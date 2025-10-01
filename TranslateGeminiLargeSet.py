import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import time
import json
import os
import sys
import signal
import logging
from datetime import datetime
from random import uniform


class DatasetTranslator:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        # Configure Gemini with API key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.setup_logging()
        self.setup_signal_handler()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("gemini_translation_log.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_signal_handler(self):
        def signal_handler(sig, frame):
            self.logger.info("Received interrupt signal. Saving progress...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def load_checkpoint(self, checkpoint_file):
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                self.logger.info(
                    f"Loaded checkpoint: {checkpoint['completed']} translations completed"
                )
                return checkpoint
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
        return {"completed": 0, "translations": []}

    def save_checkpoint(self, checkpoint_file, completed, translations):
        try:
            checkpoint = {
                "completed": completed,
                "translations": translations,
                "timestamp": datetime.now().isoformat(),
            }
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def is_rate_limit_error(self, error):
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "quota",
            "too many requests",
            "429",
            "resource_exhausted",
            "rate_limit_exceeded",
            "requests per",
            "limit exceeded",
            "throttle",
            "usage limit",
            "daily limit",
            "monthly limit",
            "per minute",
            "per hour",
        ]
        return any(ind in error_str for ind in rate_limit_indicators)

    def calculate_wait_time(self, error, base_wait=60):
        error_str = str(error).lower()
        if "minute" in error_str:
            return 70
        elif "hour" in error_str:
            return 3660
        elif "day" in error_str or "daily" in error_str:
            return 86400
        elif "quota" in error_str:
            return 3600
        else:
            return base_wait

    def wait_with_progress(self, wait_seconds, reason="Rate limit"):
        self.logger.info(
            f"{reason} detected. Waiting {wait_seconds} seconds ({wait_seconds/60:.1f} minutes)"
        )
        for remaining in tqdm(
            range(wait_seconds, 0, -1), desc=f"Waiting ({reason})", unit="sec"
        ):
            time.sleep(1)
            if remaining % 60 == 0:
                self.logger.info(
                    f"Still waiting... {remaining//60} minutes remaining"
                )
        self.logger.info("Wait period completed. Resuming translation...")

    def translate_with_retry(self, text, max_retries=5):
        base_wait_time = 1
        actual_attempt = 0

        while actual_attempt < max_retries:
            try:
                response = self.model.generate_content(
                    "Please translate this text from English to Burmese language. "
                    "Do not add any additional information or context. "
                    "Just provide the translation. [Text]: " + str(text)
                )

                if response.text and response.text.strip():
                    if hasattr(self, "_consecutive_failures"):
                        delattr(self, "_consecutive_failures")
                    return response.text.strip()
                else:
                    self.logger.warning(
                        f"Empty response for text: {str(text)[:50]}..."
                    )
                    return "Translation failed - empty response"

            except Exception as e:
                self.logger.warning(
                    f"Attempt {actual_attempt + 1} failed for text: {str(text)[:50]}... Error: {e}"
                )

                if self.is_rate_limit_error(e):
                    wait_time = self.calculate_wait_time(e)
                    self.wait_with_progress(wait_time, "Rate limit")
                    continue

                actual_attempt += 1
                if actual_attempt < max_retries:
                    wait_time = min(
                        (2 ** actual_attempt) * base_wait_time + uniform(0, 2), 300
                    )
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All retry attempts failed for text: {str(text)[:50]}..."
                    )
                    return "Translation failed - max retries exceeded"

        return "Translation failed"

    def process_in_chunks(self, df, chunk_size=100, checkpoint_interval=50):
        total_rows = len(df)
        checkpoint = self.load_checkpoint("gemini_translation_checkpoint_latest.json")
        start_index = checkpoint["completed"]
        translated_texts = checkpoint["translations"]

        self.logger.info(
            f"Starting translation from index {start_index} of {total_rows} total entries"
        )
        texts_to_process = df["text"].tolist()[start_index:]

        for i, text in enumerate(
            tqdm(
                texts_to_process,
                desc="Translating",
                initial=start_index,
                total=total_rows,
            )
        ):
            current_index = start_index + i
            translation = self.translate_with_retry(text)
            translated_texts.append(translation)

            if (current_index + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    "gemini_translation_checkpoint_latest.json",
                    current_index + 1,
                    translated_texts,
                )
                self.logger.info(
                    f"Checkpoint saved at {current_index + 1}/{total_rows}"
                )
                self.save_intermediate_results(
                    df, translated_texts, current_index + 1
                )

            sleep_time = uniform(0.5, 2.0)
            if hasattr(self, "_consecutive_failures"):
                self._consecutive_failures += (
                    1 if "failed" in translation else 0
                )
            else:
                self._consecutive_failures = (
                    1 if "failed" in translation else 0
                )
            if self._consecutive_failures > 3:
                sleep_time = uniform(2.0, 5.0)
            time.sleep(sleep_time)

        self.save_checkpoint(
            "gemini_translation_checkpoint_latest.json",
            total_rows,
            translated_texts,
        )
        self.logger.info("Translation completed!")
        return translated_texts

    def save_intermediate_results(self, df, translated_texts, completed_count):
        try:
            partial_df = pd.DataFrame(
                {
                    "labels": df["labels"].tolist()[:completed_count],
                    "original": df["text"].tolist()[:completed_count],
                    "translated": translated_texts,
                }
            )
            partial_df = self.clean_dataframe(partial_df)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Burmese_Medication_Gemini_partial_{completed_count}_{timestamp}.csv"
            partial_df.to_csv(
                filename, index=False, encoding="utf-8-sig", sep=","
            )
            self.logger.info(f"Intermediate results saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")

    def clean_dataframe(self, df):
        for col in ["translated", "original"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.replace("\r\n", " ", regex=False)
                    .str.replace("\n", " ", regex=False)
                    .str.replace("\r", " ", regex=False)
                    .str.replace("\t", " ", regex=False)
                    .str.strip()
                )
        return df

    def validate_translations(self, df):
        failed_count = df["translated"].str.contains(
            "Translation failed", na=False
        ).sum()
        empty_count = (
            df["translated"].isna().sum()
            + (df["translated"] == "").sum()
        )

        self.logger.info("Translation validation:")
        self.logger.info(f"  - Total entries: {len(df)}")
        self.logger.info(f"  - Failed translations: {failed_count}")
        self.logger.info(f"  - Empty translations: {empty_count}")
        self.logger.info(
            f"  - Success rate: {((len(df) - failed_count - empty_count) / len(df) * 100):.2f}%"
        )

    def translate_dataset(self, csv_file, chunk_size=100, checkpoint_interval=50):
        try:
            self.logger.info(f"Loading dataset from {csv_file}")
            df = pd.read_csv(csv_file)

            if "text" not in df.columns or "labels" not in df.columns:
                raise ValueError(
                    "The CSV file must contain 'text' and 'labels' columns!"
                )

            self.logger.info(
                f"Dataset loaded successfully: {len(df)} rows"
            )
            translated_texts = self.process_in_chunks(
                df, chunk_size, checkpoint_interval
            )

            result_df = pd.DataFrame(
                {
                    "labels": df["labels"].tolist(),
                    "original": df["text"].tolist(),
                    "translated": translated_texts,
                }
            )
            result_df = self.clean_dataframe(result_df)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = (
                f"Burmese_Medication_Gemini_final_{timestamp}.csv"
            )
            result_df.to_csv(
                output_filename, index=False, encoding="utf-8-sig", sep=","
            )

            self.validate_translations(result_df)
            self.logger.info(f"Final results saved to: {output_filename}")

            print("\nSample results:")
            print(result_df.head(3).to_string(index=False, max_colwidth=40))

            if os.path.exists("gemini_translation_checkpoint_latest.json"):
                os.rename(
                    "gemini_translation_checkpoint_latest.json",
                    f"gemini_translation_checkpoint_completed_{timestamp}.json",
                )

            return result_df
        except Exception as e:
            self.logger.error(f"Error in translation process: {e}")
            raise


def main():
    
    API_KEY = "YOUR_REAL_API_KEY_HERE"

    CSV_FILE = "Medication.csv"
    CHUNK_SIZE = 50
    CHECKPOINT_INTERVAL = 25

    translator = DatasetTranslator(API_KEY)

    try:
        result_df = translator.translate_dataset(
            csv_file=CSV_FILE,
            chunk_size=CHUNK_SIZE,
            checkpoint_interval=CHECKPOINT_INTERVAL,
        )
        print("\n Translation completed successfully!")
        print(f" Processed {len(result_df)} entries")

    except KeyboardInterrupt:
        print("\n Process interrupted by user. Progress has been saved.")
        print(
            " Resume by running the script again - it will continue from the last checkpoint."
        )

    except Exception as e:
        print(f"\n Error occurred: {e}")
        print(" Check the log file and checkpoint for recovery options.")


if __name__ == "__main__":
    main()
