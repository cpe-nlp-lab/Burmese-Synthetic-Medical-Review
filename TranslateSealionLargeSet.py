import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import requests
import signal
import sys
from random import uniform

class SealionDatasetTranslator:
    def __init__(self, api_key, model_name="aisingapore/Gemma-SEA-LION-v4-27B-IT"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.sea-lion.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.setup_logging()
        self.setup_signal_handler()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("sealion_translation.log"),
                logging.StreamHandler()
            ]
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
                self.logger.info(f"Resuming from checkpoint: {checkpoint['completed']} translations completed")
                return checkpoint
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
        return {"completed": 0, "translations": []}

    def save_checkpoint(self, checkpoint_file, completed, translations):
        checkpoint = {
            "completed": completed,
            "translations": translations,
            "timestamp": datetime.now().isoformat()
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def is_rate_limit_error(self, response):
        try:
            err = response.json().get("error", {}).get("message", "").lower()
            return "rate limit" in err or "too many requests" in err or "quota" in err
        except Exception:
            return False

    def translate_with_retry(self, text, max_retries=5):
        base_wait_time = 1
        actual_attempt = 0

        while actual_attempt < max_retries:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please translate this text from English to Burmese language. "
                                   f"Do not add any additional information or context. Just provide the translation. [Text]: {text}"
                    }
                ]
            }

            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    translated_text = result["choices"][0]["message"]["content"].strip()
                    if hasattr(self, "_consecutive_failures"):
                        delattr(self, "_consecutive_failures")
                    return translated_text
                else:
                    self.logger.warning(f"API error {response.status_code}: {response.text[:100]}")

                    if self.is_rate_limit_error(response):
                        wait_time = 65  # 1+ minute wait for SEA-LION’s strict 10 req/min limit
                        self.logger.info(f"Rate limit hit. Waiting {wait_time}s before retrying...")
                        time.sleep(wait_time)
                        continue

                    actual_attempt += 1
                    wait_time = min((2 ** actual_attempt) * base_wait_time + uniform(0, 2), 300)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                actual_attempt += 1
                time.sleep(5)

        self.logger.error(f"All retry attempts failed for text: {str(text)[:50]}...")
        return "Translation failed - max retries exceeded"

    def process_in_chunks(self, df, checkpoint_interval=25):
        total_rows = len(df)
        checkpoint = self.load_checkpoint("sealion_checkpoint_latest.json")
        start_index = checkpoint["completed"]
        translated_texts = checkpoint["translations"]

        self.logger.info(f"Starting translation from index {start_index} of {total_rows} total entries")
        texts_to_process = df["text"].tolist()[start_index:]

        for i, text in enumerate(tqdm(texts_to_process, desc="Translating", initial=start_index, total=total_rows)):
            current_index = start_index + i
            translation = self.translate_with_retry(str(text))
            translated_texts.append(translation)

            if (current_index + 1) % checkpoint_interval == 0:
                self.save_checkpoint("sealion_checkpoint_latest.json", current_index + 1, translated_texts)
                self.logger.info(f"Checkpoint saved at {current_index + 1}/{total_rows}")
                self.save_intermediate_results(df, translated_texts, current_index + 1)

            # SEA-LION rate limit → max 10 req/min (1 every 6s)
            time.sleep(6)

        self.save_checkpoint("sealion_checkpoint_latest.json", total_rows, translated_texts)
        self.logger.info("Translation completed!")
        return translated_texts

    def save_intermediate_results(self, df, translated_texts, completed_count):
        try:
            partial_df = pd.DataFrame({
                "labels": df["labels"].tolist()[:completed_count],
                "original": df["text"].tolist()[:completed_count],
                "translated": translated_texts
            })
            partial_df = self.clean_dataframe(partial_df)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Burmese_Medication_SEALION_partial_{completed_count}_{ts}.csv"
            partial_df.to_csv(filename, index=False, encoding="utf-8-sig")
            self.logger.info(f"Intermediate results saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")

    def clean_dataframe(self, df):
        for col in ["translated", "original"]:
            if col in df.columns:
                df[col] = (df[col].fillna("")
                           .astype(str)
                           .str.replace("\r\n", " ", regex=False)
                           .str.replace("\n", " ", regex=False)
                           .str.replace("\r", " ", regex=False)
                           .str.replace("\t", " ", regex=False)
                           .str.strip())
        return df

    def validate_translations(self, df):
        failed_count = df["translated"].str.contains("Translation failed", na=False).sum()
        empty_count = df["translated"].isna().sum() + (df["translated"] == "").sum()

        self.logger.info("Translation validation:")
        self.logger.info(f"  - Total entries: {len(df)}")
        self.logger.info(f"  - Failed translations: {failed_count}")
        self.logger.info(f"  - Empty translations: {empty_count}")
        self.logger.info(f"  - Success rate: {((len(df) - failed_count - empty_count) / len(df) * 100):.2f}%")

    def translate_dataset(self, csv_file, checkpoint_interval=25):
        try:
            self.logger.info(f"Loading dataset from {csv_file}")
            df = pd.read_csv(csv_file)

            if "text" not in df.columns or "labels" not in df.columns:
                raise ValueError("The CSV file must contain 'text' and 'labels' columns!")

            self.logger.info(f"Dataset loaded successfully: {len(df)} rows")
            translated_texts = self.process_in_chunks(df, checkpoint_interval)

            result_df = pd.DataFrame({
                "labels": df["labels"].tolist(),
                "original": df["text"].tolist(),
                "translated": translated_texts
            })
            result_df = self.clean_dataframe(result_df)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"Burmese_Medication_SEALION_final_{ts}.csv"
            result_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

            self.validate_translations(result_df)
            self.logger.info(f"Final results saved to: {output_filename}")

            print("\nSample results:")
            print(result_df.head(3).to_string(index=False, max_colwidth=40))

            if os.path.exists("sealion_checkpoint_latest.json"):
                os.rename("sealion_checkpoint_latest.json", f"sealion_checkpoint_completed_{ts}.json")

            return result_df
        except Exception as e:
            self.logger.error(f"Error in translation process: {e}")
            raise


def main():
   
    API_KEY = "SEALION API"  
    
    if not API_KEY or not API_KEY.startswith("sk-"):
        print(" Error: SEA-LION API key is missing or invalid!")
        return

    CSV_FILE = "Medication.csv"
    CHECKPOINT_INTERVAL = 50

    translator = SealionDatasetTranslator(API_KEY)

    try:
        result_df = translator.translate_dataset(
            csv_file=CSV_FILE,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
        print("\nTranslation completed successfully!")
        print(f"Processed {len(result_df)} entries")

    except KeyboardInterrupt:
        print("\n Process interrupted by user. Progress has been saved.")
        print("Resume by running the script again - it will continue from the last checkpoint.")

    except Exception as e:
        print(f"\n Error occurred: {e}")
        print("Check the log file and checkpoint for recovery options.")


if __name__ == "__main__":
    main()
