import requests
import pandas as pd
from tqdm import tqdm
import time
import os
import json
from datetime import datetime
import logging
from random import uniform
import signal
import sys

class GoogleTranslateDataset:
    def __init__(self, api_key, target_language="my"):
        self.api_key = api_key
        self.url = "https://translation.googleapis.com/language/translate/v2"
        self.target_language = target_language
        self.setup_logging()
        self.setup_signal_handler()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('google_translation_log.log'),
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
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.logger.info(f"Loaded checkpoint: {checkpoint['completed']} translations completed")
                return checkpoint
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
        return {"completed": 0, "translations": []}

    def save_checkpoint(self, checkpoint_file, completed, translations):
        try:
            checkpoint = {
                "completed": completed,
                "translations": translations,
                "timestamp": datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def translate_with_retry(self, text, max_retries=5):
        base_wait_time = 1
        actual_attempt = 0
        
        while actual_attempt < max_retries:
            try:
                params = {
                    'q': text,
                    'target': self.target_language,
                    'key': self.api_key
                }
                response = requests.post(self.url, data=params, timeout=30)

                if response.status_code == 200:
                    translated_text = response.json()['data']['translations'][0]['translatedText']
                    return translated_text
                else:
                    self.logger.warning(f"Error {response.status_code}: {response.text}")
                    if response.status_code == 429 or "limit" in response.text.lower():
                        wait_time = min((2 ** actual_attempt) * base_wait_time + uniform(0, 2), 300)
                        self.logger.info(f"Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                        actual_attempt += 1
                        continue
                    return "Translation failed"
                    
            except Exception as e:
                self.logger.warning(f"Attempt {actual_attempt + 1} failed. Error: {e}")
                actual_attempt += 1
                if actual_attempt < max_retries:
                    wait_time = min((2 ** actual_attempt) * base_wait_time + uniform(0, 2), 300)
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    return "Translation failed - max retries exceeded"
        
        return "Translation failed"

    def process_in_chunks(self, df, chunk_size=100, checkpoint_interval=50):
        total_rows = len(df)
        checkpoint = self.load_checkpoint("google_translation_checkpoint.json")
        start_index = checkpoint["completed"]
        translated_texts = checkpoint["translations"]
        
        self.logger.info(f"Starting translation from index {start_index} of {total_rows} total entries")
        texts_to_process = df['text'].tolist()[start_index:]
        
        for i, text in enumerate(tqdm(texts_to_process, desc="Translating with Google", initial=start_index, total=total_rows)):
            current_index = start_index + i
            clean_text = str(text).strip() if pd.notna(text) else ""
            translation = self.translate_with_retry(clean_text) if clean_text else "Empty text"
            translated_texts.append(translation)
            
            if (current_index + 1) % checkpoint_interval == 0:
                self.save_checkpoint("google_translation_checkpoint.json", current_index + 1, translated_texts)
                self.logger.info(f"Checkpoint saved at {current_index + 1}/{total_rows}")
                self.save_intermediate_results(df, translated_texts, current_index + 1)
            
            time.sleep(0.2)  
        
        self.save_checkpoint("google_translation_checkpoint.json", total_rows, translated_texts)
        self.logger.info("Translation completed!")
        return translated_texts

    def save_intermediate_results(self, df, translated_texts, completed_count):
        try:
            partial_df = pd.DataFrame({
                'labels': df['labels'].tolist()[:completed_count],
                'original': df['text'].tolist()[:completed_count],
                'translated': translated_texts
            })
            partial_df = self.clean_dataframe(partial_df)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'Medication_Google_partial_{completed_count}_{timestamp}.csv'
            partial_df.to_csv(filename, index=False, encoding='utf-8-sig', sep=',')
            self.logger.info(f"Intermediate results saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")

    def clean_dataframe(self, df):
        for col in ['translated', 'original']:
            if col in df.columns:
                df[col] = (df[col].fillna('')
                           .astype(str)
                           .str.replace('\r\n', ' ', regex=False)
                           .str.replace('\n', ' ', regex=False)
                           .str.replace('\r', ' ', regex=False)
                           .str.replace('\t', ' ', regex=False)
                           .str.strip())
        return df

    def validate_translations(self, df):
        failed_count = df['translated'].str.contains('Translation failed', na=False).sum()
        empty_count = df['translated'].isna().sum() + (df['translated'] == '').sum()
        self.logger.info(f"Translation validation:")
        self.logger.info(f"  - Total entries: {len(df)}")
        self.logger.info(f"  - Failed translations: {failed_count}")
        self.logger.info(f"  - Empty translations: {empty_count}")
        self.logger.info(f"  - Success rate: {((len(df) - failed_count - empty_count) / len(df) * 100):.2f}%")

    def translate_dataset(self, csv_file, chunk_size=100, checkpoint_interval=50):
        try:
            self.logger.info(f"Loading dataset from {csv_file}")
            df = pd.read_csv(csv_file)
            
            if 'text' not in df.columns or 'labels' not in df.columns:
                raise ValueError("The CSV file must contain 'text' and 'labels' columns!")
            
            self.logger.info(f"Dataset loaded successfully: {len(df)} rows")
            translated_texts = self.process_in_chunks(df, chunk_size, checkpoint_interval)
            
            result_df = pd.DataFrame({
                'labels': df['labels'].tolist(),
                'original': df['text'].tolist(),
                'translated': translated_texts
            })
            result_df = self.clean_dataframe(result_df)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'Medication_Google_final_{timestamp}.csv'
            result_df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=',')
            
            self.validate_translations(result_df)
            self.logger.info(f"Final results saved to: {output_filename}")
            print(result_df.head(3).to_string(index=False, max_colwidth=40))
            return result_df
        except Exception as e:
            self.logger.error(f"Error in translation process: {e}")
            raise


def main():
   
    API_KEY = "Translation Cloud API"  
    
    CSV_FILE = 'Medical-Condition.csv'
    CHUNK_SIZE = 50
    CHECKPOINT_INTERVAL = 25
    
    translator = GoogleTranslateDataset(API_KEY)
    
    try:
        result_df = translator.translate_dataset(
            csv_file=CSV_FILE,
            chunk_size=CHUNK_SIZE,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
        print(f"\n Google Translation completed successfully!")
        print(f" Processed {len(result_df)} entries")
    except KeyboardInterrupt:
        print(f"\n Process interrupted. Progress saved. Resume later.")
    except Exception as e:
        print(f"\n Error occurred: {e}")


if __name__ == "__main__":
    main()
