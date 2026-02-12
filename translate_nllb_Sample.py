import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

# Model and languages
CHECKPOINT = "facebook/nllb-200-1.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "mya_Mymr"
INPUT_CSV = "Splited_Medication.csv"
OUTPUT_CSV = "Splited_Medication_nllb.csv"
BATCH_SIZE = 8 

print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
    max_length=1024,
    device=0,  # GPU device 0
)
print("Model loaded.")


df = pd.read_csv(INPUT_CSV)


if 'text' not in df.columns or 'labels' not in df.columns:
    print("Error: The CSV file must contain 'text' and 'labels' columns!")
    exit()

print(f"Total rows in dataset: {len(df)}")


texts_to_translate = df['text'].tolist()
labels_list = df['labels'].tolist()

print(f"Starting translation of {len(texts_to_translate)} entries...")


translated_texts = []
batch_texts = []
batch_indices = []

def translate_batch(batch_texts):
    """Translate a batch of texts and return results"""
    if not batch_texts:
        return []
    
    try:
        outputs = translator(batch_texts)
        return [out['translation_text'] for out in outputs]
    except Exception as e:
        print(f"Batch translation error: {e}")
        return ["Translation failed"] * len(batch_texts)


for i, text in enumerate(tqdm(texts_to_translate, desc="Translating")):
    batch_texts.append(str(text))
    batch_indices.append(i)
    
   
    if len(batch_texts) == BATCH_SIZE:
        batch_translations = translate_batch(batch_texts)
        translated_texts.extend(batch_translations)
        batch_texts = []
        batch_indices = []


if batch_texts:
    batch_translations = translate_batch(batch_texts)
    translated_texts.extend(batch_translations)


result_df = pd.DataFrame({
    'labels': labels_list,
    'original': texts_to_translate,
    'translated': translated_texts
})


result_df['translated'] = result_df['translated'].astype(str).str.replace('\r\n', ' ').str.replace('\n', ' ').str.replace('\r', ' ')
result_df['original'] = result_df['original'].astype(str).str.replace('\r\n', ' ').str.replace('\n', ' ').str.replace('\r', ' ')


result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', sep=',')

print(f"\nTranslation completed! {len(translated_texts)} entries processed.")
print(f"Results saved to: {OUTPUT_CSV}")
print(f"File contains {len(result_df)} rows with columns: {', '.join(result_df.columns)}")


print(f"\nSample results:")
print(result_df.head(3).to_string(index=False, max_colwidth=40))

