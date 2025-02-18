import pandas as pd
import torch
from transformers import T5Tokenizer
import os

# ------------------ EDIT THESE PATHS AS NEEDED ------------------
INPUT_CSV  = "/Users/adrian/Athena/data/Processed Data/cleaned_annotations.csv"  
OUTPUT_PT  = "/Users/adrian/Athena/data/Processed Data/tokenized_texts.pt"           
# ----------------------------------------------------------------

# Tokenizer settings
MAX_LENGTH = 128  # Max tokens per description

def load_text_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} text descriptions.")
    return df["image_id"].tolist(), df["annotation"].tolist()

def tokenize_texts(texts, tokenizer):
    tokenized_outputs = tokenizer(
        texts,
        return_tensors="pt",  # PyTorch tensors
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )["input_ids"]  # Extract token IDs

    print(f"[INFO] Tokenized {len(texts)} text descriptions.")
    return tokenized_outputs

def main():
    print("=== Starting Text Tokenization ===")

    # 1. Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token if missing

    # 2. Load text data
    image_ids, texts = load_text_data(INPUT_CSV)

    # 3. Tokenize text descriptions
    tokenized_texts = tokenize_texts(texts, tokenizer)

    # 4. Save tokenized data
    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save({"image_ids": image_ids, "tokenized_texts": tokenized_texts}, OUTPUT_PT)
    print(f"[INFO] Tokenized text data saved to {OUTPUT_PT}")

    print("=== Text Tokenization Complete ===")

if __name__ == "__main__":
    main()
