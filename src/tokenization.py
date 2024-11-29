import pandas as pd
import torch
from transformers import GPT2Tokenizer
import os

# Paths
HUMAN_ANNOTATIONS_PATH = "Q:\Athena\data\cleaned_data\cleaned_human_annotations.csv"
ARTIFICIAL_ANNOTATIONS_PATH = "Q:\Athena\data\cleaned_data\cleaned_artificial_annotations.csv"
OUTPUT_PATH = r"Q:\Athena\data\tokenized_data\tokenized_human_annotations.pt"

# Parameters
MAX_LENGTH = 128  # Maximum token length

def load_data(human_path, artificial_path):
    """
    Load and merge human and artificial annotations into a single DataFrame.
    """
    human_df = pd.read_csv(human_path)
    artificial_df = pd.read_csv(artificial_path)

    # Combine annotations with preference for human annotations
    combined_df = artificial_df.copy()
    combined_df["annotation"] = combined_df["image_id"].map(
        dict(zip(human_df["image_id"], human_df["annotation"]))
    ).fillna(combined_df["description"])  # Use human annotations if available, otherwise artificial
    return combined_df

def tokenize_annotations(data, tokenizer, column_name="annotation"):
    """
    Tokenize text annotations using GPT-2 tokenizer.
    """
    tokenized_outputs = []
    for text in data[column_name].dropna():
        # Tokenize and pad sequences
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized_outputs.append(tokens["input_ids"])  # Save tokenized input IDs
    return torch.cat(tokenized_outputs, dim=0)

def save_tokenized_data(tokenized_data, output_path):
    """
    Save tokenized data as a PyTorch tensor file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(tokenized_data, output_path)
    print(f"Tokenized data saved to {output_path}.")

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not defined

    # Load and merge data
    data = load_data(HUMAN_ANNOTATIONS_PATH, ARTIFICIAL_ANNOTATIONS_PATH)

    # Tokenize annotations
    tokenized_data = tokenize_annotations(data, tokenizer)

    # Save tokenized data
    save_tokenized_data(tokenized_data, OUTPUT_PATH)
