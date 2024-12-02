import pandas as pd
import torch
from transformers import GPT2Tokenizer
import os

# Paths
HUMAN_ANNOTATIONS_PATH = r"Q:\Athena\data\cleaned_data\cleaned_human_annotations.csv"
ARTIFICIAL_ANNOTATIONS_PATH = r"Q:\Athena\data\cleaned_data\cleaned_artificial_annotations.csv"
OUTPUT_PATH = r"Q:\Athena\data\tokenized_data\tokenized_combined_annotations.pt"

# Parameters
MAX_LENGTH = 128  # Maximum token length

def load_data(human_path, artificial_path):
    """
    Load and merge human and artificial annotations into a single DataFrame.
    """
    # Load the data
    human_df = pd.read_csv(human_path)
    artificial_df = pd.read_csv(artificial_path)
    
    # Merge dataframes on 'image_id'
    combined_df = pd.merge(
        artificial_df[['image_id', 'description']],
        human_df[['image_id', 'annotation']],
        on='image_id',
        how='left'
    )
    
    # Create 'combined_annotation' column with preference for human annotation
    combined_df['combined_annotation'] = combined_df['annotation'].combine_first(combined_df['description'])
    
    return combined_df

def tokenize_annotations(data, tokenizer, column_name="combined_annotation"):
    """
    Tokenize text annotations using GPT-2 tokenizer.
    """
    tokenized_outputs = []
    image_ids = []
    for idx, row in data.iterrows():
        text = row[column_name]
        image_id = row['image_id']
        if pd.isna(text):
            continue  # Skip if text is NaN
        # Tokenize and pad sequences
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized_outputs.append(tokens["input_ids"])  # Save tokenized input IDs
        image_ids.append(image_id)
    # Stack the tensors to create a single tensor
    tokenized_data = torch.cat(tokenized_outputs, dim=0)  # Shape: [N, max_length]
    return tokenized_data, image_ids

def save_tokenized_data(tokenized_data_dict, output_path):
    """
    Save tokenized data and image IDs as a PyTorch tensor file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(tokenized_data_dict, output_path)
    print(f"Tokenized data saved to {output_path}.")

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not defined

    # Load and merge data
    data = load_data(HUMAN_ANNOTATIONS_PATH, ARTIFICIAL_ANNOTATIONS_PATH)

    # Tokenize annotations
    tokenized_data, image_ids = tokenize_annotations(data, tokenizer)

    # Save tokenized data and image IDs
    save_tokenized_data({'input_ids': tokenized_data, 'image_ids': image_ids}, OUTPUT_PATH)
