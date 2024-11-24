import pandas as pd
import torch
from transformers import GPT2Tokenizer
import os

# Paths to your cleaned data and where to save tokenized outputs
CLEANED_DATA_PATH = "Q:\Athena\data\cleaned_data\cleaned_human_annotations.csv"
OUTPUT_PATH = "Q:/Athena/Data/tokenized_data/tokenized_human_annotations.pt"

def load_cleaned_data(filepath):
    """
    Load cleaned text data from a CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}. Total rows: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def tokenize_text(data, column_name, tokenizer):
    """
    Tokenize text using a Hugging Face tokenizer.
    :param data: DataFrame containing the text data.
    :param column_name: The name of the column containing text to tokenize.
    :param tokenizer: Hugging Face tokenizer object.
    :return: List of tokenized text tensors.
    """
    try:
        # Set pad token for the tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenized_outputs = []
        for text in data[column_name].dropna():
            # Tokenize and pad sequences
            tokens = tokenizer(
                text, 
                return_tensors='pt', 
                padding='max_length',  # Pad shorter sequences to max length
                truncation=True,  # Truncate longer sequences
                max_length=128  # Set a fixed maximum length for consistency
            )
            tokenized_outputs.append(tokens['input_ids'])  # Save input IDs
        
        print(f"Tokenized {len(tokenized_outputs)} rows successfully.")
        return torch.cat(tokenized_outputs, dim=0)  # Combine into a single tensor
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None

def save_tokenized_data(tokenized_data, output_path):
    """
    Save tokenized data to a file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(tokenized_data, output_path)
        print(f"Tokenized data saved to {output_path}.")
    except Exception as e:
        print(f"Error saving tokenized data: {e}")

if __name__ == "__main__":
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load cleaned data
    data = load_cleaned_data(CLEANED_DATA_PATH)

    if data is not None:
        # Tokenize the text column (e.g., "annotation")
        tokenized_data = tokenize_text(data, column_name="annotation", tokenizer=tokenizer)

        if tokenized_data is not None:
            # Save the tokenized data
            save_tokenized_data(tokenized_data, OUTPUT_PATH)

# Summary:
# 1. Loaded the cleaned text data from a CSV file.
# 2. Tokenized the annotations using GPT2 tokenizer from Hugging Face, padding to a fixed maximum length and truncating as necessary.
# 3. Saved the tokenized outputs in a PyTorch tensor format for model training.
