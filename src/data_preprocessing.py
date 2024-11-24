import pandas as pd
import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the processed CSV files
processed_data_dir = "Q:\Athena\data\Processed Data"
images_file = os.path.join(processed_data_dir, "floorplan_images.csv")
human_annotations_file = os.path.join(processed_data_dir, "human_annotations.csv")
artificial_annotations_file = os.path.join(processed_data_dir, "artificial_annotations.csv")

images_df = pd.read_csv(images_file)
human_annotations_df = pd.read_csv(human_annotations_file)
artificial_annotations_df = pd.read_csv(artificial_annotations_file)

# Step 1: Cleaning the Annotations
# Remove special characters, convert to lowercase, and remove stop words from annotations
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in ENGLISH_STOP_WORDS]  # Remove stop words
    return ' '.join(text_tokens)

human_annotations_df['annotation'] = human_annotations_df['annotation'].dropna().apply(clean_text)
artificial_annotations_df['artificial_description'] = artificial_annotations_df['artificial_description'].dropna().apply(clean_text)

# Step 2: Handling Missing Values
# Replace missing values in annotations with an empty string
human_annotations_df['annotation'].fillna('', inplace=True)
artificial_annotations_df['artificial_description'].fillna('', inplace=True)

# Ensure images_df has no missing values in image_path
images_df.dropna(subset=['image_path'], inplace=True)

# Step 3: Normalization
# Normalize numerical features if available (e.g., width, height, or other possible numerical metadata)
# For demonstration purposes, we'll assume the dataset has 'Width' and 'Height' columns to normalize
if 'Width' in images_df.columns and 'Height' in images_df.columns:
    scaler = MinMaxScaler()
    images_df[['Width', 'Height']] = scaler.fit_transform(images_df[['Width', 'Height']])

# Step 4: Save Cleaned and Normalized Data
output_dir = "Processed Data/Cleaned"
os.makedirs(output_dir, exist_ok=True)

images_df.to_csv(os.path.join(output_dir, "cleaned_floorplan_images.csv"), index=False)
human_annotations_df.to_csv(os.path.join(output_dir, "cleaned_human_annotations.csv"), index=False)
artificial_annotations_df.to_csv(os.path.join(output_dir, "cleaned_artificial_annotations.csv"), index=False)

# Summary of Steps Performed:
# 1. Loaded the processed CSV files.
# 2. Cleaned the text data by removing special characters, converting to lowercase, and removing stop words.
# 3. Handled missing values by either removing them or replacing them with appropriate defaults.
# 4. Normalized numerical data using MinMaxScaler to ensure all values are in a similar range.
# 5. Saved the cleaned and normalized data to new CSV files.

print("Preprocessing completed. Cleaned and normalized data saved to 'Processed Data/Cleaned'.")
