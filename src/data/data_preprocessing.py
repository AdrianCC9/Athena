import os
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Paths to datasets
base_dir = r"Q:\adria\Documents\Tell2Design Data"
general_data_dir = os.path.join(base_dir, "General Data")
separated_data_dir = os.path.join(base_dir, "Separated Data")
output_dir = r"Q:\Athena\data\cleaned_data"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_data():
    # General Data
    images_path = os.path.join(general_data_dir, "floorplan_image")
    human_annotations_path = os.path.join(general_data_dir, "human_annotated_tags")
    artificial_annotations_path = os.path.join(general_data_dir, "Tell2Design_artificial_all.pkl")

    # Load images
    images = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".png")]
    images_df = pd.DataFrame({"image_path": images})

    # Load human annotations
    human_annotations = []
    for file in os.listdir(human_annotations_path):
        try:
            with open(os.path.join(human_annotations_path, file), "r", encoding="utf-8") as f:
                human_annotations.append({"image_id": file.replace(".txt", ".png"), "annotation": f.read().strip()})
        except UnicodeDecodeError:
            print(f"Skipping file due to encoding error: {file}")
    human_annotations_df = pd.DataFrame(human_annotations)

    # Load artificial annotations
    with open(artificial_annotations_path, "rb") as f:
        artificial_annotations = pickle.load(f)

    # Extract relevant fields from artificial annotations
    artificial_annotations_list = []
    for entry in artificial_annotations:
        artificial_annotations_list.append({
            "image_id": entry["image_id"],
            "description": entry["short_version"]["string"]
        })
    artificial_annotations_df = pd.DataFrame(artificial_annotations_list)

    return images_df, human_annotations_df, artificial_annotations_df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in ENGLISH_STOP_WORDS]  # Remove stop words
    return " ".join(text_tokens)

def preprocess_data(images_df, human_annotations_df, artificial_annotations_df):
    # Clean text in annotations
    human_annotations_df["annotation"] = human_annotations_df["annotation"].dropna().apply(clean_text)
    artificial_annotations_df["description"] = artificial_annotations_df["description"].dropna().apply(clean_text)

    # Handle missing values
    human_annotations_df["annotation"].fillna("", inplace=True)
    artificial_annotations_df["description"].fillna("", inplace=True)
    images_df.dropna(subset=["image_path"], inplace=True)

    return images_df, human_annotations_df, artificial_annotations_df

def save_preprocessed_data(images_df, human_annotations_df, artificial_annotations_df):
    images_df.to_csv(os.path.join(output_dir, "cleaned_floorplan_images.csv"), index=False)
    human_annotations_df.to_csv(os.path.join(output_dir, "cleaned_human_annotations.csv"), index=False)
    artificial_annotations_df.to_csv(os.path.join(output_dir, "cleaned_artificial_annotations.csv"), index=False)
    print("Preprocessing completed. Cleaned and normalized data saved.")

if __name__ == "__main__":
    images_df, human_annotations_df, artificial_annotations_df = load_data()

    # Preprocess main datasets
    images_df, human_annotations_df, artificial_annotations_df = preprocess_data(images_df, human_annotations_df, artificial_annotations_df)
    save_preprocessed_data(images_df, human_annotations_df, artificial_annotations_df)