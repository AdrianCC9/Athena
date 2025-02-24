import os
import csv
import pickle
import pandas as pd

# ------------------ EDIT THESE PATHS AS NEEDED ------------------
ARTIFICIAL_PKL_PATH = r"Q:\Athena\data\General Data\Tell2Design_artificial_all.pkl"   
IMAGES_DIR          = r"Q:\Athena\data\General Data\floorplan_image"                    
OUTPUT_CSV          = r"Q:\Athena\data\Processed Data\cleaned_annotations.csv" 
# ----------------------------------------------------------------

def load_artificial_annotations(pkl_path):
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)  # Load the dataset as a list of dicts
    
    records = []
    for entry in data_list:
        image_id = entry.get("image_id")
        short_description = entry.get("short_version", {}).get("string", "").strip()

        # Ensure both image ID and text exist
        if image_id and short_description:
            records.append({
                "image_id": image_id,
                "annotation": short_description
            })
    
    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} artificial annotations from PKL.")
    return df

def validate_images(df, images_dir):
    
    valid_rows = []
    missing_count = 0

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row["image_id"])
        if os.path.exists(image_path):
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARNING] {missing_count} records were dropped because the corresponding .png was not found.")
    
    return pd.DataFrame(valid_rows)

def main():
    print("=== Starting Data Cleaning (Artificial Annotations) ===")

    # 1. Load artificial dataset
    df = load_artificial_annotations(ARTIFICIAL_PKL_PATH)

    # 2. Validate that images exist
    df = validate_images(df, IMAGES_DIR)
    print(f"[INFO] After validation, {len(df)} valid image-text pairs remain.")

    # 3. Drop duplicate/empty rows
    before = len(df)
    df.drop_duplicates(subset=["image_id", "annotation"], inplace=True)
    df.dropna(subset=["annotation"], inplace=True)
    after = len(df)
    print(f"[INFO] Dropped {before - after} duplicate/empty rows. Total now: {len(df)}.")

    # 4. Save cleaned dataset to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"[INFO] Cleaned data saved to {OUTPUT_CSV}")
    print("=== Data Cleaning Complete ===")

if __name__ == "__main__":
    main()