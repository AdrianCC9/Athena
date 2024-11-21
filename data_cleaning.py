import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Define dataset paths
raw_data_file = 'organized_image_data_with_categories.csv'  # Path to your raw data
cleaned_data_file = 'cleaned_image_data.csv'  # Path to save cleaned data

# 1. Load Dataset
def load_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print("\n--- Raw Data Preview ---")
    print(data.head())
    return data

# 2. Handle Missing Values
def handle_missing_values(dataframe):
    print("\nHandling missing values...")
    before = dataframe.isnull().sum()
    dataframe.fillna(method='ffill', inplace=True)
    after = dataframe.isnull().sum()
    print("\n--- Missing Values Before and After ---")
    print("Before:")
    print(before)
    print("\nAfter:")
    print(after)
    return dataframe

# 3. Remove Duplicates
def remove_duplicates(dataframe):
    print("\nRemoving duplicates...")
    before = len(dataframe)
    dataframe.drop_duplicates(inplace=True)
    after = len(dataframe)
    print(f"\n--- Duplicates Removed: {before - after} ---")
    print(f"Remaining Rows: {after}")
    return dataframe

# 4. Handle Outliers (Example for numerical data)
def handle_outliers(dataframe, column):
    print(f"\nHandling outliers in column: {column}...")
    before = len(dataframe)
    z_scores = np.abs((dataframe[column] - dataframe[column].mean()) / dataframe[column].std())
    dataframe = dataframe[z_scores < 3]
    after = len(dataframe)
    print(f"--- Outliers Removed: {before - after} ---")
    return dataframe

# 5. Normalize/Standardize Data (e.g., image sizes or pixel values)
def normalize_pixel_values(dataframe):
    print("\nNormalizing pixel values...")
    dataframe['normalized_pixel_values'] = dataframe['image_path'].apply(lambda path: normalize_image(path))
    print("\n--- Pixel Normalization Completed ---")
    return dataframe

def normalize_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0  # Normalize pixel values to range 0-1
    return img_array

# 6. Split Data into Training, Validation, and Test Sets
def split_data(dataframe):
    print("\nSplitting data into training, validation, and test sets...")
    train_data, test_data = train_test_split(dataframe, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"\n--- Data Split ---")
    print(f"Training Data: {len(train_data)} rows")
    print(f"Validation Data: {len(val_data)} rows")
    print(f"Test Data: {len(test_data)} rows")
    return train_data, val_data, test_data

# 7. Save Cleaned Data
def save_data(dataframe, file_path):
    print(f"\nSaving cleaned data to {file_path}...")
    dataframe.to_csv(file_path, index=False)
    print(f"Saved {len(dataframe)} rows to {file_path}")

# Main Function
def main():
    # Step 1: Load Dataset
    data = load_data(raw_data_file)

    # Step 2: Clean the Data
    data = handle_missing_values(data)
    data = remove_duplicates(data)

    # (Optional) Example: Handle outliers for numerical columns
    # data = handle_outliers(data, 'some_numeric_column')

    # Step 3: Normalize/Standardize Data
    data = normalize_pixel_values(data)

    # Step 4: Split the Data
    train_data, val_data, test_data = split_data(data)

    # Step 5: Save the Cleaned Data
    save_data(train_data, 'train_data.csv')
    save_data(val_data, 'val_data.csv')
    save_data(test_data, 'test_data.csv')
    print("\nData cleaning and preprocessing completed!")

if __name__ == '__main__':
    main()
