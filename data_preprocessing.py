import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Define dataset paths
dataset_base_path = '/Users/adrian/Documents/cubicasa5k'  # Path to the main dataset folder
categories = ['colorful', 'high_quality']  # Subdirectories representing categories

# Function to load images and keep their category and path info
def load_images_from_category(category, base_path):
    """
    Loads images from a specific category (subdirectory) and returns their metadata.
    """
    category_path = os.path.join(base_path, category)
    data = []
    
    for subdir, _, files in os.walk(category_path):
        for file in files:
            if file.endswith('.png'):  # Filter PNG files
                file_path = os.path.join(subdir, file)
                img = Image.open(file_path)
                data.append({
                    'image_path': file_path,
                    'category': category,
                    'image_size': img.size,
                    'image_mode': img.mode
                })
    return pd.DataFrame(data)

# Load images from all categories
def load_all_images(base_path, categories):
    all_data = []
    for category in categories:
        category_data = load_images_from_category(category, base_path)
        all_data.append(category_data)
    return pd.concat(all_data, ignore_index=True)

# Main function to load and organize the dataset
def main():
    print("Loading images from all categories...")
    all_images = load_all_images(dataset_base_path, categories)
    
    # Preview the combined dataset
    print(all_images.head())
    
    # Optionally, save the metadata for future use
    all_images.to_csv('organized_image_data_with_categories.csv', index=False)

if __name__ == '__main__':
    main()
