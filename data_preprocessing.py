import os
from PIL import Image
import pandas as pd

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
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Include multiple image formats
                file_path = os.path.join(subdir, file)
                img = Image.open(file_path)
                data.append({
                    'image_path': file_path,
                    'category': category,
                    'file_extension': os.path.splitext(file)[1].lower(),  # Extract file extension
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

# Function to summarize dataset information
def summarize_dataset(dataframe):
    """
    Displays a summary of the dataset, including:
    - Distribution of categories
    - Count of file extensions
    - Most common image sizes
    - Most common image modes
    """
    print("\n--- Dataset Summary ---")
    
    # Distribution of categories
    print("\nCategory Distribution:")
    print(dataframe['category'].value_counts())
    
    # File extension distribution
    print("\nFile Extension Distribution:")
    print(dataframe['file_extension'].value_counts())
    
    # Most common image sizes
    print("\nTop 5 Most Common Image Sizes:")
    print(dataframe['image_size'].value_counts().head())
    
    # Most common image modes
    print("\nImage Mode Distribution:")
    print(dataframe['image_mode'].value_counts())

# Main function to load and organize the dataset
def main():
    print("Loading images from all categories...")
    all_images = load_all_images(dataset_base_path, categories)
    
    # Preview the combined dataset
    print("\nPreview of the Dataset:")
    print(all_images.head())
    
    # Display dataset summary
    summarize_dataset(all_images)
    
    # Optionally, save the metadata for future use
    all_images.to_csv('organized_image_data_with_categories.csv', index=False)

if __name__ == '__main__':
    main()
