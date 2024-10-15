# data_visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Load the preprocessed image metadata (the DataFrame saved from the preprocessing script)
metadata_file = 'organized_image_data_with_categories.csv'
image_data = pd.read_csv(metadata_file)

# Visualization of the data distribution
def visualize_category_distribution(data):
    """
    Displays a bar chart showing the number of images in each category.
    """
    category_counts = data['category'].value_counts()
    
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color=['skyblue', 'lightgreen'])
    plt.title('Number of Images per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=0)
    plt.show()

def visualize_image_size_distribution(data):
    """
    Displays scatter plots of image width and height distribution.
    """
    image_sizes = data['image_size'].apply(lambda x: eval(x))  # Convert string "(width, height)" to tuple
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    plt.figure(figsize=(10, 6))
    
    # Plot image width distribution
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='lightblue', edgecolor='black')
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    
    # Plot image height distribution
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def display_sample_images(data, n=3):
    """
    Displays sample images from each category.
    """
    categories = data['category'].unique()
    plt.figure(figsize=(12, 6))
    
    for idx, category in enumerate(categories):
        category_data = data[data['category'] == category]
        sample_images = category_data.sample(n)
        
        for i, image_path in enumerate(sample_images['image_path']):
            img = Image.open(image_path)
            plt.subplot(len(categories), n, idx * n + i + 1)
            plt.imshow(img)
            plt.title(f'{category} - Image {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function to call visualization functions
def main():
    print("Visualizing category distribution...")
    visualize_category_distribution(image_data)
    
    print("Visualizing image size distribution...")
    visualize_image_size_distribution(image_data)
    
    print(f"Displaying {3} sample images per category...")
    display_sample_images(image_data, n=3)

if __name__ == '__main__':
    main()
