import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_DIR  = "/Users/adrian/Athena/data/General Data/floorplan_image"       
NUM_IMAGES = 6                                

def visualize_images(image_dir, num_images=6):
    """
    Selects 'num_images' random PNG files from 'image_dir'
    and displays them in a grid using Matplotlib.
    """
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    random.shuffle(all_files)
    selected_files = all_files[:num_images]

    plt.figure(figsize=(12, 6))
    for i, img_name in enumerate(selected_files, 1):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # Keep full color

        plt.subplot(2, (num_images // 2), i)
        plt.imshow(img)  # Display in original color
        plt.title(img_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    print("=== Previewing Blueprint Images ===")
    visualize_images(IMAGE_DIR, NUM_IMAGES)

if __name__ == "__main__":
    main()

