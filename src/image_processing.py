import os
import torch
from PIL import Image
import torchvision.transforms as transforms

# ------------------ EDIT THESE PATHS AS NEEDED ------------------
IMAGE_DIR  = "/Users/adrian/Athena/data/General Data/floorplan_image"       
OUTPUT_PT  = "/Users/adrian/Athena/data/Processed Data/processed_images.pt"  
IMAGE_SIZE = (256, 256)  # Target size for images
# ----------------------------------------------------------------

def load_and_preprocess_images(image_dir):
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(IMAGE_SIZE),  # Resize to target size
        transforms.ToTensor(),  # Convert to PyTorch tensor (values [0,1])
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
    ])

    image_tensors = {}
    total_images = 0

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            tensor = transform(image)
            image_tensors[filename] = tensor
            total_images += 1

    print(f"[INFO] Processed {total_images} images.")
    return image_tensors

def main():
    print("=== Starting Image Preprocessing ===")

    # 1. Load and process images
    image_tensors = load_and_preprocess_images(IMAGE_DIR)

    # 2. Save processed images
    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(image_tensors, OUTPUT_PT)
    print(f"[INFO] Processed images saved to {OUTPUT_PT}")

    print("=== Image Preprocessing Complete ===")

if __name__ == "__main__":
    main()
