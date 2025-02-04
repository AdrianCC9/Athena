import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from torchvision import transforms
from PIL import Image

# Step 1: Define Dataset Class
class FloorPlanDataset(Dataset):
    def __init__(self, image_dir, text_dir, tokenizer, max_length=512, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        text_filename = image_filename.replace(".png", ".txt")
        text_path = os.path.join(self.text_dir, text_filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image": image,  # Tensor for image
            "input_ids": tokens["input_ids"].squeeze(0),  # Tokenized text
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

# Step 2: Save Preprocessed Data
def save_preprocessed_data(dataloader, save_path):
    """Saves preprocessed data as PyTorch tensor for future training."""
    torch.save(dataloader, save_path)
    print(f"✅ Preprocessed data saved at: {save_path}")

# Step 3: Main Execution
if __name__ == "__main__":
    image_folder = "/Users/adrian/Athena/data/General Data/floorplan_image"
    text_folder = "/Users/adrian/Athena/data/General Data/human_annotated_tags"
    save_path = "/Users/adrian/Athena/data/preprocessed_data.pt"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
    ])

    dataset = FloorPlanDataset(image_folder, text_folder, tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Save preprocessed data
    save_preprocessed_data(dataloader, save_path)
