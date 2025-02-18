import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms

class FloorPlanDataset(Dataset):
    def __init__(self, image_dir, text_dir, tokenizer, max_length=512, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

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
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=3, embed_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 64 * 64, embed_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_folder = "/Users/adrian/Athena/data/General Data/floorplan_image"
    text_folder = "/Users/adrian/Athena/data/General Data/human_annotated_tags"
    dataset = FloorPlanDataset(
        image_dir=image_folder,
        text_dir=text_folder,
        tokenizer=tokenizer,
        max_length=512,
        transform=image_transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    images = batch["image"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    cnn = SimpleCNN(num_channels=3, embed_dim=512)
    image_embeddings = cnn(images)
