import torch
from torch.utils.data import Dataset
import os

class FloorplanDataset(Dataset):
    def __init__(self, text_tokens_path, image_tensors_path):
        
        # Load tokenized text data
        text_data = torch.load(text_tokens_path)
        self.image_ids = text_data["image_ids"]           # list of image filenames
        self.tokenized_texts = text_data["tokenized_texts"]  # tensor of shape [N, max_length]

        # Load processed image tensors
        self.image_tensors = torch.load(image_tensors_path)  # dict { "image_id.png": tensor }

        # Sanity check
        assert len(self.image_ids) == len(self.tokenized_texts), (
            f"Mismatch between number of image IDs ({len(self.image_ids)}) "
            f"and number of tokenized texts ({len(self.tokenized_texts)})!"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]             # e.g. "16381.png"
        text_tensor = self.tokenized_texts[idx]    # shape: [max_length]

        # Retrieve the corresponding image tensor from the dictionary
        image_tensor = self.image_tensors.get(image_id, None)
        if image_tensor is None:
            raise ValueError(f"Image tensor not found for {image_id} in {list(self.image_tensors.keys())[:5]}...")

        return text_tensor, image_tensor

if __name__ == "__main__":

    TEXT_TOKENS_PATH = "/Users/adrian/Athena/data/Processed Data/tokenized_texts.pt"
    IMAGE_TENSORS_PATH = "/Users/adrian/Athena/data/Processed Data/processed_images.pt"
    
    dataset = FloorplanDataset(TEXT_TOKENS_PATH, IMAGE_TENSORS_PATH)
    print(f"[INFO] Dataset size: {len(dataset)} samples.")

    # Test accessing a sample
    sample_text, sample_image = dataset[0]
    print(f"Sample text shape: {sample_text.shape}")
    print(f"Sample image shape: {sample_image.shape}")
