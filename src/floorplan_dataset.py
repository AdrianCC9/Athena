import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FloorplanDataset(Dataset):
    def __init__(self, text_tokens_path, image_tensors_path, augment=False):
        """
        FloorplanDataset constructor.

        Args:
            text_tokens_path (str): Path to the .pt file containing {'image_ids': [...], 'tokenized_texts': ...}.
            image_tensors_path (str): Path to the .pt file containing preprocessed image tensors {filename: tensor}.
            augment (bool): Whether to apply data augmentation on-the-fly.
        """
        # 1. Load tokenized text data
        text_data = torch.load(text_tokens_path)
        self.image_ids = text_data["image_ids"]            # list of image filenames
        self.tokenized_texts = text_data["tokenized_texts"]  # tensor of shape [N, max_length]

        # 2. Load processed image tensors
        self.image_tensors = torch.load(image_tensors_path)   # dict { "image_id.png": tensor }

        # 3. Sanity check
        assert len(self.image_ids) == len(self.tokenized_texts), (
            f"Mismatch between number of image IDs ({len(self.image_ids)}) "
            f"and number of tokenized texts ({len(self.tokenized_texts)})!"
        )

        # 4. (Optional) Define augmentation transforms if requested
        #    We assume images are grayscale: shape (1, H, W)
        #    Many transforms can work directly on torch.Tensor images as of torchvision 0.8+.
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10)
                # Add more transforms here if desired, e.g.,
                # transforms.ColorJitter(brightness=0.2, contrast=0.2) - not typical for floorplans
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Get the image filename and text token
        image_id = self.image_ids[idx]             # e.g. "16381.png"
        text_tensor = self.tokenized_texts[idx]    # shape: [max_length]

        # 2. Retrieve the corresponding image tensor from the dictionary
        image_tensor = self.image_tensors.get(image_id, None)
        if image_tensor is None:
            raise ValueError(
                f"Image tensor not found for {image_id} in {list(self.image_tensors.keys())[:5]}..."
            )

        # 3. Apply augmentation if transform is defined
        if self.transform:
            # Note: The transforms will be applied in-place, returning a new augmented tensor
            image_tensor = self.transform(image_tensor)

        # 4. Return the text tensor and possibly augmented image tensor
        return text_tensor, image_tensor

if __name__ == "__main__":
    TEXT_TOKENS_PATH = "/Users/adrian/Athena/data/Processed Data/tokenized_texts.pt"
    IMAGE_TENSORS_PATH = "/Users/adrian/Athena/data/Processed Data/processed_images.pt"
    
    # Example usage: set augment=True to test random transformations
    dataset = FloorplanDataset(TEXT_TOKENS_PATH, IMAGE_TENSORS_PATH, augment=True)
    print(f"[INFO] Dataset size: {len(dataset)} samples.")

    # Test accessing a sample
    sample_text, sample_image = dataset[0]
    print(f"Sample text shape: {sample_text.shape}")
    print(f"Sample image shape: {sample_image.shape}")
