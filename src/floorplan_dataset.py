import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class FloorplanDataset(Dataset):
    def __init__(self, embeddings_path, images_csv_path, image_dir, transform=None, image_size=(128, 128)):
        
        self.embeddings = torch.load(embeddings_path)  # Load pre-tokenized text embeddings
        self.images_df = pd.read_csv(images_csv_path)  # Load image metadata
        self.image_dir = image_dir

        # Default transformations: Resize, Convert to Tensor, Normalize
        self.transform = transform if transform else Compose([
            Resize(image_size),  # Resize to the desired dimensions
            ToTensor(),  # Convert image to tensor (C, H, W)
            Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
        ])

        # Ensure dataset alignment
        assert len(self.embeddings) == len(self.images_df), \
            f"Mismatch: {len(self.embeddings)} embeddings and {len(self.images_df)} images."

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images_df)

    def __getitem__(self, idx):
        """
        Fetch a single sample (embedding and image tensor) from the dataset.
        
        :param idx: Index of the sample.
        :return: Tuple (embedding, image_tensor)
        """
        # Load the text embedding
        embedding = self.embeddings[idx]

        # Load the image
        image_path = os.path.join(self.image_dir, self.images_df.iloc[idx]['image_path'])
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply image transformations
        image_tensor = self.transform(image)

        return embedding, image_tensor


if __name__ == "__main__":
    # Paths for dataset components
    embeddings_path = "/Users/adrian/Athena/data/tokenized_annotations.pt"
    images_csv_path = "/Users/adrian/Athena/data/Cleaned Data/cleaned_floorplan_images.csv"
    image_dir = "/Users/adrian/Documents/Tell2Design Data/General Data/floorplan_image"

    # Initialize the dataset
    dataset = FloorplanDataset(
        embeddings_path=embeddings_path,
        images_csv_path=images_csv_path,
        image_dir=image_dir,
    )

    # Perform basic checks
    print(f"Dataset contains {len(dataset)} samples.")

    # Test loading a single sample
    embedding, image_tensor = dataset[0]
    print(f"Embedding shape: {embedding.shape}")
    print(f"Image tensor shape: {image_tensor.shape}")
