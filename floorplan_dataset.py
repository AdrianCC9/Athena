import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class FloorplanDataset(Dataset):
    def __init__(self, embeddings_path, images_csv_path, image_dir, transform=None):
        """
        Custom PyTorch Dataset for loading floorplan images and their associated text embeddings.
        :param embeddings_path: Path to the .pt file containing tokenized text embeddings.
        :param images_csv_path: Path to the CSV file containing cleaned image metadata.
        :param image_dir: Directory where the floorplan images are stored.
        :param transform: Optional transforms to apply to the images.
        """
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        self.images_df = pd.read_csv(images_csv_path)  # Load image metadata
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images_df)

    def __getitem__(self, idx):
        """
        Load an individual data sample (embedding and image).
        
        :param idx: Index of the sample to load.
        :return: Tuple (embedding, image_tensor)
        """
        # Get the text embedding
        embedding = self.embeddings[idx]

        # Load the image
        image_path = os.path.join(self.image_dir, self.images_df.iloc[idx]['image_path'])
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply any image transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)

        # Convert image to a tensor
        image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension (C, H, W)

        return embedding, image_tensor

if __name__ == "__main__":
    # Paths for testing
    embeddings_path = "data/tokenized_data/tokenized_human_annotations.pt"
    images_csv_path = "data/cleaned_data/cleaned_floorplan_images.csv"
    image_dir = "data/images"

    # Initialize the dataset
    dataset = FloorplanDataset(
        embeddings_path=embeddings_path,
        images_csv_path=images_csv_path,
        image_dir=image_dir,
    )

    # Perform checks
    print(f"Number of embeddings: {len(dataset.embeddings)}")
    print(f"Number of images: {len(dataset.images_df)}")

    assert len(dataset.embeddings) == len(dataset.images_df), \
        f"Mismatch: {len(dataset.embeddings)} embeddings and {len(dataset.images_df)} images."
