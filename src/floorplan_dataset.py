import torch
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class FloorplanDataset(torch.utils.data.Dataset):
    def __init__(self, human_embeddings_path, artificial_embeddings_path, images_csv_path, image_dir, transform=None, image_size=(128, 128)):
        """
        Custom PyTorch Dataset for loading floorplan images and their associated text embeddings.
        :param human_embeddings_path: Path to the .pt file containing human text embeddings.
        :param artificial_embeddings_path: Path to the .pt file containing artificial text embeddings.
        :param images_csv_path: Path to the CSV file containing cleaned image metadata.
        :param image_dir: Directory where the floorplan images are stored.
        :param transform: Optional transforms to apply to the images.
        :param image_size: Tuple indicating the desired size (H, W) of the images.
        """
        # Load human and artificial embeddings
        human_embeddings = torch.load(human_embeddings_path)
        artificial_embeddings = torch.load(artificial_embeddings_path)

        # Load image metadata
        self.images_df = pd.read_csv(images_csv_path)
        self.image_dir = image_dir

        # Map embeddings to image IDs
        human_annotations = dict(zip(self.images_df["image_id"], human_embeddings))
        artificial_annotations = dict(zip(self.images_df["image_id"], artificial_embeddings))

        # Combine human and artificial annotations
        self.embeddings = []
        for image_id in self.images_df["image_id"]:
            if image_id in human_annotations:
                self.embeddings.append(human_annotations[image_id])  # Use human annotation if available
            elif image_id in artificial_annotations:
                self.embeddings.append(artificial_annotations[image_id])  # Fallback to artificial
            else:
                raise ValueError(f"No annotation found for image ID: {image_id}")

        # Ensure embeddings are a PyTorch tensor
        self.embeddings = torch.stack(self.embeddings)

        # Default transformations: Resize, Convert to Tensor, Normalize
        self.transform = transform if transform else Compose([
            Resize(image_size),  # Resize to the desired dimensions
            ToTensor(),  # Convert image to tensor (C, H, W)
            Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
        ])

        # Final alignment check
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
    human_embeddings_path = r"Q:/Athena/data/tokenized_data/tokenized_human_annotations.pt"
    artificial_embeddings_path = "Q:/Athena/data/tokenized_data/tokenized_artificial_annotations.pt"
    images_csv_path = r"Q:/Athena/data/cleaned_data/cleaned_floorplan_images.csv"
    image_dir = r"Q:/adria/Documents/Tell2Design Data/General Data/floorplan_image"

    # Initialize the dataset
    dataset = FloorplanDataset(
        human_embeddings_path=human_embeddings_path,
        artificial_embeddings_path=artificial_embeddings_path,
        images_csv_path=images_csv_path,
        image_dir=image_dir,
    )

    # Perform basic checks
    print(f"Dataset contains {len(dataset)} samples.")

    # Test loading a single sample
    embedding, image_tensor = dataset[0]
    print(f"Embedding shape: {embedding.shape}")
    print(f"Image tensor shape: {image_tensor.shape}")
