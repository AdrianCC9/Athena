import torch
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class FloorplanDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path, images_csv_path, image_dir, transform=None, image_size=(128, 128)):
        """
        Custom PyTorch Dataset for loading floorplan images and their associated text embeddings.

        Args:
            embeddings_path (str): Path to the .pt file containing combined text embeddings and image IDs.
            images_csv_path (str): Path to the CSV file containing cleaned image metadata.
            image_dir (str): Directory where the floorplan images are stored.
            transform (callable, optional): Optional transforms to apply to the images.
            image_size (tuple, optional): Desired size (H, W) of the images.
        """
        # Load tokenized embeddings and image IDs
        data = torch.load(embeddings_path)
        self.embeddings = data['input_ids']  # Tokenized input IDs (shape: [N, max_length])
        self.image_ids = data['image_ids']  # List of image IDs

        # Load image metadata
        images_df = pd.read_csv(images_csv_path)
        images_df['image_id'] = images_df['image_path'].apply(lambda x: os.path.basename(x))
        self.image_paths = images_df.set_index('image_id')['image_path'].to_dict()

        # Verify that embeddings and image IDs are aligned
        assert len(self.embeddings) == len(self.image_ids), \
            f"Mismatch: {len(self.embeddings)} embeddings and {len(self.image_ids)} image IDs."

        # Default transformations: Resize, Convert to Tensor, Normalize
        self.transform = transform if transform else Compose([
            Resize(image_size),  # Resize to the desired dimensions
            ToTensor(),  # Convert image to tensor (C, H, W)
            Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
        ])

        # Store image directory
        self.image_dir = image_dir

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Fetch a single sample (embedding and image tensor) from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (embedding, image_tensor)
        """
        # Load the text embedding
        embedding = self.embeddings[idx]  # Shape: [max_length]

        # Get the corresponding image ID
        image_id = self.image_ids[idx]

        # Get the image path
        image_path = self.image_paths.get(image_id)
        if image_path is None:
            raise ValueError(f"No image path found for image ID: {image_id}")
        full_image_path = os.path.join(self.image_dir, os.path.basename(image_path))

        # Check if the image file exists
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image file not found: {full_image_path}")

        # Load the image
        image = Image.open(full_image_path).convert('L')  # Convert to grayscale

        # Apply image transformations
        image_tensor = self.transform(image)  # Shape: [1, H, W]

        return embedding, image_tensor

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Paths for dataset components
    embeddings_path = r"Q:/Athena/data/tokenized_data/tokenized_combined_annotations.pt"
    images_csv_path = r"Q:/Athena/data/cleaned_data/cleaned_floorplan_images.csv"
    image_dir = r"Q:/adria/Documents/Tell2Design Data/General Data/floorplan_image"

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
