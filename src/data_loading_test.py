import torch
from torch.utils.data import DataLoader
from floorplan_dataset import FloorplanDataset  # Ensure this matches your file name and class

# Paths to your data
embeddings_path = "/Users/adrian/Athena/data/tokenized_annotations.pt"
images_csv_path = "/Users/adrian/Athena/data/Cleaned Data/cleaned_floorplan_images.csv"
image_dir = "/Users/adrian/Documents/Tell2Design Data/General Data/floorplan_image"

# Initialize the FloorplanDataset
dataset = FloorplanDataset(
    embeddings_path=embeddings_path,
    images_csv_path=images_csv_path,
    image_dir=image_dir,
)

# Initialize the DataLoader
batch_size = 16  # Adjust based on your available memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test loading batches
for batch_idx, (embeddings, images) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Embeddings Shape: {embeddings.shape}")  # Expected: [batch_size, 128]
    print(f"  Images Shape: {images.shape}")  # Expected: [batch_size, 1, 128, 128] (grayscale)
    
    # Break after first batch to avoid processing the entire dataset in this test
    break
