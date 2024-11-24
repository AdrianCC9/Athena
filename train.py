import torch
from torch.utils.data import DataLoader
from model import UNetDecoder  # Import your model
from floorplan_dataset import FloorplanDataset  
from torch import nn, optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = UNetDecoder(input_dim=512, output_channels=1, img_size=128).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Loss function for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load dataset
dataset = FloorplanDataset(
    embeddings_path="data/tokenized_data/tokenized_human_annotations.pt",
    images_csv_path="data/cleaned_data/cleaned_floorplan_images.csv",
    image_dir="data/images",  # Ensure this points to the correct folder with images
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for embeddings, images in dataloader:
        embeddings, images = embeddings.to(device), images.to(device)  # Move to GPU
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
