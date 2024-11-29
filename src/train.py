import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from floorplan_dataset import FloorplanDataset  # Import your dataset
from model import FloorplanModel  # Import your model

# Paths to data
EMBEDDINGS_PATH = "Q:/Athena/data/tokenized_data/tokenized_human_annotations.pt"
IMAGES_CSV_PATH = "Q:\Athena\data\cleaned_data\cleaned_floorplan_images.csv"
IMAGE_DIR = "Q:\adria\Documents\Tell2Design Data\General Data\floorplan_image"

# Training configuration
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
OUTPUT_DIM = 128  # Same as the model's output dimension

# Initialize dataset and DataLoader
dataset = FloorplanDataset(
    embeddings_path=EMBEDDINGS_PATH,
    images_csv_path=IMAGES_CSV_PATH,
    image_dir=IMAGE_DIR
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
model = FloorplanModel(output_dim=OUTPUT_DIM)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Example: Mean Squared Error Loss for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0.0

    print(f"Starting Epoch {epoch+1}/{EPOCHS}...")

    for batch_idx, (embeddings, images) in enumerate(dataloader):
        # Move data to the same device as the model
        embeddings, images = embeddings.to(device), images.to(device)

        # Forward pass
        outputs = model(embeddings, images)

        # Compute loss
        loss = criterion(outputs, images.view(outputs.shape))  # Match output shape to expected

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print updates every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "Q:/Athena/model/floorplan_model.pth")
print("Model saved to Q:/Athena/model/floorplan_model.pth")