import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AthenaModel
from floorplan_dataset import FloorplanDataset

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU in use: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ------------------ CONFIG ------------------
BATCH_SIZE = 8            # Number of samples per batch
EPOCHS = 50               # Increased epochs for more thorough training
LEARNING_RATE = 0.0002    # Common choice for generative tasks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths to dataset files
TEXT_TOKENS_PATH = r"Q:\Athena\data\Processed Data\tokenized_texts.pt"
IMAGE_TENSORS_PATH = r"Q:\Athena\data\Processed Data\processed_images.pt"
CHECKPOINT_PATH = r"Q:\Athena\models\athena_trained.pth"

def train():
    # 1. LOAD DATASET
    print("[INFO] Loading dataset...")
    dataset = FloorplanDataset(TEXT_TOKENS_PATH, IMAGE_TENSORS_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. INIT MODEL
    print("[INFO] Initializing model...")
    model = AthenaModel().to(DEVICE)

    # 3. LOSS & OPTIMIZER
    # Mean Squared Error for comparing generated images to real images
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        total_loss = 0.0

        for batch_idx, (text_input, real_image) in enumerate(dataloader):
            # Move data to GPU if available
            text_input, real_image = text_input.to(DEVICE), real_image.to(DEVICE)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass: model generates an image
            generated_image = model(text_input, real_image)

            # Calculate MSE loss
            loss = loss_function(generated_image, real_image)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

        # (OPTIONAL) SAVE A CHECKPOINT EVERY 10 EPOCHS
        # if (epoch + 1) % 10 == 0:
        #     checkpoint_filename = f"athena_epoch_{epoch+1}.pth"
        #     torch.save(model.state_dict(), checkpoint_filename)
        #     print(f"[INFO] Saved checkpoint: {checkpoint_filename}")

    # 5. SAVE FINAL MODEL
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[INFO] Training complete! Final model saved at {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
