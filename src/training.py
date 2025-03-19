import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AthenaModel
from floorplan_dataset import FloorplanDataset
import time

# ------------------ CONFIG ------------------
BATCH_SIZE = 8            # Number of samples per batch
EPOCHS = 50               # Number of training epochs
LEARNING_RATE = 0.0002    # Common learning rate for generative tasks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths to dataset files
TEXT_TOKENS_PATH = r"Q:\Athena\data\Processed Data\tokenized_texts.pt"
IMAGE_TENSORS_PATH = r"Q:\Athena\data\Processed Data\processed_images.pt"
CHECKPOINT_PATH = r"Q:\Athena\models\athena_trained.pth"

def train():
    print("[SETUP] Checking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU in use: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. LOAD DATASET
    print("\n[INFO] Loading dataset...")
    dataset = FloorplanDataset(TEXT_TOKENS_PATH, IMAGE_TENSORS_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[INFO] Dataset loaded. Total samples: {len(dataset)}")
    print(f"[INFO] Steps per epoch: {len(dataloader)} (batch size = {BATCH_SIZE})")

    # 2. INIT MODEL
    print("\n[INFO] Initializing AthenaModel...")
    model = AthenaModel().to(DEVICE)
    print("[INFO] Model initialized and moved to device.")

    # 3. LOSS & OPTIMIZER
    print("\n[INFO] Defining loss function and optimizer...")
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    total_steps = EPOCHS * len(dataloader)  # total # of batches across all epochs
    global_step = 0  # tracks total batches processed

    start_time = time.time()  # measure total training time

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        total_loss = 0.0

        print(f"\n=== EPOCH [{epoch+1}/{EPOCHS}] ===")

        for batch_idx, (text_input, real_image) in enumerate(dataloader):
            global_step += 1

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

            # Accumulate loss for epoch average
            total_loss += loss.item()

            # Compute overall progress in training (for all epochs)
            progress_percent = (global_step / total_steps) * 100

            # Print batch details
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] | "
                  f"Step {global_step}/{total_steps} "
                  f"({progress_percent:.2f}% done) | "
                  f"Batch Loss: {loss.item():.6f}")

        # End of epoch: print average loss
        avg_loss = total_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f"[EPOCH SUMMARY] Epoch {epoch+1} finished.")
        print(f"  - Average Epoch Loss: {avg_loss:.6f}")
        print(f"  - Epoch Duration: {epoch_duration:.1f} sec")

        # (OPTIONAL) SAVE A CHECKPOINT EVERY X EPOCHS:
        # if (epoch + 1) % 10 == 0:
        #     checkpoint_filename = f"athena_epoch_{epoch+1}.pth"
        #     torch.save(model.state_dict(), checkpoint_filename)
        #     print(f"[INFO] Saved checkpoint: {checkpoint_filename}")

    # End of all epochs
    total_duration = time.time() - start_time
    print("\n[INFO] Training complete!")
    print(f"Total training time: {total_duration:.1f} seconds")

    # 5. SAVE FINAL MODEL
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[INFO] Final model saved at {CHECKPOINT_PATH}\n")

if __name__ == "__main__":
    train()
