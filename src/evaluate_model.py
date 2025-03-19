import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar
from model import AthenaModel
from floorplan_dataset import FloorplanDataset
import random

# ---------------- CONFIGURATION ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Q:/Athena/models/athena_trained.pth"
TEXT_TOKENS_PATH = "Q:/Athena/data/Processed Data/tokenized_texts.pt"
IMAGE_TENSORS_PATH = "Q:/Athena/data/Processed Data/processed_images.pt"
LOSS_LOG_PATH = "Q:/Athena/logs/loss_log.json"

MAX_SAMPLES = 5000  # Limit dataset for faster evaluation
BATCH_SIZE = 8  # Number of samples per batch

# ---------------- LOAD TRAINED MODEL ----------------
print("[INFO] Loading trained model...")
model = AthenaModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Model loaded successfully.")

# ---------------- LOAD DATASET (SUBSET) ----------------
print("[INFO] Loading dataset...")
dataset = FloorplanDataset(TEXT_TOKENS_PATH, IMAGE_TENSORS_PATH)

# Randomly sample a subset of MAX_SAMPLES
subset_indices = random.sample(range(len(dataset)), min(MAX_SAMPLES, len(dataset)))
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"[INFO] Using {len(subset_dataset)} samples for evaluation.")

# ---------------- IMAGE CONVERSION FUNCTION ----------------
def grayscale_to_rgb(image):
    """
    Convert a grayscale image (1, H, W) to RGB (3, H, W) by repeating channels.
    """
    return image.repeat(3, 1, 1)  # Converts 1-channel grayscale to 3-channel RGB

# ---------------- SSIM & PSNR COMPUTATION ----------------
def compute_ssim_psnr(real, generated):
    """
    Compute Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
    """
    real_np = real.cpu().squeeze().numpy()
    generated_np = generated.cpu().squeeze().numpy()
    ssim_score = ssim(real_np, generated_np, data_range=1.0)
    mse = np.mean((real_np - generated_np) ** 2)
    psnr_score = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
    return ssim_score, psnr_score

# ---------------- FRECHET INCEPTION DISTANCE (FID) ----------------
def compute_fid(real_images, generated_images, batch_size=4):
    """
    Compute Frechet Inception Distance (FID) using mini-batches for efficiency.
    """
    inception = inception_v3(pretrained=True, transform_input=True).to(DEVICE)
    inception.fc = torch.nn.Identity()  # Remove classification layer
    inception.eval()

    real_images_rgb = torch.stack([grayscale_to_rgb(img) for img in real_images])
    generated_images_rgb = torch.stack([grayscale_to_rgb(img) for img in generated_images])

    real_features_list, gen_features_list = [], []

    with torch.no_grad():
        for i in range(0, len(real_images_rgb), batch_size):
            real_batch = real_images_rgb[i : i + batch_size].to(DEVICE)
            gen_batch = generated_images_rgb[i : i + batch_size].to(DEVICE)

            real_features_list.append(inception(real_batch).cpu().numpy())
            gen_features_list.append(inception(gen_batch).cpu().numpy())

    real_features = np.concatenate(real_features_list, axis=0)
    gen_features = np.concatenate(gen_features_list, axis=0)

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(sigma_real @ sigma_gen))
    return fid

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two feature tensors.
    """
    tensor1 = tensor1.view(1, -1)
    tensor2 = tensor2.view(1, -1)
    return F.cosine_similarity(tensor1, tensor2).item()

# ---------------- EVALUATION LOOP WITH PROGRESS BAR ----------------
ssim_scores, psnr_scores, fid_scores, cosine_similarities = [], [], [], []

print("\n[INFO] Evaluating model performance...")
with torch.no_grad():
    for batch_idx, (text_input, real_images) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches", unit="batch"):
        text_input, real_images = text_input.to(DEVICE), real_images.to(DEVICE)
        generated_images = model(text_input, real_images)

        for i in range(len(real_images)):
            ssim_score, psnr_score = compute_ssim_psnr(real_images[i], generated_images[i])
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)

        fid_score = compute_fid(real_images, generated_images)  # Now runs faster
        fid_scores.append(fid_score)

        cosine_sim = cosine_similarity(real_images, generated_images)
        cosine_similarities.append(cosine_sim)

# Compute averages
avg_ssim = np.mean(ssim_scores)
avg_psnr = np.mean(psnr_scores)
avg_fid = np.mean(fid_scores)
avg_cosine = np.mean(cosine_similarities)

# ---------------- PRINT RESULTS ----------------
print("\n=== Evaluation Results ===")
print(f"📌 Average SSIM: {avg_ssim:.4f} (Higher is better, max = 1)")
print(f"📌 Average PSNR: {avg_psnr:.2f} dB (Higher is better, >30 is good)")
print(f"📌 Average FID: {avg_fid:.2f} (Lower is better, <50 is good)")
print(f"📌 Average Cosine Similarity: {avg_cosine:.4f} (Higher is better, max = 1)")

# ---------------- PLOT LOSS CURVE ----------------
if os.path.exists(LOSS_LOG_PATH):
    print("\n[INFO] Plotting training loss curve...")
    with open(LOSS_LOG_PATH, "r") as f:
        loss_data = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_data)), loss_data, marker='o', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
else:
    print("[WARNING] No loss log found. Skipping loss curve plot.")


[INFO] Evaluating model performance...
Processing Batches:   0%|                                                                                                                                                                              | 0/625 [00:00<?, ?batch/s]Q:\Athena\.windows_venv\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
Q:\Athena\.windows_venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=Inception_V3_Weights.IMAGENET1K_V1`. You can also use `weights=Inception_V3_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Processing Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [1:18:48<00:00,  7.57s/batch] 

=== Evaluation Results ===
📌 Average SSIM: 0.8047 (Higher is better, max = 1)
📌 Average PSNR: 14.53 dB (Higher is better, >30 is good)
📌 Average FID: 482.19-0.00j (Lower is better, <50 is good)
📌 Average Cosine Similarity: 0.9642 (Higher is better, max = 1)
[WARNING] No loss log found. Skipping loss curve plot.
PS Q:\Athena> 


