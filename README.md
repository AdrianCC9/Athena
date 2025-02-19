# Athena: AI-Powered Architectural Blueprint Generator

## **Project Overview**

**Athena** is an advanced AI-driven system designed to generate architectural blueprints from natural language descriptions. By integrating **deep learning, computer vision, and natural language processing (NLP)**, Athena translates user-provided specifications into structured architectural layouts.

This project leverages **multi-modal AI models** to process both textual and visual data, enabling an innovative approach to architectural design automation. The model learns from a dataset of annotated floorplans to generate new, context-aware blueprints based on user input.

## **Core Features**

### **1. AI-Powered Floorplan Generation**

- Accepts **natural language descriptions** of floorplans as input.
- Generates **detailed architectural blueprints** matching the given specifications.

### **2. Multi-Modal Deep Learning Architecture**

- **Text Processing:** Utilizes a **T5-based tokenizer** to convert natural language descriptions into structured representations.
- **Image Processing:** A **Convolutional Neural Network (CNN)** extracts spatial features from training floorplans.
- **Feature Fusion:** Combines text and image features to generate coherent and meaningful architectural layouts.

### **3. Scalable & Optimized Model Training**

- Implements **GPU-accelerated training** for efficient computation.
- Supports **data augmentation** techniques to enhance model generalization.
- Uses **PyTorch and torchvision** for state-of-the-art deep learning capabilities.

### **4. Model Output & Interpretation**

- Produces **256x256 grayscale blueprints**, optimized for architectural use.
- The model refines results through iterative learning cycles, ensuring high-quality output.

## **Technologies & Frameworks**

### **Machine Learning & Deep Learning**

- **PyTorch** – Core deep learning framework for model training and inference.
- **Torchvision** – Utilized for image transformations and CNN feature extraction.
- **Hugging Face Transformers** – Used for T5-based text tokenization.
- **CUDA (NVIDIA GPU Acceleration)** – Enables high-performance training on large datasets.

### **Data Processing & Visualization**

- **Pandas** – Efficiently handles large-scale dataset preprocessing.
- **NumPy** – Optimized array operations for numerical computation.
- **Matplotlib & PIL** – Provides visualization tools for blueprint images.

### **Model Architecture**

- **Text Encoder:** T5 Transformer-based tokenizer to convert textual descriptions into numerical embeddings.
- **Image Encoder:** A modified **ResNet18** CNN to process floorplan images and extract spatial features.
- **Feature Fusion:** A fully connected layer merges text embeddings with image features to create a shared latent representation.
- **Decoder:** A **transposed CNN (deconvolutional network)** generates blueprint images from the fused latent space.

## **Training Pipeline**

1. **Data Preprocessing**

   - Cleans and structures **text-image pairs** for training.
   - Tokenizes architectural descriptions using the **T5 model**.
   - Normalizes and augments blueprint images to improve model robustness.

2. **Model Training**

   - Uses **Mean Squared Error (MSE) Loss** to optimize image generation.
   - Implements **Adam optimizer** with learning rate adjustments.
   - Runs training in **mini-batches** for computational efficiency.

3. **Evaluation & Refinement**

   - Compares **generated blueprints** with ground-truth images.
   - Analyzes **loss trends** to refine architectural accuracy.
   - Saves trained models as **PyTorch checkpoints** for future inference.

## **Future Enhancements**

- **GAN-Based Floorplan Generation** – Incorporate adversarial training to improve realism.
- **3D Rendering Support** – Extend model capabilities to generate 3D architectural layouts.
- **Customizable Style Generation** – Train on diverse architectural styles for tailored outputs.
- **Real-World Data Integration** – Incorporate real-world floorplan datasets for enhanced learning.
