# Athena: AI-Driven Floor Plan Generator

## Overview
Athena is a complete end-to-end solution for automatically generating floor plan layouts based on text descriptions. By combining natural language processing (NLP) with image processing techniques, Athena can interpret user requests such as “a two-bedroom apartment with a large living room and a kitchen on the north side” and produce a corresponding floor plan blueprint.

## Key Features
- **Text-to-Floor Plan Conversion**: Accepts text input describing rooms, sizes, and relationships, and generates a tailored floor plan.
- **Large-Scale Data Support**: Trained on 80,000+ floor plan images with AI-generated annotations, ensuring broad coverage.
- **Hybrid Approach**: Starts with artificial text descriptions, then finetunes on a curated set of human-annotated floor plans for better realism.
- **Modular Code Structure**: Separate scripts for data cleaning, text tokenization, image preprocessing, dataset creation, and training.

## Tools and Technologies
- **Python** for scripting and data pipelines
- **PyTorch** for deep learning tasks and model training
- **Hugging Face Transformers (T5)** for text tokenization and encoding
- **CNN (Convolutional Neural Network)** for feature extraction from images
- **Pandas, NumPy, PIL** for data manipulation and image processing
- **Matplotlib** for data visualization

## Results
Currently **in progress**. Early testing indicates that the model learns to match floor plan images to text descriptions effectively, but we are continually refining the accuracy and realism of generated layouts.

## How to Run
1. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
