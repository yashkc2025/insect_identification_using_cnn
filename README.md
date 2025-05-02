# 🦗 Carabid Beetle Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project develops a deep learning model to classify insect species based on images with **86.26% accuracy**. The dataset consists of **63,364 insect specimens** from the **Natural History Museum, London**, focusing on **British carabid beetles**. The goal is to automate insect species identification using computer vision techniques.

![Sample Insects](https://github.com/user-attachments/assets/df9c4def-f63b-4e6f-98b1-295313bcd06a)


## Dataset

- **63,364 specimens** scanned from insect drawers
- Images labeled with **GBIF species IDs**
- Each folder corresponds to a specific insect species ID (e.g., `1035167`)
- Dataset provided by Natural History Museum, London

## Features

- 🔍 Automated insect species identification
- 🧠 Deep learning with ResNet-50 architecture
- 📊 Performance visualization and model evaluation
- 🖼️ Inference on new insect images
- 🔄 Early stopping mechanism to prevent overfitting

## Installation

```bash
# Clone the repository
git clone https://github.com/yashkc2025/insect_identification_using_cnn
```

## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
Pillow>=8.0.0
tqdm>=4.50.0
```

## Model Architecture

The project uses **ResNet-50** for image classification due to its high performance in recognizing complex patterns in images. The model is pretrained on ImageNet and fine-tuned on the insect dataset.

## Methodology

1. **Data Preprocessing**: 
   - Resize images to 224x224 pixels
   - Normalize using ImageNet mean and standard deviation
   - Split dataset into training (80%) and validation (20%) sets

2. **Training**:
   - Cross-entropy loss function
   - Adam optimizer with learning rate of 0.001
   - Early stopping based on validation accuracy
   - Model checkpointing to save best performing model

3. **Evaluation**:
   - Track training and validation loss
   - Monitor validation accuracy
   - Visualize learning curves

## Results

The model's performance is evaluated using:
- Accuracy metrics on validation set
- Loss curves to assess training convergence
- Visual inspection of classification results

![performance](https://github.com/user-attachments/assets/e8af3ecc-8a39-425e-a300-73e4fcf89b3e)

## Acknowledgments

- Natural History Museum, London for the insect specimen dataset
- PyTorch team for the deep learning framework
- ResNet developers for the CNN architecture
