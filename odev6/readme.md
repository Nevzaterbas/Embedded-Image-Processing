# EE4065 – Embedded Digital Image Processing  
## Homework 6 – Handwritten Digit Recognition using CNNs

This repository contains the implementation of the end-of-chapter application **“Handwritten Digit Recognition from Digital Images”** described in **Section 13.7** of the course textbook:

> C. Ünsalan, B. Höke, and E. Atmaca,  
> *Embedded Machine Learning with Microcontrollers: Applications on STM32 Boards*,  
> Springer Nature, 2025.

The objective of this homework is to implement handwritten digit recognition using **multiple CNN architectures** and to compare their performance under the same experimental conditions.

---

## Problem Description
Handwritten digit recognition is a classical image classification problem where grayscale images of digits (0–9) are classified into ten distinct classes. Convolutional Neural Networks (CNNs) are particularly suitable for this task due to their ability to automatically extract spatial features from images.

From an embedded systems perspective, model complexity, parameter count, and computational cost are as important as classification accuracy.

---

## Dataset
- **Dataset:** MNIST (offline dataset)
- **Classes:** 10 (digits 0–9)
- **Image preprocessing:**
  - Normalization to [0, 1]
  - Resizing to **64 × 64**
  - Conversion to 3-channel images
- **Data loading:** Implemented using `tf.data` pipeline to ensure memory-efficient training

---

## Implemented CNN Models
The following CNN models were implemented and evaluated:

### 1. SqueezeNet-like CNN
- Lightweight architecture designed for embedded and resource-constrained systems
- Uses Fire modules to reduce parameter count

### 2. EfficientNet-B0
- More complex and deeper architecture
- Optimized compound scaling of depth, width, and resolution
- Higher computational cost compared to SqueezeNet

All models were trained **from scratch** (no pretrained weights) using the same dataset.

---

## Training Configuration
- Optimizer: Adam  
- Learning rate: 0.001  
- Loss function: Sparse Categorical Cross-Entropy  
- Input resolution: **64 × 64**
- Training performed on **CPU-only environment**

Due to computational limitations, batch sizes were adjusted per model to ensure feasible training time.

---

## Experimental Results

| Model            | Test Accuracy | Parameters | Training Time (s) | Epochs | Batch Size |
|------------------|---------------|------------|-------------------|--------|------------|
| SqueezeNet-like  | 0.098         | 727,626    | 1151.6            | 8      | 64         |
| EfficientNet-B0  | 0.990         | 4,062,381  | 6951.4            | 8      | 32         |

---

## Discussion
The SqueezeNet-like model has a significantly lower parameter count, making it suitable for embedded systems with strict memory constraints. However, with limited training epochs and without pretrained weights, its classification performance remained close to random guessing.

In contrast, EfficientNet-B0 achieved very high classification accuracy, demonstrating its strong representational capacity. This improvement comes at the cost of increased model size and substantially longer training time, which may limit its applicability in resource-constrained embedded platforms.

These results highlight the trade-off between **model complexity and performance**, which is a key consideration in embedded machine learning applications.

---

## How to Run
Activate the virtual environment and run the training script:

```bash
python src/train/train_tf.py --model squeezenet --epochs 8 --batch 64 --img 64
python src/train/train_tf.py --model efficientnet_b0 --epochs 8 --batch 32 --img 64
