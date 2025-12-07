# Comparative Analysis of SOTA Models (Large Networks)

![Python](https://img.shields.io/badge/Python-3.14.1-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c)
![Device](https://img.shields.io/badge/Device-CUDA%20GPU-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📌 Project Overview
**Course:** CAP6415 - Computer Vision (Fall 2025)  
**Project Title:** SOTA Models Comparison (Large Networks)

This project conducts a rigorous comparison of relatively small State-of-the-Art (SOTA) deep learning architectures (<50 layers) for image classification. Unlike standard leaderboard benchmarks, we evaluate these models on a custom task using the **Oxford-IIIT Pet Dataset** to analyze their real-world adaptability, efficiency, and accuracy on a fine-grained classification task.

## 👥 Team Members
* **Kartavya Soni**
* **Jigar Purohit**
* **Vismay Gajera**

## 🏗️ Models Selected
We selected four distinct architectures to represent different design philosophies in computer vision. All models are initialized with ImageNet pre-trained weights and fine-tuned on the pet dataset.

| Model | Parameters | Architecture Type | Key Feature |
|-------|------------|-------------------|-------------|
| **EfficientNet-B0** | ~4.05M | Convolutional | Compound Scaling (Depth/Width/Res) |
| **MobileNet-V3-Large** | ~4.25M | Mobile-Optimized | Hardware-aware NAS & SE Blocks |
| **ResNet-34** | ~21.30M | Residual Network | Skip Connections |
| **ConvNeXt-Tiny** | ~27.85M | Modern CNN | Vision Transformer-inspired design |

## 📂 Dataset Details
* **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **Classes:** 37 Categories (Breeds of Cats and Dogs)
* **Data Format:** Images with associated `.mat` annotation files (handled automatically via Torchvision).
* **Task:** Fine-grained Image Classification.

## 🧠 Research Questions & Rigorous Analysis
To fulfill the project requirements, we aim to answer the following questions through our experiments:

1.  **Accuracy vs. Model Size:** Does the largest model (ConvNeXt-Tiny, ~28M params) actually outperform the smallest model (EfficientNet-B0, ~4M params) on a dataset of this size?
2.  **Hardware Efficiency:** Which model achieves the highest FPS (Frames Per Second) during inference on a local GPU? Is MobileNet-V3 significantly faster than ResNet-34 in a non-mobile environment?
3.  **Confusion & Errors:** Which specific breeds do *all* models fail to classify correctly? (e.g., distinguishing *Staffordshire Bull Terriers* from *Pit Bulls*).
4.  **Convergence Speed:** Which architecture reaches 80% validation accuracy in the fewest epochs?
5.  **Modern vs. Legacy:** Does the Transformer-inspired ConvNeXt architecture offer better transfer learning capabilities than the older ResNet-34 baseline?

## ⚙️ Methodology
To ensure a rigorous comparison, the project tracks the following metrics:
1.  **Top-1 Accuracy:** Final classification performance.
2.  **Training Efficiency:** Time taken to converge.
3.  **Inference Speed:** FPS (Frames Per Second) on local CUDA GPU.
4.  **Model Size:** Parameter count and memory footprint.

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.14.1 installed with the following libraries:
```bash
pip install torch torchvision pandas matplotlib scikit-learn
```

### 2. Run the Comparison
The main script handles dataset downloading, model initialization, and the training loop:
```bash
python run_comparison.py
```

## 📊 Results

| Metric | EfficientNet | MobileNet | ResNet | ConvNeXt |
| :--- | :---: | :---: | :---: | :---: |
| **Best Val Accuracy** | 92.17% | 94.49% | 93.60% | **97.48%** |
| **Training Time/Epoch** | 444.5s | **301.8s** | 310.5s | 474.1s |

> **Note:** **ConvNeXt** achieved the highest accuracy, while **MobileNet** was the fastest to train per epoch.

## 📜 License

This project is developed for educational purposes as part of the **CAP6415** curriculum.
