# 🔍 SOTA Models Comparison: Large Networks

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Course:** CAP6415 - Computer Vision (Fall 2025)  
**Project Type:** Comparative Analysis of Deep Learning Architectures

---

## 📌 Project Overview

This project conducts a rigorous comparative analysis of four distinct State-of-the-Art (SOTA) deep learning architectures. While these models are capable of scaling to "large networks," we focus on their **lightweight variants (<50 layers)** to evaluate efficiency, accuracy, and transfer learning capabilities on limited hardware resources.

Unlike standard leaderboard benchmarks (which often use ImageNet or CIFAR), this project evaluates performance on a custom fine-grained classification task using the **Oxford-IIIT Pet Dataset**.

### 👥 Team Members
* **Kartavya Soni**
* **Jigar Purohit**
* **Vismay Gajera**

---

## 🧠 Models Selected

We selected four architectures representing different "eras" and design philosophies in Computer Vision. All models were initialized with ImageNet pre-trained weights and fine-tuned.

| Model | Parameters | Architecture Philosophy | Why we chose it? |
| :--- | :--- | :--- | :--- |
| **ResNet-34** | ~21.8M | Deep Residual Learning | The standard baseline for modern CV tasks. Uses skip connections to solve vanishing gradients. |
| **EfficientNet-B0** | ~5.3M | Compound Scaling | Focuses on optimizing depth, width, and resolution simultaneously for maximum efficiency. |
| **MobileNet-V3-Large** | ~5.4M | Hardware-Aware NAS | Designed specifically for mobile/edge devices using Neural Architecture Search (NAS). |
| **ConvNeXt-Tiny** | ~28.6M | Modern CNN (Transformer-like) | A pure ConvNet modeled after Vision Transformers (ViT), featuring large kernels and layer norms. |

---

## 📂 Dataset Details

* **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **Problem Type:** Fine-Grained Image Classification
* **Classes:** 37 (Breeds of Cats and Dogs)
* **Preprocessing:**
    * Images resized to `224x224`.
    * Normalization using ImageNet mean/std.
    * **Augmentation:** Random horizontal flips and rotations were applied during training to prevent overfitting.

---

## ⚙️ Methodology

We implemented a custom training loop using **PyTorch** with the following rigorous standards:

1.  **Transfer Learning:** Freezing the backbone initially, then fine-tuning specific layers.
2.  **Optimization:** Used `AdamW` optimizer with a dynamic Learning Rate Scheduler (`ReduceLROnPlateau`).
3.  **Regularization:** Implementation of Early Stopping to prevent overfitting.
4.  **Performance Metrics:**
    * **Top-1 Accuracy:** Classification success rate.
    * **Inference Speed (FPS):** Measured on local CUDA-enabled GPU.
    * **Training Time:** Time taken to converge per epoch.

---

## 📊 Results & Analysis

After training over 10-15 epochs, the models yielded the following performance metrics on the validation set:

| Model | Accuracy (%) | Training Time (s/epoch) | Inference Speed (FPS) |
| :--- | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | **97.48%** | ~474s | Medium |
| **MobileNet-V3-L** | 94.49% | **~301s** | **High** |
| **ResNet-34** | 93.60% | ~310s | Medium |
| **EfficientNet-B0** | 92.17% | ~444s | High |

### 🔑 Key Findings
1.  **Modern Architecture Wins:** **ConvNeXt-Tiny** significantly outperformed the others, proving that modern Transformer-inspired CNN designs offer superior feature extraction for fine-grained tasks.
2.  **Efficiency Champion:** **MobileNet-V3** provided the best trade-off, achieving nearly 94.5% accuracy while being the fastest to train and run. It is the ideal candidate for real-time deployment.
3.  **The ResNet Baseline:** ResNet-34 remains a solid, stable performer but is starting to show its age compared to newer architectures like ConvNeXt.

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.10+ (Tested on 3.14.1)
* CUDA-enabled GPU (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/VismayGajera112/CAP6415_F25_project-SOTA-models-comparison.git
cd CAP6415_F25_project-SOTA-models-comparison
```

### 2. Create virtual environment and activate it
```bash
# for windows:
python -m venv .venv
.venv\Scripts\activate

#for macos/Linux
python3 -m venv .venv
source ./.venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Comparison
To download the dataset and run the training/benchmarking pipeline:

```bash 
# open the notebook and click on Run all the cells
jupyter notebook SOTA_Models_Comparison.ipynb
```

---

## 📁 Project Structure

```text
├── Data/                   # Dataset storage (downloaded automatically)
├── Results/                # Confusion matrices and training logs
├── saved_models/           # .pth files of the best performing weights
├── SOTA_Models_Comparison.ipynb  # Main source code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📜 Acknowledgements

* **Oxford-IIIT Pet Dataset:** Created by Parkhi et al.
* **PyTorch Image Models (timm):** For providing pretrained model architectures.
* **Florida Atlantic University:** CAP6415 Course Materials.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> **Note:** This project was developed for educational purposes as part of the **CAP6415** curriculum (Fall 2025).
