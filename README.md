<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:e94560&height=200&section=header&text=Brain%20Tumor%20Detection&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20%7C%20CNN%20%7C%20MRI%20Classification%20%7C%20Streamlit%20Web%20App&descAlignY=60&descSize=16&descColor=f0a0a0" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **An AI-powered web application that detects brain tumors from MRI scans using Convolutional Neural Networks — achieving 90%+ classification accuracy.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🧠 Overview

Brain tumors are one of the most critical and life-threatening conditions in medical science. Early and accurate detection dramatically improves patient outcomes. This project builds a **deep learning pipeline** that:

- Accepts MRI scan images as input
- Preprocesses and normalizes them using OpenCV
- Passes them through a trained CNN model
- Classifies whether a tumor is **present or absent**
- Displays results with **confidence scores** and **visual heatmaps** via an interactive Streamlit web app

> 💡 This project was featured in my application to the **Claude for Open Source Program by Anthropic**.

---

## 🎯 Features

- 🔬 **MRI Image Upload** — Drag-and-drop interface for uploading brain MRI scans
- 🤖 **CNN-based Prediction** — Trained deep learning model with 90%+ accuracy
- 📊 **Interactive Visualizations** — Plotly charts for prediction confidence and model performance
- 🖼️ **Image Preprocessing Pipeline** — Automated resizing, normalization, and augmentation using OpenCV
- 📈 **Performance Metrics** — Accuracy, loss curves, confusion matrix displayed in-app
- ⚡ **Real-time Results** — Instant predictions with probability scores

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | TensorFlow (CPU), PyTorch, Keras |
| **Web App** | Streamlit |
| **Image Processing** | OpenCV, Pillow |
| **Data & Visualization** | NumPy, Matplotlib, Plotly |
| **Language** | Python 3.9+ |

---

## 📂 Dataset

This project uses the **Brain MRI Images for Brain Tumor Detection** dataset.

- **Source:** [Kaggle — Brain MRI Images](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes:** 2 — `yes` (tumor present) / `no` (no tumor)
- **Total Images:** ~3,000 MRI scans
- **Format:** JPG / PNG

| Split | Images |
|---|---|
| Training | ~2,400 |
| Validation | ~300 |
| Testing | ~300 |

---

## 🏗️ Model Architecture

```
Input MRI Image (224 × 224 × 3)
        │
        ▼
┌─────────────────────┐
│  Conv2D (32)  + ReLU │  ← Feature extraction
│  MaxPooling2D        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Conv2D (64)  + ReLU │  ← Deeper features
│  MaxPooling2D        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Conv2D (128) + ReLU │  ← High-level features
│  MaxPooling2D        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Flatten             │
│  Dense (512) + ReLU  │
│  Dropout (0.5)       │
│  Dense (1) + Sigmoid │  ← Binary classification
└─────────────────────┘
        │
        ▼
  Output: Tumor / No Tumor
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/Dileep-kumarc/Brain-Tumor-Detection-using-CNN.git
cd Brain-Tumor-Detection-using-CNN
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
torch
torchvision
torchaudio
tensorflow-cpu
streamlit
numpy
pillow
matplotlib
opencv-python-headless
plotly
```

---

## 🚀 Usage

### Run the Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

### How to use the app:

1. **Upload** an MRI scan image (JPG/PNG)
2. Click **"Detect Tumor"**
3. View the **prediction result** with confidence score
4. Explore **visualizations** — probability chart, preprocessed image

### Train the model yourself

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train the CNN model
- Save the best model weights to `model/brain_tumor_model.h5`
- Display accuracy and loss curves

---

## 📊 Results

| Metric | Score |
|---|---|
| **Training Accuracy** | ~94% |
| **Validation Accuracy** | ~91% |
| **Test Accuracy** | **90%+** |
| **Loss (Test)** | ~0.21 |

> The model generalizes well across unseen MRI scans, demonstrating robust learning of tumor-indicative features.

---

## 📁 Project Structure

```
Brain-Tumor-Detection-using-CNN/
│
├── app.py                    # Streamlit web application
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
│
├── model/
│   └── brain_tumor_model.h5  # Saved trained model weights
│
├── dataset/
│   ├── yes/                  # MRI scans with tumor
│   └── no/                   # MRI scans without tumor
│
├── utils/
│   ├── preprocess.py         # Image preprocessing pipeline
│   └── visualize.py          # Visualization helpers
│
└── notebooks/
    └── EDA_and_Training.ipynb # Exploratory data analysis
```

---

#### 📦 Pre-trained Model Weights

| Model | Framework | Download |
|-------|-----------|----------|
| best_mri_classifier.pth | PyTorch | [Kaggle](https://www.kaggle.com/models/dileepkumarc/mri-and-non-mri-classification) |
| brain_tumor_classifier_augmented_finetuned.h5 | TensorFlow | [Kaggle](https://www.kaggle.com/models/dileepkumarc/mri-classifer) |
| brain_tumor_segmentation_unet.h5 | TensorFlow | [Kaggle](https://www.kaggle.com/models/dileepkumarc/segmention-of-mri) |
| tumor_size_model.h5 | TensorFlow | [Kaggle](https://www.kaggle.com/models/dileepkumarc/brain-tumor-size) |

## 👨‍💻 Author

<div align="center">

**Dileep Kumar C**
*Java Full Stack Developer | AI/ML Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dileep-kumar-3a5278268/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Dileep-kumarc)
[![Portfolio](https://img.shields.io/badge/Portfolio-e94560?style=for-the-badge&logo=vercel&logoColor=white)](https://dileepkumar-c.netlify.app)
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dileepkmrc@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*"The goal of AI is not to replace doctors, but to give them superpowers."*

⭐ **Star this repo if you found it useful!** ⭐

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e94560,50:16213e,100:1a1a2e&height=100&section=footer" width="100%"/>

</div>
