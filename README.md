# 🌐 Network Intrusion Detection System (NIDS) Using Deep Learning

A complete and user-friendly implementation of a **Network Intrusion Detection System (NIDS)** built using deep learning, machine learning, and statistical feature analysis.  
This project walks you through the full pipeline—from **PCA-based preprocessing**, to **model training**, to **model deployment**—with pre-trained models ready for immediate use.

---

## 🧭 Introduction

Traditional rule-based intrusion detection systems often fail to recognize new or evolving cyber threats.  
This project implements a **modern deep-learning-driven NIDS pipeline** that combines:

- PCA-driven dimensionality reduction  
- Deep learning autoencoders for anomaly detection  
- Machine learning classifiers (e.g., scikit-learn, XGBoost)  
- Feature engineering and statistical selection methods  

This repository is suitable for students, researchers, and engineers working on intelligent intrusion detection and cybersecurity analytics.

---

## 📚 Dataset Information

This project uses the **Cleaned & Preprocessed CICIDS2017 Dataset**, created by **Eric Anacleto Ribeiro** and published on **Kaggle** under a **CC0: Public Domain license**.

### 📄 Dataset: <a href="https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed">*Cleaned and Preprocessed CICIDS2017 Data for Machine Learning*</a>

**File Used:**
- `cicids2017_cleaned.csv` — A fully cleaned, merged, preprocessed version of CICIDS2017, containing raw but cleaned features ready for scaling and sampling after train/test split.

### 🧾 About the Original Dataset (CICIDS2017)

The original CICIDS2017 dataset captures both benign and malicious network traffic across multiple attack scenarios. While widely used, the raw files require heavy preprocessing for machine learning tasks.

### 🛠 Steps Performed by the Dataset Author

The cleaned version used in this project includes the following improvements:

#### **Data Cleaning & Integrity**
- Merged multiple CSV files into a unified dataset  
- Removed duplicate rows and duplicate columns  
- Replaced infinite values with NaN for consistency  
- Removed rows with missing values (<1% of dataset)

#### **Feature Engineering & Selection**
- Removed columns with only one unique value  
- Applied correlation analysis to remove highly correlated features (≥ 0.99)  
- Combined **Kruskal-Wallis H-test** with **Random Forest feature importance** to remove statistically irrelevant features  

#### **Label Refinement**
- Converted original `Label` column into a simplified `Attack Type` column  
- Grouped similar attack types (e.g., DoS Hulk + DoS GoldenEye → "DoS")  
- Removed extremely rare attack classes (e.g., Heartbleed, Infiltration) to avoid overfitting  

This cleaned dataset forms the **starting point** of the PCA and model training workflow in this project.

---

## 🔄 Project Workflow

This NIDS pipeline follows a structured, three-stage workflow:

### `PCA Analysis  →  NIDS Training  →  NIDS Deployment`

### 1️⃣ PCA Analysis (`PCA_Analysis.py` / `PCA_Analysis.ipynb`)
- Performed on the **Cleaned & Preprocessed CICIDS2017 dataset**  
- Reduces dimensionality  
- Extracts high-value features  
- Generates variance and correlation reports  

### 2️⃣ NIDS Training (`NIDS_Training.py`)
- Uses PCA results + cleaned dataset  
- Trains multiple models (Autoencoder, XGBoost, scikit-learn models)  
- Saves models, encoders, scalers, thresholds, and metadata to `saved_models/`

### 3️⃣ NIDS Deployment (`NIDS_Deploy.py`)
- Loads all pre-trained components  
- Processes new samples  
- Performs real-time intrusion detection  

---

## 🚀 Features

- **Deep Learning Autoencoder** for anomaly detection  
- **XGBoost + ML Pipelines** for high-accuracy classification  
- **Complete PCA Workflow** for dimensionality reduction  
- **Ready-to-Use Deployment Script**  
- **Pretrained Models Included**  
- **Feature importance analysis**  
- **Interactive notebooks for exploration**  

---

## 📁 Project Structure

```
Network-Intrusion-Detection-System/
│
├── .gradio/                              # Gradio interface cache
│   └── certificate.pem                   # SSL certificate
│
├── notebooks/                            # Interactive analysis
│   ├── Network_Intrusion_Detection_System.ipynb
│   └── PCA_Analysis.ipynb
│
├── saved_models/                         # ALL PRE-TRAINED MODELS
│   ├── trained_nids_model.pkl           # Main scikit-learn model
│   ├── trained_nids_model_xgb.model     # XGBoost model
│   ├── trained_nids_model_autoencoder.h5 # TensorFlow autoencoder
│   ├── trained_nids_model_scaler.pkl    # Feature scaler
│   ├── trained_nids_model_label_encoder.pkl # Label encoder
│   ├── trained_nids_model_features.pkl  # Feature list
│   ├── trained_nids_model_metadata.pkl  # Training metadata
│   └── trained_nids_model_threshold.pkl # Anomaly threshold
│
├── feature_importance_results.csv        # Feature analysis
├── LICENSE                              # MIT License
├── NIDS_Deploy.py                       # DEPLOYMENT SCRIPT
├── NIDS_Training.py                     # TRAINING SCRIPT
├── PCA_Analysis.py                      # PCA analysis script
├── trained_nids_simple.pkl          # Lightweight model
├── README.md                            # This file
└── requirements.txt                     # Dependencies
```
## 🎯 Quick Start - Google Colab (Recommended for beginners)

### PCA Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wIKLTrIim4bxRCpWY_f4tBAuyi8w25Qe?usp=drive_link)
1. Click the Colab badge above
2. Run the cell
3. Results are displayed at the bottom

### NIDS Training and Deployment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14nXSV3Mxhx4xEF5ut9o2RFFh3ayRrUuX?usp=sharing)

1. Click the Colab badge above
2. Select Run all
3. Interact with the Gradio interface at the bottom
---
## ⚙️ Installation

### 1. Clone the repository
```
git clone https://github.com/ScriptedLines404/Network-Intrusion-Detection-System-using-Deep-Learning.git
cd Network-Intrusion-Detection-System-using-Deep-Learning
```
### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
---

## 🧰 Usage

### 📊 PCA Analysis
Perform PCA and feature analysis:
```
python PCA_Analysis.py
```
This step generates:
* PCA variance reports
* Feature correlation files
* leaned feature lists saved to saved_models/

### 🧠 Training the Model
Train all NIDS models:
```
python NIDS_Training.py
```
The script:

* Trains ML + DL models
* Saves all artifacts automatically
* Generates metadata for deployment

### 🛡️ Deploying the NIDS
To run classification or anomaly detection:
```
python NIDS_Deploy.py
```
Loads all pre-trained models and performs inference on new network samples.

### 📑 Pretrained Models
Located in:
```
saved_models/
```
Includes:

* Autoencoder
* XGBoost classifier
* Scikit-learn model
* Scaler, encoder, metadata, threshold, features

This enables instant model deployment without the need for retraining.

---
## 🤝 Contributing

**Contributions are welcomed! 🙌**

1. 🍴 Fork the repository.
2. 🌿 Create a new branch for your feature or bugfix.
3. 🖊️ Write clear commit messages and include tests where possible.
4. 📬 Submit a pull request with a detailed description.

**Guidelines:**

* 🧹 Follow Python best practices.
* 📚 Keep code clean and well-documented.
* 📝 Update relevant documentation when making changes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

**Dataset License :** The dataset used—<a href="https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed">Cleaned and Preprocessed CICIDS2017 Data for Machine Learning</a> by Eric Anacleto Ribeiro—is under CC0: Public Domain.

---
## 🙌 Acknowledgments
* Eric Anacleto Ribeiro, for preparing the cleaned CICIDS2017 dataset used in this project

* The creators of the original CICIDS2017 dataset

* Open-source developers and the cybersecurity research community
---
## 🌟 About Me  

Hi, there!. I am Vladimir Illich Arunan, an engineering student with a deep passion for understanding the inner workings of the digital world. My goal is to master the systems that power modern technology—not just to build and innovate, but also to test their limits through cybersecurity.