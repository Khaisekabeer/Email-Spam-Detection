# 🛡️ GuardianAI | Email Spam Classification System

A comprehensive, production-ready **Email Spam Detection System** featuring multiple classification models (**Naive Bayes, SVM, and LSTM**) integrated into a premium **Streamlit** web application.

---

## 🌟 Overview

GuardianAI leverages advanced **Natural Language Processing (NLP)** and **Machine Learning** to protect your inbox. By integrating traditional statistical models with deep learning architectures, it provides a robust, multi-layered approach to spam detection.

### 🧠 Integrated Models

1.  **Multinomial Naive Bayes (Fast)**: A classic probabilistic model known for its speed and effectiveness in word-based filtering.
2.  **Linear SVM (High Precision)**: A powerful Support Vector Machine model optimized for high-dimensional text data.
3.  **Bidirectional LSTM (Deep Analytics)**: A deep learning architecture using Long Short-Term Memory units to understand the sequential context and linguistic flow of emails.

---

## 📁 Project Structure

```text
Email-Spam-Detection/
├── models/                     # 🧠 Saved Model Artifacts (.joblib, .h5)
├── app.py                      # 🚀 Premium Streamlit Web Application
├── train_nb.py                 # 🧪 Naive Bayes Training Script
├── requirements.txt            # 📦 Project Dependencies
├── Untitled49.ipynb            # 🔬 LSTM Experimental Lab
└── Email-Spam-Detector.ipynb   # 🔬 SVM Experimental Lab
```

---

## 🚀 Getting Started

### 1️⃣ Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Khaisekabeer/Email-Spam-Detection.git
cd Email-Spam-Detection
pip install -r requirements.txt
```

### 2️⃣ Model Preparation
A unified training pipeline has been implemented to train all three models (Naive Bayes, SVM, and LSTM) using the **Enron Email Dataset**. This ensures consistent preprocessing and feature extraction across all models.

Run the following command to train all models:

```bash
python train_all_models.py
```

This will:
1.  **Load** the Enron dataset from Hugging Face.
2.  **Preprocess** the text (Cleaning, Tokenization, Stopword removal, Lemmatization).
3.  **Train** Naive Bayes, SVM (with calibrated probabilities), and Bidirectional LSTM models.
4.  **Evaluate** each model with accuracy, F1 score, and confusion matrices.
5.  **Save** all model artifacts to the `models/` folder and confusion matrix plots to `plots/`.

### 3️⃣ Launch the App
Run the Streamlit application to start classifying emails:

```bash
streamlit run app.py
```

---

## 🧪 Technologies Used

*   **Python**: Core programming language.
*   **NLP Tools**: NLTK (Stopwords, Stemming).
*   **Machine Learning**: Scikit-Learn (NB, SVM, TF-IDF).
*   **Deep Learning**: TensorFlow & Keras (Bidirectional LSTM).
*   **Web Framework**: Streamlit (Premium UI with Custom CSS).
*   **Data Handling**: Pandas, NumPy.

---

## 👨‍💻 Author

**Khaisekabeer**

---

## ⭐ Support the Project
If you find this project helpful, give it a star on GitHub! 🌟
