#  | Email Spam Classification System

A comprehensive, production-ready **Email Spam Detection System** featuring multiple classification models (**Naive Bayes, SVM, and LSTM**) integrated into a premium **Streamlit** web application.

---

##  Overview

GuardianAI leverages advanced **Natural Language Processing (NLP)** and **Machine Learning** to protect your inbox. By integrating traditional statistical models with deep learning architectures, it provides a robust, multi-layered approach to spam detection.

### Integrated Models

1.  **Multinomial Naive Bayes (Fast)**: A classic probabilistic model known for its speed and effectiveness in word-based filtering.
2.  **Linear SVM (High Precision)**: A powerful Support Vector Machine model optimized for high-dimensional text data.
3.  **Bidirectional LSTM (Deep Analytics)**: A deep learning architecture using Long Short-Term Memory units to understand the sequential context and linguistic flow of emails.

---

##  Project Structure

```text
Email-Spam-Detection/
├── models/                     # Trained model artifacts
├── app.py                      # Main Streamlit application
├── train_nb.py                 # Naive Bayes training script
├── train_svm.py                # SVM training script
├── train_lstm.py               # LSTM training script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## Getting Started

### 1 Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Khaisekabeer/Email-Spam-Detection.git
cd Email-Spam-Detection
pip install -r requirements.txt
```

### 2 Model Preparation
The training pipeline is split into modular scripts to allow independent training and deployment of each model.

To train the models, run the following commands:

```bash
# Train Naive Bayes (Fast)
python train_nb.py

# Train SVM (High Precision)
python train_svm.py

# Train LSTM (Deep Analytics)
python train_lstm.py
```

This will:
1.  **Load** the Enron dataset from Hugging Face.
2.  **Preprocess** the text (Cleaning, Tokenization, Stopword removal, Lemmatization).
3.  **Train** Naive Bayes, SVM (with calibrated probabilities), and Bidirectional LSTM models.
4.  **Evaluate** each model with accuracy, F1 score, and confusion matrices.
5.  **Save** all model artifacts to the `models/` folder and confusion matrix plots to `plots/`.

### 3Launch the App
Run the Streamlit application to start classifying emails:

```bash
streamlit run app.py
```

### 4 Deploy to Streamlit Cloud
1. Commit your code and ensure the `models/` folder is included with all required artifacts:
   - `models/nb_model.joblib`
   - `models/nb_vectorizer.joblib`
   - `models/svm_model.joblib`
   - `models/svm_vectorizer.joblib`
   - `models/lstm_model.h5`
   - `models/lstm_tokenizer.joblib`
2. Push the repository to GitHub.
3. Go to https://streamlit.io/cloud and connect your GitHub account.
4. Create a new app using this repository, branch `main`, and file path `app.py`.
5. Use the existing `.streamlit/config.toml` settings if available.

> Note: The app must have the model files available at runtime. If the models are not present in the repo, deployment will fail.

---

##  Technologies Used

*   **Python**: Core programming language.
*   **NLP Tools**: NLTK (Stopwords, Stemming).
*   **Machine Learning**: Scikit-Learn (NB, SVM, TF-IDF).
*   **Deep Learning**: TensorFlow & Keras (Bidirectional LSTM).
*   **Web Framework**: Streamlit (Premium UI with Custom CSS).
*   **Data Handling**: Pandas, NumPy.

---


## Support the Project
If you find this project helpful, give it a star on GitHub! 
