import streamlit as st
import os
import joblib
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# --- Setup & Preprocessing ---
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- LSTM Architecture (Must match training) ---
def create_lstm_model():
    VOCAB_SIZE = 10000
    MAX_LEN = 150
    EMBED_DIM = 64
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
    
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

# --- App Styling ---
st.set_page_config(page_title="MailGuard | Email Spam Shield", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #1e2130;
        color: #ffffff;
        border: 1px solid #4a4e69;
        border-radius: 10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        transition: 0.3s;
    }
    .spam {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    .ham {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.4);
    }
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        background: -webkit-linear-gradient(#f8f9fa, #212529);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle-text {
        color: #8d99ae;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<p class="title-text">GuardianAI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Advanced Intelligence Email Spam Classification System</p>', unsafe_allow_html=True)

# --- Load Models (Cached) ---
@st.cache_resource
def load_assets():
    try:
        nb_model = joblib.load('models/nb_model.joblib')
        nb_vectorizer = joblib.load('models/nb_vectorizer.joblib')
        
        svm_model = joblib.load('models/svm_model.joblib')
        svm_vectorizer = joblib.load('models/svm_vectorizer.joblib')
        
        # Load LSTM via raw NumPy weights for absolute cross-version stability
        lstm_model = create_lstm_model()
        lstm_weights = joblib.load('models/lstm_weights.joblib')
        lstm_model.set_weights(lstm_weights)
        lstm_tokenizer = joblib.load('models/lstm_tokenizer.joblib')
        
        return {
            "NB": (nb_model, nb_vectorizer),
            "SVM": (svm_model, svm_vectorizer),
            "LSTM": (lstm_model, lstm_tokenizer)
        }
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have run the training scripts (train_nb.py, train_svm.py, train_lstm.py) first.")
        return None

assets = load_assets()

# --- Main Layout ---
col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.markdown("### Model Selection")
    model_choice = st.selectbox("Choose Detection Engine:", ["SVM (High Precision)", "Naive Bayes (Fast)", "LSTM (Deep Analytics)"], index=0)
    
    st.markdown("### Insight")
    if "SVM" in model_choice:
        st.info("Support Vector Machines are robust for text classification and excellent for high-dimensional feature spaces. We use a Calibrated LinearSVC.")
    elif "Naive" in model_choice:
        st.info("Naive Bayes is a classic probabilistic model known for its speed and effectiveness in word-based spam filters.")
    else:
        st.info("Bidirectional LSTM (Long Short-Term Memory) units capture sequential context, understanding the 'flow' and long-range dependencies of the email.")

with col1:
    email_input = st.text_area(" Paste Email Content Here:", placeholder="Enter the email text you want to analyze...", height=300)
    
    if st.button(" Analyze Email"):
        if not email_input.strip():
            st.warning("Please enter some text to analyze.")
        elif assets is None:
            st.error("Engine failure: Models not found.")
        else:
            with st.spinner("Analyzing linguistic patterns..."):
                start_time = time.time()
                clean_msg = preprocess(email_input)
                
                prediction = None
                confidence = None
                
                if "Naive" in model_choice:
                    model, vec = assets["NB"]
                    vec_msg = vec.transform([clean_msg])
                    prediction = model.predict(vec_msg)[0]
                    # Get probability for confidence
                    confidence = model.predict_proba(vec_msg)[0][prediction]
                    
                elif "SVM" in model_choice:
                    model, vec = assets["SVM"]
                    vec_msg = vec.transform([clean_msg])
                    prediction = model.predict(vec_msg)[0]
                    # Check for probability support (CalibratedClassifierCV)
                    if hasattr(model, "predict_proba"):
                        confidence = model.predict_proba(vec_msg)[0][prediction]
                    else:
                        confidence = 1.0 # Fallback
                    
                else: # LSTM
                    model, tok = assets["LSTM"]
                    seq = tok.texts_to_sequences([clean_msg])
                    padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
                    score = model.predict(padded)[0][0]
                    prediction = 1 if score > 0.5 else 0
                    confidence = score if prediction == 1 else (1 - score)

                processing_time = time.time() - start_time
                
                # --- Result Display ---
                if prediction == 1:
                    st.markdown(f'''
                    <div class="result-card spam">
                        <h2 style="margin:0;"> DETECTION: SPAM </h2>
                        <p style="font-size: 1.1rem; margin-top:10px;">Confidence: {confidence:.2%}</p>
                        <p style="font-size: 0.8rem; opacity: 0.8;">Analyzed in {processing_time:.3f}s</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="result-card ham">
                        <h2 style="margin:0;"> DETECTION: LEGITIMATE (HAM) </h2>
                        <p style="font-size: 1.1rem; margin-top:10px;">Confidence: {confidence:.2%}</p>
                        <p style="font-size: 0.8rem; opacity: 0.8;">Analyzed in {processing_time:.3f}s</p>
                    </div>
                    ''', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align: center; color: #4a4e69;">Secure Email Processing © 2026 GuardianAI Systems</p>', unsafe_allow_html=True)
