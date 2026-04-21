import os
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D

# --- Configuration ---
MODELS_DIR = 'models'
VOCAB_SIZE = 10000
MAX_LEN = 150
EMBED_DIM = 64

# --- Setup ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    # Lemmatize and remove stop words (skip single characters)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("--- Training Bidirectional LSTM ---")
    print("Loading dataset...")
    raw_dataset = load_dataset("SetFit/enron_spam")
    df = pd.DataFrame(raw_dataset['train'])
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Starting LSTM training (3 epochs for speed)...")
    model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=1)
    
    loss, acc = model.evaluate(X_test_pad, y_test)
    print(f"LSTM Accuracy: {acc:.4f}")
    
    # Save weights as a NumPy list for maximum cross-version compatibility
    weights = model.get_weights()
    joblib.dump(weights, os.path.join(MODELS_DIR, 'lstm_weights.joblib'))
    joblib.dump(tokenizer, os.path.join(MODELS_DIR, 'lstm_tokenizer.joblib'))
    print(f"LSTM raw weights and tokenizer saved in '{MODELS_DIR}' directory.")
