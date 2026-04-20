import os
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
MODELS_DIR = 'models'

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
        
    print("--- Training Naive Bayes ---")
    print("Loading dataset...")
    raw_dataset = load_dataset("SetFit/enron_spam")
    df = pd.DataFrame(raw_dataset['train'])
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    nb_preds = nb_model.predict(X_test_tfidf)
    print(f"NB Accuracy: {accuracy_score(y_test, nb_preds):.4f}")
    print(classification_report(y_test, nb_preds))
    
    # Save
    joblib.dump(nb_model, os.path.join(MODELS_DIR, 'nb_model.joblib'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'nb_vectorizer.joblib'))
    print(f"Naive Bayes model and vectorizer saved in '{MODELS_DIR}' directory.")
