import os
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

if __name__ == "__main__":
    print("--- Training Naive Bayes Model ---")
    print("Loading dataset...")
    raw_dataset = load_dataset("SetFit/enron_spam")
    df = pd.DataFrame(raw_dataset['train'])
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    preds = nb_model.predict(X_test_tfidf)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(tfidf, 'models/nb_vectorizer.joblib')
    joblib.dump(nb_model, 'models/nb_model.joblib')
    print("\nModel saved as models/nb_model.joblib")
