import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Download necessary NLP resources
nltk.download('stopwords')
nltk.download('punkt')

# 2. Sample balanced text dataset for training initialization
# (In a local setup, you would load your downloaded Kaggle spam.csv)
data = {
    'text': [
        'Free entry in 2 a weekly comp to win FA Cup final tickets 21st May 2005.',
        'Nah I dont think he goes to usf, he lives around here though',
        'URGENT! You have won a 1 week FREE membership in our £100,000 prize jackpot!',
        'I am typing an assignment right now, talk to you later.',
        'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!',
        'Hey what are you doing tonight? Want to hang out?',
        'Congratulations! You have been specially selected for a free holiday trip.',
        'Please call me back as soon as you get this message.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# 3. Text Preprocessing Function (The NLP Core)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers, keep only alphabets
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize words
    words = nltk.word_tokenize(text)
    # Remove stopwords and apply stemming (e.g., 'running' becomes 'run')
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Apply preprocessing to our dataset
df['cleaned_text'] = df['text'].apply(preprocess_text)

# 4. Feature Extraction (TF-IDF Vectorization)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# 5. Split Dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 6. Train the AI Model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate Model Performance
y_pred = model.predict(X_test)
print(f"Initial Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%")

# 8. Save the trained model assets for our future Streamlit UI app
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully as deployment files!")
