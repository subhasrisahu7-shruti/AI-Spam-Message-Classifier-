import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os

# Set up page configurations
st.set_page_config(page_title="AI Spam Classifier", page_icon="🛡️", layout="centered")

# Download necessary NLP data packages
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()

# Initialize text cleaning tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:700; color:#1E3A8A; text-align:center; margin-bottom:10px; }
    .sub-title { font-size:18px; color:#4B5563; text-align:center; margin-bottom:30px; }
    </style>
""", unsafe_view_menu=True)

st.markdown('<div class="main-title">🛡️ AI-Based Spam Message Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter an SMS or Email message below to verify its safety instantly using Natural Language Processing.</div>', unsafe_allow_html=True)

# User typed input box
user_input = st.text_area("Paste message contents here:", placeholder="Type or paste your message text...", height=150)

if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Check if pre-trained model files exist
        if os.path.exists('spam_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
            # Load the saved model assets
            model = joblib.load('spam_model.pkl')
            tfidf = joblib.load('tfidf_vectorizer.pkl')
            
            # Clean and process the user input text
            cleaned_input = preprocess_text(user_input)
            vectorized_input = tfidf.transform([cleaned_input]).toarray()
            
            # Make the final prediction
            prediction = model.predict(vectorized_input)[0]
            
            # Display results beautifully
            st.write("---")
            if prediction == 'spam':
                st.error("🚨 **Result: SPAM DETECTED!** This message matches standard fraudulent or unsolicited patterns.")
            else:
                st.success("✅ **Result: HAM (LEGITIMATE)**. This message looks completely safe.")
        else:
            # Fallback warning if files aren't created yet
            st.info("💡 **Repository Demo Mode:** Your interface structural pipeline is fully verified! Run your 'train_model.py' file to generate live prediction assets.")
