import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

# Streamlit UI
# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")

st.markdown("""
### Welcome!
This application predicts the sentiment of a movie review using
**Machine Learning (TF-IDF + Logistic Regression)**.

Possible predictions:
- 😊 Positive
- 😞 Negative
- 😐 Neutral (when the model is uncertain)
""")

review = st.text_area("Enter Review")

if st.button("Predict"):

    cleaned_review = preprocess_text(review)

    review_vector = tfidf.transform([cleaned_review])

    prediction = model.predict(review_vector)

    probabilities = model.predict_proba(review_vector)

    confidence = max(probabilities[0])

    if confidence < 0.60:
        st.warning("😐 Neutral")

    elif prediction[0] == "positive":
        st.success("😊 Positive")

    else:
        st.error("😞 Negative")

    st.info(f"Confidence Score: {confidence*100:.2f}%")
