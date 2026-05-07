import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load model
with open("models/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Text cleaning function
def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Prediction function
def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)

    if prediction[0] == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"

# Streamlit UI
st.title("📰 Fake News Detection System")

st.write("Enter a news article or headline below:")

news_input = st.text_area("News Text")

if st.button("Predict"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:

        result = predict_news(news_input)

        if result == "REAL NEWS":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")