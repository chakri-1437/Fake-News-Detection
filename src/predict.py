import pickle
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load stopwords
stop_words = set(stopwords.words("english"))

print("Loading model and vectorizer...")

# Load trained model
with open("models/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

print("Model loaded successfully")

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

    # Clean text
    cleaned_text = clean_text(news_text)

    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vectorized_text)

    # Return result
    if prediction[0] == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"

# User input
print("\n========== Fake News Detection ==========\n")

news = input("Enter news text:\n\n")

# Predict result
result = predict_news(news)

print("\nPrediction:", result)