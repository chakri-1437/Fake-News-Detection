import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("Loading processed dataset...")

# Load processed dataset
df = pd.read_csv("data/processed_news.csv")

print("Dataset loaded successfully")

# Remove missing values
df.dropna(inplace=True)

print("Missing values removed")

# Features and labels
X = df["text"]
y = df["label"]

print("Applying TF-IDF vectorization...")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Convert text to vectors
X = vectorizer.fit_transform(X)

print("Vectorization completed")

# Split dataset
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset split completed")

# Print dataset shapes
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Save vectorizer
with open("models/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nVectorizer saved successfully")

print("\nStep 3 completed successfully")