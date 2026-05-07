import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading processed dataset...")

# Load dataset
df = pd.read_csv("data/processed_news.csv")

print("Dataset loaded successfully")

# Remove missing values
df.dropna(inplace=True)

print("Missing values removed")

# Features and labels
X = df["text"]
y = df["label"]

print("Applying TF-IDF vectorization...")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("Vectorization completed")

# Train-test split
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset split completed")

# Model training
print("Training Logistic Regression model...")

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model training completed")

# Predictions
print("Making predictions...")

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Save model
with open("models/fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully")

print("\nStep 4 completed successfully")