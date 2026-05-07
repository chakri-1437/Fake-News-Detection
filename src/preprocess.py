import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

print("Starting preprocessing...")

# Download stopwords
nltk.download('stopwords', quiet=True)

print("Stopwords downloaded")

# Load datasets
print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

print("Datasets loaded successfully")

# Create labels
fake["label"] = 0
real["label"] = 1

print("Labels created")

# Combine datasets
df = pd.concat([fake, real])

print("Datasets combined")

# Shuffle rows
df = df.sample(frac=1, random_state=42)

# Reset index
df.reset_index(drop=True, inplace=True)

print("Dataset shuffled")

# Stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):

    # Convert to string (important safety step)
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back
    return " ".join(words)

print("Cleaning text... This may take some time.")

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

print("Text cleaning completed")

# Keep only required columns
df = df[["text", "label"]]

# Save processed dataset
df.to_csv("data/processed_news.csv", index=False)

print("\nProcessed dataset saved successfully")

# Print sample rows
print("\nSample Data:")
print(df.head())

# Print dataset shape
print("\nDataset shape:", df.shape)

print("\nPreprocessing completed successfully")