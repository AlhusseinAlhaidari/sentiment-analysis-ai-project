import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Download necessary NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

def train_and_evaluate_model(data):
    # For demonstration, we'll use a simple dataset structure
    # In a real scenario, you'd load a proper dataset
    df = pd.DataFrame(data, columns=['text', 'sentiment'])

    X = df['text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\n--- Model Training and Evaluation ---")
    print(classification_report(y_test, y_pred))
    return model, vectorizer

def analyze_sentiment_vader(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment_ml(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

if __name__ == "__main__":
    # Sample data for training (replace with a real dataset for better results)
    sample_data = [
        ("This product is amazing!", "Positive"),
        ("I hate this service.", "Negative"),
        ("It's okay, nothing special.", "Neutral"),
        ("Very happy with the purchase.", "Positive"),
        ("Terrible experience, never again.", "Negative"),
        ("The quality is decent.", "Neutral"),
        ("Absolutely fantastic!", "Positive"),
        ("Completely disappointed.", "Negative"),
        ("It works as expected.", "Neutral"),
        ("Best thing I've bought all year.", "Positive"),
    ]

    # Train and evaluate the ML model
    trained_model, trained_vectorizer = train_and_evaluate_model(sample_data)

    print("\n--- Sentiment Analysis Examples ---")
    texts_to_analyze = [
        "I love this new phone, it's so fast!",
        "The customer support was unhelpful and rude.",
        "The movie was neither good nor bad, just average.",
        "What a wonderful day!",
        "This is the worst decision ever."
    ]

    print("\n--- Using VADER Sentiment Analyzer ---")
    for text in texts_to_analyze:
        sentiment = analyze_sentiment_vader(text)
        print(f"Text: '{text}' -> Sentiment (VADER): {sentiment}")

    print("\n--- Using Trained ML Model ---")
    for text in texts_to_analyze:
        sentiment = analyze_sentiment_ml(text, trained_model, trained_vectorizer)
        print(f"Text: '{text}' -> Sentiment (ML Model): {sentiment}")

