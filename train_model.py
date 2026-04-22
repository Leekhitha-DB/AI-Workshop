import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_dataset(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or ";" not in line:
                continue
            text, label = line.split(";", 1)
            rows.append({"text": text.strip(), "label": label.strip()})
    return pd.DataFrame(rows)


def train_text_classifier(data_path: str, model_path: str, vectorizer_path: str):
    df = load_dataset(data_path)
    if df.empty:
        raise ValueError(f"No valid data found in {data_path}")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Evaluation report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")

    return model, vectorizer


def predict_emotion(text: str, model: MultinomialNB, vectorizer: TfidfVectorizer) -> str:
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]


def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier from train.txt")
    parser.add_argument("--data", default="train.txt", help="Path to the training data file")
    parser.add_argument("--model", default="text_classifier.joblib", help="Path to save the trained model")
    parser.add_argument("--vectorizer", default="tfidf_vectorizer.joblib", help="Path to save the vectorizer")
    args = parser.parse_args()

    train_text_classifier(args.data, args.model, args.vectorizer)


if __name__ == "__main__":
    main()
