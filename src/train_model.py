import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import load_and_preprocess  # your preprocessing function

def predict_with_threshold(model, X, threshold=0.5):
    """
    Predict multi-label outputs using a custom threshold instead of default 0.5
    """
    probas = []
    probas_list = model.predict_proba(X)
    for i in range(len(probas_list)):
        probas.append(probas_list[i][:, 1])  # probability of class 1

    probas = np.array(probas).T  # shape: (n_samples, n_labels)
    return (probas >= threshold).astype(int)

def train(data_path):
    # Load and preprocess data
    X, y, mlb = load_and_preprocess(data_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the model
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_tfidf, y_train)

    # Create the models directory path
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)

    # Save the vectorizer, model, and multi-label binarizer
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(model_dir, "saved_model.pkl"))
    joblib.dump(mlb, os.path.join(model_dir, "mlb.pkl"))

    print(f"✅ Models saved in: {model_dir}")

    # Make predictions with custom threshold
    y_pred = predict_with_threshold(clf, X_test_tfidf, threshold=0.3)

    # Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == '__main__':
    train('../data/go_emotions_dataset.csv')
