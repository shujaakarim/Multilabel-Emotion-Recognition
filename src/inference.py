import os
import joblib
import numpy as np

# Path to models directory (adjust relative to this file)
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

print(f"Loading vectorizer from: {os.path.join(MODEL_DIR, 'vectorizer.pkl')}")
print(f"Loading model from: {os.path.join(MODEL_DIR, 'saved_model.pkl')}")
print(f"Loading mlb from: {os.path.join(MODEL_DIR, 'mlb.pkl')}")

# Load saved objects
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "saved_model.pkl"))
mlb = joblib.load(os.path.join(MODEL_DIR, "mlb.pkl"))

print(f"Types Loaded:\nVectorizer: {type(vectorizer)}\nModel: {type(model)}\nMLB: {type(mlb)}")

def predict_with_threshold(model, X, threshold=0.5):
    """
    Predict multi-label outputs using a custom threshold instead of default 0.5
    """
    probas = []
    probas_list = model.predict_proba(X)
    for i in range(len(probas_list)):
        probas.append(probas_list[i][:, 1])  # probability of class 1

    probas = np.array(probas).T
    return (probas >= threshold).astype(int)

def predict_emotions(texts, threshold=0.3):
    """
    Given a list of texts, predict emotion labels.
    """
    # Transform texts with the vectorizer
    X_tfidf = vectorizer.transform(texts)

    # Predict with threshold
    y_pred = predict_with_threshold(model, X_tfidf, threshold)

    # Convert binary indicators back to labels
    predicted_labels = mlb.inverse_transform(y_pred)

    return predicted_labels

if __name__ == "__main__":
    # Example test texts
    test_texts = [
        "I am so happy and excited today!",
        "This is so sad and depressing.",
        "I'm angry and frustrated about this."
    ]

    predictions = predict_emotions(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}\nPredicted Emotions: {pred}\n")
