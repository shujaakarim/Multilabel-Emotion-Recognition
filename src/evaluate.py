import joblib
from sklearn.metrics import classification_report
from preprocessing import load_and_preprocess


def evaluate(data_path, threshold=0.3):
    X, y, mlb = load_and_preprocess(data_path)

    vectorizer = joblib.load('../models/vectorizer.pkl')
    clf = joblib.load('../models/saved_model.pkl')

    X_tfidf = vectorizer.transform(X)

    # If classifier supports predict_proba, use thresholding for multilabel
    if hasattr(clf, "predict_proba"):
        y_probs = clf.predict_proba(X_tfidf)

        # For multilabel, predict_proba might return a list of arrays per class
        # Stack them horizontally if needed
        if isinstance(y_probs, list):
            import numpy as np
            y_probs = np.hstack([prob[:, 1].reshape(-1, 1) for prob in y_probs])

        y_pred = (y_probs >= threshold).astype(int)
    else:
        # fallback to predict if predict_proba not available
        y_pred = clf.predict(X_tfidf)

    print(classification_report(y, y_pred, target_names=mlb.classes_, zero_division=0))


if __name__ == '__main__':
    evaluate('../data/go_emotions_dataset.csv')
