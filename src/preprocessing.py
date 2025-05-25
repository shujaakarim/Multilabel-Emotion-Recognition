import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    X = df['text'].tolist()

    # List of emotion columns (all except id, text, example_very_unclear)
    emotion_cols = df.columns.difference(['id', 'text', 'example_very_unclear']).tolist()

    # y will be a numpy array with shape (num_samples, num_emotions)
    y = df[emotion_cols].values

    # We create a dummy MultiLabelBinarizer just to hold class names for evaluation
    mlb = MultiLabelBinarizer(classes=emotion_cols)
    mlb.fit([emotion_cols])  # fit on all classes

    return X, y, mlb
