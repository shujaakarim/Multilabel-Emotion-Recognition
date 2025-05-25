import joblib

with open(r"E:\Noor_Khan\COMPUTER SCIENCE JOURNEY\Internships\Remot ( DevelopersHub )\2nd round projects\MultiLabel_Emotion_Recognition\models\vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)
print(type(vectorizer))
