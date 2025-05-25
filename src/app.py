import streamlit as st
import os
import joblib
import numpy as np

# ✅ Set page config first (must be the first Streamlit command)
st.set_page_config(page_title="Emotion Recognition", page_icon="💬", layout="centered")

# ✅ Load model components with caching
@st.cache_resource
def load_model_files():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    model = joblib.load(os.path.join(model_dir, "saved_model.pkl"))
    mlb = joblib.load(os.path.join(model_dir, "mlb.pkl"))
    return vectorizer, model, mlb

# ✅ Prediction with threshold
def predict_with_threshold(model, X, threshold=0.3):
    probas_list = model.predict_proba(X)
    probas = []
    for prob in probas_list:
        probas.append(prob[:, 1])
    probas = np.array(probas).T
    return (probas >= threshold).astype(int)

# ✅ Load model
vectorizer, model, mlb = load_model_files()

# ✅ Custom styling
st.markdown("""
    <style>
        .main-box {
            background-color: #f9fafe;
            padding: 2rem;
            border-radius: 1.5rem;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
        }
        .header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #3366cc;
            margin-bottom: 5px;
        }
        .subtext {
            text-align: center;
            color: #777;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: #999;
            margin-top: 40px;
        }
        .stButton>button {
            height: 3em;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Main UI
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header">💬 Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Enter a sentence or paragraph to detect emotions using AI</div>', unsafe_allow_html=True)

# ✅ Text input
input_text = st.text_area("✍️ Your Text:", height=150, placeholder="Write something emotional...")

# ✅ Columns for buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🎯 Predict"):
        if not input_text.strip():
            st.warning("Please enter some text before predicting.")
        else:
            X_input = vectorizer.transform([input_text])
            y_pred = predict_with_threshold(model, X_input, threshold=0.3)
            emotions = mlb.inverse_transform(y_pred)

            if emotions and len(emotions[0]) > 0:
                emoji_map = {
                    "anger": "😠", "joy": "😊", "sadness": "😢",
                    "fear": "😨", "surprise": "😲", "love": "❤️",
                    "neutral": "😐", "disgust": "🤢"
                }
                labeled_emotions = [f"{emoji_map.get(e, '❓')} {e}" for e in emotions[0]]
                st.success("Predicted Emotions:")
                st.markdown("**" + ", ".join(labeled_emotions) + "**")
            else:
                st.info("No strong emotion detected. Try writing a more emotional sentence.")

with col2:
    if st.button("🧹 Clear"):
        # Rerun to reset state
        st.session_state.clear()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Made with ❤️ by Shujaat Karim</div>', unsafe_allow_html=True)
