"""
Streamlit application for emotion detection using a pre-trained sklearn pipeline.
"""

import joblib
import re
from pathlib import Path

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "emotion_pipeline.pkl"
CONFIDENCE_THRESHOLD = 0.35

EMOTION_EMOJIS = {
    "admiration": "😍",
    "amusement": "😄",
    "anger": "😡",
    "annoyance": "😤",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "🥰",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😢",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😰",
    "optimism": "🤞",
    "pride": "🏆",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲",
    "boredom": "🥱",
    "stress": "😩",
    "neutral": "😐",
    "hope": "✨",
    "peace": "☮️",
}


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert model scores into probability values."""
    scores = np.array(scores, dtype=float)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()


def clean_text(text: str) -> str:
    """Normalize the input text before prediction."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    """Load the trained pipeline from disk."""
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("Please run: python src/train_model.py")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"❌ Unable to load model: {error}")
        return None


def predict_emotion(model, text: str) -> dict:
    """Predict emotion and return the top predictions with confidence."""
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "best_emotion": None,
            "top3": [],
            "confidence": 0.0,
            "error": "Please enter valid text.",
        }

    try:
        prediction = model.predict([cleaned])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned])[0]
        elif hasattr(model, "decision_function"):
            predicted_scores = model.decision_function([cleaned])
            if isinstance(predicted_scores, np.ndarray) and predicted_scores.ndim == 2:
                probabilities = softmax(predicted_scores[0])
            else:
                probabilities = np.array([1.0])
        else:
            raise ValueError("Model does not support probability or score output.")

        classes = np.array(model.classes_)
        top_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "emotion": classes[i],
                "probability": float(probabilities[i]),
                "emoji": EMOTION_EMOJIS.get(classes[i], "❓"),
            }
            for i in top_indices
        ]

        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        return {
            "best_emotion": prediction,
            "top3": top3,
            "confidence": confidence,
            "error": None,
        }
    except Exception as error:
        return {
            "best_emotion": None,
            "top3": [],
            "confidence": 0.0,
            "error": str(error),
        }


st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #e8e8e8; }
        .title { font-size: 48px; font-weight: 800; text-align: center; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 30px; }
        .card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 30px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin: 20px 0; }
        .emotion-display { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0; }
        .emotion-emoji { font-size: 64px; display: block; margin-bottom: 10px; }
        .emotion-label { font-size: 32px; font-weight: bold; color: white; text-transform: uppercase; letter-spacing: 2px; }
        .confidence-bar { height: 20px; background: rgba(255, 255, 255, 0.2); border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #7c3aed); border-radius: 10px; transition: width 0.5s ease; }
        .top3-item { background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 10px 0; display: flex; align-items: center; justify-content: space-between; }
        .low-confidence { background: rgba(255, 100, 100, 0.2); border: 1px solid rgba(255, 100, 100, 0.5); padding: 20px; border-radius: 10px; text-align: center; color: #ff6b6b; }
        .stTextInput > div > div > input { background: rgba(255, 255, 255, 0.1); border: 2px solid rgba(255, 255, 255, 0.2); color: white; font-size: 18px; padding: 15px; border-radius: 10px; }
        .stTextInput > div > div > input:focus { border-color: #00d4ff; outline: none; }
        .stButton > button { background: linear-gradient(90deg, #00d4ff, #7c3aed); color: white; font-size: 18px; font-weight: bold; padding: 15px 40px; border-radius: 10px; border: none; cursor: pointer; transition: transform 0.2s; }
        .stButton > button:hover { transform: scale(1.05); }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.markdown('<h1 class="title">🎭 Emotion Detection</h1>', unsafe_allow_html=True)
    model = load_model()
    if model is None:
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter your text:")

    user_input = st.text_input(
        "Text to analyze",
        placeholder="Type something... (e.g., 'I am so happy to see you!')",
        label_visibility="collapsed",
    )

    if st.button("🔍 Detect Emotion"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            result = predict_emotion(model, user_input)
            if result["error"]:
                st.error(f"❌ Error: {result['error']}")
            else:
                if result["confidence"] < CONFIDENCE_THRESHOLD:
                    st.markdown(
                        f"""
                        <div class="low-confidence">
                            <h3>⚠️ Low Confidence Prediction</h3>
                            <p>Confidence: {result['confidence']*100:.1f}%</p>
                            <p>Try a longer sentence for better results.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                emoji = EMOTION_EMOJIS.get(result["best_emotion"], "❓")
                confidence_pct = result["confidence"] * 100
                st.markdown(
                    f"""
                    <div class="emotion-display">
                        <span class="emotion-emoji">{emoji}</span>
                        <span class="emotion-label">{result['best_emotion'].upper()}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                    </div>
                    <p style="text-align: center; color: #aaa;">Confidence: {confidence_pct:.1f}%</p>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("### 🏆 Top 3 Predictions:")
                for i, item in enumerate(result["top3"], 1):
                    pct = item["probability"] * 100
                    st.markdown(
                        f"""
                        <div class="top3-item">
                            <span><b>{i}. {item['emoji']} {item['emotion'].upper()}</b></span>
                            <span style="color: #00d4ff;">{pct:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; padding: 20px;">
            <p>🎯 This app uses classical machine learning (TF-IDF + sklearn) to detect emotions in text.</p>
            <p>Supports both the supplied dataset and optional generated datasets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
