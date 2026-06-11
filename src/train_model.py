"""
Emotion Detection Model Training Script
========================================
This script trains a text-based emotion classification pipeline using TF-IDF
feature extraction and a classical scikit-learn classifier.

Usage:
    python src/train_model.py

Outputs:
    model/emotion_pipeline.pkl
"""

import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "emotion_dataset.csv"
FALLBACK_PATH = BASE_DIR / "data" / "generated_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_PATH = MODEL_DIR / "emotion_pipeline.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUPPORTED_TEXT_COLUMNS = ["text", "sentence", "message", "review"]
SUPPORTED_LABEL_COLUMNS = ["emotion", "label", "sentiment", "target"]


def clean_text(text: str) -> str:
    """Clean and normalize user text input."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find text and label columns in the dataset."""
    text_col = next((col for col in SUPPORTED_TEXT_COLUMNS if col in df.columns), None)
    label_col = next((col for col in SUPPORTED_LABEL_COLUMNS if col in df.columns), None)

    if not text_col or not label_col:
        raise ValueError(
            "Dataset must contain one text column and one label column. "
            f"Found columns: {list(df.columns)}"
        )

    return text_col, label_col


def load_data() -> pd.DataFrame:
    """Load the dataset from the configured data path."""
    if DATA_PATH.exists():
        path = DATA_PATH
    elif FALLBACK_PATH.exists():
        path = FALLBACK_PATH
        print(f"⚠️  Primary dataset not found. Falling back to {FALLBACK_PATH}")
    else:
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH} or {FALLBACK_PATH}."
        )

    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} samples from {path}")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Clean the dataset and prepare text/label columns for training."""
    text_col, label_col = detect_columns(df)

    print("\n🔧 Preprocessing data...")
    initial_count = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    print(f"  ✓ Dropped {initial_count - len(df)} rows with missing values")

    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    print(f"  ✓ Removed {before_dedup - len(df)} duplicate texts")

    df["cleaned_text"] = df[text_col].apply(clean_text)
    after_clean = len(df)
    df = df[df["cleaned_text"].str.len() > 0].copy()
    print(f"  ✓ Removed {after_clean - len(df)} empty cleaned texts")

    labels = sorted(df[label_col].unique())
    print(f"  ✓ Found {len(labels)} emotion classes")
    print(f"  ✓ Emotion labels: {labels}")
    print(f"  ✓ Final dataset size: {len(df)}")

    return df, text_col, label_col


def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """Construct the scikit-learn pipeline for training."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,
    )

    if model_type == "logistic":
        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=2,
            random_state=RANDOM_STATE,
        )
    elif model_type == "svc":
        classifier = LinearSVC(
            class_weight="balanced",
            C=2,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("tfidf", tfidf), ("model", classifier)])


def main() -> None:
    print("=" * 60)
    print("🎭 EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    df, text_col, label_col = preprocess_data(df)

    X = df["cleaned_text"]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\n🚀 Training Logistic Regression...")
    start_time = time.time()
    pipeline_lr = build_pipeline("logistic")
    pipeline_lr.fit(X_train, y_train)
    print(f"  ✓ Completed in {time.time() - start_time:.2f}s")

    print("\n📈 Evaluating Logistic Regression...")
    y_pred_lr = pipeline_lr.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"  Accuracy: {accuracy_lr:.4f}")

    print("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(pipeline_lr, X, y, cv=5, scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

    print("\n🚀 Training LinearSVC pipeline for comparison...")
    start_time = time.time()
    pipeline_svc = build_pipeline("svc")
    pipeline_svc.fit(X_train, y_train)
    print(f"  ✓ Completed in {time.time() - start_time:.2f}s")

    y_pred_svc = pipeline_svc.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    print(f"  Accuracy: {accuracy_svc:.4f}")

    if accuracy_svc > accuracy_lr:
        best_pipeline = pipeline_svc
        best_name = "LinearSVC"
        best_accuracy = accuracy_svc
    else:
        best_pipeline = pipeline_lr
        best_name = "Logistic Regression"
        best_accuracy = accuracy_lr

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, OUTPUT_PATH)
    print(f"\n💾 Saved best model ({best_name}) to {OUTPUT_PATH}")
    print(f"🎯 Selected model accuracy: {best_accuracy:.4f}")

    print("\n📋 Classification report for the selected model:")
    print(classification_report(y_test, best_pipeline.predict(X_test), zero_division=0))
    print("\n📊 Confusion matrix:")
    print(confusion_matrix(y_test, best_pipeline.predict(X_test)))


if __name__ == "__main__":
    main()
