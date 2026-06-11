<<<<<<< HEAD
# EMOTION-DETECTION-USING-TEXT
NLP-based Emotion Detection using Machine Learning and Streamlit
=======
# Emotion Detection Using Text

A polished, production-style emotion detection repository using classical NLP and machine learning. This project trains an emotion classifier from text and exposes a Streamlit web app for instant predictions.

## Overview

This repository contains a text emotion classification pipeline that trains a TF-IDF + scikit-learn model on emotion-labeled text data. The project is designed to be portfolio-ready and easy to run from the repository root.

## Features

- ✅ Clean dataset preprocessing pipeline
- ✅ TF-IDF vectorization with classical classifiers
- ✅ Model comparison between Logistic Regression and LinearSVC
- ✅ Streamlit user interface for live emotion detection
- ✅ Support for both real dataset and optional synthetic dataset generation
- ✅ Model saved as a reusable pickle pipeline

## Technologies Used

- Python 3.11
- scikit-learn
- pandas
- numpy
- Streamlit
- joblib

## Machine Learning Workflow

1. Load the dataset from `data/emotion_dataset.csv`
2. Clean and normalize text
3. Remove duplicates and invalid data
4. Train a TF-IDF + classifier pipeline
5. Compare model performance and select the best model
6. Save the best model to `model/emotion_pipeline.pkl`

## Dataset Information

- Primary dataset: `data/emotion_dataset.csv`
- Available emotions in the supplied dataset: 14 classes
- Optional synthetic dataset: `data/generated_dataset.csv`
- Use `src/dataset_generator.py` to generate extended emotion data

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python src/train_model.py
```

### Run the Streamlit app

```bash
streamlit run src/app.py
```

### Generate a synthetic dataset (optional)

```bash
python src/dataset_generator.py
```

## Project Structure

```
EMOTION-DETECTION-USING-TEXT/
├── assets/
│   ├── ui_screenshot.png
│   └── prediction_screenshot.png
├── data/
│   ├── emotion_dataset.csv
│   └── generated_dataset.csv
├── model/
│   └── emotion_pipeline.pkl
├── src/
│   ├── app.py
│   ├── dataset_generator.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Screenshots

- `assets/ui_screenshot.png` - Streamlit UI placeholder
- `assets/prediction_screenshot.png` - Sample prediction placeholder

## Future Enhancements

- Expand the dataset with more real-world examples
- Move to transformer embeddings for improved accuracy
- Add API endpoints for production deployment
- Improve text preprocessing with lemmatization and entity handling
- Add explainable AI support for model transparency

## Author

**HEMANTHL-coder**

- GitHub: https://github.com/HEMANTHL-coder/EMOTION-DETECTION-USING-TEXT
- Email: [Add your email here]
>>>>>>> d79e981 (Prepare emotion detection project for GitHub portfolio)
