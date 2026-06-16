# Emotion Detection Using Text

A professional NLP project for emotion classification using a classical machine learning pipeline and Streamlit deployment.

## Overview

This repository demonstrates a complete text-based emotion detection workflow. It includes dataset preprocessing, model training, and a Streamlit UI for live inference.

## Features

- Clean, reproducible text preprocessing
- TF-IDF feature extraction
- Model comparison between `LogisticRegression` and `LinearSVC`
- Streamlit web app for predictions
- Saved model pipeline in `model/emotion_pipeline.pkl`
- Optional synthetic dataset generation

## Technologies Used

- Python 3.11
- scikit-learn
- pandas
- numpy
- Streamlit
- joblib

## Machine Learning Workflow

1. Load the dataset from `data/emotion_dataset.csv`
2. Detect text and emotion columns automatically
3. Clean and normalize raw text
4. Remove duplicate and invalid samples
5. Train TF-IDF + classifier pipeline
6. Evaluate using train/test split and cross-validation
7. Save the best model as a pickled pipeline

## Dataset Information

- Primary dataset: `data/emotion_dataset.csv`
- Contains emotion-labeled text samples
- Optional generated dataset: `data/generated_dataset.csv`
- Synthetic dataset generator available in `src/dataset_generator.py`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Deploy on Render / Streamlit Cloud

This project is ready for Streamlit-based hosting.

1. Create a Python 3.11 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run src/app.py
   ```

For Render, use the included `Procfile` and `runtime.txt`.
For Streamlit Cloud, point the app to `src/app.py`.

## Deployment Files

- `Procfile` — starts the Streamlit server on the hosting platform
- `runtime.txt` — pins the Python runtime
- `render.yaml` — optional Render deployment configuration

### Train the model

```bash
python src/train_model.py
```

### Run the Streamlit app

```bash
streamlit run src/app.py
```

### Generate synthetic data (optional)

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

- `assets/ui_screenshot.png` — Streamlit app interface
- `assets/prediction_screenshot.png` — Example prediction output

## Future Enhancements

- Add transformer-based embeddings for improved accuracy
- Add more real-world emotion data and label coverage
- Improve preprocessing with lemmatization and stopword tuning
- Add API deployment support
- Add model explainability and confidence visualization

## Author

**HEMANTHL-coder**

- GitHub: https://github.com/HEMANTHL-coder/EMOTION-DETECTION-USING-TEXT
- Email: hemanthlpujar95@gmail.com
- LIVE DEMO LINK : https://emotion-detection-using-text-oqefv8hnygd72qqurxfzcc.streamlit.app/
