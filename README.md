# Image Classifier with Explainability

A computer vision project that classifies hand gestures as **Rock**, **Paper**, or **Scissors** using a Convolutional Neural Network (CNN).

The project demonstrates the complete machine learning workflow:

* dataset preparation
* model training
* evaluation
* prediction
* testing
* deployment with Streamlit

---

# Demo

## Example Prediction

```bash
python -m src.predict data/rock/rock01-000.png
```

Output:

```text
Prediction: rock
Confidence: 99.60%
```

## Confusion Matrix

![Confusion Matrix](docs/screenshots/confusion_matrix.png)

---

# Streamlit App

The project includes a beginner-friendly Streamlit app for image prediction.

## Run the app

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model file exists:

```text
models/rps_classifier.keras
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. Upload a `.png`, `.jpg`, or `.jpeg` image of a hand gesture.

The app will:

* show the uploaded image
* predict `rock`, `paper`, or `scissors`
* display the confidence score
* display the probability for all three classes

## If the app cannot find the model

If `models/rps_classifier.keras` is missing, the app will show an error message instead of crashing.

---

# Results

## Dataset

| Metric          | Value |
| --------------- | ----: |
| Total Images    | 2,520 |
| Training Images |       |
