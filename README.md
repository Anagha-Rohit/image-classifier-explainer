# Image Classifier with Explainability

A computer vision project that classifies hand gestures as **Rock**, **Paper**, or **Scissors** using a Convolutional Neural Network (CNN).

The project demonstrates the complete machine learning workflow:

* dataset preparation
* model training
* data augmentation
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

At the moment, the Streamlit app uses the same basic preprocessing path as training and `src.predict.py`: convert to RGB, resize to `128 x 128`, and normalize pixel values to the `0-1` range. If many real-world images are still predicted as `rock`, that should be treated as a real model weakness rather than hidden, and the class probabilities should be inspected during debugging.

## If the app cannot find the model

If `models/rps_classifier.keras` is missing, the app will show an error message instead of crashing.

---

# Real-World Testing

Validation accuracy and real-world accuracy are not always the same.

The validation split in this project comes from the same original dataset used for training, so many images share similar lighting, framing, backgrounds, and gesture styles. Because of that, validation results can look very strong even when the model is less reliable on new photos taken in everyday conditions.

To track real-world performance, use the folders below:

* `real_world_tests/rock`
* `real_world_tests/paper`
* `real_world_tests/scissors`

Add external photos to the correct folder, then run:

```bash
python real_world_evaluation.py
```

This workflow will:

* evaluate every image in `real_world_tests`
* save incorrect predictions to `docs/screenshots/real_world_failures/`
* generate `docs/real_world_results.md`

One especially useful failure to watch for is a real-world scissors image predicted as rock. That kind of mistake is a reminder that a model can look excellent on a clean validation split while still struggling with unfamiliar angles, cluttered backgrounds, or different lighting in the real world.

---

# Results

## Dataset

| Metric          | Value |
| --------------- | ----: |
| Total Images    | 2,520 |
| Training Images |       |

## Before and After Augmentation

| Model | Validation Accuracy | Real-World Accuracy | Real-World F1 Score | Notes |
|---|---:|---:|---:|---|
| Baseline CNN | 100.00% | 21.43% | 22.49% | Performs very well on the clean validation split but generalizes poorly to real-world images. |
| Augmented CNN | TBD | 35.71% | 30.00% | Real-world results improved, but the augmented validation accuracy still needs to be recorded. |

The augmented model has been trained and evaluated on the real-world test images, but this README does not yet have a documented augmented validation accuracy. That value should be added after running `evaluate.py` on `models/rps_classifier_augmented.keras`.

The baseline model achieved perfect validation accuracy on the original dataset. Real-world testing showed much weaker performance: 21.43% accuracy on 14 real-world images, with 11 mistakes. This suggests a generalization problem rather than a simple app bug. Data augmentation was added to improve robustness to lighting, cropping, camera angle, and background differences.

## Real-World Test Set

The real-world test set contains 14 images collected outside the original training dataset:

| Class | Images |
|---|---:|
| Rock | 5 |
| Paper | 5 |
| Scissors | 4 |

One important baseline failure was a real-world scissors image predicted as `rock` with 98.89% confidence. This failure is intentionally kept visible because high-confidence mistakes are useful evidence when improving a model.

## Data Augmentation

Data augmentation means showing the model slightly changed versions of the training images while it learns. For this project, the augmented training pipeline can randomly rotate, zoom, shift, and adjust brightness or contrast. These changes do not create perfect real-world coverage, but they help the model practice with images that are less centered and less clean than the original dataset.

Train the augmented model:

```bash
python train.py
```

By default, this saves:

```text
models/rps_classifier_augmented.keras
```

Evaluate the augmented model on the validation split:

```bash
python evaluate.py --model-path models/rps_classifier_augmented.keras --output-dir docs/screenshots/augmented
```

Evaluate the augmented model on real-world images:

```bash
python real_world_evaluation.py --model-path models/rps_classifier_augmented.keras --report-path docs/real_world_results_augmented.md
```

Reproduce the baseline comparison:

```bash
python real_world_evaluation.py --model-path models/rps_classifier.keras --report-path docs/real_world_results_baseline.md
```
