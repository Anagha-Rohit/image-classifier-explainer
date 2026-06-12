# Model Card

## Model Name

Rock-Paper-Scissors Image Classifier

## Intended Use

This model is intended as a beginner portfolio project for image classification and explainability practice.

## Model Type

Small convolutional neural network built with TensorFlow / Keras.

Phase 4 keeps the CNN architecture simple and adds data augmentation before the convolution layers during training.

## Input

- image of a hand gesture
- expected classes: `rock`, `paper`, `scissors`

## Output

- predicted class label
- confidence scores
- planned explainability visualization

## Training Data

Recommended dataset: TensorFlow's Rock Paper Scissors dataset by Laurence Moroney.

Expected classes:

- `rock`
- `paper`
- `scissors`

The project expects images to be stored in class folders under `data/`.

## Training Approach

The original Phase 3 model was trained on the clean dataset and saved as:

- `models/rps_classifier.keras`

The Phase 4 model uses the same dataset and the same `rock`, `paper`, `scissors` class order, but adds simple random image changes during training:

- small rotations
- small zoom changes
- small horizontal and vertical translations
- brightness variation when supported by the installed Keras version
- contrast variation

These changes are called data augmentation. They help the model see examples that are a little closer to real-world photos, where the hand may not be perfectly centered or lit the same way as the training images.

The augmented model is saved separately as:

- `models/rps_classifier_augmented.keras`

Preprocessing remains consistent across training, prediction, Streamlit, and evaluation: images are converted to RGB, resized to `128 x 128`, and scaled to the `0-1` pixel range.

## Evaluation

Evaluation is handled by `evaluate.py`.

The script loads `models/rps_classifier.keras`, evaluates a deterministic validation split, and saves:

- confusion matrix: `docs/screenshots/confusion_matrix.png`
- correct prediction examples: `docs/screenshots/correct_examples/`
- incorrect prediction examples: `docs/screenshots/incorrect_examples/`

Latest metrics:

```text
Accuracy:  100.00%
Precision: 100.00%
Recall:    100.00%
F1 score:  100.00%
```

Saved evaluation artifacts:

- confusion matrix: `docs/screenshots/confusion_matrix.png`
- correct examples saved: 6
- incorrect examples saved: 0

For Phase 4 comparison, evaluate the augmented model with:

```bash
python evaluate.py --model-path models/rps_classifier_augmented.keras --output-dir docs/screenshots/augmented
```

Real-world testing can also compare both models:

```bash
python real_world_evaluation.py --model-path models/rps_classifier.keras
python real_world_evaluation.py --model-path models/rps_classifier_augmented.keras
```

Metric meanings:

- Accuracy: how often the model is correct overall.
- Precision: when the model predicts a class, how often that prediction is correct.
- Recall: how many real examples from a class the model successfully finds.
- F1 score: one score that balances precision and recall.

## Current Limitations

- The validation split comes from the same dataset source as training, so it may be easier than real-world photos.
- The model may not work well with unusual lighting, hand angles, backgrounds, or camera quality.
- The model only predicts `rock`, `paper`, or `scissors`; it does not know when an image is unrelated.
- Metrics can look strong on a simple dataset while still hiding real-world mistakes.
- Data augmentation can improve robustness, but it does not guarantee perfect real-world performance.
- Explainability has not been added yet.

## Ethical And Practical Notes

- this project is educational, not safety-critical
- predictions may fail on unusual lighting, angles, or backgrounds
- future explainability outputs may help interpretation, but they are not perfect proof of reasoning
