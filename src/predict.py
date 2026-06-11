"""Prediction helpers for the Streamlit app."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, MODEL_PATH
from src.data_utils import load_image_array, preprocess_pil_image


def get_top_prediction(scores: Sequence[float]) -> tuple[str, float]:
    """Convert model scores into a readable class label and confidence."""
    if len(scores) != len(CLASS_NAMES):
        raise ValueError("Score count must match the number of class names.")

    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return CLASS_NAMES[best_index], float(scores[best_index])


def load_trained_model(model_path: str | Path = MODEL_PATH):
    """Load a saved Keras model from disk."""
    from tensorflow import keras

    return keras.models.load_model(model_path)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize one PIL image and scale pixels to the 0-1 range."""
    return preprocess_pil_image(image)


def predict_probabilities(image_array: np.ndarray, model) -> dict[str, float]:
    """Return one probability per class for a prepared image array."""
    batch = np.expand_dims(image_array, axis=0)
    scores = model.predict(batch, verbose=0)[0]

    if len(scores) != len(CLASS_NAMES):
        raise ValueError("Score count must match the number of class names.")

    return {
        class_name: float(score)
        for class_name, score in zip(CLASS_NAMES, scores, strict=True)
    }


def classify_image(
    image_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    model=None,
) -> tuple[str, float]:
    """Classify one image using a saved model or an already-loaded model."""
    if model is None:
        model = load_trained_model(model_path)

    image_array = load_image_array(image_path)
    probabilities = predict_probabilities(image_array, model)
    return get_top_prediction(list(probabilities.values()))


def parse_args():
    parser = ArgumentParser(description="Classify one rock-paper-scissors image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to the saved model.")
    return parser.parse_args()


def main():
    args = parse_args()
    label, confidence = classify_image(args.image_path, model_path=args.model_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
