"""Evaluate the trained rock-paper-scissors classifier."""

from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from src.config import CLASS_NAMES, DATA_DIR, MODEL_PATH, RANDOM_SEED
from src.data_utils import load_image_array, validate_dataset_directory
from src.predict import get_top_prediction, load_trained_model


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = ArgumentParser(description="Evaluate the saved rock-paper-scissors model.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Folder containing class folders.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to the saved model.")
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        help="Folder where evaluation images will be saved.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of images to use for validation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=6,
        help="Maximum correct and incorrect examples to save.",
    )
    return parser.parse_args()


def collect_image_paths(data_dir: str | Path) -> list[tuple[Path, str]]:
    """Collect image paths with their class labels."""
    dataset_path = validate_dataset_directory(data_dir)
    image_paths = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_path / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                image_paths.append((image_path, class_name))

    if not image_paths:
        raise ValueError("No images found. Add .jpg, .jpeg, or .png files to the data folders.")

    return image_paths


def get_validation_split(
    image_paths: list[tuple[Path, str]],
    validation_split: float,
) -> list[tuple[Path, str]]:
    """Return a deterministic validation split."""
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")

    shuffled_paths = image_paths.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_paths)

    validation_count = max(1, int(len(shuffled_paths) * validation_split))
    return shuffled_paths[-validation_count:]


def predict_validation_images(model, validation_paths: list[tuple[Path, str]]):
    """Run model predictions for every validation image."""
    results = []

    for image_path, true_label in validation_paths:
        image_array = load_image_array(image_path)
        batch = np.expand_dims(image_array, axis=0)
        scores = model.predict(batch, verbose=0)[0]
        predicted_label, confidence = get_top_prediction(scores)
        results.append(
            {
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
            }
        )

    return results


def build_confusion_matrix(results: list[dict]) -> np.ndarray:
    """Count true labels versus predicted labels."""
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    class_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

    for result in results:
        true_index = class_to_index[result["true_label"]]
        predicted_index = class_to_index[result["predicted_label"]]
        matrix[true_index, predicted_index] += 1

    return matrix


def calculate_metrics(confusion_matrix: np.ndarray) -> dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score."""
    total_correct = np.trace(confusion_matrix)
    total_predictions = confusion_matrix.sum()
    accuracy = total_correct / total_predictions if total_predictions else 0.0

    precision_values = []
    recall_values = []
    f1_values = []

    for index in range(len(CLASS_NAMES)):
        true_positive = confusion_matrix[index, index]
        predicted_as_class = confusion_matrix[:, index].sum()
        actual_class = confusion_matrix[index, :].sum()

        precision = true_positive / predicted_as_class if predicted_as_class else 0.0
        recall = true_positive / actual_class if actual_class else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "f1": float(np.mean(f1_values)),
    }


def save_confusion_matrix_image(confusion_matrix: np.ndarray, output_path: Path):
    """Save the confusion matrix as a PNG image."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            ax.text(
                column,
                row,
                str(confusion_matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_examples(
    results: list[dict],
    output_dir: Path,
    max_examples: int,
):
    """Save a few correct and incorrect prediction examples."""
    correct_dir = output_dir / "correct_examples"
    incorrect_dir = output_dir / "incorrect_examples"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    correct = [
        result for result in results if result["true_label"] == result["predicted_label"]
    ][:max_examples]
    incorrect = [
        result for result in results if result["true_label"] != result["predicted_label"]
    ][:max_examples]

    for index, result in enumerate(correct, start=1):
        save_labeled_example(result, correct_dir / f"correct_{index}.png")

    for index, result in enumerate(incorrect, start=1):
        save_labeled_example(result, incorrect_dir / f"incorrect_{index}.png")

    return len(correct), len(incorrect)


def save_labeled_example(result: dict, output_path: Path):
    """Save one image with its true and predicted label written above it."""
    image = Image.open(result["image_path"]).convert("RGB")
    image.thumbnail((300, 300))

    label_height = 52
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"True: {result['true_label']}", fill="black")
    draw.text(
        (8, 28),
        f"Predicted: {result['predicted_label']} ({result['confidence']:.1%})",
        fill="black",
    )
    canvas.save(output_path)


def print_metric_summary(metrics: dict[str, float]):
    print("Evaluation metrics")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1']:.2%}")


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with python train.py."
        )

    image_paths = collect_image_paths(args.data_dir)
    validation_paths = get_validation_split(image_paths, args.validation_split)
    model = load_trained_model(model_path)

    results = predict_validation_images(model, validation_paths)
    confusion_matrix = build_confusion_matrix(results)
    metrics = calculate_metrics(confusion_matrix)

    save_confusion_matrix_image(confusion_matrix, output_dir / "confusion_matrix.png")
    correct_count, incorrect_count = save_prediction_examples(
        results,
        output_dir,
        args.max_examples,
    )

    print_metric_summary(metrics)
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    print(f"Saved {correct_count} correct examples.")
    print(f"Saved {incorrect_count} incorrect examples.")


if __name__ == "__main__":
    main()
