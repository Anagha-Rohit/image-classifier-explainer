"""Evaluate the model on real-world images outside the original dataset."""

from argparse import ArgumentParser
from pathlib import Path
import shutil

from evaluate import (
    VALID_IMAGE_EXTENSIONS,
    build_confusion_matrix,
    calculate_metrics,
    save_labeled_example,
)
from src.config import CLASS_NAMES, MODEL_PATH
from src.predict import get_top_prediction, load_trained_model, predict_probabilities
from src.data_utils import load_image_array


REAL_WORLD_TEST_DIR = Path("real_world_tests")
FAILURE_OUTPUT_DIR = Path("docs/screenshots/real_world_failures")
REPORT_OUTPUT_PATH = Path("docs/real_world_results.md")


def parse_args():
    parser = ArgumentParser(description="Evaluate the model on real-world test images.")
    parser.add_argument(
        "--data-dir",
        default=REAL_WORLD_TEST_DIR,
        help="Folder containing real-world class subfolders.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--failure-dir",
        default=FAILURE_OUTPUT_DIR,
        help="Folder where incorrect predictions will be saved.",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_OUTPUT_PATH,
        help="Markdown file where the results summary will be written.",
    )
    return parser.parse_args()


def ensure_real_world_directories(data_dir: str | Path) -> Path:
    """Create the expected class folders if they do not exist yet."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        (data_path / class_name).mkdir(parents=True, exist_ok=True)

    return data_path


def collect_real_world_images(data_dir: str | Path) -> list[tuple[Path, str]]:
    """Collect labeled image paths from the real-world testing folders."""
    data_path = ensure_real_world_directories(data_dir)
    image_paths = []

    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                image_paths.append((image_path, class_name))

    return image_paths


def evaluate_real_world_images(model, image_paths: list[tuple[Path, str]]) -> list[dict]:
    """Predict every real-world image and keep detailed results."""
    results = []

    for image_path, true_label in image_paths:
        image_array = load_image_array(image_path)
        probabilities = predict_probabilities(image_array, model)
        predicted_label, confidence = get_top_prediction(list(probabilities.values()))
        results.append(
            {
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )

    return results


def get_class_counts(results: list[dict]) -> dict[str, int]:
    """Count how many real-world test images were provided for each class."""
    counts = {class_name: 0 for class_name in CLASS_NAMES}

    for result in results:
        counts[result["true_label"]] += 1

    return counts


def save_real_world_failures(results: list[dict], failure_dir: str | Path) -> list[Path]:
    """Save every incorrect prediction to the failure screenshots folder."""
    failure_path = Path(failure_dir)
    failure_path.mkdir(parents=True, exist_ok=True)

    for existing_file in failure_path.iterdir():
        if existing_file.is_file() and existing_file.name != ".gitkeep":
            existing_file.unlink()

    saved_paths = []
    failure_index = 1

    for result in results:
        if result["true_label"] == result["predicted_label"]:
            continue

        output_path = failure_path / (
            f"failure_{failure_index}_{result['true_label']}_as_{result['predicted_label']}.png"
        )
        save_labeled_example(result, output_path)
        saved_paths.append(output_path)

        original_copy_path = failure_path / (
            f"failure_{failure_index}_{result['image_path'].stem}{result['image_path'].suffix.lower()}"
        )
        shutil.copy2(result["image_path"], original_copy_path)
        saved_paths.append(original_copy_path)

        failure_index += 1

    return saved_paths


def find_scissors_predicted_as_rock(results: list[dict]) -> dict | None:
    """Return the first scissors image that the model predicted as rock."""
    for result in results:
        if result["true_label"] == "scissors" and result["predicted_label"] == "rock":
            return result

    return None


def format_class_breakdown(results: list[dict]) -> str:
    """Build a markdown table showing class-by-class counts."""
    counts = get_class_counts(results)
    lines = [
        "| Class | Images |",
        "| --- | ---: |",
    ]

    for class_name in CLASS_NAMES:
        lines.append(f"| {class_name} | {counts[class_name]} |")

    return "\n".join(lines)


def build_real_world_report(
    results: list[dict],
    report_path: str | Path,
    failure_dir: str | Path,
) -> Path:
    """Write a markdown report summarizing real-world test performance."""
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        output_path.write_text(
            "\n".join(
                [
                    "# Real-World Test Results",
                    "",
                    "No real-world test images were found yet.",
                    "",
                    "Add `.png`, `.jpg`, or `.jpeg` files to these folders:",
                    "",
                    "* `real_world_tests/rock`",
                    "* `real_world_tests/paper`",
                    "* `real_world_tests/scissors`",
                    "",
                    "After that, run `python real_world_evaluation.py` to generate a real report.",
                    "",
                    "This workflow is important because a model can score very highly on a clean validation split and still struggle on new camera angles, lighting conditions, backgrounds, and hand positions.",
                    "",
                    "When a real-world scissors image is predicted as rock, it should be treated as a valuable failure example rather than hidden. The workflow will save that mistake to `docs/screenshots/real_world_failures/` and include it in this report.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    confusion_matrix = build_confusion_matrix(results)
    metrics = calculate_metrics(confusion_matrix)
    total_images = len(results)
    mistakes = [result for result in results if result["true_label"] != result["predicted_label"]]
    failure_example = find_scissors_predicted_as_rock(results)

    lines = [
        "# Real-World Test Results",
        "",
        "This report measures how the trained model behaves on images outside the original dataset.",
        "",
        "## Summary",
        "",
        f"* Total real-world images: {total_images}",
        f"* Accuracy: {metrics['accuracy']:.2%}",
        f"* Precision: {metrics['precision']:.2%}",
        f"* Recall: {metrics['recall']:.2%}",
        f"* F1 score: {metrics['f1']:.2%}",
        f"* Mistakes saved to `{Path(failure_dir).as_posix()}`: {len(mistakes)}",
        "",
        "## Class Coverage",
        "",
        format_class_breakdown(results),
        "",
        "## Why Real-World Results Can Be Worse",
        "",
        "Validation accuracy can look very high because the validation split comes from the same dataset distribution as training: similar backgrounds, framing, lighting, and hand positions.",
        "Real-world images are usually messier. Different cameras, shadows, cluttered rooms, partial hands, and unusual angles can push the model away from the patterns it learned during training.",
        "That means strong validation accuracy is useful, but it is not the same thing as strong real-world generalization.",
        "",
        "## Example Failure",
        "",
    ]

    if failure_example is not None:
        lines.extend(
            [
                f"A real-world scissors image was predicted as rock: `{failure_example['image_path'].as_posix()}`.",
                f"The model gave `rock` a confidence score of {failure_example['confidence']:.2%}, even though the true label was `scissors`.",
                "This kind of mistake is exactly why real-world testing matters: the model may rely on background cues, framing, or gesture shape details that looked reliable in the original dataset but do not transfer cleanly to new images.",
            ]
        )
    else:
        lines.extend(
            [
                "No `scissors -> rock` error appeared in this run.",
                "If that failure shows up in a future run, this report will call it out explicitly because it is a useful example of a high-validation model still making a real-world mistake.",
            ]
        )

    lines.extend(
        [
            "",
            "## Detailed Predictions",
            "",
            "| Image | True Label | Predicted Label | Confidence |",
            "| --- | --- | --- | ---: |",
        ]
    )

    for result in results:
        lines.append(
            "| "
            f"{result['image_path'].as_posix()} | "
            f"{result['true_label']} | "
            f"{result['predicted_label']} | "
            f"{result['confidence']:.2%} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def print_summary(results: list[dict], report_path: Path, failure_dir: Path):
    """Print a short terminal summary after evaluation."""
    print("Real-world evaluation complete.")
    print(f"Images evaluated: {len(results)}")
    print(f"Report saved to {report_path}")
    print(f"Failure examples saved to {failure_dir}")


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    failure_dir = Path(args.failure_dir)
    report_path = Path(args.report_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with python train.py."
        )

    image_paths = collect_real_world_images(args.data_dir)

    if not image_paths:
        report_output_path = build_real_world_report([], report_path, failure_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)
        print("No real-world images found yet.")
        print(f"Created folders under {Path(args.data_dir)}")
        print(f"Report saved to {report_output_path}")
        return

    model = load_trained_model(model_path)
    results = evaluate_real_world_images(model, image_paths)
    save_real_world_failures(results, failure_dir)
    report_output_path = build_real_world_report(results, report_path, failure_dir)
    print_summary(results, report_output_path, failure_dir)


if __name__ == "__main__":
    main()
