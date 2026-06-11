"""Helpers for loading and preparing images."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMAGE_SIZE, RANDOM_SEED


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Resize one PIL image and scale pixels to the 0-1 range."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def get_data_directory() -> Path:
    """Return the default location where a dataset can be stored later."""
    return Path(DATA_DIR)


def describe_expected_structure() -> dict[str, str]:
    """Show the class folders expected for a simple image dataset."""
    return {
        "rock": "images of the rock hand sign",
        "paper": "images of the paper hand sign",
        "scissors": "images of the scissors hand sign",
    }


def validate_dataset_directory(data_dir: str | Path = DATA_DIR) -> Path:
    """Check that the dataset folder contains one subfolder per class."""
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. "
            "Create data/rock, data/paper, and data/scissors first."
        )

    missing_classes = [
        class_name for class_name in CLASS_NAMES if not (dataset_path / class_name).is_dir()
    ]
    if missing_classes:
        raise FileNotFoundError(
            "Missing class folders: "
            + ", ".join(missing_classes)
            + ". Expected data/rock, data/paper, and data/scissors."
        )

    return dataset_path


def load_image_array(image_path: str | Path) -> np.ndarray:
    """Load one image, resize it, and scale pixels to the 0-1 range."""
    image = Image.open(image_path)
    return preprocess_pil_image(image)


def load_training_datasets(
    data_dir: str | Path = DATA_DIR,
    validation_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
):
    """Load train and validation datasets from class-named folders."""
    dataset_path = validate_dataset_directory(data_dir)

    from tensorflow import keras

    train_dataset = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=RANDOM_SEED,
    )
    validation_dataset = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=RANDOM_SEED,
    )

    normalization = keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda images, labels: (normalization(images), labels))
    validation_dataset = validation_dataset.map(
        lambda images, labels: (normalization(images), labels)
    )

    return train_dataset, validation_dataset
