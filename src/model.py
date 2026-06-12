"""Starter model utilities for the future classifier."""

from src.config import CLASS_NAMES, IMAGE_SIZE


def build_data_augmentation():
    """Create simple image changes that help the model handle real-world variation."""
    from tensorflow import keras

    augmentation_layers = [
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.10),
        keras.layers.RandomTranslation(0.10, 0.10),
    ]

    if hasattr(keras.layers, "RandomBrightness"):
        augmentation_layers.append(
            keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0))
        )

    augmentation_layers.append(keras.layers.RandomContrast(0.15))

    return keras.Sequential(
        augmentation_layers,
        name="data_augmentation",
    )


def build_baseline_model(use_augmentation: bool = False):
    """Create a small CNN for rock-paper-scissors classification."""
    from tensorflow import keras

    layers = [keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))]

    if use_augmentation:
        layers.append(build_data_augmentation())

    layers.extend(
        [
            keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )

    model = keras.Sequential(
        layers,
        name="rps_classifier_augmented" if use_augmentation else "rps_classifier",
    )
    return model


def compile_model(model):
    """Compile the model with beginner-friendly defaults."""
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
