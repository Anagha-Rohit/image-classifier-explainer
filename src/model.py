"""Starter model utilities for the future classifier."""

from src.config import CLASS_NAMES, IMAGE_SIZE


def build_baseline_model():
    """Create a small CNN skeleton for a future training step."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
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
    return model


def compile_model(model):
    """Compile the model with beginner-friendly defaults."""
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
