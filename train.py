"""Train a simple rock-paper-scissors image classifier."""

from argparse import ArgumentParser
from pathlib import Path

from src.config import BATCH_SIZE, DATA_DIR, MODEL_PATH
from src.data_utils import load_training_datasets
from src.model import build_baseline_model, compile_model


def parse_args():
    parser = ArgumentParser(description="Train a simple rock-paper-scissors CNN.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Folder containing class folders.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Where to save the model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch.")
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset, validation_dataset = load_training_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = compile_model(build_baseline_model())
    model.summary()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
