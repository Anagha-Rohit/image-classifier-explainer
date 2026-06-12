from src.config import AUGMENTED_MODEL_PATH
from train import parse_args


def test_train_defaults_to_augmented_model_path(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py"])

    args = parse_args()

    assert args.model_path == AUGMENTED_MODEL_PATH
    assert args.no_augmentation is False


def test_train_can_disable_augmentation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--no-augmentation"])

    args = parse_args()

    assert args.no_augmentation is True
