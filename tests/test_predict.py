from PIL import Image
import numpy as np
import pytest

from src.config import IMAGE_SIZE
from src.predict import classify_image, get_top_prediction, predict_probabilities, prepare_image


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape[0] == 1
        return [[0.1, 0.8, 0.1]]


def test_get_top_prediction_returns_label_and_confidence():
    label, confidence = get_top_prediction([0.2, 0.3, 0.5])

    assert label == "scissors"
    assert confidence == 0.5


def test_get_top_prediction_rejects_wrong_score_count():
    with pytest.raises(ValueError):
        get_top_prediction([0.5, 0.5])


def test_classify_image_with_loaded_model_shape(tmp_path):
    image_path = tmp_path / "paper.png"
    Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)

    label, confidence = classify_image(image_path, model=FakeModel())

    assert label == "paper"
    assert confidence == 0.8


def test_prepare_image_resizes_and_normalizes_pil_image():
    image = Image.new("RGB", (32, 18), color=(0, 128, 255))

    image_array = prepare_image(image)

    assert image_array.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert image_array.dtype == np.float32
    assert image_array.min() >= 0.0
    assert image_array.max() <= 1.0


def test_predict_probabilities_returns_all_classes():
    image_array = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

    probabilities = predict_probabilities(image_array, FakeModel())

    assert probabilities == {"rock": 0.1, "paper": 0.8, "scissors": 0.1}
