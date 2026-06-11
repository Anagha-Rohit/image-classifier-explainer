import numpy as np
import pytest

from evaluate import build_confusion_matrix, calculate_metrics, get_validation_split


def test_get_validation_split_is_deterministic():
    image_paths = [(f"image_{index}.jpg", "rock") for index in range(10)]

    first_split = get_validation_split(image_paths, validation_split=0.2)
    second_split = get_validation_split(image_paths, validation_split=0.2)

    assert first_split == second_split
    assert len(first_split) == 2


def test_get_validation_split_rejects_invalid_values():
    with pytest.raises(ValueError):
        get_validation_split([("image.jpg", "rock")], validation_split=1.0)


def test_build_confusion_matrix_counts_predictions():
    results = [
        {"true_label": "rock", "predicted_label": "rock"},
        {"true_label": "rock", "predicted_label": "paper"},
        {"true_label": "paper", "predicted_label": "paper"},
        {"true_label": "scissors", "predicted_label": "rock"},
    ]

    matrix = build_confusion_matrix(results)

    assert matrix.tolist() == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_calculate_metrics_returns_macro_scores():
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )

    metrics = calculate_metrics(matrix)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    assert metrics["recall"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert metrics["f1"] == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)
