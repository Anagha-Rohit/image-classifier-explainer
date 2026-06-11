from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import IMAGE_SIZE
from src.data_utils import load_image_array, preprocess_pil_image, validate_dataset_directory


def test_load_image_array_resizes_and_normalizes(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 16), color=(255, 0, 0)).save(image_path)

    image_array = load_image_array(image_path)

    assert image_array.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert image_array.dtype == np.float32
    assert image_array.min() >= 0.0
    assert image_array.max() <= 1.0


def test_validate_dataset_directory_requires_class_folders(tmp_path):
    (tmp_path / "rock").mkdir()

    with pytest.raises(FileNotFoundError):
        validate_dataset_directory(tmp_path)


def test_validate_dataset_directory_accepts_expected_class_folders(tmp_path):
    for class_name in ["rock", "paper", "scissors"]:
        Path(tmp_path / class_name).mkdir()

    assert validate_dataset_directory(tmp_path) == tmp_path


def test_preprocess_pil_image_matches_load_image_array(tmp_path):
    image_path = tmp_path / "sample.png"
    pil_image = Image.new("RGB", (24, 16), color=(10, 20, 30))
    pil_image.save(image_path)

    from_path = load_image_array(image_path)
    from_pil = preprocess_pil_image(pil_image)

    assert np.allclose(from_path, from_pil)
