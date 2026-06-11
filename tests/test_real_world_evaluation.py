from pathlib import Path

from real_world_evaluation import (
    build_real_world_report,
    collect_real_world_images,
    ensure_real_world_directories,
    find_scissors_predicted_as_rock,
)


def test_ensure_real_world_directories_creates_expected_folders(tmp_path):
    data_dir = ensure_real_world_directories(tmp_path / "real_world_tests")

    assert data_dir.exists()
    assert (data_dir / "rock").is_dir()
    assert (data_dir / "paper").is_dir()
    assert (data_dir / "scissors").is_dir()


def test_collect_real_world_images_returns_labeled_paths(tmp_path):
    data_dir = ensure_real_world_directories(tmp_path / "real_world_tests")
    image_path = data_dir / "scissors" / "outside_sample.png"
    image_path.write_bytes(b"fake-image-content")

    image_paths = collect_real_world_images(data_dir)

    assert image_paths == [(image_path, "scissors")]


def test_find_scissors_predicted_as_rock_returns_matching_failure():
    results = [
        {"true_label": "paper", "predicted_label": "paper"},
        {"true_label": "scissors", "predicted_label": "rock", "image_path": Path("x.png")},
    ]

    failure = find_scissors_predicted_as_rock(results)

    assert failure == results[1]


def test_build_real_world_report_handles_empty_results(tmp_path):
    report_path = tmp_path / "real_world_results.md"

    build_real_world_report([], report_path, tmp_path / "failures")

    report_text = report_path.read_text(encoding="utf-8")
    assert "No real-world test images were found yet." in report_text
    assert "scissors image is predicted as rock" in report_text


def test_build_real_world_report_mentions_scissors_failure_when_present(tmp_path):
    report_path = tmp_path / "real_world_results.md"
    results = [
        {
            "image_path": Path("real_world_tests/scissors/example.png"),
            "true_label": "scissors",
            "predicted_label": "rock",
            "confidence": 0.72,
            "probabilities": {"rock": 0.72, "paper": 0.18, "scissors": 0.10},
        }
    ]

    build_real_world_report(results, report_path, tmp_path / "failures")

    report_text = report_path.read_text(encoding="utf-8")
    assert "A real-world scissors image was predicted as rock" in report_text
    assert "Validation accuracy can look very high" in report_text
