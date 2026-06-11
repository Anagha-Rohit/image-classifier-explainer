from pathlib import Path


def test_required_project_files_exist():
    required_paths = [
        "README.md",
        "MODEL_CARD.md",
        "LIMITATIONS.md",
        "FUTURE_WORK.md",
        "app.py",
        "train.py",
        "evaluate.py",
        "requirements.txt",
        "docs/how-it-works.md",
        "docs/architecture.md",
        "src/config.py",
        "src/model.py",
        "src/predict.py",
        "src/explainability.py",
        "tests/test_project_structure.py",
    ]

    for relative_path in required_paths:
        assert Path(relative_path).exists(), f"Missing required path: {relative_path}"
