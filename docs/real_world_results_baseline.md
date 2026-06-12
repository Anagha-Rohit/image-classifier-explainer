# Real-World Test Results

This report measures how the trained model behaves on images outside the original dataset.

## Summary

* Total real-world images: 14
* Accuracy: 21.43%
* Precision: 24.21%
* Recall: 21.67%
* F1 score: 22.49%
* Mistakes saved to `docs/screenshots/real_world_failures`: 11

## Class Coverage

| Class | Images |
| --- | ---: |
| rock | 5 |
| paper | 5 |
| scissors | 4 |

## Why Real-World Results Can Be Worse

Validation accuracy can look very high because the validation split comes from the same dataset distribution as training: similar backgrounds, framing, lighting, and hand positions.
Real-world images are usually messier. Different cameras, shadows, cluttered rooms, partial hands, and unusual angles can push the model away from the patterns it learned during training.
That means strong validation accuracy is useful, but it is not the same thing as strong real-world generalization.

## Example Failure

A real-world scissors image was predicted as rock: `real_world_tests/scissors/images-2.jpeg`.
The model gave `rock` a confidence score of 98.89%, even though the true label was `scissors`.
This kind of mistake is exactly why real-world testing matters: the model may rely on background cues, framing, or gesture shape details that looked reliable in the original dataset but do not transfer cleanly to new images.

## Detailed Predictions

| Image | True Label | Predicted Label | Confidence |
| --- | --- | --- | ---: |
| real_world_tests/rock/download.jpeg | rock | paper | 77.54% |
| real_world_tests/rock/images (1).jpeg | rock | paper | 100.00% |
| real_world_tests/rock/images (2).jpeg | rock | paper | 50.64% |
| real_world_tests/rock/images-3.jpeg | rock | rock | 100.00% |
| real_world_tests/rock/images.jpeg | rock | paper | 94.68% |
| real_world_tests/paper/74144ac70b25504f70817cf4962f295a_t.jpeg | paper | scissors | 89.94% |
| real_world_tests/paper/Rock-paper-scissors_(paper).png | paper | rock | 100.00% |
| real_world_tests/paper/images (3).jpeg | paper | paper | 99.57% |
| real_world_tests/paper/images (4).jpeg | paper | rock | 62.35% |
| real_world_tests/paper/images (5).jpeg | paper | scissors | 80.05% |
| real_world_tests/scissors/images (6).jpeg | scissors | paper | 99.70% |
| real_world_tests/scissors/images (7).jpeg | scissors | scissors | 99.69% |
| real_world_tests/scissors/images-2.jpeg | scissors | rock | 98.89% |
| real_world_tests/scissors/scissors-gesture-on-left-hand-for-concept-of-rock-paper-scissors-game-isolated-on-white-background-free-photo.jpeg | scissors | paper | 87.33% |
