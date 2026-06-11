# Real-World Test Results

No real-world test images were found yet.

Add `.png`, `.jpg`, or `.jpeg` files to these folders:

* `real_world_tests/rock`
* `real_world_tests/paper`
* `real_world_tests/scissors`

After that, run `python real_world_evaluation.py` to generate a real report.

This workflow is important because a model can score very highly on a clean validation split and still struggle on new camera angles, lighting conditions, backgrounds, and hand positions.

When a real-world scissors image is predicted as rock, it should be treated as a valuable failure example rather than hidden. The workflow will save that mistake to `docs/screenshots/real_world_failures/` and include it in this report.
