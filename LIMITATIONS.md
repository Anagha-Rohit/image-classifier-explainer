# Limitations

Although the model achieved 100% accuracy on the validation split of the Rock Paper Scissors dataset, this should not be interpreted as perfect real-world performance.

The dataset used in this project is relatively small, clean, and well-controlled. Most images are centered, clearly labeled, and captured under similar conditions. As a result, the model may perform significantly worse when deployed on images that differ from the training data.

## Known Limitations

### Lighting Conditions

The model was trained on images captured under relatively consistent lighting. Predictions may become less reliable when images are:

* too dark
* overexposed
* affected by shadows
* captured in unusual lighting conditions

### Background Clutter

Many training images contain simple backgrounds. Complex or distracting backgrounds may reduce prediction accuracy because the model can learn patterns that are not directly related to the hand gesture.

### Camera Angle and Hand Position

The model may struggle with:

* unusual hand orientations
* partially visible hands
* hands positioned near the edge of the image
* perspectives that were not well represented in the training dataset

### Generalization

A high validation score on a controlled dataset does not guarantee strong performance on completely new images.

The model has not been evaluated on:

* webcam streams
* mobile phone photos
* images from different cameras
* images collected from a broader group of users

### Explainability Limitations

Future explainability techniques such as Grad-CAM can help visualize which regions of an image influence predictions. However, these visualizations should be treated as helpful insights rather than complete explanations of the model's reasoning process.

## Future Improvements

Potential ways to improve the model include:

* collecting more diverse training images
* adding data augmentation
* testing on real-world photographs
* evaluating performance across different users and environments
* improving model robustness to lighting and background variations

These limitations represent opportunities for future learning and development as the project evolves.
