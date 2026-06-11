# How It Works

This project will follow a simple machine learning pipeline:

## 1. Input Image

The user uploads an image of a hand showing either rock, paper, or scissors.

## 2. Preprocessing

Before prediction, the image will be:

- resized to a fixed shape
- converted into numeric pixel values
- normalized so the model receives consistent input

## 3. Model Prediction

A small image classification model will process the image and output a score for each class:

- rock
- paper
- scissors

The class with the highest score becomes the prediction.

## 4. Explainability

After the base classifier works, the project will add a simple explainability technique.

The goal is to help answer a question like:

"Which part of the image influenced the model most?"

One possible method is Grad-CAM, which can highlight important image regions using a heatmap.

## 5. Streamlit Demo

The final user flow in Streamlit will be:

1. upload image
2. preview image
3. get prediction
4. see confidence scores
5. view explainability output

## Why Explainability Matters

Explainability is useful because it helps us:

- debug incorrect predictions
- understand model behavior
- communicate results more clearly
- build better responsible AI habits

