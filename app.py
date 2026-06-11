"""Streamlit app for rock-paper-scissors image prediction."""

from pathlib import Path

from PIL import Image
import streamlit as st

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from src.predict import get_top_prediction, load_trained_model, predict_probabilities
from src.data_utils import preprocess_pil_image


st.set_page_config(page_title="RPS Prediction App", page_icon="🖼️", layout="centered")


@st.cache_resource
def get_model(model_path: str = MODEL_PATH):
    """Load and cache the trained model once per app session."""
    return load_trained_model(model_path)


def main():
    st.title("Rock, Paper, Scissors Prediction App")
    st.write(
        """
        Upload a hand-gesture image and the trained model will predict whether it is
        rock, paper, or scissors. This app focuses only on prediction for Phase 3.
        """
    )
    st.caption(
        "Debug note: the app uses the same class order and pixel scaling as training: "
        f"`rock, paper, scissors`, resized to `{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}`, "
        "then normalized to `0-1`."
    )

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        st.error(
            f"Model file not found at `{model_file}`. "
            "Train the model first so the app can load `models/rps_classifier.keras`."
        )
        st.stop()

    try:
        model = get_model()
    except OSError as error:
        st.error(f"Could not load the model file: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Something went wrong while loading the model: {error}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Choose a photo of a hand showing rock, paper, or scissors.",
    )

    if uploaded_file is None:
        st.info("Upload an image to see the prediction results.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        image_array = preprocess_pil_image(image)
        probabilities = predict_probabilities(image_array, model)
        predicted_class, confidence = get_top_prediction(list(probabilities.values()))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.subheader("Prediction Result")
    st.success(f"Predicted class: {predicted_class}")
    st.write(f"Confidence score: {confidence:.2%}")

    st.subheader("Class Probabilities")
    probability_data = {
        class_name.title(): probability for class_name, probability in probabilities.items()
    }
    st.bar_chart(probability_data)

    for class_name, probability in probabilities.items():
        st.write(f"{class_name.title()}: {probability:.2%}")

    st.subheader("Prediction Debug")
    st.write(f"Class label order: {', '.join(CLASS_NAMES)}")
    st.write(f"Preprocessed image shape: {image_array.shape}")
    st.write(
        "Pixel range after preprocessing: "
        f"{image_array.min():.3f} to {image_array.max():.3f}"
    )
    st.info(
        "If many real-world images are predicted as rock, we should treat that as a real "
        "model limitation, not hide it. The class probabilities above help us inspect that behavior."
    )


if __name__ == "__main__":
    main()
