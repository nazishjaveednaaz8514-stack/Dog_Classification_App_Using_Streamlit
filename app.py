import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def prepare_image(image_file):
    """Prepare the image for prediction by resizing and preprocessing it."""
    img = Image.open(image_file)  # Open the image
    img = img.resize((224, 224))  # Resize to 224x224 as ResNet50 expects this size
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for ResNet50
    return img_array

def classify_image(preprocessed_image):
    """Classify the image and return the top prediction."""
    preds = model.predict(preprocessed_image)  # Get predictions
    decoded = decode_predictions(preds, top=1)[0]  # Decode the top prediction
    class_name, confidence = decoded[0][1], decoded[0][2]
    return class_name, confidence

def is_cat(predicted_class):
    """Check if the predicted class corresponds to a cat."""
    cat_keywords = ["cat", "kitten", "siamese", "tabby"]
    for keyword in cat_keywords:
        if keyword in predicted_class.lower():
            return True
    return False

# Streamlit User Interface
st.title("Cat Image Classification App")
st.write("This app predicts whether an uploaded image is of a cat.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    preprocessed_image = prepare_image(uploaded_file)

    # Classify the image
    class_name, confidence = classify_image(preprocessed_image)

    # Check if the image is of a cat
    if is_cat(class_name):
        st.success(f"The image is classified as a CAT with {confidence * 100:.2f}% confidence!")
    else:
        st.error(f"The image is NOT a cat. It is classified as a {class_name} with {confidence * 100:.2f}% confidence.")
