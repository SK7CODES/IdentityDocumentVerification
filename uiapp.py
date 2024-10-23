import streamlit as st
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import re

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the trained model
model = load_model('model/final_document_type_classifier.h5')

# Define class labels
class_labels = {0: 'Aadhar_Card', 1: 'Driving_License', 2: 'Passport'}

# Functions for prediction and information extraction
def predict_document_type(img):
    """Preprocess image and predict document type."""
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

def extract_document_info(img, document_type):
    """Extract specific information based on document type."""
    extracted_text = pytesseract.image_to_string(img)
    if document_type == "Aadhar_Card":
        result = re.findall(r'\b\d{4}\s\d{4}\s\d{4}\b', extracted_text)
        return f"Aadhaar Number: {result[0]}" if result else "Aadhaar Number not found."
    elif document_type == "Driving_License":
        result = re.findall(r'\b[A-Z]/[A-Z]{2}/\d{3}/\d{6}/\d{4}\b', extracted_text)
        return f"License Number: {result[0]}" if result else "License Number not found."
    elif document_type == "Passport":
        result = re.findall(r'\b[A-Z]{1}[0-9]{7}\b', extracted_text)
        return f"Passport Number: {result[0]}" if result else "Passport Number not found."

# Streamlit UI
st.set_page_config(page_title="Document Classifier & Extractor", page_icon="📝", layout="wide")

# Title and introduction
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📄 Document Type Classification & Information Extraction</h1>", unsafe_allow_html=True)
st.write("Upload a document, select the type, and let the model classify and extract key information!")

# Create two columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Step 1: Select Document Type")
    selected_doc_type = st.selectbox(
        "Which type of document are you uploading?", 
        ["Aadhar_Card", "Driving_License", "Passport"]
    )

    st.markdown("### Step 2: Upload the Image")
    uploaded_file = st.file_uploader("Upload your document image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

with col2:
    st.markdown("### Preview and Result")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Predict document type
        predicted_type = predict_document_type(img)
        st.success(f"**Predicted Document Type:** {predicted_type}")

        # Check if the predicted type matches the selected type
        if predicted_type == selected_doc_type:
            st.write("✅ The document type matches your selection.")
            extracted_info = extract_document_info(img, predicted_type)
            st.info(f"**Extracted Information:**\n{extracted_info}")
        else:
            st.error(f"⚠️ Mismatch: The uploaded document is classified as **{predicted_type}**, but you selected **{selected_doc_type}**.")
    else:
        st.write("👈 Please upload an image to proceed.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Streamlit</p>", 
    unsafe_allow_html=True
)
