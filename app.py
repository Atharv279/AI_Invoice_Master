import streamlit as st
import json
import pandas as pd
from PIL import Image, ImageEnhance
import pytesseract
import os
import google.generativeai as genai
import numpy as np
import cv2
import re

# Load API key from .env file
from dotenv import load_dotenv
load_dotenv()

# Set Streamlit Page Config
st.set_page_config(page_title="Multivoice Invoice Extractor")

# Custom Styling for buttons and inputs
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
    }
    .stHeader {
        color: #4CAF50;
    }
    .stFileUploader>div>div {
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Display Logo and Title
st.image("E:\INVOICEEXTRACTOR\logo.png", width=100)  # Your logo
st.title("AI Invoice Master")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Image", "Settings", "FAQ"])

# Initialize the uploaded_file variable globally
uploaded_file = None

if page == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader_1")

    if uploaded_file is not None:
        # Display Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Image Preprocessing
        with st.expander("Preprocess Image"):
            rotate_angle = st.slider("Rotate image", 0, 360, 0)
            if rotate_angle != 0:
                image = image.rotate(rotate_angle)
                st.image(image, caption="Rotated Image", use_container_width=True)

            # Cropping functionality
            st.text("Crop the image: Select a region of interest.")
            crop_x = st.slider("X position", 0, image.width, 0)
            crop_y = st.slider("Y position", 0, image.height, 0)
            crop_width = st.slider("Width", 1, image.width, image.width)
            crop_height = st.slider("Height", 1, image.height, image.height)

            if crop_x != 0 or crop_y != 0 or crop_width != image.width or crop_height != image.height:
                cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
                st.image(cropped_image, caption="Cropped Image", use_container_width=True)

            # Brightness/Contrast adjustment
            enhancer = ImageEnhance.Brightness(image)
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
            image = enhancer.enhance(brightness)

            enhancer = ImageEnhance.Contrast(image)
            contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
            image = enhancer.enhance(contrast)
            st.image(image, caption="Adjusted Image", use_container_width=True)

        # Preprocessing Function: Apply Thresholding
        def preprocess_image(image):
            # Convert to grayscale
            gray = np.array(image.convert("L"))
            # Apply thresholding to improve clarity
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            return Image.fromarray(thresh)

        # Apply preprocessing before OCR
        preprocessed_image = preprocess_image(image)

        # Extract text from the image after preprocessing
        def extract_text_from_image(image):
            text = pytesseract.image_to_string(image)
            return text

        extracted_text = extract_text_from_image(preprocessed_image)

        # Display extracted text for debugging
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        # Function to check if the extracted text contains invoice-related keywords
        def is_invoice(text):
            keywords = ["invoice", "invoice number", "total amount", "amount", "subtotal", "billing", "date", "due date"]
            text = text.lower()  # Convert text to lowercase for case-insensitive search
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", text):  # Check for whole word match
                    return True
            return False

        if is_invoice(extracted_text):
            st.write("This is likely an invoice. Proceeding with further processing...")
        else:
            st.write("This document does not appear to be an invoice. Please upload a valid invoice.")

# Interactive Q&A with Gemini model
input_prompt = """
You are an expert in understanding invoices. We will upload an image as an invoice, 
and you will have to answer any questions based on the uploaded invoice image.
"""

if uploaded_file is not None:
    # Prepare the model for interaction
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Extract key details from the invoice text
    def extract_invoice_details(text):
        # Example pattern to find invoice number and total amount
        invoice_number = re.search(r"invoice number:\s*(\S+)", text, re.IGNORECASE)
        total_amount = re.search(r"total amount:\s*([\d,]+)", text, re.IGNORECASE)
        
        details = {}
        if invoice_number:
            details["Invoice Number"] = invoice_number.group(1)
        if total_amount:
            details["Total Amount"] = total_amount.group(1)
        
        return details

    invoice_details = extract_invoice_details(extracted_text)

    if st.button("Ask a Question"):
        user_question = st.text_input("Ask about the invoice details:")
        
        # Provide more structured answers based on extracted details
        response = ""
        if "vendor" in user_question.lower():
            response = "Sorry, vendor information is not available in this invoice."
        elif "invoice number" in user_question.lower():
            response = f"Invoice Number: {invoice_details.get('Invoice Number', 'Not found')}"
        elif "total amount" in user_question.lower():
            response = f"Total Amount: {invoice_details.get('Total Amount', 'Not found')}"
        else:
            response = "Sorry, I couldn't find the information you're looking for."

        st.subheader("Answer:")
        st.write(response)


# Add options to download the extracted data in JSON or CSV format
    if invoice_details:
        # Convert the details to JSON
        json_data = json.dumps(invoice_details, indent=4)
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name="invoice_details.json",
            mime="application/json",
            use_container_width=True
        )

        # Convert the details to CSV
        df = pd.DataFrame([invoice_details])
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="invoice_details.csv",
            mime="text/csv",
            use_container_width=True
        )

elif page == "Settings":
    st.header("Settings")
    st.write("Here you can customize your experience. For example, change image processing options or upload preferences.")

elif page == "FAQ":
    st.header("Frequently Asked Questions")
    st.write("Q: How do I upload an invoice?")
    st.write("A: Simply use the 'Upload Image' section to choose an invoice image file.")
    st.write("Q: How can I adjust the image?")
    st.write("A: Use the 'Preprocess Image' section to rotate, crop, or adjust brightness and contrast.")
