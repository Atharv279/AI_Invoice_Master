# AI Invoice Master

**AI Invoice Master** is an AI-powered tool that extracts key details from invoices in multiple languages. This application leverages **Tesseract OCR** for text extraction, **Google Gemini AI** for question answering, and Streamlit for a user-friendly interface.

## Features

- **Image Preprocessing:** Rotate, crop, and adjust brightness/contrast of uploaded images to enhance the quality for OCR processing.
- **OCR Text Extraction:** Extract text from invoice images using **Tesseract OCR**.
- **Invoice Detection:** Automatically detects if the uploaded image contains invoice-related information (e.g., invoice number, total amount).
- **Interactive Q&A:** Allows users to ask questions based on the extracted invoice data using **Google Gemini AI**.
- **Download Data:** Export extracted invoice details as **JSON** or **CSV** files for further use.

