# Install EasyOCR
!pip install easyocr

from google.colab import files
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import io
import matplotlib.pyplot as plt
import cv2

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to preprocess image for OCR
def preprocess_image_for_ocr(img):
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast and brightness
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3)  # Increase contrast more aggressively
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(2)  # Increase brightness

    # Convert to NumPy array for further processing
    img_np = np.array(img)
    
    # Apply adaptive thresholding
    img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Use morphological operations to improve text structure
    kernel = np.ones((2, 2), np.uint8)
    img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL Image
    img = Image.fromarray(img_np)
    
    # Resize image for better clarity
    width, height = img.size
    new_size = (width * 2, height * 2)  # Double the size
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Apply sharpening filter
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

# Function to extract text from image using EasyOCR
def extract_text_from_image(img):
    img = preprocess_image_for_ocr(img)
    img_np = np.array(img)
    results = reader.readtext(img_np, detail=0, paragraph=True)
    text = ' '.join(results)
    return text

# Function to handle uploaded images
def process_uploaded_image(uploaded_file):
    # Open the image from bytes
    img = Image.open(io.BytesIO(uploaded_file))

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Extract text from the image
    extracted_text = extract_text_from_image(img)
    print("Extracted Text from Image:")
    print(extracted_text)

# Upload and process images
uploaded = files.upload()

# Process each uploaded file
for filename in uploaded.keys():
    print(f'Processing file: {filename}')
    process_uploaded_image(uploaded[filename])
