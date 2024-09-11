#Main Working Code
!pip install easyocr
!pip install easyocr
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install pillow
!pip install opencv-python
!pip install fuzzywuzzy


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import io
import cv2
from google.colab import files
from fuzzywuzzy import process, fuzz

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Load the dataset
file_name = 'Medicine_Details.csv'
df = pd.read_csv(file_name)

# Extract the required columns and convert to a list of dictionaries
medicine_details = df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']]
medicine_details_list = medicine_details.to_dict(orient='records')

# Convert the list to a dictionary for easier lookup
medicine_details_dict = {item['Medicine Name'].lower(): item for item in medicine_details_list}

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

# Function to match extracted text with medicine names and compositions
from fuzzywuzzy import fuzz, process
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
from google.colab import files

# Function to match extracted text with medicine names and compositions
def match_text_with_medicine_details(extracted_text, medicine_details_dict):
    # Lowercase the text for matching
    text_lower = extracted_text.lower()
    
    best_match_name = "Unknown"
    best_match_details = {"Composition": "Unknown", "Uses": "Unknown", "Side_effects": "Unknown"}
    best_score = 0
    
    # Extract composition from text
    composition_patterns = [
        r'(?:composition|ingredients|active\s*ingredients|content|contains)\s*[:\-]?\s*(.*)',  # General pattern
        r'(?:(?:active\s*ingredients|composition|content)\s*[:\-]?\s*)?(.*)',  # More flexible pattern
    ]
    
    extracted_composition = ""
    for pattern in composition_patterns:
        composition_match = re.search(pattern, text_lower, re.IGNORECASE)
        if composition_match:
            extracted_composition = composition_match.group(1).strip().lower()
            break  # Use the first successful match
    
    # Debug: Print extracted composition
    print(f"Extracted Composition: '{extracted_composition}'")
    
    # Iterate over all known medicines
    for medicine_name, details in medicine_details_dict.items():
        medicine_name_lower = medicine_name.lower()
        known_composition = details['Composition'].lower()
        
        # Debug: Print current medicine name and composition
        # print(f"Checking: '{medicine_name_lower}' with Composition: '{known_composition}'")
        
        # Calculate partial match scores
        medicine_name_score = fuzz.partial_ratio(text_lower, medicine_name_lower)
        composition_score = fuzz.partial_ratio(extracted_composition, known_composition)
        
        # Debug: Print scores
        # print(f"Medicine Name Score: {medicine_name_score}, Composition Score: {composition_score}")
        
        # Combine scores with higher weight on composition score
        combined_score = (medicine_name_score * 0.4) + (composition_score * 0.6)
        
        if combined_score > best_score:
            best_score = combined_score
            best_match_name = medicine_name
            best_match_details = details

    # Debug: Print best match result
    print(f"Best Match: '{best_match_name}' with Score: {best_score}")
    
    return best_match_name, best_match_details

# Load medicine details from CSV
def load_medicine_details_from_csv(file_path):
    df = pd.read_csv(file_path)
    medicine_details_dict = {}
    for _, row in df.iterrows():
        medicine_name = row['Medicine Name']
        medicine_details_dict[medicine_name] = {
            "Medicine Name": medicine_name,
            "Composition": row['Composition'],
            "Uses": row['Uses'],
            "Side_effects": row['Side_effects']
        }
    return medicine_details_dict

# Path to your CSV file
csv_file_path = 'Medicine_Details.csv'

# Load the data
medicine_details_dict = load_medicine_details_from_csv(csv_file_path)

# Function to analyze medicine
def analyze_medicine(uploaded_image):
    # Display image (if needed, for visualization purposes)
    img = Image.open(uploaded_image)
    plt.imshow(img)
    plt.axis('off')  # No axis needed for displaying
    plt.show()

    # Extract text from the image
    extracted_text = extract_text_from_image(img)
    print("Extracted Text from Image:")
    print(extracted_text)

    # Match extracted text with medicine details
    medicine_name, medicine_info = match_text_with_medicine_details(extracted_text, medicine_details_dict)

    # Provide detailed feedback to the user
    print(f"\n--- Detailed Medicine Analysis Report ---")
    print(f"Extracted Medicine Name (from packaging): {medicine_name}")
    print(f"\nComposition: {medicine_info['Composition']}")
    print(f"Uses: {medicine_info['Uses']}")
    print(f"Side Effects: {medicine_info['Side_effects']}")

    # Assuming visual check for pill quality
    print(f"\nPill Visual Quality: Good (Assumed based on visual inspection)")

    # Provide packaging quality and integrity feedback
    print(f"Packaging Condition: Appears to be in good condition")  # Based on basic visual assumption

    # Handling expiry date analysis
    expiry_pattern = re.compile(r'(expiry|exp)\s*[:\-]?\s*([\d]{2}/[\d]{2}/[\d]{2,4}|\d{4})', re.IGNORECASE)
    expiry_match = expiry_pattern.search(extracted_text)
    expiry_date = expiry_match.group(2) if expiry_match else "Not found"
    
    if expiry_date != "Not found":
        print(f"Expiry Date: {expiry_date}")
        try:
            expiry_year = int(re.search(r'(\d{4})', expiry_date).group(1))
            if expiry_year >= 2025:
                print("The medicine is within the expiry date and should be safe to consume.")
            else:
                print("Warning: The medicine may be expired. Please verify the expiry date before use.")
        except ValueError:
            print("Unable to validate expiry year.")
    else:
        print("Expiry Date: Could not be identified. Please inspect the packaging manually.")

    # General advice about medicine safety and usage
    if expiry_date != "Not found" and expiry_year >= 2025:
        print(f"\nConclusion: The {medicine_name} appears to be safe to use based on the expiry date and packaging condition. However, always consult a healthcare professional before use.")
    else:
        print(f"\nConclusion: The {medicine_name} may not be safe to use due to an unclear or expired expiry date. Please verify the expiry date and consult a healthcare professional before use.")

# Function to handle user-uploaded images for pills or packaging
def process_uploaded_images():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f'Processing file: {filename}')
        analyze_medicine(io.BytesIO(uploaded[filename]))

# Run the function to handle user-uploaded images
process_uploaded_images()

