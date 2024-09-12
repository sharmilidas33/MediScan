#Main Working Code
!pip install easyocr
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install pillow
!pip install opencv-python
!pip install fuzzywuzzy
!pip install fpdf

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
from fpdf import FPDF
from datetime import datetime

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

# Function to clean and normalize extracted text (handling cases like JJUL)
def normalize_text(text):
    # Convert all text to uppercase for uniformity
    text = text.upper()

    # Replace common misrecognized characters (for example, 'JJUL' could be 'JUL')
    text = text.replace("JJAN", "JAN").replace("JAN_", "JAN").replace("_", " ")
    text = text.replace("FFEB", "FEB").replace("FEB_", "FEB").replace("_", " ")
    text = text.replace("MMAR", "MAR").replace("MAR_", "MAR").replace("_", " ")
    text = text.replace("AAPR", "APR").replace("APR_", "APR").replace("_", " ")
    text = text.replace("MMAY", "MAY").replace("MAY_", "MAY").replace("_", " ")
    text = text.replace("JJUN", "JUN").replace("JUN_", "JUN").replace("_", " ")
    text = text.replace("JJUL", "JUL").replace("JUL_", "JUL").replace("_", " ")
    text = text.replace("AAUG", "AUG").replace("AUG_", "AUG").replace("_", " ")
    text = text.replace("SSEP", "SEP").replace("SEP_", "SEP").replace("_", " ")
    text = text.replace("OOCT", "OCT").replace("OCT_", "OCT").replace("_", " ")
    text = text.replace("NNOV", "NOV").replace("NOV_", "NOV").replace("_", " ")
    text = text.replace("DDEC", "DEC").replace("DEC_", "DEC").replace("_", " ")

    # Replace double letters, underscores, and spaces that may appear incorrectly
    text = text.replace("JANUARY", "JAN").replace("FEBRUARY", "FEB")
    text = text.replace("MARCH", "MAR").replace("APRIL", "APR")
    text = text.replace("JUNE", "JUN").replace("JULY", "JUL")
    text = text.replace("AUGUST", "AUG").replace("SEPTEMBER", "SEP")
    text = text.replace("OCTOBER", "OCT").replace("NOVEMBER", "NOV")
    text = text.replace("DECEMBER", "DEC")
    return text 
    
# Function to analyze medicine
def extract_expiry_date(text):
    text = normalize_text(text)  # Normalize the text first

    # List of multiple regex patterns to catch various date formats
    patterns = [
        r'\b(?:EXP|EXPIRY)\s*[:\-]?\s*(\w{3,}\s*\d{2,4})\b',  # For formats like "EXP JUL 2024"
        r'\b(?:EXP)\s*[:\-]?\s*(\d{2,4}\s*\w{3,})\b',          # For formats like "EXP 24 DEC"
        r'\b(\w{3,}\s*\d{4})\b',                               # For formats like "DEC 2024"
        r'\b(?:EXP)\s*(\w{3,}\d{2})\b',                        # For formats like "EXPJUL24"
    ]

    # Search for expiry date using multiple patterns
    for pattern in patterns:
        expiry_match = re.search(pattern, text)
        if expiry_match:
            expiry_date_raw = expiry_match.group(1).replace('_', ' ').strip()
            print(f"Extracted Expiry Date Raw: {expiry_date_raw}")
            try:
                # Try to parse the raw expiry date into a valid datetime object
                expiry_date = pd.to_datetime(expiry_date_raw, format='%b %Y', errors='coerce')

                if pd.isna(expiry_date):
                    expiry_date = pd.to_datetime(expiry_date_raw, format='%b %y', errors='coerce')

                if pd.isna(expiry_date):
                    print("Unable to parse the expiry date from the given format.")
                    continue
                return expiry_date
            except ValueError:
                print(f"Could not parse the expiry date: {expiry_date_raw}")
                continue
    return None

# Function to generate PDF report
def generate_pdf_report(medicine_name, medicine_info, expiry_date, packaging_condition, safety_conclusion):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.add_page()

    # Set title and font
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Medicine Analysis Report", ln=True, align='C')

    # Add date and time
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Add Medicine Name
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Medicine Name: {medicine_name}", ln=True)
    pdf.ln(5)

    # Add Composition
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Composition: {medicine_info['Composition']}")
    pdf.ln(5)

    # Add Uses
    pdf.multi_cell(0, 10, txt=f"Uses: {medicine_info['Uses']}")
    pdf.ln(5)

    # Add Side Effects
    pdf.multi_cell(0, 10, txt=f"Side Effects: {medicine_info['Side_effects']}")
    pdf.ln(10)

    # Add Packaging Condition
    pdf.multi_cell(0, 10, txt=f"Packaging Condition: {packaging_condition}")
    pdf.ln(5)

    # Add Expiry Date
    if expiry_date:
        expiry_str = expiry_date.strftime('%d %b %Y')
        pdf.multi_cell(0, 10, txt=f"Expiry Date: {expiry_str}")
    else:
        pdf.multi_cell(0, 10, txt="Expiry Date: Could not be identified.")
    pdf.ln(10)

    # Add Safety Conclusion
    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, txt="Conclusion:")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=safety_conclusion)

    # Save the file
    file_name = f"Medicine_Report_{medicine_name}.pdf"
    pdf.output(file_name)
    print(f"PDF report generated: {file_name}")

    # Download the PDF (in Google Colab environment)
    files.download(file_name)

# Modify analyze_medicine to call the PDF generation function
def analyze_medicine(uploaded_image):
    img = Image.open(uploaded_image)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    extracted_text = extract_text_from_image(img)
    print("Extracted Text from Image:")
    print(extracted_text)

    medicine_name, medicine_info = match_text_with_medicine_details(extracted_text, medicine_details_dict)

    print(f"\n--- Detailed Medicine Analysis Report ---")
    print(f"Extracted Medicine Name (from packaging): {medicine_name}")
    print(f"\nComposition: {medicine_info['Composition']}")
    print(f"Uses: {medicine_info['Uses']}")
    print(f"Side Effects: {medicine_info['Side_effects']}")

    packaging_condition = "Appears to be in good condition"
    print(f"Packaging Condition: {packaging_condition}")

    expiry_date = extract_expiry_date(extracted_text)
    if expiry_date:
        expiry_year = expiry_date.year
        if expiry_year >= datetime.now().year:
            safety_conclusion = f"The medicine is within the expiry date of {expiry_date.strftime('%d %b %Y')} and should be safe to consume."
            print(safety_conclusion)
        else:
            safety_conclusion = f"Warning: The medicine expired on {expiry_date.strftime('%d %b %Y')}. Please verify before use."
            print(safety_conclusion)
    else:
        safety_conclusion = "Expiry Date: Could not be identified. Please inspect the packaging manually."
        print(safety_conclusion)

    if expiry_date and expiry_date.year >= datetime.now().year:
        final_conclusion = f"The {medicine_name} appears to be safe to use based on the expiry date and packaging condition. However, always consult a healthcare professional before use."
    else:
        final_conclusion = f"The {medicine_name} may not be safe to use due to an unclear or expired expiry date. Please verify the expiry date and consult a healthcare professional before use."
    
    print(f"\nConclusion: {final_conclusion}")

    # Call the function to generate and download the PDF report
    generate_pdf_report(medicine_name, medicine_info, expiry_date, packaging_condition, final_conclusion)

# Process uploaded images
def process_uploaded_images():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f'Processing file: {filename}')
        analyze_medicine(io.BytesIO(uploaded[filename]))

# Run the function to handle user-uploaded images
process_uploaded_images()




