from fuzzywuzzy import fuzz
import re
import pandas as pd

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
        print(f"Medicine Name Score: {medicine_name_score}, Composition Score: {composition_score}")
        
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

# Test the function with known values
test_text = "12x 5x 10 Tabiets Paracetamol Tablets IP 500 mg PARACIP-500 Ti _ kctoe Fu 500 @qpla J0 2 3"
result = match_text_with_medicine_details(test_text, medicine_details_dict)
print("Test Match Result:")
print(result)
