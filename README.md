# MediScan: Medicine Analysis and Verification System

MediScan is an intelligent system designed to analyze and verify medicine details from images, ensuring that users can make informed decisions about their medication. Utilizing Optical Character Recognition (OCR) and fuzzy matching techniques, MediScan extracts relevant information from medicine packaging and cross-references it with a comprehensive database to provide detailed feedback on the medicine's safety and usability.

## Features

- **OCR Processing**: Extracts text from medicine packaging images using EasyOCR.
- **Text Matching**: Uses fuzzy matching to identify medicine names and compositions from extracted text.
- **Detailed Analysis**: Provides information about the medicine's composition, uses, side effects, and expiry date.
- **Visual and Packaging Quality Assessment**: Assesses the visual quality of the medicine and packaging condition.
- **Expiry Date Verification**: Checks the expiry date and provides warnings if the medicine may be expired.

## Below is the Output and The Sample taken 
- Output - ![medreport](https://github.com/user-attachments/assets/311bc6f3-929d-4639-979e-8b9f2387a1ca)

## Installation

To use MediScan, follow these steps to set up the environment:

1. **Clone the repository:**

    ```bash
    https://github.com/sharmilidas33/MediScan.git
    ```

2. **Install the required packages:**

    You can use `pip` to install the necessary Python packages:

    ```bash
    pip install easyocr pandas numpy matplotlib opencv-python pillow fuzzywuzzy
    ```

3. **Upload the Medicine Details CSV:**

    Place your CSV file containing medicine details (`Medicine_Details.csv`) in the project directory.

## Usage

1. **Prepare your image files:**

    Ensure that you have images of medicine packaging saved on your local machine. The images should be clear and readable.

2. **Run the script:**

    Execute the Python script to process and analyze the images. The script will handle image uploads, perform OCR, match the extracted text with medicine details, and provide a detailed report.

    ```python
    python mediscan.py
    ```

    Alternatively, if you are using Google Colab, you can run the provided code cells to upload and process images.

## Functions

- **`preprocess_image_for_ocr(img)`**: Preprocesses the image to enhance text readability for OCR.
- **`extract_text_from_image(img)`**: Extracts text from the preprocessed image using EasyOCR.
- **`match_text_with_medicine_details(extracted_text, medicine_details_dict)`**: Matches the extracted text with medicine details and provides the best match.
- **`analyze_medicine(uploaded_image)`**: Analyzes the uploaded image, provides detailed feedback, and verifies the expiry date.
- **`process_uploaded_images()`**: Handles user-uploaded images and processes them for analysis.

## Contributing

Contributions to MediScan are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact [your email address].

Happy scanning and stay safe!

