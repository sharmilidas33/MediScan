from google.colab import files

# Upload the file
uploaded = files.upload()

# Print the uploaded file names
print(uploaded.keys())

#For listing the data
# import pandas as pd

# # Load the dataset (adjust the filename if necessary)
# file_name = 'Medicine_Details.csv'  # Correct filename with .csv extension
# df = pd.read_csv(file_name)  # Default delimiter for CSV is a comma

# # Extract the required columns
# medicine_details = df[['Medicine Name', 'Composition', 'Uses', 'Side_effects']]

# # Convert the extracted data to a list of dictionaries for easier processing
# medicine_details_list = medicine_details.to_dict(orient='records')

# # Print the result
# for item in medicine_details_list:
#     print(item)
