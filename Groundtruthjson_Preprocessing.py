import os
import pandas as pd
import json
import re

def sort_files(files):
    """ Custom sort files based on the numeric part if present, or alphabetically otherwise. """
    def file_key(f):
        # Extract numbers as integers for sorting if present, default to filename for alphabetic sort
        num = re.search(r'(\d+)', f)
        if num:
            return int(num.group(1))
        return f
    return sorted(files, key=file_key)

# Defining the structure of our dataframe
columns = ["Issue Number", "Node", "Metric", "Aggregation", "Timestamp", "Root Cause Node", "Root Cause Metric", "Reproducibility Command"]
data = []

# Directory containing the JSON files
json_directory = "Target_low_test"

# List all JSON files in the directory
json_files = [os.path.join(json_directory, f) for f in os.listdir(json_directory) if f.endswith('.json')]

# Sort files to maintain a consistent and correct order
json_files = sort_files(json_files)

# Read and process each JSON file
for i, file_path in enumerate(json_files):
    try:
        with open(file_path, 'r') as file:
            print(f"Processing file: {file_path}")  # Debugging statement
            issue_data = json.load(file)
            row = [
                i,  # Issue number
                issue_data['target']['node'],
                issue_data['target']['metric'],
                issue_data['target']['agg'],
                pd.to_datetime(issue_data['target']['timestamp'], unit='s'),
                issue_data['root_cause']['node'],
                issue_data['root_cause'].get('metric'),
                issue_data['metadata']['reproducibility']['command']
            ]
            data.append(row)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")  # Error handling

# Creating the DataFrame
issues_df = pd.DataFrame(data, columns=columns)

# Save DataFrame to an Excel file
excel_filename = "issues_data.csv"
csv_path = os.path.join(json_directory, excel_filename)
issues_df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")
