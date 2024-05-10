import os
import pandas as pd
import re

def normalize_service_name(name):
    """
    Remove trailing numbers and additional unwanted characters from service names.
    """
    name = re.sub(r'[\d\.]+$', '', name)
    return name

def transpose_csv_files(input_directory):
    """
    Transpose CSV files in the given directory, normalize service names, and convert timestamps.
    """
    print("Checking directory:", input_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            print(f"Found CSV file: {filename}")
            try:
                data = pd.read_csv(file_path)
                print(f"Data loaded for {filename}: {data.head()}")
                timestamps = pd.to_numeric(data.iloc[3:, 0], errors='coerce')
                data.iloc[3:, 0] = pd.to_datetime(timestamps, unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
                data = data.set_index('microservice')
                data_transposed = data.transpose()
                data_transposed_reset = data_transposed.reset_index()
                data_transposed_reset.rename(columns={'index': 'microservice'}, inplace=True)
                data_transposed_reset['microservice'] = data_transposed_reset['microservice'].apply(normalize_service_name)
                output_filename = f'transposed_{filename}'
                output_file_path = os.path.join(input_directory, output_filename)
                data_transposed_reset.to_csv(output_file_path, index=False)
                print(f"Transposed file saved: {output_file_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

def transform_issue_dataset(file_path, output_directory):
    """
    Transform the issue dataset from a transposed and normalized CSV format.
    """
    print(f"Starting transformation of {file_path}")
    issue_df = pd.read_csv(file_path)
    if issue_df.empty:
        print(f"No data in file {file_path}. Exiting transformation.")
        return

    try:
        issue_df['metric_statistic'] = issue_df['metric'] + '_' + issue_df['statistic']
        issue_df.drop(['metric', 'statistic', 'unix_timestamp'], axis=1, inplace=True)
        melted_df = pd.melt(issue_df, id_vars=['microservice', 'metric_statistic'], var_name="timestamp", value_name="value")
        pivoted_df = melted_df.pivot_table(index=["microservice", "timestamp"], columns="metric_statistic", values="value", aggfunc='mean').reset_index()
        pivoted_df.fillna(0, inplace=True)
        output_path = os.path.join(output_directory, f'transformed_{os.path.basename(file_path)}')
        pivoted_df.to_csv(output_path, index=False)
        print(f"Transformed file saved to {output_path}")
    except Exception as e:
        print(f"Error during transformation: {e}")

# Example usage
input_directory = 'Metrics_temp2_test'
output_directory = 'Transformed_Metrics_temp2_test'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

transpose_csv_files(input_directory)

for file in os.listdir(input_directory):
    if file.startswith('transposed_') and file.endswith('.csv'):
        file_path = os.path.join(input_directory, file)
        transform_issue_dataset(file_path, output_directory)
