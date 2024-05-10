import pandas as pd
import re
import json
from datetime import datetime

def load_path_sets(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_nodes_in_text(text):
    root_cause_pattern = r"The singular root cause node among the dependencies and dependents is (.*?)(?=\. The primary)"
    target_node_pattern = r"The primary target node among the dependents directly impacted by this anomaly is (.*?)(?=\.)"
    root_cause_match = re.search(root_cause_pattern, text, re.DOTALL)
    target_node_match = re.search(target_node_pattern, text, re.DOTALL)
    root_cause = root_cause_match.group(1).strip() if root_cause_match else None
    target_node = target_node_match.group(1).strip() if target_node_match else None
    return root_cause, target_node

def evaluate_model_response(root_cause, target_node, path_sets, top_n=3):
    if not root_cause or not target_node:
        return {}, "Incorrect", 0

    path_scores = [(path, abs(path.index(root_cause) - path.index(target_node)))
                   for path in path_sets if root_cause in path and target_node in path]
    path_scores.sort(key=lambda x: x[1])  # Sort by closeness between nodes

    top_paths = {f"Top {i+1} Path": x[0] for i, x in enumerate(path_scores[:top_n])}

    evaluation = 'Correct' if any(root_cause in path and target_node in path for path in top_paths.values()) else 'Incorrect'
    score = 1 if evaluation == 'Correct' else 0

    return top_paths, evaluation, score

def process_csv_data(file_path, path_sets, top_n=3):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    results = []

    for _, row in data.iterrows():
        root_cause, target_node = find_nodes_in_text(row['Summary'])
        valid_paths, evaluation, score = evaluate_model_response(root_cause, target_node, path_sets, top_n)
        result_row = {
            'Issue_ID': row['Issue_ID'],
            "Scenario": row['Scenario'],
            'Service': row['Service'],
            'Timestamp': row['Timestamp'],
            'MAD Score': row['MAD Score'],
            'Affected Metric': row['Affected Metric'],
            'Prompt': row['Prompt'],
            'Hypothesis': row['Hypothesis'],
            'Summary': row['Summary'],
            'Root Cause': root_cause,
            'Target Node': target_node,
            'Evaluation': evaluation,
            'Score': score,
            **valid_paths  # Unpack top paths directly into the row
        }
        results.append(result_row)

    return pd.DataFrame(results)

# Usage Example
path_sets = load_path_sets('path_sets.json')
file_path = r"Path_evaluation_OpenHermesMistral_V2.0.csv"
processed_data = process_csv_data(file_path, path_sets, top_n=3)  # Adjust top_n as needed

# Generating a timestamped filename for output
output_file_path = f"mixtral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
processed_data.to_csv(output_file_path, index=False)
print(f"Data saved to {output_file_path}")
