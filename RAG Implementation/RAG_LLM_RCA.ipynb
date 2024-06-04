!pip show pinecone-client
!pip install --upgrade pinecone-client

import pandas as pd
import json
from scipy.stats import median_abs_deviation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import pinecone

# Ensure all tensor operations use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Pinecone client
api_key = '13b1255d-5993-4098-ae44-2b1e0cd62e22'
print("Initializing Pinecone client...")
pc = pinecone.Pinecone(api_key=api_key)
index_name = "synthetic-data-v2"

# Connect to the index
print(f"Connecting to Pinecone index '{index_name}'...")
index = pc.Index(name=index_name)
print("Connected to Pinecone index.")

# Load the dataset and service information
file_path = '/content/synthetic_data_1_1_issue.csv'
json_path = '/content/path_sets.json'
print(f"Loading dataset from '{file_path}'...")
metrics_data = pd.read_csv(file_path)
metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'])
print("Dataset loaded successfully.")

print(f"Loading JSON data from '{json_path}'...")
with open(json_path, 'r') as f:
    data = json.load(f)
    print("JSON data loaded successfully.")
    print("Sample JSON data:", data[:5])

# Process the list of lists in the JSON file to build the service_info dictionary
service_info = {}
for path in data:
    if len(path) > 1:
        for i in range(len(path) - 1):
            service = path[i]
            dependency = path[i + 1]
            if service not in service_info:
                service_info[service] = {'dependencies': [], 'dependents': []}
            if dependency not in service_info[service]['dependencies']:
                service_info[service]['dependencies'].append(dependency)
            if dependency not in service_info:
                service_info[dependency] = {'dependencies': [], 'dependents': []}
            if service not in service_info[dependency]['dependents']:
                service_info[dependency]['dependents'].append(service)

print("Service info processed successfully.")
print(f"Sample service info: {list(service_info.items())[:5]}")

# Load models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B").to(device)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device.index) # Use GPU if available
model_scoring = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)
print("Models loaded successfully.")

def service_exists(service_name):
    """Check if a service exists in the loaded service information."""
    exists = service_name in service_info
    print(f"Service '{service_name}' exists: {exists}")
    return exists

def calculate_mad_scores(dataframe):
    """Calculate MAD scores."""
    print("Calculating MAD scores...")
    metrics = ['availability_Average', 'latency_Average', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'requests_Sum']
    mad_columns = [m + '_MAD' for m in metrics]
    for m in metrics:
        dataframe[m + '_MAD'] = dataframe.groupby('microservice')[m].transform(lambda x: median_abs_deviation(x, scale='normal'))
    dataframe['Max_MAD_Score'] = dataframe[mad_columns].max(axis=1)
    dataframe['Metric_With_Max_MAD'] = dataframe[mad_columns].idxmax(axis=1).str.replace('_MAD', '')
    print("MAD scores calculated successfully.")
    return dataframe.loc[dataframe.groupby('microservice')['Max_MAD_Score'].idxmax()]


def retrieve_related_anomalies(service_name, top_k=3):
    """Retrieve related anomalies and generate textual representations for LLM processing."""
    print(f"Retrieving textual representations for service: {service_name}")
    query_text = service_name
    query_embedding = model_scoring.encode(query_text, convert_to_tensor=True).cpu().numpy().tolist()
    metadata_filter = {"root_cause_node": True}

    try:
        query_result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=metadata_filter)
        print(f"Retrieved {len(query_result['matches'])} matches.")
        if not query_result['matches']:
            print("No matches found in the Pinecone index.")
            return ["Previously, no historical logs have been logged for issues matching the current query parameters."]

        return [match['metadata']['textual_representation'] for match in query_result['matches']]
    except Exception as e:
        print("Query failed with error:", str(e))
        return ["Previously, no historical logs have been logged for issues matching the current query parameters."]


def generate_analysis_prompt(service_name, mad_score, affected_metric, dependencies, dependents, retrieval_results):
    dependencies_formatted = ', '.join(dependencies) if dependencies else 'None'
    dependents_formatted = ', '.join(dependents) if dependents else 'None'
    historical_anomalies = "\n".join([f"{idx + 1}. {anomaly}" for idx, anomaly in enumerate(retrieval_results)])

    prompt = f"""
    An anomaly with a Median Absolute Deviation (MAD) score of {mad_score} has been detected in the '{service_name}' service's '{affected_metric}' metric, indicating a substantial deviation impacting its performance. This service is a critical component of a pet adoption website's microservices architecture.

    **Dependencies involved**: {dependencies_formatted}
    **Dependents impacted**: {dependents_formatted}

    **Historical related anomalies**:
    {historical_anomalies}

    Your analysis should focus on identifying a singular root cause from among the dependencies that directly contributes to the anomaly in the '{affected_metric}' metric. Consider each dependency's role and potential issues that could lead to such a deviation.

    Additionally, pinpoint the primary dependent (target node) that is most directly affected by this anomaly. This should be the service that relies on '{service_name}' and would face the most significant impact due to the anomaly in '{affected_metric}'.

    Please provide a concise and focused hypothesis on:
    1. The singular root cause node among the dependencies.
    2. The primary target node among the dependents directly impacted by this anomaly.

    Your analysis will guide subsequent investigation and mitigation efforts."""
    print(prompt)
    return prompt


def clean_hypothesis(hypothesis):
    """Process the hypothesis to refine and extract the relevant section."""
    lines = hypothesis.split('\n')
    final_hypothesis_start = next((i for i, line in enumerate(lines) if "Final Hypothesis:" in line), None)
    cleaned_hypothesis = '\n'.join(lines[final_hypothesis_start + 1:]) if final_hypothesis_start is not None else hypothesis
    print(f"Cleaned Hypothesis: {cleaned_hypothesis}")
    return cleaned_hypothesis

def analyze_root_cause(anomaly_row):
    """Analyze root cause."""
    service_name = anomaly_row['microservice']
    if not service_exists(service_name):
        return "Service information not found.", "", []

    retrieval_results = retrieve_related_anomalies(service_name)
    prompt = generate_analysis_prompt(
        service_name,
        anomaly_row['Max_MAD_Score'],
        anomaly_row['Metric_With_Max_MAD'],
        service_info[service_name]['dependencies'],
        service_info[service_name]['dependents'],
        retrieval_results
    )

    print(f"Retrieving and generating response based on the prompt...")
    response = text_generator(prompt, max_new_tokens=100, num_return_sequences=1, temperature=0.7, top_p=0.9)[0]['generated_text']
    cleaned_response = clean_hypothesis(response)
    return prompt, cleaned_response


def main():
    """Main function to analyze top services."""
    try:
        print("Starting main function...")
        metrics_data_filtered = calculate_mad_scores(metrics_data)
        top_services = metrics_data_filtered.sort_values(by='Max_MAD_Score', ascending=False).head(1)

        results = []

        for _, row in top_services.iterrows():
            print(f"Analyzing root cause for service: {row['microservice']}")
            if service_exists(row['microservice']):
                prompt, cleaned_response = analyze_root_cause(row)
                results.append({
                    'Service': row['microservice'],
                    'Timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'MAD Score': row['Max_MAD_Score'],
                    'Affected Metric': row['Metric_With_Max_MAD'],
                    'Prompt': prompt,
                    'Hypothesis': cleaned_response,
                })
                print(f"Analysis complete for service: {row['microservice']}")
            else:
                print(f"Service {row['microservice']} not found in service information.")

        results_df = pd.DataFrame(results)
        results_df.to_csv('/content/output_June3_v2.csv', index=False)
        print("Analysis results saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
