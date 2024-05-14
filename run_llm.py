import pandas as pd
import json
from scipy.stats import median_abs_deviation
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util

# Load the dataset and service information
file_path = '/path_to_csv'
json_path = '/path_to_json'
metrics_data = pd.read_csv(file_path)
metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'])

with open(json_path, 'r') as f:
    data = json.load(f)
    service_info = {service['name']: {'dependencies': service['dependencies'], 'dependents': service['dependents']} for service in data['services']}

# Load models
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)
model_scoring = SentenceTransformer('distilbert-base-nli-mean-tokens')

def service_exists(service_name):
    """Check if a service exists in the loaded service information."""
    return service_name in service_info

def calculate_mad_scores(dataframe):
    """Calculate MAD scores."""
    metrics = ['availability_Average', 'latency_Average', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'requests_Sum']
    mad_columns = [m + '_MAD' for m in metrics]
    for m in metrics:
        dataframe[m + '_MAD'] = dataframe.groupby('microservice')[m].transform(lambda x: median_abs_deviation(x, scale='normal'))
    dataframe['Max_MAD_Score'] = dataframe[mad_columns].max(axis=1)
    dataframe['Metric_With_Max_MAD'] = dataframe[mad_columns].idxmax(axis=1).str.replace('_MAD', '')
    return dataframe.loc[dataframe.groupby('microservice')['Max_MAD_Score'].idxmax()]


def preprocess_input(dependencies, dependents):
    """Preprocess input to ensure clear, structured prompts."""
    dependencies_formatted = ', '.join(dependencies) if dependencies else 'None'
    dependents_formatted = ', '.join(dependents) if dependents else 'None'
    return dependencies_formatted, dependents_formatted

def clean_hypothesis(hypothesis):
    """Process the hypothesis to refine and extract the relevant section."""
    lines = hypothesis.split('\n')
    final_hypothesis_start = next((i for i, line in enumerate(lines) if "Final Hypothesis:" in line), None)
    return '\n'.join(lines[final_hypothesis_start + 1:]) if final_hypothesis_start is not None else hypothesis

def generate_analysis_prompt(service_name, mad_score, affected_metric, dependencies, dependents):
    """Generate a structured analysis prompt for the model."""
    dependencies_formatted, dependents_formatted = preprocess_input(dependencies, dependents)
    prompt = f"""
    An anomaly with a Median Absolute Deviation (MAD) score of {mad_score} has been detected in the {service_name} service's {affected_metric} metric, indicating a substantial deviation impacting its performance. This service is a critical component of a pet adoption website's microservices architecture.

    Dependencies involved include: {dependencies_formatted}.
    The service also serves as a crucial dependency for: {dependents_formatted}.

    Your analysis should focus on identifying a singular root cause from among the dependencies. This cause should directly contribute to the anomaly in the {affected_metric} metric. Consider each dependency's role and potential issues that could lead to such a deviation.

    Additionally, pinpoint the primary dependent (target node) that is most directly affected by this anomaly. This should be the service that relies on {service_name} and would face the most significant impact due to the anomaly in {affected_metric}.

    Please provide a concise and focused hypothesis on:
    1. The singular root cause node among the dependencies and dependents.
    2. The primary target node among the dependents directly impacted by this anomaly.

    Your analysis will guide subsequent investigation and mitigation efforts.
    """
    return prompt.strip()

def analyze_root_cause(anomaly_row):
    """Analyze root cause."""
    service_name = anomaly_row['microservice']
    if not service_exists(service_name):
        return "Service information not found.", "", []

    prompt = generate_analysis_prompt(
        service_name,
        anomaly_row['Max_MAD_Score'],
        anomaly_row['Metric_With_Max_MAD'],
        service_info[service_name]['dependencies'],
        service_info[service_name]['dependents']
    )

    response = text_generator(prompt, max_length=1200, num_return_sequences=1, temperature=0.7, top_p=0.9)[0]['generated_text']
    cleaned_response = clean_hypothesis(response)
    return prompt, cleaned_response

def main():
    """Main function to analyze top services."""
    metrics_data_filtered = calculate_mad_scores(metrics_data)
    top_services = metrics_data_filtered.sort_values(by='Max_MAD_Score', ascending=False).head(2)
    results = []

    for _, row in top_services.iterrows():
        prompt, cleaned_response = analyze_root_cause(row)
        results.append({
            'Service': row['microservice'],
            'Timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'MAD Score': row['Max_MAD_Score'],
            'Affected Metric': row['Metric_With_Max_MAD'],
            'Prompt': prompt,
            'Hypothesis': cleaned_response,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('/content/RESULTS_MISTRAL_ISSUE2_TEMP2_TEST.csv', index=False)
    print("Analysis results saved.")

if __name__ == "__main__":
    main()
