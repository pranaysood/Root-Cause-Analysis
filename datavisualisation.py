import pandas as pd
import matplotlib.pyplot as plt

def load_and_visualize(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Iterate through each unique scenario in the dataset
    for scenario in data['Scenario'].unique():
        print(f"Visualizations for: {scenario}")
        scenario_data = data[data['Scenario'] == scenario]

        # Plotting MAD Score Distribution
        plt.figure(figsize=(12, 6))
        plt.hist(scenario_data['MAD Score'], bins=10, color='skyblue', alpha=0.7)
        plt.title(f'Distribution of MAD Scores for {scenario}')
        plt.xlabel('MAD Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # Plotting Evaluation Results - Count each unique evaluation per issue
        issue_evaluation_counts = scenario_data.groupby(['Issue_ID', 'Evaluation']).size().groupby(level=1).sum()
        colors = {'Correct': 'green', 'Incorrect': 'red'}  # Color mapping
        issue_evaluation_counts.plot(kind='bar', color=[colors[eval] for eval in issue_evaluation_counts.index], alpha=0.8)
        plt.title(f'Evaluation of Issue Resolutions for {scenario}')
        plt.xlabel('Evaluation')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Plotting Service Name Counts by Evaluation Type
        service_evaluation = scenario_data.groupby('Service')['Evaluation'].value_counts().unstack().fillna(0)
        # Ensure both 'Correct' and 'Incorrect' are present
        service_evaluation = service_evaluation.reindex(columns=['Correct', 'Incorrect'], fill_value=0)
        service_evaluation[['Correct', 'Incorrect']].plot(kind='barh', stacked=True, color=['green', 'red'], alpha=0.75)
        plt.title(f'Service Name Count by Evaluation Type for {scenario}')
        plt.xlabel('Count')
        plt.ylabel('Service')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Plotting Correct vs Incorrect Predictions per Issue
        issue_evaluation = scenario_data.groupby('Issue_ID')['Evaluation'].value_counts().unstack().fillna(0)
        # Ensure both 'Correct' and 'Incorrect' are present
        issue_evaluation = issue_evaluation.reindex(columns=['Correct', 'Incorrect'], fill_value=0)
        issue_evaluation[['Correct', 'Incorrect']].plot(kind='bar', stacked=True, color=['green', 'red'], alpha=0.75)
        plt.title(f'Correct vs Incorrect Predictions per Issue for {scenario}')
        plt.xlabel('Issue ID')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# Example usage
load_and_visualize('/content/Path_Based_Evaluation_Mixtral.csv')
