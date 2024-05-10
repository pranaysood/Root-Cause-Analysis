# Root Cause Analysis of Cloud Microservices

## Description
This project, supervised by Dr. Yan Liu and conducted by Gayathiri Elambooranan and Pranay Sood, aims to enhance the understanding and application of root cause analysis in microservice environments. Utilizing the PetShop Dataset, the project employs large language models (LLMs) and traditional models to simulate and analyze real-world microservice performance issues, leveraging data-driven insights and zero-shot learning capabilities.

## Installation
To set up this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pranaysood/Root-Cause-Analysis.git
   cd Root-Cause-Analysis

2. **Install required libraries:**
   ```
   #Ensure Python is installed on your system, and then run:
   pip install -r requirements.txt
   
## Data Setup:
Download the PetShop Dataset from this link: 
https://github.com/amazon-science/petshop-root-cause-analysis 
and place it in the dataset/ directory.


### 3. **Usage**

This section will detail how to run the scripts or use the models. We need specifics on what commands to run, what each script does, or how to interact with the application if it's software with a UI.

```markdown
## Usage
To analyze microservice issues using our models, follow these steps:

1. Prepare the data:
   Run the data preprocessing script to format the PetShop dataset for analysis.

   ```bash
   python preprocess_data.py

2. Run the analysis:
    To start the root cause analysis, use the following command:
    python analyze.py

3. Evaluate the models:
    Assess the model predictions against ground truth:
    python evaluate.py
```

  ## Features
  - Data-Driven Insights: Utilizes the PetShop Dataset to simulate real-world microservice performance issues, offering a robust platform for testing root cause analysis methods.
  - Dual Model Analysis: Employs both traditional models and large language models (LLMs) to explore different approaches in identifying the root causes of service disruptions.
  - Zero-Shot Learning: Implements zero-shot learning capabilities to evaluate model performance without prior specific training on the dataset.
  - Evaluation Metrics: Includes a comprehensive set of evaluation metrics to assess the accuracy and effectiveness of the models.

## Contributing
Contributions are welcome, and we value your input! Here are some ways you can contribute:
- **Issues:** Feel free to post issues on GitHub if you find bugs or have suggestions.
- **Pull Requests:** We accept pull requests. Please fork the repository, make your changes, and submit a pull request for review.
- **Feedback:** Provide feedback on the usage of the models and suggest improvements.

## Credits
- Dr. Yan Liu: Project Supervisor
- Gayathiri Elambooranan and Pranay Sood: Researchers and Developers
- The PetShop Dataset Authors: For providing a dataset specifically designed to aid in root cause analysis in microservices environments.

Special thanks to everyone who contributed to this project, providing insights, reviews, and testing.





