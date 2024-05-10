Job-Analysis-Dashboard-Text-Mining
Job Analysis Dashboard: Enhancing Workforce Insights with Streamlit, Plotly, and SpaCy (Text Mining)

This repository contains a Streamlit application designed to perform data visualization and text mining on job-related data. The dashboard allows users to upload an Excel file, filter and visualize the data, and analyze text in job responsibilities.

📋 Table of Contents
Features
Data Upload
Data Visualization
Text Mining
Usage
Code Overview
Main Functions
Streamlit Components
Installation
Contributing
License
Acknowledgements
✨ Features
Data Upload
📂 Upload an Excel file containing job data.
📝 Display column names for verification.
Data Visualization
🔍 Filtered Data View: Option to filter data by specific columns.
📊 Job Titles Count: Visualize the count of distinct job titles by department and division.
⏱️ Job Titles with Specific Hours: Visualize job titles with total hours per month greater than or less than 176.
📈 Distinct Job Titles by Division/Department: Bar chart visualization of distinct job titles.
🌍 Department and Job Title Counts by Country: Visualize distinct counts of departments and job titles by country.
📋 Main Responsibility Counts: Bar chart for counts of main responsibilities filtered by specific criteria.
Text Mining
🧠 TF-IDF Analysis: Calculate and display TF-IDF scores for the "Main Responsibility" column.
🔤 Word Frequency Analysis: Analyze and visualize the frequency of words in the "Main Responsibility" column, excluding custom stop words.
🚀 Usage
Run the Streamlit application:
bash
Copy code
streamlit run app.py
Upload an Excel file using the sidebar file uploader.
Explore the data using the provided tabs:
Visualizations: Filter and visualize job data.
Text Mining: Perform TF-IDF analysis and word frequency analysis on job responsibilities.
🛠️ Code Overview
Main Functions
custom_tokenizer(text): Tokenizes and filters out numeric tokens.
load_spacy_model(name): Loads the SpaCy model.
preprocess_text(text): Preprocesses text using custom tokenization and SpaCy lemmatization.
Streamlit Components
Sidebar: File uploader and initial filter options.
Tabs: Separate sections for visualizations and text mining.
Check Boxes: Toggle visibility of various visualizations and analyses.
Expanders: Expandable sections to view filtered data.
📝 Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/job-analysis-dashboard.git
Navigate to the project directory:
bash
Copy code
cd job-analysis-dashboard
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Download the SpaCy model:
bash
Copy code
python -m spacy download en_core_web_sm
🤝 Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a Pull Request.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙌 Acknowledgements
Streamlit
Plotly
SpaCy
