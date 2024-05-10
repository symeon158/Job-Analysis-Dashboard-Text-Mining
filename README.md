# Job Analysis Dashboard

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Upload Instructions](#file-upload-instructions)
6. [Visualization Details](#visualization-details)
7. [Text Mining Details](#text-mining-details)
8. [Contributing](#contributing)
9. [License](#license)
10. [Full Code](#full-code)

## Introduction
The Job Analysis Dashboard is a comprehensive tool designed to analyze and visualize job-related data from Excel files. It provides insights into job titles, departments, divisions, and main responsibilities.

## Features
- **File Upload**: Upload Excel files for analysis.
- **Data Filtering**: Initial and advanced filtering options.
- **Visualizations**:
  - Job Titles Count by Division
  - Job Titles with Hours >< 176
  - Distinct Job Titles Count by Division/Department
  - Department and Job Title Counts by Country
  - Main Responsibility Counts
- **Text Mining**:
  - TF-IDF Analysis
  - Word Frequency Analysis

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/job-analysis-dashboard.git

Usage
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Upload your Excel file using the sidebar file uploader.
Use the checkboxes and filters to explore the visualizations and text mining features.
File Upload Instructions
Ensure your Excel file has the required columns such as DIVISION:, DEPARTMENT, JOB TITLE:, A/A, Sum Total hours per month, Country, Main Responsibility , and Category.
Visualization Details
Job Titles Count by Division: Displays the count of distinct job titles within each division.
Job Titles with Hours >< 176: Shows job titles with monthly hours greater or less than 176.
Distinct Job Titles Count by Division/Department: Visualizes the count of distinct job titles by either division or department.
Department and Job Title Counts by Country: Displays the count of distinct departments and job titles across different countries.
Main Responsibility Counts: Shows counts of main responsibilities by department and job title.
Text Mining Details
TF-IDF Analysis:
Lemmatizes text from the 'Main Responsibility' column.
Displays top TF-IDF scores and keywords.
Word Frequency Analysis:
Counts and visualizes the most frequent words in the 'Main Responsibility' column, excluding custom stop words.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch:
bash
Copy code
git checkout -b feature-branch
Make your changes and commit them:
bash
Copy code
git commit -m "Add new feature"
Push to the branch:
bash
Copy code
git push origin feature-branch
Create a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Full Code
python
Copy code
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import plotly.graph_objects as go
import spacy
import re

def custom_tokenizer(text):
    # Tokenize by whitespace
    tokens = text.split()
    # Filter out tokens that are entirely numeric
    tokens = [token for token in tokens if not token.isdigit() and not re.fullmatch(r





