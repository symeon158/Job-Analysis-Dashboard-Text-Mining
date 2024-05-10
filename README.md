# Job Analysis Dashboard

I developed a powerful Job Analysis Dashboard leveraging Streamlit, Plotly, and SpaCy to transform raw job data into actionable insights. This interactive tool is designed for HR professionals and analysts to seamlessly upload and visualize job data, providing a comprehensive view of job titles, departments, and responsibilities across various divisions and countries.


## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Upload Instructions](#file-upload-instructions)
6. [Visualization Details](#visualization-details)
7. [Text Mining Details](#text-mining-details)
8. [Full Code](#full-code)

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

## Usage
Run the Streamlit app:
streamlit run app.py
Upload your Excel file using the sidebar file uploader.
Use the checkboxes and filters to explore the visualizations and text mining features.
File Upload Instructions
Ensure your Excel file has the required columns such as DIVISION:, DEPARTMENT, JOB TITLE:, A/A, Sum Total hours per month, Country, Main Responsibility , and Category.
## Visualization Details
Job Titles Count by Division: Displays the count of distinct job titles within each division.
Job Titles with Hours >< 176: Shows job titles with monthly hours greater or less than 176.
Distinct Job Titles Count by Division/Department: Visualizes the count of distinct job titles by either division or department.
Department and Job Title Counts by Country: Displays the count of distinct departments and job titles across different countries.
Main Responsibility Counts: Shows counts of main responsibilities by department and job title.

![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/230e1ef2-beb4-42c4-bcb4-e58a099a9f4c)
![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/50ff96e5-bff2-483f-9064-5444fe112b94)
![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/076adcb3-c109-47d5-bb5d-4bd21232cf53)
![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/98c4cf8a-22ae-4ee8-b038-a24ae51b7637)

## Text Mining Details
TF-IDF Analysis:
Lemmatizes text from the 'Main Responsibility' column.
Displays top TF-IDF scores and keywords.
Word Frequency Analysis:
Counts and visualizes the most frequent words in the 'Main Responsibility' column, excluding custom stop words.

![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/a6deb13e-a2ee-4a57-ab48-fc2e7b09eb61)
![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/b472b594-fa21-4785-8bd7-5a17c575e28f)
![image](https://github.com/symeon158/Job-Analysis-Dashboard-Text-Mining/assets/106148298/e309250f-04b5-4ba9-bcc5-f39da6c0687e)

## Full Code
```python
# Import necessary libraries for Streamlit, data handling, and visualizations
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import plotly.graph_objects as go
import spacy
import re

# Custom tokenizer function to filter out digits from text
def custom_tokenizer(text):
    tokens = text.split()
    tokens = [token for token in tokens if not token.isdigit() and not re.fullmatch(r'\d+', token)]
    filtered_text = " ".join(tokens)
    return filtered_text

# Function to load the spaCy language model with caching to improve performance
@st.cache_resource
def load_spacy_model(name):
    return spacy.load(name)

# Load the English language model from spaCy
nlp = load_spacy_model("en_core_web_sm")

# Function to preprocess text: tokenization and lemmatization
def preprocess_text(text):
    filtered_text = custom_tokenizer(text)
    doc = nlp(filtered_text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

# Main function that defines the Streamlit app
def main():
    st.title('Job Analysis Dashboard')

    # File uploader widget in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Column names in the uploaded file:", df.columns.tolist())

        # Create tabs for visualizations and text mining
        tab1, tab2 = st.tabs(["Visualizations", "Text Mining"])
        with tab1:
            filtered_df = df
            column_names = df.columns.tolist()

            # Sidebar checkbox for initial filter options
            if st.sidebar.checkbox('Show Initial Filter Options'):
                selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
                unique_values = df[selected_column].unique()
                selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
                filtered_df = df[df[selected_column] == selected_value]

            # Expandable section to view filtered data
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_df)

            # Checkbox to show job titles count
            if st.checkbox('Show Job Titles Count'):
                job_titles_df = filtered_df[filtered_df['A/A'] == 1]
                job_titles = job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])['JOB TITLE:'].count().reset_index(name='Distinct Job Titles Count')
                fig = px.bar(job_titles, x='JOB TITLE:', y='Distinct Job Titles Count', color='DIVISION: ', title='Job Titles Count by Division')
                st.plotly_chart(fig)
                st.dataframe(job_titles)

            # Checkbox to show job titles with hours > or < 176
            if st.checkbox('Show Job Titles><176'):
                filtered_df_1 = filtered_df[filtered_df['A/A'] == 1]
                job_titles_hours = filtered_df_1[(filtered_df_1['Sum Total hours per month'] > 176) | (filtered_df_1['Sum Total hours per month'] < 176)]
                job_titles_176 = job_titles_hours.groupby(['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])['JOB TITLE:'].count().reset_index(name='Job Titles Count Hours')
                fig = px.bar(job_titles_176, x='JOB TITLE:', y='Job Titles Count Hours', color='DIVISION: ', title='Job Titles Count ><176')
                st.plotly_chart(fig)
                st.dataframe(job_titles_176)

            # Checkbox to show general visualization of job titles count by division or department
            if st.checkbox('Show Visualization'):
                unique_job_titles_df = filtered_df.drop_duplicates(subset=['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])
                job_title_counts = unique_job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT'])['JOB TITLE:'].count().reset_index(name='Distinct Job Titles Count')
                visualization_column = st.selectbox('Visualize by', ['DIVISION: ', 'DEPARTMENT'])

                if visualization_column == 'DIVISION: ':
                    fig_data = job_title_counts.groupby('DIVISION: ')['Distinct Job Titles Count'].sum().reset_index()
                else:
                    fig_data = job_title_counts.groupby('DEPARTMENT')['Distinct Job Titles Count'].sum().reset_index()

                fig = px.bar(fig_data, x=visualization_column, y='Distinct Job Titles Count', title=f'Distinct Job Titles Count by {visualization_column}')
                st.plotly_chart(fig)

            # Visualization for distinct department and job title counts by country
            distinct_df = df.drop_duplicates(subset=['Country', 'DEPARTMENT', 'JOB TITLE:'])
            department_counts = distinct_df.groupby('Country')['DEPARTMENT'].nunique().reset_index(name='Distinct Departments')
            job_title_counts = distinct_df.groupby('Country')['JOB TITLE:'].nunique().reset_index(name='Distinct Job Titles')
            merged_counts = department_counts.merge(job_title_counts, on='Country')

            if st.checkbox('Show Department Counts Visualization'):
                fig_dept = px.bar(merged_counts, x='Country', y='Distinct Departments', title='Distinct Department Counts by Country')
                st.plotly_chart(fig_dept)

            if st.checkbox('Show Job Title Counts Visualization'):
                fig_title = px.bar(merged_counts, x='Country', y='Distinct Job Titles', title='Distinct Job Title Counts by Country')
                st.plotly_chart(fig_title)

            # Visualization for main responsibility counts by department and job title
            if 'Main Responsibility ' in filtered_df.columns and 'DEPARTMENT' in filtered_df.columns and 'JOB TITLE:' in filtered_df.columns:
                df_non_blank_responsibility = filtered_df[filtered_df['Main Responsibility '].notna()]
                counts = df_non_blank_responsibility.groupby(['DIVISION: ','DEPARTMENT', 'JOB TITLE:'])["A/A"].max().reset_index().rename(columns={'A/A': 'Count'})
                st.dataframe(counts)
                filtered_counts = counts[(counts['Count'] < 8) | (counts['Count'] > 8)]
                filtered_counts['Color'] = filtered_counts['Count'].apply(lambda x: 'red' if x < 8 else 'blue')

                fig = go.Figure()
                for _, row in filtered_counts.iterrows():
                    fig.add_trace(go.Bar(x=[f"{row['DEPARTMENT']} - {row['JOB TITLE:']}"], y=[row['Count']],
                                        name=f"{row['DEPARTMENT']} - {row['JOB TITLE:']}",
                                        marker_color=row['Color']))

                fig.update_layout(title_text='Main Responsibility Counts by Department and Job Title (Filtered)',
                                  xaxis_title="Department - Job Title",
                                  yaxis_title="Count of Main Responsibilities",
                                  xaxis={'categoryorder':'total descending'},
                                  width=800,
                                  showlegend=False)
                st.plotly_chart(fig)

        # Text Mining tab for TF-IDF and word frequency analysis
        with tab2:
            st.header("Text Mining")
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_df)
            st.write("TF-IDF Scores:")
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ['&', 'on', 'alumil', 'new', 'daily', 'monthly', '/', 'albania', 'weekly', 'annual', 'aluminium', 'kosovo','yearly','aluminum','kosovo.', 'controll','']

            # Sidebar checkbox for advanced filter options
            if st.sidebar.checkbox('Show Advanced Filter Options'):
                if 'Category' in column_names:
                    categories = df['Category'].unique()
                    selected_category = st.sidebar.selectbox('Filter by Category', ['All'] + list(categories))
                    if selected_category != 'All':
                        df = df[df['Category'] == selected_category]

                if 'Country' in column_names:
                    countries = df['Country'].unique()
                    selected_country = st.sidebar.selectbox('Filter by Country', ['All'] + list(countries))
                    if selected_country != 'All':
                        df = df[df['Country'] == selected_country]

                if 'JOB TITLE:' in column_names:
                    job_titles = df['JOB TITLE:'].unique()
                    selected_job_title = st.sidebar.selectbox('Filter by JOB TITLE:', ['All'] + list(job_titles))
                    if selected_job_title != 'All':
                        df = df[df['JOB TITLE:'] == selected_job_title]

            # TF-IDF analysis on 'Main Responsibility' column
            if 'Main Responsibility ' in df.columns:
                responsibilities = df['Main Responsibility '].dropna().tolist()
                if responsibilities and st.checkbox('Perform TF-IDF Analysis on "Main Responsibility"'):
                    lemmatized_responsibilities = [" ".join([token.lemma_ for token in nlp(custom_tokenizer(text))]) for text in responsibilities]
                    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=10)
                    tfidf_matrix = vectorizer.fit_transform(lemmatized_responsibilities)
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

                    st.write("TF-IDF Scores for 'Main Responsibility'")
                    st.dataframe(tfidf_df.head())
                    data = tfidf_df.head().values

                    styled_df = tfidf_df.head().style.background_gradient(cmap='viridis')
                    st.dataframe(styled_df)

                    top_keywords = tfidf_df.apply(lambda s: s.abs().nlargest(3).index.tolist(), axis=1)
                    st.write("Top Keywords for Each Responsibility (Sample)")
                    st.dataframe(top_keywords.head())

            # Word frequency analysis on 'Main Responsibility' column
            if st.checkbox('Perform Word Frequency Analysis on "Main Responsibility"'):
                responsibilities = df['Main Responsibility '].dropna().tolist()
                all_words = ' '.join(responsibilities).lower().split()
                filtered_words = [word for word in all_words if word not in custom_stop_words]
                word_counts = Counter(filtered_words)
                most_common_words = word_counts.most_common(10)
                words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

                st.write("Most Frequent Words in 'Main Responsibility' (excluding common stop words)")
                st.dataframe(words_df)

                fig = px.bar(words_df, x='Word', y='Frequency', title="Top 10 Most Frequent Words (excluding common stop words)")
                st.plotly_chart(fig)

    else:
        st.error("Upload the file!")

# Execute the main function when the script is run
if __name__ == '__main__':
    main()

  
      else:
          st.error("Upload the file!")
  
  
  
  if __name__ == '__main__':
      main()



