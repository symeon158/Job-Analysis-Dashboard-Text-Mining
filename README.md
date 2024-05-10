# Job Analysis Dashboard

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
## Text Mining Details
TF-IDF Analysis:
Lemmatizes text from the 'Main Responsibility' column.
Displays top TF-IDF scores and keywords.
Word Frequency Analysis:
Counts and visualizes the most frequent words in the 'Main Responsibility' column, excluding custom stop words.

## Full Code
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
    tokens = [token for token in tokens if not token.isdigit() and not re.fullmatch(r'\d+', token)]
    # Re-join tokens into a string for SpaCy processing
    filtered_text = " ".join(tokens)
    return filtered_text


@ st.cache_resource
def load_spacy_model(name):
    return spacy.load(name)

nlp = load_spacy_model("en_core_web_sm")



def preprocess_text(text):
    # First, apply the custom tokenization and filtering
    filtered_text = custom_tokenizer(text)
    # Then, process the filtered text with SpaCy for lemmatization
    doc = nlp(filtered_text)
    # Extract lemmas as a single string
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def main():
    st.title('Job Analysis Dashboard')

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        # Display the column names for verification
        st.write("Column names in the uploaded file:", df.columns.tolist())

        # Create tabs
        tab1, tab2 = st.tabs(["Visualizations", "Text Mining"])
        # Tab 1: Visualizations
        with tab1:
            # Initialize filter variables
            filtered_df = df
            column_names = df.columns.tolist()

            # Initial filtering options in the sidebar
            if st.sidebar.checkbox('Show Initial Filter Options'):
                selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
                unique_values = df[selected_column].unique()
                selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
                
                # Filter the dataframe based on selected value and column
                filtered_df = df[df[selected_column] == selected_value]

            # Display the filtered dataframe within an expander
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_df)
            # Visualization Section
            if st.checkbox('Show Job Titles Count'):
                # Filter the dataset where 'A/A' equals 1
                job_titles_df = filtered_df[filtered_df['A/A'] == 1]
                # Now, group by DIVISION: and DEPARTMENT to count distinct JOB TITLE:
                job_titles = job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])['JOB TITLE:'].count().reset_index(name='Distinct Job Titles Count')
                #job_titles = job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT'])['JOB TITLE:'].nunique().reset_index(name='Distinct Job Titles Count')
                #fig = px.bar(job_titles, x='JOB TITLE:', y='Distinct Job Titles Count', color='JOB TITLE:', title='Distinct Job Titles Count by Department and Division')
                fig = px.bar(job_titles, x='JOB TITLE:', y='Distinct Job Titles Count', color='DIVISION: ', title='Job Titles Count by Division')

                st.plotly_chart(fig)
                st.dataframe(job_titles)

            # Visualization Section
            if st.checkbox('Show Job Titles><176'):
                # Filter the dataset where 'A/A' equals 1
                #job_titles_hours = filtered_df[(filtered_df['Sum Total hours per month'] > 176) & (filtered_df['A/A'] == 1) & (filtered_df['Sum Total hours per month'] < 176)]
                # Assuming 'df' is your original DataFrame
                filtered_df_1 = filtered_df[filtered_df['A/A'] == 1]
                job_titles_hours = filtered_df_1[(filtered_df_1['Sum Total hours per month'] > 176) | (filtered_df_1['Sum Total hours per month'] < 176)]
                # Now, group by DIVISION: and DEPARTMENT to count distinct JOB TITLE:
                job_titles_176 = job_titles_hours.groupby(['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])['JOB TITLE:'].count().reset_index(name='Job Titles Count Hours')
                #job_titles = job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT'])['JOB TITLE:'].nunique().reset_index(name='Distinct Job Titles Count')
                #fig = px.bar(job_titles, x='JOB TITLE:', y='Distinct Job Titles Count', color='JOB TITLE:', title='Distinct Job Titles Count by Department and Division')
                fig = px.bar(job_titles_176, x='JOB TITLE:', y='Job Titles Count Hours', color='DIVISION: ', title='Job Titles Count ><176')

                st.plotly_chart(fig)
                st.dataframe(job_titles_176)
                

            
            if st.checkbox('Show Visualization'):
                # Dropping duplicates to ensure distinct JOB TITLE: counts within each DIVISION: and DEPARTMENT
                unique_job_titles_df = filtered_df.drop_duplicates(subset=['DIVISION: ', 'DEPARTMENT', 'JOB TITLE:'])
                
                # Now, group by DIVISION: and DEPARTMENT to count distinct JOB TITLE:
                job_title_counts = unique_job_titles_df.groupby(['DIVISION: ', 'DEPARTMENT'])['JOB TITLE:'].count().reset_index(name='Distinct Job Titles Count')
                
                # Choosing what to visualize: Division or Department with distinct counts of Job Titles
                visualization_column = st.selectbox('Visualize by', ['DIVISION: ', 'DEPARTMENT'])
                
                # Depending on the choice, aggregate data to visualize
                if visualization_column == 'DIVISION: ':
                    fig_data = job_title_counts.groupby('DIVISION: ')['Distinct Job Titles Count'].sum().reset_index()
                else:  # 'DEPARTMENT'
                    fig_data = job_title_counts.groupby('DEPARTMENT')['Distinct Job Titles Count'].sum().reset_index()
                
                # Creating a bar chart with Plotly Express
                fig = px.bar(fig_data, x=visualization_column, y='Distinct Job Titles Count', title=f'Distinct Job Titles Count by {visualization_column}')
                st.plotly_chart(fig)

                # Dropping duplicates to find distinct combinations of Country, DEPARTMENT, and JOB TITLE:
            distinct_df = df.drop_duplicates(subset=['Country', 'DEPARTMENT', 'JOB TITLE:'])

            # Now, for visualization, we need to count these distinct combinations by Country
            department_counts = distinct_df.groupby('Country')['DEPARTMENT'].nunique().reset_index(name='Distinct Departments')
            job_title_counts = distinct_df.groupby('Country')['JOB TITLE:'].nunique().reset_index(name='Distinct Job Titles')

            # Merging the counts back for a comprehensive dataframe
            merged_counts = department_counts.merge(job_title_counts, on='Country')

            # Visualization
            if st.checkbox('Show Department Counts Visualization'):
                fig_dept = px.bar(merged_counts, x='Country', y='Distinct Departments', title='Distinct Department Counts by Country')
                st.plotly_chart(fig_dept)

            if st.checkbox('Show Job Title Counts Visualization'):
                fig_title = px.bar(merged_counts, x='Country', y='Distinct Job Titles', title='Distinct Job Title Counts by Country')
                st.plotly_chart(fig_title)

            # Group by DEPARTMENT and JOB TITLE: and count 'Main Responsibility'
            if 'Main Responsibility ' in filtered_df.columns and 'DEPARTMENT' in filtered_df.columns and 'JOB TITLE:' in filtered_df.columns:
                # Group and count
                # Filter rows where 'main_responsibility' is not blank (non-NaN in this context)
                df_non_blank_responsibility = filtered_df[filtered_df['Main Responsibility '].notna()]

                # Group by 'country', 'division:', 'department', 'job_title:' and find the maximum 'a/a' for each group
                counts = df_non_blank_responsibility.groupby(['DIVISION: ','DEPARTMENT', 'JOB TITLE:'])["A/A"].max().reset_index().rename(columns={'A/A': 'Count'})
                st.dataframe(counts)
                # counts = filtered_df.groupby(['DIVISION: ','DEPARTMENT', 'JOB TITLE:'])["A/A"].count().reset_index(name='Count')
                # st.dataframe(counts)
                # Filter to keep only counts under 8 or above 8
                filtered_counts = counts[(counts['Count'] < 8) | (counts['Count'] > 8)]

                # Apply conditional formatting
                filtered_counts['Color'] = filtered_counts['Count'].apply(lambda x: 'red' if x < 8 else 'blue')

                # Plot with specified width
                fig = go.Figure()
                for _, row in filtered_counts.iterrows():
                    fig.add_trace(go.Bar(x=[f"{row['DEPARTMENT']} - {row['JOB TITLE:']}"], y=[row['Count']],
                                            name=f"{row['DEPARTMENT']} - {row['JOB TITLE:']}",
                                            marker_color=row['Color']))

                fig.update_layout(title_text='Main Responsibility Counts by Department and Job Title (Filtered)',
                                    xaxis_title="Department - Job Title",
                                    yaxis_title="Count of Main Responsibilities",
                                    xaxis={'categoryorder':'total descending'},
                                    width=800,  # Set the width to 1200 pixels
                                    showlegend=False)

                st.plotly_chart(fig)

        with tab2:
            st.header("Text Mining")
            # Display some text mining results, e.g., TF-IDF scores
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_df)
            st.write("TF-IDF Scores:")
            # Extend the stop words list with your specific words
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ['&', 'on', 'alumil', 'new', 'daily', 'monthly', '/', 'albania', 'weekly', 'annual', 'aluminium', 'kosovo','yearly','aluminum','kosovo.', 'controll','']

            # Advanced filtering options in the sidebar for Category and JOB TITLE:
            if st.sidebar.checkbox('Show Advanced Filter Options'):
                # Filtering by Category
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

                # Filtering by JOB TITLE:
                if 'JOB TITLE:' in column_names:
                    job_titles = df['JOB TITLE:'].unique()
                    selected_job_title = st.sidebar.selectbox('Filter by JOB TITLE:', ['All'] + list(job_titles))
                    if selected_job_title != 'All':
                        df = df[df['JOB TITLE:'] == selected_job_title]

            # TF-IDF Analysis for "Main Responsibility" with Lemmatization using SpaCy
            if 'Main Responsibility ' in df.columns:
                responsibilities = df['Main Responsibility '].dropna().tolist()
                if responsibilities and st.checkbox('Perform TF-IDF Analysis on "Main Responsibility"'):
                    #Lemmatize the responsibilities using SpaCy
                    lemmatized_responsibilities = [" ".join([token.lemma_ for token in nlp(custom_tokenizer(text))]) for text in responsibilities]
                    #preprocessed_texts = [preprocess_text(text) for text in responsibilities]
                    # TF-IDF Vectorizer
                    # vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                    # tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
                    # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=10)
                    tfidf_matrix = vectorizer.fit_transform(lemmatized_responsibilities)
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

                    # Display the DataFrame with TF-IDF scores
                    st.write("TF-IDF Scores for 'Main Responsibility'")
                    st.dataframe(tfidf_df.head())
                    data = tfidf_df.head().values

                    # Apply the background_gradient
                    styled_df = tfidf_df.head().style.background_gradient(cmap='viridis')

                    # Display the styled DataFrame in Streamlit
                    st.dataframe(styled_df)

                    # Display top keywords for each responsibility (optional detailed view)
                    top_keywords = tfidf_df.apply(lambda s: s.abs().nlargest(3).index.tolist(), axis=1)
                    st.write("Top Keywords for Each Responsibility (Sample)")
                    st.dataframe(top_keywords.head())

            # Word Frequency Analysis with extended stop words
            if st.checkbox('Perform Word Frequency Analysis on "Main Responsibility"'):
                responsibilities = df['Main Responsibility '].dropna().tolist()
                all_words = ' '.join(responsibilities).lower().split()
                # Filter out the custom stop words
                filtered_words = [word for word in all_words if word not in custom_stop_words]
                word_counts = Counter(filtered_words)
                most_common_words = word_counts.most_common(10)  # Adjust as needed
                words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

                # Display the most common words excluding the custom stop words
                st.write("Most Frequent Words in 'Main Responsibility' (excluding common stop words)")
                st.dataframe(words_df)

                # Visualization of word frequencies
                fig = px.bar(words_df, x='Word', y='Frequency', title="Top 10 Most Frequent Words (excluding common stop words)")
                st.plotly_chart(fig)

            #else:
                #st.error("The 'Main Responsibility' column was not found in the uploaded file.")

            


            else:
                st.warning("Check the box to see the plot!")
    else:
        st.error("Upload the file!")
        #st.error("The 'Country' column was not found in the uploaded file.")


if __name__ == '__main__':
    main()



