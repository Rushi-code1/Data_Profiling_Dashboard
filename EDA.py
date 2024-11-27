import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from fpdf import FPDF
import os
import streamlit as st

def load_data(file_path):
    """
    Load dataset from various file formats and convert to pandas DataFrame.
    """
    try:
        # Get the file extension from the uploaded file's name
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            return pd.read_json(uploaded_file)
        elif file_extension == 'txt':
            return pd.read_csv(uploaded_file, delimiter="\t")
        else:
            raise ValueError("Unsupported file format! Please provide a CSV, Excel, JSON, or TXT file.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def data_understanding(df):
    """
    Perform basic understanding of the dataset.
    """
    st.subheader("Data Understanding")
    st.write(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("\nColumn Data Types:")
    st.write(df.dtypes)
    st.write("\nDataset Preview (First 5 Rows):")
    st.write(df.head())
    
    st.write("\nDataset Statistics (Numerical Columns):")
    st.write(df.describe())

def handle_missing_values(df):
    """
    Handle missing values by imputing with median for numeric columns and mode for categorical columns.
    """
    st.subheader("Handling Missing Values")
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values for numeric columns (impute with median)
    for col in numeric_cols:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        st.write(f"Imputed missing values in numeric column '{col}' with median: {median_value}")
    
    # Handle missing values for categorical columns (impute with mode)
    for col in categorical_cols:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        st.write(f"Imputed missing values in categorical column '{col}' with mode: {mode_value}")
    
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataframe.
    """
    initial_shape = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_rows = initial_shape - df.shape[0]
    st.write(f"\nRemoved {removed_rows} duplicate rows.")
    return df

def detect_outliers(df):
    """
    Detect outliers using the IQR method. Outliers are values outside the range of Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
    """
    st.subheader("Outlier Detection (Using IQR Method)")
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if outliers.shape[0] > 0:
            outlier_info[col] = outliers
            st.write(f"Outliers detected in column '{col}': {outliers.shape[0]} rows")
        else:
            st.write(f"No outliers detected in column '{col}'.")
    
    return outlier_info

def univariate_analysis(df):
    """
    Perform univariate analysis to visualize the distribution of variables.
    """
    st.subheader("Univariate Analysis")
    # Numerical feature distributions (Histograms and Boxplots)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot histogram
        axes[0].hist(df[col], bins=30, color='lightblue', edgecolor='black')
        axes[0].set_title(f'{col} - Histogram')
        
        # Plot boxplot
        sns.boxplot(x=df[col], color='lightgreen', ax=axes[1])
        axes[1].set_title(f'{col} - Boxplot')
        
        st.pyplot(fig)
    
    # Categorical feature distributions (Bar plots)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=df[col], palette='Set2', ax=ax)
        ax.set_title(f'{col} - Count Plot')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

def bivariate_analysis(df):
    """
    Perform bivariate analysis to explore relationships between two variables.
    """
    st.subheader("Bivariate Analysis")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        # Scatter plots for pairwise numeric relationships
        pairplot_fig = sns.pairplot(df[numeric_cols])
        st.pyplot(pairplot_fig)
    
    # Grouped bar plots for categorical vs. numeric relationships
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=cat_col, y=num_col, data=df, palette='Set1', ax=ax)
            ax.set_title(f'{cat_col} vs {num_col} - Grouped Bar Plot')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

def multivariate_analysis(df):
    """
    Perform multivariate analysis to explore relationships among multiple variables.
    """
    st.subheader("Multivariate Analysis")
    
    # Pair plots for multiple numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        pairplot_fig = sns.pairplot(df[numeric_cols], hue=numeric_cols[0], palette='coolwarm')
        st.pyplot(pairplot_fig)

    # Box plots grouped by categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=cat_col, y=num_col, data=df, palette='Set1', ax=ax)
            ax.set_title(f'{cat_col} vs {num_col} - Box Plot')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    # Cluster Visualization (KMeans)
    if len(numeric_cols) >= 2:
        st.write("\nPerforming KMeans Clustering (2 clusters)")
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[numeric_cols].dropna())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], hue='Cluster', data=df, palette='Set2', ax=ax)
        ax.set_title("Cluster Visualization using KMeans")
        st.pyplot(fig)

def generate_report(df, outliers, file_name="eda_report.pdf"):
    """
    Generate a PDF report with insights from the analysis, including visualizations.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Exploratory Data Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Dataset Overview
    pdf.cell(200, 10, txt="1. Dataset Overview", ln=True)
    pdf.multi_cell(0, 10, f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    pdf.multi_cell(0, 10, "Columns:\n" + str(df.columns.tolist()) + "\n")
    pdf.multi_cell(0, 10, "Basic Statistics:\n")
    pdf.multi_cell(0, 10, str(df.describe().to_string()) + "\n")

    # Outlier Information
    pdf.ln(10)
    pdf.cell(200, 10, txt="2. Outlier Detection", ln=True)
    for col, outlier_data in outliers.items():
        pdf.multi_cell(0, 10, f"Outliers detected in column '{col}' with {outlier_data.shape[0]} rows.\n")

    # Visualizations (save the figures temporarily and include in the report)
    pdf.ln(10)
    pdf.cell(200, 10, txt="3. Visualizations", ln=True)

    # Univariate Analysis Plots (Save and include images)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Plot Histogram and Boxplot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df[col].hist(bins=30, color='lightblue', edgecolor='black')
        plt.title(f'{col} - Histogram')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f'{col} - Boxplot')

        plt.tight_layout()
        img_path = f"hist_box_{col}.png"
        plt.savefig(img_path)
        plt.close()

        # Add the image to PDF
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"{col} - Histogram and Boxplot", ln=True)
        pdf.image(img_path, x=10, w=180)
        os.remove(img_path)  # Delete the image file after adding it to the PDF

    # Bivariate Analysis (Add scatter plot, correlation heatmap)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap")
        heatmap_path = "correlation_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()

        pdf.ln(5)
        pdf.cell(200, 10, txt="Correlation Heatmap", ln=True)
        pdf.image(heatmap_path, x=10, w=180)
        os.remove(heatmap_path)

    # Conclusion and Suggestions
    pdf.ln(10)
    pdf.cell(200, 10, txt="4. Conclusion and Suggestions", ln=True)
    pdf.multi_cell(0, 10, "Suggestions for Next Steps:\n - Feature Engineering\n - Handle Outliers\n - Perform Model Building\n")

    # Output the report
    pdf.output(file_name)
    st.write(f"\nPDF report generated: {file_name}")

# Streamlit interface
st.title("Exploratory Data Analysis (EDA)")

uploaded_file = st.file_uploader("Upload a Dataset", type=["csv", "xls", "xlsx", "json", "txt"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        # Data Understanding
        data_understanding(df)
        
        # Data Cleaning
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        
        # Outlier Detection
        outliers = detect_outliers(df)
        
        # Univariate and Bivariate Analysis
        univariate_analysis(df)
        bivariate_analysis(df)
        
        # Multivariate Analysis
        multivariate_analysis(df)
        
        # Generate PDF Report
        if st.button("Generate PDF Report"):
            generate_report(df, outliers)