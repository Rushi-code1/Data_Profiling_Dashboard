import pandas as pd
from EDA import EDAProcessor

def main():
    # Load data
    df = pd.read_csv('Carbon_(CO2)_Emissions_by_Country.csv')
    
    # Specify the columns dynamically
    target_column = 'Kilotons of Co2'  # Replace with your actual target column name
    feature_columns = df.drop(columns=target_column).columns  # Specify columns to analyze dynamically
    # Initialize the EDAProcessor
    eda_processor = EDAProcessor(data=df, target_col=target_column)

    # Perform EDA tasks
    eda_processor.data_info()
    eda_processor.clean_data()
    eda_processor.summary_statistics(columns=feature_columns)
    eda_processor.correlation_matrix(columns=feature_columns)
    eda_processor.visualize_all_features(columns=feature_columns)
    eda_processor.missing_data_pattern()
    eda_processor.detect_outliers(columns=feature_columns)
    df = eda_processor.encode_categorical(columns=['Country', 'Region'], encoding_type='label')
    eda_processor.scale_features(columns=feature_columns, method='standard')
    eda_processor.feature_importance(target_col=target_column)

if __name__ == "__main__":
    main()
