import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats
import missingno as msno

class EDAProcessor:
    def __init__(self, data: pd.DataFrame, target_col: str = None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        self.df = data
        self.target = target_col
        
        if self.target:
            if self.target not in self.df.columns:
                raise ValueError(f"Target column '{self.target}' not found in the DataFrame.")
            print(f"Target column set to: {self.target}")
        else:
            print("No target column specified.")
        
    def data_info(self):
        """Displays basic information about the dataframe: shape, column types, and memory usage."""
        try:
            print(f"Shape of the data: {self.df.shape}")
            print(f"Columns and data types:\n{self.df.dtypes}")
            print(f"Memory usage: {self.df.memory_usage(deep=True)}")
            return self.df.info()
        except Exception as e:
            print(f"Error in data_info: {e}")
    
    def clean_data(self):
        """Cleans the data by removing rows with missing values and duplicates."""
        try:
            print("Cleaning data...")
            self.df = self.df.dropna()
            self.df = self.df.drop_duplicates()
            print("Data cleaned successfully.")
            return self.df
        except Exception as e:
            print(f"Error in clean_data: {e}")
    
    def summary_statistics(self, columns=None):
        """Displays summary statistics for selected columns or the entire dataframe."""
        try:
            if columns is not None:
                # Validate if all columns exist in the DataFrame
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
                data_to_describe = self.df[columns]
            else:
                data_to_describe = self.df
            
            print(f"Numerical summary:\n{data_to_describe.describe()}")
            print(f"Categorical summary:\n{data_to_describe.describe(include=['object'])}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in summary_statistics: {e}")

        
    def correlation_matrix(self, columns=None):
        """Displays a heatmap of the correlation matrix for numeric columns."""
        try:
            if columns is not None:
                # Validate if all columns exist in the DataFrame
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
                corr_data = self.df[columns].select_dtypes(include=[np.number])
            else:
                corr_data = self.df.select_dtypes(include=[np.number])
            
            if corr_data.empty:
                raise ValueError("No numeric columns found to calculate correlation.")
            
            corr = corr_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in correlation_matrix: {e}")
            corr = corr_data.corr()
            plt.figure(figsize=(10,8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()
        except Exception as e:
                print(f"Error in correlation_matrix: {e}")
        
    def visualize_all_features(self, columns=None):
        """Visualizes all selected features (univariate and bivariate)."""
        try:
            print("Visualizing all selected features...")
            if columns is None:
                columns = self.df.columns

            continuous_columns = self.df[columns].select_dtypes(include=[np.number]).columns
            categorical_columns = self.df[columns].select_dtypes(include=['object']).columns

            for col in continuous_columns:
                self.univariate_continuous(col)

            for col in categorical_columns:
                self.univariate_categorical(col)
            
            for col1 in continuous_columns:
                for col2 in continuous_columns:
                    if col1 != col2:
                        self.bivariate_continuous(col1, col2)

            for col1 in continuous_columns:
                for col2 in categorical_columns:
                    self.bivariate_continuous_categorical(col1, col2)
                
        except Exception as e:
            print(f"Error in visualize_all_features: {e}")

    def univariate_continuous(self, col):
        """Visualizes distribution and boxplot for continuous columns."""
        try:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")
            
            plt.figure(figsize=(12,6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
            
            plt.figure(figsize=(12,6))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in univariate_continuous: {e}")

    def univariate_categorical(self, col):
        """Visualizes countplot for categorical columns."""
        try:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")
            
            plt.figure(figsize=(12,6))
            sns.countplot(x=self.df[col])
            plt.title(f'Countplot of {col}')
            plt.show()
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in univariate_categorical: {e}")

    def bivariate_continuous(self, col1, col2):
        """Visualizes scatterplot between two continuous columns."""
        try:
            if col1 not in self.df.columns or col2 not in self.df.columns:
                raise ValueError(f"Columns '{col1}' or '{col2}' not found in the DataFrame.")
            
            plt.figure(figsize=(12,6))
            sns.scatterplot(x=self.df[col1], y=self.df[col2])
            plt.title(f'Scatterplot between {col1} and {col2}')
            plt.show()
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in bivariate_continuous: {e}")

    def bivariate_continuous_categorical(self, cont_col, cat_col):
        """Visualizes boxplot of continuous vs categorical columns."""
        try:
            if cont_col not in self.df.columns or cat_col not in self.df.columns:
                raise ValueError(f"Columns '{cont_col}' or '{cat_col}' not found in the DataFrame.")
            
            plt.figure(figsize=(12,6))
            sns.boxplot(x=self.df[cat_col], y=self.df[cont_col])
            plt.title(f'{cont_col} by {cat_col}')
            plt.show()
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in bivariate_continuous_categorical: {e}")

    def missing_data_pattern(self):
        """Visualizes the missing data pattern in the dataframe."""
        try:
            msno.matrix(self.df)
            plt.show()
        except Exception as e:
            print(f"Error in missing_data_pattern: {e}")

    def detect_outliers(self, columns=None):
        """Detects outliers in the continuous columns."""
        try:
            # Use all numeric columns if no specific columns are provided
            if columns is None:
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            else:
                # Validate if specified columns exist
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
                
                # Filter numeric columns from the specified list
                numeric_columns = [col for col in columns if col in self.df.select_dtypes(include=[np.number]).columns]
            
            if not numeric_columns:
                raise ValueError("No numeric columns found for outlier detection.")
            
            # Detect outliers for each numeric column
            for col in numeric_columns:
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers = self.df[col][z_scores > 3]
                print(f"Outliers detected in {col}:\n{outliers}")
                return outliers
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in detect_outliers: {e}")


    def scale_features(self, columns=None, method='standard'):
        """Scales features using StandardScaler or MinMaxScaler."""
        try:
            if columns is None:
                # Select all numeric columns by default
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            else:
                # Validate if all specified columns exist
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
                
                # Filter only numeric columns
                numeric_columns = [col for col in columns if col in self.df.select_dtypes(include=[np.number]).columns]

            if not numeric_columns:
                raise ValueError("No numeric columns found to scale.")
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be either 'standard' or 'minmax'.")
            
            # Scale only the numeric columns
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
            print(f"Features {numeric_columns} scaled using {method} scaling.")
            return self.df
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in scale_features: {e}")

    
    def feature_importance(self, target_col):
        """Displays feature importance using Random Forest."""
        try:
            if target_col not in self.df.columns:
                raise ValueError(f"Target column '{target_col}' not found.")
            X = self.df.drop(columns=[target_col,"Date"])
            y = self.df[target_col]
            
            if y.nunique() > 10:  # Continuous target (regression)
                model = RandomForestRegressor()
            else:  # Categorical target (classification)
                model = RandomForestClassifier()

            model.fit(X, y)
            importance = model.feature_importances_
            feature_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            print(f"Feature importance:\n{feature_imp}")
            return feature_imp
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in feature_importance: {e}")
            
            
    def encode_categorical(self, columns=None, encoding_type='label'):
        """Encodes categorical variables into numeric."""
        try:
            if columns is None:
                # Select all categorical columns by default
                cat_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            else:
                # Validate if all specified columns exist in the DataFrame
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
                
                # Filter only categorical columns from the specified list
                cat_columns = [col for col in columns if col in self.df.select_dtypes(include=['object']).columns]

            if not cat_columns:  # Check if the list is empty
                print("No categorical columns found for encoding.")
                return self.df
            
            if encoding_type == 'label':
                # Label encoding
                for col in cat_columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    print(f"Column '{col}' encoded using Label Encoding.")
            elif encoding_type == 'onehot':
                # One-hot encoding
                self.df = pd.get_dummies(self.df, columns=cat_columns, drop_first=True)
                print(f"Columns {cat_columns} encoded using One-Hot Encoding.")
            else:
                raise ValueError("Unsupported encoding_type. Use 'label' or 'onehot'.")

            return self.df
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error in encode_categorical: {e}")
