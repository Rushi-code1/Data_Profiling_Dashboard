import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
from scipy import stats
import io

# Set Streamlit page configuration with a custom theme
st.set_page_config(
    page_title="📊 Automated Data Profiling Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set global Seaborn style for consistency
sns.set_style("whitegrid")

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["figure.figsize"] = (8, 6)

# Initialize session state for report content
if "report_sections" not in st.session_state:
    st.session_state["report_sections"] = []

# Helper function to safely convert DataFrame to string by escaping backslashes
def safe_to_string(df):
    return df.to_string().replace("\\", "\\\\")

# Function to load data
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Function to generate automated insights
def generate_insights():
    insights = []
    # Example insights based on collected report sections
    # You can enhance this with more sophisticated rules or even integrate NLP models
    # For simplicity, we'll use some basic rules here

    # Extract top correlations
    report_sections = st.session_state["report_sections"]
    top_corr = None
    for section in report_sections:
        if "Top Correlations with" in section:
            lines = section.split("\n")
            for line in lines:
                if "Correlation" in line and ":" in line:
                    # Assuming the format is "Feature: Correlation"
                    parts = line.split(":")
                    if len(parts) == 2:
                        feature = parts[0].strip().strip("*")
                        try:
                            corr_value = float(parts[1].strip())
                            if top_corr is None or abs(corr_value) > abs(top_corr[1]):
                                top_corr = (feature, corr_value)
                        except ValueError:
                            continue  # Skip lines that don't have a valid float
    if top_corr:
        feature, corr = top_corr
        if abs(corr) > 0.7:
            insights.append(
                f"The feature **{feature}** has a strong correlation ({corr:.2f}) with the target variable. Consider investigating this relationship further."
            )
        elif abs(corr) > 0.5:
            insights.append(
                f"The feature **{feature}** has a moderate correlation ({corr:.2f}) with the target variable."
            )
        else:
            insights.append(
                f"The feature **{feature}** has a weak correlation ({corr:.2f}) with the target variable."
            )

    # Check for missing values
    for section in report_sections:
        if "**Missing Values Handling:**" in section:
            if "Dropped rows with missing values." in section:
                insights.append(
                    "Rows with missing values were dropped to ensure data quality."
                )
            elif "Filled missing values with mean." in section:
                insights.append(
                    "Missing numerical values were filled with the mean to maintain data integrity."
                )
            elif "Filled missing values with median." in section:
                insights.append(
                    "Missing numerical values were filled with the median to maintain data integrity."
                )
            elif "Filled missing values with mode." in section:
                insights.append(
                    "Missing categorical values were filled with the mode to maintain data integrity."
                )

    # Check for multicollinearity
    for section in report_sections:
        if "**Variance Inflation Factor (VIF):**" in section:
            lines = section.split("\n")
            high_vif_features = []
            for line in lines:
                if line.startswith("feature") or line.startswith("VIF"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    feature = parts[0]
                    try:
                        vif = float(parts[1])
                        if vif > 10:
                            high_vif_features.append(f"{feature} (VIF={vif:.2f})")
                    except ValueError:
                        continue  # Skip lines that don't have a valid float
            if high_vif_features:
                features_str = ", ".join(high_vif_features)
                insights.append(
                    f"The following features exhibit high multicollinearity: {features_str}. Consider removing or combining these features."
                )
            else:
                insights.append(
                    "No significant multicollinearity detected among the features."
                )

    # Check for normality of residuals
    for section in report_sections:
        if "**Shapiro-Wilk Test:**" in section:
            if "normally distributed" in section:
                insights.append(
                    "Residuals are normally distributed, validating the assumption for linear regression."
                )
            else:
                insights.append(
                    "Residuals are not normally distributed, which may affect the linear regression model's performance."
                )

    # General insights
    if not insights:
        insights.append("No significant insights were detected based on the current analysis.")

    return insights

# EDA Theme 1: Data Understanding
def data_understanding(data):
    st.subheader("1. Data Understanding")

    # Business Context (Placeholder)
    st.write("### Business Context")
    st.write(
        """
    *Provide a brief description of the business problem or objective behind this dataset.*
    *For example: Analyzing customer churn to improve retention strategies.*
    """
    )

    # Dataset Overview
    st.write("### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of Rows:** {data.shape[0]}")
    with col2:
        st.write(f"**Number of Columns:** {data.shape[1]}")

    # Data Types Identification
    st.write("### Data Types")
    st.write(safe_to_string(data.dtypes.astype(str).to_frame(name="Data Type")))

    # Initial Data Preview
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Summary Statistics
    st.write("### Summary Statistics")
    st.dataframe(data.describe(include="all").transpose())

    # Store section content for the report
    report = f"""
    ## 1. Data Understanding
    
    **Business Context:**
    Provide a brief description of the business problem or objective behind this dataset.
    
    **Dataset Overview:**
    - **Number of Rows:** {data.shape[0]}
    - **Number of Columns:** {data.shape[1]}
    
    **Data Types:**
    {safe_to_string(data.dtypes.astype(str).to_frame(name="Data Type"))}
    
    **Data Preview:**
    {safe_to_string(data.head())}
    
    **Summary Statistics:**
    {safe_to_string(data.describe(include="all").transpose())}
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 2: Data Cleaning
def data_cleaning(data):
    st.subheader("2. Data Cleaning")

    # Missing Values
    st.write("### Missing Values")
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write(safe_to_string(missing.to_frame(name="Missing Values")))
    else:
        st.write("No missing values found.")

    # Handle Missing Values
    if not missing.empty:
        st.write("### Handle Missing Values")
        fill_method = st.selectbox(
            "Select fill method:",
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
        )
        if st.button("Apply Missing Value Handling"):
            if fill_method == "Drop Rows":
                data.dropna(inplace=True)
                st.write("Dropped rows with missing values.")
                report = f"**Missing Values Handling:** Dropped rows with missing values."
            elif fill_method == "Fill with Mean":
                for col in missing.index:
                    if data[col].dtype in ["float64", "int64"]:
                        data[col].fillna(data[col].mean(), inplace=True)
                st.write("Filled missing values with mean.")
                report = f"**Missing Values Handling:** Filled missing values with mean."
            elif fill_method == "Fill with Median":
                for col in missing.index:
                    if data[col].dtype in ["float64", "int64"]:
                        data[col].fillna(data[col].median(), inplace=True)
                st.write("Filled missing values with median.")
                report = f"**Missing Values Handling:** Filled missing values with median."
            elif fill_method == "Fill with Mode":
                for col in missing.index:
                    if data[col].dtype == "object":
                        data[col].fillna(data[col].mode()[0], inplace=True)
                st.write("Filled missing values with mode.")
                report = f"**Missing Values Handling:** Filled missing values with mode."
            st.session_state["report_sections"].append(report)

    # Remove Duplicates
    st.write("### Remove Duplicates")
    if st.button("Remove Duplicates"):
        initial_shape = data.shape
        data.drop_duplicates(inplace=True)
        final_shape = data.shape
        removed = initial_shape[0] - final_shape[0]
        st.write(f"Removed {removed} duplicate rows.")
        report = f"**Duplicates Removal:** Removed {removed} duplicate rows."
        st.session_state["report_sections"].append(report)

    # Handle Outliers
    st.write("### Handle Outliers")
    if st.checkbox("Handle Outliers for Numerical Columns"):
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        outlier_info = []
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = data.shape[0]
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            after = data.shape[0]
            removed = before - after
            if removed > 0:
                outlier_info.append(f"Removed {removed} outliers from **{col}**.")
                st.write(f"Removed {removed} outliers from **{col}**.")
        if outlier_info:
            report = "**Outliers Handling:**\n" + "\n".join(outlier_info)
            st.session_state["report_sections"].append(report)

    return data

# EDA Theme 3: Univariate Analysis
def univariate_analysis(data):
    st.subheader("3. Univariate Analysis")

    # Numerical Variables
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    st.write("### Numerical Variables")
    num_summary = {}
    for col in numerical_cols:
        st.write(f"**{col}**")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax, color="#1f77b4")
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

        # Collect summary statistics
        desc = data[col].describe().to_dict()
        num_summary[col] = desc

    # Categorical Variables
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    st.write("### Categorical Variables")
    cat_summary = {}
    for col in categorical_cols:
        st.write(f"**{col}**")
        fig, ax = plt.subplots()
        sns.countplot(
            y=data[col], ax=ax, order=data[col].value_counts().index, palette="viridis"
        )
        plt.title(f"Count of {col}")
        st.pyplot(fig)

        # Collect value counts
        counts = data[col].value_counts().to_dict()
        cat_summary[col] = counts

    # Store section content for the report
    report = f"""
    ## 3. Univariate Analysis
    
    **Numerical Variables:**
    {safe_to_string(pd.DataFrame(num_summary).transpose())}
    
    **Categorical Variables:**
    {safe_to_string(pd.DataFrame(cat_summary).transpose())}
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 4: Bivariate Analysis
def bivariate_analysis(data, target_variable):
    st.subheader("4. Bivariate Analysis")

    # Numerical vs Numerical
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    st.write("### Numerical vs Numerical")
    num_corr = {}
    for col in numerical_cols:
        if col != target_variable:
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[col], y=data[target_variable], ax=ax, color="#ff7f0e")
            plt.title(f"{col} vs {target_variable}")
            st.pyplot(fig)

            # Calculate correlation
            corr = data[col].corr(data[target_variable])
            num_corr[col] = corr

    # Pair Plot
    if st.checkbox("Show Pair Plot for Numerical Features"):
        if len(numerical_cols) >= 2:
            st.write("#### Pair Plot")
            pairplot_fig = sns.pairplot(data[numerical_cols], diag_kind="kde", corner=True)
            st.pyplot(pairplot_fig)
        else:
            st.write("Not enough numerical columns for pair plot.")

    # Numerical vs Categorical
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    st.write("### Numerical vs Categorical")
    num_cat_summary = {}
    for col in categorical_cols:
        if col != target_variable:
            # Box Plot
            fig, ax = plt.subplots()
            sns.boxplot(x=data[col], y=data[target_variable], ax=ax, palette="Set2")
            plt.title(f"{target_variable} by {col}")
            st.pyplot(fig)

            # Violin Plot
            fig, ax = plt.subplots()
            sns.violinplot(x=data[col], y=data[target_variable], ax=ax, palette="Set2")
            plt.title(f"Violin Plot of {target_variable} by {col}")
            st.pyplot(fig)

            # Collect statistics
            group_stats = data.groupby(col)[target_variable].describe().to_dict()
            num_cat_summary[col] = group_stats

    # Categorical vs Categorical
    st.write("### Categorical vs Categorical")
    cat_cat_summary = {}
    if len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1 = categorical_cols[i]
                col2 = categorical_cols[j]
                fig, ax = plt.subplots()
                sns.countplot(x=col1, hue=col2, data=data, ax=ax, palette="magma")
                plt.title(f"Count of {col1} vs {col2}")
                st.pyplot(fig)

                # Cross-tabulation
                ct = pd.crosstab(data[col1], data[col2])
                cat_cat_summary[f"{col1} vs {col2}"] = ct.to_dict()
    else:
        st.write(
            "Not enough categorical columns for Categorical vs Categorical analysis."
        )

    # Store section content for the report
    report = f"""
    ## 4. Bivariate Analysis
    
    **Numerical vs Numerical Correlations:**
    {safe_to_string(pd.Series(num_corr).to_frame(name="Correlation"))}
    
    **Numerical vs Categorical Statistics:**
    {safe_to_string(pd.DataFrame(num_cat_summary).transpose())}
    
    **Categorical vs Categorical Cross-tabulation:**
    {safe_to_string(pd.DataFrame({k: pd.Series(v).to_dict() for k, v in cat_cat_summary.items()}).transpose())}
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 5: Multivariate Analysis
def multivariate_analysis(data):
    st.subheader("5. Multivariate Analysis")

    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_cols) >= 3:
        st.write("### Pair Plot of Numerical Features")
        pairplot_fig = sns.pairplot(data[numerical_cols], diag_kind="kde", corner=True)
        st.pyplot(pairplot_fig)

    st.write("### Advanced Correlation Heatmap")
    if not numerical_cols.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            data[numerical_cols].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
        )
        plt.title("Advanced Correlation Heatmap")
        st.pyplot(fig)

    # Optional: PCA Visualization
    pca_performed = False
    if st.checkbox("Perform PCA and visualize"):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        pca_data = data[numerical_cols].dropna()
        if pca_data.empty:
            st.write("No data available for PCA.")
        else:
            principal_components = pca.fit_transform(pca_data)
            pc_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
            st.write("### PCA Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(x="PC1", y="PC2", data=pc_df, ax=ax, palette="Set3")
            plt.title("PCA Scatter Plot")
            st.pyplot(fig)

            # Explained variance
            explained_var = pca.explained_variance_ratio_
            st.write(f"**Explained Variance by PC1:** {explained_var[0]:.2f}")
            st.write(f"**Explained Variance by PC2:** {explained_var[1]:.2f}")

            pca_performed = True

            # Store section content for the report
            report = f"""
            ## 5. Multivariate Analysis
            
            **Advanced Correlation Heatmap:**
            {safe_to_string(data[numerical_cols].corr())}
            
            **PCA Explained Variance:**
            - **PC1:** {explained_var[0]:.2f}
            - **PC2:** {explained_var[1]:.2f}
            """
            st.session_state["report_sections"].append(report)

    if not pca_performed:
        # Store section content for the report without PCA
        report = f"""
        ## 5. Multivariate Analysis
        
        **Advanced Correlation Heatmap:**
        {safe_to_string(data[numerical_cols].corr())}
        """
        st.session_state["report_sections"].append(report)

# EDA Theme 6: Feature Engineering
def feature_engineering(data):
    st.subheader("6. Feature Engineering")

    # Creating Interaction Terms
    if st.checkbox("Create Interaction Terms"):
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        interaction_terms = []
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                new_col = f"{col1}_x_{col2}"
                data[new_col] = data[col1] * data[col2]
                interaction_terms.append(new_col)
                st.write(f"Created interaction term: **{new_col}**")
        if not interaction_terms:
            st.write("No numerical columns available to create interaction terms.")

        # Store in report
        if interaction_terms:
            report = f"**Feature Engineering:** Created interaction terms: {', '.join(interaction_terms)}."
            st.session_state["report_sections"].append(report)

    # Encoding Categorical Variables
    if st.checkbox("Encode Categorical Variables"):
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            before_cols = data.shape[1]
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
            after_cols = data.shape[1]
            st.write(
                f"Encoded categorical variables using one-hot encoding. Added {after_cols - before_cols} new columns."
            )

            # Store in report
            report = f"**Feature Engineering:** Encoded categorical variables using one-hot encoding. Added {after_cols - before_cols} new columns."
            st.session_state["report_sections"].append(report)
        else:
            st.write("No categorical columns to encode.")

    # Feature Scaling
    if st.checkbox("Apply Feature Scaling"):
        scaler = StandardScaler()
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        before_cols = data[numerical_cols].copy()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        st.write("Applied Standard Scaling to numerical features.")

        # Store in report
        report = f"**Feature Engineering:** Applied Standard Scaling to numerical features."
        st.session_state["report_sections"].append(report)

    return data

# EDA Theme 7: Correlation Analysis
def correlation_analysis(data, target_variable):
    st.subheader("7. Correlation Analysis")

    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_cols) >= 2:
        st.write("### Correlation Coefficients")
        corr = data[numerical_cols].corr()
        st.write(safe_to_string(corr))

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax
        )
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

        # Correlation with Target Variable
        if target_variable in numerical_cols:
            st.write(f"### Correlations with Target Variable: **{target_variable}**")
            corr_target = corr[target_variable].sort_values(ascending=False)
            st.write(safe_to_string(corr_target))
    else:
        st.write("Not enough numerical columns for correlation analysis.")

    # Store section content for the report
    report = f"""
    ## 7. Correlation Analysis
    
    **Correlation Coefficients:**
    {safe_to_string(corr)}
    
    **Top Correlations with {target_variable}:**
    {safe_to_string(corr[target_variable].sort_values(ascending=False).to_frame(name="Correlation"))}
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 8: Advanced Visualization
def advanced_visualization(data, target_variable):
    st.subheader("8. Advanced Visualization")

    # Interactive Plotly Scatter Plot
    if st.checkbox("Interactive Scatter Plot"):
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(numerical_cols) >= 2:
            x_col = st.selectbox(
                "Select X-axis:",
                numerical_cols,
                index=0,
                key="scatter_x",
            )
            y_col = st.selectbox(
                "Select Y-axis:",
                numerical_cols,
                index=1,
                key="scatter_y",
            )
            color_col = st.selectbox(
                "Select Color Grouping (optional):",
                [None] + list(numerical_cols),
                index=0,
                key="scatter_color",
            )
            if color_col and color_col != "None":
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{x_col} vs {y_col}",
                    hover_data=data.columns,
                )
            else:
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"{x_col} vs {y_col}",
                    hover_data=data.columns,
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numerical columns for scatter plot.")

    # Interactive Box Plot
    if st.checkbox("Interactive Box Plot"):
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            cat_col = st.selectbox(
                "Select Categorical Variable:", categorical_cols, key="box_cat"
            )
            num_col = st.selectbox(
                "Select Numerical Variable:", numerical_cols, key="box_num"
            )
            fig = px.box(
                data,
                x=cat_col,
                y=num_col,
                title=f"Box Plot of {num_col} by {cat_col}",
                color=cat_col,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(
                "Ensure there are both categorical and numerical columns for this plot."
            )

    # Interactive Correlation Heatmap
    if st.checkbox("Interactive Correlation Heatmap"):
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(numerical_cols) > 0:
            corr = data[numerical_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Interactive Correlation Heatmap",
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numerical columns available for correlation heatmap.")

    # Store section content for the report
    report = f"""
    ## 8. Advanced Visualization
    
    **Interactive Scatter Plot:**
    - Users can select X and Y axes to visualize relationships.
    - Optional color grouping enhances data interpretation.
    
    **Interactive Box Plot:**
    - Allows comparison of numerical variables across different categories.
    
    **Interactive Correlation Heatmap:**
    - Dynamic heatmap to explore correlations between numerical variables.
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 9: Summary and Insights
def summary_insights(data, target_variable):
    st.subheader("9. Summary and Insights")

    # Key Metrics
    st.write("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("**Total Rows**", data.shape[0])
    with col2:
        st.metric("**Total Columns**", data.shape[1])
    with col3:
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
        st.metric("**Numerical Columns**", len(numerical_cols))
    with col4:
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        st.metric("**Categorical Columns**", len(categorical_cols))

    # Top Correlations with Target Variable
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if target_variable in numerical_cols:
        st.write(f"### Top Correlations with **{target_variable}**")
        top_corr = data.corr()[target_variable].abs().sort_values(
            ascending=False
        ).drop(target_variable).head(5)
        st.write(safe_to_string(top_corr.to_frame(name="Correlation")))
    else:
        st.write(
            f"**{target_variable}** is not a numerical column. Correlation analysis not applicable."
        )

    # Generate Automated Insights
    insights = generate_insights()
    st.write("### Automated Insights")
    for insight in insights:
        st.write(f"- {insight}")

    # Store section content for the report
    report = f"""
    ## 9. Summary and Insights
    
    **Key Metrics:**
    - **Total Rows:** {data.shape[0]}
    - **Total Columns:** {data.shape[1]}
    - **Numerical Columns:** {len(numerical_cols)}
    - **Categorical Columns:** {len(categorical_cols)}
    
    **Top Correlations with {target_variable}:**
    {safe_to_string(top_corr.to_frame(name="Correlation"))}
    
    **Automated Insights:**
    {"chr(10)".join([f"- {insight}" for insight in insights])}
    """
    st.session_state["report_sections"].append(report)

# EDA Theme 10: Model Assumptions Checks
def model_assumptions(data, target_variable):
    st.subheader("10. Model Assumptions Checks (Linear Regression)")

    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if target_variable not in numerical_cols:
        st.warning(
            "Target variable is not numerical. Model assumptions checks are applicable to regression models only."
        )
        return

    # Prepare data for assumptions checks
    X = data[numerical_cols].drop(columns=[target_variable])
    y = data[target_variable]

    # Add a constant for intercept
    X_const = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid
    fitted = model.fittedvalues

    # 1. Linearity
    st.write("### 1. Linearity")
    st.write(
        "Check if the relationship between each independent variable and the dependent variable is linear."
    )
    linearity_reports = []
    for col in X.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[col], y=y, ax=ax, color="#2ca02c")
        sns.lineplot(x=data[col], y=model.predict(X_const), ax=ax, color="red")
        plt.title(f"Linearity Check: {col} vs {target_variable}")
        st.pyplot(fig)

        # Assess linearity (placeholder for automated assessment)
        linearity_reports.append(f"**{col}**: Visual inspection suggests a linear relationship.")

    # 2. Independence (Durbin-Watson)
    st.write("### 2. Independence")
    dw = durbin_watson(residuals)
    st.write(f"**Durbin-Watson Statistic:** {dw:.2f}")
    st.write(
        """
    - **Interpretation:**
      - A value around 2 indicates no autocorrelation.
      - Values approaching 0 suggest positive autocorrelation.
      - Values toward 4 indicate negative autocorrelation.
    """
    )
    independence_report = f"**Durbin-Watson Statistic:** {dw:.2f} suggests {'no' if 1.5 < dw < 2.5 else 'some'} autocorrelation."
    st.session_state["report_sections"].append(independence_report)

    # 3. Homoscedasticity
    st.write("### 3. Homoscedasticity")
    fig, ax = plt.subplots()
    sns.scatterplot(x=fitted, y=residuals, ax=ax, color="#e377c2")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    st.pyplot(fig)
    homoscedasticity_report = "Residuals vs Fitted Values plot shows [insert assessment based on plot]."
    st.session_state["report_sections"].append(homoscedasticity_report)

    # 4. Normality of Residuals
    st.write("### 4. Normality of Residuals")
    fig, ax = plt.subplots()
    qqplot(residuals, line="s", ax=ax)
    plt.title("Q-Q Plot of Residuals")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax, color="#bcbd22")
    plt.title("Histogram of Residuals")
    st.pyplot(fig)

    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    st.write(f"**Shapiro-Wilk Test Statistic:** {shapiro_stat:.2f}")
    st.write(f"**Shapiro-Wilk Test p-value:** {shapiro_p:.4f}")
    st.write(
        """
    - **Interpretation:**
      - A p-value > 0.05 indicates that residuals are normally distributed.
      - A p-value ≤ 0.05 suggests deviation from normality.
    """
    )
    normality_report = f"**Shapiro-Wilk Test:** p-value = {shapiro_p:.4f} suggests residuals are {'normally distributed' if shapiro_p > 0.05 else 'not normally distributed'}."
    st.session_state["report_sections"].append(normality_report)

    # 5. Multicollinearity (VIF)
    st.write("### 5. Multicollinearity (Variance Inflation Factor - VIF)")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]
    st.write(safe_to_string(vif_data))
    st.write(
        """
    - **Interpretation:**
      - **VIF < 5:** No multicollinearity.
      - **5 ≤ VIF < 10:** Moderate multicollinearity.
      - **VIF ≥ 10:** High multicollinearity.
      
      High multicollinearity may require removing or combining features.
    """
    )
    vif_report = f"**Variance Inflation Factor (VIF):**\n{safe_to_string(vif_data)}"
    st.session_state["report_sections"].append(vif_report)

# Machine Learning Theme: Model Training
def run_machine_learning(data, target_variable):
    st.subheader("11. Machine Learning Model Training")

    if target_variable in data.columns:
        X = data.drop(columns=[target_variable])
        y = data[target_variable]

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Check if target variable is categorical or numerical
        if y.dtype in ["object", "category"]:
            st.write("**Target variable is categorical. Classification models will be used.**")
            # Example: Logistic Regression for classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Encode target variable
            y_encoded = y.astype("category").cat.codes

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )

            # Feature Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit Logistic Regression
            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display results
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.2f}")

            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.title("Confusion Matrix")
            st.pyplot(fig)

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Store section content for the report
            report = f"""
            ## 11. Machine Learning Model Training
            
            **Model:** Logistic Regression
            
            **Accuracy:** {accuracy:.2f}
            
            **Confusion Matrix:**
            {safe_to_string(pd.DataFrame(cm))}
            
            **Classification Report:**
            {classification_report(y_test, y_pred)}
            """
            st.session_state["report_sections"].append(report)

        else:
            st.write("**Target variable is numerical. Regression models will be used.**")
            # Example: Linear Regression for regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Feature Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit a basic linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display results
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.2f}")

            # Display model coefficients
            st.write("### Model Coefficients:")
            coef = pd.Series(model.coef_, index=X.columns)
            st.write(safe_to_string(coef.sort_values(ascending=False).to_frame()))

            # Store section content for the report
            report = f"""
            ## 11. Machine Learning Model Training
            
            **Model:** Linear Regression
            
            **Mean Squared Error (MSE):** {mse:.2f}
            **R-squared (R²):** {r2:.2f}
            
            **Model Coefficients:**
            {safe_to_string(coef.sort_values(ascending=False).to_frame())}
            """
            st.session_state["report_sections"].append(report)
    else:
        st.write("Selected target variable is not present in the dataset.")

# Function to generate a full EDA and Modeling report
def generate_full_report():
    report_content = "\n\n".join(st.session_state["report_sections"])
    return report_content

# Main function to run the Streamlit app
def main():
    st.title("📊 Automated Data Profiling Dashboard")
    st.markdown(
        """
    Welcome to the **Automated Data Profiling Dashboard**! Upload your CSV dataset below to perform an extensive Exploratory Data Analysis (EDA) and prepare your data for machine learning model building.
    """
    )

    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            # Allow user to select the target variable
            st.subheader("🔍 Select Target Variable")
            target_variable = st.selectbox(
                "Select the target variable for analysis:",
                options=data.columns.tolist(),
            )

            # Generate the EDA report
            if st.button("📈 Generate EDA Report"):
                with st.spinner("Generating EDA report..."):
                    # Clear previous report sections
                    st.session_state["report_sections"] = []

                    # Perform EDA Themes
                    data_understanding(data)
                    data = data_cleaning(data)
                    univariate_analysis(data)
                    bivariate_analysis(data, target_variable)
                    multivariate_analysis(data)
                    data = feature_engineering(data)
                    correlation_analysis(data, target_variable)
                    advanced_visualization(data, target_variable)
                    summary_insights(data, target_variable)
                    model_assumptions(data, target_variable)
                    run_machine_learning(data, target_variable)

                st.success("✅ EDA Report Generated Successfully!")

    # Display Download Report button only after report is generated
    if len(st.session_state["report_sections"]) > 0:
        st.subheader("📥 Download EDA Report")
        report = generate_full_report()
        # Convert the report to a downloadable file
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="EDA_Report.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()


