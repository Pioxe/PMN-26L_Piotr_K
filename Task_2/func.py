import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

def load_heart_data():
    # Attempts to fetch data using UCI repository ID. If it fails, loads from a local CSV file.
 
    file_path = "heart.csv"
    # Standard column names for the Cleveland Heart Disease dataset
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    try:
        print("Attempting to fetch data from UCI via ID 45...")
        heart_disease = fetch_ucirepo(id=45)
        
        # Combine features and targets into one DataFrame
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        
        # Ensure column names are exactly as want
        df.columns = columns
        
        source = "UCI Repository API (ID 45)"
        return df, source
        
    except Exception as e:
        print(f"Web API failed (Error: {e}).")
        
        # Fallback to local file
        if os.path.exists(file_path):
            
            df = pd.read_csv(file_path, header=None, names=columns, na_values="?")
            source = f"Local file ({file_path})"
            return df, source
        else:
            print(f"Error: No data found via API or local file '{file_path}'.")
            return None, "No data available"

def dataset_summary(df, source):
    print("\n" + "="*40)
    print("DATASET SUMMARY")
    print("="*40)
    print(f"Data source    : {source}")
    print(f"Number of rows : {df.shape[0]}") 
    print(f"Number of columns : {df.shape[1]}")

def dataset_overview(df):
    print("\n" + "="*40)
    print("DATASET OVERVIEW (First 5 rows)")
    print("="*40)
    print(df.head())

def dataset_missing_values_analysis(df):
    print("\n" + "="*40)
    print("MISSING VALUES ANALYSIS")
    print("="*40)
    # Calculate the sum of null values for each feature
    missing = df.isnull().sum()
    # Display only the columns that actually contain missing values
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

def dataset_variable_distribution(df):
    print("\n" + "="*40)
    print("VARIABLE DISTRIBUTION (Target)")
    print("="*40)
    # Count occurrences of each unique value in the 'target' column
    print(df['target'].value_counts())

def dataset_check_categorical_health(df):
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    print("\n" + "="*40)
    print("CATEGORICAL COLUMN ANALYSIS")
    print("="*40)
    # Iterate through each categorical column to check its status
    for col in categorical_cols:
        if col in df.columns:
            # Retrieve unique categories present in the column
            unique_values = df[col].unique()
            # Count missing values specifically for this column
            nan_count = df[col].isnull().sum()
            # :10 and :2 ensure the output is aligned and easy to read
            print(f"Column: {col:10} | Absence (NaN): {nan_count:2} | Unique values: {unique_values}")
def dataset_show_rows_with_nan(df):
    print("\n" + "="*40)
    print("ROWS WITH MISSING VALUES (NaN)")
    print("="*40)
    # Filter the DataFrame to isolate rows where any column contains a null value
    # axis=1 checks across columns for each row
    nan_rows = df[df.isnull().any(axis=1)]
    # Logic to handle the display based on whether gaps were found
    if not nan_rows.empty:
        print(nan_rows)
    else:
        print("No rows with NaN found.")
def display_analysis(df, source):
    if df is not None:
        #Basic structural info
        dataset_summary(df, source)
        #Manual data inspection
        dataset_overview(df)
        #Comprehensive missing data check
        dataset_missing_values_analysis(df)
        #Detailed of categorical health features
        dataset_check_categorical_health(df)
        #Class balance check Healthy vs. Sick
        dataset_variable_distribution(df)
    else:
        print("Analysis impossible: DataFrame is None.")





if __name__ == "__main__":
    heart_df, data_source = load_heart_data()
    display_analysis(heart_df, data_source)