import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_and_explore_data(file_path):
    """Loads the dataset and performs initial exploration."""
    try:
        df = pd.read_csv(file_path)
        print("--- Raw Data Summary ---")
        print(f"Shape: {df.shape}")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def clean_data(df):
    """Performs comprehensive data cleaning."""
    cleaned_df = df.copy()
    
    # Handle Missing Values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    num_cols = cleaned_df.select_dtypes(include=['number']).columns
    cat_cols = cleaned_df.select_dtypes(include=['object']).columns
    
    cleaned_df[num_cols] = num_imputer.fit_transform(cleaned_df[num_cols])
    cleaned_df[cat_cols] = cat_imputer.fit_transform(cleaned_df[cat_cols])
    
    # Standardize Categorical Data
    for col in cat_cols:
        cleaned_df[col] = cleaned_df[col].str.strip().str.title()
    
    # Remove Duplicates
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    final_count = len(cleaned_df)
    print(f"Removed {initial_count - final_count} duplicate rows.")
    
    print("--- Data Cleaning Complete ---")
    return cleaned_df

def engineer_features(df):
    """Creates new features from the cleaned data."""
    # Example: Create energy consumption per capita
    if 'total_energy_use' in df.columns and 'household_size' in df.columns:
        df['energy_per_capita'] = df['total_energy_use'] / df['household_size']
        print("--- Feature Engineering Complete ---")
        print("New feature 'energy_per_capita' added.")
    else:
        print("Required columns for feature engineering not found.")
    return df

def export_data(df, output_path):
    """Exports the cleaned DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"--- Exporting Data ---")
    print(f"Cleaned dataset successfully saved to '{output_path}'")

def main():
    """Main function to orchestrate the data wrangling pipeline."""
    input_file = 'household_energy_raw.csv'
    output_file = 'cleaned_household_energy_data.csv'
    
    print("=== STARTING DATA WRANGLING PIPELINE ===")
    
    # Step 1: Load and Explore
    raw_df = load_and_explore_data(input_file)
    if raw_df is None:
        return
    
    # Step 2: Clean
    cleaned_df = clean_data(raw_df)
    
    # Step 3: Engineer Features
    final_df = engineer_features(cleaned_df)
    
    # Step 4: Export
    export_data(final_df, output_file)
    
    print("=== DATA WRANGLING PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
