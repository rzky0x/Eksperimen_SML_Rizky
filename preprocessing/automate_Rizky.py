"""
automate_Rizky.py
Automated preprocessing script for Wine Quality dataset.
Converts the raw experimentation steps into reusable functions.

Author: Rizky
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data(filepath=None):
    """
    Load Wine Quality dataset from local file or UCI repository.
    
    Parameters:
        filepath (str): Path to CSV file. If None, downloads from UCI.
    
    Returns:
        pd.DataFrame: Raw wine quality dataframe.
    """
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading data from local file: {filepath}")
        df = pd.read_csv(filepath, sep=';')
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        print(f"[INFO] Downloading data from UCI repository...")
        df = pd.read_csv(url, sep=';')
    
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(df):
    """
    Perform basic EDA on the dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        dict: EDA summary statistics.
    """
    eda_summary = {}
    
    # Basic info
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Shape
    print(f"\n[EDA] Shape: {df.shape}")
    eda_summary['shape'] = df.shape
    
    # Data types
    print(f"\n[EDA] Data Types:\n{df.dtypes}")
    eda_summary['dtypes'] = df.dtypes.to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    print(f"\n[EDA] Missing Values:\n{missing}")
    eda_summary['missing_values'] = missing.to_dict()
    
    # Descriptive statistics
    print(f"\n[EDA] Descriptive Statistics:\n{df.describe()}")
    eda_summary['describe'] = df.describe().to_dict()
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n[EDA] Duplicate Rows: {duplicates}")
    eda_summary['duplicates'] = duplicates
    
    # Target distribution (quality)
    if 'quality' in df.columns:
        print(f"\n[EDA] Quality Distribution:\n{df['quality'].value_counts().sort_index()}")
        eda_summary['quality_distribution'] = df['quality'].value_counts().sort_index().to_dict()
    
    return eda_summary


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the wine quality dataset:
    - Handle missing values
    - Remove duplicates
    - Create binary target (good: quality > 5, bad: quality <= 5)
    - Feature scaling with StandardScaler
    - Train/test split (stratified)
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
        test_size (float): Test set proportion.
        random_state (int): Random seed.
    
    Returns:
        dict: Dictionary containing preprocessed data splits and scaler.
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    df_processed = df.copy()
    
    # --- Handle Missing Values ---
    missing_before = df_processed.isnull().sum().sum()
    print(f"\n[PREPROCESS] Missing values before: {missing_before}")
    
    # Fill numeric columns with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"  - Filled '{col}' missing values with median: {median_val}")
    
    missing_after = df_processed.isnull().sum().sum()
    print(f"[PREPROCESS] Missing values after: {missing_after}")
    
    # --- Remove Duplicates ---
    duplicates_before = df_processed.duplicated().sum()
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = duplicates_before - df_processed.duplicated().sum()
    print(f"\n[PREPROCESS] Duplicates removed: {duplicates_removed}")
    print(f"[PREPROCESS] Dataset size after removing duplicates: {df_processed.shape[0]}")
    
    # --- Create Binary Target ---
    print(f"\n[PREPROCESS] Creating binary target variable...")
    print(f"  Original quality range: {df_processed['quality'].min()} - {df_processed['quality'].max()}")
    df_processed['quality_label'] = (df_processed['quality'] > 5).astype(int)
    print(f"  Binary distribution:")
    print(f"    Bad  (quality <= 5): {(df_processed['quality_label'] == 0).sum()}")
    print(f"    Good (quality >  5): {(df_processed['quality_label'] == 1).sum()}")
    
    # --- Separate Features and Target ---
    feature_cols = [col for col in df_processed.columns if col not in ['quality', 'quality_label']]
    X = df_processed[feature_cols].copy()
    y = df_processed['quality_label'].copy()
    
    print(f"\n[PREPROCESS] Features ({len(feature_cols)}): {feature_cols}")
    
    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    print(f"\n[PREPROCESS] Train/Test Split (test_size={test_size}):")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_cols, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_cols, 
        index=X_test.index
    )
    print(f"\n[PREPROCESS] Feature scaling applied (StandardScaler)")
    
    result = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    return result


# ============================================================
# 4. SAVE PREPROCESSED DATA
# ============================================================

def save_preprocessed(data, output_dir="wine_quality_preprocessing"):
    """
    Save preprocessed data to CSV files.
    
    Parameters:
        data (dict): Dictionary with X_train, X_test, y_train, y_test.
        output_dir (str): Output directory path.
    """
    print("\n" + "="*60)
    print("SAVING PREPROCESSED DATA")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target for saving
    train_df = data['X_train'].copy()
    train_df['quality_label'] = data['y_train']
    
    test_df = data['X_test'].copy()
    test_df['quality_label'] = data['y_test']
    
    # Save files
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n[SAVE] Train data saved to: {train_path} ({train_df.shape[0]} rows)")
    print(f"[SAVE] Test data saved to:  {test_path} ({test_df.shape[0]} rows)")
    
    # Also save full preprocessed dataset
    full_df = pd.concat([train_df, test_df], axis=0)
    full_path = os.path.join(output_dir, "wine_quality_preprocessed.csv")
    full_df.to_csv(full_path, index=False)
    print(f"[SAVE] Full preprocessed data saved to: {full_path} ({full_df.shape[0]} rows)")
    
    return {
        'train_path': train_path,
        'test_path': test_path,
        'full_path': full_path
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("WINE QUALITY PREPROCESSING PIPELINE")
    print("Author: Rizky")
    print("="*60)
    
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(os.path.dirname(script_dir), "wine_quality_raw")
    raw_file = os.path.join(raw_data_dir, "winequality-red.csv")
    output_dir = os.path.join(script_dir, "wine_quality_preprocessing")
    
    # Step 1: Load data
    df = load_data(raw_file)
    
    # Step 2: EDA
    eda_summary = perform_eda(df)
    
    # Step 3: Preprocess
    preprocessed_data = preprocess_data(df, test_size=0.2, random_state=42)
    
    # Step 4: Save
    saved_paths = save_preprocessed(preprocessed_data, output_dir=output_dir)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    return preprocessed_data, saved_paths


if __name__ == "__main__":
    main()
