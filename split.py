import pandas as pd
import numpy as np
import os

def split_data(input_file, output_dir_train, output_dir_test, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split time series data into training, validation, and test sets.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_dir_train (str): Directory for training and validation files
    output_dir_test (str): Directory for test files
    train_ratio (float): Proportion for training (default: 0.6)
    val_ratio (float): Proportion for validation (default: 0.2)
    test_ratio (float): Proportion for test (default: 0.2)
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Read the data
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Ensure Date column is datetime and data is sorted chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate split indices (for time series, we split chronologically)
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remaining goes to test
    
    print(f"Split sizes:")
    print(f"  Training: {n_train} rows ({n_train/n_total*100:.1f}%)")
    print(f"  Validation: {n_val} rows ({n_val/n_total*100:.1f}%)")
    print(f"  Test: {n_test} rows ({n_test/n_total*100:.1f}%)")
    
    # Split the data chronologically
    train_data = df.iloc[:n_train].copy()
    val_data = df.iloc[n_train:n_train+n_val].copy()
    test_data = df.iloc[n_train+n_val:].copy()
    
    print(f"Date ranges:")
    print(f"  Training: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"  Validation: {val_data['Date'].min()} to {val_data['Date'].max()}")
    print(f"  Test: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    return train_data, val_data, test_data

def save_splits(train_data, val_data, test_data, base_filename, output_dir_train, output_dir_test):
    """
    Save the split datasets to appropriate directories.
    
    Parameters:
    train_data, val_data, test_data (DataFrame): Split datasets
    base_filename (str): Base filename (e.g., 'OMXS30')
    output_dir_train (str): Directory for training and validation files
    output_dir_test (str): Directory for test files
    """
    
    # Ensure output directories exist
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    
    # Define output file paths
    train_file = os.path.join(output_dir_train, f"{base_filename}_train.csv")
    val_file = os.path.join(output_dir_train, f"{base_filename}_val.csv")
    test_file = os.path.join(output_dir_test, f"{base_filename}_test.csv")
    
    # Save the files
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"\nFiles saved:")
    print(f"  Training: {train_file} ({len(train_data)} rows)")
    print(f"  Validation: {val_file} ({len(val_data)} rows)")
    print(f"  Test: {test_file} ({len(test_data)} rows)")
    
    return train_file, val_file, test_file

def main():
    """
    Main function to split OMXS30 data into training, validation, and test sets.
    """
    
    # Define file paths
    input_file = "data/OMXS30.csv"
    output_dir_train = "training"
    output_dir_test = "test"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    try:
        # Split the data
        train_data, val_data, test_data = split_data(
            input_file=input_file,
            output_dir_train=output_dir_train,
            output_dir_test=output_dir_test,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Save the splits
        train_file, val_file, test_file = save_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            base_filename="OMXS30",
            output_dir_train=output_dir_train,
            output_dir_test=output_dir_test
        )
        
        print(f"\n{'='*50}")
        print("Data split completed successfully!")
        print(f"{'='*50}")
        
        # Display some basic statistics
        print(f"\nBasic statistics:")
        print(f"Training set:")
        print(f"  Rows: {len(train_data)}")
        print(f"  Non-null targets: {train_data['Target'].notna().sum()}")
        
        print(f"Validation set:")
        print(f"  Rows: {len(val_data)}")
        print(f"  Non-null targets: {val_data['Target'].notna().sum()}")
        
        print(f"Test set:")
        print(f"  Rows: {len(test_data)}")
        print(f"  Non-null targets: {test_data['Target'].notna().sum()}")
        
    except Exception as e:
        print(f"Error during data splitting: {str(e)}")
        raise

if __name__ == "__main__":
    main()
