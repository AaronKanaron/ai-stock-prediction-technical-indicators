import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

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

def discover_csv_files(data_dir="data"):
    """
    Discover all CSV files in the specified data directory.
    
    Parameters:
    data_dir (str): Path to the data directory
    
    Returns:
    list: List of CSV file paths
    """
    csv_pattern = os.path.join(data_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {data_dir} directory!")
        return []
    
    # Sort files for consistent processing order
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files in {data_dir}:")
    for i, file in enumerate(csv_files, 1):
        file_name = os.path.basename(file)
        print(f"  {i}. {file_name}")
    
    return csv_files

def get_base_filename(file_path):
    """
    Extract base filename without extension from file path.
    
    Parameters:
    file_path (str): Full path to the file
    
    Returns:
    str: Base filename without extension
    """
    return Path(file_path).stem

def process_single_file(input_file, output_dir_train, output_dir_test, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Process a single CSV file: split and save the data.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_dir_train (str): Directory for training and validation files
    output_dir_test (str): Directory for test files
    train_ratio, val_ratio, test_ratio (float): Split ratios
    
    Returns:
    dict: Summary statistics for the processed file
    """
    base_filename = get_base_filename(input_file)
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_filename}")
    print(f"{'='*60}")
    
    try:
        # Split the data
        train_data, val_data, test_data = split_data(
            input_file=input_file,
            output_dir_train=output_dir_train,
            output_dir_test=output_dir_test,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Save the splits
        train_file, val_file, test_file = save_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            base_filename=base_filename,
            output_dir_train=output_dir_train,
            output_dir_test=output_dir_test
        )
        
        # Collect statistics
        stats = {
            'filename': base_filename,
            'status': 'SUCCESS',
            'total_rows': len(train_data) + len(val_data) + len(test_data),
            'train_rows': len(train_data),
            'val_rows': len(val_data),
            'test_rows': len(test_data),
            'train_targets': train_data['Target'].notna().sum() if 'Target' in train_data.columns else 0,
            'val_targets': val_data['Target'].notna().sum() if 'Target' in val_data.columns else 0,
            'test_targets': test_data['Target'].notna().sum() if 'Target' in test_data.columns else 0,
            'date_range': f"{train_data['Date'].min().strftime('%Y-%m-%d')} to {test_data['Date'].max().strftime('%Y-%m-%d')}"
        }
        
        print(f"\n✅ {base_filename} processed successfully!")
        print(f"   Total rows: {stats['total_rows']}")
        print(f"   Date range: {stats['date_range']}")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ Error processing {base_filename}: {str(e)}")
        return {
            'filename': base_filename,
            'status': 'ERROR',
            'error': str(e)
        }

def print_summary_report(results):
    """
    Print a comprehensive summary report of all processed files.
    
    Parameters:
    results (list): List of processing results for each file
    """
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY REPORT")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'ERROR']
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total files processed: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        print(f"\n✅ SUCCESSFUL PROCESSING:")
        print(f"{'Filename':<20} {'Total Rows':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Date Range':<25}")
        print(f"{'-'*80}")
        
        total_train = total_val = total_test = 0
        for result in successful:
            print(f"{result['filename']:<20} {result['total_rows']:<12} {result['train_rows']:<8} "
                  f"{result['val_rows']:<8} {result['test_rows']:<8} {result['date_range']:<25}")
            total_train += result['train_rows']
            total_val += result['val_rows']
            total_test += result['test_rows']
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} {total_train+total_val+total_test:<12} {total_train:<8} {total_val:<8} {total_test:<8}")
    
    if failed:
        print(f"\n❌ FAILED PROCESSING:")
        for result in failed:
            print(f"   {result['filename']}: {result['error']}")
    
    print(f"\n📁 OUTPUT DIRECTORIES:")
    print(f"   Training & Validation files: training/")
    print(f"   Test files: test/")

def main():
    """
    Main function to split all CSV files in the data directory.
    """
    
    # Configuration
    data_dir = "data"
    output_dir_train = "training"
    output_dir_test = "test"
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    print(f"{'='*80}")
    print("BATCH DATA SPLITTING - PROCESSING ALL CSV FILES")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Split ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    # Discover all CSV files
    csv_files = discover_csv_files(data_dir)
    if not csv_files:
        return
    
    # Process each file
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Starting processing...")
        
        result = process_single_file(
            input_file=csv_file,
            output_dir_train=output_dir_train,
            output_dir_test=output_dir_test,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        results.append(result)
    
    # Print comprehensive summary
    print_summary_report(results)
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETED!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
