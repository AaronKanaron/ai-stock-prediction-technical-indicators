import pandas as pd
import numpy as np
import ta
import os

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Parameters:
    df (DataFrame): DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    DataFrame: Original data with added technical indicators
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure Date is datetime and set as index for calculations
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Simple Moving Averages (SMA)
    data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
    data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    
    # RSI (Relative Strength Index) - 14 days
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD (12-26-9 configuration)
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_histogram'] = macd.macd_diff()
    
    # Bollinger Bands (20-day with 2 standard deviations)
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_middle'] = bollinger.bollinger_mavg()
    data['BB_lower'] = bollinger.bollinger_lband()
    data['BB_width'] = data['BB_upper'] - data['BB_lower']
    
    # Rolling Standard Deviation (20 days)
    data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
    
    # Daily Return (Return_1d)
    data['Return_1d'] = data['Close'].pct_change()
    
    # Enhanced Features
    # Momentum features (lagged to avoid lookahead bias)
    data['Return_5d_lag'] = data['Close'].pct_change(5).shift(1)
    data['Return_10d_lag'] = data['Close'].pct_change(10).shift(1)
        
    # Trend features
    data['SMA_ratio'] = data['Close'] / data['SMA_20']
    data['Price_change_20d'] = data['Close'] / data['Close'].shift(20) - 1
    
    # Target: 1 if 5-day future return > 2%, 0 otherwise
    threshold = 0.02 
    data['Future_Close_5d'] = data['Close'].shift(-5)
    data['Return_5d'] = (data['Future_Close_5d'] - data['Close']) / data['Close']
    data['Target'] = 1
    data.loc[data['Return_5d'] > threshold, 'Target'] = 2
    data.loc[data['Return_5d'] < -threshold, 'Target'] = 0
    
    # Remove the helper column
    data = data.drop(['Future_Close_5d', 'Return_5d'], axis=1)
    
    # Store original shape for reporting
    original_shape = data.shape
    
    # Remove rows with NaN values (typically the first rows due to technical indicators)
    data_cleaned = data.dropna()
    
    # Remove the last 5 rows since they don't have valid targets (5-day look-ahead)
    if len(data_cleaned) > 5:
        data_cleaned = data_cleaned.iloc[:-5]
    
    # Print cleaning statistics
    rows_removed_nan = original_shape[0] - len(data.dropna())
    rows_removed_total = original_shape[0] - len(data_cleaned)
    print(f"Data cleaning summary:")
    print(f"  Original rows: {original_shape[0]}")
    print(f"  Rows with NaN removed: {rows_removed_nan}")
    print(f"  Last 5 rows removed (target look-ahead): 5")
    print(f"  Total rows removed: {rows_removed_total}")
    print(f"  Final clean dataset: {len(data_cleaned)} rows")
    
    return data_cleaned

def process_stock_data(input_file, output_file):
    """
    Process a single stock data file and save with technical indicators
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    print(f"Processing {input_file}...")
    
    # Read the data
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Save to output file
    df_with_indicators.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print(f"Final data shape: {df_with_indicators.shape}")
    
    # Show first few rows for verification
    print("\nFirst 5 rows of processed data:")
    print(df_with_indicators.head())
    print("\nColumn names:")
    print(df_with_indicators.columns.tolist())
    
    return df_with_indicators

def main():
    """
    Main function to process both OMXS30 and SP500 data files
    """
    # Define file paths
    rawdata_dir = "rawdata"
    data_dir = "data"
    
    files_to_process = [
        ("atlascopco_rawdata.csv", "atlascopco_rawdata.csv"),
        ("electrolux_rawdata.csv", "electrolux_rawdata.csv"),
        ("ericsson_rawdata.csv", "ericsson_rawdata.csv"),
        ("getinge_rawdata.csv", "getinge_rawdata.csv"),
        ("handelsbanken_rawdata.csv", "handelsbanken_rawdata.csv"),
        ("hmb_rawdata.csv", "hmb_rawdata.csv"),
        ("investor_rawdata.csv", "investor_rawdata.csv"),
        ("nordea_rawdata.csv", "nordea_rawdata.csv"),
        ("sandvik_rawdata.csv", "sandvik_rawdata.csv"),
        ("seb_rawdata.csv", "seb_rawdata.csv"),
        ("skf_rawdata.csv", "skf_rawdata.csv"),
        ("swedbank_rawdata.csv", "swedbank_rawdata.csv"),
        ("telia_rawdata.csv", "telia_rawdata.csv"),
        ("volvo_rawdata.csv", "volvo_rawdata.csv")
    ]
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Process each file
    for input_filename, output_filename in files_to_process:
        input_path = os.path.join(rawdata_dir, input_filename)
        output_path = os.path.join(data_dir, output_filename)
        
        if os.path.exists(input_path):
            try:
                df = process_stock_data(input_path, output_path)
                print(f"\n{'='*50}")
                print(f"Successfully processed {input_filename}")
                print(f"Output saved to {output_path}")
                print(f"{'='*50}\n")
            except Exception as e:
                print(f"Error processing {input_filename}: {str(e)}")
        else:
            print(f"Warning: {input_path} not found!")

if __name__ == "__main__":
    main()
