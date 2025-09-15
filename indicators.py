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
    
    # Target: 1 if next day's return > 0, 0 otherwise
    data['Next_Return'] = data['Return_1d'].shift(-1)
    data['Target'] = (data['Next_Return'] > 0).astype(int)
    
    # Remove the helper column
    data = data.drop('Next_Return', axis=1)
    
    return data

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
        ("OMXS30_10year_data.csv", "OMXS30_with_indicators.csv"),
        ("SP500_10year_data.csv", "SP500_with_indicators.csv")
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
