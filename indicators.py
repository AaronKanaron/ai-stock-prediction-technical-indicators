import pandas as pd
import numpy as np
import ta
import os

class TechnicalIndicators:
    """
    Class to calculate technical indicators for stock data
    """
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
    
    def Price_vs_SMA(self, window):
        sma = ta.trend.sma_indicator(self.df['Close'], window=window)
        return ((self.df['Close'] - sma) / sma) * 100
    
    @staticmethod
    def calculate_volume_ratio(data, window=20):
        volume_avg = data['Volume'].rolling(window=window).mean()
        volume_ratio = data['Volume'] / volume_avg
        return volume_ratio
    
    @staticmethod
    def calculate_volume_rank(data, window=50):
        """Beräknar volymens percentilrank över rullande period"""
        def percentile_rank(series):
            if len(series) < 2:
                return 0.5
            return (series < series.iloc[-1]).sum() / len(series)
        
        volume_rank = data['Volume'].rolling(window=window).apply(percentile_rank, raw=False)
        return volume_rank
    
    @staticmethod
    def calculate_macd_normalized(data, window=20):
        """"Z-score normalisering av MACD"""
        macd_mean = data['MACD'].rolling(window=window).mean()
        macd_std = data['MACD'].rolling(window=window).std()
        macd_normalized = (data['MACD'] - macd_mean) / macd_std
        return macd_normalized
    
    @staticmethod
    def calculate_trend_strength(data):
        """Beräkna trend strength baserat på close position i daily range"""
        daily_range = data['High'] - data['Low']
        
        # Undvik division med noll
        daily_range = daily_range.where(daily_range > 0, np.nan)
        
        # Position av close inom dagens range (0 = Low, 1 = High)
        trend_strength = (data['Close'] - data['Low']) / daily_range
        
        # Konvertera till -1 till +1 skala (-1 = nära Low, +1 = nära High)
        trend_strength = (trend_strength - 0.5) * 2
        
        return trend_strength.fillna(0)




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
    
    # Price vs SMA indicators
    # Detta är för att generalisera modellen för alla aktier
    # En skillnad på +5% SMA är samma oavsett aktiepris
    data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
    data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    
    data['Price_vs_SMA5'] = ((data['Close'] - data['SMA_5']) / data['SMA_5']) * 100
    data['Price_vs_SMA10'] = ((data['Close'] - data['SMA_10']) / data['SMA_10']) * 100
    data['Price_vs_SMA20'] = ((data['Close'] - data['SMA_20']) / data['SMA_20']) * 100
    data['Price_vs_SMA50'] = ((data['Close'] - data['SMA_50']) / data['SMA_50']) * 100
    
    #kategoriskt trend, 1 = uppåt, -1 = nedåt, 0 = ingen förändring | optimiserat för boosting algoritm
    data['SMA5_trend'] = np.where(data['SMA_5'] > data['SMA_5'].shift(1), 1,
                        np.where(data['SMA_5'] < data['SMA_5'].shift(1), -1, 0))

    data['SMA20_trend'] = np.where(data['SMA_20'] > data['SMA_20'].shift(1), 1,
                        np.where(data['SMA_20'] < data['SMA_20'].shift(1), -1, 0))

    # Ta bort de ursprungliga SMA kolumnerna efter att ha skapat Price_vs_SMA
    data = data.drop(['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50'], axis=1)
    
    # RSI (Relative Strength Index) - 14 och 7 dagar
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    data['RSI_7'] = ta.momentum.rsi(data['Close'], window=7)

    # RSI kategorier
    data['RSI_oversold'] = np.where(data['RSI_14'] <= 25, 1, 0)
    data['RSI_overbought'] = np.where(data['RSI_14'] >= 75, 1, 0) 
    data['RSI_neutral'] = np.where((data['RSI_14'] > 25) & (data['RSI_14'] < 75), 1, 0)

    
    # Bollinger Bands (20-day with 2 standard deviations)
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_middle'] = bollinger.bollinger_mavg()
    data['BB_lower'] = bollinger.bollinger_lband()
    data['BB_width'] = data['BB_upper'] - data['BB_lower']
    
    # BB_position: Normaliserad position (0 = vid lower band, 1 = vid upper band)
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    data['BB_position'] = data['BB_position'].clip(lower=0, upper=1)

    bb_width_threshold = data['BB_width'].rolling(window=100).quantile(0.25)
    data['BB_squeeze'] = np.where(data['BB_width'] <= bb_width_threshold, 1, 0)

    # Drop BB columns if they exist
    existing_bb_cols = [col for col in ['BB_upper', 'BB_middle', 'BB_lower'] if col in data.columns]
    if existing_bb_cols: data.drop(existing_bb_cols, axis=1, inplace=True)
    
    # Volume ratio och rank
    data['Volume_ratio'] = TechnicalIndicators.calculate_volume_ratio(data, window=20).fillna(1).clip(upper=10)
    data['Volume_rank'] = TechnicalIndicators.calculate_volume_rank(data, window=50)
    
    # MACD (12-26-9 configuration)
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_histogram'] = macd.macd_diff()    
    data['MACD_normalized'] = TechnicalIndicators.calculate_macd_normalized(data, window=50)
    data.drop(['MACD'], axis=1, inplace=True)
    
    # Rolling Standard Deviation (20 days)
    data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
    
    # Daily Returns 
    data['Return_1d'] = data['Close'].pct_change()    
    data['Return_3d'] = data['Close'].pct_change(3)
    data['Return_5d'] = data['Close'].pct_change(5)
    data['Return_1d_lag1'] = data['Return_1d'].shift(1)
    data['Return_3d_lag1'] = data['Return_3d'].shift(1)
    data['Return_5d_lag1'] = data['Return_5d'].shift(1)

    data['Volatility_5d'] = data['Return_1d'].rolling(window=5).std()
    data['Volatility_20d'] = data['Return_1d'].rolling(window=20).std()
    data['Volatility_ratio'] = data['Volatility_5d'] / data['Volatility_20d']
    
    data['Daily_range'] = (data['High'] - data['Low']) / data['Close']
    data['Gap'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
    data['Trend_strength'] = TechnicalIndicators.calculate_trend_strength(data)
    
    # Trend features
    # data['SMA_ratio'] = data['Close'] / data['SMA_20']
    # data['Price_change_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
    #========================
    # Målvariabelns definition
    #========================
    threshold = 0.015
    data['Future_Close_xd'] = data['Close'].shift(-5)
    data['Return_xd'] = (data['Future_Close_xd'] - data['Close']) / data['Close']
    data['Target'] = 1
    data.loc[data['Return_xd'] > threshold, 'Target'] = 2
    data.loc[data['Return_xd'] < -threshold, 'Target'] = 0
    data = data.drop(['Future_Close_xd', "Return_xd"], axis=1)
    
    # data['ATR_20'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=20)
    # data['Threshold'] = 0.7 * data['ATR_20'] / data['Close']
    # data['Future_Close_3d'] = data['Close'].shift(-3)
    # data['Return_3d'] = (data['Future_Close_3d'] - data['Close']) / data['Close']
    # data['Target'] = np.where(data['Return_3d'] > data['Threshold'], 2,
    #              np.where(data['Return_3d'] < -data['Threshold'], 0, 1))
    # data = data.drop(['Future_Close_3d', 'Return_3d', 'ATR_20', 'Threshold'], axis=1)
    
    #========================
    
    # Store original shape for reporting
    original_shape = data.shape
    
    # Remove rows with NaN values (typically the first rows due to technical indicators)
    data_cleaned = data.dropna()

    # Remove the last 3 rows since they don't have valid targets (3-day look-ahead)
    if len(data_cleaned) > 3:
        data_cleaned = data_cleaned.iloc[:-3]

    
    # Keep only Date, technical indicators, and Target - drop original OHLCV data
    columns_to_keep = ['Date', 'Target']  # Always keep these
    
    # Add all technical indicator columns (everything except original OHLCV)
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in data_cleaned.columns:
        if col not in original_cols and col not in columns_to_keep:
            columns_to_keep.append(col)
    
    data_cleaned = data_cleaned[columns_to_keep]
    
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
    output_file = output_file.replace("_rawdata", "")
    
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
    rawdata_dir = "rawdata"
    data_dir = "data"
    
    files_to_process = [
        ("abb_rawdata.csv", "abb.csv"),
        ("addtech_rawdata.csv", "addtech.csv"),
        ("alfa_rawdata.csv", "alfa.csv"),
        ("assa_rawdata.csv", "assa.csv"),
        ("astrazeneca_rawdata.csv", "astrazeneca.csv"),
        ("atlascopco_rawdata.csv", "atlascopco.csv"),
        ("boliden_rawdata.csv", "boliden.csv"),
        ("epiroc_rawdata.csv", "epiroc.csv"),
        ("eqt_rawdata.csv", "eqt.csv"),
        ("ericsson_rawdata.csv", "ericsson.csv"),
        ("essity_rawdata.csv", "essity.csv"),
        ("evolution_rawdata.csv", "evolution.csv"),
        ("handelsbanken_rawdata.csv", "handelsbanken.csv"),
        ("hexagon_rawdata.csv", "hexagon.csv"),
        ("hmb_rawdata.csv", "hmb.csv"),
        ("industrivarden_rawdata.csv", "industrivarden.csv"),
        ("investor_rawdata.csv", "investor.csv"),
        ("lifco_rawdata.csv", "lifco.csv"),
        ("nibe_rawdata.csv", "nibe.csv"),
        ("nordea_rawdata.csv", "nordea.csv"),
        ("OMXS30_rawdata.csv", "OMXS30.csv"),
        ("saab_rawdata.csv", "saab.csv"),
        ("sandvik_rawdata.csv", "sandvik.csv"),
        ("sca_rawdata.csv", "sca.csv"),
        ("seb_rawdata.csv", "seb.csv"),
        ("skanska_rawdata.csv", "skanska.csv"),
        ("skf_rawdata.csv", "skf.csv"),
        ("swedbank_rawdata.csv", "swedbank.csv"),
        ("tele2_rawdata.csv", "tele2.csv"),
        ("telia_rawdata.csv", "telia.csv"),
        ("volvo_rawdata.csv", "volvo.csv"),
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
