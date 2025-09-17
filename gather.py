import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def gather_stock_data():
    """
    Gather 10 years of daily stock data for OMXS30 and S&P 500
    """
    # Define the stock symbols
    symbols = {
        # 'OMXS30': '^OMX',
        # 'SP500': '^GSPC',
        # Big tech companies
        # 'AAPL': 'AAPL',
        # 'MSFT': 'MSFT',
        "volvo": "VOLV-B.ST",
        "investor": "INVE-A.ST",
        "atlascopco": "ATCO-A.ST",
        "ericsson": "ERIC-B.ST",
        "hmb": "HM-B.ST",
        "swedbank": "SWED-A.ST",
        "handelsbanken": "SHB-A.ST",
        "seb": "SEB-A.ST",
        "nordea": "NDA-SE.ST",
        "sandvik": "SAND.ST",
        "skf": "SKF-B.ST",
        "electrolux": "ELUX-B.ST",
        "telia": "TELIA.ST",
        "getinge": "GETI-B.ST",
    }
    
    # Calculate date range (10 years from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)  # Approximately 10 years
    
    print(f"Gathering data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create data directory if it doesn't exist
    data_dir = "rawdata"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Gather data for each symbol
    for name, symbol in symbols.items():
        print(f"\nFetching data for {name} ({symbol})...")
        
        try:
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                print(f"No data found for {symbol}")
                continue
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Select only the required columns: Date, Open, High, Low, Close, Volume
            columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[columns_to_keep]
            
            # Format the Date column
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            # Save to CSV file
            filename = f"{data_dir}/{name}_rawdata.csv"
            data.to_csv(filename, index=False)
            
            print(f"Successfully saved {len(data)} records to {filename}")
            print(f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
            
            # Display first few rows as preview
            print(f"\nPreview of {name} data:")
            print(data.head())
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

if __name__ == "__main__":
    print("Starting stock data collection...")
    gather_stock_data()
    print("\nData collection completed!")
