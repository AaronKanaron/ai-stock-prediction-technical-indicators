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
        'OMXS30': '^OMX',
        

        "hmb": "HM-B.ST",
        "skf": "SKF-B.ST",
        "saab": "SAAB-B.ST",
        "nibe": "NIBE-B.ST",
        "boliden": "BOL.ST",
        "ericsson": "ERIC-B.ST",
        "alfa": "ALFA.ST",
        "essity": "ESSITY-B.ST",
        "telia": "TELIA.ST",
        "epiroc": "EPI-A.ST",
        "atlascopco": "ATCO-A.ST",
        "handelsbanken": "SHB-A.ST",
        "investor": "INVE-B.ST",
        "swedbank": "SWED-A.ST",
        "tele2": "TEL2-B.ST",
        "abb": "ABB.ST",
        "evolution": "EVO.ST",
        "sandvik": "SAND.ST",
        "sca": "SCA-B.ST",
        "skanska": "SKA-B.ST",
        "volvo": "VOLV-B.ST",
        "seb": "SEB-A.ST",
        "astrazeneca": "AZN",
        "assa": "ASSA-B.ST",
        "nordea": "NDA-SE.ST",
        "industrivarden": "INDU-C.ST",
        "lifco": "LIFCO-B.ST",
        "addtech": "ADDT-B.ST",
        "hexagon": "HEXA-B.ST",
        "eqt": "EQT.ST",
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
