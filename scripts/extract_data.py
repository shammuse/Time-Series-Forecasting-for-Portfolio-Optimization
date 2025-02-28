import os
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(symbol, start_date, end_date):
    """
    Downloads, cleans, and saves historical stock data for a given symbol.
    
    Parameters:
    - symbol (str): Stock symbol (e.g., 'TSLA').
    - start_date (str): Start date for historical data (YYYY-MM-DD).
    - end_date (str): End date for historical data (YYYY-MM-DD).
    
    Returns:
    - None: Saves the data to a CSV file named `symbol_data.csv`.
    """
    try:
        logging.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        
        # Download the historical data
        df = yf.download(symbol, start=start_date, end=end_date)
        
        # Reset the index to ensure the 'Date' column is saved as a column, not as an index
        df.reset_index(inplace=True)
        
        # Clean the data (fill missing values, drop nulls, etc.)
        df.fillna(method='ffill', inplace=True)  # Forward fill missing data
        df.dropna(inplace=True)  # Drop any remaining missing data
        
        # Create the ../data folder if it doesn't exist
        data_folder = '../data'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        # Save the data to a CSV file in the ../data folder
        file_name = f"{data_folder}/{symbol}_data.csv"  # Save in the ../data folder
        df.to_csv(file_name, index=False)
        
        logging.info(f"Data for {symbol} saved successfully to {file_name}")
        print(f"Data for {symbol} downloaded and saved as {file_name}")
        
    except Exception as e:
        logging.error(f"Error processing data for {symbol}: {e}")
        print(f"Error processing data for {symbol}: {e}")

# Define your variables
symbols = ['TSLA', 'BND', 'SPY']  # List of stock symbols
start_date = '2015-01-01'  # Start date
end_date = '2025-01-31'  # End date

# Loop through each symbol and call the function
for symbol in symbols:
    get_data(symbol, start_date, end_date)