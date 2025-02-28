import yfinance as yf
import logging

# Configure logging
logging.basicConfig(filename='../log/data_processing.log', level=logging.INFO, 
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
        
        # Save the data to a CSV file with the format symbol_data.csv
        file_name = f"../data/{symbol}_data.csv"
        df.to_csv(file_name, index=False)
        
        logging.info(f"Data for {symbol} saved successfully to ../data/{file_name}")
        print(f"Data for {symbol} downloaded and saved as ../data/{file_name}")
        
    except Exception as e:
        logging.error(f"Error processing data for {symbol}: {e}")
        print(f"Error processing data for {symbol}: {e}")