from utils import download_yfinance_data, merge_dataframes, fix_columns
from calculate_indicators import calculate_indicators
import os
from pathlib import Path
import pandas as pd 

def get_ticker_data(ticker_name: str, start_date: str | None) -> None:
    """
    Downloads and stores raw data at data_lake and data with technical indicators at data_warehouse.
    If the ticker already exists in data_lake, it will merge the new data with the existing data.
    If the ticker does not exist, it will create a new file with the ticker name.

    Args:
        ticker_name (str): The name of the ticker to download data for.
        start_date (str | None): The start date for downloading data. If None, will download historical data from the beginning.
    """
    # receive rabbitmq message with ticker and start date to download data
    print(f"Received ticker: {ticker_name} with start date: {start_date}")

    data_warehouse = Path('/data_warehouse')
    data_lake = Path('/data_lake')

    ticker_csv = ticker_name + ".csv"

    new_data = download_yfinance_data(ticker=ticker_name, start_date=start_date)
    fixed_data = fix_columns(new_data)
    
    print(f"Data for {ticker_name} downloaded and stored in data lake.")

    # calculate technical indicators
    ticker_csv = ticker_name + ".csv"
    up_data_with_indicators = calculate_indicators(fixed_data)
    if ticker_csv in os.listdir(data_lake):
        # Merge with existing csv's, if they exist
        merged_raw = merge_dataframes([fixed_data, pd.read_csv(data_lake / ticker_csv)])
        merged_raw.to_csv(data_lake / ticker_csv, index=False)
        print(f"Data {ticker_name} merged and saved to data lake.")
        merged_indicators = merge_dataframes([up_data_with_indicators, pd.read_csv(data_warehouse / f"{ticker_name}_indicators.csv")])
        merged_indicators.to_csv(data_warehouse / f"{ticker_name}_indicators.csv", index=False)
        print(f"Data {ticker_name} merged and saved to data warehouse.")
    else:
        print(f'Data {ticker_name} not found in data lake, saving new data...')
        # Save the new data to a csv file
        up_data_with_indicators.to_csv(data_lake / ticker_csv, index=False)
        up_data_with_indicators.to_csv(data_warehouse / f"{ticker_name}_indicators.csv", index=False)
        print(f"Data {ticker_name} saved to data lake and indicators to data warehouse.")


if __name__ == "__main__":
    # Example usage
    ticker_name = "AAPL"
    start_date = "2020-01-01"
    get_ticker_data(ticker_name, start_date)
    ticker_name = "RACE"
    start_date = None
    get_ticker_data(ticker_name, start_date)
    print("ETL process completed.")