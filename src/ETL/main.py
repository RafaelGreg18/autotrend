from utils import check_ticker_last_date, download_yfinance_data, merge_dataframes, fix_columns
from calculate_indicators import calculate_indicators
import os
from pathlib import Path
import pandas as pd 

def main():
    # For each ticker in the list of tickers:
    tickers = ['RACE', 'AML.L', 'MBG.DE']
    data_warehouse = Path('/data_warehouse')
    data_lake = Path('/data_lake')
    update_dates = [check_ticker_last_date(data_path=data_lake, ticker_name=ticker) for ticker in tickers]
    updated_data = []
    for ticker, update_date in zip(tickers, update_dates):
        new_data = download_yfinance_data(ticker=ticker, start_date=update_date)
        fixed_columns_data = fix_columns(new_data)
        updated_data.append(fixed_columns_data) 
    print("Data download completed.")
    # calculate technical indicators

    for ticker, up_data in zip(tickers, updated_data):
        ticker_csv = ticker + ".csv"
        up_data_with_indicators = calculate_indicators(up_data)
        if ticker_csv in os.listdir(data_lake):
            # Merge with existing csv's, if they exist
            merged_raw = merge_dataframes([up_data, pd.read_csv(data_lake / ticker_csv)])
            merged_raw.to_csv(data_lake / ticker_csv, index=False)
            print(f"Data {ticker} merged and saved to data lake.")
            merged_indicators = merge_dataframes([up_data_with_indicators, pd.read_csv(data_warehouse / f"{ticker}_indicators.csv")])
            merged_indicators.to_csv(data_warehouse / f"{ticker}_indicators.csv", index=False)
            print(f"Data {ticker} merged and saved to data warehouse.")
        else:
            print(f'Data {ticker} not found in data lake, saving new data...')
            # Save the new data to a csv file
            up_data.to_csv(data_lake / ticker_csv, index=False)
            up_data_with_indicators.to_csv(data_warehouse / f"{ticker}_indicators.csv", index=False)
            print(f"Data {ticker} saved to data lake and indicators to data warehouse.")


    for df in os.listdir(data_lake):
        # remove suffix .csv
        ticker_name = df.removesuffix('.csv')
        print(f"\nProcessing indicators for {ticker_name}...")
        # Calculate technical indicators
        calculate_indicators(ticker_raw_data_path=data_lake,
                             data_warehouse_path=data_warehouse,
                             ticker_name=ticker_name)
        print(f"Indicators for {ticker_name} processed and saved to {data_warehouse / f'{ticker_name}_indicators.csv'}")
        
if __name__ == "__main__":
    main()