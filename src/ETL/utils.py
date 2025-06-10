import yfinance
from time import sleep
import logging
from pathlib import Path
import pandas as pd
from datetime import date

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_yfinance_data(ticker: str, start_date: str | None = None) -> dict:
    # Ensure data directory exists
    try:
        logger.info(f"Downloading data for {ticker}...")
        
        # Add delay to avoid hitting API limits
        sleep(1)

        if start_date is None:
            ticker_data = yfinance.download(ticker, period='max', interval='1d')
        else:
            ticker_data = yfinance.download(ticker, start=start_date, interval='1d')
        # removing the multi-index columns
        ticker_data.columns = [col[0] for col in ticker_data.columns]
        if ticker_data.empty:
            logger.warning(f"No data returned for {ticker}")
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
    
    return ticker_data


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of DataFrames into a single DataFrame.
    
    Args:
        dataframes: List of DataFrames to merge
    
    Returns:
        Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

def get_ticker_update_date(data_path: Path, ticker_name: str) -> str | None:
    file_path = data_path / f"{ticker_name}.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    if df.empty or 'Date' not in df.columns:
        return None

    last_date = df['Date'].iloc[-1]
    today = date.today().strftime('%Y-%m-%d')
    last_date_dt = pd.to_datetime(last_date)
    last_date_str = last_date_dt.strftime('%Y-%m-%d')
    
    # Check if the last date in the data is today or later
    if last_date_str == today:
        return None
    else:
        next_day = (last_date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Data for {ticker_name} is not up to date. Last date in data: {last_date_str}, today: {today}")
        return next_day
    
def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column names in the DataFrame to ensure consistency.
    
    Args:
        df: DataFrame with columns to fix
    Returns:
        DataFrame with fixed column names
    """
    df['Date'] = df.index.astype(str)
    df.reset_index(drop=True, inplace=True)
    return df