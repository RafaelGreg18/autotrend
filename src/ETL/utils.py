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
    
def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column names in the DataFrame to ensure consistency.
    
    Args:
        df: DataFrame with columns to fix
    Returns:
        DataFrame with fixed column names
    """
    # removing the multi-index columns
    df.columns = [col[0] for col in df.columns]
    df['Date'] = df.index.astype(str)
    df.reset_index(drop=True, inplace=True)
    return df