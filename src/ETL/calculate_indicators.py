import pandas as pd

def calculate_indicators(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data: SMA, MACD and RSI
    
    Args:
        ticker_data: DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
        
    Returns:
        DataFrame with added technical indicators
    """

    df = raw_data.copy()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Close_Change'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)

    df = df.drop(columns=['Date', 'EMA_12', 'EMA_26'])
    df = df.dropna(subset=['SMA_20', 'MACD', 'RSI', 'Close_Change'])

    features_to_scale = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'MACD', 'RSI'
    ]

    for feat in features_to_scale:
        min_val = df[feat].min()
        max_val = df[feat].max()
        df[feat] = (df[feat] - min_val) / (max_val - min_val)

    return df