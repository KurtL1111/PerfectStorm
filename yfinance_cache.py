import yfinance as yf

def get_stock_history(symbol, interval='1d', period='1y'):
    """
    Get stock history data using yfinance
    
    Parameters:
    - symbol: Stock symbol
    - interval: Data interval (1d, 1wk, 1mo, etc.)
    - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
    - history: DataFrame with stock history
    """
    return yf.download(symbol, interval=interval, period=period)
