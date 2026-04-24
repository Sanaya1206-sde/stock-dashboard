# indicators.py
import pandas as pd

def calculate_rsi(df, period=14):
    """
    RSI (Relative Strength Index) — tells if stock is
    overbought (>70) or oversold (<30).
    """
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df):
    """
    MACD — shows trend direction and momentum.
    When MACD crosses Signal line = buy/sell signal.
    """
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist']   = df['MACD'] - df['Signal']
    return df

def calculate_bollinger(df, period=20):
    """
    Bollinger Bands — shows price volatility.
    Price near upper band = expensive, lower band = cheap.
    """
    df['MA20']       = df['Close'].rolling(period).mean()
    df['Upper_Band'] = df['MA20'] + 2 * df['Close'].rolling(period).std()
    df['Lower_Band'] = df['MA20'] - 2 * df['Close'].rolling(period).std()
    return df