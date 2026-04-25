# model.py — uses scikit-learn instead of TensorFlow (works on Python 3.14)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def predict_prices(df, days=30):
    data = df['Close'].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Create features — use last 30 days to predict next day
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1,1)).flatten()
    actual      = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    return actual, predictions, model, scaler