import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest
)
from alpaca.trading.enums import (
    AssetStatus,
    OrderSide,
    OrderType,
    TimeInForce
)
from alpaca.common.exceptions import APIError

import yfinance as yf


import os
import json
import torch
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca trading and historical data clients
trade_client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
historical_data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Get account information
account = trade_client.get_account()
print(f"${account.buying_power} is available as buying power.")
balance_change = float(account.equity) - float(account.last_equity)
print(f"Today's portfolio balance change: ${balance_change}")

# Check if AAPL is tradable
aapl_asset = trade_client.get_asset('AAPL')
if aapl_asset.tradable:
    print('We can trade AAPL.')

# Get list of assets which are options enabled
req = GetAssetsRequest(
    attributes="options_enabled"
)
assets = trade_client.get_all_assets(req)
print("Options-enabled assets:", assets[:2])

# Define functions for buy, sell, data fetching, and trading logic
def buy(symbol, qty):
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trade_client.submit_order(order_request)
        print("Buy order submitted successfully.")
        return order
    except APIError as e:
        print(f"Error submitting buy order: {e}")

def sell(symbol, qty):
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trade_client.submit_order(order_request)
        print("Sell order submitted successfully.")
        return order
    except APIError as e:
        print(f"Error submitting sell order: {e}")

def get_recent_data(symbol, lookback=30, interval="1d"):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)

        # Fetch data from Yahoo Finance
        recent_bars = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        print("Fetched recent bars:", recent_bars)

        # Convert to DataFrame and prepare columns
        recent_bars.reset_index(drop=False, inplace=True)
        
        # Sample feature engineering (modify based on actual data structure)
        X = torch.tensor(recent_bars.iloc[-lookback:].values.reshape(1, lookback, -1), dtype=torch.float32)

        return X
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def get_model_prediction(symbol):
    X = get_recent_data(symbol)
    if X is None:
        print("Error: Could not fetch data for prediction.")
        return "hold", 0.0

    # Example prediction logic
    with torch.no_grad():
        prediction = 0.2  # Placeholder prediction

    buy_threshold = 0.6
    sell_threshold = 0.4

    if prediction >= buy_threshold:
        action = 'buy'
    elif prediction <= sell_threshold:
        action = 'sell'
    else:
        action = 'hold'

    return action, prediction

def trade(symbol, qty):
    action, prediction = get_model_prediction(symbol)
    print(f"Prediction for {symbol}: {prediction:.2f} -> Action: {action}")

    if action == 'buy':
        print(f"Buying {qty} shares of {symbol}...")
        # buy(symbol, qty)
    elif action == 'sell':
        print(f"Selling {qty} shares of {symbol}...")
        # sell(symbol, qty)
    else:
        print("Holding position, no action taken.")

# Example usage
trade('AAPL', qty=1)