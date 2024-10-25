import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import alpaca_trade_api.rest as rest
from torch.utils.data import TensorDataset, DataLoader

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_historical_data(symbol, timeframe, start_date, end_date, limit=10000):
    # Fetch data from Alpaca
    try:
        bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date, limit=limit).df
        bars['time'] = pd.to_datetime(bars.index)
        return bars[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error while fetching historical data: {e}")
        return None

def prepare_data(symbol, start_date, end_date, timeframe, is_filter=False, limit=4000, is_training=True, sc=None):
    df = get_historical_data(symbol, timeframe, start_date, end_date, limit)
    df['interval_evolution'] = df['close'].diff().shift(-1).fillna(0)
    
    # Scale data
    if is_training:
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(df[['open', 'high', 'low', 'close', 'volume', 'interval_evolution']])
    dataset_scaled = sc.transform(df[['open', 'high', 'low', 'close', 'volume', 'interval_evolution']])
    return df, dataset_scaled, sc

def create_test_loader(symbol, start_date, end_date, timeframe, backcandles, sc, is_filter, limit=1000):
    df, dataset_scaled, _ = prepare_data(symbol, start_date, end_date, timeframe, is_filter, limit, is_training=False, sc=sc)
    X_dataset = dataset_scaled[['open', 'high', 'low', 'close', 'volume']]
    y_dataset = dataset_scaled[['interval_evolution']].values
    # Sliding window feature set
    X, y = [], []
    for i in range(backcandles, len(X_dataset) - 1):
        window = [X_dataset.iloc[i-backcandles:i, j].values for j in range(X_dataset.shape[1])]
        X.append(np.array(window))
    X, y = np.array(X).transpose(0, 2, 1), np.array(y_dataset[1 + backcandles:, -1]).reshape(-1, 1)
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(test_dataset, batch_size=1, shuffle=False), df, dataset_scaled

def prepare_training_validation_data(dataset_scaled, backcandles, train_ratio=0.8):
    X_dataset = dataset_scaled[['open', 'high', 'low', 'close', 'volume']]
    y_dataset = dataset_scaled[['interval_evolution']].values

    # Create sliding window feature set
    X = []
    y = []
    for i in range(backcandles, len(X_dataset) - 1):
        window = []
        for j in range(X_dataset.shape[1]):
            window.append(X_dataset.iloc[i-backcandles:i, j].values)
        X.append(np.array(window))
    X = np.array(X)
    X = X.transpose(0, 2, 1)

    y = np.array(y_dataset[1 + backcandles:, -1])
    y = np.reshape(y, (len(y), 1))

    # Split data into training and validation datasets
    train_size = int(train_ratio * len(X))
    train_dataset = TensorDataset(torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(y[:train_size], dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(X[train_size:], dtype=torch.float32), torch.tensor(y[train_size:], dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader