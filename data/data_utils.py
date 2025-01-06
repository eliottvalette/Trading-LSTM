import os
import torch
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import alpaca_trade_api.rest as rest
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from data.features_engineering import features_engineering
from sklearn.preprocessing import MinMaxScaler
import numpy as np


load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY_V3')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY_V3')
BASE_URL = 'https://data.alpaca.markets/v2/stocks/bars'

print(f"API_KEY:{API_KEY} SECRET_KEY:{SECRET_KEY}" )

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

def filter_close(df):
    start_date = min(df['time'])
    end_date = max(df['time'])
    filtered_rows = []

    for date in pd.date_range(start=start_date, end=end_date):
        date_str = date.strftime('%Y-%m-%d')

        try:
            calendar = api.get_calendar(date_str)[0]
            open_time, close_time = calendar.open, calendar.close
            for index, row in df.iterrows():
                if row['time'].date() == date.date() and open_time <= row['time'].time() <= close_time:
                    filtered_rows.append(row)
        except IndexError:
            # Handle case when the market was closed on a specific date (e.g., weekend or holiday)
            continue
    
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df.index = np.arange(len(filtered_df))
    
    return filtered_df

def prepare_data_from_preloads(final_symbol, timeframe, is_filter, backcandles=60):
    final_df, final_df_scaled = None, None
    # Convert timeframe
    timeframe = str(timeframe)

    # First pass: find and fit the scaler on the final symbol
    for file in os.listdir('data/preloads'):
        if timeframe in file and final_symbol in file:
            df = pd.read_csv('data/preloads/' + file)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if is_filter:
                df = filter_close(df)

            # Perform feature engineering
            df, train_cols = features_engineering(df, backcandles)

            # Save df
            final_df = df

            final_df_scaled = df[train_cols + ['target']]
            final_df_scaled.index = final_df['time']


    # Second pass: apply the scaler to other files
    for file in os.listdir('data/preloads'):
        if timeframe in file and final_symbol not in file:
            df = pd.read_csv('data/preloads/' + file)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if is_filter:
                df = filter_close(df)

            # Perform feature engineering
            df, train_cols = features_engineering(df, backcandles)

            # Save df
            final_df = pd.concat([df, final_df])

            new_scaled_df = df[train_cols + ['target']]
            new_scaled_df.index = df['time']
            final_df_scaled = pd.concat([new_scaled_df, final_df_scaled])

    final_df_scaled = pd.DataFrame(final_df_scaled, columns=train_cols + ['target'], index=final_df['time'])

    return final_df, final_df_scaled, train_cols

def prepare_data(symbol, start_date, end_date, timeframe, is_filter=False, backcandles=60):
    df = get_historical_data(symbol, timeframe, start_date, end_date)

    if is_filter :
        # Filter to keep only open market hours
        df = filter_close(df)
    
    df, train_cols = features_engineering(df, backcandles)
    print(train_cols)
    
    df_scaled = df[train_cols + ['target']]
    df_scaled.index = df['time']

    return df, df_scaled, train_cols

def training_loaders(dataframe, dataset_scaled, backcandles, train_cols, buy_threshold, sell_threshold, train_ratio=0.95):
    X_dataset = dataset_scaled[train_cols]
    y_dataset = dataframe['target'].values

    # Create sliding window feature set
    X, y = [], []
    for i in range(backcandles, len(X_dataset) - 1):  # Adjust to len(X_dataset) - 1 to prevent index out of bounds
        window = X_dataset.iloc[i - backcandles : i].values
        X.append(window)
        y.append(y_dataset[i - 1])  # Ensure each label aligns with the end of its window

    X = np.array(X) # Match now [batch_size, seq_len, num_feat]
    y = np.array(y)

    # Split data into training and validation datasets
    batch_size = 16
    train_size = len(X) - 30 * batch_size

    # Convert data to PyTorch DataLoader objects
    train_dataset = TensorDataset(torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(y[:train_size], dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(X[train_size:], dtype=torch.float32), torch.tensor(y[train_size:], dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def create_test_loader(dataframe, dataset_scaled, backcandles, train_cols):
    X_dataset = dataset_scaled[train_cols]
    y_dataset = dataframe['target'].values

    # Create sliding window feature set
    X, y = [], []
    for i in range(backcandles, len(X_dataset) - 1):  # Adjust to len(X_dataset) - 1 to prevent index out of bounds
        window = X_dataset.iloc[i - backcandles : i].values
        X.append(window)
        y.append(y_dataset[i - 1])  # Ensure each label aligns with the end of its window

    X = np.array(X) # Match now [batch_size, seq_len, num_feat]
    y = np.array(y)

    print('target descritpion :', dataframe['target'].describe())

    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)