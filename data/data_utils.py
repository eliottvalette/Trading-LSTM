import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import alpaca_trade_api.rest as rest
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from data.features_engineering import features_engineering

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

def label_data(y, buy_threshold, sell_threshold):
    labels = []
    for change in y:
        labels.append( 1 if change > 0 else 0 )
    return labels


def prepare_data_from_preloads(final_symbol, timeframe, is_filter, is_training=True, sc=None, backcandles=60):
    final_df, final_dataset_scaled = None, None
    # Convert timeframe
    timeframe = str(timeframe)

    # First pass: find and fit the scaler on the final symbol
    for file in os.listdir('data/preloads'):
        if timeframe in file and final_symbol in file:
            df = pd.read_csv('data/preloads/' + file)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if is_filter:
                df = filter_close(df)

            # Categorize hours and calculate close change
            df['hour'] = df['time'].dt.hour
            df['close_pct_change'] = df['close'].diff().fillna(0).clip(lower=-10, upper=10)


            # Perform feature engineering
            df, train_cols = features_engineering(df, backcandles)
            print(train_cols)

            sc = MinMaxScaler()
            sc.fit(df[train_cols + ['close_pct_change']])

            final_df = df
            final_dataset_scaled = sc.transform(final_df[train_cols + ['close_pct_change']])
            final_dataset_scaled = pd.DataFrame(final_dataset_scaled, columns=train_cols + ['close_pct_change'], index=df['time'])

    # Second pass: apply the scaler to other files
    for file in os.listdir('data/preloads'):
        if timeframe in file and final_symbol not in file:
            df = pd.read_csv('data/preloads/' + file)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if is_filter:
                df = filter_close(df)

            # Categorize hours and calculate close change
            df['hour'] = df['time'].dt.hour
            df['close_pct_change'] = df['close'].diff().fillna(0).clip(lower=-10, upper=10)

            # Perform feature engineering
            df, _ = features_engineering(df, backcandles)
            new_scaled_df = sc.transform(df[train_cols + ['close_pct_change']])

            final_df = pd.concat([df, final_df])
            new_scaled_df = pd.DataFrame(new_scaled_df, columns=train_cols + ['close_pct_change'], index=df['time'])
            final_dataset_scaled = pd.concat([new_scaled_df, final_dataset_scaled])


    final_dataset_scaled = pd.DataFrame(final_dataset_scaled, columns=train_cols + ['close_pct_change'], index=final_df['time'])
    return final_df, final_dataset_scaled, sc, train_cols

def prepare_data(symbol, start_date, end_date, timeframe, is_filter=False, limit=4000, is_training=True, sc=None, backcandles=60):
    df = get_historical_data(symbol, timeframe, start_date, end_date, limit)
    if is_filter :
        # Filter to keep only open market hours
        df = filter_close(df)

    # Categorize Hours
    df['hour'] = df['time'].dt.hour

    df['close_pct_change'] = df['close'].pct_change().shift(-1).fillna(0).clip(lower=-10, upper=10)
    df, train_cols = features_engineering(df, backcandles)
    print(train_cols)

    # Scale data
    if is_training:
        sc = MinMaxScaler()
        sc.fit(df[train_cols + ['close_pct_change']])

    dataset_scaled = sc.transform(df[train_cols + ['close_pct_change']])
    dataset_scaled = pd.DataFrame(dataset_scaled, columns=train_cols + ['close_pct_change'], index= df['time'])

    return df, dataset_scaled, sc, train_cols

def training_loaders(dataframe, dataset_scaled, backcandles, train_cols, buy_threshold, sell_threshold, train_ratio = 0.95):
    X_dataset = dataset_scaled[train_cols]
    y_dataset = dataframe['close_pct_change'].values

    # Create sliding window feature set
    X, y = [], []
    for i in range(backcandles, len(X_dataset) - 1):
        window = [X_dataset.iloc[i-backcandles:i, j].values for j in range(X_dataset.shape[1])]
        X.append(np.array(window))
    X = np.array(X).transpose(0, 2, 1)
    
    y = label_data(y_dataset[backcandles :-1], buy_threshold, sell_threshold)
    y = np.array(y)

    print('close_pct_change :', dataframe['close_pct_change'].describe()) 
    print('buy_sell_label :', np.unique(y, return_counts=True))

    # Split data into training and validation datasets
    train_size = max(int(len(X) * train_ratio), len(X) - 2176)

    # Convert data to PyTorch DataLoader objects
    train_dataset = TensorDataset(torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(y[:train_size], dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(X[train_size:], dtype=torch.float32), torch.tensor(y[train_size:], dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def create_test_loader(dataframe, dataset_scaled, backcandles, train_cols, buy_threshold, sell_threshold):
    X_dataset = dataset_scaled[train_cols]
    y_dataset = dataframe['close_pct_change'].values

    # Create sliding window feature set
    X, y = [], []
    for i in range(backcandles, len(X_dataset) - 1):
        window = [X_dataset.iloc[i-backcandles:i, j].values for j in range(X_dataset.shape[1])]
        X.append(np.array(window))
    X = np.array(X).transpose(0, 2, 1)
    
    y = label_data(y_dataset[backcandles :-1], buy_threshold, sell_threshold)
    y = np.array(y)

    print('close_pct_change :', dataframe['close_pct_change'].describe())
    print('buy_sell_label :', np.unique(y, return_counts=True))

    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)