import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import alpaca_trade_api.rest as rest
from pytz import timezone
from models.models import LSTMModel, CNNModel, GradBOOSTModel, EnsemblingModel, DirectionalMSELoss
from torch.utils.data import TensorDataset, DataLoader
from data.features_engineering import features_engineering
from config import Config
import torch
import numpy as np
import time


# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API client
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Set timezone to Eastern Time (US markets)
us_eastern = timezone('America/New_York')

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device : {device} \n")

# Set up constants and configurations
config = Config()

def load_ensemble_model(lstm_path, cnn_path, gradboost_path, embedding_dim, hidden_dim, num_layers, dropout_prob, num_leaves, max_depth, learning_rate, n_estimators):
    lstm_model = LSTMModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob)
    cnn_model = CNNModel(embedding_dim=embedding_dim, dropout_prob=dropout_prob)
    
    # Use weights_only=True to mitigate warning and security risk
    lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
    cnn_model.load_state_dict(torch.load(cnn_path, weights_only=True))

    # Move models to device
    lstm_model = lstm_model.to(device)
    cnn_model = cnn_model.to(device)

    # Load GradBOOST model
    gradboost_model = GradBOOSTModel(num_leaves, max_depth, learning_rate, n_estimators)
    gradboost_model.load(gradboost_path)

    return lstm_model, cnn_model, gradboost_model


def get_intraday_data(symbol, timeframe, lookback):
    # Calculate start time in US Eastern Time
    print("Current time in US Eastern Time is: ", datetime.now(us_eastern))
    start_time = (datetime.now(us_eastern) - timedelta(hours=lookback * 3)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_time} in US Eastern Time to the latest available")

    try:
        # Fetch data from Alpaca, only specifying start time
        bars = api.get_bars(symbol, timeframe, start=start_time).df
        
        # Ensure times are displayed in US Eastern Time
        bars.index = bars.index.tz_convert(us_eastern)
        bars['time'] = bars.index
        
        # Return only relevant columns
        return bars[['time', 'open', 'high', 'low', 'close', 'volume']][-30:] # keep last 30 rows
    except Exception as e:
        print(f"Error while fetching historical data: {e}")
        return None

# Function to predict action based on the model and Alpaca’s 15-minute delayed data
def predict_action(data, model):
    prediction = model.predict(data)
    return prediction

# Function to confirm trade based on real-time last price before executing
def auto_trade(symbol, model, trade_allocation, lookback):
    timeframe = rest.TimeFrame.Minute

    # Get delayed data for the model to analyze
    data = get_intraday_data(symbol, timeframe, lookback)
    data_df, train_cols = features_engineering(data, lookback)

    data_df_scaled = data_df[train_cols]

    data_list = []
    data_list.append(data_df_scaled.iloc[0 : 30].values)
    data_list = np.array(data_list)
    data_dataset = TensorDataset(torch.tensor(data_list, dtype=torch.float32))
    data_loader = DataLoader(data_dataset, batch_size=1, shuffle=False)
    
    for features in data_loader :
        features = features[0].to(device)  # Access the tensor inside the list and move it to the device
        action = model(features).cpu().detach().numpy().flatten()[0]

        # Determine the amount to trade based on trade_allocation and current capital
        account = api.get_account()
        capital = float(account.buying_power)
        trade_amount = round(capital * trade_allocation, 2)

        # Check existing position
        positions = api.list_positions()
        current_position = next((p for p in positions if p.symbol == symbol), None)
        current_side = current_position.side if current_position else None

        print('\npositions : ', positions)

        if action == 1 and (current_side != 'long'):
            # If previous action was short or no position, close existing short position
            if current_side == 'short':
                api.close_position(symbol)

            # Extract the current price
            current_price = api.get_latest_trade('AAPL').price

            print(f"Buying ${trade_amount:.2f} of {symbol} at {current_price}")

            api.submit_order(
                symbol=symbol,
                notional=trade_amount,
                side='buy',
                type='market',
                time_in_force='day'
            )

        elif action == 0 and (current_side != 'short'):
            # If previous action was buy or no position, close existing long position
            if current_side == 'long':
                api.close_position(symbol)

            # Extract the current price
            current_price = api.get_latest_trade('AAPL').price

            print(f"Shorting ${trade_amount:.2f} of {symbol} at {current_price}")

            api.submit_order(
                symbol=symbol,
                notional=trade_amount, 
                side='sell',
                type='market',
                time_in_force='day'
            )    


# Load individual models
lstm_model, cnn_model, gradboost_model = load_ensemble_model(
    lstm_path="saved_weights/Best_LSTM_F1_0.6813.pth",
    cnn_path="saved_weights/Best_CNN_F1_0.4095.pth",
    gradboost_path="saved_weights/LGBM_F1_0.5720.txt",
    embedding_dim=13,
    hidden_dim=128,
    num_layers=3, 
    dropout_prob=0.2,
    num_leaves=256, 
    max_depth=5, 
    learning_rate=0.05, 
    n_estimators=1000
)

# Initialize the ensemble
ensemble_model = EnsemblingModel(lstm_model, cnn_model, gradboost_model)
ensemble_model.to(device)

past_time = ''
previous_action = -1
while True :
    current_time = datetime.now(us_eastern).strftime('%Y-%m-%d %H %M')
    if past_time != current_time :
        past_time = current_time
        auto_trade(symbol="AAPL", model=ensemble_model, trade_allocation=0.01, lookback = 30)
    
    time.sleep(2)