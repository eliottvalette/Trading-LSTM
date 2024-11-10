from config import Config
from data.data_utils import get_historical_data_alpaca, get_historical_data_yf
from data.features_engineering import features_engineering
import alpaca_trade_api.rest as rest
import torch.nn as nn
import pandas as pd

import torch
import numpy as np
import random as rd


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rd.seed(seed)

config=Config()

symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'FB', 'NVDA', 'PYPL', 'ADBE', 'NFLX']

def preload_historical_data(symbol) : 
    start_date = '2023-01-01'
    end_date = '2024-08-01'
    timeframe = '15m'
    limit = 4*364*24

    if timeframe == '15m':
        timeframe_alpaca = '15Min'
        df = get_historical_data_alpaca(
            symbol=symbol, 
            timeframe=timeframe_alpaca,
            start_date=start_date, 
            end_date=end_date,
            limit=limit
        )

    else :
        df = get_historical_data_yf(
            symbol=symbol, 
            timeframe=timeframe,
            start_date=start_date, 
            end_date=end_date,
            limit= limit
            )

    df.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_yf.csv', index=False)

    # df_eng, _ = features_engineering(df, 30)
    # df_eng.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_yf_eng.csv', index=False)

for symbol in symbols:
    preload_historical_data(symbol)