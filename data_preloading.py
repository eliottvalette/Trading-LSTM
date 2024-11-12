from config import Config
from data.data_utils import get_historical_data
from data.features_engineering import features_engineering
import alpaca_trade_api.rest as rest
import torch.nn as nn
import pandas as pd

import torch
import numpy as np
import random as rd
import datetime

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rd.seed(seed)

config=Config()

symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']

def preload_historical_data(symbol) : 
    start_date = config.start_date
    end_date = config.end_date
    timeframe = config.timeframe
    limit = 4*364*24

    df = get_historical_data(
            symbol=symbol, 
            timeframe=timeframe,
            start_date=start_date, 
            end_date=end_date,
            limit= limit
            )

    df.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}.csv', index=False)

    # df_eng, _ = features_engineering(df, 30)
    # df_eng.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_yf_eng.csv', index=False)

for symbol in symbols:
    preload_historical_data(symbol)