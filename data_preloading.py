from config import Config
from data.data_utils import get_historical_data, get_historical_data_yf
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

symbol = 'NVDA'
start_date = '2023-01-01'
end_date = '2024-01-01'
timeframe = '1h'
is_filter = False
limit = 4*364*24
backcandles = config.backcandles

df = get_historical_data_yf(
    symbol=symbol, 
    timeframe=timeframe,
    start_date=start_date, 
    end_date=end_date,
    limit= limit)

df.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_yf.csv', index=False)
# dataset_scaled.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_scaled.csv', index=False)
