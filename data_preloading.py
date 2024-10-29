from config import Config
from data.data_utils import prepare_data, create_test_loader, training_loaders
from models.model import LSTMModel
from models.train import run_training
from models.evaluate import simulate_investment
from utils.visualisation import plot_training_metrics
from torch.optim import Adam, lr_scheduler
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

symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
timeframe = rest.TimeFrame(1, rest.TimeFrameUnit.Hour)
is_filter = False
limit = 4*364*24
is_training = True
backcandles = 60

df, dataset_scaled, train_sc, train_cols = prepare_data(
    symbol=symbol, 
    start_date=start_date, 
    end_date=end_date,
    timeframe=timeframe, 
    is_filter=is_filter, 
    limit= limit, 
    is_training=is_training,
    backcandles=backcandles)

df.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}.csv', index=False)
dataset_scaled.to_csv(f'data/preloads/{symbol}_{start_date}_{end_date}_{timeframe}_scaled.csv', index=False)
