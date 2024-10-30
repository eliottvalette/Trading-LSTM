from datetime import datetime, timedelta
import alpaca_trade_api.rest as rest
import os

class Config:
    def __init__(self):
        self.symbol = 'AAPL'
        self.backcandles = 30
        self.timeframe = rest.TimeFrame(1, rest.TimeFrameUnit.Hour)
        self.start_date = '2023-01-01'
        self.end_date = '2024-05-01'
        self.test_start_date = self.end_date
        self.test_end_date = '2024-10-01'
        self.initial_capital = 10_000
        self.shares_owned = 0
        self.num_epochs = 20
        self.buy_threshold = 0.002
        self.sell_threshold = -0.002
        self.decision_threshold = 0.5
        self.use_preloads = False