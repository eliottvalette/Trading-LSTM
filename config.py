from datetime import datetime, timedelta
import alpaca_trade_api.rest as rest
import os

class Config:
    def __init__(self):
        self.symbol = 'NVDA'
        self.backcandles = 120
        self.timeframe = rest.TimeFrame(1, rest.TimeFrameUnit.Hour)
        self.start_date = '2023-01-01'
        self.start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        self.end_date = (self.start_date_dt + timedelta(days= 365 + 200)).strftime('%Y-%m-%d')
        self.test_start_date = '2024-08-01'
        self.test_end_date = '2024-10-01'
        self.initial_capital = 100
        self.shares_owned = 0
        self.num_epochs = 20
        self.buy_threshold = 0.02
        self.sell_threshold = -0.02