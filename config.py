from datetime import datetime, timedelta
import alpaca_trade_api.rest as rest
import os

# Get current Day
previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
two_month_ago = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
one_year_ago = (datetime.now() - timedelta(days=364)).strftime('%Y-%m-%d')

class Config:
    def __init__(self):
        self.symbol = 'AAPL'
        self.backcandles = 30
        self.timeframe = rest.TimeFrame.Minute
        self.start_date = two_month_ago
        self.end_date = three_days_ago
        self.test_start_date = self.end_date
        self.test_end_date = previous_day
        self.initial_capital = 10_000
        self.shares_owned = 0
        self.num_epochs = 4
        self.buy_threshold = 0.002
        self.sell_threshold = -0.002
        self.use_preloads = False