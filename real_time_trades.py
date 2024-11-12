import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import alpaca_trade_api.rest as rest
from pytz import timezone

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API client
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Set timezone to Eastern Time (US markets)
us_eastern = timezone('America/New_York')

def get_intraday_data(symbol, timeframe, lookback_hours):
    # Calculate start time in US Eastern Time
    print("Current time in US Eastern Time is: ", datetime.now(us_eastern))
    start_time = (datetime.now(us_eastern) - timedelta(hours=lookback_hours)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_time} in US Eastern Time to the latest available")

    try:
        # Fetch data from Alpaca, only specifying start time
        bars = api.get_bars(symbol, timeframe, start=start_time).df
        
        # Ensure times are displayed in US Eastern Time
        bars.index = bars.index.tz_convert(us_eastern)
        bars['time'] = bars.index
        
        # Return only relevant columns
        return bars[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error while fetching historical data: {e}")
        return None

# Example usage
timeframe = rest.TimeFrame.Minute
intraday_data = get_intraday_data(symbol="AAPL", timeframe=timeframe, lookback_hours=30)
print(intraday_data)
