import pandas as pd
import numpy as np
def calculate_rsi(data, window=14):
        delta = data['close_pct_change'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return rsi

def stochastic_oscillator(data, window=14):
    lowest_low = data['low_pct_diff'].rolling(window=window).min()
    highest_high = data['high_pct_diff'].rolling(window=window).max()
    stochastic = ((data['close_pct_change'] - lowest_low) / (highest_high - lowest_low))
    return stochastic

def calculate_macd(data, short_window=12, long_window=26):
        macd = data['close_pct_change'].ewm(span=short_window, adjust=False).mean() * 1_000 - data['close_pct_change'].ewm(span=long_window, adjust=False).mean() * 1_000
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return macd , signal_line 

def calculate_atr(data, window=14):
    high_low = data['high_pct_diff'] - data['low_pct_diff']
    high_close = (data['high_pct_diff'] - data['close_pct_change'].shift()).abs()
    low_close = (data['low_pct_diff'] - data['close_pct_change'].shift()).abs()
    
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close})
    
    atr = true_range.max(axis=1).rolling(window=window).mean()
    return atr
       

def features_engineering(df, backcandles):
    # Baseline
    df['close_pct_change'] = df['close'].pct_change().fillna(0).clip(lower=-10, upper=10)

    # Open
    df['open_pct_diff'] = (df['open'] - df['close']) / df['close'] * 100

    # High 
    df['high_pct_diff'] = (df['high'] - df['close']) / df['close'] * 100
    
    # Low 
    df['low_pct_diff'] = (df['low'] - df['close']) / df['close'] * 100
    
    # Volume :
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    
    # EMA
    df['EMA'] = df['close_pct_change'].ewm(span=backcandles, adjust=False).mean() * 1_000

    # RSI
    df['RSI'] = calculate_rsi(df)

    # MACD
    df['MACD'], df['Signal Line'] = calculate_macd(df)

    # Bollinger Bands
    df['Upper Band'] = df['EMA'] + 2 * df['close_pct_change'].rolling(window=20).std()
    df['Lower Band'] = df['EMA'] - 2 * df['close_pct_change'].rolling(window=20).std()

    # Stochastic Oscillator
    df['Stochastic'] = stochastic_oscillator(df)

    # ATR
    df['ATR'] = calculate_atr(df)

    # Target
    df['target'] = df['close_pct_change'].shift(-1)

    # Small adjustments
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # Drop value dependant columns
    df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    
    train_cols = [col for col in df.columns if col not in ['time', 'target'] + ['close']] # Obviously to drop + [maybe noisy]
    
    return df, train_cols