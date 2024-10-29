import pandas as pd
def calculate_rsi(data, window=14):
        delta = data['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def stochastic_oscillator(data, window=14):
        lowest_low = data['low'].rolling(window=window).min()
        highest_high = data['high'].rolling(window=window).max()
        stochastic = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return stochastic

def features_engineering(df, backcandles):
    # 1
    df['SMA'] = df['close'].rolling(window=backcandles).mean().fillna(0)

    # 2
    df['EMA'] = df['close'].ewm(span=backcandles, adjust=False).mean()

    # 3
    df['RSI'] = calculate_rsi(df)

    # 4
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = short_ema - long_ema
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5
    df['Upper Band'] = df['SMA'] + 2 * df['close'].rolling(window=20).std()
    df['Lower Band'] = df['SMA'] - 2 * df['close'].rolling(window=20).std()

    # 6
    df['Stochastic'] = stochastic_oscillator(df)

    # 7
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close})
    df['ATR'] = true_range.max(axis=1).rolling(window=14).mean()

    df.fillna(0, inplace=True)

    train_cols = [col for col in df.columns if col not in ['time', 'volume', 'close_pct_change']]
    
    return df, train_cols