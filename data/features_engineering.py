
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
        data['Lowest Low'] = data['low'].rolling(window=window).min()
        data['Highest High'] = data['high'].rolling(window=window).max()
        data['Stochastic'] = 100 * ((data['close'] - data['Lowest Low']) / (data['Highest High'] - data['Lowest Low']))
        return data['Stochastic']

def features_engineering(df, backcandles) :
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
    df['High-Low'] = df['high'] - df['low']
    df['High-close'] = (df['high'] - df['close'].shift()).abs()
    df['Low-close'] = (df['low'] - df['close'].shift()).abs()
    true_range = df[['High-Low', 'High-close', 'Low-close']].max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # 8
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    df.fillna(0, inplace=True)

    train_cols = [col for col in df.columns if col not in ['time', 'close_change']]

    return df, train_cols