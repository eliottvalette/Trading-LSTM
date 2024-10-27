from data.data_utils import prepare_data
import alpaca_trade_api.rest as rest


symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2021-12-31'
timeframe = rest.TimeFrame(1, rest.TimeFrameUnit.Hour)
is_filter = False,
limit = 365 * 24 * 4
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