import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def plot_portfolio_value(timestamps, portfolio_values, buy_sell_annotations=None):
    """Plot portfolio value over time with optional annotations for buy/sell actions."""
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, portfolio_values, label='Portfolio Value', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value in USD')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)

    # Add buy/sell annotations if provided
    if buy_sell_annotations:
        for i, (action, price, time) in enumerate(buy_sell_annotations):
            color = 'green' if action == 'Buy' else 'red'
            plt.annotate(f"{action} @ {price:.2f}", 
                         (time, price),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center', 
                         color=color,
                         fontsize=8)

    plt.savefig('logs/portfolio.png')

def plot_actual_stock_price(timestamps, current_prices):
    """Plot actual stock price over time."""
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, current_prices, label='Actual Stock Price', color='green')
    plt.xlabel('Time')
    plt.ylabel('Stock Price in USD')
    plt.title('Actual Stock Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/stock.png')

def simulate_investment(model, dataloader, capital, shares_owned, scaler, buy_threshold, sell_threshold, test_df, backcandles, train_cols):
    model.eval()
    true_evolutions = []
    predicted_evolutions = []
    current_prices = []
    portfolio_values = []
    timestamps = []
    initial_capital = capital
    buy_sell_annotations = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        # Correct the timestamp index by adding the backcandles delay
        timestamps.append(test_df.iloc[step + backcandles, 0])

        current_price_index = train_cols.index('close')
        current_price_norm = features[:, -1, current_price_index].numpy().flatten()[0]
        
        # Create dummy arrays to inverse transform
        dummy_pred = np.zeros((1, len(train_cols) + 1))
        dummy_pred[:, current_price_index] = current_price_norm
        current_price = scaler.inverse_transform(dummy_pred)[:, current_price_index][0]


        # Predicted and true prices in normalized form
        order_prediction = model(features).detach().numpy().flatten().argmax()
        true_best_order = targets.numpy().flatten().argmax()

        # Simulate investment based on model prediction
        if order_prediction == 0 and capital >= current_price:
            # Buy one share if the predicted action is "Buy" and capital allows
            shares_owned += 1
            capital -= current_price * 1.01  # Add 1% commission
            buy_sell_annotations.append(['Buy', current_price, timestamps[-1]])
            print(f"Bought 1 share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On :{timestamps[-1]}")

        elif order_prediction == 2 and shares_owned > 0:
            # Sell one share if the predicted action is "Sell" and shares are available
            shares_owned -= 1
            capital += current_price
            buy_sell_annotations.append(['Sell', current_price, timestamps[-1]])
            print(f"Sold 1 share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On :{timestamps[-1]}")

        # Calculate portfolio value after each decision
        portfolio_value = capital + shares_owned * current_price
        portfolio_values.append(portfolio_value)
        current_prices.append(current_price)

    # Convert timestamps to a datetime format
    timestamps = pd.to_datetime(timestamps)

    # Final Portfolio Summary
    print("Final Portfolio Value: {:.2f}".format(portfolio_values[-1]))
    print("Augmentation of the portfolio: {:.2%}".format((portfolio_values[-1] - initial_capital) / initial_capital))

    # Plot results
    plot_portfolio_value(timestamps, portfolio_values, buy_sell_annotations)
    plot_actual_stock_price(timestamps, current_prices)

    buy_sell_annotations_df = pd.DataFrame(buy_sell_annotations, columns=['Action', 'Price', 'Timestamp'])

    print(buy_sell_annotations_df)
