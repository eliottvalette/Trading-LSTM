import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        for action, price, time in buy_sell_annotations:
            # Find the portfolio value at the time of the action
            if time in timestamps.values:
                idx = timestamps.get_loc(time)
                portfolio_value_at_action = portfolio_values[idx]
                
                if action == 'Buy':
                    plt.annotate('Buy', xy=(time, portfolio_value_at_action),
                                 xytext=(time, portfolio_value_at_action * 1.01),
                                 arrowprops=dict(facecolor='green', shrink=0.05),
                                 fontsize=12, color='green', ha='center')
                elif action == 'Sell':
                    plt.annotate('Sell', xy=(time, portfolio_value_at_action),
                                 xytext=(time, portfolio_value_at_action * 0.99),
                                 arrowprops=dict(facecolor='red', shrink=0.05),
                                 fontsize=12, color='red', ha='center')

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

def plot_confusion_matrix(predicted_orders, best_orders):
    """Plot confusion matrix comparing predicted and actual orders."""
    cm = confusion_matrix(predicted_orders, best_orders)
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Orders')
    plt.ylabel('Actual Orders')
    plt.title('Confusion Matrix')
    plt.savefig('logs/confusion_matrix.png')

def simulate_investment(model, dataloader, capital, shares_owned, scaler, buy_threshold, sell_threshold, test_df, backcandles, train_cols):
    model.eval()

    predicted_orders = []
    best_orders = []
    
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
        order_prediction_weighted = model(features).detach().numpy().flatten()
        order_prediction = order_prediction_weighted.argmax()
        true_best_order = targets.numpy().flatten().argmax()

        # Simulate investment based on model prediction
        if order_prediction == 0 and capital >= current_price:
            # Buy all the shares you can
            nb_share_affordable = int(capital // current_price)
            shares_owned += nb_share_affordable
            capital -= nb_share_affordable * current_price * 1.01  # Add 1% commission
            buy_sell_annotations.append(['Buy', current_price, timestamps[-1]])
            print(f"Bought {nb_share_affordable} share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On :{timestamps[-1]}")

        elif order_prediction == 2 and shares_owned > 0:
            # Sell all the shares you have
            capital += current_price * shares_owned * 0.99  # Subtract 1% commission
            shares_owned_temp = shares_owned
            shares_owned = 0
            buy_sell_annotations.append(['Sell', current_price, timestamps[-1]])
            print(f"Sold {shares_owned_temp} share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On :{timestamps[-1]}")

        # Calculate portfolio value after each decision
        portfolio_value = capital + shares_owned * current_price
        portfolio_values.append(portfolio_value)
        current_prices.append(current_price)

        # Store predicted and true orders
        predicted_orders.append(order_prediction)
        best_orders.append(true_best_order)

    # Convert timestamps to a datetime format
    timestamps = pd.to_datetime(timestamps)

    # Final Portfolio Summary
    print("Final Portfolio Value: {:.2f}".format(portfolio_values[-1]))
    print("Augmentation of the portfolio: {:.2%}".format((portfolio_values[-1] - initial_capital) / initial_capital))

    # Plot results
    plot_portfolio_value(timestamps, portfolio_values, buy_sell_annotations)
    plot_actual_stock_price(timestamps, current_prices)
    plot_confusion_matrix(predicted_orders, best_orders)

    buy_sell_annotations_df = pd.DataFrame(buy_sell_annotations, columns=['Action', 'Price', 'Timestamp'])

    print(buy_sell_annotations_df)
