import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_portfolio_value(timestamps, portfolio_values, annotations_df, add_annotation = False):
    """Plot portfolio value over time with optional annotations for buy/sell actions."""
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, portfolio_values, label='Portfolio Value', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value in USD')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)

    # Add buy/sell annotations if provided
    if len(annotations_df) > 0 and add_annotation:
        for action, portfolio_value, time in annotations_df.to_records(index=False):
            arrowprops = dict(facecolor='green' if action == 'Buy' else 'red',
                              shrink = 0.0005,
                              headlength = 6,
                              headwidth = 6
                              )
            if action == 'Buy':
                plt.annotate('', xy=(time, portfolio_value + 9),
                             xytext=(time, portfolio_value + 10), # to flip the arrow upside down
                             arrowprops=arrowprops,
                             fontsize=12, color='green', ha='center', zorder=5)
            elif action == 'Sell':
                plt.annotate('', xy=(time, portfolio_value),
                             xytext=(time, portfolio_value + 1),
                             arrowprops=arrowprops,
                             fontsize=12, color='red', ha='center', zorder=5)

    plt.savefig('logs/portfolio.png', dpi=300)
    plt.close()

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

def simulate_investment(model, dataloader, capital, shares_owned, scaler, buy_threshold, sell_threshold, test_df, backcandles, train_cols, device, trade_allocation = 0.1):
    model.eval()

    predicted_orders = []
    best_orders = []
    
    current_prices = []
    portfolio_values = []
    timestamps = []
    initial_capital = capital
    buy_sell_annotations = []

    commission = 0.00

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        features = features.to(device)
        targets = targets.to(device)

        # Correct the timestamp index by adding the backcandles delay
        timestamps.append(test_df.iloc[step + backcandles, 0])

        current_price_index = train_cols.index('close')
        current_price_norm = features[:, -1, current_price_index].cpu().detach().numpy().flatten()[0]

        # Create dummy arrays to inverse transform
        dummy_pred = np.zeros((1, len(train_cols) + 1))
        dummy_pred[:, current_price_index] = current_price_norm
        current_price = scaler.inverse_transform(dummy_pred)[:, current_price_index][0]

        # Predicted and true prices in normalized form
        order_prediction_weighted = model(features).cpu().detach().numpy().flatten()
        order_prediction = order_prediction_weighted.argmax()
        true_best_order = targets.cpu().numpy().flatten().argmax()

        # Simulate investment based on model prediction
        if order_prediction == 0 and capital >= current_price :
            # Calculate position size based on trade allocation
            investment_amount = capital * trade_allocation
            nb_share_affordable = int(investment_amount // current_price)
            if nb_share_affordable > 0:
                shares_owned += nb_share_affordable
                total_cost = nb_share_affordable * current_price * (1 + commission)
                capital -= total_cost
                portfolio_value_temp = capital + shares_owned * current_price
                buy_sell_annotations.append(['Buy', portfolio_value_temp, timestamps[-1]])
                # print(f"Bought {nb_share_affordable} shares at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On: {timestamps[-1]}")

        elif order_prediction == 2 and shares_owned > 0:
            # Sell all shares
            total_revenue = shares_owned * current_price * (1 - commission)
            capital += total_revenue
            shares_sold = shares_owned
            shares_owned = 0
            portfolio_value_temp = capital
            buy_sell_annotations.append(['Sell', portfolio_value_temp, timestamps[-1]])
            # print(f"Sold {shares_sold} shares at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On: {timestamps[-1]}")

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

    buy_sell_annotations_df = pd.DataFrame(buy_sell_annotations, columns=['Action', 'Portfolio', 'Timestamp'])
    buy_sell_annotations_df['Timestamp'] = pd.to_datetime(buy_sell_annotations_df['Timestamp'])
    buy_sell_annotations_df.to_csv('logs/buy_sell_annotations.csv')

    # Plot results
    plot_portfolio_value(timestamps, portfolio_values, buy_sell_annotations_df)
    plot_actual_stock_price(timestamps, current_prices)
    plot_confusion_matrix(predicted_orders, best_orders)

    print(buy_sell_annotations_df)
