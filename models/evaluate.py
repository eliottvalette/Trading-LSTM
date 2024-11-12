import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_portfolio_value(timestamps, portfolio_values, annotations_df, model_name, add_annotation = True):
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
        for action, portfolio_value, price, time in annotations_df.to_records(index=False):
            arrowprops = dict(facecolor='green' if action == 'Buy' else 'red',
                              shrink = 0.0005,
                              headlength = 6,
                              headwidth = 6
                              )
            if action == 'Buy':
                plt.annotate('', xy=(time, portfolio_value),
                             xytext=(time, portfolio_value + 2), # to flip the arrow upside down
                             arrowprops=arrowprops,
                             fontsize=12, color='green', ha='center', zorder=5)
            elif action == 'Sell':
                plt.annotate('', xy=(time, portfolio_value),
                             xytext=(time, portfolio_value + 1),
                             arrowprops=arrowprops,
                             fontsize=12, color='red', ha='center', zorder=5)

    plt.savefig(f'logs/{model_name}_portfolio.png', dpi=300)
    plt.close()

def plot_actual_stock_price(timestamps, current_prices, annotations_df, model_name, add_annotation = True):
    """Plot actual stock price over time."""
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, current_prices, label='Actual Stock Price', color='green')
    plt.xlabel('Time')
    plt.ylabel('Stock Price in USD')
    plt.title('Actual Stock Price Over Time')
    plt.legend()
    plt.grid(True)
    # Add buy/sell annotations if provided
    if len(annotations_df) > 0 and add_annotation:
        for action, portfolio_value, price, time in annotations_df.to_records(index=False):
            arrowprops = dict(facecolor='green' if action == 'Buy' else 'red',
                              shrink = 0.0005,
                              headlength = 6,
                              headwidth = 6
                              )
            if action == 'Buy':
                plt.annotate('', xy=(time, price + 0.1),
                             xytext=(time, price + 0.2), # to flip the arrow upside down
                             arrowprops=arrowprops,
                             fontsize=12, color='green', ha='center', zorder=5)
            elif action == 'Sell':
                plt.annotate('', xy=(time, price + 0.1),
                             xytext=(time, price + 0.2),
                             arrowprops=arrowprops,
                             fontsize=12, color='red', ha='center', zorder=5)
                
    plt.savefig(f'logs/{model_name}_annotated_stock.png')

def plot_confusion_matrix(predicted_orders, best_orders, title, model_name):
    """Plot confusion matrix comparing predicted and actual orders."""
    cm = confusion_matrix(best_orders, predicted_orders)
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Orders')
    plt.ylabel('Actual Orders')
    plt.title('Confusion Matrix')
    plt.savefig(f'logs/{model_name}_confusion_matrix_evaluate_{title}.png')

def simulate_investment(model, dataloader, capital, shares_owned, test_df, backcandles, train_cols, trade_decision_threshold, device, model_name, trade_allocation=0.1):
    model.lstm_model.eval()
    model.cnn_model.eval()

    predicted_orders = []
    best_orders = []

    current_prices = []
    portfolio_values = []
    timestamps = []
    initial_capital = capital
    buy_sell_annotations = []

    commission = 0.0
    previous_BHS_pred = None
    investment_amount = 0
    price_when_invested = None

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        features = features.to(device)
        targets = targets.to(device)

        # Correct the timestamp index by adding the backcandles delay
        timestamps.append(test_df.iloc[step + backcandles, 0])

        # Get the current price from the dataframe
        current_price_index = 1
        current_price = test_df.iloc[step + backcandles, current_price_index]

        # Predicted and true prices in normalized form
        order_prediction = model(features).cpu().detach().numpy().flatten()
        BHS_pred = order_prediction

        true_best_order = 1 if targets.cpu().numpy().flatten() > 0 else 0

        # Simulate investment based on model prediction
        if BHS_pred == 1 and (previous_BHS_pred != 1 or previous_BHS_pred is None):
            # If currently in a sell position, close it before buying
            if investment_amount > 0 and price_when_invested is not None:
                price_difference = (current_price - price_when_invested) / price_when_invested
                revenue_generated = investment_amount * (1 + price_difference)
                capital += revenue_generated
                buy_sell_annotations.append(['Sell', capital, current_price, timestamps[-1]])
                print(f"Closed Sell at {current_price:.2f}, Capital: {capital:.2f}, On: {timestamps[-1]}")

            # Buy using trade allocation of current capital
            investment_amount = capital * trade_allocation
            capital -= investment_amount
            price_when_invested = current_price
            buy_sell_annotations.append(['Buy', capital + investment_amount, current_price, timestamps[-1]])
            print(f"Bought at {current_price:.2f}, Capital: {capital:.2f}, On: {timestamps[-1]}")

        elif BHS_pred == 0 and (previous_BHS_pred == 1):
            # If currently in a buy position, close it before selling
            if investment_amount > 0 and price_when_invested is not None:
                price_difference = (current_price - price_when_invested) / price_when_invested
                revenue_generated = investment_amount * (1 + price_difference)
                capital += revenue_generated
                buy_sell_annotations.append(['Sell', capital, current_price, timestamps[-1]])
                print(f"Closed Buy at {current_price:.2f}, Capital: {capital:.2f}, On: {timestamps[-1]}")

            # Sell using trade allocation of current capital
            investment_amount = capital * trade_allocation
            capital -= investment_amount
            price_when_invested = current_price
            buy_sell_annotations.append(['Sell', capital + investment_amount, current_price, timestamps[-1]])
            print(f"Sold at {current_price:.2f}, Capital: {capital:.2f}, On: {timestamps[-1]}")

        # Calculate portfolio value after each decision
        if investment_amount > 0:
            portfolio_value = capital + investment_amount * ((current_price - price_when_invested) / price_when_invested + 1)
        else:
            portfolio_value = capital

        portfolio_values.append(portfolio_value)
        current_prices.append(current_price)

        # Store predicted and true orders
        predicted_orders.append(BHS_pred)
        best_orders.append(true_best_order)

        # Update previous BHS prediction
        previous_BHS_pred = BHS_pred

    # Convert timestamps to a datetime format
    timestamps = pd.to_datetime(timestamps)

    # Final Portfolio Summary
    print("Final Portfolio Value: {:.2f}".format(portfolio_values[-1]))
    print("Augmentation of the portfolio: {:.2%}".format((portfolio_values[-1] - initial_capital) / initial_capital))

    buy_sell_annotations_df = pd.DataFrame(buy_sell_annotations, columns=['Action', 'Portfolio', 'Price', 'Timestamp'])
    buy_sell_annotations_df['Timestamp'] = pd.to_datetime(buy_sell_annotations_df['Timestamp'])
    buy_sell_annotations_df.to_csv(f'logs/buy_sell_annotations_{model_name}.csv')

    # Plot results
    plot_portfolio_value(timestamps, portfolio_values, buy_sell_annotations_df, model_name=model_name)
    plot_actual_stock_price(timestamps, current_prices, buy_sell_annotations_df, model_name=model_name)
    plot_confusion_matrix(predicted_orders, best_orders, 'Global', model_name=model_name)

    print(buy_sell_annotations_df)

