import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def plot_portfolio_value(timestamps, portfolio_values):
    """Plot portfolio value over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, portfolio_values, label='Portfolio Value', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value in USD')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/portfolio.png')


def plot_actual_stock_price(timestamps, current_prices):
    """Plot actual stock price over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, current_prices, label='Actual Stock Price', color='green')
    plt.xlabel('Time')
    plt.ylabel('Stock Price in USD')
    plt.title('Actual Stock Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/stock.png')


def plot_evolutions(timestamps, true_evolutions, predicted_evolutions):
    """Plot true and predicted evolutions over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, true_evolutions, label='True Evolution', color='red')
    plt.plot(timestamps, predicted_evolutions, label='Predicted Evolution', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Price Change Prediction (Original Scale)')
    plt.title('Evolution Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/evolutions.png')

def simulate_investment(model, dataloader, capital, shares_owned, scaler, buy_threshold, sell_threshold, test_df, backcandles):
    model.eval()
    true_evolutions = []
    predicted_evolutions = []
    current_prices = []
    portfolio_values = []
    timestamps = []
    initial_capital = capital

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        # Correct the timestamp index by adding the backcandles delay
        timestamps.append(test_df.iloc[step + backcandles, 0])
        current_price = features[:, -1, 3].numpy().flatten()[0] # Current closing price [fourth column of the last row]

        # Predicted and true prices in normalized form
        predicted_evolution_norm = model(features).detach().numpy().flatten()
        true_evolution_norm = targets.numpy().flatten()

        # Create dummy arrays to inverse transform
        dummy_pred = np.zeros((1, 6))
        dummy_pred[:, 5] = predicted_evolution_norm
        dummy_target = np.zeros((1, 6))
        dummy_target[:, 5] = true_evolution_norm

        predicted_evolution = scaler.inverse_transform(dummy_pred)[:, 5][0]
        true_evolution = scaler.inverse_transform(dummy_target)[:, 5][0]


        # Simulate investment based on model prediction
        if predicted_evolution >= buy_threshold and capital >= current_price:
            # Buy one share if the predicted change is positive and capital allows
            shares_owned += 1
            capital -= current_price
            # print(f"Bought 1 share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On step : {step}")

        elif predicted_evolution <= sell_threshold and shares_owned > 0:
            # Sell one share if the predicted change is negative and shares are available
            shares_owned -= 1
            capital += current_price
            # print(f"Sold 1 share at {current_price:.2f}, Capital: {capital:.2f}, Shares Owned: {shares_owned}, On step : {step}")

        # Calculate portfolio value after each decision
        portfolio_value = capital + shares_owned * current_price
        portfolio_values.append(portfolio_value)

        true_evolutions.append(true_evolution)
        predicted_evolutions.append(predicted_evolution)
        current_prices.append(current_price)

    # Convert timestamps to a datetime format
    timestamps = pd.to_datetime(timestamps)

    print("Final Portfolio Value: {:.2f}".format(portfolio_values[-1]))
    print("Augmentation of the portfolio: {:.2%}".format((portfolio_values[-1] - initial_capital) / initial_capital))

    # Plot results
    plot_portfolio_value(timestamps, portfolio_values)
    plot_actual_stock_price(timestamps, current_prices)
    plot_evolutions(timestamps, true_evolutions, predicted_evolutions)

    predicted_evolutions_df = pd.DataFrame(predicted_evolutions, columns=['Predicted Evolution'])
    print(predicted_evolutions_df.describe())
