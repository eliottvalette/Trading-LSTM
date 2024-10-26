import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def simulate_investment(model, dataloader, capital, shares_owned, scaler, buy_threshold, sell_threshold):
    model.eval()
    true_evolutions = []
    predicted_evolutions = []
    current_prices = []
    portfolio_values = []
    initial_capital = capital

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        # Predicted and true prices in normalized form
        predicted_price_norm = model(features).detach().numpy().flatten()
        true_price_norm = targets.numpy().flatten()

        # Create dummy arrays to inverse transform
        dummy_pred = np.zeros((1, 5))
        dummy_pred[:, 3] = predicted_price_norm
        dummy_target = np.zeros((1, 5))
        dummy_target[:, 3] = true_price_norm

        # Inverse transform to get prices in original scale
        predicted_price_original = scaler.inverse_transform(dummy_pred)[:, 3][0]
        true_price_original = scaler.inverse_transform(dummy_target)[:, 3][0]

        # Current price in original scale
        current_price_norm = features[:, -1, 3].numpy().flatten()[0]
        dummy_price = np.zeros((1, 5))
        dummy_price[:, 3] = current_price_norm
        current_price = scaler.inverse_transform(dummy_price)[:, 3][0]

        # Calculate price increase percentage
        predicted_evolution = (predicted_price_original - current_price) / current_price
        true_evolution = (true_price_original - current_price) / current_price

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


    print("Final Portfolio Value: {:.2f}".format(portfolio_values[-1]))
    print("Augmentation of the portfolio: {:.2%}".format((portfolio_values[-1] - initial_capital) / initial_capital))

    # Plot final results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value in USD')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/portfolio.png')

    # Plot Actual Stock
    plt.figure(figsize=(12, 6))
    plt.plot(current_prices, label='Actual Stock Price', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price in USD')
    plt.title('Actual Stock Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/stock.png')

    # Plot Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(true_evolutions, label='True Evolution', color='red')
    plt.plot(predicted_evolutions, label='Predicted Evolution', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Price Change Prediction (Original Scale)')
    plt.title('Evolution Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/evolutions.png')

    predicted_evolutions_df = pd.DataFrame(predicted_evolutions, columns=['Predicted Evolution'])
    print(predicted_evolutions_df.describe())
