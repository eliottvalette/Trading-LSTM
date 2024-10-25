from config import Config
from data.data_utils import prepare_data, create_test_loader, prepare_training_validation_data
from models.model import LSTMModel
from models.train import run_training
from models.evaluate import simulate_investment
from utils.visualisation import plot_training_metrics
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import pandas as pd

def criterion(outputs, targets):
    loss = nn.MSELoss()(outputs, targets)
    return loss

if __name__ == "__main__":
    # Set up constants and configurations
    config = Config()

    # Prepare data
    df, dataset_scaled, sc = prepare_data(symbol=config.symbol, 
                                          start_date=config.start_date, 
                                          end_date=config.end_date,
                                          timeframe=config.timeframe, 
                                          is_filter=False, 
                                          limit=4000, 
                                          is_training=True)

    # Convert dataset_scaled to DataFrame if it is still in numpy format
    dataset_scaled = pd.DataFrame(dataset_scaled, columns=['open', 'high', 'low', 'close', 'volume', 'interval_evolution'])

    # Prepare training and validation data
    train_loader, valid_loader = prepare_training_validation_data(dataset_scaled, config.backcandles)

    # Initialize model, optimizer, and scheduler
    lstm_model = LSTMModel(embedding_dim=5, hidden_dim=64)
    optimizer = Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    trained_model, history = run_training(
        model=lstm_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config.num_epochs,
    )

    # Plot training metrics
    plot_training_metrics(history)

    # Create test loader and simulate investment
    test_loader, df_test, dataset_scaled_test = create_test_loader(
        symbol = config.symbol, 
        start_date = config.test_start_date, 
        end_date = config.test_end_date,
        timeframe = config.timeframe, 
        backcandles= config.backcandles, 
        sc = sc, 
        is_filter=False, 
        limit=1000
    )

    # Run simulation
    simulate_investment(model = trained_model, 
            dataloader = test_loader, 
            capital = config.initial_capital, 
            shares_owned = config.shares_owned, 
            scaler = sc,
            buy_threshold = 0.01,
            sell_threshold = -0.01
            )
