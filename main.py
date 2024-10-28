from config import Config
from data.data_utils import prepare_data_from_preloads, prepare_data, create_test_loader, training_loaders
from models.model import LSTMModel
from models.train import run_training
from models.evaluate import simulate_investment
from utils.visualisation import plot_training_metrics
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import pandas as pd

import torch
import numpy as np
import random as rd

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rd.seed(seed)

def criterion(outputs, targets):
    loss = nn.MSELoss()(outputs, targets)
    return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    # Set up constants and configurations
    config = Config()

    # Prepare data
    if config.use_preloads :
        df, dataset_scaled, train_sc, train_cols = prepare_data_from_preloads(
            final_symbol=config.symbol,
            timeframe=config.timeframe, 
            is_filter=False, 
            is_training=True,
            backcandles=config.backcandles
        )
    else :
        df, dataset_scaled, train_sc, train_cols = prepare_data(
            symbol=config.symbol, 
            start_date=config.start_date, 
            end_date=config.end_date,
            timeframe=config.timeframe, 
            is_filter=False, 
            limit= 30_000, 
            is_training=True,
            backcandles=config.backcandles)

    # Prepare training and validation data
    train_loader, valid_loader = training_loaders(
        dataset_scaled=dataset_scaled, 
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
        )

    # Initialize model, optimizer, and scheduler
    lstm_model = LSTMModel(embedding_dim=len(train_cols), 
                           hidden_dim = 128,
                           num_layers = 2, 
                           dropout_prob = 0.2)
    lstm_model = lstm_model.to(device)
    
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
        device=device
    )

    # Plot training metrics
    plot_training_metrics(history)

     # Prepare test data
    test_df, test_dataset_scaled, _, _ = prepare_data(symbol=config.symbol, 
                                          start_date=config.test_start_date, 
                                          end_date=config.test_end_date,
                                          timeframe=config.timeframe, 
                                          is_filter=False, 
                                          limit= 20_000, 
                                          is_training=False,
                                          sc = train_sc,
                                          backcandles=config.backcandles)

    # Create test loader
    test_loader = create_test_loader(
        dataset_scaled=test_dataset_scaled,
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold, 
    )

    # Run simulation
    simulate_investment(model = trained_model, 
            dataloader = test_loader, 
            capital = config.initial_capital, 
            shares_owned = config.shares_owned, 
            scaler = train_sc,
            buy_threshold = config.buy_threshold,
            sell_threshold = config.sell_threshold,
            test_df = test_df,
            backcandles=config.backcandles,
            train_cols=train_cols,
            device = device
            )
