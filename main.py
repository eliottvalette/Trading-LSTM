from config import Config
from data.data_utils import prepare_data, create_test_loader, training_loaders
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
    # Set up constants and configurations
    config = Config()

    # Prepare data
    df, dataset_scaled, train_sc = prepare_data(symbol=config.symbol, 
                                          start_date=config.start_date, 
                                          end_date=config.end_date,
                                          timeframe=config.timeframe, 
                                          is_filter=False, 
                                          limit= 4 * 365 * 12, 
                                          is_training=True)

    # Prepare training and validation data
    train_loader, valid_loader = training_loaders(dataset_scaled, config.backcandles)

    # Initialize model, optimizer, and scheduler
    lstm_model = LSTMModel(embedding_dim=5, 
                           hidden_dim = 64,
                           num_layers=2, 
                           dropout_prob=0.2)
    
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

     # Prepare test data
    test_df, test_dataset_scaled, _ = prepare_data(symbol=config.symbol, 
                                          start_date=config.test_start_date, 
                                          end_date=config.test_end_date,
                                          timeframe=config.timeframe, 
                                          is_filter=False, 
                                          limit= 1 * 305 * 12, 
                                          is_training=False,
                                          sc = train_sc)

    # Create test loader
    test_loader = create_test_loader(
        dataset_scaled=test_dataset_scaled,
        backcandles=config.backcandles,
    )

    # Run simulation
    simulate_investment(model = trained_model, 
            dataloader = test_loader, 
            capital = config.initial_capital, 
            shares_owned = config.shares_owned, 
            scaler = train_sc,
            buy_threshold = 0.01,
            sell_threshold = -0.01
            )
