from config import Config
from data.data_utils import prepare_data_from_preloads, prepare_data, create_test_loader, training_loaders
from models.models import LSTMModel, CNNModel
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
    loss = nn.BCELoss()(outputs, targets)
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
        dataframe = df,
        dataset_scaled=dataset_scaled, 
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
        )

    # Initialize LSTM model, optimizer, and scheduler
    lstm_model = LSTMModel(embedding_dim=len(train_cols), 
                           hidden_dim = 128,
                           num_layers = 2, 
                           dropout_prob = 0.2)
    lstm_model = lstm_model.to(device)
    
    optimizer_lstm = Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_lstm = lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='min', factor=0.1, patience=3)

    # Train the LSTM model
    trained_model_lstm, history_lstm = run_training(
        model=lstm_model,
        model_name='LSTM', 
        decision_threshold=config.decision_threshold,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer_lstm,
        scheduler=scheduler_lstm,
        criterion=criterion,
        num_epochs=config.num_epochs,
        device=device
    )

    # Plot training metrics
    plot_training_metrics(history_lstm, 'LSTM')

    # Initialize the CNN model, optimizer and scheduler
    cnn_model = CNNModel(embedding_dim=len(train_cols), 
                           hidden_dim = 128,
                           num_layers = 2, 
                           dropout_prob = 0.2)
    
    cnn_model = cnn_model.to(device)

    optimizer_cnn = Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_cnn = lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.1, patience=3)

    # Train the CNN model
    trained_model_cnn, history_cnn = run_training(
        model=cnn_model,
        model_name='CNN',
        decision_threshold=config.decision_threshold,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer_cnn,
        scheduler=scheduler_cnn,
        criterion=criterion,
        num_epochs=config.num_epochs,
        device=device
    )

    # Plot training metrics
    plot_training_metrics(history_cnn, 'CNN')

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
        dataframe = test_df,
        dataset_scaled=test_dataset_scaled,
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold, 
    )

    while input('Quit ?') != 'q' :
        trade_decision_threshold = float(input('trade_decision_threshold? '))
        # Run simulation on LSTM
        simulate_investment(model = trained_model_lstm, 
                dataloader = test_loader, 
                capital = config.initial_capital, 
                shares_owned = config.shares_owned, 
                scaler = train_sc,
                test_df = test_df,
                backcandles=config.backcandles,
                train_cols=train_cols,
                decision_threshold=config.decision_threshold, # to differenciate buy or sell signal for conf matrix
                trade_decision_threshold=trade_decision_threshold, # to keep only confident signals for the simulation
                device = device,
                model_name = 'LSTM'
                )
        
        # Run simulation on CNN
        simulate_investment(model = trained_model_cnn, 
                dataloader = test_loader, 
                capital = config.initial_capital, 
                shares_owned = config.shares_owned, 
                scaler = train_sc,
                test_df = test_df,
                backcandles=config.backcandles,
                train_cols=train_cols,
                decision_threshold=config.decision_threshold, # to differenciate buy or sell signal for conf matrix
                trade_decision_threshold=trade_decision_threshold, # to keep only confident signals for the simulation
                device = device,
                model_name = 'CNN'
                )

