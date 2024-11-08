from config import Config
from data.data_utils import prepare_data_from_preloads, prepare_data, create_test_loader, training_loaders
from models.models import LSTMModel, CNNModel, GradBOOSTModel, EnsemblingModel
from models.train import run_training, run_training_LGBM
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
    print("Using device : ", device)

    # Set up constants and configurations
    config = Config()

    # Prepare data
    if config.use_preloads :
        df, dataset_scaled, train_cols = prepare_data_from_preloads(
            final_symbol=config.symbol,
            timeframe=config.timeframe, 
            is_filter=False, 
            backcandles=config.backcandles)
    else :
        df, dataset_scaled, train_cols = prepare_data(
            symbol=config.symbol, 
            start_date=config.start_date, 
            end_date=config.end_date,
            timeframe=config.timeframe, 
            is_filter=False, 
            backcandles=config.backcandles)

    # Prepare training and validation data
    train_loader, valid_loader = training_loaders(
        dataframe = df,
        dataset_scaled=dataset_scaled, 
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold)

    # Initialize LSTM model, optimizer, and scheduler
    lstm_model = LSTMModel(embedding_dim=len(train_cols), 
                           hidden_dim = 128,
                           num_layers = 2, 
                           dropout_prob = 0.2)
    lstm_model = lstm_model.to(device)
    
    optimizer_lstm = Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_lstm = lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='min', factor=0.5, patience=5)

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
        device=device)

    # Plot training metrics
    plot_training_metrics(history_lstm, 'LSTM')

    # Initialize the CNN model, optimizer and scheduler
    cnn_model = CNNModel(embedding_dim=len(train_cols), 
                           hidden_dim = 128,
                           num_layers = 2, 
                           dropout_prob = 0.2)
    
    cnn_model = cnn_model.to(device)

    optimizer_cnn = Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_cnn = lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.5, patience=3)

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
        device=device)

    # Plot training metrics
    plot_training_metrics(history_cnn, 'CNN')

    # Initialize the LGBM model
    gradboost_model = GradBOOSTModel(num_leaves=128, 
                           max_depth=5, 
                           learning_rate=0.05, 
                           n_estimators=100)

    # Train the LGBM model
    trained_model_gradboost, history_gradboost = run_training_LGBM(
        model=gradboost_model,
        model_name='LGBM',
        decision_threshold=config.decision_threshold,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config.num_epochs,
        device=device)
    
    # Initialize the Ensembling model
    ensembling_model = EnsemblingModel(trained_model_lstm, trained_model_cnn, trained_model_gradboost)

    ensembling_model.to(device)

    optimizer_ensembling = Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_ensembling = lr_scheduler.ReduceLROnPlateau(optimizer_ensembling, mode='min', factor=0.5, patience=3)

    # Prepare test data
    test_df, test_dataset_scaled, _ = prepare_data(symbol=config.symbol, 
                                          start_date=config.test_start_date, 
                                          end_date=config.test_end_date,
                                          timeframe=config.timeframe, 
                                          is_filter=False,
                                          backcandles=config.backcandles)

    # Create test loader
    test_loader = create_test_loader(dataframe = test_df,
        dataset_scaled=test_dataset_scaled,
        backcandles=config.backcandles,
        train_cols=train_cols,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold)

    # Run simulation
    trade_decision_threshold = 0.02
    simulate_investment(model = ensembling_model, 
            dataloader = test_loader, 
            dataframe = test_df,
            capital = config.initial_capital, 
            shares_owned = config.shares_owned, 
            test_df = test_df,
            backcandles=config.backcandles,
            train_cols=train_cols,
            decision_threshold=config.decision_threshold, # to differenciate buy or sell signal for conf matrix
            trade_decision_threshold=trade_decision_threshold, # to keep only confident signals for the simulation
            device = device,
            model_name = 'ENSEMBLING')


    
        
    
