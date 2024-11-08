import copy
import time
from tqdm import tqdm
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, title, file_title, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Sell', 'Buy'],
                yticklabels=['Sell', 'Buy'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.savefig(f'logs/confusion_matrix_{file_title}_{model_name}.png')
    plt.close()


def train_one_epoch(model, model_name, decision_threshold, optimizer, criterion, dataloader, epoch, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    train_targets = []
    train_predictions = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:

        features = features.to(device)
        targets = targets.to(device)

        batch_size = features.size(0)
        
        optimizer.zero_grad()
        
        predicted_evolution = model(features).squeeze(-1)
        
        loss = criterion(predicted_evolution, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        train_targets.extend(targets.cpu().numpy())
        train_predictions.extend(np.where(predicted_evolution.cpu().detach().numpy() > decision_threshold, 1, 0))
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss)

    if epoch == 20 :
        plot_confusion_matrix(train_targets, train_predictions, title="Training Set Confusion Matrix", file_title = "train", model_name = model_name)

    return epoch_loss

def valid_one_epoch(model, model_name, decision_threshold, criterion, dataloader, epoch, device):    
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    
    valid_targets = []
    valid_predictions_probs = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:
        features = features.to(device)
        targets = targets.to(device)
        batch_size = features.size(0)

        with torch.no_grad():
            predicted_evolution = model(features).squeeze(-1)

        # Compute the loss
        loss = criterion(predicted_evolution, targets)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        # Append current batch results instead of overwriting
        valid_targets.extend(targets.cpu().numpy())
        valid_predictions_probs.extend(predicted_evolution.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    valid_predictions = np.where(np.array(valid_predictions_probs) > decision_threshold, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(valid_targets, valid_predictions)

    print(f"Validation Metrics - Accuracy: {accuracy:.4f}")

    if epoch == 20:
        plot_confusion_matrix(valid_targets, valid_predictions, title="Validation Set Confusion Matrix", file_title="valid", model_name=model_name)

    return epoch_loss, accuracy


def run_training(model, model_name, decision_threshold, train_loader, valid_loader, optimizer, scheduler, criterion, num_epochs, device):
    # Confirm that it is running on GPU
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    
    # Deep copies the initial model weights to save the best model later
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_loss = 30
    
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        
        print(f"EPOCH [{epoch}/{num_epochs}]")
        
        # Train for one epoch
        train_epoch_loss = train_one_epoch(model = model, 
                                           model_name = model_name,
                                           decision_threshold = decision_threshold,
                                           optimizer = optimizer, 
                                           criterion = criterion, 
                                           dataloader = train_loader,
                                           epoch = epoch,
                                           device = device)
        # Valid for one epoch
        val_epoch_loss, val_accuracy= valid_one_epoch(model = model,
                                         model_name = model_name,
                                         decision_threshold = decision_threshold,
                                         criterion = criterion, 
                                         dataloader = valid_loader,
                                         epoch = epoch,
                                         device = device)
        
        # Save the metrics in the history dictionary
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Accuracy'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save the model if it's getting better results
        if best_loss >= val_epoch_loss or True:
            print(f"Best Loss Improved ({best_loss} ---> {val_epoch_loss})")

            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save the model based on Loss Score improvement
            PATH = f"saved_weights/LSTM_Loss_{val_epoch_loss:.4f}_epoch{epoch}.bin"
            torch.save(model.state_dict(), PATH)

            print("Model Saved")

        print()

        scheduler.step(val_epoch_loss)
    
    end = time.time()
    
    # Display the training time
    time_elapsed = end - start # in seconds
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600 , (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.6f}".format(best_loss))

    FINAL_PATH = f"saved_weights/Best_LSTM_Loss_{val_epoch_loss:.8f}.bin"
    torch.save(model.state_dict(), FINAL_PATH)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def run_training_LGBM(model, model_name, decision_threshold, train_loader, valid_loader, num_epochs, device):
    print()
    print(f"Running training for LGBM model")
    history = defaultdict(list)
    
    # Collect training data
    train_features, train_targets = [], []
    for features, targets in train_loader:
        # Reshape features to 2D: [Batch, Channels * Sequence_length]
        reshaped_features = features.view(features.size(0), -1).cpu().numpy()
        train_features.append(reshaped_features)
        train_targets.extend(targets.cpu().numpy())
    
    # Fit the LGBM model on aggregated training data
    train_features = np.concatenate(train_features, axis=0)
    model.fit(train_features, train_targets)
    
    # Collect validation data and predict
    valid_features, valid_targets = [], []
    valid_predictions = []
    for features, targets in valid_loader:
        reshaped_features = features.view(features.size(0), -1).cpu().numpy()
        valid_features.append(reshaped_features)
        valid_targets.extend(targets.cpu().numpy())
    
    # Predict using the LGBM model on validation set
    valid_features = np.concatenate(valid_features, axis=0)
    valid_predictions = model.predict(valid_features)
    
    # Calculate metrics
    accuracy = accuracy_score(valid_targets, valid_predictions)    
    print(f"Validation Metrics - Accuracy: {accuracy:.4f}")

    plot_confusion_matrix(valid_targets, valid_predictions, title="Validation Set Confusion Matrix", file_title = "valid", model_name = model_name)        

        
    return model, history