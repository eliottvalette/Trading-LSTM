import copy
import time
from tqdm import tqdm
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import Config

config = Config()

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

def plot_predicted_evolution(y_true, y_pred, title, file_title, model_name):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Labels', color='blue')
    plt.plot(y_pred, label='Predicted Labels', color='red')
    plt.xlabel('Time')
    plt.ylabel('Predicted Evolution')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/predicted_evolution_{file_title}_{model_name}.png')
    plt.close()

def train_one_epoch(model, model_name, optimizer, criterion, dataloader, epoch, device):
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
        train_predictions.extend(predicted_evolution.cpu().detach().numpy())
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss)

    if epoch == config.num_epochs  :
        predictions_binary = np.where(np.array(train_predictions) > 0, 1, 0)
        targets_binary = np.where(np.array(train_targets) > 0, 1, 0)

        plot_confusion_matrix(targets_binary, predictions_binary, title="Training Set Confusion Matrix", file_title = "train", model_name = model_name)

    return epoch_loss

def valid_one_epoch(model, model_name, criterion, dataloader, epoch, device):    
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

    valid_predictions_binary = np.where(np.array(valid_predictions_probs) > 0, 1, 0)
    targets_binary = np.where(np.array(valid_targets) > 0, 1, 0)
    
    # Calculate F1 score and accuracy
    f1 = f1_score(targets_binary, valid_predictions_binary)
    accuracy = accuracy_score(targets_binary, valid_predictions_binary)

    print(f"Validation Metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    if epoch == config.num_epochs :
        plot_confusion_matrix(targets_binary, valid_predictions_binary, title="Validation Set Confusion Matrix", file_title="valid", model_name=model_name)
        plot_predicted_evolution(valid_targets, valid_predictions_probs, title="Predicted Evolution", file_title="valid", model_name=model_name)
        
        print(pd.Series(valid_predictions_probs).describe())

    return epoch_loss, accuracy, f1


def run_training(model, model_name, train_loader, valid_loader, optimizer, scheduler, criterion, num_epochs, device):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    print(f'Training the model with {model_name} architecture')
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_f1 = 0  # Initialize the best F1 score
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        
        print(f"EPOCH [{epoch}/{num_epochs}]")
        
        # Train for one epoch
        train_epoch_loss = train_one_epoch(model = model, 
                                           model_name = model_name,
                                           optimizer = optimizer, 
                                           criterion = criterion, 
                                           dataloader = train_loader,
                                           epoch = epoch,
                                           device = device)
        # Validate for one epoch
        val_epoch_loss, val_accuracy, val_f1 = valid_one_epoch(model = model,
                                                               model_name = model_name,
                                                               criterion = criterion, 
                                                               dataloader = valid_loader,
                                                               epoch = epoch,
                                                               device = device)
        
        # Save metrics in history
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Accuracy'].append(val_accuracy)
        history['F1 Score'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save the model based on the F1 Score improvement
        if best_f1 <= val_f1:  # Compare current F1 score with best F1 score
            print(f"Best F1 Score Improved ({best_f1} ---> {val_f1})")

            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save the model based on F1 Score improvement
            PATH = f"saved_weights/{model_name}_F1_{val_f1:.4f}_epoch{epoch}.txt"
            # torch.save(model.state_dict(), PATH)
            print("Model Saved")

        print()

        scheduler.step(val_epoch_loss)
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600 , (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best F1 Score: {:.4f}".format(best_f1))

    FINAL_PATH = f"saved_weights/Best_{model_name}_F1_{best_f1:.4f}.pth"
    torch.save(model.state_dict(), FINAL_PATH)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def run_training_LGBM(model, model_name, train_loader, valid_loader, num_epochs, device):
    print(f"\nRunning training for LGBM model")

    history = defaultdict(list)
    
    # Collect training data
    train_features, train_targets = [], []
    for features, targets in train_loader:
        # Reshape features to 2D: [Batch, Channels * Sequence_length]
        reshaped_features = features.view(features.size(0), -1).cpu().numpy()
        train_features.append(reshaped_features)
        train_targets.extend((targets.cpu().numpy() > 0).astype(int))
    
    # Fit the LGBM model on aggregated training data
    train_features = np.concatenate(train_features, axis=0)
    model.fit(train_features, train_targets)
    
    # Collect validation data and predict
    valid_features, valid_targets = [], []
    valid_predictions = []
    for features, targets in valid_loader:
        reshaped_features = features.view(features.size(0), -1).cpu().numpy()
        valid_features.append(reshaped_features)
        valid_targets.extend((targets.cpu().numpy() > 0).astype(int))
    
    # Predict using the LGBM model on validation set
    valid_features = np.concatenate(valid_features, axis=0)
    valid_predictions = model.predict(valid_features)
    
    accuracy = accuracy_score(valid_targets, valid_predictions)
    f1 = f1_score(valid_targets, valid_predictions)
    print(f"Validation Metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    plot_confusion_matrix(valid_targets, valid_predictions, title="Validation Set Confusion Matrix", file_title="valid", model_name=model_name)

    # Save the model
    PATH = f"saved_weights/{model_name}_F1_{f1:.4f}.txt"
    model.save(PATH)

    return model, history