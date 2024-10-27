import copy
import time
from tqdm import tqdm
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_one_epoch(model, optimizer, criterion, dataloader, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:

        batch_size = features.size(0)
        
        optimizer.zero_grad()
        
        predicted_evolution = model(features)
        
        loss = criterion(predicted_evolution, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss)
    return epoch_loss

def valid_one_epoch(model, criterion, dataloader, epoch):    
    model.eval()
    dataset_size = 0
    running_loss = 0.0

    # Lists to store true and predicted values for metrics
    all_targets = []
    all_predictions = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:

        batch_size = features.size(0)

        with torch.no_grad():
            predicted_evolution = model(features)      

        # Compute the loss
        loss = criterion(predicted_evolution, targets)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size


        all_targets.extend(targets.argmax(dim=1).cpu().numpy())        
        all_predictions.extend(predicted_evolution.argmax(dim=1).cpu().numpy())

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)   
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='macro')  

    print(f"Validation Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return epoch_loss, accuracy, f1

def run_training(model, train_loader, valid_loader, optimizer, scheduler, criterion, num_epochs):
    
    start = time.time()
    
    # Deep copies the initial model weights to save the best model later
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_loss = 30
    
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        
        print(f"EPOCH [{epoch}/{num_epochs}]")
        
        # Train for one epoch
        train_epoch_loss = train_one_epoch(model = model, 
                                           optimizer = optimizer, 
                                           criterion = criterion, 
                                           dataloader = train_loader,
                                           epoch = epoch)
        # Valid for one epoch
        val_epoch_loss, val_accuracy, val_f1 = valid_one_epoch(model = model, 
                                         criterion = criterion, 
                                         dataloader = valid_loader,
                                         epoch = epoch)
        
        # Save the metrics in the history dictionary
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Accuracy'].append(val_accuracy)
        history['F1 Score'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save the model if it's getting better results
        if best_loss >= val_epoch_loss:
            print(f"Best Loss Improved ({best_loss} ---> {val_epoch_loss})")

            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save the model based on F1 Score improvement
            PATH = f"saved_weights/LSTM_Loss_{val_epoch_loss:.4f}_epoch{epoch}.bin"
            torch.save(model.state_dict(), PATH)

            print("Model Saved")

        print()

        scheduler.step()
    
    end = time.time()
    
    # Display the training time
    time_elapsed = end - start # in seconds
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600 , (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best F1: {:.6f}".format(best_loss))

    FINAL_PATH = f"saved_weights/Best_LSTM_Loss_{val_epoch_loss:.8f}.bin"
    torch.save(model.state_dict(), FINAL_PATH)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history