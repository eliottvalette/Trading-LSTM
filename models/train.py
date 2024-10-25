import copy
import time
from tqdm import tqdm
from collections import defaultdict
import torch

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
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (features, targets) in bar:

        batch_size = features.size(0)

        predicted_evolution = model(features)       

        # Compute the loss
        loss = criterion(predicted_evolution, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss)        
    return epoch_loss

def run_training(model, train_loader, valid_loader, optimizer, scheduler, criterion, num_epochs):
    
    start = time.time()
    
    # Deep copies the initial model weights to save the best model later
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_epoch_loss = 300
    
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
        val_epoch_loss = valid_one_epoch(model = model, 
                                         criterion = criterion, 
                                         dataloader = valid_loader,
                                         epoch = epoch)
        
        # Save the loss and score in the history dict
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save the model if it's getting better results
        if best_epoch_loss >= val_epoch_loss:
            print(f"{best_epoch_loss} Validation AUROC Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            
            best_epoch_loss = val_epoch_loss
            
            # Deepcopy the weights
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Saves the weights in the working directory
            PATH = f"saved_weights/LSTM_Loss_{val_epoch_loss:.4f}_epoch{epoch}.bin".format(val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            
            print(f"Model Saved")
            
        print()

        scheduler.step()
    
    end = time.time()
    
    # Display the training time
    time_elapsed = end - start # in seconds
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600 , (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best LOSS: {:.6f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history