import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def ensemble_predict(lstm_model, cnn_model, gradboost_model, dataloader, device, decision_threshold=0.5, strategy='soft'):
    lstm_model.eval()
    cnn_model.eval()
    ensemble_predictions = []
    true_labels = []
    
    for features, targets in dataloader:
        features = features.to(device)
        true_labels.extend(targets.cpu().numpy())

        # Predict probabilities from LSTM and CNN models
        with torch.no_grad():
            lstm_probs = lstm_model(features).squeeze(-1).cpu().numpy()
            cnn_probs = cnn_model(features).squeeze(-1).cpu().numpy()
        
        # Reshape features for GradBOOSTModel and predict probabilities
        gradboost_features = features.view(features.size(0), -1).cpu().numpy()
        gradboost_probs = gradboost_model.lgbm_model.predict_proba(gradboost_features)[:, 1] # Probability of class 1 with is equivalent of a binary prediction after a sigmoid

        # Combine predictions based on strategy
        if strategy == 'soft':
            avg_probs = (lstm_probs + cnn_probs + gradboost_probs) / 3
            ensemble_preds = (avg_probs > decision_threshold).astype(int)
        elif strategy == 'hard':
            lstm_preds = (lstm_probs > decision_threshold).astype(int)
            cnn_preds = (cnn_probs > decision_threshold).astype(int)
            gradboost_preds = (gradboost_probs > decision_threshold).astype(int)
            # Majority voting
            ensemble_preds = (lstm_preds + cnn_preds + gradboost_preds >= 2).astype(int)
        else:
            raise ValueError("Unsupported strategy. Use 'soft' or 'hard'.")

        ensemble_predictions.extend(ensemble_preds)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    cm = confusion_matrix(true_labels, ensemble_predictions)
    print(f"Ensemble Accuracy ({strategy} voting): {accuracy:.4f}")
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Orders')
    plt.ylabel('Actual Orders')
    plt.title('Confusion Matrix')
    plt.savefig(f'logs/confusion_matrix_ensembling.png')
    
    return accuracy, cm