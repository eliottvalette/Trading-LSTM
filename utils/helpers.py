from sklearn.metrics import f1_score
import numpy as np

def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.0, 0.99, 0.01)  # Experiment with values around 0.5
    best_threshold, best_f1 = 0.5, 0  # Default to 0.5
    for threshold in thresholds:
        preds = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold, best_f1