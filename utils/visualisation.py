
import matplotlib.pyplot as plt
import os

def plot_training_metrics(history):
    epochs = range(1, len(history['Train Loss']) + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['Train Loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['Valid Loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('logs/metrics.png')
