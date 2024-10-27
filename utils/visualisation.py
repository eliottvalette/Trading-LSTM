import matplotlib.pyplot as plt
import os

def plot_training_metrics(history):
    epochs = range(1, len(history['Train Loss']) + 1)
    plt.figure(figsize=(18, 10))
    
    # Plot Training Loss and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['Train Loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['Valid Loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['Accuracy'], label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['F1 Score'], label='F1 Score', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()

    # Plot Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    plt.savefig('logs/metrics.png')
