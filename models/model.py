import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_prob, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out[:, -1, :])
        lstm_out = self.dropout(lstm_out)
        x = self.relu(self.fc1(lstm_out))
        return self.fc2(x)
