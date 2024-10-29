import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout_prob, 
                            batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        output = self.sigmoid(tag_space)
        return output
