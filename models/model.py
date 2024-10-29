import torch 
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
        
        self.fc1_lstm = nn.Linear(hidden_dim, 64)
        self.fc2_lstm = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_cnn = nn.Linear(embedding_dim * 4, 64)
        self.fc2_cnn = nn.Linear(64, 64)

        self.fc1_cat = nn.Linear(128, 64)
        self.fc2_cat = nn.Linear(64, 1)

        self.Leakyrelu = nn.LeakyReLU()

    def forward(self, features):
        # LSTM Part :
        lstm_out, _ = self.lstm(features)
        tag_space = self.Leakyrelu(self.fc1_lstm(lstm_out[:, -1, :]))
        tag_space = self.fc2_lstm(tag_space)

        # CNN Part :
        cnn_out = self.conv1(features.permute(0, 2, 1))
        cnn_out = self.pool1(cnn_out)
        cnn_out = self.conv2(cnn_out)
        cnn_out = self.pool2(cnn_out)
        cnn_out = self.Leakyrelu(self.fc1_cnn(cnn_out))
        cnn_out = self.fc2_cnn(cnn_out)

        # Concatenate LSTM and CNN outputs :
        concat_features = torch.cat((tag_space, cnn_out), dim=1)
        concat_features = self.Leakyrelu(self.fc1_cat(concat_features))
        output = self.fc2_cat(concat_features)

        # Final output :
        output = self.sigmoid(output)
        return output