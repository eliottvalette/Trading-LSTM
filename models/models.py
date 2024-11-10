import torch 
import torch.nn as nn
import lightgbm as lgb

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
        self.dropout_lstm = nn.Dropout(p = dropout_prob)
        self.fc2_lstm = nn.Linear(64, 1)

        self.bn_lstm = nn.LayerNorm(hidden_dim)

        self.Leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, features):        
        # LSTM Part:
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]  # Extract the last output in the sequence

        # Apply BatchNorm1d only if batch size is greater than 1
        if last_hidden.size(0) > 1:
            last_hidden = self.bn_lstm(last_hidden)

        # Fully connected layers
        tag_space = self.Leakyrelu(self.fc1_lstm(last_hidden))
        tag_space = self.fc2_lstm(tag_space)

        # Final output:
        output = self.sigmoid(tag_space)
        return output

class CNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout_prob=0.2):
        super(CNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(embedding_dim * 2)  # Add BatchNorm1d here
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(embedding_dim * 4)  # Add BatchNorm1d here
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1_cnn = nn.Linear(embedding_dim * 4, 16)
        self.dropout_cnn = nn.Dropout(p = dropout_prob)
        self.fc2_cnn = nn.Linear(16, 1)

        self.Leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, features):
        # CNN Part :
        cnn_out = self.conv1(features.permute(0, 2, 1))
        cnn_out = self.bn1(cnn_out)  # Apply BatchNorm1d after convolution
        cnn_out = self.pool1(cnn_out)

        cnn_out = self.conv2(cnn_out)
        cnn_out = self.bn2(cnn_out)  # Apply BatchNorm1d after convolution
        cnn_out = self.pool2(cnn_out)

        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out, _ = torch.max(cnn_out, dim=1)
        cnn_out = self.Leakyrelu(self.fc1_cnn(cnn_out))
        cnn_out = self.fc2_cnn(cnn_out)

        # Final output :
        output = self.sigmoid(cnn_out)
        return output
    
class GradBOOSTModel:
    def __init__(self, num_leaves=128, max_depth=5, learning_rate=0.05, n_estimators=100):
        self.lgbm_model = lgb.LGBMClassifier(
            num_leaves=128, 
            max_depth=5, 
            learning_rate=0.05, 
            n_estimators=100,
            objective='binary',
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
            metric='binary_logloss'
        )
                           

    def fit(self, features, targets):
        # Fit the model with features and targets
        self.lgbm_model.fit(features, targets)

    def predict(self, features):
        return self.lgbm_model.predict(features)
        
class EnsemblingModel(nn.Module):
    def __init__(self, lstm_model, cnn_model, gradboost_model):
        super(EnsemblingModel, self).__init__()
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.gradboost_model = gradboost_model

    def forward(self, features):
        device = features.device  # Get the device of the input tensor

        # Get the predictions from each model
        lstm_output = torch.round(self.lstm_model(features))
        cnn_output = torch.round(self.cnn_model(features))
        
        # Prepare features for GradBOOSTModel and predict
        gradboost_features = features.view(features.size(0), -1).cpu().numpy()
        gradboost_output = torch.tensor(
            self.gradboost_model.predict(gradboost_features),
            device=device,
            dtype=torch.float32
        ).unsqueeze(1)
        gradboost_output = torch.round(gradboost_output)

        # Stack predictions and perform manual majority voting
        predictions = torch.cat([lstm_output, cnn_output, gradboost_output], dim=1)  # Shape: [batch_size, num_models]
        
        # Calculate majority vote by summing up predictions
        vote_sums = predictions.sum(dim=1)  # Sum of votes along the model axis
        majority_vote = (vote_sums >= 2).float()  # Majority if 2 or more models predict 1, else 0

        return majority_vote
