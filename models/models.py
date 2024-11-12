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
        tag_space = self.dropout_lstm(tag_space)
        output = self.fc2_lstm(tag_space)

        return output

class CNNModel(nn.Module):
    def __init__(self, embedding_dim, dropout_prob=0.2):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(embedding_dim * 2)  # Add BatchNorm1d here
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.conv2 = nn.Conv1d(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(embedding_dim * 4)  # Add BatchNorm1d here
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc1_cnn = nn.Linear(embedding_dim * 4, 16)
        self.dropout_cnn = nn.Dropout(p = dropout_prob)
        self.fc2_cnn = nn.Linear(16, 1)

        self.Leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        # CNN Part :
        cnn_out_1 = self.conv1(features.permute(0, 2, 1))
        cnn_out_1 = self.bn1(cnn_out_1)  # Apply BatchNorm1d after convolution
        cnn_out_1 = self.pool1(cnn_out_1)
        cnn_out_1 = self.dropout1(cnn_out_1)

        cnn_out_2 = self.conv2(cnn_out_1)
        cnn_out_2 = self.bn2(cnn_out_2)  # Apply BatchNorm1d after convolution
        cnn_out_2 = self.pool2(cnn_out_2)
        cnn_out_2 = self.dropout2(cnn_out_2)

        cnn_out_3 = cnn_out_2.permute(0, 2, 1)
        cnn_out_3, _ = torch.max(cnn_out_3, dim=1)
        cnn_out_3 = self.Leakyrelu(self.fc1_cnn(cnn_out_3))
        cnn_out_3 = self.dropout_cnn(cnn_out_3)
        output = self.fc2_cnn(cnn_out_3)

        return output
    
class GradBOOSTModel:
    def __init__(self, num_leaves, max_depth, learning_rate, n_estimators):
        self.lgbm_model = lgb.LGBMClassifier(
            num_leaves=num_leaves, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            n_estimators=n_estimators,
            objective='binary',
            random_state=42,
            verbosity=-1,
            metric='binary_logloss'
        )
                           

    def fit(self, features, targets):
        # Fit the model with features and targets
        self.lgbm_model.fit(features, targets)

    def predict(self, features):
        return self.lgbm_model.predict(features)

    def save(self, path):
        self.lgbm_model.booster_.save_model(path)
        
    def load(self, path):
        self.lgbm_model = lgb.Booster(model_file=path)
        
class EnsemblingModel(nn.Module):
    def __init__(self, lstm_model, cnn_model, gradboost_model):
        super(EnsemblingModel, self).__init__()
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.gradboost_model = gradboost_model

        self.decision_threshold = 0.0

    def forward(self, features):
        device = features.device  # Get the device of the input tensor

        # Get the predictions from each model and apply decision threshold
        lstm_output = self.lstm_model(features)
        cnn_output = self.cnn_model(features)

        # Convert LSTM and CNN predictions from `pct_change` to binary labels
        lstm_prediction = (lstm_output > self.decision_threshold).float()
        cnn_prediction = (cnn_output > self.decision_threshold).float()
        
        # Prepare features for GradBOOSTModel and predict
        gradboost_features = features.view(features.size(0), -1).cpu().numpy()
        gradboost_output = torch.tensor(
            self.gradboost_model.predict(gradboost_features),
            device=device,
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Convert GradBOOST output to binary by rounding
        gradboost_prediction = torch.round(gradboost_output)

        # Stack predictions and perform manual majority voting
        predictions = torch.cat([lstm_prediction, cnn_prediction, gradboost_prediction], dim=1)  # Shape: [batch_size, num_models]
        
        # Calculate majority vote by summing up predictions
        vote_sums = predictions.sum(dim=1)  # Sum of votes along the model axis
        majority_vote = (vote_sums >= 2).float()  # Majority if 2 or more models predict 1, else 0

        return majority_vote
    
class DirectionalMSELoss(nn.Module):
    def __init__(self, penalty_factor=2.0):
        super(DirectionalMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.penalty_factor = penalty_factor

    def forward(self, predictions, targets):
        # Calculate standard MSE
        mse_loss = self.mse(predictions, targets) * 1_000 # Manually increase the loss to make it more significant
        
        # Calculate directional mismatch: 1 if direction is wrong, 0 if correct
        direction_mismatch = (torch.sign(predictions) != torch.sign(targets)).float()
        
        # Apply penalty factor to mismatched directions
        directional_penalty = direction_mismatch * self.penalty_factor
        
        # Combine penalized loss
        penalized_loss = mse_loss * (1 + directional_penalty.mean())
        return penalized_loss
