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

        self.Leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, features):
        # LSTM Part :
        lstm_out, _ = self.lstm(features)
        tag_space = self.Leakyrelu(self.fc1_lstm(lstm_out[:, -1, :]))
        tag_space = self.fc2_lstm(tag_space)
        
        # Final output :
        output = self.sigmoid(tag_space)
        return output

class CNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout_prob=0.2):
        super(CNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_cnn = nn.Linear(embedding_dim * 4, 16)
        self.dropout_cnn = nn.Dropout(p = dropout_prob)
        self.fc2_cnn = nn.Linear(16, 1)

        self.Leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, features):
        # CNN Part :
        cnn_out = self.conv1(features.permute(0, 2, 1))
        cnn_out = self.pool1(cnn_out)
        cnn_out = self.conv2(cnn_out)
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

        lstm_output = self.lstm_model(features)
        cnn_output = self.cnn_model(features)
        
        # Prepare features for GradBOOSTModel and predict
        gradboost_features = features.view(features.size(0), -1).cpu().numpy()
        gradboost_output = torch.tensor(
            self.gradboost_model.predict(gradboost_features),
            device=device,
            dtype=torch.float32
        ).unsqueeze(1)

        weights = [0.3, 0.3, 0.4]

        # Weighted sum of the model outputs
        ensemble_output = (
            weights[0] * lstm_output +
            weights[1] * cnn_output +
            weights[2] * gradboost_output
        )
        
        return ensemble_output
