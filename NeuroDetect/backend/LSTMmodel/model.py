"""
LSTM Fraud Detection Model Architecture
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMFraudDetector(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        sequence_length=10
    ):
        """
        LSTM-based Fraud Detection Model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            sequence_length: Length of input sequences
        """
        super(LSTMFraudDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism (optional)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Fraud probability (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        ).unsqueeze(1)
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(attention_weights, lstm_out).squeeze(1)
        
        # Final classification
        output = self.fc(context)
        
        return output
    
    def predict(self, x):
        """
        Make prediction (evaluation mode)
        
        Args:
            x: Input tensor
            
        Returns:
            Fraud probability
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_fraud_score(self, x):
        """
        Get fraud score for a single transaction or batch
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            Fraud scores as numpy array
        """
        predictions = self.predict(x)
        return predictions.cpu().numpy().flatten()
