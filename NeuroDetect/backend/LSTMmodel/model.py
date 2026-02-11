"""
Enhanced LSTM Fraud Detection Model Architecture
with attention mechanism and batch normalization
"""

import torch
import torch.nn as nn
import numpy as np


class EnhancedFraudLSTM(nn.Module):
    """
    Enhanced LSTM with attention mechanism, batch normalization, and bidirectional processing
    
    This model uses:
    - Bidirectional LSTM for better context understanding
    - Attention mechanism to focus on important time steps
    - Batch normalization for stable training
    - Deep fully connected layers for classification
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of LSTM layers (default: 3)
        dropout: Dropout rate (default: 0.4)
        bidirectional: Use bidirectional LSTM (default: True)
        use_attention: Use attention mechanism (default: True)
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.4, 
                 bidirectional=True, use_attention=True):
        super(EnhancedFraudLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 128),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            output: Fraud probability (batch_size, 1)
            attention_weights: Attention weights if use_attention=True, else None
        """
        # Apply batch normalization
        batch_size, seq_len, features = x.size()
        x = x.transpose(1, 2).contiguous()  # (batch, features, seq_len)
        x = self.batch_norm(x)
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, features)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention or last hidden state
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_output_dim)
        else:
            # Use last hidden state
            if self.bidirectional:
                context_vector = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context_vector = hidden[-1]
            attention_weights = None
        
        # Final classification
        output = self.fc(context_vector)
        
        return output, attention_weights
    
    def predict(self, x):
        """
        Make prediction in evaluation mode
        
        Args:
            x: Input tensor
            
        Returns:
            Fraud probability
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
            return output
    
    def get_fraud_score(self, x):
        """
        Get fraud score for a single transaction or batch
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            Fraud scores as numpy array
        """
        predictions, _ = self.predict(x)
        return predictions.cpu().numpy().flatten()


class LSTMFraudDetector(nn.Module):
    """
    Legacy LSTM Fraud Detection Model (kept for backward compatibility)
    
    For new projects, use EnhancedFraudLSTM instead
    """
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
