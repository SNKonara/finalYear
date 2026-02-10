"""
LSTM Data Preprocessor
Handles sequence creation and feature engineering for LSTM models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class LSTMPreprocessor:
    def __init__(self, sequence_length=10):
        """
        Initialize LSTM preprocessor
        
        Args:
            sequence_length: Number of transactions in each sequence
        """
        self.sequence_length = sequence_length
        self.scaler = None
        self.feature_names = None
        
    def create_sequences(self, X, y=None):
        """
        Create sequences from transaction data
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Labels (optional)
            
        Returns:
            Sequences (n_sequences, sequence_length, n_features)
            Labels (n_sequences,) if y provided
        """
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length + 1
        
        # Create sequences
        X_seq = np.array([
            X[i:i + self.sequence_length] 
            for i in range(n_sequences)
        ])
        
        if y is not None:
            # Use the label of the last transaction in each sequence
            y_seq = np.array([
                y[i + self.sequence_length - 1] 
                for i in range(n_sequences)
            ])
            return X_seq, y_seq
        
        return X_seq
    
    def preprocess(self, df, is_training=True):
        """
        Preprocess DataFrame for LSTM model
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            
        Returns:
            X_seq: Sequences (n_sequences, sequence_length, n_features)
            y_seq: Labels (n_sequences,) if training
            feature_names: List of feature names
        """
        # Feature engineering (reuse from AEmodel)
        from ..AEmodel.preprocessor import DataPreprocessor
        
        ae_preprocessor = DataPreprocessor()
        X, y, feature_names = ae_preprocessor.preprocess(df, is_training=is_training)
        
        # Scale features
        if is_training:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_names
        else:
            if self.scaler is None:
                raise ValueError("Preprocessor not fitted. Load saved preprocessor first.")
            X_scaled = self.scaler.transform(X)
        
        # Create sequences
        if y is not None:
            X_seq, y_seq = self.create_sequences(X_scaled, y)
            return X_seq, y_seq, feature_names
        else:
            X_seq = self.create_sequences(X_scaled)
            return X_seq, None, feature_names
    
    def save_preprocessor(self, filepath):
        """Save preprocessor state"""
        state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        joblib.dump(state, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor state"""
        state = joblib.load(filepath)
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.sequence_length = state['sequence_length']
        print(f"✅ Preprocessor loaded from {filepath}")
