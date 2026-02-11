"""
LSTM Data Preprocessor - Enhanced with class imbalance handling
Handles sequence creation, feature engineering, and balanced sampling for LSTM models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def create_balanced_sequences(data, labels, sequence_length=10, step=1, fraud_boost_factor=5):
    """
    Create sequences with balanced class distribution
    
    This function oversamples fraud sequences and undersamples normal sequences
    to create a more balanced training dataset
    
    Args:
        data: Feature array (n_samples, n_features)
        labels: Target labels (n_samples,)
        sequence_length: Length of each sequence
        step: Step size for sequence creation (not used in current implementation)
        fraud_boost_factor: Factor by which to oversample fraud sequences
        
    Returns:
        sequences: Array of sequences (n_sequences, sequence_length, n_features)
        sequence_labels: Labels for each sequence (n_sequences,)
    """
    sequences = []
    sequence_labels = []
    
    # Find fraud and normal indices
    fraud_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    # Create sequences for fraud transactions (oversample)
    for i in fraud_indices:
        if i >= sequence_length - 1:
            for _ in range(fraud_boost_factor):  # Oversample fraud sequences
                start_idx = i - sequence_length + 1
                sequence = data[start_idx:i+1]
                label = labels[i]
                sequences.append(sequence)
                sequence_labels.append(label)
    
    # Create sequences for normal transactions (undersample)
    np.random.shuffle(normal_indices)
    normal_samples = min(len(fraud_indices) * fraud_boost_factor * 2, len(normal_indices))
    
    for i in normal_indices[:normal_samples]:
        if i >= sequence_length - 1:
            start_idx = i - sequence_length + 1
            sequence = data[start_idx:i+1]
            label = labels[i]
            sequences.append(sequence)
            sequence_labels.append(label)
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(sequences))
    sequences = sequences[shuffle_idx]
    sequence_labels = sequence_labels[shuffle_idx]
    
    return sequences, sequence_labels


def prepare_improved_lstm_data(df_preprocessed, sequence_length=10, step=1):
    """
    Prepare balanced data for LSTM with train/val/test split
    
    Args:
        df_preprocessed: Preprocessed dataframe with features and 'is_fraud' column
        sequence_length: Length of each sequence
        step: Step size for sequence creation
        
    Returns:
        X_train, y_train: Training sequences and labels
        X_val, y_val: Validation sequences and labels
        X_test, y_test: Test sequences and labels
        feature_names: List of feature names
    """
    # Separate features and target
    X = df_preprocessed.drop('is_fraud', axis=1).values
    y = df_preprocessed['is_fraud'].values
    
    feature_names = df_preprocessed.drop('is_fraud', axis=1).columns.tolist()
    
    print(f"Total transactions: {len(X)}")
    print(f"Fraud percentage: {(y.sum()/len(y))*100:.4f}%")
    
    # Create balanced sequences
    X_sequences, y_sequences = create_balanced_sequences(
        X, y, 
        sequence_length=sequence_length,
        step=step,
        fraud_boost_factor=10  # Strong oversampling for fraud
    )
    
    print(f"\nAfter balancing:")
    print(f"Total sequences: {len(X_sequences)}")
    print(f"Normal sequences: {(y_sequences == 0).sum()} ({(y_sequences == 0).sum()/len(y_sequences)*100:.2f}%)")
    print(f"Fraud sequences: {(y_sequences == 1).sum()} ({(y_sequences == 1).sum()/len(y_sequences)*100:.2f}%)")
    
    # Split into train/val/test (70/15/15)
    total_samples = len(X_sequences)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    
    X_val = X_sequences[train_size:train_size + val_size]
    y_val = y_sequences[train_size:train_size + val_size]
    
    X_test = X_sequences[train_size + val_size:]
    y_test = y_sequences[train_size + val_size:]
    
    print(f"\nDataset splits:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


class LSTMPreprocessor:
    """
    LSTM Preprocessor with scaling and sequence creation capabilities
    """
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
    
    def fit_transform(self, X_train):
        """
        Fit scaler and transform training data
        
        Args:
            X_train: Training sequences (n_sequences, sequence_length, n_features)
            
        Returns:
            X_train_scaled: Scaled training sequences
        """
        self.scaler = StandardScaler()
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(original_shape)
        return X_train_scaled
    
    def transform(self, X):
        """
        Transform data using fitted scaler
        
        Args:
            X: Input sequences (n_sequences, sequence_length, n_features)
            
        Returns:
            X_scaled: Scaled sequences
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        return X_scaled
    
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
