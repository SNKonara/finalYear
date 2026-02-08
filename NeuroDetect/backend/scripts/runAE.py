"""
FIRST SCRIPT TO RUN: Train and save the model
"""
import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

# Add parent directory to path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from AEmodel.preprocessor import DataPreprocessor
from AEmodel.model import FraudAutoencoder, AutoencoderTrainer

def train_simple_model():
    """First-time training script - run this FIRST"""
    
    print("="*60)
    print("FIRST-TIME MODEL TRAINING")
    print("="*60)
    
    # 1. Load your dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv("C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv")
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("✗ ERROR: Could not find 'dataset/fraudTrain.csv'")
        print("Please make sure your dataset is in the 'dataset/' folder")
        return
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Keep only required columns (adjust based on your dataset)
    required_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                    'is_fraud', 'trans_date_trans_time', 'category', 'gender']
    
    # Check if all columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        print("Using available columns...")
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols + ['is_fraud']]
    else:
        df = df[required_cols]
    
    X_scaled, y, feature_names = preprocessor.preprocess(df, is_training=True)
    print(f"✓ Preprocessing complete")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Samples: {X_scaled.shape[0]}")
    print(f"  Fraud percentage: {y.mean()*100:.2f}%")
    print(f"  Top categories: {preprocessor.top_categories}")
    print(f"  Category columns: {preprocessor.category_columns}")
    
    # 3. Split data (normal transactions only for training)
    print("\n3. Splitting data...")
    normal_indices = np.where(y == 0)[0]
    fraud_indices = np.where(y == 1)[0]
    
    np.random.shuffle(normal_indices)
    train_size = int(0.8 * len(normal_indices))
    
    X_train = X_scaled[normal_indices[:train_size]]
    X_val = X_scaled[normal_indices[train_size:train_size + 100000]]  # Limit validation size
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    
    # 4. Create and train model using AutoencoderTrainer
    print("\n4. Creating and training model...")
    input_dim = X_train.shape[1]
    
    # Initialize trainer
    trainer = AutoencoderTrainer()
    
    # Train model with the trainer class
    print(f"✓ Using device: {trainer.device}")
    model = trainer.train(
        X_train=X_train,
        X_val=X_val,
        input_dim=input_dim,
        epochs=100,  # Using more epochs with early stopping
        batch_size=128,
        patience=10
    )
    
    print("✓ Training complete!")
    
    # 5. Create test set for threshold determination
    print("\n5. Determining optimal threshold...")
    X_test_normal = X_scaled[normal_indices[train_size:train_size + 50000]]
    X_test_fraud = X_scaled[fraud_indices]
    X_test = np.vstack([X_test_normal, X_test_fraud])
    y_test = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(X_test_fraud))])
    
    # Calculate reconstruction errors using model's built-in method
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
    errors = model.get_reconstruction_error(X_test_tensor)
    
    # Calculate threshold (95th percentile of normal errors)
    normal_errors = errors[y_test == 0]
    threshold = np.percentile(normal_errors, 95)
    
    # Make predictions
    y_pred = (errors > threshold).astype(int)
    
    # Calculate metrics

    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"✓ Optimal threshold: {threshold:.6f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # 6. Save everything
    print("\n6. Saving model and components...")
    
    # Create saved_models directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model using trainer's save method
    model_path = 'saved_models/autoencoder.pth'
    trainer.save_model(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save preprocessor (includes scaler AND parameters)
    scaler_path = 'saved_models/scaler.pkl'
    preprocessor.save_preprocessor(scaler_path)
    print(f"✓ Preprocessor saved (scaler + parameters)")
    
    # Save threshold
    threshold_path = 'saved_models/threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)
    print(f"✓ Threshold saved to: {threshold_path}")
    
    # Save feature names (for reference)
    features_path = 'saved_models/features.json'
    with open(features_path, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'num_features': len(feature_names),
            'top_categories': preprocessor.top_categories,
            'category_columns': preprocessor.category_columns
        }, f, indent=2)
    print(f"✓ Feature names saved to: {features_path}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModel Information:")
    print(f"  - Input features: {input_dim}")
    print(f"  - Architecture: 128-64-16 (encoder)")
    print(f"  - Dropout rate: 0.000287")
    print(f"  - Learning rate: 0.000608939")
    print(f"  - Weight decay: 1.5073e-05")
    print(f"  - Loss function: Huber Loss")
    print(f"  - Top categories: {preprocessor.top_categories}")
    print(f"  - Category columns: {preprocessor.category_columns}")
    print("\nNext steps:")
    print("1. Run batch processing: python scripts/02_batch_process.py")
    print("2. Test real-time: python scripts/03_realtime_test.py")
    print("\nSaved files:")
    print(f"  - Model: saved_models/autoencoder.pth")
    print(f"  - Preprocessor: saved_models/scaler.pkl")
    print(f"  - Threshold: saved_models/threshold.json")
    print(f"  - Features info: saved_models/features.json")
    
    return True

if __name__ == "__main__":
    train_simple_model()