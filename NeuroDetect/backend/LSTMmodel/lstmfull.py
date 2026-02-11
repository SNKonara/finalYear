"""
Enhanced LSTM Fraud Detection - Main Training Script
This script uses the organized modules to train the enhanced LSTM model
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import from organized modules
from LSTMmodel.preprocessor import prepare_improved_lstm_data
from LSTMmodel.train import train_enhanced_lstm, comprehensive_evaluation
from LSTMmodel.save_load import save_model, save_training_history

# Set random seeds for reproducibility
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def run_enhanced_lstm_training(
    df_preprocessed,
    sequence_length=10,
    epochs=100,
    batch_size=128,
    learning_rate=0.0005,
    output_dir='saved_models'
):
    """
    Main function to run the enhanced LSTM training pipeline
    
    Args:
        df_preprocessed: Preprocessed dataframe with 'is_fraud' column
        sequence_length: Length of transaction sequences
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        output_dir: Directory to save the trained model
        
    Returns:
        model: Trained model
        scaler: Fitted scaler
        results: Evaluation results dictionary
    """
    print("="*80)
    print("ENHANCED LSTM FOR FRAUD DETECTION WITH CLASS IMBALANCE HANDLING")
    print("="*80)
    
    # Step 1: Prepare balanced sequential data
    print("\n1. Preparing balanced sequential data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_improved_lstm_data(
        df_preprocessed,
        sequence_length=sequence_length
    )
    
    # Step 2: Train enhanced model
    print("\n2. Training enhanced LSTM model...")
    model, scaler, train_losses, val_losses, train_f1, val_f1 = train_enhanced_lstm(
        X_train, y_train, X_val, y_val, feature_names,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Step 3: Comprehensive evaluation
    print("\n3. Comprehensive evaluation on test set...")
    results = comprehensive_evaluation(model, scaler, X_test, y_test, threshold_tuning=True)
    
    # Step 4: Save model
    print("\n4. Saving model...")
    model_path, metadata_path = save_model(
        model, scaler, feature_names, results, 
        output_dir=output_dir,
        filename='enhanced_lstm_fraud_model.pth'
    )
    
    # Step 5: Save training history
    print("\n5. Saving training history...")
    history_path = save_training_history(
        train_losses, val_losses, train_f1, val_f1,
        output_dir=output_dir,
        filename='lstm_training_history.json'
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Training history saved to: {history_path}")
    
    print(f"\nFinal Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Fraud F1: {results['fraud_f1']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print(f"  PR AUC: {results['pr_auc']:.4f}")
    
    return model, scaler, results


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use this training script:
    
    1. First, preprocess your data using your preprocessing function
    2. Then call run_enhanced_lstm_training with the preprocessed data
    
    Example:
    --------
    from your_preprocessing_module import preprocessing
    
    # Load data
    df = pd.read_csv('dataset/fraudTrain.csv')
    
    # Preprocess
    df_preprocessed = preprocessing(df)
    
    # Train model
    model, scaler, results = run_enhanced_lstm_training(
        df_preprocessed=df_preprocessed,
        sequence_length=10,
        epochs=100,
        batch_size=128,
        learning_rate=0.0005,
        output_dir='saved_models'
    )
    """
    
    print("\n" + "="*80)
    print("LSTM TRAINING SCRIPT")
    print("="*80)
    print("\nThis script is ready to use!")
    print("\nTo train the model:")
    print("1. Import your preprocessing function")
    print("2. Preprocess your data")
    print("3. Call run_enhanced_lstm_training(df_preprocessed)")
    print("\nSee the example in the __main__ block above.")
    print("="*80)
