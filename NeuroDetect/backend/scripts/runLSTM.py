"""
LSTM Model Training Script
Run this to train the Enhanced LSTM Fraud Detection model
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing from AEmodel
from AEmodel.preprocessor import DataPreprocessor

# Import LSTM modules
from LSTMmodel.preprocessor import prepare_improved_lstm_data
from LSTMmodel.train import train_enhanced_lstm, comprehensive_evaluation
from LSTMmodel.save_load import save_model, save_training_history

def main(args):
    """Main training function"""
    
    # Set random seeds
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("="*80)
    print("ENHANCED LSTM FRAUD DETECTION TRAINING")
    print("="*80)
    
    # Step 1: Load raw data
    print(f"\n1. Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"   ✓ Loaded {len(df)} transactions")
    
    # Step 2: Preprocess data using AEmodel preprocessor
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Keep only required columns
    required_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                    'is_fraud', 'trans_date_trans_time', 'category', 'gender']
    
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    
    X_scaled, y, feature_names = preprocessor.preprocess(df, is_training=True)
    
    # Reconstruct dataframe for LSTM preprocessing
    df_preprocessed = pd.DataFrame(X_scaled, columns=feature_names)
    df_preprocessed['is_fraud'] = y
    
    print(f"   ✓ Preprocessed {len(df_preprocessed)} transactions")
    print(f"   ✓ Features: {len(feature_names)}")
    print(f"   ✓ Fraud percentage: {(y.sum()/len(y))*100:.4f}%")
    
    # Step 3: Prepare balanced sequential data for LSTM
    print("\n3. Preparing balanced sequential data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_improved_lstm_data(
        df_preprocessed,
        sequence_length=args.sequence_length
    )
    
    # Step 4: Train enhanced LSTM model
    print("\n4. Training Enhanced LSTM model...")
    model, scaler, train_losses, val_losses, train_f1, val_f1 = train_enhanced_lstm(
        X_train, y_train, X_val, y_val, feature_names,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Step 5: Comprehensive evaluation
    print("\n5. Evaluating on test set...")
    results = comprehensive_evaluation(model, scaler, X_test, y_test, threshold_tuning=True)
    
    # Step 6: Save model
    print("\n6. Saving model...")
    model_path, metadata_path = save_model(
        model, scaler, feature_names, results, 
        output_dir=args.output_dir,
        filename=args.output_model
    )
    
    # Step 7: Save training history
    print("\n7. Saving training history...")
    history_path = save_training_history(
        train_losses, val_losses, train_f1, val_f1,
        output_dir=args.output_dir,
        filename='lstm_training_history.json'
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Training history saved to: {history_path}")
    
    print(f"\nFinal Performance:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  F1 Score:    {results['f1']:.4f}")
    print(f"  Fraud F1:    {results['fraud_f1']:.4f}")
    print(f"  ROC AUC:     {results['roc_auc']:.4f}")
    print(f"  PR AUC:      {results['pr_auc']:.4f}")
    print(f"  Threshold:   {results['optimal_threshold']:.4f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced LSTM Fraud Detection Model")
    
    # Data arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="dataset/fraudTrain.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../saved_models",
        help="Directory to save model and metadata"
    )
    parser.add_argument(
        "--output-model", 
        type=str, 
        default="enhanced_lstm_fraud_model.pth", 
        help="Model filename"
    )
    
    # Training arguments
    parser.add_argument("--sequence-length", type=int, default=10, help="Length of transaction sequences")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Initial learning rate")

    args = parser.parse_args()
    main(args)