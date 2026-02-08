"""
SECOND SCRIPT TO RUN: Evaluate the trained model
Run this AFTER 01_train_model.py
"""
import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AEmodel.preprocessor import DataPreprocessor

class FraudAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, 
                 latent_dim=16, dropout_rate=0.000287):
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim2, latent_dim),
            torch.nn.ReLU()
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim2, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim1, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def evaluate_model():
    """Evaluate model with multiple thresholds - matches your notebook"""
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # 1. Load saved components
    print("\n1. Loading saved model components...")
    try:
        # Load model
        checkpoint = torch.load('C:/finalYear/saved_models/autoencoder.pth')
        input_dim = checkpoint['input_dim']
        
        model = FraudAutoencoder(
            input_dim=input_dim,
            hidden_dim1=128,
            hidden_dim2=64,
            latent_dim=16,
            dropout_rate=0.000287
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler
        scaler = joblib.load('C:/finalYear/saved_models/scaler.pkl')
        
        # Load threshold
        with open('C:/finalYear/saved_models/threshold.json', 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data['threshold']
        
        # Load features info
        with open('C:/finalYear/saved_models/features.json', 'r') as f:
            features_data = json.load(f)
            top_categories = features_data.get('top_categories', {})
            category_columns = features_data.get('category_columns', [])
        
        print("✓ All components loaded successfully")
        
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        print("Please run train_model.py first to train and save the model")
        return
    
    # 2. Load dataset for evaluation
    print("\n2. Loading dataset for evaluation...")
    try:
        df = pd.read_csv('C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv')
        print(f"✓ Dataset loaded: {df.shape[0]} rows")
    except FileNotFoundError:
        print("✗ ERROR: Could not find dataset file")
        print("Please ensure 'C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv' exists")
        return
    
    # 3. Prepare test data (EXACTLY like your notebook)
    print("\n3. Preparing test dataset...")
    preprocessor = DataPreprocessor()
    preprocessor.scaler = scaler
    preprocessor.top_categories = top_categories
    preprocessor.category_columns = category_columns
    
    # Preprocess
    X_scaled, y, _ = preprocessor.preprocess(df, is_training=False)
    
    # Get indices
    normal_data = np.where(y == 0)[0]
    fraud_data = np.where(y == 1)[0]
    
    # Create balanced test set (same logic as notebook)
    np.random.seed(42)
    
    # Use 50,000 normal transactions or 10x fraud count (whichever is smaller)
    test_normal_size = min(50000, len(fraud_data) * 10)
    test_normal_indices = np.random.choice(
        normal_data,
        test_normal_size,
        replace=False
    )
    
    # Use all fraud
    test_fraud_indices = fraud_data
    
    # Combine
    X_test_normal = X_scaled[test_normal_indices]
    y_test_normal = y[test_normal_indices]
    
    X_test_fraud = X_scaled[test_fraud_indices]
    y_test_fraud = y[test_fraud_indices]
    
    X_test = np.vstack([X_test_normal, X_test_fraud])
    y_test = np.hstack([y_test_normal, y_test_fraud])
    
    print(f"✓ Test dataset prepared:")
    print(f"  Normal transactions: {len(X_test_normal)}")
    print(f"  Fraud transactions: {len(X_test_fraud)}")
    print(f"  Total test samples: {len(X_test)}")
    print(f"  Fraud percentage in test: {(len(test_fraud_indices) / len(X_test)) * 100:.2f}%")
    
    # 4. Calculate reconstruction errors
    print("\n4. Calculating reconstruction errors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        errors = reconstruction_errors.cpu().numpy()
    
    # Get normal errors for threshold calculation
    normal_errors = errors[y_test == 0]
    
    # 5. Evaluate with multiple thresholds (EXACTLY like your notebook)
    print("\n5. Evaluating with multiple thresholds...")
    print("\n" + "="*70)
    print("MODEL EVALUATION - TESTING MULTIPLE THRESHOLDS")
    print("="*70)
    
    # Calculate ROC AUC (threshold-independent)
    roc_auc = roc_auc_score(y_test, errors)
    print(f"ROC AUC Score (threshold-independent): {roc_auc:.4f}\n")
    
    # Test different percentiles
    threshold_percentiles = [90, 95, 98, 99]
    results_dict = {}
    
    for percentile in threshold_percentiles:
        # Calculate threshold
        threshold = np.percentile(normal_errors, percentile)
        y_pred = (errors > threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store results
        results_dict[percentile] = {
            'threshold': threshold,
            'y_pred': y_pred,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print results
        print("="*70)
        print(f"THRESHOLD PERCENTILE: {percentile}%")
        print("="*70)
        print(f"Threshold Value: {threshold:.6f}")
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Normal:  {cm[0,0]:6d}  |  False Fraud: {cm[0,1]:6d}")
        print(f"  False Normal: {cm[1,0]:6d}  |  True Fraud:  {cm[1,1]:6d}")
        print()
    
    # 6. Find best threshold
    best_percentile = max(results_dict.keys(), key=lambda k: results_dict[k]['f1_score'])
    best_results = results_dict[best_percentile]
    
    print("="*70)
    print("BEST THRESHOLD SELECTED")
    print("="*70)
    print(f"Best Percentile: {best_percentile}%")
    print(f"Best Threshold: {best_results['threshold']:.6f}")
    print(f"Best F1-Score: {best_results['f1_score']:.4f}")
    print(f"Precision: {best_results['precision']:.4f}")
    print(f"Recall: {best_results['recall']:.4f}")
    print("="*70)
    
    # 7. Detailed classification report for best threshold
    print(f"\nDetailed Classification Report (Percentile {best_percentile}%):")
    print(classification_report(y_test, best_results['y_pred'], 
                                target_names=['Normal', 'Fraud']))
    
    # 8. Save evaluation results
    print("\n6. Saving evaluation results...")
    os.makedirs('results/evaluation', exist_ok=True)
    
    # Save metrics summary
    summary = {
        'best_percentile': int(best_percentile),
        'best_threshold': float(best_results['threshold']),
        'best_f1_score': float(best_results['f1_score']),
        'best_precision': float(best_results['precision']),
        'best_recall': float(best_results['recall']),
        'roc_auc': float(roc_auc),
        'test_set_size': int(len(X_test)),
        'normal_samples': int(len(X_test_normal)),
        'fraud_samples': int(len(X_test_fraud)),
        'fraud_percentage': float((len(X_test_fraud) / len(X_test)) * 100)
    }
    
    with open('results/evaluation/evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    detailed_results = {}
    for perc in threshold_percentiles:
        detailed_results[f'percentile_{perc}'] = {
            'threshold': float(results_dict[perc]['threshold']),
            'precision': float(results_dict[perc]['precision']),
            'recall': float(results_dict[perc]['recall']),
            'f1_score': float(results_dict[perc]['f1_score']),
            'confusion_matrix': results_dict[perc]['confusion_matrix'].tolist()
        }
    
    with open('results/evaluation/detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("✓ Evaluation results saved to 'results/evaluation/' folder")
    
    # 9. Compare with current threshold
    current_y_pred = (errors > threshold).astype(int)  # 'threshold' is from saved file
    current_precision = precision_score(y_test, current_y_pred, zero_division=0)
    current_recall = recall_score(y_test, current_y_pred, zero_division=0)
    current_f1 = f1_score(y_test, current_y_pred, zero_division=0)
    
    print("\n" + "="*70)
    print("CURRENT THRESHOLD vs BEST THRESHOLD COMPARISON")
    print("="*70)
    print(f"Current Threshold (from saved file): {threshold:.6f}")
    print(f"  Precision: {current_precision:.4f}")
    print(f"  Recall:    {current_recall:.4f}")
    print(f"  F1-Score:  {current_f1:.4f}")
    print()
    print(f"Best Threshold ({best_percentile}% percentile): {best_results['threshold']:.6f}")
    print(f"  Precision: {best_results['precision']:.4f}")
    print(f"  Recall:    {best_results['recall']:.4f}")
    print(f"  F1-Score:  {best_results['f1_score']:.4f}")
    
    # Update threshold if better one found
    if best_results['f1_score'] > current_f1:
        print(f"\n⚠️  Better threshold found! Updating saved threshold...")
        with open('saved_models/threshold.json', 'w') as f:
            json.dump({'threshold': float(best_results['threshold'])}, f)
        print(f"✓ Updated threshold to {best_results['threshold']:.6f}")
    
    return summary

if __name__ == "__main__":
    evaluate_model()