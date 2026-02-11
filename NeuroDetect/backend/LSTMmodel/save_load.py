"""
Model Saving and Loading Utilities for LSTM Fraud Detection
"""

import torch
import joblib
import json
from datetime import datetime
from pathlib import Path

from .model import EnhancedFraudLSTM


def save_model(model, scaler, feature_names, results, output_dir='saved_models', 
               filename='enhanced_lstm_fraud_model.pth'):
    """
    Save the trained model, scaler, and metadata
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        results: Dictionary containing evaluation results
        output_dir: Directory to save the model
        filename: Name of the model file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full path for model file
    model_path = output_path / filename
    
    # Create model package
    model_package = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': feature_names,
        'model_class': EnhancedFraudLSTM.__name__,
        'model_config': {
            'input_dim': len(feature_names),
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.4,
            'bidirectional': True,
            'use_attention': True
        },
        'results': {
            'optimal_threshold': float(results['optimal_threshold']),
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'roc_auc': float(results['roc_auc']),
            'pr_auc': float(results['pr_auc']),
            'fraud_precision': float(results['fraud_precision']),
            'fraud_recall': float(results['fraud_recall']),
            'fraud_f1': float(results['fraud_f1'])
        },
        'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save model package
    joblib.dump(model_package, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save metadata separately as JSON
    metadata = {
        'feature_names': feature_names,
        'input_shape': [10, len(feature_names)],  # [sequence_length, n_features]
        'performance': {
            'accuracy': float(results['accuracy']),
            'f1_score': float(results['f1']),
            'fraud_precision': float(results['fraud_precision']),
            'fraud_recall': float(results['fraud_recall']),
            'fraud_f1': float(results['fraud_f1']),
            'roc_auc': float(results['roc_auc']),
            'pr_auc': float(results['pr_auc']),
            'optimal_threshold': float(results['optimal_threshold'])
        },
        'model_config': model_package['model_config'],
        'training_date': model_package['save_date']
    }
    
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    return str(model_path), str(metadata_path)


def load_model(filepath, device=None):
    """
    Load a saved model
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        model: Loaded PyTorch model
        scaler: Loaded StandardScaler
        feature_names: List of feature names
        model_config: Model configuration dictionary
        results: Training/evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model package
    model_package = joblib.load(filepath)
    
    # Extract components
    model_config = model_package['model_config']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    results = model_package.get('results', {})
    
    # Reconstruct model
    model = EnhancedFraudLSTM(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        bidirectional=model_config['bidirectional'],
        use_attention=model_config['use_attention']
    )
    
    # Load state dict
    model.load_state_dict(model_package['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {filepath}")
    print(f"  Device: {device}")
    print(f"  Input dim: {model_config['input_dim']}")
    print(f"  Save date: {model_package.get('save_date', 'Unknown')}")
    
    if results:
        print(f"\n  Model Performance:")
        print(f"    F1 Score: {results.get('f1', 'N/A')}")
        print(f"    Fraud F1: {results.get('fraud_f1', 'N/A')}")
        print(f"    ROC AUC: {results.get('roc_auc', 'N/A')}")
    
    return model, scaler, feature_names, model_config, results


def save_training_history(train_losses, val_losses, train_f1_scores, val_f1_scores, 
                          output_dir='saved_models', filename='training_history.json'):
    """
    Save training history to JSON file
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_f1_scores: List of training F1 scores
        val_f1_scores: List of validation F1 scores
        output_dir: Directory to save the history
        filename: Name of the history file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    history_path = output_path / filename
    
    history = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_f1_scores': [float(x) for x in train_f1_scores],
        'val_f1_scores': [float(x) for x in val_f1_scores],
        'num_epochs': len(train_losses),
        'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"✓ Training history saved to {history_path}")
    
    return str(history_path)


def load_training_history(filepath):
    """
    Load training history from JSON file
    
    Args:
        filepath: Path to the history file
        
    Returns:
        dict: Dictionary containing training history
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    print(f"✓ Training history loaded from {filepath}")
    print(f"  Number of epochs: {history['num_epochs']}")
    print(f"  Save date: {history.get('save_date', 'Unknown')}")
    
    return history


def export_model_for_inference(model, scaler, feature_names, optimal_threshold,
                               output_dir='saved_models', 
                               filename='lstm_inference_model.pth'):
    """
    Export a lightweight model package for inference only
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        optimal_threshold: Optimal decision threshold
        output_dir: Directory to save the model
        filename: Name of the model file
        
    Returns:
        str: Path to the saved model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / filename
    
    # Create lightweight package (no training results)
    inference_package = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': feature_names,
        'optimal_threshold': float(optimal_threshold),
        'model_config': {
            'input_dim': len(feature_names),
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.4,
            'bidirectional': True,
            'use_attention': True
        },
        'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_class': EnhancedFraudLSTM.__name__
    }
    
    torch.save(inference_package, model_path)
    print(f"✓ Inference model exported to {model_path}")
    
    return str(model_path)


def load_inference_model(filepath, device=None):
    """
    Load an inference-only model
    
    Args:
        filepath: Path to the saved inference model
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        scaler: Loaded scaler
        feature_names: List of feature names
        optimal_threshold: Optimal decision threshold
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load inference package
    package = torch.load(filepath, map_location=device)
    
    # Extract components
    model_config = package['model_config']
    scaler = package['scaler']
    feature_names = package['feature_names']
    optimal_threshold = package['optimal_threshold']
    
    # Reconstruct model
    model = EnhancedFraudLSTM(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        bidirectional=model_config['bidirectional'],
        use_attention=model_config['use_attention']
    )
    
    model.load_state_dict(package['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Inference model loaded from {filepath}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    
    return model, scaler, feature_names, optimal_threshold
