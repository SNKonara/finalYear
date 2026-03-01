"""
Quick script to optimize threshold on already trained SNN model
Use this to find better threshold without retraining
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    print("ERROR: snnTorch not installed. Installing...")
    os.system("pip install snntorch")
    import snntorch as snn
    from snntorch import surrogate

# Import model class from snnfull
sys.path.append(os.path.dirname(__file__))
from snnfull import SpikingFraudDetector, DataEngineer, optimize_threshold, plot_threshold_analysis, plot_confusion_matrix


def load_model_and_data(model_dir):
    """Load trained model and prepare data"""
    print(f"Loading model from: {model_dir}")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(model_dir, 'model.pth')
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate model
    model = SpikingFraudDetector(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        output_size=checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    config = checkpoint['config']
    print(f"✓ Model loaded")
    print(f"  Time steps: {config['time_steps']}")
    print(f"  Hidden size: {config['hidden_size']}")
    
    # Load and prepare data
    print(f"\nPreparing validation data...")
    data_path = "C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv"
    
    engineer = DataEngineer(data_path)
    X, y = engineer.load_and_engineer_features(sample_size=config.get('sample_size'))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        engineer.stratified_split_and_normalize(X, y)
    
    # Create validation loader
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    batch_size = config.get('batch_size', 1024)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return model, val_loader, test_loader, config, device, model_dir


def evaluate_with_threshold(model, test_loader, num_steps, threshold, device):
    """Evaluate model with specific threshold"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data, num_steps)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(target.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Apply threshold
    y_pred = (all_probs >= threshold).astype(int)
    
    return {
        'accuracy': np.mean(y_pred == all_labels),
        'precision': precision_score(all_labels, y_pred, zero_division=0),
        'recall': recall_score(all_labels, y_pred, zero_division=0),
        'f1': f1_score(all_labels, y_pred, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'predictions': y_pred,
        'labels': all_labels,
        'probabilities': all_probs
    }


def main():
    """Optimize threshold on trained model"""
    
    # Find most recent model directory
    model_base = 'snn_models'
    if not os.path.exists(model_base):
        print(f"ERROR: No models found in {model_base}/")
        print("Please train a model first using snnfull.py")
        return
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(model_base) if d.startswith('snn_')]
    if not model_dirs:
        print(f"ERROR: No SNN models found in {model_base}/")
        return
    
    # Sort by date and get most recent
    model_dirs.sort(reverse=True)
    print("\nAvailable models:")
    for i, d in enumerate(model_dirs[:5], 1):
        print(f"  {i}. {d}")
    
    choice = input(f"\nSelect model (1-{min(5, len(model_dirs))}) or press Enter for most recent: ").strip()
    
    if choice == "":
        selected_dir = model_dirs[0]
    else:
        try:
            idx = int(choice) - 1
            selected_dir = model_dirs[idx]
        except (ValueError, IndexError):
            print("Invalid choice. Using most recent model.")
            selected_dir = model_dirs[0]
    
    model_path = os.path.join(model_base, selected_dir)
    
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION ON TRAINED MODEL")
    print("="*60)
    print(f"Model: {selected_dir}")
    
    # Load model and data
    model, val_loader, test_loader, config, device, model_dir = load_model_and_data(model_path)
    num_steps = config['time_steps']
    
    # Optimize threshold on validation set
    print("\n" + "="*60)
    print("OPTIMIZING THRESHOLD (on validation set)")
    print("="*60)
    
    # Try different strategies
    strategies = ['f1', 'f2', 'f0.5']
    strategy_names = {
        'f1': 'Balanced (F1)',
        'f2': 'Recall-Focused (F2)',
        'f0.5': 'Precision-Focused (F0.5)'
    }
    
    print("\nAvailable strategies:")
    for i, (strat, name) in enumerate(strategy_names.items(), 1):
        print(f"  {i}. {name}")
    
    strat_choice = input(f"\nSelect strategy (1-3) or press Enter for F1: ").strip()
    
    if strat_choice == "":
        selected_strategy = 'f1'
    else:
        try:
            strat_idx = int(strat_choice) - 1
            selected_strategy = list(strategies)[strat_idx]
        except (ValueError, IndexError):
            print("Invalid choice. Using F1.")
            selected_strategy = 'f1'
    
    threshold_results = optimize_threshold(
        model, val_loader, num_steps, device, strategy=selected_strategy
    )
    optimal_threshold = threshold_results['optimal_threshold']
    
    # Evaluate on test set with optimized threshold
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_with_threshold(
        model, test_loader, num_steps, optimal_threshold, device
    )
    
    print(f"\nTest Results (Threshold = {optimal_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    
    # Compare with default threshold
    default_metrics = evaluate_with_threshold(
        model, test_loader, num_steps, 0.5, device
    )
    
    print(f"\nComparison with Default Threshold (0.5):")
    print(f"  Default - Precision: {default_metrics['precision']:.4f}, Recall: {default_metrics['recall']:.4f}, F1: {default_metrics['f1']:.4f}")
    print(f"  Optimal - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    if default_metrics['f1'] > 0:
        improvement = (test_metrics['f1'] - default_metrics['f1']) / default_metrics['f1'] * 100
        print(f"  F1 Improvement: {improvement:+.1f}%")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Update model with new threshold
    checkpoint_path = os.path.join(model_dir, 'model.pth')
    checkpoint = torch.load(checkpoint_path)
    checkpoint['optimal_threshold'] = optimal_threshold
    checkpoint['threshold_strategy'] = selected_strategy
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Model updated with optimal threshold: {checkpoint_path}")
    
    # Save threshold info
    threshold_path = os.path.join(model_dir, 'optimal_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({
            'optimal_threshold': float(optimal_threshold),
            'strategy': selected_strategy,
            'validation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                  for k, v in threshold_results['metrics'].items()},
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                           for k, v in test_metrics.items() 
                           if k not in ['predictions', 'labels', 'probabilities']}
        }, f, indent=2)
    print(f"✓ Threshold info saved: {threshold_path}")
    
    # Save plots
    threshold_plot_path = os.path.join(model_dir, 'threshold_analysis_updated.png')
    plot_threshold_analysis(threshold_results, threshold_plot_path)
    
    cm_path = os.path.join(model_dir, 'confusion_matrix_updated.png')
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], cm_path)
    
    print(f"\n✓ All results saved to: {model_dir}")
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
