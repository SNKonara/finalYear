"""
Spiking Neural Network (SNN) Training for Fraud Detection
Uses snnTorch library with LIF neurons
Professional pipeline with debug, hyperparameter tuning, and final training
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import snnTorch - we'll need to install it
try:
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import utils
except ImportError:
    print("ERROR: snnTorch not installed. Installing...")
    os.system("pip install snntorch")
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import utils

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class DataEngineer:
    """Handle data loading, preprocessing, and feature engineering"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_engineer_features(self, sample_size=None):
        """Load data and create same 24 features as other models"""
        print(f"\nLoading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        if sample_size:
            print(f"Sampling {sample_size:,} rows...")
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        print(f"Dataset loaded: {len(df):,} rows")
        print(f"Fraud rate: {df['is_fraud'].mean()*100:.2%}")
        
        # Feature engineering (matching other models)
        print("\nEngineering features...")
        
        # Parse datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # Time features
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day_of_month'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month
        
        # Distance calculation
        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 + 
            (df['long'] - df['merch_long'])**2
        )
        
        # Derived features
        df['log_amt'] = np.log1p(df['amt'])
        df['amt_per_pop'] = df['amt'] / (df['city_pop'] + 1)
        
        # Cyclical time encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Category encoding (one-hot top categories)
        top_categories = [
            'gas_transport', 'grocery_pos', 'home', 'shopping_pos',
            'kids_pets', 'shopping_net', 'entertainment', 'food_dining'
        ]
        
        for cat in top_categories:
            df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
        
        # Gender encoding
        df['gender_M'] = (df['gender'] == 'M').astype(int)
        
        # Select 24 features (matching features.json)
        self.feature_names = [
            'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'hour', 'day_of_week', 'day_of_month', 'month', 'distance',
            'log_amt', 'amt_per_pop', 'hour_sin', 'hour_cos',
            'cat_food_dining', 'cat_gas_transport', 'cat_grocery_pos',
            'cat_home', 'cat_kids_pets', 'cat_other', 'cat_shopping_net',
            'cat_shopping_pos', 'gender_M'
        ]
        
        # Handle "other" category
        other_cats = set(df['category'].unique()) - set(top_categories)
        df['cat_other'] = df['category'].isin(other_cats).astype(int)
        
        X = df[self.feature_names].values
        y = df['is_fraud'].values
        
        print(f"✓ Features created: {X.shape[1]} features")
        print(f"✓ Feature names: {self.feature_names}")
        
        return X, y
    
    def stratified_split_and_normalize(self, X, y, test_size=0.2, val_size=0.1):
        """Stratified split and normalization"""
        print("\nSplitting data (stratified)...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        print(f"Train set: {len(X_train):,} samples ({y_train.mean()*100:.2%} fraud)")
        print(f"Val set: {len(X_val):,} samples ({y_val.mean()*100:.2%} fraud)")
        print(f"Test set: {len(X_test):,} samples ({y_test.mean()*100:.2%} fraud)")
        
        # Normalize using StandardScaler (fit on train only)
        print("\nNormalizing features...")
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        print("✓ Normalization complete")
        
        return (X_train_norm, y_train), (X_val_norm, y_val), (X_test_norm, y_test)


class SpikingFraudDetector(nn.Module):
    """
    Spiking Neural Network for Fraud Detection
    Architecture:
    - Input: 24 features
    - FC1: 24 → 64, LIF
    - FC2: 64 → 64, LIF
    - FC3: 64 → 2 (output layer)
    """
    
    def __init__(self, input_size=24, hidden_size=64, output_size=2, 
                 beta=0.95, spike_grad=None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Spike gradient surrogate (for backprop through spikes)
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()
        
        # Layer 1: FC + LIF
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 2: FC + LIF
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 3: FC (output)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, num_steps):
        """
        Forward pass through time
        x: (batch, features)
        num_steps: number of time steps
        Returns: (batch, output_size) - spike counts
        """
        batch_size = x.size(0)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record output spikes
        spk_rec = []
        
        # Simulate through time
        for _ in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output layer (no spiking)
            out = self.fc3(spk2)
            spk_rec.append(out)
        
        # Sum outputs over time (spike count)
        spk_rec = torch.stack(spk_rec, dim=0)  # (time, batch, output)
        output = torch.sum(spk_rec, dim=0)  # (batch, output)
        
        return output


class SNNTrainer:
    """Professional SNN trainer with all bells and whistles"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = None  # Will be set based on class weights
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
    def set_class_weights(self, y_train):
        """Calculate and set class weights for imbalanced data"""
        print("\nCalculating class weights...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"Class 0 (normal) weight: {class_weights[0]:.4f}")
        print(f"Class 1 (fraud) weight: {class_weights[1]:.4f}")
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    def train_epoch(self, train_loader, num_steps):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data, num_steps)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Predictions
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return avg_loss, accuracy, f1
    
    def evaluate(self, val_loader, num_steps):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                output = self.model(data, num_steps)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Get probabilities using softmax
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Fraud probability
                
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate comprehensive metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self, train_loader, val_loader, num_steps, epochs, patience=3):
        """Full training loop with early stopping"""
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, num_steps)
            
            # Validate
            val_metrics = self.evaluate(val_loader, num_steps)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print("  ✓ New best model!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping triggered (patience={patience})")
                    break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Best model restored (F1: {best_val_f1:.4f})")
        
        return self.history


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(history['train_f1'], label='Train')
    axes[2].plot(history['val_f1'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def optimize_threshold(model, val_loader, num_steps, device, strategy='f1'):
    """Optimize classification threshold for better precision/recall balance
    
    Args:
        model: Trained SNN model
        val_loader: Validation data loader
        num_steps: Number of time steps for SNN
        device: torch device
        strategy: 'f1' (default), 'f2' (favor recall), 'f0.5' (favor precision), 'balanced'
    
    Returns:
        dict: Contains optimal_threshold, metrics at that threshold, and threshold analysis
    """
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    # Get all predictions
    print("Collecting predictions...")
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            output = model(data, num_steps)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Fraud probability
            all_labels.extend(target.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Test thresholds from 0.1 to 0.9
    print("Testing thresholds from 0.1 to 0.9...")
    thresholds = np.arange(0.1, 0.91, 0.01)
    results = []
    
    for thresh in thresholds:
        y_pred = (all_probs >= thresh).astype(int)
        
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        
        # F-beta scores for different strategies
        from sklearn.metrics import fbeta_score
        f2 = fbeta_score(all_labels, y_pred, beta=2, zero_division=0)  # Favor recall
        f05 = fbeta_score(all_labels, y_pred, beta=0.5, zero_division=0)  # Favor precision
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'f0.5': f05
        })
    
    # Find optimal threshold based on strategy
    if strategy == 'f1':
        best_result = max(results, key=lambda x: x['f1'])
        metric_name = 'F1 Score'
    elif strategy == 'f2':
        best_result = max(results, key=lambda x: x['f2'])
        metric_name = 'F2 Score (Recall-focused)'
    elif strategy == 'f0.5':
        best_result = max(results, key=lambda x: x['f0.5'])
        metric_name = 'F0.5 Score (Precision-focused)'
    elif strategy == 'balanced':
        # Maximize geometric mean of precision and recall
        for r in results:
            r['gmean'] = np.sqrt(r['precision'] * r['recall'])
        best_result = max(results, key=lambda x: x['gmean'])
        metric_name = 'G-Mean (Balanced)'
    else:
        best_result = max(results, key=lambda x: x['f1'])
        metric_name = 'F1 Score'
    
    optimal_threshold = best_result['threshold']
    
    print(f"\nOptimization Strategy: {strategy.upper()}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"\nMetrics at Optimal Threshold:")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1 Score:  {best_result['f1']:.4f}")
    print(f"  {metric_name}: {best_result.get(strategy, best_result['f1']):.4f}")
    
    # Show comparison with default threshold (0.5)
    default_result = next((r for r in results if abs(r['threshold'] - 0.5) < 0.01), results[40])
    print(f"\nComparison with Default Threshold (0.5):")
    print(f"  Default - Precision: {default_result['precision']:.4f}, Recall: {default_result['recall']:.4f}, F1: {default_result['f1']:.4f}")
    print(f"  Optimal - Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}, F1: {best_result['f1']:.4f}")
    print(f"  Improvement: Precision: {(best_result['precision']-default_result['precision'])*100:+.2f}%, "
          f"Recall: {(best_result['recall']-default_result['recall'])*100:+.2f}%, "
          f"F1: {(best_result['f1']-default_result['f1'])*100:+.2f}%")
    
    return {
        'optimal_threshold': optimal_threshold,
        'metrics': best_result,
        'threshold_analysis': results,
        'strategy': strategy
    }


def plot_threshold_analysis(threshold_results, save_path):
    """Plot precision-recall curve vs threshold"""
    results = threshold_results['threshold_analysis']
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    ax1.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax1.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    ax1.axvline(x=threshold_results['optimal_threshold'], color='red', 
                linestyle=':', label=f'Optimal ({threshold_results["optimal_threshold"]:.3f})', linewidth=2)
    ax1.axvline(x=0.5, color='gray', linestyle=':', label='Default (0.5)', linewidth=1)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Precision-Recall Curve
    ax2.plot(recalls, precisions, linewidth=2, color='blue')
    opt_metrics = threshold_results['metrics']
    ax2.scatter([opt_metrics['recall']], [opt_metrics['precision']], 
                color='red', s=200, zorder=5, label=f'Optimal (T={threshold_results["optimal_threshold"]:.3f})', marker='*')
    
    # Mark default threshold
    default_idx = next((i for i, r in enumerate(results) if abs(r['threshold'] - 0.5) < 0.01), 40)
    ax2.scatter([recalls[default_idx]], [precisions[default_idx]], 
                color='gray', s=100, zorder=5, label='Default (T=0.5)', marker='o')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Threshold analysis plot saved to: {save_path}")
    plt.close()


def run_experiment(config, save_dir='snn_models'):
    """Run a single training experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"snn_{config['name']}_{timestamp}"
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"EXPERIMENT: {config['name']}")
    print("="*60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # 1. Data Engineering
    data_path = "C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv"
    engineer = DataEngineer(data_path)
    
    X, y = engineer.load_and_engineer_features(sample_size=config['sample_size'])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        engineer.stratified_split_and_normalize(X, y)
    
    # 2. Create DataLoaders
    print("\nCreating DataLoaders...")
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    # Use pin_memory for faster GPU transfer
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=0, pin_memory=pin_memory)
    
    print(f"✓ DataLoaders created (batch_size={config['batch_size']})")
    
    # 3. Create Model with GPU
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Training will be VERY slow on CPU.")
        print("Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"GPU CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Current GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"✓ GPU acceleration enabled!")
    else:
        print(f"✗ CPU mode (slow)")
    print(f"{'='*60}")
    
    model = SpikingFraudDetector(
        input_size=24,
        hidden_size=config['hidden_size'],
        output_size=2
    )
    
    print(f"\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Create Trainer and set class weights
    trainer = SNNTrainer(model, device=device, learning_rate=config['lr'])
    trainer.set_class_weights(y_train)
    
    # 5. Train
    history = trainer.train(
        train_loader, val_loader,
        num_steps=config['time_steps'],
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # 6. Optimize Threshold on Validation Set
    threshold_results = optimize_threshold(
        model, val_loader, config['time_steps'], device, strategy='f1'
    )
    optimal_threshold = threshold_results['optimal_threshold']
    
    # 7. Evaluate on test set with OPTIMIZED threshold
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET (with Optimized Threshold)")
    print("="*60)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            output = model(data, config['time_steps'])
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(target.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Apply optimized threshold
    y_pred_optimal = (all_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics with optimized threshold
    test_metrics = {
        'accuracy': np.mean(y_pred_optimal == all_labels),
        'precision': precision_score(all_labels, y_pred_optimal, zero_division=0),
        'recall': recall_score(all_labels, y_pred_optimal, zero_division=0),
        'f1': f1_score(all_labels, y_pred_optimal, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'predictions': y_pred_optimal,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    print(f"\nTest Results (Threshold = {optimal_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    
    # Also show results with default threshold for comparison
    y_pred_default = (all_probs >= 0.5).astype(int)
    default_metrics = {
        'precision': precision_score(all_labels, y_pred_default, zero_division=0),
        'recall': recall_score(all_labels, y_pred_default, zero_division=0),
        'f1': f1_score(all_labels, y_pred_default, zero_division=0)
    }
    
    print(f"\nComparison with Default Threshold (0.5):")
    print(f"  Default - Precision: {default_metrics['precision']:.4f}, Recall: {default_metrics['recall']:.4f}, F1: {default_metrics['f1']:.4f}")
    print(f"  Optimal - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    improvement = (test_metrics['f1'] - default_metrics['f1']) / default_metrics['f1'] * 100 if default_metrics['f1'] > 0 else 0
    print(f"  F1 Improvement: {improvement:+.1f}%")
    
    # 8. Save everything
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model with optimal threshold
    model_path = os.path.join(exp_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_size': 24,
        'hidden_size': config['hidden_size'],
        'output_size': 2,
        'optimal_threshold': optimal_threshold,
        'threshold_strategy': threshold_results['strategy']
    }, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save optimal threshold separately for easy access
    threshold_path = os.path.join(exp_dir, 'optimal_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({
            'optimal_threshold': float(optimal_threshold),
            'strategy': threshold_results['strategy'],
            'validation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                  for k, v in threshold_results['metrics'].items()}
        }, f, indent=2)
    print(f"✓ Optimal threshold saved: {threshold_path}")
    
    # Save scaler
    scaler_path = os.path.join(exp_dir, 'scaler.pkl')
    import joblib
    joblib.dump(engineer.scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save training history
    history_path = os.path.join(exp_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    # Save test metrics (including optimal threshold)
    metrics_path = os.path.join(exp_dir, 'test_metrics.json')
    metrics_to_save = {
        'optimal_threshold': float(optimal_threshold),
        'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                        for k, v in test_metrics.items() 
                        if k not in ['predictions', 'labels', 'probabilities']},
        'threshold_validation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                        for k, v in threshold_results['metrics'].items()}
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"✓ Test metrics saved: {metrics_path}")
    
    # Save feature names
    features_path = os.path.join(exp_dir, 'features.json')
    with open(features_path, 'w') as f:
        json.dump({'feature_names': engineer.feature_names, 'num_features': 24}, f, indent=2)
    print(f"✓ Features saved: {features_path}")
    
    # Plot training curves
    plot_path = os.path.join(exp_dir, 'training_curves.png')
    plot_training_history(history, plot_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(exp_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], cm_path)
    
    # Plot threshold analysis
    threshold_plot_path = os.path.join(exp_dir, 'threshold_analysis.png')
    plot_threshold_analysis(threshold_results, threshold_plot_path)
    
    print(f"\n✓ All results saved to: {exp_dir}")
    
    return {
        'exp_dir': exp_dir,
        'test_metrics': test_metrics,
        'history': history
    }


def main():
    """Main training pipeline - 3 stages"""
    
    print("="*60)
    print("SPIKING NEURAL NETWORK TRAINING PIPELINE")
    print("="*60)
    
    # Check GPU availability at start
    if torch.cuda.is_available():
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠ WARNING: No GPU detected. Training will be extremely slow!")
        print("  Consider using Google Colab or a machine with CUDA-capable GPU.")
    
    os.makedirs('snn_models', exist_ok=True)
    
    # ========================================
    # STAGE 1: DEBUG (100K samples, 5 epochs)
    # ========================================
    print("\n\n")
    print("█"*60)
    print("STAGE 1: DEBUG")
    print("█"*60)
    
    debug_config = {
        'name': 'debug',
        'sample_size': 100000,
        'epochs': 5,
        'time_steps': 25,
        'batch_size': 1024,
        'hidden_size': 64,
        'lr': 0.001,
        'patience': 999  # No early stopping for debug
    }
    
    debug_results = run_experiment(debug_config)
    
    # Check if loss decreased
    if debug_results['history']['train_loss'][-1] < debug_results['history']['train_loss'][0]:
        print("\n✓ DEBUG PASSED: Loss decreased!")
    else:
        print("\n✗ DEBUG FAILED: Loss did not decrease. Check your setup.")
        return
    
    # ========================================
    # STAGE 2: HYPERPARAMETER TUNING
    # ========================================
    print("\n\n")
    print("█"*60)
    print("STAGE 2: HYPERPARAMETER TUNING")
    print("█"*60)
    
    tuning_configs = [
        {
            'name': 'tune_timesteps20_hidden64',
            'sample_size': 500000,
            'epochs': 10,
            'time_steps': 20,
            'batch_size': 1024,
            'hidden_size': 64,
            'lr': 0.001,
            'patience': 3
        },
        {
            'name': 'tune_timesteps30_hidden64',
            'sample_size': 500000,
            'epochs': 10,
            'time_steps': 30,
            'batch_size': 1024,
            'hidden_size': 64,
            'lr': 0.001,
            'patience': 3
        },
        {
            'name': 'tune_timesteps25_hidden128',
            'sample_size': 500000,
            'epochs': 10,
            'time_steps': 25,
            'batch_size': 1024,
            'hidden_size': 128,
            'lr': 0.001,
            'patience': 3
        }
    ]
    
    tuning_results = []
    for config in tuning_configs:
        result = run_experiment(config)
        tuning_results.append({
            'config': config,
            'f1': result['test_metrics']['f1'],
            'auc': result['test_metrics']['auc']
        })
    
    # Find best config
    best_tuning = max(tuning_results, key=lambda x: x['f1'])
    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS:")
    print("="*60)
    print(f"Time Steps: {best_tuning['config']['time_steps']}")
    print(f"Hidden Size: {best_tuning['config']['hidden_size']}")
    print(f"F1 Score: {best_tuning['f1']:.4f}")
    print(f"AUC-ROC: {best_tuning['auc']:.4f}")
    
    # ========================================
    # STAGE 3: FINAL TRAINING
    # ========================================
    print("\n\n")
    print("█"*60)
    print("STAGE 3: FINAL TRAINING (Full Dataset)")
    print("█"*60)
    
    final_config = {
        'name': 'final',
        'sample_size': None,  # Use full dataset
        'epochs': 20,
        'time_steps': best_tuning['config']['time_steps'],
        'batch_size': 1024,
        'hidden_size': best_tuning['config']['hidden_size'],
        'lr': 0.001,
        'patience': 3
    }
    
    final_results = run_experiment(final_config)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n\n")
    print("█"*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("█"*60)
    
    print(f"\nFinal Model Performance (with Optimized Threshold):")
    print(f"  Accuracy:  {final_results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {final_results['test_metrics']['precision']:.4f}")
    print(f"  Recall:    {final_results['test_metrics']['recall']:.4f}")
    print(f"  F1 Score:  {final_results['test_metrics']['f1']:.4f}")
    print(f"  AUC-ROC:   {final_results['test_metrics']['auc']:.4f}")
    
    print(f"\nFinal model saved to: {final_results['exp_dir']}")
    print(f"\nNOTE: Model uses optimized classification threshold for better precision/recall balance.")
    
    # Save summary
    summary = {
        'debug': {
            'config': debug_config,
            'test_f1': debug_results['test_metrics']['f1']
        },
        'tuning': tuning_results,
        'best_hyperparams': best_tuning['config'],
        'final': {
            'config': final_config,
            'metrics': {k: v for k, v in final_results['test_metrics'].items() 
                       if k not in ['predictions', 'labels']}
        }
    }
    
    summary_path = 'snn_models/training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Training summary saved to: {summary_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
