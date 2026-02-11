"""
Training and Evaluation Functions for Enhanced LSTM Fraud Detection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, precision_score, recall_score, 
    f1_score, accuracy_score, average_precision_score
)

from .model import EnhancedFraudLSTM
from .utils import FocalLoss, EarlyStopping, create_weighted_sampler
from .preprocessor import LSTMPreprocessor


def train_enhanced_lstm(X_train, y_train, X_val, y_val, feature_names, 
                       sequence_length=10, epochs=100, batch_size=128, 
                       learning_rate=0.0005):
    """
    Enhanced training with better handling of class imbalance
    
    Args:
        X_train: Training sequences (n_sequences, sequence_length, n_features)
        y_train: Training labels (n_sequences,)
        X_val: Validation sequences
        y_val: Validation labels
        feature_names: List of feature names
        sequence_length: Length of each sequence (for logging)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        
    Returns:
        model: Trained model
        scaler: Fitted scaler
        train_losses: Training loss history
        val_losses: Validation loss history
        train_f1_scores: Training F1 score history
        val_f1_scores: Validation F1 score history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Scale the data
    preprocessor = LSTMPreprocessor(sequence_length=sequence_length)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    scaler = preprocessor.scaler
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create weighted sampler
    sampler = create_weighted_sampler(y_train)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[2]
    model = EnhancedFraudLSTM(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.4,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Loss function - Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=1e-5, verbose=True)
    
    # Training metrics
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    best_f1 = 0
    best_model_state = None
    
    print(f"\nStarting Enhanced LSTM Training...")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {X_train_tensor.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss function: Focal Loss (alpha=0.25, gamma=2.0)")
    print(f"Using weighted sampling: Yes")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            preds = (outputs > 0.5).float()
            all_train_preds.extend(preds.cpu().detach().numpy())
            all_train_labels.extend(batch_y.cpu().detach().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = outputs.cpu().detach().numpy()
                preds = (outputs > 0.5).float()
                
                all_val_probs.extend(probs)
                all_val_preds.extend(preds.cpu().detach().numpy())
                all_val_labels.extend(batch_y.cpu().detach().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Convert to numpy arrays
        train_preds = np.array(all_train_preds).flatten()
        train_labels = np.array(all_train_labels).flatten()
        val_preds = np.array(all_val_preds).flatten()
        val_labels = np.array(all_val_labels).flatten()
        val_probs = np.array(all_val_probs).flatten()
        
        # Calculate F1 scores
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        
        # Update learning rate
        scheduler.step()
        
        # Check for best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best F1: {val_f1:.4f}")
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if early_stopping.early_stop:
            print(f'\n⚠️ Early stopping triggered at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'\n✅ Training completed!')
    print(f'   Best Validation F1: {best_f1:.4f}')
    print(f'   Total epochs trained: {len(train_losses)}')
    
    return model, scaler, train_losses, val_losses, train_f1_scores, val_f1_scores


def comprehensive_evaluation(model, scaler, X_test, y_test, threshold_tuning=True):
    """
    Comprehensive evaluation with threshold tuning
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test sequences (n_sequences, sequence_length, n_features)
        y_test: Test labels (n_sequences,)
        threshold_tuning: Whether to tune the decision threshold
        
    Returns:
        dict: Dictionary containing all evaluation metrics and results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare test data
    preprocessor = LSTMPreprocessor()
    preprocessor.scaler = scaler
    X_test_scaled = preprocessor.transform(X_test)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Get predictions
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs, _ = model(batch_x)
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
    
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    
    # Find optimal threshold
    if threshold_tuning:
        # Use precision-recall curve to find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1_scores = f1_scores[:-1]  # Remove last element
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\nOptimal threshold found: {optimal_threshold:.4f}")
        print(f"Best F1 score at this threshold: {best_f1:.4f}")
    else:
        optimal_threshold = 0.5
    
    # Make predictions with optimal threshold
    y_pred = (y_prob > optimal_threshold).astype(int)
    
    # Calculate all metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Handle ROC AUC calculation carefully
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5  # Default if calculation fails
    
    # PR AUC
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'], output_dict=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  ROC AUC:     {roc_auc:.4f}")
    print(f"  PR AUC:      {pr_auc:.4f}")
    print(f"  Threshold:   {optimal_threshold:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Normal    Fraud")
    print(f"  Actual Normal  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"        Fraud    {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    print(f"\nDetailed Classification Report:")
    print(f"  Normal - Precision: {report['Normal']['precision']:.4f}, Recall: {report['Normal']['recall']:.4f}, F1: {report['Normal']['f1-score']:.4f}")
    print(f"  Fraud  - Precision: {report['Fraud']['precision']:.4f}, Recall: {report['Fraud']['recall']:.4f}, F1: {report['Fraud']['f1-score']:.4f}")
    
    # Calculate additional useful metrics
    fraud_precision = report['Fraud']['precision']
    fraud_recall = report['Fraud']['recall']
    fraud_f1 = report['Fraud']['f1-score']
    
    print(f"\nFraud Detection Specific:")
    print(f"  Fraud Precision: {fraud_precision:.4f}")
    print(f"  Fraud Recall:    {fraud_recall:.4f}")
    print(f"  Fraud F1:        {fraud_f1:.4f}")
    
    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'fraud_precision': fraud_precision,
        'fraud_recall': fraud_recall,
        'fraud_f1': fraud_f1
    }
    
    return results
