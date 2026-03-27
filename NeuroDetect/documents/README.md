# Enhanced LSTM Fraud Detection - Organized Structure

## Overview
This folder contains a well-organized, production-ready LSTM fraud detection system with class imbalance handling, attention mechanism, and comprehensive evaluation metrics.

## File Structure

```
LSTMmodel/
├── __init__.py           # Module exports
├── model.py              # Enhanced LSTM model architecture
├── preprocessor.py       # Data preprocessing and sequence creation
├── train.py              # Training and evaluation functions
├── utils.py              # Helper classes (FocalLoss, EarlyStopping, etc.)
├── save_load.py          # Model saving and loading utilities
└── lstmfull.py           # Main training script
```

## Components

### 1. **model.py** - Model Architecture
Contains two model classes:

- **`EnhancedFraudLSTM`** (Recommended): Advanced LSTM with:
  - Bidirectional LSTM layers
  - Attention mechanism
  - Batch normalization
  - Deep fully connected layers
  - Dropout regularization

- **`LSTMFraudDetector`** (Legacy): Simpler LSTM model for backward compatibility

### 2. **preprocessor.py** - Data Preprocessing
Contains functions for:

- **`create_balanced_sequences()`**: Creates balanced sequences by oversampling fraud and undersampling normal transactions
- **`prepare_improved_lstm_data()`**: Complete data preparation pipeline with train/val/test split
- **`LSTMPreprocessor`**: Class for scaling and sequence creation

### 3. **train.py** - Training & Evaluation
Contains:

- **`train_enhanced_lstm()`**: Training function with:
  - Focal Loss for class imbalance
  - Weighted random sampling
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate scheduler
  - Early stopping
  - Gradient clipping

- **`comprehensive_evaluation()`**: Evaluation function with:
  - Optimal threshold tuning using PR curve
  - Complete metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
  - Confusion matrix
  - Detailed classification report

### 4. **utils.py** - Utility Classes
Contains:

- **`FocalLoss`**: Custom loss function for class imbalance
- **`EarlyStopping`**: Early stopping callback
- **`create_weighted_sampler()`**: Creates weighted sampler for balanced batches
- **`calculate_class_weights()`**: Calculates class weights for loss functions

### 5. **save_load.py** - Model Persistence
Contains:

- **`save_model()`**: Save model with metadata
- **`load_model()`**: Load saved model
- **`save_training_history()`**: Save training metrics
- **`load_training_history()`**: Load training metrics
- **`export_model_for_inference()`**: Export lightweight inference model
- **`load_inference_model()`**: Load inference model

### 6. **lstmfull.py** - Main Training Script
Simplified training script that orchestrates all components.

## Usage

### Basic Training

```python
from LSTMmodel import run_enhanced_lstm_training

# Assuming you have preprocessed data with 'is_fraud' column
model, scaler, results = run_enhanced_lstm_training(
    df_preprocessed=df_preprocessed,
    sequence_length=10,
    epochs=100,
    batch_size=128,
    learning_rate=0.0005,
    output_dir='saved_models'
)
```

### Step-by-Step Training

```python
from LSTMmodel import (
    prepare_improved_lstm_data,
    train_enhanced_lstm,
    comprehensive_evaluation,
    save_model
)

# Step 1: Prepare data
X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_improved_lstm_data(
    df_preprocessed,
    sequence_length=10
)

# Step 2: Train model
model, scaler, train_losses, val_losses, train_f1, val_f1 = train_enhanced_lstm(
    X_train, y_train, X_val, y_val, feature_names,
    sequence_length=10,
    epochs=100,
    batch_size=128,
    learning_rate=0.0005
)

# Step 3: Evaluate
results = comprehensive_evaluation(
    model, scaler, X_test, y_test, 
    threshold_tuning=True
)

# Step 4: Save
save_model(model, scaler, feature_names, results, output_dir='saved_models')
```

### Loading a Trained Model

```python
from LSTMmodel import load_model

model, scaler, feature_names, model_config, results = load_model(
    'saved_models/enhanced_lstm_fraud_model.pth'
)

# Make predictions
predictions = model.predict(X_test_tensor)
```

## Key Features

### 1. **Class Imbalance Handling**
- Oversampling fraud sequences (10x boost factor)
- Undersampling normal sequences
- Weighted random sampling during training
- Focal Loss (alpha=0.25, gamma=2.0)

### 2. **Model Architecture**
- Bidirectional LSTM (2 layers, 256 hidden units)
- Attention mechanism for focusing on important time steps
- Batch normalization for stable training
- Deep classification head (5 layers)
- Dropout (0.4) for regularization

### 3. **Training Optimizations**
- AdamW optimizer with weight decay (1e-4)
- Cosine annealing with warm restarts
- Gradient clipping (max norm 1.0)
- Early stopping (patience=15)
- Best model checkpoint saving

### 4. **Evaluation**
- Optimal threshold tuning using precision-recall curve
- Comprehensive metrics
- Confusion matrix analysis
- Per-class performance metrics

## Model Performance

The enhanced LSTM model addresses the common issue of predicting all zeros by:

1. **Balanced Sequence Generation**: Creates roughly equal fraud/normal sequences
2. **Focal Loss**: Focuses learning on hard examples
3. **Weighted Sampling**: Ensures balanced batches during training
4. **Threshold Tuning**: Finds optimal decision threshold instead of using 0.5

Expected improvements:
- Better fraud detection recall
- Balanced precision/recall trade-off
- Higher F1 score on fraud class
- No "all zeros" predictions

## Hyperparameters

Default hyperparameters (tuned for fraud detection):

```python
{
    'sequence_length': 10,           # Transactions per sequence
    'hidden_dim': 256,               # LSTM hidden size
    'num_layers': 2,                 # LSTM layers
    'dropout': 0.4,                  # Dropout rate
    'bidirectional': True,           # Bidirectional LSTM
    'use_attention': True,           # Attention mechanism
    'learning_rate': 0.0005,         # Initial learning rate
    'batch_size': 128,               # Training batch size
    'epochs': 100,                   # Maximum epochs
    'focal_alpha': 0.25,             # Focal loss alpha
    'focal_gamma': 2.0,              # Focal loss gamma
    'fraud_boost_factor': 10         # Fraud oversampling factor
}
```

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
joblib>=1.0.0
```

## Notes

1. **Preprocessing**: This module expects preprocessed data with an 'is_fraud' column. Use your existing preprocessing function before calling the LSTM training.

2. **GPU Support**: Automatically uses CUDA if available for faster training.

3. **Memory**: Sequence creation can be memory-intensive for large datasets. Adjust `fraud_boost_factor` if needed.

4. **Threshold**: The model finds the optimal decision threshold automatically. Don't use the default 0.5.

## Troubleshooting

### Issue: Model predicts all zeros
**Solution**: This is fixed! The enhancements ensure balanced training.

### Issue: Out of memory
**Solution**: Reduce `fraud_boost_factor` in `prepare_improved_lstm_data()` or decrease `batch_size`.

### Issue: Training too slow
**Solution**: Reduce `epochs`, `hidden_dim`, or `sequence_length`.

### Issue: Overfitting
**Solution**: Increase `dropout` rate or add more data augmentation.

## Citation

This enhanced LSTM implementation uses best practices for fraud detection with imbalanced data, including focal loss, attention mechanisms, and careful threshold tuning.
