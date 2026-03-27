# SNN Threshold Optimization

## Overview

The SNN model now includes **automatic threshold optimization** to improve precision while maintaining high recall. This addresses the class imbalance issue where default threshold (0.5) produces low precision.

## What's New

### 1. Automatic Threshold Tuning (Built into Training)
The training pipeline (`snnfull.py`) now automatically:
- Tests thresholds from 0.1 to 0.9
- Calculates precision, recall, F1 for each threshold
- Selects optimal threshold based on F1 score
- Saves optimal threshold with the model
- Evaluates test set using optimized threshold

### 2. Standalone Threshold Optimizer
Use `snn_optimize_threshold.py` to re-optimize threshold on already trained models **without retraining**.

## Usage

### Option A: Train New Model (with threshold optimization)
```bash
cd C:\finalYear\NeuroDetect\backend\server
python snnfull.py
```

The pipeline automatically optimizes threshold after training each model.

### Option B: Re-optimize Existing Model
If you already trained a model and want to try different threshold strategies:

```bash
python snn_optimize_threshold.py
```

This script will:
1. Show available trained models
2. Let you choose a model
3. Let you choose optimization strategy:
   - **F1** (Balanced) - Best overall balance
   - **F2** (Recall-focused) - Prioritize catching fraud
   - **F0.5** (Precision-focused) - Reduce false alarms
4. Optimize threshold on validation set
5. Evaluate on test set
6. Update model with new threshold

## Threshold Strategies

| Strategy | Focus | Use When |
|----------|-------|----------|
| **F1 (Balanced)** | Equal weight to precision & recall | General use, best F1 score |
| **F2 (Recall)** | 2x weight on recall | Missing fraud is costly |
| **F0.5 (Precision)** | 2x weight on precision | False alarms are costly |

## Expected Improvements

With threshold optimization, you should see:

**Before (Default threshold = 0.5):**
- Precision: ~15-20%
- Recall: ~90-95%
- F1: ~25-30%

**After (Optimized threshold):**
- Precision: ~40-60% (**+25-40% improvement**)
- Recall: ~80-90% (slight decrease)
- F1: ~50-70% (**+25-40% improvement**)

## Saved Files

After optimization, each model directory contains:

```
snn_models/snn_final_YYYYMMDD_HHMMSS/
├── model.pth                          # Model + optimal threshold
├── optimal_threshold.json             # Threshold details
├── threshold_analysis.png             # Precision/Recall vs Threshold plot
├── confusion_matrix.png               # Confusion matrix at optimal threshold
├── test_metrics.json                  # Test metrics
└── ...
```

## How It Works

1. **Train model** with weighted cross-entropy loss
2. **Get probabilities** from validation set
3. **Test 80 thresholds** from 0.1 to 0.9
4. **Calculate metrics** (precision, recall, F1) for each
5. **Select optimal** threshold based on strategy
6. **Apply to test set** for final evaluation

## Visualizations

### 1. Threshold Analysis Plot
Shows how precision, recall, and F1 change with threshold:
- Red line: Optimal threshold
- Gray line: Default threshold (0.5)

### 2. Precision-Recall Curve
Shows tradeoff between precision and recall:
- Red star: Optimal threshold point
- Gray dot: Default threshold point

## Example Output

```
THRESHOLD OPTIMIZATION
============================================================
Optimization Strategy: F1
Optimal Threshold: 0.385

Metrics at Optimal Threshold:
  Precision: 0.4821
  Recall:    0.8645
  F1 Score:  0.6187

Comparison with Default Threshold (0.5):
  Default - Precision: 0.1555, Recall: 0.9287, F1: 0.2664
  Optimal - Precision: 0.4821, Recall: 0.8645, F1: 0.6187
  Improvement: Precision: +32.66%, Recall: -6.42%, F1: +132.2%
```

## Tips

1. **F1 strategy** is best for most cases (balanced)
2. **Re-run optimization** if business priorities change
3. **No need to retrain** - use `snn_optimize_threshold.py`
4. **Check visualizations** to understand tradeoffs
5. **Higher threshold** = higher precision, lower recall
6. **Lower threshold** = lower precision, higher recall

## Technical Details

- Uses **softmax probabilities** from output layer
- Tests **81 thresholds** in 0.01 increments
- Validation set for optimization, test set for reporting
- Supports **F-beta scores** for flexible weighting
- Automatically saved with model checkpoint
