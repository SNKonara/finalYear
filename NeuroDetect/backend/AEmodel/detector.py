import torch
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetector:
    """Fraud detection system using autoencoder"""
    
    def __init__(self, model, threshold=None, device=None):
        """
        Initialize fraud detector
        
        Args:
            model: Trained autoencoder model
            threshold: Detection threshold (if None, will be determined)
            device: Device to run inference on
        """
        self.model = model
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def calculate_reconstruction_errors(self, X):
        """
        Calculate reconstruction errors for input data
        
        Args:
            X: Input data (numpy array)
            
        Returns:
            Reconstruction errors
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return reconstruction_errors.cpu().numpy()
    
    def determine_threshold(self, X_normal, percentile=95):
        """
        Determine threshold based on normal transactions
        
        Args:
            X_normal: Normal transaction data
            percentile: Percentile to use for threshold
            
        Returns:
            Optimal threshold
        """
        errors = self.calculate_reconstruction_errors(X_normal)
        self.threshold = np.percentile(errors, percentile)
        return self.threshold
    
    def predict(self, X, threshold=None):
        """
        Predict fraud labels
        
        Args:
            X: Input data
            threshold: Custom threshold (uses self.threshold if None)
            
        Returns:
            Predictions (1 for fraud, 0 for normal)
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call determine_threshold() first.")
            threshold = self.threshold
        
        errors = self.calculate_reconstruction_errors(X)
        predictions = (errors > threshold).astype(int)
        
        return predictions, errors
    
    def evaluate(self, X_test, y_test, threshold_percentiles=[90, 95, 98, 99]):
        """
        Evaluate model with multiple thresholds
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold_percentiles: Percentiles to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        errors = self.calculate_reconstruction_errors(X_test)
        normal_errors = errors[y_test == 0]
        
        results = {}
        
        print("="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for percentile in threshold_percentiles:
            threshold = np.percentile(normal_errors, percentile)
            y_pred = (errors > threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            results[percentile] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm
            }
            
            print(f"\nPercentile {percentile}% (Threshold: {threshold:.6f}):")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        # Find best threshold based on F1-score
        best_percentile = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_result = results[best_percentile]
        
        print("\n" + "="*70)
        print(f"BEST THRESHOLD: {best_percentile}%")
        print(f"Threshold Value: {best_result['threshold']:.6f}")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print("="*70)
        
        # Set optimal threshold
        self.threshold = best_result['threshold']
        
        return results, best_result
    
    def plot_errors_distribution(self, X_normal, X_fraud=None):
        """
        Plot distribution of reconstruction errors
        
        Args:
            X_normal: Normal transactions
            X_fraud: Fraud transactions (optional)
        """
        normal_errors = self.calculate_reconstruction_errors(X_normal)
        
        plt.figure(figsize=(12, 6))
        
        # Plot normal errors
        plt.subplot(1, 2, 1)
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue')
        if self.threshold:
            plt.axvline(x=self.threshold, color='red', linestyle='--', 
                       label=f'Threshold: {self.threshold:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Normal Transactions Error Distribution')
        plt.legend()
        
        if X_fraud is not None:
            fraud_errors = self.calculate_reconstruction_errors(X_fraud)
            
            plt.subplot(1, 2, 2)
            plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue')
            plt.hist(fraud_errors, bins=50, alpha=0.7, label='Fraud', color='red')
            if self.threshold:
                plt.axvline(x=self.threshold, color='black', linestyle='--', 
                           label=f'Threshold: {self.threshold:.4f}')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Normal vs Fraud Error Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.show()