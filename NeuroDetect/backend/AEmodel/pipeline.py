import pandas as pd
import numpy as np
import torch
import joblib
from typing import List, Dict, Any, Optional
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BatchProcessor:
    """Batch processing pipeline for fraud detection"""
    
    def __init__(self, model_path, scaler_path, threshold=None):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            threshold: Detection threshold
        """
        from .preprocessor import DataPreprocessor
        from .model import AutoencoderTrainer
        from .detector import FraudDetector
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(scaler_path)
        self.preprocessor.load_scaler(scaler_path)
        
        # Load model
        self.trainer = AutoencoderTrainer()
        self.model = self.trainer.load_model(model_path)
        
        # Initialize detector
        self.detector = FraudDetector(self.model, threshold)
        
    def process_batch(self, df, output_format='dataframe'):
        """
        Process a batch of transactions
        
        Args:
            df: Input DataFrame with transactions
            output_format: 'dataframe' or 'json'
            
        Returns:
            Processed results
        """
        # Preprocess
        X_processed, _, _ = self.preprocessor.preprocess(df, is_training=False)
        
        # Predict
        predictions, errors = self.detector.predict(X_processed)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['reconstruction_error'] = errors
        df_result['is_fraud_predicted'] = predictions
        df_result['fraud_probability'] = errors / (errors.max() + 1e-10)  # Normalized score
        
        # Add risk categories
        df_result['risk_category'] = pd.cut(
            df_result['fraud_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        if output_format == 'json':
            # Convert to JSON format
            results = []
            for idx, row in df_result.iterrows():
                result = {
                    'transaction_id': idx,
                    'amount': float(row.get('amt', 0)),
                    'reconstruction_error': float(row['reconstruction_error']),
                    'is_fraud_predicted': int(row['is_fraud_predicted']),
                    'fraud_probability': float(row['fraud_probability']),
                    'risk_category': str(row['risk_category']),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            return json.dumps(results, indent=2)
        
        return df_result
    
    def process_file(self, input_path, output_path=None, chunksize=10000):
        """
        Process large file in chunks
        
        Args:
            input_path: Input file path
            output_path: Output file path
            chunksize: Chunk size for processing
        """
        all_results = []
        
        # Process in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
            print(f"Processing chunk {chunk_num + 1}...")
            result_chunk = self.process_batch(chunk)
            all_results.append(result_chunk)
            
            # Save intermediate results if output path provided
            if output_path:
                # Create output directory if it doesn't exist
                import os
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save chunk
                chunk_output = f"{output_path}_chunk_{chunk_num}.csv"
                result_chunk.to_csv(chunk_output, index=False)
                print(f"Saved chunk to {chunk_output}")
        
        # Combine all results
        final_result = pd.concat(all_results, ignore_index=True)
        
        if output_path:
            final_result.to_csv(output_path, index=False)
            print(f"\nProcessing complete. Results saved to {output_path}")
        
        return final_result

class RealTimeProcessor:
    """Real-time fraud detection processor"""
    
    def __init__(self, model_path, scaler_path, threshold=None):
        """
        Initialize real-time processor
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            threshold: Detection threshold
        """
        from .preprocessor import DataPreprocessor
        from .model import AutoencoderTrainer
        from .detector import FraudDetector
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(scaler_path)
        self.preprocessor.load_scaler(scaler_path)
        
        # Load model
        self.trainer = AutoencoderTrainer()
        self.model = self.trainer.load_model(model_path)
        
        # Initialize detector
        self.detector = FraudDetector(self.model, threshold)
        
        # Statistics
        self.processed_count = 0
        self.fraud_count = 0
        self.latencies = []
        
    def process_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single transaction in real-time
        
        Args:
            transaction_data: Dictionary with transaction data
            
        Returns:
            Dictionary with fraud detection results
        """
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Preprocess
        try:
            X_processed, _, _ = self.preprocessor.preprocess(df, is_training=False)
        except Exception as e:
            return {
                'error': str(e),
                'is_fraud_predicted': -1,
                'timestamp': datetime.now().isoformat()
            }
        
        # Predict
        predictions, errors = self.detector.predict(X_processed)
        
        # Calculate latency
        latency = time.time() - start_time
        self.latencies.append(latency)
        
        # Update statistics
        self.processed_count += 1
        if predictions[0] == 1:
            self.fraud_count += 1
        
        # Prepare response
        result = {
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'reconstruction_error': float(errors[0]),
            'is_fraud_predicted': int(predictions[0]),
            'fraud_probability': float(errors[0] / (self.detector.threshold + 1e-10)),
            'processing_latency_ms': round(latency * 1000, 2),
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0'
        }
        
        # Add risk assessment
        if result['fraud_probability'] < 0.3:
            result['risk_level'] = 'Low'
        elif result['fraud_probability'] < 0.7:
            result['risk_level'] = 'Medium'
        else:
            result['risk_level'] = 'High'
        
        return result
    
    def process_stream(self, transaction_stream, output_callback=None):
        """
        Process a stream of transactions
        
        Args:
            transaction_stream: Iterable of transaction data
            output_callback: Function to call with results
        """
        results = []
        
        for transaction in transaction_stream:
            result = self.process_transaction(transaction)
            results.append(result)
            
            if output_callback:
                output_callback(result)
        
        return results
    
    def get_statistics(self):
        """Get processing statistics"""
        if not self.latencies:
            avg_latency = 0
        else:
            avg_latency = np.mean(self.latencies) * 1000  # Convert to ms
        
        stats = {
            'total_processed': self.processed_count,
            'fraud_detected': self.fraud_count,
            'fraud_rate_percent': (self.fraud_count / self.processed_count * 100) if self.processed_count > 0 else 0,
            'avg_processing_time_ms': round(avg_latency, 2),
            'current_threshold': float(self.detector.threshold) if self.detector.threshold else None
        }
        
        return stats