# websocket_unified_server.py
"""
Unified WebSocket Server for Autoencoder, LSTM, and SNN Fraud Detection
Supports model selection capability on a single port
"""
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import re
import os
import sys
import io
import joblib
from datetime import datetime, timedelta
import time
import torch
from collections import deque

# Force UTF-8 output on Windows (cp1252 terminal cannot encode emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Autoencoder imports
from AEmodel.preprocessor import DataPreprocessor
from AEmodel.model import FraudAutoencoder

# LSTM imports
from LSTMmodel.save_load import load_model as load_lstm_model

# SNN imports
from SNNmodel.customer_behavior_snn import SpikingFraudDetector

# Database import
from database.mongodb import get_mongodb_instance


class UnifiedFraudDetectionServer:
    def __init__(self, dataset_path, stream_speed=1.0, sequence_length=10):
        """
        Unified server supporting both Autoencoder and LSTM detection
        
        Args:
            dataset_path: Path to CSV dataset
            stream_speed: Records per second (default: 1 record/sec)
            sequence_length: Length of transaction sequences for LSTM (default: 10)
        """
        self.dataset = self.load_dataset(dataset_path)
        self.stream_speed = stream_speed
        self.sequence_length = sequence_length
        
        # Client management
        self.streaming_clients = {}
        self.stream_tasks = {}
        self.client_totals = {}
        self.client_sequences = {}
        self.client_models = {}  # Track which model each client is using
        self.clients = set()
        self.always_streaming = True  # Stream 24/7 without stopping
        self.hourly_cleanup_task = None
        self.cleanup_interval_seconds = 3600  # Check for cleanup once per hour
        
        # Autoencoder components
        self.ae_model = None
        self.ae_preprocessor = None
        self.ae_threshold = None
        self.ae_device = None
        self.ae_model_info = {}
        
        # LSTM components
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_feature_names = []
        self.lstm_model_config = {}
        self.lstm_results = {}
        self.lstm_threshold = 0.5
        self.lstm_device = None
        self.lstm_model_info = {}

        # SNN components
        self.snn_model = None
        self.snn_scaler = None
        self.snn_feature_names = []
        self.snn_profiles = {}
        self.snn_metadata = {}
        self.snn_threshold = 0.5
        self.snn_threshold_scale = 1.0
        self.snn_global_threshold = 0.5
        self.snn_unknown_customer_policy = 'global'
        self.snn_time_steps = 20
        self.snn_device = None
        self.snn_model_info = {}
        
        # Statistics (separate for each model)
        self.stats = {
            'autoencoder': {
                'total_processed': 0,
                'fraud_detected': 0,
                'preprocessing_errors': 0,
                'dataset_loops': 0,
                'avg_processing_time': 0,
                'start_time': time.time(),
                'risk_distribution': {'Low': 0, 'Medium': 0, 'High': 0},
                'prediction_history': []
            },
            'lstm': {
                'total_processed': 0,
                'fraud_detected': 0,
                'preprocessing_errors': 0,
                'dataset_loops': 0,
                'avg_processing_time': 0,
                'start_time': time.time(),
                'risk_distribution': {'Low': 0, 'Medium-Low': 0, 'Medium-High': 0, 'High': 0},
                'prediction_history': []
            },
            'snn': {
                'total_processed': 0,
                'fraud_detected': 0,
                'preprocessing_errors': 0,
                'dataset_loops': 0,
                'avg_processing_time': 0,
                'start_time': time.time(),
                'risk_distribution': {'Low': 0, 'Medium-Low': 0, 'Medium-High': 0, 'High': 0},
                'prediction_history': []
            }
        }
        
        # MongoDB connection
        self.db = None
        
        print(f" Loaded dataset with {len(self.dataset)} records")
        
        # Connect to database
        self.connect_database()
        
        # Load both models
        self.load_autoencoder_model()
        self.load_lstm_model()
        self.load_snn_model()

    def _build_stats_payload(self, model_type):
        stats = self.stats[model_type]
        elapsed = time.time() - stats['start_time']
        throughput = stats['total_processed'] / elapsed if elapsed > 0 else 0
        fraud_rate = (stats['fraud_detected'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
        success_rate = ((stats['total_processed'] - stats['preprocessing_errors']) / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 100
        return {
            'total_processed': stats['total_processed'],
            'fraud_detected': stats['fraud_detected'],
            'preprocessing_errors': stats['preprocessing_errors'],
            'dataset_loops': stats['dataset_loops'],
            'avg_processing_time': round(stats['avg_processing_time'], 2),
            'throughput_tps': round(throughput, 2),
            'fraud_rate': round(fraud_rate, 2),
            'success_rate': round(success_rate, 2),
            'risk_distribution': stats['risk_distribution'],
            'prediction_history': stats['prediction_history'][-20:]
        }

    def _reset_stats_for_model(self, model_type):
        if model_type == 'autoencoder':
            risk_distribution = {'Low': 0, 'Medium': 0, 'High': 0}
        else:
            risk_distribution = {'Low': 0, 'Medium-Low': 0, 'Medium-High': 0, 'High': 0}

        self.stats[model_type] = {
            'total_processed': 0,
            'fraud_detected': 0,
            'preprocessing_errors': 0,
            'dataset_loops': 0,
            'avg_processing_time': 0,
            'start_time': time.time(),
            'risk_distribution': risk_distribution,
            'prediction_history': []
        }

    def _extract_autoencoder_architecture(self, state_dict, input_dim):
        layer_keys = ['encoder.0.weight', 'encoder.3.weight', 'encoder.6.weight']
        hidden_layers = []
        for key in layer_keys:
            tensor = state_dict.get(key)
            if tensor is not None and len(tensor.shape) == 2:
                hidden_layers.append(int(tensor.shape[0]))

        if not hidden_layers:
            hidden_layers = [128, 64, 16]

        architecture = '-'.join(str(layer) for layer in hidden_layers)
        readable = f"Input ({input_dim}) -> {' -> '.join(str(layer) for layer in hidden_layers)} -> Output ({input_dim})"
        return architecture, readable
    
    def connect_database(self):
        """Initialize MongoDB connection"""
        try:
            print("\n️  Checking MongoDB configuration...")
            self.db = get_mongodb_instance()
            if self.db.connected:
                print(" MongoDB connected - results will be saved to database")
            else:
                print("ℹ️  MongoDB disabled - results will be saved to local files only")
        except Exception as e:
            print(f"️  MongoDB setup error: {e}")
            self.db = None
    
    def load_autoencoder_model(self):
        """Load the Autoencoder fraud detection model"""
        print("\n Loading Autoencoder Model...")
        
        try:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'saved_models')
            
            # Load model checkpoint
            checkpoint_path = os.path.join(model_dir, 'autoencoder.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            input_dim = checkpoint['input_dim']
            state_dict = checkpoint['model_state_dict']
            architecture, readable_architecture = self._extract_autoencoder_architecture(state_dict, input_dim)
            
            # Initialize model
            self.ae_model = FraudAutoencoder(
                input_dim=input_dim,
                hidden_dim1=128,
                hidden_dim2=64,
                latent_dim=16,
                dropout_rate=0.000287
            )
            self.ae_model.load_state_dict(state_dict)
            self.ae_model.eval()
            
            # Load preprocessor
            self.ae_preprocessor = DataPreprocessor()
            preprocessor_path = os.path.join(model_dir, 'scaler.pkl')
            self.ae_preprocessor.load_preprocessor(preprocessor_path)
            
            # Load threshold
            threshold_path = os.path.join(model_dir, 'threshold.json')
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.ae_threshold = threshold_data['threshold']
            
            # Set device
            self.ae_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ae_model = self.ae_model.to(self.ae_device)
            
            # Load features.json
            features_path = os.path.join(model_dir, 'features.json')
            feature_info = {}
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    feature_info = json.load(f)
            
            # Store model info
            self.ae_model_info = {
                'model_type': 'Autoencoder',
                'input_dim': input_dim,
                'architecture': architecture,
                'architecture_readable': readable_architecture,
                'threshold': float(self.ae_threshold),
                'device': str(self.ae_device),
                'expected_features': input_dim,
                'feature_names': checkpoint.get('feature_names', feature_info.get('feature_names', [])),
                'num_features': feature_info.get('num_features', input_dim),
                'performance': {},
            }
            
            print(f" Autoencoder Model Loaded Successfully")
            print(f"   Input dimensions: {input_dim}")
            print(f"   Threshold: {self.ae_threshold:.6f}")
            print(f"   Device: {self.ae_device}")
            
        except Exception as e:
            print(f" Failed to load Autoencoder model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_lstm_model(self):
        """Load the LSTM fraud detection model"""
        print("\n Loading LSTM Model...")
        
        try:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'saved_models')
            model_path = os.path.join(model_dir, 'enhanced_lstm_fraud_model.pth')
            
            # Load model
            self.lstm_model, self.lstm_scaler, self.lstm_feature_names, self.lstm_model_config, self.lstm_results = load_lstm_model(model_path)
            
            # Set device
            self.lstm_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lstm_model = self.lstm_model.to(self.lstm_device)
            self.lstm_model.eval()
            
            # Get optimal threshold
            self.lstm_threshold = self.lstm_results.get('optimal_threshold', 0.5)
            
            # Store model info
            self.lstm_model_info = {
                'model_type': 'LSTM',
                'input_dim': self.lstm_model_config['input_dim'],
                'sequence_length': self.sequence_length,
                'architecture': f"BiLSTM-{self.lstm_model_config['hidden_dim']}-{self.lstm_model_config['num_layers']}L + Attention",
                'threshold': float(self.lstm_threshold),
                'device': str(self.lstm_device),
                'expected_features': len(self.lstm_feature_names),
                'feature_names': self.lstm_feature_names,
                'hidden_size': int(self.lstm_model_config.get('hidden_dim', 256)),
                'num_layers': int(self.lstm_model_config.get('num_layers', 2)),
                'performance': {
                    'accuracy': self.lstm_results.get('accuracy', 0),
                    'precision': self.lstm_results.get('fraud_precision', 0),
                    'recall': self.lstm_results.get('fraud_recall', 0),
                    'f1': self.lstm_results.get('fraud_f1', self.lstm_results.get('f1', 0)),
                    'f1_score': self.lstm_results.get('f1', 0),
                    'fraud_f1': self.lstm_results.get('fraud_f1', 0),
                    'roc_auc': self.lstm_results.get('roc_auc', 0),
                    'auc': self.lstm_results.get('roc_auc', 0),
                }
            }
            
            print(f" LSTM Model Loaded Successfully")
            print(f"   Input dimensions: {self.lstm_model_config['input_dim']}")
            print(f"   Sequence length: {self.sequence_length}")
            print(f"   Threshold: {self.lstm_threshold:.6f}")
            print(f"   F1 Score: {self.lstm_results.get('f1', 0):.4f}")
            
        except Exception as e:
            print(f" Failed to load LSTM model: {e}")
            import traceback
            traceback.print_exc()

    def load_snn_model(self):
        """Load the packaged customer-centric SNN fraud detection model"""
        print("\n Loading SNN Model...")

        try:
            snn_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'snn_models',
                'final_customer_snn_package'
            )

            model_path = os.path.join(snn_dir, 'snn_customer_classifier.pth')
            scaler_path = os.path.join(snn_dir, 'scaler.pkl')
            features_path = os.path.join(snn_dir, 'features.json')
            profiles_path = os.path.join(snn_dir, 'customer_profiles.json')
            metadata_path = os.path.join(snn_dir, 'training_metadata.json')

            checkpoint = torch.load(model_path, map_location='cpu')

            self.snn_model = SpikingFraudDetector(
                input_size=int(checkpoint.get('input_size', 24)),
                hidden_size=int(checkpoint.get('hidden_size', 64)),
                output_size=int(checkpoint.get('output_size', 2)),
                beta=float(checkpoint.get('beta', 0.95)),
            )
            self.snn_model.load_state_dict(checkpoint['model_state_dict'])

            self.snn_scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                with open(features_path, 'r', encoding='utf-8') as f:
                    feat_info = json.load(f)
                    self.snn_feature_names = feat_info.get('feature_names', checkpoint.get('feature_names', []))
            else:
                self.snn_feature_names = checkpoint.get('feature_names', [])

            if os.path.exists(profiles_path):
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    self.snn_profiles = json.load(f)
            else:
                self.snn_profiles = {}

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.snn_metadata = json.load(f)
            else:
                self.snn_metadata = {}

            threshold_policy = self.snn_metadata.get('threshold_policy', {})
            test_metrics = self.snn_metadata.get('test_metrics', {})

            self.snn_threshold = float(test_metrics.get('optimal_threshold', checkpoint.get('optimal_threshold', 0.5)))
            self.snn_threshold_scale = float(threshold_policy.get('threshold_scale', 1.0))
            self.snn_global_threshold = float(threshold_policy.get('global_threshold', self.snn_threshold))
            self.snn_unknown_customer_policy = str(threshold_policy.get('unknown_customer_policy', 'global'))
            self.snn_time_steps = int(self.snn_metadata.get('time_steps', checkpoint.get('time_steps', 20)))

            self.snn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.snn_model = self.snn_model.to(self.snn_device)
            self.snn_model.eval()

            self.snn_model_info = {
                'model_type': 'SNN',
                'input_dim': int(checkpoint.get('input_size', len(self.snn_feature_names) or 24)),
                'architecture': f"SNN-FC{int(checkpoint.get('input_size', 24))}-{int(checkpoint.get('hidden_size', 64))}-{int(checkpoint.get('hidden_size', 64))}-2",
                'threshold': float(self.snn_threshold),
                'threshold_scale': float(self.snn_threshold_scale),
                'global_threshold': float(self.snn_global_threshold),
                'unknown_customer_policy': self.snn_unknown_customer_policy,
                'time_steps': int(self.snn_time_steps),
                'device': str(self.snn_device),
                'expected_features': len(self.snn_feature_names),
                'feature_names': self.snn_feature_names,
                'performance': {
                    'accuracy': self.snn_metadata.get('test_metrics', {}).get('accuracy', 0),
                    'precision': self.snn_metadata.get('test_metrics', {}).get('precision', 0),
                    'recall': self.snn_metadata.get('test_metrics', {}).get('recall', 0),
                    'f1': self.snn_metadata.get('test_metrics', {}).get('f1', 0),
                    'auc': self.snn_metadata.get('test_metrics', {}).get('auc', 0),
                }
            }

            print(f" SNN Model Loaded Successfully")
            print(f"   Input dimensions: {self.snn_model_info['input_dim']}")
            print(f"   Time steps: {self.snn_time_steps}")
            print(f"   Threshold: {self.snn_threshold:.6f} (scale={self.snn_threshold_scale:.2f})")
            print(f"   F1 Score: {self.snn_model_info['performance'].get('f1', 0):.4f}")

        except Exception as e:
            print(f" Failed to load SNN model: {e}")
            import traceback
            traceback.print_exc()

    def _build_snn_feature_vector(self, transaction_data):
        """Build SNN feature vector using same feature engineering as training."""
        tx = dict(transaction_data)

        cc_num = str(tx.get('cc_num', ''))
        category = str(tx.get('category', 'unknown'))
        gender = str(tx.get('gender', 'U'))

        def _to_float(value, default=0.0):
            try:
                if value is None:
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        amt = _to_float(tx.get('amt', 0.0))
        lat = _to_float(tx.get('lat', 0.0))
        lon = _to_float(tx.get('long', 0.0))
        city_pop = _to_float(tx.get('city_pop', 0.0))
        merch_lat = _to_float(tx.get('merch_lat', 0.0))
        merch_lon = _to_float(tx.get('merch_long', 0.0))

        dt_value = tx.get('trans_date_trans_time')
        dt = pd.to_datetime(dt_value, errors='coerce')
        if pd.isna(dt):
            hour = int(_to_float(tx.get('hour', 0)))
            day_of_week = int(_to_float(tx.get('day_of_week', 0)))
            day_of_month = int(_to_float(tx.get('day_of_month', 1)))
            month = int(_to_float(tx.get('month', 1)))
        else:
            hour = int(dt.hour)
            day_of_week = int(dt.dayofweek)
            day_of_month = int(dt.day)
            month = int(dt.month)

        distance = float(np.sqrt((lat - merch_lat) ** 2 + (lon - merch_lon) ** 2))
        log_amt = float(np.log1p(max(amt, 0.0)))
        amt_per_pop = float(amt / (city_pop + 1.0))
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))

        top_categories = [
            'gas_transport', 'grocery_pos', 'home', 'shopping_pos',
            'kids_pets', 'shopping_net', 'entertainment', 'food_dining'
        ]

        feature_map = {
            'amt': amt,
            'lat': lat,
            'long': lon,
            'city_pop': city_pop,
            'merch_lat': merch_lat,
            'merch_long': merch_lon,
            'hour': float(hour),
            'day_of_week': float(day_of_week),
            'day_of_month': float(day_of_month),
            'month': float(month),
            'distance': distance,
            'log_amt': log_amt,
            'amt_per_pop': amt_per_pop,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'cat_food_dining': float(category == 'food_dining'),
            'cat_gas_transport': float(category == 'gas_transport'),
            'cat_grocery_pos': float(category == 'grocery_pos'),
            'cat_home': float(category == 'home'),
            'cat_kids_pets': float(category == 'kids_pets'),
            'cat_other': float(category not in top_categories),
            'cat_shopping_net': float(category == 'shopping_net'),
            'cat_shopping_pos': float(category == 'shopping_pos'),
            'gender_M': float(gender == 'M'),
        }

        vector = np.array([feature_map.get(fname, 0.0) for fname in self.snn_feature_names], dtype=np.float32)
        return cc_num, vector

    def detect_fraud_snn(self, transaction_data, model_key='snn'):
        """Run customer-centric SNN fraud detection for a single transaction."""
        if self.snn_model is None:
            return None

        start_time = time.time()

        try:
            cc_num, feature_vector = self._build_snn_feature_vector(transaction_data)

            if feature_vector.shape[0] != len(self.snn_feature_names):
                self.stats[model_key]['preprocessing_errors'] += 1
                return None

            x_scaled = self.snn_scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32)
            x_tensor = torch.FloatTensor(x_scaled).to(self.snn_device)

            with torch.no_grad():
                output = self.snn_model(x_tensor, num_steps=self.snn_time_steps)
                prob = float(torch.softmax(output, dim=1)[0, 1].item())

            profile = self.snn_profiles.get(str(cc_num))
            if profile is None and self.snn_unknown_customer_policy == 'skip':
                return {
                    'detection_status': 'skipped_unknown_customer',
                    'is_fraud': False,
                    'fraud_probability': prob,
                    'decision_threshold': None,
                    'known_customer': False,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'SNN'
                }

            if profile is None:
                decision_threshold = max(self.snn_threshold, self.snn_global_threshold)
                known_customer = False
            else:
                decision_threshold = max(self.snn_threshold, float(profile.get('fraud_prob_threshold', self.snn_threshold)))
                known_customer = True

            raw_decision_threshold = float(decision_threshold * self.snn_threshold_scale)
            # Probabilities from softmax are in [0, 1), so keep threshold in a valid range.
            decision_threshold = float(np.clip(raw_decision_threshold, 0.0, 0.999999))
            is_fraud = prob >= decision_threshold
            confidence = prob if is_fraud else (1.0 - prob)

            if is_fraud:
                risk = 'High'
            elif prob < 0.3:
                risk = 'Low'
            elif prob < 0.5:
                risk = 'Medium-Low'
            else:
                risk = 'Medium-High'

            processing_time = (time.time() - start_time) * 1000

            self.stats[model_key]['total_processed'] += 1
            if is_fraud:
                self.stats[model_key]['fraud_detected'] += 1

            self.stats[model_key]['risk_distribution'][risk] = self.stats[model_key]['risk_distribution'].get(risk, 0) + 1

            if self.stats[model_key]['total_processed'] > 0:
                total_time = (self.stats[model_key]['avg_processing_time'] * (self.stats[model_key]['total_processed'] - 1)) + processing_time
                self.stats[model_key]['avg_processing_time'] = total_time / self.stats[model_key]['total_processed']

            result = {
                'fraud_probability': float(prob),
                'decision_threshold': float(decision_threshold),
                'raw_decision_threshold': float(raw_decision_threshold),
                'base_threshold': float(self.snn_threshold),
                'global_threshold': float(self.snn_global_threshold),
                'threshold_scale': float(self.snn_threshold_scale),
                'is_fraud': bool(is_fraud),
                'confidence': float(confidence),
                'risk_level': risk,
                'known_customer': bool(known_customer),
                'unknown_customer_policy': self.snn_unknown_customer_policy,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'SNN'
            }

            self.stats[model_key]['prediction_history'].append({
                'score': float(prob),
                'is_fraud': bool(is_fraud),
                'risk': risk,
                'timestamp': result['timestamp']
            })
            if len(self.stats[model_key]['prediction_history']) > 100:
                self.stats[model_key]['prediction_history'].pop(0)

            return result

        except Exception as e:
            self.stats[model_key]['preprocessing_errors'] += 1
            print(f"️ SNN detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_fraud_autoencoder(self, transaction_data, model_key='autoencoder'):
        """Run Autoencoder fraud detection"""
        if self.ae_model is None:
            return None
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Preprocess
            X_processed, _, _ = self.ae_preprocessor.preprocess(df, is_training=False)
            
            # Verify dimensions
            if X_processed.shape[1] != self.ae_model_info['expected_features']:
                self.stats[model_key]['preprocessing_errors'] += 1
                return None
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_processed).to(self.ae_device)
            
            # Get reconstruction error
            with torch.no_grad():
                error_value = self.ae_model.get_reconstruction_error(X_tensor)[0]
            
            # Determine if fraud
            is_fraud = error_value > self.ae_threshold
            fraud_prob = min(error_value / self.ae_threshold, 1.0)
            
            # Determine risk level
            if is_fraud:
                risk = "High"
            elif fraud_prob < 0.7:
                risk = "Low"
            else:
                risk = "Medium"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats[model_key]['total_processed'] += 1
            if is_fraud:
                self.stats[model_key]['fraud_detected'] += 1
            
            self.stats[model_key]['risk_distribution'][risk] = self.stats[model_key]['risk_distribution'].get(risk, 0) + 1
            
            if self.stats[model_key]['total_processed'] > 0:
                total_time = (self.stats[model_key]['avg_processing_time'] * (self.stats[model_key]['total_processed'] - 1)) + processing_time
                self.stats[model_key]['avg_processing_time'] = total_time / self.stats[model_key]['total_processed']
            
            # Create result
            result = {
                'reconstruction_error': float(error_value),
                'threshold': float(self.ae_threshold),
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'risk_level': risk,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Autoencoder'
            }
            
            # Store prediction in history
            self.stats[model_key]['prediction_history'].append({
                'error': float(error_value),
                'is_fraud': bool(is_fraud),
                'risk': risk,
                'timestamp': result['timestamp']
            })
            if len(self.stats[model_key]['prediction_history']) > 100:
                self.stats[model_key]['prediction_history'].pop(0)
            
            return result
            
        except Exception as e:
            self.stats[model_key]['preprocessing_errors'] += 1
            print(f"️ Autoencoder detection error: {e}")
            return None
    
    def detect_fraud_lstm(self, transaction_sequence, client_id, model_key='lstm'):
        """Run LSTM fraud detection"""
        if self.lstm_model is None:
            return None
        
        if len(transaction_sequence) < self.sequence_length:
            return None
        
        start_time = time.time()
        
        try:
            # Take last sequence_length transactions
            sequence_data = list(transaction_sequence)[-self.sequence_length:]
            
            # Convert to DataFrame
            df = pd.DataFrame(sequence_data)
            
            # Prepare features
            X_features = []
            for idx, row in df.iterrows():
                feature_vector = []
                for feat_name in self.lstm_feature_names:
                    if feat_name in row:
                        feature_vector.append(row[feat_name])
                    else:
                        feature_vector.append(0.0)
                X_features.append(feature_vector)
            
            X_features = np.array(X_features)
            
            # Verify dimensions
            if X_features.shape[1] != len(self.lstm_feature_names):
                self.stats[model_key]['preprocessing_errors'] += 1
                return None
            
            # Scale features
            X_scaled = self.lstm_scaler.transform(X_features)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.lstm_device)
            
            # Get prediction
            with torch.no_grad():
                output = self.lstm_model(X_tensor)
                # LSTM model returns (output, attention_weights) tuple
                if isinstance(output, tuple):
                    output = output[0]
                fraud_score = output.item()
            
            # Determine if fraud
            is_fraud = fraud_score >= self.lstm_threshold
            confidence = fraud_score if is_fraud else (1 - fraud_score)
            
            # Determine risk level
            if is_fraud:
                risk = "High"
            elif fraud_score < 0.3:
                risk = "Low"
            elif fraud_score < 0.5:
                risk = "Medium-Low"
            else:
                risk = "Medium-High"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats[model_key]['total_processed'] += 1
            if is_fraud:
                self.stats[model_key]['fraud_detected'] += 1
            
            self.stats[model_key]['risk_distribution'][risk] = self.stats[model_key]['risk_distribution'].get(risk, 0) + 1
            
            if self.stats[model_key]['total_processed'] > 0:
                total_time = (self.stats[model_key]['avg_processing_time'] * (self.stats[model_key]['total_processed'] - 1)) + processing_time
                self.stats[model_key]['avg_processing_time'] = total_time / self.stats[model_key]['total_processed']
            
            # Create result
            result = {
                'fraud_score': float(fraud_score),
                'optimal_threshold': float(self.lstm_threshold),
                'is_fraud': bool(is_fraud),
                'confidence': float(confidence),
                'risk_level': risk,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'sequence_length': self.sequence_length,
                'model_type': 'LSTM'
            }
            
            # Store prediction in history
            self.stats[model_key]['prediction_history'].append({
                'score': float(fraud_score),
                'is_fraud': bool(is_fraud),
                'risk': risk,
                'timestamp': result['timestamp']
            })
            if len(self.stats[model_key]['prediction_history']) > 100:
                self.stats[model_key]['prediction_history'].pop(0)
            
            return result
            
        except Exception as e:
            self.stats[model_key]['preprocessing_errors'] += 1
            print(f"️ LSTM detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_dataset(self, dataset_path):
        """Load and prepare dataset"""
        print(f" Loading dataset from: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Extract datetime features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date_trans_time'].dt.hour
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['day_of_month'] = df['trans_date_trans_time'].dt.day
            df['month'] = df['trans_date_trans_time'].dt.month
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1
        
        # Handle category column
        if 'category' in df.columns:
            top_categories = ['gas_transport', 'grocery_pos', 'home', 
                            'shopping_pos', 'food_dining', 'kids_pets', 
                            'shopping_net', 'other']
            for cat in top_categories:
                df[f'cat_{cat}'] = (df['category'] == cat).astype(int)
        
        # Handle gender
        if 'gender' in df.columns:
            df['gender_M'] = (df['gender'] == 'M').astype(int)
        else:
            df['gender_M'] = 0
        
        # Add transaction_id
        if 'transaction_id' not in df.columns and 'trans_num' in df.columns:
            df['transaction_id'] = df['trans_num']
        elif 'transaction_id' not in df.columns:
            df['transaction_id'] = [f'TXN_{i:06d}' for i in range(len(df))]
        
        # Feature engineering
        if 'distance' not in df.columns and 'lat' in df.columns and 'merch_lat' in df.columns:
            df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        elif 'distance' not in df.columns:
            df['distance'] = 0.0
        
        if 'log_amt' not in df.columns:
            df['log_amt'] = np.log1p(df['amt'])
        
        if 'amt_per_pop' not in df.columns and 'city_pop' in df.columns:
            df['amt_per_pop'] = df['amt'] / (df['city_pop'] + 1)
        elif 'amt_per_pop' not in df.columns:
            df['amt_per_pop'] = 0.0
        
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        
        if 'hour_cos' not in df.columns:
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        records = df.to_dict('records')
        
        print(f" Loaded dataset with {len(records)} records")
        
        return records
    
    def prepare_record(self, record, index):
        """Prepare a record for streaming"""
        prepared = {}
        for key, value in record.items():
            if isinstance(value, (np.integer, np.int64)):
                prepared[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                prepared[key] = float(value)
            elif isinstance(value, np.ndarray):
                prepared[key] = value.tolist()
            elif pd.isna(value):
                prepared[key] = None
            elif isinstance(value, pd.Timestamp):
                prepared[key] = value.isoformat()
            else:
                prepared[key] = value
        
        prepared['stream_index'] = index
        prepared['stream_timestamp'] = datetime.now().isoformat()
        prepared['is_fraud'] = bool(record.get('is_fraud', 0)) if 'is_fraud' in record else False
        
        try:
            amt = prepared.get('amt', 0) if prepared.get('amt', 0) is not None else 0
            prepared['amt'] = float(amt)
        except Exception:
            prepared['amt'] = 0.0
        
        return prepared
    
    async def stream_to_client(self, websocket):
        """Stream dataset records to client with selected model detection"""
        client_id = id(websocket)
        model_type = self.client_models.get(websocket, 'autoencoder')
        
        print(f" Starting {model_type.upper()} stream to client {client_id}")
        
        if websocket not in self.client_totals:
            self.client_totals[websocket] = 0.0
        
        index = 0
        total = len(self.dataset)
        
        while self.streaming_clients.get(websocket, False):
            try:
                model_type = self.client_models.get(websocket, 'autoencoder')

                # Initialize LSTM sequence buffer only when needed.
                if model_type == 'lstm' and client_id not in self.client_sequences:
                    self.client_sequences[client_id] = deque(maxlen=self.sequence_length)

                # Get and prepare record
                record = self.dataset[index]
                prepared_record = self.prepare_record(record, index)
                
                # Update running total
                try:
                    self.client_totals[websocket] += float(prepared_record.get('amt', 0) or 0)
                except Exception:
                    pass
                
                # Prepare payload
                payload = dict(prepared_record)
                payload['stream_total_amount'] = self.client_totals.get(websocket, 0.0)
                payload['active_model'] = model_type
                
                # Perform detection based on selected model
                if model_type == 'lstm':
                    # LSTM detection
                    self.client_sequences[client_id].append(prepared_record)
                    detection_result = self.detect_fraud_lstm(self.client_sequences[client_id], client_id)
                    
                    if detection_result:
                        payload.update(detection_result)
                        payload['transaction_data'] = dict(prepared_record)
                        payload['transaction_id'] = prepared_record.get('transaction_id', f'TXN_{index:06d}')
                    else:
                        if len(self.client_sequences[client_id]) < self.sequence_length:
                            payload['detection_status'] = 'building_sequence'
                            payload['sequence_progress'] = f"{len(self.client_sequences[client_id])}/{self.sequence_length}"
                        else:
                            payload['detection_error'] = True
                elif model_type == 'snn':
                    # SNN detection
                    detection_result = self.detect_fraud_snn(prepared_record)

                    if detection_result:
                        payload.update(detection_result)
                        payload['transaction_data'] = dict(prepared_record)
                        payload['transaction_id'] = prepared_record.get('transaction_id', f'TXN_{index:06d}')
                    else:
                        payload['detection_error'] = True
                else:
                    # Autoencoder detection
                    detection_result = self.detect_fraud_autoencoder(prepared_record)
                    
                    if detection_result:
                        payload.update(detection_result)
                        payload['transaction_data'] = dict(prepared_record)
                        payload['transaction_id'] = prepared_record.get('transaction_id', f'TXN_{index:06d}')
                    else:
                        payload['detection_error'] = True
                
                # Save to MongoDB if detection was successful
                if detection_result and self.db and self.db.connected:
                    try:
                        db_payload = dict(payload)
                        db_payload['inserted_at'] = datetime.now().isoformat()
                        
                        result_id = self.db.insert_fraud_result(db_payload)
                        
                        if result_id:
                            payload['db_id'] = result_id
                            payload['inserted_at'] = db_payload['inserted_at']
                        
                        if db_payload.get('is_fraud'):
                            self.db.insert_immediate_alert(db_payload)
                            
                    except Exception as e:
                        print(f"   ️ MongoDB save error: {e}")
                
                # Send record to frontend
                await websocket.send(json.dumps(payload))
                
                # Move to next record
                index += 1
                
                if index >= total:
                    print(f" End of dataset reached — restarting from beginning ({model_type.upper()})")
                    self.stats[model_type]['dataset_loops'] += 1
                    index = 0
                
                # Control streaming speed
                await asyncio.sleep(1.0 / self.stream_speed)
                
            except websockets.exceptions.ConnectionClosed:
                print("️ Client disconnected during stream")
                break
            except Exception as e:
                print(f" Error sending record: {e}")
                import traceback
                traceback.print_exc()
                index += 1
        
        print(f" {model_type.upper()} stream stopped for client {client_id}")
        
        # Clean up
        if client_id in self.client_sequences:
            del self.client_sequences[client_id]
        if websocket not in self.clients:
            self.client_totals.pop(websocket, None)
    
    async def handler(self, websocket):
        """Handle client commands"""
        client_id = id(websocket)
        self.clients.add(websocket)
        
        # Default to autoencoder model
        self.client_models[websocket] = 'autoencoder'
        
        print(f" Client connected (ID: {client_id})")

        if self.always_streaming and not self.streaming_clients.get(websocket, False):
            self.streaming_clients[websocket] = True
            self.stream_tasks[websocket] = asyncio.create_task(self.stream_to_client(websocket))
            await websocket.send(json.dumps({
                "status": "streaming_started",
                "streaming": True,
                "stream_speed": self.stream_speed,
                "model_type": self.client_models.get(websocket, 'autoencoder'),
                "always_streaming": True
            }))
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command", "").lower()
                    
                    if command == "ping":
                        await websocket.send(json.dumps({
                            "response": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif command == "set_model":
                        model_type = data.get("model", "autoencoder").lower()
                        if model_type in ['autoencoder', 'lstm', 'snn']:
                            old_model = self.client_models.get(websocket, 'autoencoder')
                            self.client_models[websocket] = model_type

                            # Reset LSTM sequence state when switching in/out of LSTM mode.
                            if old_model == 'lstm' or model_type == 'lstm':
                                self.client_sequences[client_id] = deque(maxlen=self.sequence_length)

                            print(f" Client {client_id} switched from {old_model} to {model_type}")
                            
                            await websocket.send(json.dumps({
                                "status": "model_changed",
                                "model": model_type,
                                "previous_model": old_model
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "error": f"Invalid model type: {model_type}. Use 'autoencoder', 'lstm', or 'snn'"
                            }))
                    
                    elif command == "get_model_info":
                        model_type = self.client_models.get(websocket, 'autoencoder')
                        if model_type == 'lstm':
                            model_info = self.lstm_model_info
                        elif model_type == 'snn':
                            model_info = self.snn_model_info
                        else:
                            model_info = self.ae_model_info
                        
                        await websocket.send(json.dumps({
                            "model_info": model_info,
                            "active_model": model_type,
                            "available_models": ["autoencoder", "lstm", "snn"]
                        }))
                    
                    elif command == "get_stats":
                        model_type = self.client_models.get(websocket, 'autoencoder')
                        await websocket.send(json.dumps({
                            "stats": self._build_stats_payload(model_type),
                            "active_model": model_type
                        }))

                    elif command == "reset_stats":
                        model_type = self.client_models.get(websocket, 'autoencoder')
                        self._reset_stats_for_model(model_type)
                        self.client_totals[websocket] = 0.0

                        if model_type == 'lstm':
                            self.client_sequences[client_id] = deque(maxlen=self.sequence_length)

                        await websocket.send(json.dumps({
                            "status": "stats_reset",
                            "active_model": model_type,
                            "stats": self._build_stats_payload(model_type)
                        }))
                    
                    elif command == "start_stream":
                        if not self.streaming_clients.get(websocket, False):
                            model_type = self.client_models.get(websocket, 'autoencoder')
                            print(f"️ Start {model_type.upper()} stream for client {client_id}")

                            self.streaming_clients[websocket] = True
                            task = asyncio.create_task(self.stream_to_client(websocket))
                            self.stream_tasks[websocket] = task

                            await websocket.send(json.dumps({
                                "status": "streaming_started",
                                "streaming": True,
                                "stream_speed": self.stream_speed,
                                "model_type": model_type,
                                "always_streaming": self.always_streaming
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "status": "already_streaming",
                                "streaming": True,
                                "always_streaming": self.always_streaming
                            }))
                    
                    elif command == "stop_stream":
                        if self.always_streaming:
                            await websocket.send(json.dumps({
                                "status": "always_on_streaming",
                                "streaming": True,
                                "message": "Streaming is always enabled and cannot be stopped from frontend.",
                                "always_streaming": True
                            }))
                        else:
                            print(f"️ Stop stream for client {client_id}")
                            self.streaming_clients[websocket] = False
                            task = self.stream_tasks.pop(websocket, None)
                            if task and not task.done():
                                try:
                                    task.cancel()
                                except Exception:
                                    pass

                            await websocket.send(json.dumps({
                                "status": "stopped",
                                "streaming": False,
                                "stream_total_amount": self.client_totals.get(websocket, 0.0)
                            }))
                            self.client_totals[websocket] = 0.0
                    
                    elif command == "get_status":
                        model_type = self.client_models.get(websocket, 'autoencoder')
                        await websocket.send(json.dumps({
                            "streaming": self.streaming_clients.get(websocket, False),
                            "total_records": len(self.dataset),
                            "stream_speed": self.stream_speed,
                            "active_clients": len(self.clients),
                            "always_streaming": self.always_streaming,
                            "hourly_cleanup_enabled": bool(self.db and self.db.connected),
                            "stream_total_amount": self.client_totals.get(websocket, 0.0),
                            "active_model": model_type,
                            "available_models": ["autoencoder", "lstm", "snn"],
                            "stats": self._build_stats_payload(model_type)
                        }))
                    
                    elif command == "set_speed":
                        speed = data.get("speed")
                        if isinstance(speed, (int, float)) and speed > 0:
                            self.stream_speed = speed
                            print(f"️ Stream speed changed to {speed} records/sec")
                            await websocket.send(json.dumps({
                                "status": "speed_updated",
                                "speed": self.stream_speed
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "error": "Invalid speed value"
                            }))
                    
                    else:
                        await websocket.send(json.dumps({
                            "error": f"Unknown command: {command}"
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON"
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f" Client {client_id} disconnected")
        finally:
            print(f" Cleaning up client {client_id}")
            self.clients.discard(websocket)
            self.streaming_clients.pop(websocket, None)
            self.client_totals.pop(websocket, None)
            self.client_models.pop(websocket, None)
            if client_id in self.client_sequences:
                del self.client_sequences[client_id]
            task = self.stream_tasks.pop(websocket, None)
            if task and not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass

    async def run_hourly_cleanup(self):
        """Create hourly reports and remove detailed stream/prediction documents.
        Runs once per hour aligned to the clock hour boundary.
        """
        print(" Hourly MongoDB cleanup task started (interval: 1 hour)")

        while True:
            try:
                # Sleep until the next top-of-hour boundary first
                now = datetime.utcnow()
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
                wait_seconds = max(0.0, (next_hour - now).total_seconds())
                await asyncio.sleep(wait_seconds)

                if self.db and self.db.connected:
                    cleanup_result = self.db.summarize_and_cleanup_previous_hour()
                    if cleanup_result.get('processed'):
                        print(
                            f" Hourly report saved for {cleanup_result.get('hour_start')} "
                            f"and raw docs cleaned: {cleanup_result.get('deleted', {})}"
                        )
                    else:
                        reason = cleanup_result.get('reason', 'unknown')
                        print(f" Hourly cleanup skipped: {reason}")
            except asyncio.CancelledError:
                print(" Hourly cleanup task cancelled")
                raise
            except Exception as e:
                print(f"️ Hourly cleanup task error: {e}")


async def main():
    """Main server function"""
    # Configuration
    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'fraudTest.csv'
    )
    HOST = "0.0.0.0"
    PORT = 8765  # Single unified port
    STREAM_SPEED = 1.0
    SEQUENCE_LENGTH = 10
    
    print("="*80)
    print("[>>] UNIFIED FRAUD DETECTION WEBSOCKET SERVER")
    print("="*80)
    print(f"[DIR] Dataset: {DATASET_PATH}")
    print(f"[NET] Host: {HOST}")
    print(f"[PORT] Port: {PORT}")
    print(f"[SPD] Stream Speed: {STREAM_SPEED} records/sec")
    print(f"[SEQ] LSTM Sequence Length: {SEQUENCE_LENGTH}")
    print(f"[AI]  Models: Autoencoder + LSTM + SNN (switchable)")
    print("="*80)
    
    # Create server instance
    server = UnifiedFraudDetectionServer(
        dataset_path=DATASET_PATH,
        stream_speed=STREAM_SPEED,
        sequence_length=SEQUENCE_LENGTH
    )

    # Start background hourly summarize/cleanup task.
    server.hourly_cleanup_task = asyncio.create_task(server.run_hourly_cleanup())
    
    # Start WebSocket server
    print(f"\n[OK] Server starting on ws://{HOST}:{PORT}")
    print("   Waiting for connections...\n")
    
    async with websockets.serve(server.handler, HOST, PORT, ping_interval=30, ping_timeout=10):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[--] Server shutting down...")
    except Exception as e:
        print(f"\n[ERR] Server error: {e}")
        import traceback
        traceback.print_exc()
