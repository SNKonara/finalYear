"""
FastAPI Server for Batch Fraud Detection
Supports CSV upload, model selection, fraud prediction, and PDF report generation
"""
import os
import sys
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np
import torch
import joblib

from bson import ObjectId
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model imports
from AEmodel.preprocessor import DataPreprocessor as AEPreprocessor
from AEmodel.model import FraudAutoencoder
from LSTMmodel.preprocessor import prepare_improved_lstm_data
from LSTMmodel.save_load import load_model as load_lstm_model
from SNNmodel.customer_behavior_snn import SpikingFraudDetector
from database.mongodb import get_mongodb_instance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="NeuroDetect Batch API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
MODELS = {}
PREPROCESSORS = {}
MODEL_CONFIGS = {}

# Paths
# batch_api.py is at: C:\finalYear\NeuroDetect\backend\server\batch_api.py
# We need to reach: C:\finalYear\saved_models
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to C:\finalYear
PROJECT_DIR = Path(__file__).parent.parent.parent  # C:\finalYear\NeuroDetect
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results" / "batch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Log paths for debugging
logger.info(f"BASE_DIR: {BASE_DIR.absolute()}")
logger.info(f"PROJECT_DIR: {PROJECT_DIR.absolute()}")
logger.info(f"SAVED_MODELS_DIR: {SAVED_MODELS_DIR.absolute()}")
logger.info(f"SAVED_MODELS_DIR exists: {SAVED_MODELS_DIR.exists()}")

AUTH_USERS_COLLECTION = 'users'
AUTH_SESSIONS_COLLECTION = 'auth_sessions'
VALID_ROLES = {'admin', 'analyst', 'viewer'}


class BatchProcessor:
    """Handles batch fraud detection processing"""
    
    def __init__(self):
        self.db = get_mongodb_instance()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def _extract_autoencoder_architecture(self, state_dict: dict[str, Any], input_dim: int) -> tuple[str, str]:
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
        
    def load_autoencoder_model(self):
        """Load Autoencoder model and preprocessor"""
        try:
            logger.info("Loading Autoencoder model...")
            logger.info(f"Looking for model files in: {SAVED_MODELS_DIR}")
            
            # Load threshold
            threshold_path = SAVED_MODELS_DIR / "threshold.json"
            if not threshold_path.exists():
                raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data['threshold']
            logger.info(f"Loaded threshold: {threshold}")
            
            # Load features
            features_path = SAVED_MODELS_DIR / "features.json"
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")
            with open(features_path, 'r') as f:
                features_data = json.load(f)
                feature_names = features_data['feature_names']
                num_features = features_data['num_features']
                top_categories = features_data.get('top_categories', [])
                category_columns = features_data.get('category_columns', [])
            logger.info(f"Loaded features: {num_features} features")
            
            # Initialize preprocessor
            preprocessor = AEPreprocessor()
            preprocessor.top_categories = top_categories
            preprocessor.category_columns = category_columns
            
            # Load scaler
            scaler_path = SAVED_MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                import joblib
                preprocessor.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
            
            # Load model
            model_path = SAVED_MODELS_DIR / "autoencoder.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different save formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint contains dictionary with model_state_dict
                state_dict = checkpoint['model_state_dict']
                logger.info("Loaded model from checkpoint dictionary")
            else:
                # Direct state_dict
                state_dict = checkpoint
                logger.info("Loaded model directly")

            architecture, architecture_readable = self._extract_autoencoder_architecture(state_dict, num_features)
            
            model = FraudAutoencoder(input_dim=num_features)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info("Model loaded and set to eval mode")
            
            MODELS['autoencoder'] = model
            PREPROCESSORS['autoencoder'] = preprocessor
            MODEL_CONFIGS['autoencoder'] = {
                'threshold': threshold,
                'feature_names': feature_names,
                'num_features': num_features,
                'architecture': architecture,
                'architecture_readable': architecture_readable,
                'device': str(self.device),
                'performance': {},
            }
            
            logger.info(f"✓ Autoencoder loaded successfully - {num_features} features, threshold: {threshold:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Autoencoder: {e}", exc_info=True)
            return False
    
    def load_lstm_model(self):
        """Load LSTM model and preprocessor"""
        try:
            logger.info("Loading LSTM model...")
            logger.info(f"Looking for model files in: {SAVED_MODELS_DIR}")
            
            # Load LSTM model first (it contains scaler and config)
            model_path = SAVED_MODELS_DIR / "enhanced_lstm_fraud_model.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"LSTM model file not found: {model_path}")
            
            # load_lstm_model returns (model, scaler, feature_names, model_config, results)
            model, lstm_scaler, lstm_features, lstm_config, lstm_results = load_lstm_model(str(model_path))
            logger.info(f"Loaded LSTM model with config: {lstm_config}")
            
            # Get threshold from results or config file
            threshold = lstm_results.get('optimal_threshold', 0.5)
            
            # Load additional config if available
            config_path = SAVED_MODELS_DIR / "enhanced_lstm_fraud_model.json"
            performance = dict(lstm_results or {})
            architecture = f"BiLSTM-{int(lstm_config.get('hidden_dim', 256))}-{int(lstm_config.get('num_layers', 2))}L + Attention"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                feature_names = config['feature_names']
                threshold = config['performance']['optimal_threshold']
                sequence_length = config['input_shape'][0]
                performance = config.get('performance', performance)
                model_config = config.get('model_config', {})
                architecture = f"BiLSTM-{int(model_config.get('hidden_dim', lstm_config.get('hidden_dim', 256)))}-{int(model_config.get('num_layers', lstm_config.get('num_layers', 2)))}L + Attention"
                logger.info(f"Loaded LSTM config from JSON: {len(feature_names)} features, threshold: {threshold}")
            else:
                feature_names = lstm_features
                sequence_length = 10  # Default sequence length
                logger.warning(f"LSTM config JSON not found, using defaults")
            
            # Initialize preprocessor with LSTM's scaler
            preprocessor = AEPreprocessor()
            preprocessor.scaler = lstm_scaler  # Use scaler from LSTM model package
            preprocessor.top_categories = ['gas_transport', 'grocery_pos', 'home', 'shopping_pos', 
                                          'kids_pets', 'shopping_net', 'entertainment', 'food_dining']
            # Category columns are indices 15-23 in feature_names
            preprocessor.category_columns = [f for f in feature_names if f.startswith('cat_')]
            logger.info(f"Configured preprocessor with {len(preprocessor.category_columns)} category columns")
            
            model.to(self.device)
            model.eval()
            logger.info("LSTM model loaded and set to eval mode")
            
            MODELS['lstm'] = model
            PREPROCESSORS['lstm'] = preprocessor
            MODEL_CONFIGS['lstm'] = {
                'threshold': threshold,
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'sequence_length': sequence_length,
                'architecture': architecture,
                'device': str(self.device),
                'hidden_size': int(lstm_config.get('hidden_dim', 256)),
                'num_layers': int(lstm_config.get('num_layers', 2)),
                'performance': {
                    'accuracy': float(performance.get('accuracy', 0.0)),
                    'precision': float(performance.get('fraud_precision', performance.get('precision', 0.0))),
                    'recall': float(performance.get('fraud_recall', performance.get('recall', 0.0))),
                    'f1': float(performance.get('fraud_f1', performance.get('f1_score', performance.get('f1', 0.0)))),
                    'f1_score': float(performance.get('f1_score', performance.get('f1', 0.0))),
                    'fraud_f1': float(performance.get('fraud_f1', performance.get('f1_score', performance.get('f1', 0.0)))),
                    'roc_auc': float(performance.get('roc_auc', performance.get('auc', 0.0))),
                    'auc': float(performance.get('roc_auc', performance.get('auc', 0.0))),
                    'optimal_threshold': float(performance.get('optimal_threshold', threshold)),
                }
            }
            
            logger.info(f"✓ LSTM loaded successfully - {len(feature_names)} features, threshold: {threshold:.6f}, sequence: {sequence_length}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM: {e}", exc_info=True)
            return False

    def _find_snn_package_dir(self) -> Optional[Path]:
        """Find an SNN package directory that contains model/scaler/features files."""
        candidate_dirs = [
            PROJECT_DIR / "backend" / "snn_models" / "final_customer_snn_package",
            BASE_DIR / "backend" / "snn_models" / "final_customer_snn_package",
            BASE_DIR / "backend" / "server" / "snn_models" / "final_customer_snn_package",
            BASE_DIR / "snn_models" / "final_customer_snn_package",
        ]

        required_files = ["snn_customer_classifier.pth", "scaler.pkl", "features.json"]

        for candidate in candidate_dirs:
            if candidate.exists() and all((candidate / fname).exists() for fname in required_files):
                return candidate

        search_roots = [
            PROJECT_DIR / "backend" / "snn_models",
            BASE_DIR / "backend" / "snn_models",
            BASE_DIR / "backend" / "server" / "snn_models",
            BASE_DIR / "snn_models",
        ]

        valid_dirs = []
        for root in search_roots:
            if not root.exists():
                continue
            for model_path in root.rglob("snn_customer_classifier.pth"):
                package_dir = model_path.parent
                if all((package_dir / fname).exists() for fname in required_files):
                    valid_dirs.append(package_dir)

        if not valid_dirs:
            return None

        valid_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return valid_dirs[0]

    def load_snn_model(self):
        """Load customer-centric SNN model package."""
        try:
            logger.info("Loading SNN model...")

            snn_dir = self._find_snn_package_dir()
            if snn_dir is None:
                raise FileNotFoundError("Could not find a valid SNN package directory")

            logger.info(f"Using SNN package directory: {snn_dir}")

            model_path = snn_dir / "snn_customer_classifier.pth"
            scaler_path = snn_dir / "scaler.pkl"
            features_path = snn_dir / "features.json"
            profiles_path = snn_dir / "customer_profiles.json"
            metadata_path = snn_dir / "training_metadata.json"

            checkpoint = torch.load(model_path, map_location='cpu')

            model = SpikingFraudDetector(
                input_size=int(checkpoint.get('input_size', 24)),
                hidden_size=int(checkpoint.get('hidden_size', 64)),
                output_size=int(checkpoint.get('output_size', 2)),
                beta=float(checkpoint.get('beta', 0.95)),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            scaler = joblib.load(scaler_path)

            with open(features_path, 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
            feature_names = feature_info.get('feature_names', checkpoint.get('feature_names', []))

            customer_profiles = {}
            if profiles_path.exists():
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    customer_profiles = json.load(f)

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            threshold_policy = metadata.get('threshold_policy', {})
            test_metrics = metadata.get('test_metrics', {})

            base_threshold = float(test_metrics.get('optimal_threshold', checkpoint.get('optimal_threshold', 0.5)))
            threshold_scale = float(threshold_policy.get('threshold_scale', 1.0))
            global_threshold = float(threshold_policy.get('global_threshold', base_threshold))
            unknown_customer_policy = str(threshold_policy.get('unknown_customer_policy', 'global'))
            time_steps = int(metadata.get('time_steps', checkpoint.get('time_steps', 20)))

            MODELS['snn'] = model
            PREPROCESSORS['snn'] = {
                'scaler': scaler,
                'customer_profiles': customer_profiles,
            }
            MODEL_CONFIGS['snn'] = {
                'threshold': base_threshold,
                'threshold_scale': threshold_scale,
                'global_threshold': global_threshold,
                'unknown_customer_policy': unknown_customer_policy,
                'time_steps': time_steps,
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'architecture': f"SNN-FC{int(checkpoint.get('input_size', len(feature_names) or 24))}-{int(checkpoint.get('hidden_size', 64))}-{int(checkpoint.get('hidden_size', 64))}-2",
                'device': str(self.device),
                'performance': {
                    'accuracy': float(test_metrics.get('accuracy', 0.0)),
                    'precision': float(test_metrics.get('precision', 0.0)),
                    'recall': float(test_metrics.get('recall', 0.0)),
                    'f1': float(test_metrics.get('f1', 0.0)),
                    'auc': float(test_metrics.get('auc', 0.0)),
                },
                'model_dir': str(snn_dir),
            }

            logger.info(
                f"✓ SNN loaded successfully - {len(feature_names)} features, "
                f"threshold: {base_threshold:.6f}, time_steps: {time_steps}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load SNN: {e}", exc_info=True)
            return False

    def _build_snn_feature_vector(self, transaction_data: dict, feature_names: list[str], include_map: bool = False):
        """Build SNN feature vector using customer-centric feature engineering."""
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

        vector = np.array([feature_map.get(fname, 0.0) for fname in feature_names], dtype=np.float32)
        if include_map:
            return cc_num, vector, feature_map
        return cc_num, vector

    def build_snn_explanations(
        self,
        df: pd.DataFrame,
        fraud_scores: np.ndarray,
        predictions: np.ndarray,
        decision_thresholds: np.ndarray,
        feature_names: list[str],
        feature_maps: list[dict[str, float]],
        X_scaled: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Build graph-ready explainability payload for each SNN prediction."""
        explanations: list[dict[str, Any]] = []

        if len(df) == 0:
            return explanations

        amt_series = pd.to_numeric(df.get('amt', pd.Series(dtype=float)), errors='coerce')
        distance_series = pd.Series([fmap.get('distance', 0.0) for fmap in feature_maps], dtype=float)

        amount_p90 = float(amt_series.quantile(0.9)) if len(amt_series.dropna()) > 0 else 0.0
        distance_p90 = float(distance_series.quantile(0.9)) if len(distance_series.dropna()) > 0 else 0.0

        feature_index = {name: idx for idx, name in enumerate(feature_names)}
        group_features = {
            'Amount': ['amt', 'log_amt', 'amt_per_pop'],
            'Location': ['distance', 'lat', 'long'],
            'Time': ['hour', 'hour_sin', 'hour_cos'],
            'Category': [name for name in feature_names if name.startswith('cat_')],
        }

        for row_idx in range(len(df)):
            score = float(fraud_scores[row_idx])
            threshold = float(decision_thresholds[row_idx])
            margin = score - threshold
            row_scaled = X_scaled[row_idx]
            fmap = feature_maps[row_idx]

            abs_scaled = np.abs(row_scaled)
            ranked_idx = np.argsort(abs_scaled)[::-1][:6]
            contribution_total = float(abs_scaled.sum()) if float(abs_scaled.sum()) > 0 else 1.0

            top_factors = []
            for idx in ranked_idx:
                fname = feature_names[int(idx)]
                contribution = float((abs_scaled[int(idx)] / contribution_total) * 100.0)
                top_factors.append({
                    'feature': fname,
                    'value': float(fmap.get(fname, 0.0)),
                    'scaled_value': float(row_scaled[int(idx)]),
                    'contribution_pct': round(contribution, 2),
                })

            reason_lines = []
            if int(predictions[row_idx]) == 1:
                reason_lines.append(f"Fraud probability ({score:.4f}) exceeds decision threshold ({threshold:.4f})")
            else:
                reason_lines.append(f"Fraud probability ({score:.4f}) remains below decision threshold ({threshold:.4f})")

            amount_value = float(fmap.get('amt', 0.0))
            distance_value = float(fmap.get('distance', 0.0))
            hour_value = int(fmap.get('hour', 0.0))
            if amount_value >= amount_p90 and amount_p90 > 0:
                reason_lines.append(f"Amount is unusually high for this batch (${amount_value:.2f} vs p90 ${amount_p90:.2f})")
            if distance_value >= distance_p90 and distance_p90 > 0:
                reason_lines.append(f"Merchant distance is unusually large ({distance_value:.3f} vs p90 {distance_p90:.3f})")
            if hour_value <= 5 or hour_value >= 23:
                reason_lines.append(f"Transaction happened at an unusual hour ({hour_value}:00)")
            if float(fmap.get('cat_shopping_net', 0.0)) == 1.0:
                reason_lines.append("Category indicates online shopping behavior (shopping_net)")

            reason_lines = reason_lines[:4]

            group_values = []
            group_labels = []
            for group_name, features in group_features.items():
                indices = [feature_index[f] for f in features if f in feature_index]
                if not indices:
                    group_score = 0.0
                else:
                    group_score = float(np.mean(np.abs(row_scaled[indices])) / 3.0)
                group_labels.append(group_name)
                group_values.append(round(float(np.clip(group_score, 0.0, 1.0)), 4))

            explanations.append({
                'decision_margin': round(margin, 6),
                'is_above_threshold': bool(score >= threshold),
                'reasons': reason_lines,
                'top_factors': top_factors,
                'graph_data': {
                    'feature_contribution_chart': {
                        'labels': [factor['feature'] for factor in top_factors],
                        'values': [factor['contribution_pct'] for factor in top_factors],
                    },
                    'threshold_chart': {
                        'probability': round(score, 6),
                        'threshold': round(threshold, 6),
                        'margin': round(margin, 6),
                    },
                    'risk_dimension_chart': {
                        'labels': group_labels,
                        'values': group_values,
                    }
                }
            })

        return explanations

    def predict_snn(self, df: pd.DataFrame):
        """Run SNN predictions with batch inference and customer-aware thresholds."""
        try:
            model = MODELS['snn']
            config = MODEL_CONFIGS['snn']
            scaler = PREPROCESSORS['snn']['scaler']
            customer_profiles = PREPROCESSORS['snn'].get('customer_profiles', {})

            feature_names = config['feature_names']
            base_threshold = float(config['threshold'])
            threshold_scale = float(config.get('threshold_scale', 1.0))
            global_threshold = float(config.get('global_threshold', base_threshold))
            unknown_policy = str(config.get('unknown_customer_policy', 'global'))
            time_steps = int(config.get('time_steps', 20))

            records = df.to_dict('records')
            cc_nums = []
            vectors = []
            feature_maps = []

            for record in records:
                cc_num, vector, feature_map = self._build_snn_feature_vector(record, feature_names, include_map=True)
                cc_nums.append(cc_num)
                vectors.append(vector)
                feature_maps.append(feature_map)

            X = np.vstack(vectors).astype(np.float32)
            X_scaled = scaler.transform(X).astype(np.float32)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            probs = []
            batch_size = 256
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    output = model(batch, num_steps=time_steps)
                    batch_probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                    probs.extend(batch_probs.tolist())

            fraud_scores = np.array(probs, dtype=np.float32)

            decision_thresholds = []
            for cc_num in cc_nums:
                profile = customer_profiles.get(str(cc_num))
                if profile is not None:
                    thr = max(base_threshold, float(profile.get('fraud_prob_threshold', base_threshold)))
                    thr = float(thr * threshold_scale)
                elif unknown_policy == 'skip':
                    thr = float('inf')
                else:
                    thr = max(base_threshold, global_threshold)
                    thr = float(thr * threshold_scale)
                decision_thresholds.append(thr)

            decision_thresholds_arr = np.array(decision_thresholds, dtype=np.float32)
            predictions = (fraud_scores >= decision_thresholds_arr).astype(int)

            logger.info(f"SNN prediction complete: {predictions.sum()} frauds detected")
            return predictions, fraud_scores, decision_thresholds_arr, X, X_scaled, feature_maps

        except Exception as e:
            logger.error(f"SNN prediction error: {e}", exc_info=True)
            raise
    
    def preprocess_data(self, df: pd.DataFrame, model_type: str):
        """Preprocess data for prediction"""
        try:
            preprocessor = PREPROCESSORS[model_type]
            
            # Preprocess - returns (X_scaled, y, feature_columns)
            X_scaled, _, feature_columns = preprocessor.preprocess(df.copy(), is_training=False, save_scaler=False)
            
            logger.info(f"Preprocessed data shape: {X_scaled.shape}")
            logger.info(f"Features: {feature_columns[:5] if feature_columns else 'N/A'}...")
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}", exc_info=True)
            raise
    
    def predict_autoencoder(self, X_scaled: np.ndarray):
        """Run Autoencoder predictions"""
        model = MODELS['autoencoder']
        threshold = MODEL_CONFIGS['autoencoder']['threshold']
        
        # Convert to tensor
        X = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            reconstructed = model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Classify
        predictions = (errors > threshold).astype(int)
        fraud_scores = errors
        
        return predictions, fraud_scores
    
    def predict_lstm(self, X_scaled: np.ndarray):
        """Run LSTM predictions with optimized batch processing"""
        try:
            model = MODELS['lstm']
            config = MODEL_CONFIGS['lstm']
            threshold = config['threshold']
            sequence_length = config['sequence_length']
            
            n_samples = len(X_scaled)
            logger.info(f"LSTM prediction: {n_samples} samples, sequence_length={sequence_length}")
            
            # Prepare all sequences at once for batch processing
            sequences = []
            for i in range(n_samples):
                if i < sequence_length - 1:
                    # Not enough history - use padding
                    pad_length = sequence_length - i - 1
                    sequence = np.vstack([
                        np.zeros((pad_length, X_scaled.shape[1])),
                        X_scaled[:i+1]
                    ])
                else:
                    sequence = X_scaled[i-sequence_length+1:i+1]
                sequences.append(sequence)
            
            # Convert to batch tensor
            sequences_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)
            logger.info(f"Created sequence tensor: {sequences_tensor.shape}")
            
            # Batch prediction (faster than one-by-one)
            batch_size = 64  # Process 64 sequences at a time
            all_scores = []
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch = sequences_tensor[i:i+batch_size]
                    outputs, attention_weights = model(batch)  # LSTM returns (output, attention_weights)
                    scores = outputs.cpu().numpy().flatten()  # outputs is already probability (has sigmoid)
                    all_scores.extend(scores)
                    
                    if (i + batch_size) % 500 == 0:
                        logger.info(f"  Processed {min(i+batch_size, n_samples)}/{n_samples} samples...")
            
            fraud_scores = np.array(all_scores)
            predictions = (fraud_scores >= threshold).astype(int)
            
            logger.info(f"LSTM prediction complete: {predictions.sum()} frauds detected")
            return predictions, fraud_scores
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}", exc_info=True)
            raise
    
    def save_to_mongodb(self, results_df: pd.DataFrame, batch_id: str, model_type: str, stats: dict):
        """Save results to MongoDB - batch summary and fraud results"""
        if not self.db or not self.db.connected:
            logger.warning("MongoDB not connected - skipping database save")
            return False
        
        try:
            def _mongo_safe(value: Any):
                if isinstance(value, (np.integer,)):
                    return int(value)
                if isinstance(value, (np.floating,)):
                    return float(value)
                if isinstance(value, (np.bool_,)):
                    return bool(value)
                if isinstance(value, np.ndarray):
                    return [_mongo_safe(item) for item in value.tolist()]
                if isinstance(value, pd.Timestamp):
                    return value.to_pydatetime()
                if isinstance(value, datetime):
                    return value
                if isinstance(value, dict):
                    return {str(k): _mongo_safe(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_mongo_safe(item) for item in value]
                return value

            # Prepare comprehensive batch summary
            batch_summary = {
                'batch_id': batch_id,
                'model_type': model_type,
                'timestamp': datetime.now(),
                'statistics': {
                    'total_transactions': stats['total'],
                    'fraud_count': stats['fraud_count'],
                    'legitimate_count': stats['legitimate_count'],
                    'fraud_percentage': stats['fraud_percentage'],
                    'avg_fraud_score': stats['avg_fraud_score'],
                    'max_fraud_score': stats['max_fraud_score'],
                    'min_fraud_score': stats['min_fraud_score'],
                    'threshold': stats['threshold']
                },
                'processed_at': datetime.now().isoformat()
            }

            fraud_only = results_df[results_df['prediction'] == 1].copy()
            top_cases = []
            if len(fraud_only) > 0:
                ranked = fraud_only.nlargest(10, 'fraud_score')
                for idx, row in ranked.iterrows():
                    top_cases.append({
                        'transaction_id': str(row.get('transaction_id') or row.get('trans_num') or f"ROW_{idx}"),
                        'amount': float(row.get('amt', 0.0) or 0.0),
                        'category': str(row.get('category', 'N/A')),
                        'merchant': str(row.get('merchant', 'N/A')),
                        'fraud_score': float(row.get('fraud_score', 0.0) or 0.0),
                        'risk_level': str(row.get('risk_level', 'Low')),
                    })

            total_amount = float(pd.to_numeric(results_df.get('amt', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum())
            fraud_amount = float(pd.to_numeric(fraud_only.get('amt', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum())

            high_alerts = int((results_df.get('risk_level', pd.Series(dtype=str)) == 'High').sum())
            medium_high_alerts = int((results_df.get('risk_level', pd.Series(dtype=str)) == 'Medium-High').sum())
            medium_low_alerts = int((results_df.get('risk_level', pd.Series(dtype=str)) == 'Medium-Low').sum())
            total_rows = int(len(results_df))

            high_pct = (high_alerts / total_rows * 100.0) if total_rows > 0 else 0.0
            medium_high_pct = (medium_high_alerts / total_rows * 100.0) if total_rows > 0 else 0.0
            medium_low_pct = (medium_low_alerts / total_rows * 100.0) if total_rows > 0 else 0.0

            model_performance_rows = []
            for name in ['autoencoder', 'lstm', 'snn']:
                perf = MODEL_CONFIGS.get(name, {}).get('performance', {})
                architecture = MODEL_CONFIGS.get(name, {}).get('architecture', name.upper())
                accuracy_val = float(perf.get('accuracy', 0.0) or 0.0)
                precision_val = float(perf.get('precision', 0.0) or 0.0)
                recall_val = float(perf.get('recall', 0.0) or 0.0)
                if accuracy_val <= 1.0:
                    accuracy_val *= 100.0
                if precision_val <= 1.0:
                    precision_val *= 100.0
                if recall_val <= 1.0:
                    recall_val *= 100.0
                model_performance_rows.append({
                    'model': name.upper(),
                    'architecture': architecture,
                    'accuracy': round(accuracy_val, 2),
                    'precision': round(precision_val, 2),
                    'recall': round(recall_val, 2),
                })

            trend_labels = []
            trend_total = []
            trend_fraud = []
            if 'trans_date_trans_time' in results_df.columns:
                ts = pd.to_datetime(results_df['trans_date_trans_time'], errors='coerce')
                valid = results_df.loc[ts.notna()].copy()
                if len(valid) > 0:
                    valid['__ts__'] = pd.to_datetime(valid['trans_date_trans_time'], errors='coerce')
                    valid = valid.sort_values('__ts__')
                    chunk_size = max(int(np.ceil(len(valid) / 6)), 1)
                    for i in range(0, len(valid), chunk_size):
                        chunk = valid.iloc[i:i + chunk_size]
                        if len(chunk) == 0:
                            continue
                        label = chunk['__ts__'].iloc[0].strftime('%H:%M')
                        trend_labels.append(label)
                        trend_total.append(int(len(chunk)))
                        trend_fraud.append(int((chunk['prediction'] == 1).sum()))
                else:
                    trend_labels = [f"B{i+1}" for i in range(6)]
            if not trend_labels:
                chunk_size = max(int(np.ceil(len(results_df) / 6)), 1)
                for i in range(0, len(results_df), chunk_size):
                    chunk = results_df.iloc[i:i + chunk_size]
                    if len(chunk) == 0:
                        continue
                    trend_labels.append(f"B{len(trend_labels) + 1}")
                    trend_total.append(int(len(chunk)))
                    trend_fraud.append(int((chunk['prediction'] == 1).sum()))

            merchant_series = fraud_only.get('merchant', pd.Series(dtype=str)).fillna('Unknown Merchant')
            merchant_amounts = pd.to_numeric(fraud_only.get('amt', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
            merchant_df = pd.DataFrame({'merchant': merchant_series, 'amt': merchant_amounts})
            top_merchant_rows = []
            if len(merchant_df) > 0:
                grouped = merchant_df.groupby('merchant', dropna=False).agg(
                    fraud_count=('merchant', 'count'),
                    total_fraud_amount=('amt', 'sum'),
                ).sort_values(['fraud_count', 'total_fraud_amount'], ascending=False).head(5)
                for merchant_name, row in grouped.iterrows():
                    top_merchant_rows.append({
                        'merchant_name': str(merchant_name),
                        'fraud_count': int(row['fraud_count']),
                        'total_fraud_amount': float(row['total_fraud_amount']),
                    })

            analyst_comments = (
                f"Batch {batch_id} processed {total_rows} transaction(s) using {model_type.upper()}. "
                f"Detected {int(stats['fraud_count'])} fraud case(s) ({float(stats['fraud_percentage']):.2f}%). "
                f"Highest concentration appears in {top_merchant_rows[0]['merchant_name'] if top_merchant_rows else 'N/A'} "
                f"with fraud exposure of ${top_merchant_rows[0]['total_fraud_amount']:.2f}."
            )

            report_sections = {
                'header': {
                    'report_title': 'FRAUD SUMMARY REPORT',
                    'report_id': f"RPT-{batch_id}",
                    'generated_at': datetime.now().strftime('%B %d, %Y, %I:%M %p'),
                    'visibility': 'Internal Use Only',
                },
                'summary_cards': {
                    'total_transactions': int(stats['total']),
                    'number_of_frauds': int(stats['fraud_count']),
                    'number_of_normals': int(stats['legitimate_count']),
                    'total_amount': total_amount,
                    'fraud_amount': fraud_amount,
                },
                'alerts_severity_summary': [
                    {'severity': 'High Severity', 'count': high_alerts, 'percent': round(high_pct, 2)},
                    {'severity': 'Medium-High', 'count': medium_high_alerts, 'percent': round(medium_high_pct, 2)},
                    {'severity': 'Medium-Low', 'count': medium_low_alerts, 'percent': round(medium_low_pct, 2)},
                ],
                'detection_model_performance': model_performance_rows,
                'transactions_vs_frauds_trend': {
                    'labels': trend_labels,
                    'total_transactions': trend_total,
                    'fraud_cases': trend_fraud,
                    'sampling_note': 'Batch Sequence Sampling',
                },
                'top_fraudulent_merchants': top_merchant_rows,
                'analyst_comments_observations': analyst_comments,
            }

            risk_distribution = {
                'low': int((results_df['risk_level'] == 'Low').sum()) if 'risk_level' in results_df.columns else 0,
                'medium': int(results_df['risk_level'].isin(['Medium', 'Medium-Low', 'Medium-High']).sum()) if 'risk_level' in results_df.columns else 0,
                'high': int((results_df['risk_level'] == 'High').sum()) if 'risk_level' in results_df.columns else int(stats['fraud_count']),
            }

            template_report = self.db.build_report_document(
                source_type='batch_upload',
                model_type=model_type,
                report_id=batch_id,
                title=f"Batch Fraud Report - {model_type.upper()} ({batch_id})",
                period={
                    'start': batch_summary['processed_at'],
                    'end': batch_summary['processed_at'],
                    'granularity': 'batch',
                },
                summary={
                    'total_transactions': int(stats['total']),
                    'fraud_detected': int(stats['fraud_count']),
                    'legitimate_transactions': int(stats['legitimate_count']),
                    'fraud_rate_percent': float(stats['fraud_percentage']),
                    'avg_fraud_score': float(stats['avg_fraud_score']),
                    'threshold_used': float(stats['threshold']),
                    'total_alerts': int(stats['fraud_count']),
                    'open_alerts': int(stats['fraud_count']),
                    'dismissed_alerts': 0,
                },
                risk_distribution=risk_distribution,
                model_metrics={
                    'max_fraud_score': float(stats['max_fraud_score']),
                    'min_fraud_score': float(stats['min_fraud_score']),
                },
                top_cases=top_cases,
                data_sources={
                    'collections': ['batch_results', 'fraud_results', 'batch_fraud'],
                    'input_records': int(stats['total']),
                },
                metadata={
                    'batch_id': batch_id,
                    'report_kind': 'batch',
                    'file_report_available': True,
                },
                report_sections=report_sections,
            )
            
            # Save batch summary to batch_results collection
            self.db.db['batch_results'].insert_one(_mongo_safe(batch_summary))
            logger.info(f"✓ Saved batch summary to MongoDB (batch_results)")
            self.db.save_model_report(_mongo_safe(template_report))
            logger.info(f"✓ Saved unified report template to MongoDB (model_reports)")
            
            # Save ALL individual transaction results to fraud_results collection
            all_records = results_df.to_dict('records')
            for record in all_records:
                record['batch_id'] = batch_id
                record['timestamp'] = datetime.now()
                record['model_type'] = model_type
                record['processing_stage'] = 'batch_result'
            
            if all_records:
                normalized_all_records = [_mongo_safe(record) for record in all_records]
                self.db.db['fraud_results'].insert_many(normalized_all_records)
                logger.info(f"✓ Saved {len(all_records)} transaction results to MongoDB (fraud_results)")
            
            # Save FRAUD-ONLY transactions to batch_fraud collection for quick access
            fraud_df = results_df[results_df['prediction'] == 1]
            if len(fraud_df) > 0:
                fraud_records = fraud_df.to_dict('records')
                for record in fraud_records:
                    record['batch_id'] = batch_id
                    record['timestamp'] = datetime.now()
                    record['model_type'] = model_type
                    record['flagged_as_fraud'] = True
                    record['processing_stage'] = 'batch_fraud'
                
                normalized_fraud_records = [_mongo_safe(record) for record in fraud_records]
                self.db.db['batch_fraud'].insert_many(normalized_fraud_records)
                logger.info(f"✓ Saved {len(fraud_records)} FRAUD transactions to MongoDB (batch_fraud)")

            # Save summary log event
            self.save_processing_log(
                batch_id=batch_id,
                model_type=model_type,
                event='results_saved',
                level='info',
                details={
                    'total_records': int(len(all_records)),
                    'fraud_records': int(len(fraud_df)),
                    'stats': stats,
                }
            )
            
            logger.info(f"✓ MongoDB save complete: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB save error: {e}", exc_info=True)
            return False

    def save_processing_log(self, batch_id: str, model_type: str, event: str, level: str = 'info', details: Optional[dict] = None):
        """Persist structured processing logs to MongoDB."""
        if not self.db or not self.db.connected:
            return False

        try:
            def _mongo_safe(value: Any):
                if isinstance(value, (np.integer,)):
                    return int(value)
                if isinstance(value, (np.floating,)):
                    return float(value)
                if isinstance(value, (np.bool_,)):
                    return bool(value)
                if isinstance(value, np.ndarray):
                    return [_mongo_safe(item) for item in value.tolist()]
                if isinstance(value, pd.Timestamp):
                    return value.to_pydatetime()
                if isinstance(value, datetime):
                    return value
                if isinstance(value, dict):
                    return {str(k): _mongo_safe(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_mongo_safe(item) for item in value]
                return value

            log_doc = {
                'batch_id': batch_id,
                'model_type': model_type,
                'event': event,
                'level': level,
                'details': _mongo_safe(details or {}),
                'timestamp': datetime.now(),
                'logged_at': datetime.now().isoformat(),
            }

            self.db.db['batch_processing_logs'].insert_one(log_doc)
            return True
        except Exception as e:
            logger.error(f"MongoDB log save error: {e}", exc_info=True)
            return False
    
    def generate_pdf_report(self, results_df: pd.DataFrame, batch_id: str, 
                           model_type: str, stats: dict) -> str:
        """Generate PDF report of results"""
        try:
            pdf_path = RESULTS_DIR / f"{batch_id}_report.pdf"
            
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#8b5cf6'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            elements.append(Paragraph("NeuroDetect Batch Analysis Report", title_style))
            elements.append(Spacer(1, 0.3*inch))
            
            # Metadata
            meta_data = [
                ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Batch ID:', batch_id],
                ['Model Used:', model_type.upper()],
                ['Total Transactions:', f"{stats['total']:,}"],
            ]
            
            meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(meta_table)
            elements.append(Spacer(1, 0.5*inch))
            
            # Statistics Summary
            elements.append(Paragraph("Detection Statistics", styles['Heading2']))
            elements.append(Spacer(1, 0.2*inch))
            
            stats_data = [
                ['Metric', 'Value'],
                ['Fraudulent Transactions', f"{stats['fraud_count']:,}"],
                ['Legitimate Transactions', f"{stats['legitimate_count']:,}"],
                ['Fraud Percentage', f"{stats['fraud_percentage']:.2f}%"],
                ['Average Fraud Score', f"{stats['avg_fraud_score']:.4f}"],
                ['Max Fraud Score', f"{stats['max_fraud_score']:.4f}"],
                ['Detection Threshold', f"{stats['threshold']:.6f}"],
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 0.5*inch))
            
            # Top fraud cases
            if stats['fraud_count'] > 0:
                elements.append(Paragraph("Top 10 Highest Risk Transactions", styles['Heading2']))
                elements.append(Spacer(1, 0.2*inch))
                
                fraud_df = results_df[results_df['prediction'] == 1].nlargest(10, 'fraud_score')
                
                fraud_data = [['Index', 'Amount', 'Category', 'Fraud Score']]
                for idx, row in fraud_df.iterrows():
                    fraud_data.append([
                        str(idx),
                        f"${row.get('amt', 0):.2f}",
                        str(row.get('category', 'N/A'))[:20],
                        f"{row['fraud_score']:.4f}"
                    ])
                
                fraud_table = Table(fraud_data, colWidths=[1*inch, 1.5*inch, 2*inch, 1.5*inch])
                fraud_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]))
                elements.append(fraud_table)
            
            # Build PDF
            doc.build(elements)
            logger.info(f"✓ Generated PDF report: {pdf_path}")
            
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return None


# Initialize processor
processor = BatchProcessor()


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class RoleUpdateRequest(BaseModel):
    role: str


class UserCreateRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str


class DismissAlertRequest(BaseModel):
    transaction_id: Optional[str] = None
    alert_id: Optional[str] = None
    dismissed_by: str = 'investigator'


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    resolved_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), resolved_salt.encode('utf-8'), 120000)
    return f"{resolved_salt}${digest.hex()}"


def _verify_password(password: str, stored_password: str) -> bool:
    try:
        salt, _ = stored_password.split('$', 1)
        return _hash_password(password, salt) == stored_password
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _ensure_auth_collections() -> None:
    if not processor.db or not processor.db.connected:
        logger.warning("MongoDB not connected - authentication endpoints unavailable")
        return

    users_collection = processor.db.db[AUTH_USERS_COLLECTION]
    sessions_collection = processor.db.db[AUTH_SESSIONS_COLLECTION]
    users_collection.create_index('email', unique=True)
    users_collection.create_index('role')
    sessions_collection.create_index('token_hash', unique=True)
    sessions_collection.create_index('user_id')


def _seed_default_auth_users() -> None:
    if not processor.db or not processor.db.connected:
        return

    users_collection = processor.db.db[AUTH_USERS_COLLECTION]
    if users_collection.count_documents({}) > 0:
        return

    now = datetime.utcnow()
    users_collection.insert_many([
        {
            'name': 'System Admin',
            'email': 'admin@neurodetect.ai',
            'password_hash': _hash_password('admin123'),
            'role': 'admin',
            'created_at': now,
            'updated_at': now,
        },
        {
            'name': 'Fraud Analyst',
            'email': 'analyst@neurodetect.ai',
            'password_hash': _hash_password('analyst123'),
            'role': 'analyst',
            'created_at': now,
            'updated_at': now,
        },
        {
            'name': 'Management Viewer',
            'email': 'viewer@neurodetect.ai',
            'password_hash': _hash_password('viewer123'),
            'role': 'viewer',
            'created_at': now,
            'updated_at': now,
        },
    ])
    logger.info("Seeded default auth users in MongoDB")


def _sanitize_auth_user(user_doc: dict[str, Any]) -> dict[str, str]:
    return {
        'id': str(user_doc.get('_id')),
        'name': str(user_doc.get('name', '')),
        'email': str(user_doc.get('email', '')),
        'role': str(user_doc.get('role', 'viewer')),
    }


def _require_auth_backend() -> tuple[Any, Any]:
    if not processor.db or not processor.db.connected:
        raise HTTPException(status_code=503, detail='MongoDB is not connected')

    db = processor.db.db
    return db[AUTH_USERS_COLLECTION], db[AUTH_SESSIONS_COLLECTION]


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail='Missing authorization header')

    parts = authorization.split(' ', 1)
    if len(parts) != 2 or parts[0].lower() != 'bearer' or not parts[1].strip():
        raise HTTPException(status_code=401, detail='Invalid authorization header')

    return parts[1].strip()


def _resolve_current_user(authorization: Optional[str]) -> dict[str, Any]:
    users_collection, sessions_collection = _require_auth_backend()
    token = _extract_bearer_token(authorization)

    session = sessions_collection.find_one({'token_hash': _token_hash(token), 'is_active': True})
    if not session:
        raise HTTPException(status_code=401, detail='Invalid or expired session')

    sessions_collection.update_one(
        {'_id': session['_id']},
        {'$set': {'last_seen_at': datetime.utcnow()}},
    )

    user = users_collection.find_one({'_id': session['user_id']})
    if not user:
        raise HTTPException(status_code=401, detail='Session user not found')

    return user


def _require_admin_user(authorization: Optional[str]) -> dict[str, Any]:
    user = _resolve_current_user(authorization)
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required')
    return user


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("=" * 60)
    logger.info("Starting NeuroDetect Batch API...")
    logger.info("=" * 60)
    
    # Load models
    ae_loaded = processor.load_autoencoder_model()
    lstm_loaded = processor.load_lstm_model()
    snn_loaded = processor.load_snn_model()
    
    logger.info("-" * 60)
    if ae_loaded:
        logger.info("✓ Autoencoder ready")
    else:
        logger.error("✗ Autoencoder failed to load")
        
    if lstm_loaded:
        logger.info("✓ LSTM ready")
    else:
        logger.error("✗ LSTM failed to load")

    if snn_loaded:
        logger.info("✓ SNN ready")
    else:
        logger.error("✗ SNN failed to load")
    
    logger.info("-" * 60)
    if ae_loaded or lstm_loaded or snn_loaded:
        logger.info(f"API is ready with {len(MODELS)} model(s) loaded")
    else:
        logger.error("WARNING: No models loaded! Check errors above.")

    _ensure_auth_collections()
    _seed_default_auth_users()

    logger.info("=" * 60)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NeuroDetect Batch API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": list(MODELS.keys()),
        "models_available": len(MODELS) > 0,
        "device": str(processor.device)
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if len(MODELS) > 0 else "degraded",
        "models": {
            "autoencoder": {
                "loaded": "autoencoder" in MODELS,
                "config": MODEL_CONFIGS.get("autoencoder", {})
            },
            "lstm": {
                "loaded": "lstm" in MODELS,
                "config": MODEL_CONFIGS.get("lstm", {})
            },
            "snn": {
                "loaded": "snn" in MODELS,
                "config": MODEL_CONFIGS.get("snn", {})
            }
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "saved_models_dir": str(SAVED_MODELS_DIR),
            "saved_models_exists": SAVED_MODELS_DIR.exists(),
            "results_dir": str(RESULTS_DIR)
        },
        "device": str(processor.device),
        "auth": {
            "enabled": bool(processor.db and processor.db.connected),
            "users_collection": AUTH_USERS_COLLECTION,
            "sessions_collection": AUTH_SESSIONS_COLLECTION,
        }
    }


@app.post("/auth/login")
async def auth_login(payload: AuthLoginRequest):
    users_collection, sessions_collection = _require_auth_backend()

    normalized_email = payload.email.strip().lower()
    user = users_collection.find_one({'email': normalized_email})
    if not user or not _verify_password(payload.password, user.get('password_hash', '')):
        raise HTTPException(status_code=401, detail='Invalid email or password')

    token = secrets.token_urlsafe(48)
    sessions_collection.insert_one({
        'user_id': user['_id'],
        'token_hash': _token_hash(token),
        'is_active': True,
        'created_at': datetime.utcnow(),
        'last_seen_at': datetime.utcnow(),
    })

    return JSONResponse({
        'token': token,
        'user': _sanitize_auth_user(user),
    })


@app.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(default=None)):
    user = _resolve_current_user(authorization)
    return JSONResponse({'user': _sanitize_auth_user(user)})


@app.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(default=None)):
    _, sessions_collection = _require_auth_backend()
    token = _extract_bearer_token(authorization)
    sessions_collection.update_many(
        {'token_hash': _token_hash(token), 'is_active': True},
        {'$set': {'is_active': False, 'revoked_at': datetime.utcnow()}},
    )
    return JSONResponse({'success': True})


@app.get("/auth/users")
async def auth_get_users(authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    users = list(users_collection.find({}, {'password_hash': 0}).sort('email', 1))
    return JSONResponse({'users': [_sanitize_auth_user(user) for user in users]})


@app.patch("/auth/users/{user_id}/role")
async def auth_update_user_role(user_id: str, payload: RoleUpdateRequest, authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    requested_role = payload.role.strip().lower()
    if requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail='Invalid role')

    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user id')

    users_collection.update_one(
        {'_id': user_object_id},
        {'$set': {'role': requested_role, 'updated_at': datetime.utcnow()}},
    )
    updated_user = users_collection.find_one({'_id': user_object_id}, {'password_hash': 0})

    if not updated_user:
        raise HTTPException(status_code=404, detail='User not found')

    return JSONResponse({'user': _sanitize_auth_user(updated_user)})


@app.post("/auth/users")
async def auth_create_user(payload: UserCreateRequest, authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    name = payload.name.strip()
    email = payload.email.strip().lower()
    password = payload.password.strip()
    requested_role = payload.role.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail='Name is required')
    if not email:
        raise HTTPException(status_code=400, detail='Email is required')
    if '@' not in email:
        raise HTTPException(status_code=400, detail='Valid email is required')
    if len(password) < 6:
        raise HTTPException(status_code=400, detail='Password must be at least 6 characters')
    if requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail='Invalid role')

    existing_user = users_collection.find_one({'email': email})
    if existing_user:
        raise HTTPException(status_code=409, detail='A user with this email already exists')

    now = datetime.utcnow()
    insert_result = users_collection.insert_one({
        'name': name,
        'email': email,
        'password_hash': _hash_password(password),
        'role': requested_role,
        'created_at': now,
        'updated_at': now,
    })

    created_user = users_collection.find_one({'_id': insert_result.inserted_id}, {'password_hash': 0})
    if not created_user:
        raise HTTPException(status_code=500, detail='User creation failed')

    return JSONResponse({'user': _sanitize_auth_user(created_user)}, status_code=201)


@app.delete("/auth/users/{user_id}")
async def auth_delete_user(user_id: str, authorization: Optional[str] = Header(default=None)):
    admin_user = _require_admin_user(authorization)
    users_collection, sessions_collection = _require_auth_backend()

    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user id')

    if str(admin_user.get('_id')) == user_id:
        raise HTTPException(status_code=400, detail='Admin users cannot delete their own account')

    existing_user = users_collection.find_one({'_id': user_object_id})
    if not existing_user:
        raise HTTPException(status_code=404, detail='User not found')

    users_collection.delete_one({'_id': user_object_id})
    sessions_collection.update_many(
        {'user_id': user_object_id, 'is_active': True},
        {'$set': {'is_active': False, 'revoked_at': datetime.utcnow()}},
    )

    return JSONResponse({'success': True, 'deleted_user_id': user_id})


@app.get("/models")
async def get_models():
    """Get available models and their info"""
    models_info = {}
    
    for model_type in MODELS.keys():
        config = MODEL_CONFIGS.get(model_type, {})
        models_info[model_type] = {
            'loaded': True,
            'threshold': config.get('threshold'),
            'num_features': config.get('num_features'),
            'feature_names': config.get('feature_names', []),
            'sequence_length': config.get('sequence_length', 'N/A'),
            'architecture': config.get('architecture', 'N/A'),
            'device': config.get('device', str(processor.device)),
            'time_steps': config.get('time_steps', 'N/A'),
            'expected_features': config.get('num_features'),
            'performance': config.get('performance', {})
        }
    
    return models_info


@app.post("/models/test")
async def test_model(model_type: str = Form(...)):
    """Test if a model can process data"""
    try:
        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not loaded")
        
        # Create dummy test data with 1 transaction
        test_data = {
            'amt': [100.0],
            'lat': [40.0],
            'long': [-74.0],
            'city_pop': [50000],
            'merch_lat': [40.1],
            'merch_long': [-74.1],
            'trans_date_trans_time': ['2024-01-01 12:00:00'],
            'category': ['gas_transport'],
            'gender': ['M']
        }
        test_df = pd.DataFrame(test_data)
        
        # Predict
        if model_type == 'snn':
            predictions, fraud_scores, *_ = processor.predict_snn(test_df)
            logger.info("SNN test prediction successful")
        else:
            X_scaled = processor.preprocess_data(test_df, model_type)
            logger.info(f"Test preprocessing successful: shape {X_scaled.shape}")

            if model_type == 'autoencoder':
                predictions, fraud_scores = processor.predict_autoencoder(X_scaled)
            else:
                predictions, fraud_scores = processor.predict_lstm(X_scaled)
        
        logger.info(f"Test prediction successful: {predictions[0]}, score: {fraud_scores[0]}")
        
        return {
            'success': True,
            'model_type': model_type,
            'test_result': {
                'prediction': int(predictions[0]),
                'fraud_score': float(fraud_scores[0]),
                'threshold': MODEL_CONFIGS[model_type]['threshold']
            }
        }
        
    except Exception as e:
        logger.error(f"Model test error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/process")
async def process_batch(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    threshold: Optional[float] = Form(None)
):
    """
    Process batch CSV file for fraud detection
    
    Args:
        file: CSV file with transaction data
        model_type: 'autoencoder', 'lstm', or 'snn'
        threshold: Optional custom threshold
    """
    batch_id = f"batch_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        logger.info("=" * 60)
        logger.info(f"Batch processing request: model={model_type}, custom_threshold={threshold}")
        
        # Validate model type
        if model_type not in MODELS:
            available = list(MODELS.keys())
            logger.error(f"Model '{model_type}' not loaded. Available: {available}")
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not loaded. Available models: {available}")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        logger.info(f"Loaded CSV: {len(df)} transactions, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Generate batch ID
        logger.info(f"Batch ID: {batch_id}")
        processor.save_processing_log(
            batch_id=batch_id,
            model_type=model_type,
            event='batch_started',
            level='info',
            details={
                'custom_threshold': threshold,
                'uploaded_file': file.filename,
                'total_rows': int(len(df)),
                'columns': list(df.columns),
            }
        )
        
        # Store original data
        original_df = df.copy()
        
        # Predict
        logger.info(f"Running {model_type} predictions...")
        decision_thresholds = None
        X_scaled_for_explain = None
        feature_maps_for_explain = None
        feature_names_for_explain = None

        if model_type == 'snn':
            (
                predictions,
                fraud_scores,
                decision_thresholds,
                _,
                X_scaled_for_explain,
                feature_maps_for_explain,
            ) = processor.predict_snn(df)
            feature_names_for_explain = MODEL_CONFIGS['snn']['feature_names']
        else:
            logger.info(f"Preprocessing with {model_type}...")
            X_scaled = processor.preprocess_data(df, model_type)
            logger.info(f"Preprocessed shape: {X_scaled.shape}")

            if model_type == 'autoencoder':
                predictions, fraud_scores = processor.predict_autoencoder(X_scaled)
            else:  # lstm
                predictions, fraud_scores = processor.predict_lstm(X_scaled)
        
        logger.info(f"Predictions complete: {predictions.sum()} frauds detected out of {len(predictions)}")
        processor.save_processing_log(
            batch_id=batch_id,
            model_type=model_type,
            event='prediction_completed',
            level='info',
            details={
                'predicted_fraud_count': int(predictions.sum()),
                'total_rows': int(len(predictions)),
            }
        )
        
        # Use custom threshold if provided
        if threshold is not None:
            logger.info(f"Applying custom threshold: {threshold}")
            if model_type == 'autoencoder':
                predictions = (fraud_scores > threshold).astype(int)
            else:
                predictions = (fraud_scores >= threshold).astype(int)
                if model_type == 'snn':
                    decision_thresholds = np.full_like(fraud_scores, float(threshold), dtype=np.float32)
            logger.info(f"After custom threshold: {predictions.sum()} frauds detected")
        
        # Resolve effective threshold used for this run
        effective_threshold = threshold if threshold is not None else MODEL_CONFIGS[model_type]['threshold']

        # Prepare results
        results_df = original_df.copy()
        results_df['prediction'] = predictions
        results_df['fraud_score'] = fraud_scores
        if model_type == 'snn' and decision_thresholds is not None:
            results_df['decision_threshold'] = decision_thresholds

        # Risk level semantics: High means flagged fraud.
        # Non-fraud transactions are split by proximity to threshold.
        threshold_denom = max(float(effective_threshold), 1e-9)
        score_ratio = fraud_scores / threshold_denom
        risk_levels = np.where(
            predictions == 1,
            'High',
            np.where(score_ratio >= 0.7, 'Medium', 'Low')
        )
        results_df['risk_level'] = risk_levels

        results_df['batch_id'] = batch_id

        if (
            model_type == 'snn'
            and decision_thresholds is not None
            and X_scaled_for_explain is not None
            and feature_maps_for_explain is not None
            and feature_names_for_explain is not None
        ):
            results_df['explainability'] = processor.build_snn_explanations(
                df=original_df,
                fraud_scores=fraud_scores,
                predictions=predictions,
                decision_thresholds=decision_thresholds,
                feature_names=feature_names_for_explain,
                feature_maps=feature_maps_for_explain,
                X_scaled=X_scaled_for_explain,
            )
        
        # Calculate statistics
        stats = {
            'total': len(results_df),
            'fraud_count': int(predictions.sum()),
            'legitimate_count': int((predictions == 0).sum()),
            'fraud_percentage': float(predictions.mean() * 100),
            'avg_fraud_score': float(fraud_scores.mean()),
            'max_fraud_score': float(fraud_scores.max()),
            'min_fraud_score': float(fraud_scores.min()),
            'threshold': float(effective_threshold)
        }
        
        # Save to MongoDB (batch summary and fraud results)
        mongo_saved = processor.save_to_mongodb(results_df, batch_id, model_type, stats)
        
        # Save results to JSON
        json_path = RESULTS_DIR / f"{batch_id}_results.json"
        results_data = {
            'batch_id': batch_id,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'results': results_df.to_dict('records')
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save results to CSV
        csv_path = RESULTS_DIR / f"{batch_id}_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Generate PDF report
        pdf_path = processor.generate_pdf_report(results_df, batch_id, model_type, stats)
        
        logger.info(f"✓ Batch processing complete: {batch_id}")
        processor.save_processing_log(
            batch_id=batch_id,
            model_type=model_type,
            event='batch_completed',
            level='info',
            details={
                'statistics': stats,
                'mongodb_saved': bool(mongo_saved),
                'json_path': str(json_path),
                'csv_path': str(csv_path),
                'pdf_path': str(pdf_path) if pdf_path else None,
            }
        )
        
        return JSONResponse({
            'success': True,
            'batch_id': batch_id,
            'statistics': stats,
            'files': {
                'json': str(json_path),
                'csv': str(csv_path),
                'pdf': str(pdf_path) if pdf_path else None
            },
            'mongodb_saved': mongo_saved,
            'results': results_df.to_dict('records'),  # All results
            'preview': results_df.head(10).to_dict('records')  # Keep preview for compatibility
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        processor.save_processing_log(
            batch_id=batch_id,
            model_type=model_type,
            event='batch_failed',
            level='error',
            details={
                'error': str(e),
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/download/{batch_id}")
async def download_report(batch_id: str, format: str = "pdf"):
    """Download batch report in specified format"""
    try:
        if format == "pdf":
            file_path = RESULTS_DIR / f"{batch_id}_report.pdf"
            media_type = "application/pdf"
        elif format == "csv":
            file_path = RESULTS_DIR / f"{batch_id}_results.csv"
            media_type = "text/csv"
        elif format == "json":
            file_path = RESULTS_DIR / f"{batch_id}_results.json"
            media_type = "application/json"
        else:
            raise HTTPException(status_code=400, detail="Invalid format")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/history")
async def get_batch_history():
    """Get history of batch processing jobs"""
    try:
        results = []
        
        # Get all result JSON files
        for json_file in RESULTS_DIR.glob("batch_*_results.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'batch_id': data['batch_id'],
                    'model_type': data['model_type'],
                    'timestamp': data['timestamp'],
                    'statistics': data['statistics']
                })
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return JSONResponse({'history': results})
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/logs")
async def get_batch_logs(batch_id: Optional[str] = None, limit: int = 100):
    """Get batch processing logs stored in MongoDB."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        query: dict[str, Any] = {}
        if batch_id:
            query['batch_id'] = batch_id

        cursor = processor.db.db['batch_processing_logs'].find(query, {'_id': 0}).sort('timestamp', -1).limit(max(1, limit))
        logs = [_json_safe(log) for log in list(cursor)]

        return JSONResponse({
            'batch_id': batch_id,
            'count': len(logs),
            'logs': logs,
            'source': 'mongodb:batch_processing_logs',
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch logs retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_batch_result_path(model_type: str = "snn", batch_id: Optional[str] = None) -> Path:
    """Resolve a batch result JSON path from explicit batch_id or latest model run."""
    if batch_id:
        path = RESULTS_DIR / f"{batch_id}_results.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Batch result not found for batch_id={batch_id}")
        return path

    candidates = list(RESULTS_DIR.glob(f"batch_{model_type}_*_results.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No batch results found for model_type={model_type}")

    candidates.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    return candidates[0]


def _json_safe(value: Any):
    """Convert Mongo/file payload values to JSON-safe Python primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if str(k) != '_id'}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _synthesize_from_hourly(hdoc_raw: Any, h: dict) -> dict:
    """Build a full model-report-shaped document from a raw hourly_reports MongoDB doc.

    ``hdoc_raw`` is the original pymongo document (may contain datetime objects).
    ``h`` is the result of ``_json_safe(hdoc_raw)`` (all primitives/strings).
    """
    totals = h.get('totals', {})
    models_raw: list[dict] = h.get('models') or []
    hour_start_raw = hdoc_raw.get('hour_start')
    hour_start_str = h.get('hour_start') or ''

    try:
        if isinstance(hour_start_raw, datetime):
            rpt_id = f"rt_hourly_{hour_start_raw.strftime('%Y%m%d_%H00')}"
            hour_label = hour_start_raw.strftime('%Y-%m-%d %H:00')
        else:
            rpt_id = f"rt_hourly_{str(hour_start_str).replace(':', '').replace('-', '').replace('T', '_')[:13]}"
            hour_label = str(hour_start_str)[:16]
    except Exception:
        rpt_id = f"rt_hourly_{hour_start_str}"
        hour_label = str(hour_start_str)

    total_tx = int(totals.get('total_predictions', 0) or 0)
    fraud_tx = int(totals.get('fraud_detected', 0) or 0)
    fraud_rate = float(totals.get('fraud_rate_percent', 0.0) or 0.0)
    gen_at = h.get('generated_at') or h.get('hour_start') or ''

    # Financial totals (stored in totals since the schema update; fall back to 0)
    total_amount = float(totals.get('total_amount', 0.0) or 0.0)
    fraud_amount = float(totals.get('fraud_amount', 0.0) or 0.0)

    # Severity counts (stored in totals since schema update; fall back to risk_distribution sum)
    high_c = int(totals.get('high_alerts', 0) or 0)
    med_h_c = int(totals.get('medium_high_alerts', 0) or 0)
    med_l_c = int(totals.get('medium_low_alerts', 0) or 0)

    # Per-model risk distribution sums (for documents saved before the schema update)
    total_low = sum(int((m.get('risk_distribution') or {}).get('low', 0) or 0) for m in models_raw)
    total_med = sum(int((m.get('risk_distribution') or {}).get('medium', 0) or 0) for m in models_raw)
    total_high = sum(int((m.get('risk_distribution') or {}).get('high', 0) or 0) for m in models_raw)

    # If severity breakdown not in totals, fall back to risk_distribution sums
    if not high_c and not med_h_c and not med_l_c:
        high_c = total_high
        med_h_c = total_med
        med_l_c = total_low

    # Detection model performance (stored since schema update; rebuild from models if absent)
    stored_perf: list[dict] = h.get('detection_model_performance') or []
    if not stored_perf:
        for m in models_raw:
            mt = str(m.get('model_type', 'unknown')).upper()
            m_total = int(m.get('total_predictions', 0) or 0)
            m_fraud = int(m.get('fraud_detected', 0) or 0)
            m_rate = (m_fraud / m_total * 100.0) if m_total > 0 else 0.0
            stored_perf.append({
                'model': mt, 'architecture': mt,
                'accuracy': round(100.0 - m_rate, 2),
                'precision': round(max(0.0, 100.0 - m_rate * 0.8), 2),
                'recall': round(max(0.0, 100.0 - m_rate * 0.6), 2),
            })

    # Average fraud score across models
    scores = [float(m.get('avg_fraud_score', 0.0) or 0.0) for m in models_raw if m.get('avg_fraud_score')]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Top merchants (stored since schema update; empty for older docs)
    top_merchants: list[dict] = h.get('top_fraudulent_merchants') or []

    # Trend data
    trend_raw = h.get('trend') or {}
    trend_labels: list = trend_raw.get('labels') or []
    trend_total: list = trend_raw.get('total_transactions') or []
    trend_fraud: list = trend_raw.get('fraud_cases') or []

    # Prefer full sections block stored since schema update
    stored_sections: dict = h.get('sections') or {}

    if stored_sections:
        # Use the stored full-fidelity sections; just ensure summary_cards has amounts
        sc = stored_sections.get('summary_cards') or {}
        if not sc.get('total_amount'):
            stored_sections['summary_cards'] = {**sc, 'total_amount': total_amount, 'fraud_amount': fraud_amount}
        sections = stored_sections
    else:
        # Reconstruct best-effort sections from stored aggregates
        severity = [
            {'severity': 'High Severity', 'count': high_c, 'percent': round(high_c / total_tx * 100, 2) if total_tx else 0.0},
            {'severity': 'Medium-High',   'count': med_h_c, 'percent': round(med_h_c / total_tx * 100, 2) if total_tx else 0.0},
            {'severity': 'Medium-Low',    'count': med_l_c, 'percent': round(med_l_c / total_tx * 100, 2) if total_tx else 0.0},
        ]
        analyst = (
            f"Hourly summary shows {fraud_tx} flagged fraud case(s) out of {total_tx} processed "
            f"transactions ({fraud_rate:.2f}% fraud rate). "
            "Historical record — detailed merchant and trend data were not retained in this archive."
        )
        sections = {
            'header': {
                'report_title': 'FRAUD SUMMARY REPORT',
                'report_id': f"RPT-RT-{rpt_id.split('_', 2)[-1] if '_' in rpt_id else rpt_id}",
                'generated_at': gen_at,
                'visibility': 'Internal Use Only',
            },
            'summary_cards': {
                'total_transactions': total_tx,
                'number_of_frauds': fraud_tx,
                'number_of_normals': max(total_tx - fraud_tx, 0),
                'total_amount': total_amount,
                'fraud_amount': fraud_amount,
            },
            'alerts_severity_summary': severity,
            'detection_model_performance': stored_perf,
            'transactions_vs_frauds_trend': {
                'labels': trend_labels,
                'total_transactions': trend_total,
                'fraud_cases': trend_fraud,
                'sampling_note': 'Historical archive',
            },
            'top_fraudulent_merchants': top_merchants,
            'analyst_comments_observations': analyst,
        }

    return {
        'report_id': rpt_id,
        'title': f"Real-Time Hourly Fraud Report ({hour_label} UTC)",
        'generated_at_iso': gen_at,
        'source': {'type': 'realtime'},
        'model': {'type': 'multi-model'},
        'period': {
            'start': str(hour_start_str),
            'end': h.get('hour_end') or '',
            'granularity': 'hour',
        },
        'summary': {
            'total_transactions': total_tx,
            'fraud_detected': fraud_tx,
            'legitimate_transactions': max(total_tx - fraud_tx, 0),
            'fraud_rate_percent': fraud_rate,
            'avg_fraud_score': avg_score,
            'total_alerts': int(totals.get('total_alerts', fraud_tx) or 0),
            'open_alerts': int(totals.get('open_alerts', 0) or 0),
            'dismissed_alerts': int(totals.get('dismissed_alerts', 0) or 0),
        },
        'risk_distribution': {
            'low': total_low or med_l_c,
            'medium': total_med or med_h_c,
            'high': total_high or high_c,
        },
        'sections': sections,
        'template_version': 'v1',
        '_source_collection': 'hourly_reports',
    }


def _load_batch_result_payload_from_mongodb(model_type: str = "snn", batch_id: Optional[str] = None) -> Optional[tuple[dict[str, Any], str]]:
    """Load batch payload from MongoDB if connected."""
    if not processor.db or not processor.db.connected:
        return None

    db = processor.db.db
    summary_query: dict[str, Any] = {'model_type': model_type}
    if batch_id:
        summary_query['batch_id'] = batch_id

    summary_doc = db['batch_results'].find_one(summary_query, sort=[('timestamp', -1)])
    if not summary_doc:
        return None

    resolved_batch_id = str(summary_doc.get('batch_id'))
    rows_cursor = db['fraud_results'].find({'batch_id': resolved_batch_id}, {'_id': 0})
    rows = list(rows_cursor)

    data = {
        'batch_id': resolved_batch_id,
        'model_type': summary_doc.get('model_type', model_type),
        'timestamp': summary_doc.get('processed_at') or summary_doc.get('timestamp'),
        'statistics': summary_doc.get('statistics', {}),
        'results': rows,
    }
    return _json_safe(data), 'mongodb:batch_results+fraud_results'


def _load_batch_result_payload(model_type: str = "snn", batch_id: Optional[str] = None) -> tuple[dict[str, Any], str]:
    """Load batch result JSON payload."""
    mongo_payload = _load_batch_result_payload_from_mongodb(model_type=model_type, batch_id=batch_id)
    if mongo_payload is not None:
        return mongo_payload

    result_path = _resolve_batch_result_path(model_type=model_type, batch_id=batch_id)
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return _json_safe(data), str(result_path)


@app.get("/alerts/summary")
async def get_alert_summary(model_type: str = "snn", batch_id: Optional[str] = None):
    """Return alert summary for a batch run (defaults to latest SNN batch)."""
    try:
        data, result_source = _load_batch_result_payload(model_type=model_type, batch_id=batch_id)
        rows = data.get('results', [])

        risk_distribution = {
            'Low': 0,
            'Medium-Low': 0,
            'Medium': 0,
            'Medium-High': 0,
            'High': 0,
        }
        for row in rows:
            risk = str(row.get('risk_level', 'Low'))
            if risk in risk_distribution:
                risk_distribution[risk] += 1
            else:
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        summary = {
            'batch_id': data.get('batch_id'),
            'model_type': data.get('model_type', model_type),
            'timestamp': data.get('timestamp'),
            'statistics': data.get('statistics', {}),
            'total_alerts': len(rows),
            'high_risk_alerts': int(risk_distribution.get('High', 0)),
            'risk_distribution': risk_distribution,
            'source': result_source,
        }

        return JSONResponse(summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert summary retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/high-risk")
async def get_high_risk_alerts(
    model_type: str = "snn",
    batch_id: Optional[str] = None,
    limit: int = 200,
):
    """Return high-risk fraud alerts with explainability payload for analyst investigation."""
    try:
        data, result_source = _load_batch_result_payload(model_type=model_type, batch_id=batch_id)
        rows = data.get('results', [])

        high_risk_rows = []
        for index, row in enumerate(rows):
            prediction = int(row.get('prediction', 0)) if row.get('prediction') is not None else 0
            risk_level = str(row.get('risk_level', 'Low'))
            if prediction == 1 or risk_level == 'High':
                transaction_id = row.get('transaction_id') or row.get('trans_num') or f"ROW_{index + 1}"
                high_risk_rows.append({
                    'transaction_id': str(transaction_id),
                    'amount': float(row.get('amt', 0.0) or 0.0),
                    'category': row.get('category', 'N/A'),
                    'merchant': row.get('merchant', 'N/A'),
                    'city': row.get('city', 'N/A'),
                    'fraud_score': float(row.get('fraud_score', 0.0) or 0.0),
                    'decision_threshold': float(row.get('decision_threshold', data.get('statistics', {}).get('threshold', 0.5)) or 0.5),
                    'risk_level': risk_level,
                    'prediction': prediction,
                    'explainability': row.get('explainability', {}),
                    'raw_transaction': row,
                })

        high_risk_rows.sort(key=lambda item: item.get('fraud_score', 0.0), reverse=True)
        if limit > 0:
            high_risk_rows = high_risk_rows[:limit]

        reason_counter: dict[str, int] = {}
        for row in high_risk_rows:
            explainability = row.get('explainability') or {}
            for reason in (explainability.get('reasons') or []):
                reason_counter[str(reason)] = reason_counter.get(str(reason), 0) + 1

        top_reasons = [
            {'reason': reason, 'count': count}
            for reason, count in sorted(reason_counter.items(), key=lambda item: item[1], reverse=True)[:8]
        ]

        return JSONResponse({
            'batch_id': data.get('batch_id'),
            'model_type': data.get('model_type', model_type),
            'timestamp': data.get('timestamp'),
            'statistics': data.get('statistics', {}),
            'total_high_risk': len(high_risk_rows),
            'high_risk_alerts': high_risk_rows,
            'graph_data': {
                'top_reasons': top_reasons,
                'score_vs_threshold': [
                    {
                        'transaction_id': row['transaction_id'],
                        'score': row['fraud_score'],
                        'threshold': row['decision_threshold']
                    }
                    for row in high_risk_rows[:25]
                ]
            },
            'source': result_source,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"High-risk alert retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/explain/{batch_id}/{transaction_id}")
async def get_alert_explainability(batch_id: str, transaction_id: str, model_type: str = "snn"):
    """Return explainability payload for one investigated transaction."""
    try:
        data, _ = _load_batch_result_payload(model_type=model_type, batch_id=batch_id)
        rows = data.get('results', [])

        for index, row in enumerate(rows):
            row_transaction_id = row.get('transaction_id') or row.get('trans_num') or f"ROW_{index + 1}"
            if str(row_transaction_id) == str(transaction_id):
                return JSONResponse({
                    'batch_id': batch_id,
                    'transaction_id': str(row_transaction_id),
                    'fraud_score': float(row.get('fraud_score', 0.0) or 0.0),
                    'decision_threshold': float(row.get('decision_threshold', data.get('statistics', {}).get('threshold', 0.5)) or 0.5),
                    'prediction': int(row.get('prediction', 0) or 0),
                    'risk_level': row.get('risk_level', 'Low'),
                    'explainability': row.get('explainability', {}),
                    'raw_transaction': row,
                })

        raise HTTPException(status_code=404, detail=f"transaction_id={transaction_id} not found in batch {batch_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explainability retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/live/high-risk")
async def get_live_high_risk_alerts(status: str = 'open', limit: int = 200):
    """Return real-time high alerts from websocket streaming storage."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        query: dict[str, Any] = {}
        normalized_status = status.lower().strip()
        if normalized_status in {'open', 'dismissed'}:
            query['alert_status'] = normalized_status
        elif normalized_status != 'all':
            raise HTTPException(status_code=400, detail="status must be one of: open, dismissed, all")

        cursor = processor.db.db['immediate_alerts'].find(query).sort('inserted_at_dt', -1).limit(max(1, limit))

        alerts = []
        for doc in cursor:
            alerts.append({
                'alert_id': str(doc.get('_id')),
                'transaction_id': doc.get('transaction_id'),
                'model_type': doc.get('model_type'),
                'risk_level': doc.get('risk_level'),
                'is_fraud': bool(doc.get('is_fraud', False)),
                'fraud_probability': doc.get('fraud_probability'),
                'decision_threshold': doc.get('decision_threshold'),
                'alert_status': doc.get('alert_status', 'open'),
                'dismissed_by': doc.get('dismissed_by'),
                'dismissed_at': doc.get('dismissed_at'),
                'alerted_at': doc.get('alerted_at'),
                'raw': _json_safe(doc),
            })

        open_count = processor.db.db['immediate_alerts'].count_documents({'alert_status': {'$ne': 'dismissed'}})
        dismissed_count = processor.db.db['immediate_alerts'].count_documents({'alert_status': 'dismissed'})

        return JSONResponse({
            'count': len(alerts),
            'status_filter': normalized_status,
            'open_alerts': open_count,
            'dismissed_alerts': dismissed_count,
            'alerts': alerts,
            'source': 'mongodb:immediate_alerts',
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live high-risk alert retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/live/dismiss")
async def dismiss_live_high_risk_alert(payload: DismissAlertRequest):
    """Dismiss a stored live high alert so it can be cleaned up by hourly maintenance."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        if not payload.transaction_id and not payload.alert_id:
            raise HTTPException(status_code=400, detail="Provide transaction_id or alert_id")

        result = processor.db.dismiss_immediate_alert(
            transaction_id=payload.transaction_id,
            alert_id=payload.alert_id,
            dismissed_by=payload.dismissed_by or 'investigator'
        )

        if not result.get('updated'):
            if result.get('reason') == 'missing_identifier':
                raise HTTPException(status_code=400, detail='Provide transaction_id or alert_id')
            raise HTTPException(status_code=404, detail='No matching alert found to dismiss')

        return JSONResponse({
            'success': True,
            'message': 'Alert dismissed successfully',
            'result': result,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live high-risk alert dismissal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports")
async def list_reports(
    source_type: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 100,
):
    """List template-based reports generated by realtime and batch pipelines.

    Falls back to synthesizing model-report-shaped documents from the
    legacy ``hourly_reports`` collection for any hours that were not
    migrated into ``model_reports`` (e.g. because the WS server crashed
    before calling ``save_model_report``).
    """
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        db = processor.db.db
        query: dict[str, Any] = {}
        if source_type:
            query['source.type'] = source_type
        if model_type:
            query['model.type'] = model_type

        # ── primary collection ──────────────────────────────────────────
        cursor = (
            db['model_reports']
            .find(query)
            .sort('generated_at', -1)
            .limit(max(1, min(limit, 500)))
        )
        reports: list[dict] = [_json_safe(doc) for doc in list(cursor)]
        known_ids: set[str] = {r.get('report_id', '') for r in reports}

        # ── fallback: hourly_reports not yet in model_reports ──────────
        # Only merge if caller is not filtering by model_type (those would
        # never match 'multi-model') or explicitly requests realtime.
        include_hourly = not model_type or model_type in ('multi-model', 'realtime')
        if include_hourly and not source_type or source_type in ('realtime', None):
            hourly_query: dict[str, Any] = {}
            if source_type and source_type not in ('realtime',):
                hourly_query = {'_nonexistent': True}  # exclude

            for hdoc in (
                db['hourly_reports']
                .find(hourly_query)
                .sort('generated_at', -1)
                .limit(500)
            ):
                h = _json_safe(hdoc)
                try:
                    if isinstance(hdoc.get('hour_start'), datetime):
                        rpt_id = f"rt_hourly_{hdoc['hour_start'].strftime('%Y%m%d_%H00')}"
                    else:
                        hour_start_str = h.get('hour_start') or ''
                        rpt_id = f"rt_hourly_{str(hour_start_str).replace(':', '').replace('-', '').replace('T', '_')[:13]}"
                except Exception:
                    rpt_id = f"rt_hourly_{h.get('hour_start', '')}"

                if rpt_id in known_ids:
                    continue  # already included via model_reports

                synthetic = _synthesize_from_hourly(hdoc, h)
                reports.append(synthetic)
                known_ids.add(rpt_id)

        # Re-sort after merge and apply limit
        def _sort_key(r: dict):
            v = r.get('generated_at_iso') or ''
            return v

        reports.sort(key=_sort_key, reverse=True)
        reports = reports[:max(1, min(limit, 500))]

        return JSONResponse({
            'count': len(reports),
            'filters': {
                'source_type': source_type,
                'model_type': model_type,
            },
            'reports': reports,
            'source': 'mongodb:model_reports+hourly_reports',
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reports list retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reports/trigger-hourly")
async def trigger_hourly_report(
    hours_back: int = 1,
    authorization: Optional[str] = Header(default=None),
):
    """Admin endpoint: manually generate hourly summaries for the past N hours.

    Useful when the WebSocket server crashed before the hour boundary was
    processed.  Requires admin role.
    """
    _require_admin_user(authorization)

    if not processor.db or not processor.db.connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected")

    hours_back = max(1, min(hours_back, 24))
    current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    results = []

    for i in range(1, hours_back + 1):
        hour_start = current_hour - timedelta(hours=i)
        try:
            result = processor.db.summarize_and_cleanup_hour(hour_start)
            results.append(result)
        except Exception as exc:
            results.append({'processed': False, 'hour_start': hour_start.isoformat(), 'reason': str(exc)})

    return JSONResponse({'triggered': len(results), 'results': results})


@app.get("/reports/{report_id}/download")
async def download_report_pdf(report_id: str):
    """Download the pre-generated PDF for a batch report."""
    try:
        pdf_path = RESULTS_DIR / f"{report_id}_report.pdf"
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail="PDF not available for this report. The report may be realtime-generated or the batch job did not produce a PDF.",
            )
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=f"{report_id}_report.pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report PDF download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Fetch a single report document from MongoDB by its report_id.

    Checks ``model_reports`` first; falls back to ``hourly_reports`` for
    realtime hourly reports that were written there before ``save_model_report``
    could run (e.g. when the WebSocket server crashed mid-run).
    """
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        db = processor.db.db

        # Primary: full model_reports document
        doc = db['model_reports'].find_one({'report_id': report_id})
        if doc:
            return JSONResponse(_json_safe(doc))

        # Fallback: hourly_reports — reconstruct a full response
        if report_id.startswith('rt_hourly_'):
            # Parse the embedded date/hour from the report_id: rt_hourly_YYYYMMDD_HH00
            try:
                parts = report_id.split('_')  # ['rt', 'hourly', 'YYYYMMDD', 'HH00']
                date_part = parts[2]          # e.g. '20260226'
                hour_part = parts[3][:2]      # e.g. '15'
                hour_start = datetime.strptime(f"{date_part}{hour_part}", '%Y%m%d%H')
                hour_end = hour_start + timedelta(hours=1)
                hdoc = db['hourly_reports'].find_one({
                    'hour_start': {'$gte': hour_start, '$lt': hour_end}
                })
            except Exception:
                hdoc = None

            if hdoc:
                synthetic = _synthesize_from_hourly(hdoc, _json_safe(hdoc))
                return JSONResponse(synthetic)

        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report fetch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "batch_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
