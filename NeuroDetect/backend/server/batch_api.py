"""
FastAPI Server for Batch Fraud Detection
Supports CSV upload, model selection, fraud prediction, and PDF report generation
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import torch
import joblib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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


class BatchProcessor:
    """Handles batch fraud detection processing"""
    
    def __init__(self):
        self.db = get_mongodb_instance()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
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
                'num_features': num_features
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
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                feature_names = config['feature_names']
                threshold = config['performance']['optimal_threshold']
                sequence_length = config['input_shape'][0]
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
                'sequence_length': sequence_length
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

    def _build_snn_feature_vector(self, transaction_data: dict, feature_names: list[str]):
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
        return cc_num, vector

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

            for record in records:
                cc_num, vector = self._build_snn_feature_vector(record, feature_names)
                cc_nums.append(cc_num)
                vectors.append(vector)

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
            return predictions, fraud_scores

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
            
            # Save batch summary to batch_results collection
            self.db.db['batch_results'].insert_one(batch_summary)
            logger.info(f"✓ Saved batch summary to MongoDB (batch_results)")
            
            # Save ALL individual transaction results to fraud_results collection
            all_records = results_df.to_dict('records')
            for record in all_records:
                record['batch_id'] = batch_id
                record['timestamp'] = datetime.now()
                record['model_type'] = model_type
                # Convert numpy types to Python types
                for key in ['prediction', 'fraud_score']:
                    if key in record:
                        record[key] = float(record[key])
            
            if all_records:
                self.db.db['fraud_results'].insert_many(all_records)
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
                    # Convert numpy types
                    for key in ['prediction', 'fraud_score']:
                        if key in record:
                            record[key] = float(record[key])
                
                self.db.db['batch_fraud'].insert_many(fraud_records)
                logger.info(f"✓ Saved {len(fraud_records)} FRAUD transactions to MongoDB (batch_fraud)")
            
            logger.info(f"✓ MongoDB save complete: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB save error: {e}", exc_info=True)
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
        "device": str(processor.device)
    }


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
            predictions, fraud_scores = processor.predict_snn(test_df)
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
        batch_id = f"batch_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Batch ID: {batch_id}")
        
        # Store original data
        original_df = df.copy()
        
        # Predict
        logger.info(f"Running {model_type} predictions...")
        if model_type == 'snn':
            predictions, fraud_scores = processor.predict_snn(df)
        else:
            logger.info(f"Preprocessing with {model_type}...")
            X_scaled = processor.preprocess_data(df, model_type)
            logger.info(f"Preprocessed shape: {X_scaled.shape}")

            if model_type == 'autoencoder':
                predictions, fraud_scores = processor.predict_autoencoder(X_scaled)
            else:  # lstm
                predictions, fraud_scores = processor.predict_lstm(X_scaled)
        
        logger.info(f"Predictions complete: {predictions.sum()} frauds detected out of {len(predictions)}")
        
        # Use custom threshold if provided
        if threshold is not None:
            logger.info(f"Applying custom threshold: {threshold}")
            if model_type == 'autoencoder':
                predictions = (fraud_scores > threshold).astype(int)
            else:
                predictions = (fraud_scores >= threshold).astype(int)
            logger.info(f"After custom threshold: {predictions.sum()} frauds detected")
        
        # Resolve effective threshold used for this run
        effective_threshold = threshold if threshold is not None else MODEL_CONFIGS[model_type]['threshold']

        # Prepare results
        results_df = original_df.copy()
        results_df['prediction'] = predictions
        results_df['fraud_score'] = fraud_scores

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


if __name__ == "__main__":
    uvicorn.run(
        "batch_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
