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
BASE_DIR = Path(__file__).parent.parent.parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results" / "batch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
            
            # Load threshold
            threshold_path = SAVED_MODELS_DIR / "threshold.json"
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data['threshold']
            
            # Load features
            features_path = SAVED_MODELS_DIR / "features.json"
            with open(features_path, 'r') as f:
                features_data = json.load(f)
                feature_names = features_data['feature_names']
                num_features = features_data['num_features']
                top_categories = features_data.get('top_categories', [])
                category_columns = features_data.get('category_columns', [])
            
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
            
            # Load model
            model = FraudAutoencoder(input_dim=num_features)
            model_path = SAVED_MODELS_DIR / "autoencoder.pth"
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            MODELS['autoencoder'] = model
            PREPROCESSORS['autoencoder'] = preprocessor
            MODEL_CONFIGS['autoencoder'] = {
                'threshold': threshold,
                'feature_names': feature_names,
                'num_features': num_features
            }
            
            logger.info(f"✓ Autoencoder loaded - {num_features} features, threshold: {threshold:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Autoencoder: {e}")
            return False
    
    def load_lstm_model(self):
        """Load LSTM model and preprocessor"""
        try:
            logger.info("Loading LSTM model...")
            
            # Load model config
            config_path = SAVED_MODELS_DIR / "enhanced_lstm_fraud_model.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            feature_names = config['feature_names']
            threshold = config['performance']['optimal_threshold']
            
            # Initialize preprocessor
            preprocessor = AEPreprocessor()  # LSTM uses same preprocessing
            preprocessor.top_categories = ['gas_transport', 'grocery_pos', 'home', 'shopping_pos', 
                                          'kids_pets', 'shopping_net', 'entertainment', 'food_dining']
            preprocessor.category_columns = config['feature_names'][15:23]  # Category columns
            
            # Load scaler
            scaler_path = SAVED_MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                import joblib
                preprocessor.scaler = joblib.load(scaler_path)
            
            # Load LSTM model
            model_path = SAVED_MODELS_DIR / "enhanced_lstm_fraud_model.pth"
            model = load_lstm_model(str(model_path))
            model.to(self.device)
            model.eval()
            
            MODELS['lstm'] = model
            PREPROCESSORS['lstm'] = preprocessor
            MODEL_CONFIGS['lstm'] = {
                'threshold': threshold,
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'sequence_length': config['input_shape'][0]
            }
            
            logger.info(f"✓ LSTM loaded - {len(feature_names)} features, threshold: {threshold:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM: {e}")
            return False
    
    def preprocess_data(self, df: pd.DataFrame, model_type: str):
        """Preprocess data for prediction"""
        try:
            preprocessor = PREPROCESSORS[model_type]
            
            # Preprocess
            df_processed = preprocessor.preprocess(df, is_training=False, save_scaler=False)
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def predict_autoencoder(self, df_processed: pd.DataFrame):
        """Run Autoencoder predictions"""
        model = MODELS['autoencoder']
        threshold = MODEL_CONFIGS['autoencoder']['threshold']
        
        # Convert to tensor
        X = torch.FloatTensor(df_processed.values).to(self.device)
        
        # Predict
        with torch.no_grad():
            reconstructed = model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Classify
        predictions = (errors > threshold).astype(int)
        fraud_scores = errors
        
        return predictions, fraud_scores
    
    def predict_lstm(self, df_processed: pd.DataFrame):
        """Run LSTM predictions with sequence handling"""
        model = MODELS['lstm']
        config = MODEL_CONFIGS['lstm']
        threshold = config['threshold']
        sequence_length = config['sequence_length']
        
        X = df_processed.values
        n_samples = len(X)
        
        predictions = []
        fraud_scores = []
        
        # Create sequences
        for i in range(n_samples):
            if i < sequence_length - 1:
                # Not enough history - use padding
                pad_length = sequence_length - i - 1
                sequence = np.vstack([
                    np.zeros((pad_length, X.shape[1])),
                    X[:i+1]
                ])
            else:
                sequence = X[i-sequence_length+1:i+1]
            
            # Convert to tensor
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = model(seq_tensor)
                score = torch.sigmoid(output).cpu().item()
            
            predictions.append(1 if score >= threshold else 0)
            fraud_scores.append(score)
        
        return np.array(predictions), np.array(fraud_scores)
    
    def save_to_mongodb(self, results_df: pd.DataFrame, batch_id: str, model_type: str):
        """Save results to MongoDB"""
        if not self.db or not self.db.connected:
            logger.warning("MongoDB not connected - skipping database save")
            return False
        
        try:
            # Prepare batch metadata
            batch_metadata = {
                'batch_id': batch_id,
                'model_type': model_type,
                'timestamp': datetime.now(),
                'total_transactions': len(results_df),
                'fraud_detected': int(results_df['prediction'].sum()),
                'fraud_percentage': float(results_df['prediction'].mean() * 100)
            }
            
            # Save batch metadata
            self.db.db['batch_results'].insert_one(batch_metadata)
            
            # Save individual results
            records = results_df.to_dict('records')
            for record in records:
                record['batch_id'] = batch_id
                record['timestamp'] = datetime.now()
                record['model_type'] = model_type
            
            self.db.db['fraud_results'].insert_many(records)
            
            logger.info(f"✓ Saved {len(records)} results to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB save error: {e}")
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
    logger.info("Starting NeuroDetect Batch API...")
    
    # Load models
    ae_loaded = processor.load_autoencoder_model()
    lstm_loaded = processor.load_lstm_model()
    
    if ae_loaded:
        logger.info("✓ Autoencoder ready")
    if lstm_loaded:
        logger.info("✓ LSTM ready")
    
    logger.info("API is ready to accept requests")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NeuroDetect Batch API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": list(MODELS.keys())
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
            'feature_names': config.get('feature_names')
        }
    
    return models_info


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
        model_type: 'autoencoder' or 'lstm'
        threshold: Optional custom threshold
    """
    try:
        # Validate model type
        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not loaded")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        logger.info(f"Processing {len(df)} transactions with {model_type}")
        
        # Generate batch ID
        batch_id = f"batch_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store original data
        original_df = df.copy()
        
        # Preprocess
        df_processed = processor.preprocess_data(df, model_type)
        
        # Predict
        if model_type == 'autoencoder':
            predictions, fraud_scores = processor.predict_autoencoder(df_processed)
        else:  # lstm
            predictions, fraud_scores = processor.predict_lstm(df_processed)
        
        # Use custom threshold if provided
        if threshold is not None:
            if model_type == 'autoencoder':
                predictions = (fraud_scores > threshold).astype(int)
            else:
                predictions = (fraud_scores >= threshold).astype(int)
        
        # Prepare results
        results_df = original_df.copy()
        results_df['prediction'] = predictions
        results_df['fraud_score'] = fraud_scores
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
            'threshold': threshold if threshold else MODEL_CONFIGS[model_type]['threshold']
        }
        
        # Save to MongoDB
        mongo_saved = processor.save_to_mongodb(results_df, batch_id, model_type)
        
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
            'preview': results_df.head(10).to_dict('records')
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
