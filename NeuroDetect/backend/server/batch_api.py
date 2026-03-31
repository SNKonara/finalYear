"""
NeuroDetect Batch API — entry point.
Business logic is split across dedicated modules:
  shared_state.py          — global dicts & path constants
  batch_processor.py       — BatchProcessor class + singleton
  api_utils.py             — shared utility helpers
  routes_auth.py           — /auth/* endpoints
  routes_batch.py          — /batch/* endpoints
  routes_alerts.py         — /alerts/* endpoints
  routes_investigations.py — /investigations/* endpoints
  routes_reports.py        — /reports/* endpoints
"""
import os
import sys
import json
import logging
import hashlib
import random
import secrets
from datetime import datetime, timedelta, timezone
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart

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







# ── Shared state & processor ──────────────────────────────────────────────────
from shared_state import (
    MODELS, PREPROCESSORS, MODEL_CONFIGS,
    BASE_DIR, PROJECT_DIR, SAVED_MODELS_DIR, RESULTS_DIR,
    AUTH_USERS_COLLECTION, AUTH_SESSIONS_COLLECTION,
)
from batch_processor import processor

# ── Route modules ─────────────────────────────────────────────────────────────
from routes_auth import (
    router as auth_router,
    _ensure_auth_collections,
    _seed_default_auth_users,
)
from routes_batch import router as batch_router
from routes_alerts import router as alerts_router
from routes_investigations import router as investigations_router
from routes_reports import router as reports_router

app.include_router(auth_router)
app.include_router(batch_router)
app.include_router(alerts_router)
app.include_router(investigations_router)
app.include_router(reports_router)

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



if __name__ == "__main__":
    uvicorn.run(
        "batch_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
