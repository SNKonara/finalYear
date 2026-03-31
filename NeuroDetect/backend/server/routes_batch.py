"""
Batch processing routes.
Routes: /batch/process  /batch/download/{batch_id}
        /batch/history  /batch/logs
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api_utils import _json_safe, _load_batch_result_payload
from batch_processor import processor
from shared_state import MODEL_CONFIGS, MODELS, RESULTS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/batch", tags=["batch"])

@router.post("/process")
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


@router.get("/download/{batch_id}")
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


@router.get("/history")
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


@router.get("/logs")
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


