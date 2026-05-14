"""
Batch processing routes.
Routes: /batch/process  /batch/download/{batch_id}
        /batch/history  /batch/logs
"""
import json
import logging
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from api_utils import _json_safe, _load_batch_result_payload
from batch_processor import processor
from shared_state import MODEL_CONFIGS, MODELS, RESULTS_DIR, SAVED_MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/batch", tags=["batch"])

THRESHOLD_OVERRIDES_PATH = SAVED_MODELS_DIR / "threshold_overrides.json"
THRESHOLD_AUDIT_COLLECTION = "threshold_audit_logs"


class ThresholdUpdateRequest(BaseModel):
    model_type: str
    threshold: float
    persist: bool = True
    reason: Optional[str] = None


def _resolve_actor(authorization: Optional[str]) -> dict[str, str]:
    if not authorization:
        return {'email': 'system', 'role': 'system'}

    try:
        from routes_auth import _resolve_current_user

        user = _resolve_current_user(authorization)
        return {
            'email': str(user.get('email', 'unknown')),
            'role': str(user.get('role', 'unknown')),
        }
    except Exception:
        return {'email': 'unknown', 'role': 'unknown'}


def _write_threshold_audit(
    *,
    model_type: str,
    previous_threshold: float,
    new_threshold: float,
    persisted: bool,
    actor: dict[str, str],
    reason: Optional[str],
) -> bool:
    db_client = getattr(processor, 'db', None)
    if not db_client or not getattr(db_client, 'connected', False):
        return False

    try:
        payload = {
            'event_type': 'threshold_update',
            'model_type': model_type,
            'previous_threshold': float(previous_threshold),
            'new_threshold': float(new_threshold),
            'persisted': bool(persisted),
            'reason': (reason or '').strip() or None,
            'changed_at': datetime.utcnow().isoformat(),
            'changed_at_dt': datetime.utcnow(),
            'actor': {
                'email': actor.get('email', 'unknown'),
                'role': actor.get('role', 'unknown'),
            },
        }
        db_client.db[THRESHOLD_AUDIT_COLLECTION].insert_one(payload)
        return True
    except Exception as error:
        logger.warning(f"Failed to write threshold audit event: {error}")
        return False


def _validate_threshold(model_type: str, threshold: float) -> None:
    if model_type == 'autoencoder':
        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail='Autoencoder threshold must be between 0 and 1')
    else:
        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail='Threshold must be between 0 and 1')


def _to_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'fraud', 'yes'}:
            return 1
        if lowered in {'0', 'false', 'legit', 'legitimate', 'no'}:
            return 0
        try:
            return 1 if float(lowered) >= 0.5 else 0
        except ValueError:
            return None
    return None


def _compute_labeled_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute accuracy-style metrics when ground-truth labels exist in input payload."""
    if 'prediction' not in results_df.columns:
        return {}

    truth_column = None
    if 'is_fraud' in results_df.columns:
        truth_column = 'is_fraud'
    elif 'H1' in results_df.columns:
        truth_column = 'H1'

    if truth_column is None:
        return {}

    truth_series = results_df[truth_column].map(_to_binary_label)
    pred_series = results_df['prediction'].map(_to_binary_label)
    valid_mask = truth_series.notna() & pred_series.notna()
    if int(valid_mask.sum()) == 0:
        return {}

    truth = truth_series[valid_mask].astype(int)
    pred = pred_series[valid_mask].astype(int)

    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    total = int(len(truth))
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if positives > 0 and (tp + fn) > 0 else None
    f1 = float((2 * precision * recall) / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else None
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    specificity = float(tn / (tn + fp)) if negatives > 0 and (tn + fp) > 0 else None
    false_positive_rate = float(fp / negatives) if negatives > 0 else None

    return {
        'truth_column': truth_column,
        'labeled_count': total,
        'positive_labels': positives,
        'negative_labels': negatives,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'accuracy_percent': round(accuracy * 100.0, 2),
        'precision_percent': round(precision * 100.0, 2) if precision is not None else None,
        'recall_percent': round(recall * 100.0, 2) if recall is not None else None,
        'f1_percent': round(f1 * 100.0, 2) if f1 is not None else None,
        'specificity_percent': round(specificity * 100.0, 2) if specificity is not None else None,
        'false_positive_rate_percent': round(false_positive_rate * 100.0, 2) if false_positive_rate is not None else None,
        'single_class_labels': positives == 0 or negatives == 0,
    }


def _load_threshold_overrides() -> dict[str, float]:
    if not THRESHOLD_OVERRIDES_PATH.exists():
        return {}
    try:
        with open(THRESHOLD_OVERRIDES_PATH, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        raw = payload.get('overrides', payload)
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in raw.items()
            if key in {'autoencoder', 'lstm', 'snn'}
        }
    except Exception as error:
        logger.warning(f"Failed to read threshold overrides: {error}")
        return {}


def apply_saved_threshold_overrides() -> dict[str, float]:
    """Apply saved threshold overrides to in-memory model config at startup."""
    overrides = _load_threshold_overrides()
    applied: dict[str, float] = {}
    for model_type, override in overrides.items():
        if model_type in MODEL_CONFIGS:
            MODEL_CONFIGS[model_type]['threshold'] = float(override)
            applied[model_type] = float(override)
    if applied:
        logger.info(f"Applied threshold overrides: {applied}")
    return applied


def _persist_threshold_overrides(overrides: dict[str, float]) -> None:
    THRESHOLD_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'updated_at': datetime.utcnow().isoformat(),
        'overrides': overrides,
    }
    with open(THRESHOLD_OVERRIDES_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


@router.post('/threshold')
async def update_model_threshold(
    payload: ThresholdUpdateRequest,
    authorization: Optional[str] = Header(default=None),
):
    model_type = payload.model_type.strip().lower()
    if model_type not in {'autoencoder', 'lstm', 'snn'}:
        raise HTTPException(status_code=400, detail='model_type must be one of: autoencoder, lstm, snn')

    threshold_value = float(payload.threshold)
    _validate_threshold(model_type, threshold_value)

    if model_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Model '{model_type}' is not loaded")

    previous_threshold = float(MODEL_CONFIGS[model_type]['threshold'])
    MODEL_CONFIGS[model_type]['threshold'] = threshold_value
    persisted = False

    if payload.persist:
        overrides = _load_threshold_overrides()
        overrides[model_type] = threshold_value
        _persist_threshold_overrides(overrides)
        persisted = True

    actor = _resolve_actor(authorization)
    audit_logged = _write_threshold_audit(
        model_type=model_type,
        previous_threshold=previous_threshold,
        new_threshold=threshold_value,
        persisted=persisted,
        actor=actor,
        reason=payload.reason,
    )

    return JSONResponse({
        'success': True,
        'model_type': model_type,
        'previous_threshold': previous_threshold,
        'threshold': threshold_value,
        'persisted': persisted,
        'audit_logged': audit_logged,
        'updated_by': actor,
    })

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
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail='MongoDB must be connected (mongodb-only persistence mode)')

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

        labeled_metrics = _compute_labeled_metrics(results_df)
        if labeled_metrics:
            stats['labeled_metrics'] = labeled_metrics
            stats['accuracy'] = labeled_metrics['accuracy']
            stats['precision'] = labeled_metrics.get('precision')
            stats['recall'] = labeled_metrics.get('recall')
            stats['f1'] = labeled_metrics.get('f1')
            stats['specificity'] = labeled_metrics.get('specificity')
            stats['false_positive_rate'] = labeled_metrics.get('false_positive_rate')

            model_perf = MODEL_CONFIGS.setdefault(model_type, {}).setdefault('performance', {})
            model_perf['accuracy'] = labeled_metrics['accuracy']
            if labeled_metrics.get('precision') is not None:
                model_perf['precision'] = labeled_metrics['precision']
            if labeled_metrics.get('recall') is not None:
                model_perf['recall'] = labeled_metrics['recall']
            if labeled_metrics.get('f1') is not None:
                model_perf['f1'] = labeled_metrics['f1']
                model_perf['f1_score'] = labeled_metrics['f1']
            if labeled_metrics.get('specificity') is not None:
                model_perf['specificity'] = labeled_metrics['specificity']
            if labeled_metrics.get('false_positive_rate') is not None:
                model_perf['false_positive_rate'] = labeled_metrics['false_positive_rate']
            model_perf['runtime_validated_at'] = datetime.utcnow().isoformat()

            if labeled_metrics.get('single_class_labels'):
                logger.info(
                    f"Runtime labeled metrics ({model_type}) on single-class batch: "
                    f"accuracy={labeled_metrics['accuracy_percent']:.2f}% "
                    f"specificity={labeled_metrics.get('specificity_percent')} "
                    f"fpr={labeled_metrics.get('false_positive_rate_percent')}"
                )
            else:
                def _fmt(v):
                    return f"{v:.2f}" if v is not None else "N/A"
                logger.info(
                    f"Runtime labeled metrics ({model_type}): "
                    f"accuracy={_fmt(labeled_metrics.get('accuracy_percent'))}% "
                    f"precision={_fmt(labeled_metrics.get('precision_percent'))}% "
                    f"recall={_fmt(labeled_metrics.get('recall_percent'))}% "
                    f"f1={_fmt(labeled_metrics.get('f1_percent'))}%"
                )
        
        # Save to MongoDB (batch summary and transaction results)
        mongo_saved = processor.save_to_mongodb(results_df, batch_id, model_type, stats)
        if not mongo_saved:
            raise HTTPException(status_code=500, detail='Failed to persist batch results to MongoDB')
        
        logger.info(f"✓ Batch processing complete: {batch_id}")
        processor.save_processing_log(
            batch_id=batch_id,
            model_type=model_type,
            event='batch_completed',
            level='info',
            details={
                'statistics': stats,
                'mongodb_saved': bool(mongo_saved),
                'storage_mode': 'mongodb_only',
            }
        )

        download_base = f"/batch/download/{batch_id}"
        
        return JSONResponse({
            'success': True,
            'batch_id': batch_id,
            'statistics': stats,
            'files': {
                'json': f"{download_base}?format=json",
                'csv': f"{download_base}?format=csv",
                'pdf': f"{download_base}?format=pdf",
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
    """Download batch report in specified format from MongoDB-backed payload."""
    try:
        if format not in {"pdf", "csv", "json"}:
            raise HTTPException(status_code=400, detail="Invalid format")

        data, _ = _load_batch_result_payload(batch_id=batch_id)
        results = data.get('results', [])
        model_type = str(data.get('model_type', 'unknown'))

        if format == 'json':
            payload = json.dumps(data, indent=2, default=str)
            headers = {'Content-Disposition': f'attachment; filename={batch_id}_results.json'}
            return Response(content=payload, media_type='application/json', headers=headers)

        if format == 'csv':
            df = pd.DataFrame(results)
            csv_content = df.to_csv(index=False)
            headers = {'Content-Disposition': f'attachment; filename={batch_id}_results.csv'}
            return Response(content=csv_content, media_type='text/csv', headers=headers)

        # format == 'pdf': generate on-demand PDF in memory (no local file persistence)
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle(f"{batch_id}_report")
        pdf.setFont('Helvetica-Bold', 14)
        pdf.drawString(40, 760, f"NeuroDetect Batch Report - {model_type.upper()}")
        pdf.setFont('Helvetica', 10)
        pdf.drawString(40, 742, f"Batch ID: {batch_id}")
        pdf.drawString(40, 728, f"Generated: {datetime.now().isoformat()}")

        stats = data.get('statistics', {}) or {}
        y = 705
        for key in ['total', 'fraud_count', 'legitimate_count', 'fraud_percentage', 'avg_fraud_score', 'max_fraud_score', 'threshold']:
            if key in stats:
                pdf.drawString(40, y, f"{key}: {stats.get(key)}")
                y -= 14

        pdf.drawString(40, y - 8, f"Transactions in report: {len(results)}")
        pdf.showPage()
        pdf.save()
        buffer.seek(0)

        headers = {'Content-Disposition': f'attachment; filename={batch_id}_report.pdf'}
        return StreamingResponse(buffer, media_type='application/pdf', headers=headers)
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_batch_history():
    """Get history of batch processing jobs from MongoDB."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        cursor = processor.db.db['batch_results'].find({}, {'_id': 0}).sort('timestamp', -1)
        history = []
        for row in list(cursor):
            history.append({
                'batch_id': row.get('batch_id'),
                'model_type': row.get('model_type'),
                'timestamp': row.get('processed_at') or row.get('timestamp'),
                'statistics': row.get('statistics', {}),
            })

        return JSONResponse({'history': _json_safe(history), 'source': 'mongodb:batch_results'})
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        if isinstance(e, HTTPException):
            raise e
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


