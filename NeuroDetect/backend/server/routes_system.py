"""
System overview routes.
Routes: /system/overview
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from batch_processor import processor
from shared_state import AUTH_SESSIONS_COLLECTION, AUTH_USERS_COLLECTION, MODEL_CONFIGS, MODELS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["system"])


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if value in (None, ''):
        return None
    return str(value)


def _build_model_entry(model_type: str) -> dict[str, Any]:
    config = MODEL_CONFIGS.get(model_type, {})
    performance = config.get('performance', {}) if isinstance(config.get('performance'), dict) else {}
    return {
        'model_type': model_type,
        'display_name': 'Autoencoder' if model_type == 'autoencoder' else model_type.upper(),
        'loaded': model_type in MODELS,
        'threshold': config.get('threshold'),
        'feature_count': config.get('num_features'),
        'sequence_length': config.get('sequence_length'),
        'time_steps': config.get('time_steps'),
        'architecture': config.get('architecture'),
        'performance': {
            'accuracy': performance.get('accuracy'),
            'precision': performance.get('precision'),
            'recall': performance.get('recall'),
            'f1_score': performance.get('f1_score'),
        },
    }


@router.get('/overview')
async def get_system_overview():
    try:
        now = datetime.utcnow()
        last_24_hours = now - timedelta(hours=24)
        mongo_connected = bool(processor.db and processor.db.connected)

        model_entries = [_build_model_entry(model_type) for model_type in ('autoencoder', 'lstm', 'snn')]
        loaded_models = sum(1 for entry in model_entries if entry['loaded'])

        overview: dict[str, Any] = {
            'generated_at': now.isoformat(),
            'read_only': True,
            'status': 'healthy' if mongo_connected and loaded_models == len(model_entries) else 'degraded',
            'services': {
                'api': {
                    'status': 'healthy' if loaded_models > 0 else 'degraded',
                    'detail': f'{loaded_models} of {len(model_entries)} models loaded',
                },
                'mongodb': {
                    'status': 'healthy' if mongo_connected else 'offline',
                    'detail': 'MongoDB connected' if mongo_connected else 'MongoDB unavailable',
                },
                'auth': {
                    'status': 'healthy' if mongo_connected else 'offline',
                    'detail': 'Session-backed authentication active' if mongo_connected else 'Authentication backend unavailable',
                },
            },
            'kpis': {
                'loaded_models': loaded_models,
                'expected_models': len(model_entries),
                'open_alerts': 0,
                'resolved_frauds': 0,
                'reports_total': 0,
                'batch_runs_last_24h': 0,
                'active_sessions': 0,
                'users_total': 0,
            },
            'models': model_entries,
            'monitoring': {
                'open_alerts': 0,
                'dismissed_alerts': 0,
                'false_positives': 0,
                'escalated_alerts': 0,
                'resolved_frauds_total': 0,
                'resolved_frauds_last_24h': 0,
                'alert_mix': {
                    'autoencoder': 0,
                    'lstm': 0,
                    'snn': 0,
                },
            },
            'reports': {
                'model_reports_total': 0,
                'hourly_reports_total': 0,
                'latest_generated_at': None,
            },
            'operations': {
                'recent_batches': [],
                'recent_events': [],
            },
            'sources': [],
        }

        if not mongo_connected:
            return JSONResponse(overview)

        db = processor.db.db
        immediate_alerts = db['immediate_alerts']
        resolved_frauds = db['resolved_frauds']
        batch_results = db['batch_results']
        model_reports = db['model_reports']
        hourly_reports = db['hourly_reports']
        auth_users = db[AUTH_USERS_COLLECTION]
        auth_sessions = db[AUTH_SESSIONS_COLLECTION]

        open_alerts = immediate_alerts.count_documents({'alert_status': {'$ne': 'dismissed'}})
        dismissed_alerts = immediate_alerts.count_documents({'alert_status': 'dismissed'})
        false_positives = immediate_alerts.count_documents({'alert_status': 'false_positive'})
        escalated_alerts = immediate_alerts.count_documents({'alert_status': 'escalated'})
        resolved_total = resolved_frauds.count_documents({})
        resolved_last_24h = resolved_frauds.count_documents({'resolved_at_dt': {'$gte': last_24_hours}})
        batch_runs_last_24h = batch_results.count_documents({'timestamp': {'$gte': last_24_hours}})
        batch_runs_total = batch_results.count_documents({})
        model_reports_total = model_reports.count_documents({})
        hourly_reports_total = hourly_reports.count_documents({})
        users_total = auth_users.count_documents({})
        active_sessions = auth_sessions.count_documents({'is_active': True, 'expires_at_dt': {'$gt': now}})

        alert_mix = {
            'autoencoder': immediate_alerts.count_documents({'model_type': {'$in': ['Autoencoder', 'autoencoder', 'AE', 'ae']}, 'alert_status': {'$ne': 'dismissed'}}),
            'lstm': immediate_alerts.count_documents({'model_type': {'$in': ['LSTM', 'lstm']}, 'alert_status': {'$ne': 'dismissed'}}),
            'snn': immediate_alerts.count_documents({'model_type': {'$in': ['SNN', 'snn']}, 'alert_status': {'$ne': 'dismissed'}}),
        }

        latest_model_report = model_reports.find_one({}, sort=[('generated_at', -1)])
        latest_hourly_report = hourly_reports.find_one({}, sort=[('generated_at', -1)])
        latest_batch = batch_results.find_one({}, sort=[('timestamp', -1)])
        latest_alert = immediate_alerts.find_one({}, sort=[('inserted_at_dt', -1)])
        latest_resolved = resolved_frauds.find_one({}, sort=[('resolved_at_dt', -1)])

        recent_batches = []
        for row in batch_results.find({}, {'_id': 0}).sort('timestamp', -1).limit(5):
            statistics = row.get('statistics') or {}
            recent_batches.append({
                'batch_id': row.get('batch_id'),
                'model_type': row.get('model_type'),
                'timestamp': _iso(row.get('processed_at') or row.get('timestamp')),
                'fraud_count': int(statistics.get('fraud_count', 0) or 0),
                'total_transactions': int(statistics.get('total', 0) or 0),
                'threshold': statistics.get('threshold'),
            })

        recent_events = []
        if latest_batch:
            recent_events.append({
                'type': 'batch',
                'title': f"Latest batch run: {str(latest_batch.get('model_type') or 'unknown').upper()}",
                'timestamp': _iso(latest_batch.get('processed_at') or latest_batch.get('timestamp')),
                'detail': f"Batch {latest_batch.get('batch_id', 'N/A')} persisted to MongoDB.",
            })
        if latest_alert:
            recent_events.append({
                'type': 'alert',
                'title': f"Latest queued alert: {latest_alert.get('risk_level', 'High')} risk",
                'timestamp': _iso(latest_alert.get('inserted_at_dt') or latest_alert.get('inserted_at') or latest_alert.get('alerted_at')),
                'detail': f"Transaction {latest_alert.get('transaction_id', 'N/A')} remains in the immediate alerts queue.",
            })
        if latest_resolved:
            recent_events.append({
                'type': 'resolution',
                'title': 'Most recent confirmed fraud archived',
                'timestamp': _iso(latest_resolved.get('resolved_at_dt') or latest_resolved.get('resolved_at')),
                'detail': f"{latest_resolved.get('transaction_id', 'N/A')} resolved by {((latest_resolved.get('resolved_by') or {}).get('name')) or 'analyst'}.",
            })

        latest_report_at = None
        if latest_model_report or latest_hourly_report:
            report_candidates = [
                _iso((latest_model_report or {}).get('generated_at')),
                _iso((latest_hourly_report or {}).get('generated_at')),
            ]
            report_candidates = [candidate for candidate in report_candidates if candidate]
            latest_report_at = max(report_candidates) if report_candidates else None

        overview['status'] = 'healthy' if loaded_models == len(model_entries) else 'degraded'
        overview['kpis'] = {
            'loaded_models': loaded_models,
            'expected_models': len(model_entries),
            'open_alerts': open_alerts,
            'resolved_frauds': resolved_total,
            'reports_total': model_reports_total + hourly_reports_total,
            'batch_runs_last_24h': batch_runs_last_24h,
            'active_sessions': active_sessions,
            'users_total': users_total,
        }
        overview['monitoring'] = {
            'open_alerts': open_alerts,
            'dismissed_alerts': dismissed_alerts,
            'false_positives': false_positives,
            'escalated_alerts': escalated_alerts,
            'resolved_frauds_total': resolved_total,
            'resolved_frauds_last_24h': resolved_last_24h,
            'alert_mix': alert_mix,
        }
        overview['reports'] = {
            'model_reports_total': model_reports_total,
            'hourly_reports_total': hourly_reports_total,
            'latest_generated_at': latest_report_at,
        }
        overview['operations'] = {
            'batch_runs_total': batch_runs_total,
            'recent_batches': recent_batches,
            'recent_events': recent_events,
        }
        overview['sources'] = [
            'mongodb:immediate_alerts',
            'mongodb:resolved_frauds',
            'mongodb:batch_results',
            'mongodb:model_reports',
            'mongodb:hourly_reports',
            f'mongodb:{AUTH_USERS_COLLECTION}',
            f'mongodb:{AUTH_SESSIONS_COLLECTION}',
        ]

        return JSONResponse(overview)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f'System overview retrieval error: {exc}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))