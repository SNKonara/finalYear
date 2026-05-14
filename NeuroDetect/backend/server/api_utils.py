"""
Shared utility functions for NeuroDetect Batch API route handlers.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from shared_state import RESULTS_DIR
from batch_processor import processor  # noqa: E402

logger = logging.getLogger(__name__)

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


def _as_datetime(value: Any) -> Optional[datetime]:
    """Parse various timestamp representations into a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:
            return None
    return None


def _normalize_model_label(model_value: Any) -> str:
    raw = str(model_value or '').strip().lower()
    if raw in {'ae', 'autoencoder'}:
        return 'autoencoder'
    if raw in {'lstm'}:
        return 'lstm'
    if raw in {'snn'}:
        return 'snn'
    return raw or 'unknown'


def _model_display_name(model_label: str) -> str:
    if model_label == 'autoencoder':
        return 'Autoencoder'
    if model_label == 'lstm':
        return 'LSTM'
    if model_label == 'snn':
        return 'SNN'
    return model_label.title()


def _build_alert_title(alert_doc: dict[str, Any], tx_doc: dict[str, Any]) -> str:
    merchant = str(
        tx_doc.get('merchant')
        or alert_doc.get('merchant')
        or tx_doc.get('category')
        or alert_doc.get('category')
        or 'Unknown merchant'
    )
    transaction_id = str(alert_doc.get('transaction_id') or tx_doc.get('transaction_id') or 'N/A')
    return f"Fraud alert at {merchant} ({transaction_id})"


def _extract_evidence_rows(
    tx_doc: dict[str, Any],
    alert_doc: Optional[dict[str, Any]] = None,
    max_rows: int = 8,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ad = alert_doc or {}
    fraud_score = float(ad.get('fraud_probability') or 0.0) or float(ad.get('fraud_score') or 0.0)
    threshold = float(ad.get('decision_threshold') or 0.0) or float(ad.get('optimal_threshold') or 0.5) or 0.5
    recon_error = float(ad.get('reconstruction_error') or 0.0)
    ae_threshold = float(ad.get('threshold') or 0.05) or 0.05
    ts = str(ad.get('alerted_at') or ad.get('inserted_at') or '')

    def _deviation(key: str, value: Any) -> str:
        try:
            if key in ('fraud_probability', 'fraud_score') and fraud_score > 0:
                pct = abs((fraud_score - threshold) / threshold * 100) if threshold > 0 else 0.0
                direction = 'above' if fraud_score >= threshold else 'below'
                return f'{pct:.1f}% {direction} threshold'
            if key == 'reconstruction_error' and ae_threshold > 0:
                ratio = recon_error / ae_threshold
                tag = '>1 = anomaly' if ratio > 1 else 'normal range'
                return f'{ratio:.2f}\u00d7 threshold ({tag})'
            if key == 'amt':
                v = float(value)
                if v > 1000:
                    return 'Very high amount (>$1,000)'
                if v > 500:
                    return 'Elevated amount (>$500)'
                if v < 5:
                    return 'Micro-transaction (<$5)'
                return 'Within normal range'
            if key == 'hour':
                h = int(float(value))
                if h <= 5 or h >= 23:
                    return f'Off-hours ({h}:00 \u2014 unusual window)'
                return 'Business hours'
        except Exception:
            pass
        return 'N/A'

    PRIORITY = ['amt', 'category', 'merchant', 'city', 'state', 'hour',
                'distance', 'fraud_probability', 'fraud_score', 'reconstruction_error']
    for key in PRIORITY:
        if key not in tx_doc:
            continue
        value = tx_doc[key]
        if isinstance(value, (dict, list, tuple)):
            continue
        rows.append({
            'event_id': f'EV-{len(rows) + 1:03d}',
            'timestamp': ts,
            'attribute': str(key),
            'raw_value': str(value),
            'deviation': _deviation(key, value),
        })
        if len(rows) >= max_rows:
            return rows

    for key, value in tx_doc.items():
        if key in {'_id', 'inserted_at_dt', 'dismissed_at_dt'} or key in PRIORITY:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        rows.append({
            'event_id': f'EV-{len(rows) + 1:03d}',
            'timestamp': ts,
            'attribute': str(key),
            'raw_value': str(value),
            'deviation': 'N/A',
        })
        if len(rows) >= max_rows:
            break
    return rows


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
            stored_perf.append({
                'model': mt,
                'architecture': mt,
                'accuracy': None,
                'precision': None,
                'recall': None,
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


