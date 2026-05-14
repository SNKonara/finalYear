"""
Investigation routes.
Routes: /investigations/alerts  /investigations/alerts/{alert_id}
        /investigations/alerts/resolve-fraud
        /investigations/alerts/false-positive
        /investigations/alerts/escalate
"""
import logging
import os
import base64
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib import parse, request as urllib_request
from urllib.error import URLError, HTTPError

import api_utils
from bson import ObjectId
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api_utils import (
    _build_alert_title,
    _extract_evidence_rows,
    _json_safe,
    _model_display_name,
    _normalize_model_label,
)
from shared_state import AUTH_USERS_COLLECTION
from batch_processor import processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/investigations", tags=["investigations"])
ALERT_NOTIFICATION_COLLECTION = 'alert_notifications'


class ResolveFraudRequest(BaseModel):
    transaction_id: Optional[str] = None
    alert_id: Optional[str] = None
    analyst_id: Optional[str] = None
    analyst_name: str
    analyst_email: Optional[str] = None
    analyst_role: str = 'analyst'
    resolution_note: Optional[str] = None


class FalsePositiveRequest(BaseModel):
    transaction_id: Optional[str] = None
    alert_id: Optional[str] = None
    analyst_name: str = 'investigator'
    analyst_note: Optional[str] = None


class EscalateRequest(BaseModel):
    transaction_id: Optional[str] = None
    alert_id: Optional[str] = None
    analyst_name: str = 'investigator'
    escalated_to: Optional[str] = None
    escalation_note: Optional[str] = None


def _normalize_phone_number(phone_number: Any) -> str:
    return str(phone_number or '').strip()


def _send_sms_notification(phone_number: str, message: str) -> dict[str, Any]:
    account_sid = os.getenv('TWILIO_ACCOUNT_SID', '').strip()
    auth_token = os.getenv('TWILIO_AUTH_TOKEN', '').strip()
    from_number = os.getenv('TWILIO_FROM_NUMBER', '').strip()

    if account_sid and auth_token and from_number:
        url = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json'
        body = parse.urlencode({
            'To': phone_number,
            'From': from_number,
            'Body': message,
        }).encode('utf-8')
        basic_auth = base64.b64encode(f'{account_sid}:{auth_token}'.encode('utf-8')).decode('ascii')
        req = urllib_request.Request(
            url,
            data=body,
            method='POST',
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f'Basic {basic_auth}',
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=10) as response:
                return {'sent': True, 'provider': 'twilio', 'status': response.status}
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            logger.warning("SMS delivery failed for %s: %s", phone_number, exc)
            return {'sent': False, 'provider': 'twilio', 'reason': str(exc)}

    return {'sent': False, 'provider': 'queue', 'reason': 'sms_provider_not_configured'}


def _notify_senior_analysts(alert_doc: dict[str, Any], analyst_name: str, escalation_note: Optional[str]) -> list[dict[str, Any]]:
    if not processor.db or not processor.db.connected:
        return []

    db = processor.db.db
    users_collection = db[AUTH_USERS_COLLECTION]
    notifications_collection = db[ALERT_NOTIFICATION_COLLECTION]
    senior_users = list(users_collection.find({
        'role': 'senior_analyst',
        'phone_number': {'$exists': True, '$nin': ['', None]},
    }, {'password_hash': 0}))

    alert_title = str(alert_doc.get('title') or 'Fraud alert')
    transaction_id = str(alert_doc.get('transaction_id') or '')
    message = f"Fraud alert escalated: {transaction_id}. {alert_title}."
    if escalation_note:
        message = f"{message} Note: {escalation_note.strip()}"

    now_utc = datetime.utcnow()
    notifications: list[dict[str, Any]] = []
    for user in senior_users:
        phone_number = _normalize_phone_number(user.get('phone_number'))
        if not phone_number:
            continue

        delivery = _send_sms_notification(phone_number, message)
        record = {
            'type': 'senior_analyst_escalation',
            'alert_id': str(alert_doc.get('_id')),
            'transaction_id': transaction_id,
            'recipient_user_id': str(user.get('_id')),
            'recipient_name': str(user.get('name') or ''),
            'recipient_phone_number': phone_number,
            'message': message,
            'delivery': delivery,
            'created_at': now_utc.isoformat(),
            'created_at_dt': now_utc,
            'created_by': analyst_name,
        }
        notifications_collection.insert_one(record)
        notifications.append(record)

    return notifications


@router.get("/alerts")
async def get_investigation_alerts(hours: int = 24, limit: int = 200, status: str = 'open'):
    """Return investigation-ready alerts from MongoDB for the last N hours."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        hours = max(1, min(hours, 168))
        limit = max(1, min(limit, 500))
        status_value = status.strip().lower()
        if status_value not in {'open', 'dismissed', 'resolved_fraud', 'escalated', 'all'}:
            raise HTTPException(status_code=400, detail="status must be one of: open, dismissed, resolved_fraud, escalated, all")

        db = processor.db.db
        since = datetime.utcnow() - timedelta(hours=hours)

        query: dict[str, Any] = {
            'inserted_at_dt': {'$gte': since},
            '$or': [
                {'is_fraud': True},
                {'risk_level': {'$in': ['High', 'Medium-High']}},
            ],
        }
        if status_value != 'all':
            query['alert_status'] = status_value

        cursor = db['immediate_alerts'].find(query).sort('inserted_at_dt', -1).limit(limit)

        model_base_query = dict(query)
        model_counts = {
            'autoencoder': db['immediate_alerts'].count_documents({
                **model_base_query,
                'model_type': {'$in': ['Autoencoder', 'autoencoder', 'AE', 'ae']},
            }),
            'lstm': db['immediate_alerts'].count_documents({
                **model_base_query,
                'model_type': {'$in': ['LSTM', 'lstm']},
            }),
            'snn': db['immediate_alerts'].count_documents({
                **model_base_query,
                'model_type': {'$in': ['SNN', 'snn']},
            }),
        }

        alerts: list[dict[str, Any]] = []

        for doc in cursor:
            transaction_id = str(doc.get('transaction_id') or 'N/A')
            model_label = _normalize_model_label(doc.get('model_type'))

            fraud_score = float(doc.get('fraud_probability', 0.0) or 0.0)
            threshold = float(doc.get('decision_threshold', 0.5) or 0.5)
            risk_percent = max(0, min(100, int(round(fraud_score * 100))))

            inserted_dt = api_utils._as_datetime(doc.get('inserted_at_dt')) or api_utils._as_datetime(doc.get('inserted_at'))
            opened_at = inserted_dt.isoformat() if inserted_dt else str(doc.get('inserted_at') or doc.get('alerted_at') or '')

            tx_data = doc.get('transaction_data') or {}
            tx_doc = tx_data if isinstance(tx_data, dict) else {}
            alerts.append({
                'alert_id': str(doc.get('_id')),
                'transaction_id': transaction_id,
                'title': _build_alert_title(doc, tx_doc),
                'model_type': model_label,
                'model_label': _model_display_name(model_label),
                'risk_percent': risk_percent,
                'fraud_score': fraud_score,
                'decision_threshold': threshold,
                'risk_level': str(doc.get('risk_level') or 'High'),
                'is_fraud': bool(doc.get('is_fraud', False)),
                'status': str(doc.get('alert_status') or 'open'),
                'opened_at': opened_at,
                'opened_ago_minutes': int(max(0, (datetime.utcnow() - inserted_dt).total_seconds() // 60)) if inserted_dt else None,
                'amount': float(tx_doc.get('amt', 0.0) or 0.0),
                'merchant': str(tx_doc.get('merchant') or tx_doc.get('category') or 'N/A'),
                'category': str(tx_doc.get('category') or 'N/A'),
            })

        return JSONResponse({
            'hours': hours,
            'count': len(alerts),
            'status_filter': status_value,
            'model_counts': {
                'autoencoder': model_counts['autoencoder'],
                'lstm': model_counts['lstm'],
                'snn': model_counts['snn'],
                'total': sum(model_counts.values()),
            },
            'alerts': alerts,
            'source': 'mongodb:immediate_alerts',
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Investigation alerts retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}")
async def get_investigation_alert_detail(alert_id: str):
    """Return full transaction details for one selected fraud alert."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        db = processor.db.db
        doc = None
        if ObjectId is not None:
            try:
                doc = db['immediate_alerts'].find_one({'_id': ObjectId(alert_id)})
            except Exception:
                doc = None
        if doc is None:
            doc = db['immediate_alerts'].find_one({'_id': alert_id})

        source_collection = 'mongodb:immediate_alerts'
        if not doc:
            doc = db['resolved_frauds'].find_one({'source_alert_id': alert_id})
            source_collection = 'mongodb:resolved_frauds'

        if not doc:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

        tx_doc = doc.get('transaction_data') if isinstance(doc.get('transaction_data'), dict) else {}
        if not tx_doc and isinstance(doc.get('transaction_context'), dict):
            tx_doc = dict(doc.get('transaction_context') or {})
            tx_doc['transaction_id'] = doc.get('transaction_id')
            tx_doc['merchant'] = doc.get('merchant')
            tx_doc['category'] = doc.get('category')
            tx_doc['city'] = doc.get('city')
            tx_doc['state'] = doc.get('state')
            tx_doc['job'] = doc.get('job')
            tx_doc['amt'] = doc.get('amount')
        model_label = _normalize_model_label(doc.get('model_type'))

        fraud_score = (
            float(doc.get('fraud_probability') or 0.0)
            or float(doc.get('fraud_score') or 0.0)
        )
        threshold = (
            float(doc.get('decision_threshold') or 0.0)
            or float(doc.get('optimal_threshold') or 0.5)
            or 0.5
        )
        explainability_data = processor.build_investigation_explainability(doc)

        return JSONResponse({
            'alert_id': str(doc.get('_id')),
            'transaction_id': str(doc.get('transaction_id') or tx_doc.get('transaction_id') or 'N/A'),
            'title': _build_alert_title(doc, tx_doc),
            'model_type': model_label,
            'model_label': _model_display_name(model_label),
            'risk_level': str(doc.get('risk_level') or 'High'),
            'fraud_score': fraud_score,
            'decision_threshold': threshold,
            'risk_percent': max(0, min(100, int(round(fraud_score * 100)))),
            'is_fraud': bool(doc.get('is_fraud', False)),
            'status': str(doc.get('alert_status') or 'open'),
            'alerted_at': str(doc.get('alerted_at') or doc.get('inserted_at') or ''),
            'dismissed_at': str(doc.get('dismissed_at') or ''),
            'resolved_at': str(doc.get('resolved_at') or ''),
            'resolved_by': _json_safe(doc.get('resolved_by') or {}),
            'amount': float(tx_doc.get('amt', 0.0) or 0.0),
            'merchant': str(tx_doc.get('merchant') or tx_doc.get('category') or 'N/A'),
            'category': str(tx_doc.get('category') or 'N/A'),
            'city': str(tx_doc.get('city') or 'N/A'),
            'state': str(tx_doc.get('state') or 'N/A'),
            'job': str(tx_doc.get('job') or 'N/A'),
            'raw_transaction': _json_safe(tx_doc),
            'raw_alert': _json_safe(doc) if source_collection == 'mongodb:immediate_alerts' else {},
            'evidence': _json_safe(doc.get('evidence')) if source_collection == 'mongodb:resolved_frauds' else _extract_evidence_rows(tx_doc, doc),
            'source': source_collection,
            'confidence_pct': explainability_data.get('confidence_pct', round(fraud_score * 100.0, 1)),
            'score_series': explainability_data.get('score_series', []),
            'explainability': {
                'reasons': explainability_data.get('reasons', []),
                'top_factors': explainability_data.get('top_factors', []),
                'feature_contribution_chart': explainability_data.get('feature_contribution_chart', {'labels': [], 'values': []}),
                'threshold_chart': explainability_data.get('threshold_chart', {}),
                'risk_dimension_chart': explainability_data.get('risk_dimension_chart', {'labels': [], 'values': []}),
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Investigation alert detail retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/resolve-fraud")
async def resolve_investigation_alert_as_fraud(payload: ResolveFraudRequest):
    """Resolve a queued alert as confirmed fraud and archive only compact details."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        if not payload.transaction_id and not payload.alert_id:
            raise HTTPException(status_code=400, detail="Provide transaction_id or alert_id")

        analyst_name = payload.analyst_name.strip()
        if not analyst_name:
            raise HTTPException(status_code=400, detail="Analyst name is required")

        result = processor.db.resolve_immediate_alert_as_fraud(
            transaction_id=payload.transaction_id,
            alert_id=payload.alert_id,
            analyst={
                'id': payload.analyst_id or '',
                'name': analyst_name,
                'email': payload.analyst_email or '',
                'role': payload.analyst_role or 'analyst',
            },
            resolution_note=payload.resolution_note,
        )

        if not result.get('updated'):
            if result.get('reason') == 'missing_identifier':
                raise HTTPException(status_code=400, detail='Provide transaction_id or alert_id')
            if result.get('reason') == 'already_resolved':
                raise HTTPException(status_code=409, detail='Alert has already been resolved as fraud')
            if result.get('reason') == 'not_found':
                raise HTTPException(status_code=404, detail='No matching alert found to resolve')
            raise HTTPException(status_code=500, detail=str(result.get('reason') or 'Resolution failed'))

        return JSONResponse({
            'success': True,
            'message': 'Alert moved to resolved fraud archive',
            'result': result,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolve fraud alert error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/false-positive")
async def mark_investigation_alert_false_positive(payload: FalsePositiveRequest):
    """Mark a queued alert as a false positive."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        if not payload.transaction_id and not payload.alert_id:
            raise HTTPException(status_code=400, detail="Provide transaction_id or alert_id")

        db = processor.db.db
        query: dict[str, Any] = {}
        if payload.alert_id:
            if ObjectId is not None:
                try:
                    query['_id'] = ObjectId(payload.alert_id)
                except Exception:
                    query['_id'] = payload.alert_id
            else:
                query['_id'] = payload.alert_id
        if payload.transaction_id and not query:
            query['transaction_id'] = payload.transaction_id

        now_utc = datetime.utcnow()
        result = db['immediate_alerts'].update_one(
            query,
            {'$set': {
                'alert_status': 'false_positive',
                'false_positive_at': now_utc.isoformat(),
                'false_positive_at_dt': now_utc,
                'false_positive_by': (payload.analyst_name or 'investigator').strip(),
                'false_positive_note': payload.analyst_note or '',
            }},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="No matching alert found")

        return JSONResponse({
            'success': True,
            'message': 'Alert marked as false positive',
            'modified': result.modified_count > 0,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark false positive error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/escalate")
async def escalate_investigation_alert(payload: EscalateRequest):
    """Escalate a queued alert to a senior analyst."""
    try:
        if not processor.db or not processor.db.connected:
            raise HTTPException(status_code=503, detail="MongoDB is not connected")

        if not payload.transaction_id and not payload.alert_id:
            raise HTTPException(status_code=400, detail="Provide transaction_id or alert_id")

        db = processor.db.db
        query: dict[str, Any] = {}
        if payload.alert_id:
            if ObjectId is not None:
                try:
                    query['_id'] = ObjectId(payload.alert_id)
                except Exception:
                    query['_id'] = payload.alert_id
            else:
                query['_id'] = payload.alert_id
        if payload.transaction_id and not query:
            query['transaction_id'] = payload.transaction_id

        now_utc = datetime.utcnow()
        result = db['immediate_alerts'].update_one(
            query,
            {'$set': {
                'alert_status': 'escalated',
                'escalated_at': now_utc.isoformat(),
                'escalated_at_dt': now_utc,
                'escalated_by': (payload.analyst_name or 'investigator').strip(),
                'escalated_to': payload.escalated_to or 'senior_analyst',
                'escalation_note': payload.escalation_note or '',
            }},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="No matching alert found")

        alert_doc = db['immediate_alerts'].find_one(query) or {
            '_id': payload.alert_id or payload.transaction_id,
            'transaction_id': payload.transaction_id or '',
            'title': 'Fraud alert',
        }
        notifications = _notify_senior_analysts(alert_doc, (payload.analyst_name or 'investigator').strip(), payload.escalation_note)

        return JSONResponse({
            'success': True,
            'message': 'Alert escalated to senior analyst',
            'modified': result.modified_count > 0,
            'notifications_sent': len(notifications),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Escalate alert error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


