"""
Alert routes.
Routes: /alerts/summary  /alerts/high-risk
        /alerts/explain/{batch_id}/{transaction_id}
        /alerts/live/high-risk  /alerts/live/dismiss
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api_utils import _json_safe, _load_batch_result_payload
from batch_processor import processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


class DismissAlertRequest(BaseModel):
    transaction_id: Optional[str] = None
    alert_id: Optional[str] = None
    dismissed_by: str = 'investigator'


@router.get("/summary")
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


@router.get("/high-risk")
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


@router.get("/explain/{batch_id}/{transaction_id}")
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


@router.get("/live/high-risk")
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


@router.post("/live/dismiss")
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


