"""
Report routes.
Routes: /reports  /reports/trigger-hourly
        /reports/{report_id}/download  /reports/{report_id}
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd

from api_utils import _json_safe, _synthesize_from_hourly
from batch_processor import processor
from routes_auth import _require_admin_user
from shared_state import RESULTS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])


def _build_df_and_stats_from_report(report: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str, str]:
    """Map a report payload into batch-style dataframe + stats for shared PDF template."""
    report_id = str(report.get('report_id') or f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    model_type = str((report.get('model') or {}).get('type') or 'unknown').lower()

    summary = report.get('summary') or {}
    sections = report.get('sections') or {}
    summary_cards = sections.get('summary_cards') or {}
    trend = sections.get('transactions_vs_frauds_trend') or {}
    merchants = sections.get('top_fraudulent_merchants') or []

    total = int(summary.get('total_transactions') or summary_cards.get('total_transactions') or 0)
    fraud_count = int(summary.get('fraud_detected') or summary_cards.get('number_of_frauds') or 0)
    threshold = float(summary.get('threshold_used') or summary.get('threshold') or 0.0)
    avg_score = float(summary.get('avg_fraud_score') or 0.0)
    fraud_rate_percent = float(summary.get('fraud_rate_percent') or (fraud_count / total * 100 if total > 0 else 0.0))

    labels = trend.get('labels') or []
    total_series = [int(x or 0) for x in (trend.get('total_transactions') or [])]
    fraud_series = [int(x or 0) for x in (trend.get('fraud_cases') or [])]
    default_ts = datetime.utcnow().replace(second=0, microsecond=0)

    rows: list[dict[str, Any]] = []

    if labels and total_series and len(total_series) == len(labels):
        safe_fraud_series = fraud_series if len(fraud_series) == len(labels) else [0] * len(labels)
        for idx, label in enumerate(labels):
            bucket_total = max(0, int(total_series[idx]))
            bucket_fraud = max(0, min(bucket_total, int(safe_fraud_series[idx])))
            for i in range(bucket_total):
                flagged = i < bucket_fraud
                rows.append({
                    'prediction': 1 if flagged else 0,
                    'fraud_score': avg_score if flagged else 0.0,
                    'risk_level': 'High' if flagged else 'Low',
                    'merchant': (merchants[i % len(merchants)].get('merchant_name') if flagged and merchants else 'Unknown Merchant') or 'Unknown Merchant',
                    'category': 'historical_summary',
                    'amt': float(summary_cards.get('fraud_amount', 0) / max(fraud_count, 1)) if flagged else float((summary_cards.get('total_amount', 0) - summary_cards.get('fraud_amount', 0)) / max(total - fraud_count, 1)) if total > fraud_count else 0.0,
                    'trans_date_trans_time': f"{default_ts.strftime('%Y-%m-%d')} {str(label)}",
                })

    if not rows and total > 0:
        for idx in range(total):
            flagged = idx < fraud_count
            rows.append({
                'prediction': 1 if flagged else 0,
                'fraud_score': avg_score if flagged else 0.0,
                'risk_level': 'High' if flagged else 'Low',
                'merchant': (merchants[idx % len(merchants)].get('merchant_name') if flagged and merchants else 'Unknown Merchant') or 'Unknown Merchant',
                'category': 'historical_summary',
                'amt': float(summary_cards.get('fraud_amount', 0) / max(fraud_count, 1)) if flagged else float((summary_cards.get('total_amount', 0) - summary_cards.get('fraud_amount', 0)) / max(total - fraud_count, 1)) if total > fraud_count else 0.0,
                'trans_date_trans_time': (default_ts + timedelta(minutes=idx)).strftime('%Y-%m-%d %H:%M:%S'),
            })

    if not rows:
        rows = []

    if len(rows) > 600:
        step = max(1, len(rows) // 600)
        rows = rows[::step][:600]

    df = pd.DataFrame(rows, columns=['prediction', 'fraud_score', 'risk_level', 'merchant', 'category', 'amt', 'trans_date_trans_time'])

    derived_total = int(len(df))
    derived_fraud = int((df['prediction'] == 1).sum())
    stats = {
        'total': derived_total,
        'fraud_count': derived_fraud,
        'legitimate_count': max(derived_total - derived_fraud, 0),
        'fraud_percentage': (derived_fraud / derived_total * 100.0) if derived_total > 0 else 0.0,
        'avg_fraud_score': float(df['fraud_score'].mean()) if 'fraud_score' in df.columns else avg_score,
        'max_fraud_score': float(df['fraud_score'].max()) if 'fraud_score' in df.columns else avg_score,
        'min_fraud_score': float(df['fraud_score'].min()) if 'fraud_score' in df.columns else 0.0,
        'threshold': threshold,
    }

    if total > 0 and fraud_count >= 0:
        stats['total'] = total
        stats['fraud_count'] = fraud_count
        stats['legitimate_count'] = max(total - fraud_count, 0)
        stats['fraud_percentage'] = fraud_rate_percent

    return df, stats, report_id, model_type


def _load_report_document(report_id: str) -> dict[str, Any]:
    """Fetch a report payload from model_reports or synthesize from hourly_reports."""
    if not processor.db or not processor.db.connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected")

    db = processor.db.db

    doc = db['model_reports'].find_one({'report_id': report_id})
    if doc:
        return _json_safe(doc)

    if report_id.startswith('rt_hourly_'):
        try:
            parts = report_id.split('_')
            date_part = parts[2]
            hour_part = parts[3][:2]
            hour_start = datetime.strptime(f"{date_part}{hour_part}", '%Y%m%d%H')
            hour_end = hour_start + timedelta(hours=1)
            hdoc = db['hourly_reports'].find_one({
                'hour_start': {'$gte': hour_start, '$lt': hour_end}
            })
        except Exception:
            hdoc = None

        if hdoc:
            return _synthesize_from_hourly(hdoc, _json_safe(hdoc))

    raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")


@router.get("")
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
        if include_hourly and (not source_type or source_type in ('realtime', None)):
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


@router.post("/trigger-hourly")
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


@router.get("/{report_id}/download")
async def download_report_pdf(report_id: str):
    """Download report PDF, using pre-generated file or on-demand generation."""
    try:
        pdf_path = RESULTS_DIR / f"{report_id}_report.pdf"
        if not pdf_path.exists():
            report_payload = _load_report_document(report_id)
            results_df, stats, derived_report_id, model_type = _build_df_and_stats_from_report(report_payload)
            generated_path = processor.generate_pdf_report(results_df, derived_report_id, model_type, stats)
            if not generated_path:
                raise HTTPException(status_code=500, detail='Failed to generate PDF report')
            return FileResponse(path=str(generated_path), media_type='application/pdf', filename=f"{derived_report_id}_report.pdf")

        return FileResponse(path=str(pdf_path), media_type="application/pdf", filename=f"{report_id}_report.pdf")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report PDF download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}")
async def get_report(report_id: str):
    """Fetch a single report document from MongoDB by its report_id.

    Checks ``model_reports`` first; falls back to ``hourly_reports`` for
    realtime hourly reports that were written there before ``save_model_report``
    could run (e.g. when the WebSocket server crashed mid-run).
    """
    try:
        report_payload = _load_report_document(report_id)
        return JSONResponse(report_payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report fetch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


