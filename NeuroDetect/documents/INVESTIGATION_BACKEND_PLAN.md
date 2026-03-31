# Investigation Detail Page — Backend Implementation Plan

## Context

The **Investigation Detail** page (`src/components/pages/InvestigationDetail.tsx`) currently displays placeholder/synthetic data:
- **0.0% Model Confidence** — `fraud_score` stored as 0 in MongoDB because the field name mismatch (`fraud_probability` vs `fraud_score`)
- **0% Feature Contributions** — `buildFeatureContributions()` is a pure frontend heuristic (risk × fixed multipliers), not real model output
- **Synthetic Chart Series** — `chartSeries` is a hardcoded 9-point array pattern, not real score history
- **Generic Evidence Rows** — `_extract_evidence_rows()` always sets `deviation: 'N/A'`, no model context
- **Missing Action Endpoints** — "Mark as False Positive" and "Escalate" buttons have no backend

---

## Step Map

### Step 1 — Fix fraud_score / confidence_pct bug (Quick Fix)
**File:** `backend/server/batch_api.py`  
**Where:** `GET /investigations/alerts/{alert_id}` route (≈ line 2837)  
**Problem:** Response reads `fraud_score` from alert doc but real-time alerts store the field as `fraud_probability`. The confidence ring shows 0.0% because the field is missing.  
**Fix:**
- When building the detail response, try `fraud_score` first, then fall back to `fraud_probability`
- Add `confidence_pct` field = `round(score * 100, 1)` to the response  
**Status:** [x] DONE — Build `build_investigation_explainability()` in batch_api.py
**File:** `backend/server/batch_api.py`  
**Where:** New standalone method on `FraudDetectionAPI` class, near the existing `build_snn_explanations()` (≈ line 490)  
**Purpose:** Single entry point that reads a stored `alert_doc` from MongoDB and produces real explainability output — no re-inference needed, everything is derived from the stored data.

**Input:** `alert_doc` dict (as returned from MongoDB)  
**Output:**
```json
{
  "reasons": ["string", ...],
  "top_factors": [{"feature": "str", "value": "str", "contribution_pct": 0-100}, ...],
  "score_series": [0.12, 0.18, ...],
  "feature_contribution_chart": {"labels": [...], "values": [...]},
  "threshold_chart": {"probability": 0.0, "threshold": 0.0, "margin": 0.0},
  "risk_dimension_chart": {"labels": [...], "values": [...]},
  "confidence_pct": 0.0
}
```

**Sub-functions to add inside or alongside:**
- `_build_snn_explain(alert_doc)` — uses stored `transaction_data`, computes feature group aggregates (Amount, Location, Time, Category)
- `_build_ae_explain(alert_doc)` — uses `reconstruction_error` / `threshold`; distributes error across feature groups
- `_build_lstm_explain(alert_doc)` — uses `fraud_score` / `decision_threshold`; produces sequence-style score_series  
**Status:** [ ] TODO

---

### Step 3 — Extend `GET /investigations/alerts/{alert_id}` response
**File:** `backend/server/batch_api.py`  
**Where:** Route handler at ≈ line 2837  
**Change:** After fetching `alert_doc`, call `self.build_investigation_explainability(alert_doc)` and merge the result into the JSON response.  
**New fields added to response:**
- `confidence_pct` — float, 0–100
- `score_series` — list[float], 9 values for the sparkline chart
- `explainability.reasons` — list[str]
- `explainability.top_factors` — list of factor objects
- `explainability.feature_contribution_chart` — `{labels, values}`
- `explainability.threshold_chart` — `{probability, threshold, margin}`
- `explainability.risk_dimension_chart` — `{labels, values}`  
**Status:** [ ] TODO

---

### Step 4 — Update `InvestigationDetailResponse` TypeScript type
**File:** `src/components/pages/InvestigationDetail.tsx`  
**Where:** Interface definition at the top of the file  
**Change:** Add `confidence_pct`, `score_series`, and nested `explainability` to the interface.  
**Status:** [ ] TODO

---

### Step 5 — Wire real data into the InvestigationDetail UI
**File:** `src/components/pages/InvestigationDetail.tsx`  
**Changes:**
1. **Confidence ring** — replace `(detail.fraud_score * 100).toFixed(1)` with `detail.confidence_pct?.toFixed(1) ?? '0.0'`
2. **Feature contributions** — replace `buildFeatureContributions(detail)` call with `detail.explainability?.top_factors` mapped to `{label, value}` pairs; fall back to heuristic if explainability absent
3. **Chart series** — replace hardcoded `chartSeries` array with `detail.score_series ?? [default...]`
4. **Evidence rows** — already consumed from `detail.evidence`; no change needed if backend Step 2/3 improve the evidence builder
5. **Reasons list** — optionally render `detail.explainability?.reasons` in the forensic timeline  
**Status:** [ ] TODO

---

### Step 6 — Model-aware evidence rows (improve `_extract_evidence_rows`)
**File:** `backend/server/batch_api.py`  
**Where:** `_extract_evidence_rows()` helper at ≈ line 2295  
**Change:** Make it model-aware — AE includes `reconstruction_error` vs `threshold` comparison; LSTM includes `fraud_score` vs threshold; SNN includes `fraud_probability` vs `decision_threshold`. Add real numeric deviation where possible.  
**Status:** [ ] TODO

---

### Step 7 — Add `POST /investigations/alerts/false-positive` endpoint
**File:** `backend/server/batch_api.py`  
**Pattern:** Mirror `resolve-fraud` endpoint (≈ line 2911)  
**Action:** Sets `alert_status = 'false_positive'` in `immediate_alerts`, optionally moves to audit log  
**Status:** [ ] TODO

---

### Step 8 — Add `POST /investigations/alerts/escalate` endpoint
**File:** `backend/server/batch_api.py`  
**Pattern:** Mirror `resolve-fraud` endpoint  
**Action:** Sets `alert_status = 'escalated'`, stores `escalated_by`, `escalated_at`, optional `notes`  
**Status:** [ ] TODO

---

### Step 9 — (Optional) Attach lightweight explainability at WebSocket save time
**File:** `backend/server/websocket_unified.py`  
**Why:** So new alerts already have explainability pre-computed in MongoDB; Step 2 would then just read it instead of computing on the fly.  
**Change:**
- Modify `_build_snn_feature_vector()` to return 3-tuple `(cc_num, feature_vector, feature_map)`
- Compute lightweight explainability in each `detect_fraud_*()` method
- Attach to `db_payload` before `insert_immediate_alert()` call  
**Status:** [ ] TODO (deferred — on-demand compute in Step 2 is sufficient for now)

---

## Execution Order

```
Step 1  →  Step 2  →  Step 3  →  Step 4  →  Step 5
                ↓
           Step 6 (can be done in parallel with Step 2)
                ↓
           Step 7  →  Step 8
                ↓
           Step 9 (optional — improves future alerts)
```

## Files Touched

| File | Steps |
|------|-------|
| `backend/server/batch_api.py` | 1, 2, 3, 6, 7, 8 |
| `src/components/pages/InvestigationDetail.tsx` | 4, 5 |
| `backend/server/websocket_unified.py` | 9 (optional) |

## Definition of Done

- [x] Investigation detail page shows real confidence % (not 0.0%)
- [x] Feature contributions show model-derived factor names and percentages
- [x] Chart sparkline reflects real score trajectory (model-grounded score_series)
- [x] Evidence rows show numeric deviations for the relevant model metric
- [x] "Mark as False Positive" button calls backend successfully
- [x] "Escalate" button calls backend successfully
- [x] No TypeScript errors in InvestigationDetail.tsx
- [x] No Python errors in batch_api.py
