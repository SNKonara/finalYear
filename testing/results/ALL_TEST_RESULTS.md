# NeuroDetect Consolidated Test Results

Last updated: 2026-05-13

This document consolidates all currently available test result artifacts under `testing/results` into a single report.

## 1. Source Files Included

- `smoke_results.md`
- `autoencoder_results.md`
- `lstm_results.md`
- `snn_results.md`
- `integration_results.md`
- `cross_day_validation_report_20260426.md`
- `cross_day_validation_20260426_100429.json`
- `cross_day_validation_20260426_100537.json`
- `cross_day_validation_20260426_101516.json`
- `cross_day_validation_20260426_101623.json`
- `cross_day_validation_20260426_101742.json`

## 2. Overall Snapshot

| Area | Status | Key Notes |
|---|---|---|
| Smoke/API sanity | PASS | Critical routes responded 200 in smoke scope |
| Autoencoder model suite | PASS with 1 BLOCKED case | Numeric edge dataset case pending |
| LSTM model suite | PASS with 1 BLOCKED case | Small-file sequence case pending |
| SNN model suite | PASS with 2 BLOCKED cases | Personalized dataset + deep explainability checks pending |
| API + Frontend integration | PASS with follow-up risk | Lint gate failed (31 issues), route walkthrough partial |
| Cross-day validation (main report) | PASS | LSTM stable, AE/SNN threshold tuning recommended |

## 3. Smoke Test Results

Run info:
- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)
- Backend URL: `http://127.0.0.1:8000`
- Command: `python backend/scripts/smoke_api.py --base-url http://127.0.0.1:8000 --all-models`

| Endpoint | Status | Notes |
|---|---|---|
| GET /investigations/alerts | PASS | 200 OK, alerts=1 |
| POST /batch/threshold (autoencoder) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (autoencoder) | PASS | 200 OK, total=20, fraud_count=0 |
| POST /batch/threshold (lstm) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (lstm) | PASS | 200 OK, total=20, fraud_count=6 |
| POST /batch/threshold (snn) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (snn) | PASS | 200 OK, total=20, fraud_count=2 |

Summary:
- Overall: PASS
- Blocking issues: None in smoke scope

## 4. Model-Specific Result Logs

### 4.1 Autoencoder Results

Run info:
- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)

| Case ID | Result | Evidence | Notes |
|---|---|---|---|
| AE-01 | PASS | GET /models -> 200, autoencoder present | Metadata available and loaded |
| AE-02 | PASS | POST /models/test (autoencoder) -> 200 | success=true |
| AE-03 | PASS | POST /batch/process threshold 0.0005 -> 200 | fraud_count=1 on sample20 |
| AE-04 | PASS | POST /batch/process threshold 0.0002 -> 200 | Lower threshold increased fraud_count to 6 |
| AE-05 | PASS | POST /batch/process threshold 0.001 -> 200 | Higher threshold reduced fraud_count to 1 |
| AE-06 | PASS | Malformed CSV via LSTM path -> 500 with missing-columns detail | Error surfaced clearly; no crash observed |
| AE-07 | BLOCKED | Numeric edge-value CSV not executed | Schedule next cycle |

Metrics:
- Baseline threshold: 0.0005
- Fraud count baseline: 1
- Low threshold count: 6
- High threshold count: 1

### 4.2 LSTM Results

Run info:
- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)

| Case ID | Result | Evidence | Notes |
|---|---|---|---|
| LS-01 | PASS | GET /models -> 200, lstm present | Metadata available |
| LS-02 | PASS | POST /models/test (lstm) -> 200 | success=true |
| LS-03 | PASS | POST /batch/process threshold 0.5 -> 200 | fraud_count=6 |
| LS-04 | PASS | 3 threshold runs all 200 | No timeout/crash in sweep |
| LS-05 | PASS | Threshold sweep: 0.15->15, 0.5->6, 0.85->1 | Monotonic sensitivity observed |
| LS-06 | BLOCKED | 1-2 row sequence file not executed separately | Add targeted small-file test next cycle |
| LS-07 | PASS | Malformed CSV -> 500 with missing-columns detail | Clear validation-style detail |

Metrics:
- Baseline threshold: 0.5
- Fraud counts: 15 (t=0.15), 6 (t=0.5), 1 (t=0.85)

### 4.3 SNN Results

Run info:
- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)

| Case ID | Result | Evidence | Notes |
|---|---|---|---|
| SN-01 | PASS | GET /models -> 200, snn present | Metadata available |
| SN-02 | PASS | POST /models/test (snn) -> 200 | success=true |
| SN-03 | PASS | POST /batch/process threshold 0.5 -> 200 | fraud_count=2 |
| SN-04 | PASS | Threshold sweep successful | 0.15->12, 0.5->2, 0.85->0 |
| SN-05 | BLOCKED | Personalized/mixed-customer dataset not executed | Needs customer-profile test dataset |
| SN-06 | BLOCKED | Explainability payload deep validation not executed | Add explicit schema assertions next cycle |
| SN-07 | PASS | Threshold extremes stable outputs | No crash/overflow |

Metrics:
- Baseline threshold: 0.5
- Fraud counts: low(0.15)=12, baseline(0.5)=2, high(0.85)=0

## 5. API + Frontend Integration Results

Run info:
- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)

### 5.1 API Cases

| Case ID | Result | Evidence | Notes |
|---|---|---|---|
| API-01 | PASS | GET /health and GET /models returned 200 | All 3 models visible |
| API-02 | PASS | POST /batch/process passed for all models with sample20 | 200 responses across sweeps |
| API-03 | PASS | /batch/download/{batch_id}?format=json|csv|pdf returned 200 | Content-Types verified |
| API-04 | PASS | Threshold tuning affected fraud counts | Sensitivity behaved as expected |
| API-05 | PASS | POST /batch/threshold persist=true returned 200 | `saved_models/threshold_overrides.json` updated (lstm: 0.4321) |
| API-06 | PASS | Invalid model threshold request returned 400 | Validation path works |
| API-07 | PASS | Malformed CSV returned handled error payload | Missing-columns detail present |

### 5.2 Frontend Regression Snapshot

- Build status: PASS (`npm run build`)
- Lint status: FAIL (`npm run lint` -> 31 issues: 25 errors, 6 warnings)
- Frontend entry route availability: PASS (`http://localhost:5173/` returned 200)
- Backend routes used by frontend: PASS (`/models`, `/batch/history` returned 200)
- Reports page UX update: PASS
- Real-time pages: Revalidation pending (not fully exercised end-to-end)

### 5.3 Follow-up

- Open defect/risk: Frontend lint gate failing (31 issues)
- Suggested next cycle: Fix lint issues, then perform screenshot-backed full route walkthrough

## 6. Cross-Day Validation Results

### 6.1 Main Cross-Day Report

Source: `cross_day_validation_report_20260426.md`

Summary from report:
- Overall Status: PASS
- LSTM: Stable and deterministic across repeated runs
- Autoencoder: Low sensitivity on sample20; threshold tuning suggested
- SNN: Default threshold too conservative; lower-threshold tuning suggested
- Threshold monotonicity: PASS across tested models
- Sequence sensitivity test: Skipped (requires shuffled/sorted variants)

### 6.2 Cross-Day JSON Runs

| File | test_execution | overall_status |
|---|---|---|
| `cross_day_validation_20260426_100429.json` | 2026-04-26T04:34:29.754022 | FAIL |
| `cross_day_validation_20260426_100537.json` | 2026-04-26T04:35:41.548151 | PASS |
| `cross_day_validation_20260426_101516.json` | 2026-04-26T04:45:17.915247 | PASS |
| `cross_day_validation_20260426_101623.json` | 2026-04-26T04:46:26.579081 | PASS |
| `cross_day_validation_20260426_101742.json` | 2026-04-26T04:47:45.423493 | PASS |

Interpretation:
- Earliest run (`100429`) failed during setup/format issues.
- Subsequent runs converged to PASS and show stabilized validation execution.

## 7. Outstanding Gaps Across All Results

- AE-07: Autoencoder numeric edge-case dataset not executed
- LS-06: LSTM 1-2 row sequence test not executed
- SN-05: SNN personalized/mixed-customer dataset not executed
- SN-06: SNN deep explainability payload schema validation not executed
- Frontend lint quality gate currently failing
- Full screenshot-backed end-to-end frontend route walkthrough still pending

## 8. Final Consolidated Status

Current overall testing posture: **PASS with known gaps**.

- Core backend/API flows: Passing
- Model inference and threshold behaviors: Passing with tuning recommendations for AE/SNN
- Integration: Passing with frontend lint debt
- Cross-day validation: Passing on latest runs
- Remaining work: Close blocked cases and frontend lint issues before hard release gate
