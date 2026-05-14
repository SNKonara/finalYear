# NeuroDetect Unified Testing Documentation

Last updated: 2026-05-13

This document consolidates all testing plans, checklists, and model-specific validation notes previously split across multiple files in the `testing/` folder.

## 1. Objectives

1. Validate critical backend and frontend flows after code changes.
2. Validate model reliability for Autoencoder, LSTM, and SNN.
3. Detect data drift and model performance anomalies over time.
4. Investigate suspicious model performance metrics (for example, LSTM 99.5% config accuracy).
5. Track execution outcomes in repeatable result logs.

## 2. Test Scope

- Backend API health and route behavior
- Model load and inference correctness
- Batch processing and threshold behavior
- Cross-day consistency and performance stability
- LSTM sequence sensitivity and score calibration
- Data drift detection via fraud-rate trends
- WebSocket streaming and model switching
- Frontend workflow regressions (especially batch upload)

## 3. Test Hierarchy and Runtime Order

### 3.1 Quick Gate (Always Run First, ~5 min)

- Health check and model metadata check
- Basic smoke validation
- Batch page load check

### 3.2 Core Model Validation (Per Change, ~15 min)

- Cross-model contract tests for Autoencoder, LSTM, SNN
- Threshold behavior checks
- Edge-case input handling

### 3.3 Cross-Day Validation (Daily During Active Development, ~30 min)

- Baseline consistency (3 runs per model)
- Threshold sweep monotonicity
- LSTM sequence sensitivity and score calibration
- Data-drift indicators from fraud-rate and score distribution trends

### 3.4 Integration and Frontend Regression (Per Feature)

- API endpoint contracts and artifact downloads
- Frontend route, navigation, theme, and workflow regressions
- WebSocket stability and model switching behavior

## 4. Primary Commands

```bash
# Frontend
npm run dev

# Backend (all servers)
python backend/start_all_servers.py

# Backend (individual)
cd backend/server && python batch_api.py
cd backend/server && python websocket_unified.py

# API smoke test
python backend/scripts/smoke_api.py --base-url http://127.0.0.1:8000 --all-models

# Cross-day validation
python backend/test_cross_day_validation.py

# Optional: apply runtime threshold recommendations (non-persistent)
python backend/test_cross_day_validation.py --apply-recommendations

# Optional: persist recommended thresholds
python backend/test_cross_day_validation.py --apply-recommendations --persist-recommendations

# Build + lint
npm run build
npm run lint
```

## 5. Environment and Preconditions

- Python environment active and dependencies installed
- Frontend dependencies installed (`npm install`)
- Backend running on port 8000
- Frontend app starts and renders
- MongoDB reachable for history and investigation routes
- Test data available:
  - `backend/dataset/fraudTest_sample20.csv`
  - `backend/dataset/fraudTest_mixed40.csv`

Additional data to prepare as needed:
- Empty or invalid-schema CSV
- CSV missing required fields
- CSV with extreme numeric values
- CSV with category/gender edge cases
- Larger samples (500-1000 rows) for drift and scalability checks

## 6. Quick Execution Checklist

Date baseline: 2026-04-26  
Tester baseline: Automated Cross-Day Validation  
Environment baseline: Local Windows, Python 3.x, CUDA enabled, MongoDB

### 6.1 Pre-Run Setup

- [x] Python dependencies ready
- [x] Frontend dependencies installed
- [x] Backend starts without startup exceptions
- [x] Frontend starts and loads

### 6.2 Quick Gate

- [x] `GET /health` returns 200 with all models loaded
- [x] `GET /models` returns model metadata
- [x] Smoke script previously tested
- [x] Batch upload page loads without UI break

### 6.3 Model Test Execution (Baseline State)

- [x] Cross-day validation completed - PASS
- [x] Autoencoder tests: Stable, lower sensitivity vs SNN
- [x] LSTM tests: Production-usable with caveats
- [x] SNN tests: Improved after threshold clamping

### 6.4 Integration and Frontend (Track Per Release)

- [ ] API + batch integration tests completed
- [ ] WebSocket model-switching tests completed
- [ ] Frontend regression checks completed

### 6.5 Exit Criteria

- [x] No unresolved P0/P1 defects from model testing
- [x] Known issues documented
- [x] Result reports generated and stored

## 7. Cross-Model Core API Cases

| ID | Scenario | Steps | Expected Result |
|---|---|---|---|
| M-01 | Model loaded in metadata | `GET /models` | Model appears with `loaded: true` and threshold present |
| M-02 | Single model self-test | `POST /models/test` with each model | 200 response with `prediction` and `fraud_score` |
| M-03 | Batch inference success | `POST /batch/process` with sample data | `success: true`, stats and files returned |
| M-04 | Threshold update (non-persistent) | `POST /batch/threshold` with `persist=false` | 200 response, updated threshold returned |
| M-05 | Threshold sensitivity | Run same file with low vs high threshold | Fraud count changes in expected direction |
| M-06 | Invalid model type | Send unsupported model | 4xx with clear validation error |
| M-07 | Invalid CSV schema | Upload malformed CSV | Error is handled, no server crash |
| M-08 | Output contract consistency | Inspect JSON response keys | Required fields always present |

## 8. Model-Specific Validation

### 8.1 Autoencoder

Scope: Validate reconstruction-error pipeline for metadata and batch endpoints.

Test cases:

| ID | Test | Procedure | Expected |
|---|---|---|---|
| AE-01 | Metadata availability | `GET /models` | `autoencoder.loaded == true` |
| AE-02 | Self-test endpoint | `POST /models/test` with model_type=autoencoder | 200 plus valid `prediction`, `fraud_score`, `threshold` |
| AE-03 | Batch happy path | Upload sample CSV via `/batch/process` | `success: true`, stats populated |
| AE-04 | Very low threshold | Update threshold low, rerun batch | Higher fraud-count tendency |
| AE-05 | High threshold | Update threshold high, rerun batch | Lower fraud-count tendency |
| AE-06 | Missing columns | Upload CSV missing required field | Graceful validation/processing error |
| AE-07 | Numeric edge values | Extreme amount and geo values | No crash, deterministic response structure |

Observability:
- `statistics.threshold`
- `statistics.fraud_count`
- Average `fraud_score`
- Any preprocessing warnings/errors

Pass criteria:
- No unhandled exceptions
- Response contract preserved
- Threshold tuning affects output predictably
- `statistics.labeled_metrics` present when `is_fraud` exists

### 8.2 LSTM

Scope: Validate sequence-aware behavior, score calibration, and stability.

Known concern:
- Config reports suspiciously high performance (99.5% accuracy, 99.8% recall), which may reflect training-data bias or overfitting.

Core and diagnostic tests:

| ID | Test | Procedure | Expected |
|---|---|---|---|
| LS-01 | Metadata availability | `GET /models` | `lstm.loaded == true` with sequence metadata |
| LS-02 | Self-test endpoint | `POST /models/test` with model_type=lstm | 200 with `fraud_score` in [0, 1] |
| LS-03 | Batch happy path | `/batch/process` with sample data | Success with populated stats |
| LS-04 | Throughput spot check | 3 consecutive batch requests | Stable latency |
| LS-05 | Threshold sweep | Low/medium/high threshold reruns | Fraud count monotonic with threshold |
| LS-06 | Small file handling | 1-2 row sample | No sequence-processing crash |
| LS-07 | Feature mismatch | Wrong/partial schema upload | Graceful failure |
| LS-08 | Score calibration | Repeat `/models/test` 10x | Scores in [0, 1], not clustered at extremes |
| LS-09 | Fraud-rate consistency | Run same file 3x | Fraud count within small variance |
| LS-10 | Sequence order sensitivity | Shuffle/sort same data and rerun | Same fraud_count and similar average score |
| LS-11 | Threshold monotonicity | Sweep threshold 0.3 to 0.9 | Fraud count decreases or stays the same |
| LS-12 | Score distribution | Analyze histogram | Useful separation of likely fraud vs legit |
| LS-13 | Field accuracy vs config | Compare runtime to config metrics | Document gap and probable cause |

Red flags:
- High run-to-run variance on identical inputs
- Non-monotonic threshold behavior
- Order-dependent predictions on equivalent data
- Scores collapsing to only 0 or 1

### 8.3 SNN

Scope: Validate customer-personalized scoring path, threshold handling, and payload safety.

Test cases:

| ID | Test | Procedure | Expected |
|---|---|---|---|
| SN-01 | Metadata availability | `GET /models` | `snn.loaded == true` with expected metadata |
| SN-02 | Self-test endpoint | `POST /models/test` with model_type=snn | 200 with valid prediction and score |
| SN-03 | Batch happy path | `/batch/process` with sample data | Success with stats |
| SN-04 | Threshold update | `/batch/threshold` with persist=false | Threshold reflected in run |
| SN-05 | Personalized behavior | Run mixed-customer sample | No runtime failures, contract intact |
| SN-06 | Explainability payload | Inspect response fields | JSON-safe, stable structure |
| SN-07 | Threshold extremes | Near-zero and near-one threshold tests | No invalid math or overflow |

Risk-focused checks:
- Customer profile edge-case robustness
- Consistent risk-level generation
- Score bounds and numeric safety

Pass criteria:
- Stable endpoints under realistic and edge inputs
- Predictable threshold behavior (clamped to [0, 0.95])
- No schema-breaking regressions
- Non-zero recall on mixed-class test sets

## 9. API and Integration Plan

Priority endpoints:
- `GET /health`
- `GET /models`
- `POST /models/test`
- `POST /batch/process`
- `POST /batch/threshold`
- `GET /investigations/alerts`
- `GET /batch/history`
- `GET /batch/download/{batch_id}?format=csv|json|pdf`

Integration cases:

| ID | Scenario | Expected |
|---|---|---|
| API-01 | Health check | 200 and loaded models visible |
| API-02 | Batch process success | Success payload includes statistics and files |
| API-03 | Download artifacts | CSV, JSON, PDF downloads work for valid batch ID |
| API-04 | Threshold update non-persistent | Threshold updates for active run |
| API-05 | Threshold update persistent | Override survives restart |
| API-06 | Alerts route contract | Returns `alerts` array with expected fields |
| API-07 | Invalid inputs | Clear error response without server crash |

Frontend contract checks:
- Batch upload results table parses numeric fields safely
- Real-time model pages parse/display numeric fields safely
- Investigations list/detail pages remain stable
- Display formatting handles string-versus-number conversions safely

## 10. Frontend Regression Plan

Core routes:
- `/`
- `/lstmreal`
- `/snnreal`
- `/batch-upload`
- `/investigations`
- `/reports`
- `/user-management`

Checklist:

### 10.1 Navigation and Layout

- [ ] Single top navbar visible (no duplicate headers)
- [ ] Sidebar navigation works on all pages
- [ ] Role-based route access remains correct

### 10.2 Theme Behavior

- [ ] Dark/light mode switch updates page surfaces correctly
- [ ] Batch upload layout matches active theme
- [ ] No hardcoded dark-only artifacts in light mode

### 10.3 Batch Upload Workflow

- [ ] Model cards selectable
- [ ] File drag/drop and browse upload works
- [ ] Process action triggers request and shows progress
- [ ] Results tab and table render correctly
- [ ] Threshold tuning can rerun processing

### 10.4 Real-Time Dashboards

- [ ] WebSocket connects and stream appears
- [ ] Model switching works without UI break
- [ ] Stats and graphs update without runtime errors

### 10.5 Authentication and Form Validation Test Table

Use this table for manual UI testing focused on login, password validation, and button behavior.

| ID | Area | Scenario | Steps | Expected Result |
|---|---|---|---|---|
| FE-AUTH-01 | Login | Login button disabled on empty form | Open login page, leave email/password empty | Login button remains disabled or shows validation messages; no request is sent |
| FE-AUTH-02 | Login | Invalid email format | Enter invalid email (for example `user@`), enter any password, click login | Inline email validation appears; login does not proceed |
| FE-AUTH-03 | Login | Short/invalid password format | Enter valid email + short password violating rules, click login | Password validation message shown; submit blocked |
| FE-AUTH-04 | Login | Wrong credentials | Enter unregistered/incorrect credentials, click login | Error toast/message shown; user stays on login page; no crash |
| FE-AUTH-05 | Login | Successful login | Enter valid credentials and click login | Redirect to authorized landing page; auth token/session stored correctly |
| FE-AUTH-06 | Login | Multiple rapid clicks on login button | Fill valid form and click login repeatedly quickly | Only one request is processed (or duplicates safely handled); button shows loading/disabled state |
| FE-AUTH-07 | Login | Enter key submits form | Fill valid login form and press Enter | Same behavior as clicking login button |
| FE-AUTH-08 | Password Field | Show/Hide password toggle | Type password, toggle visibility icon/button | Password text visibility toggles correctly; value preserved |
| FE-AUTH-09 | Password Validation | Missing uppercase/lowercase/number/special char | Enter passwords missing each rule one by one | Rule-specific feedback appears accurately |
| FE-AUTH-10 | Password Validation | Minimum length boundary | Test exactly min-1 and min length characters | Min-1 fails with message; min length passes validation |
| FE-AUTH-11 | Password Validation | Trim and whitespace behavior | Enter password with leading/trailing spaces | Behavior matches spec (either preserved intentionally or trimmed consistently) |
| FE-AUTH-12 | Forgot Password | Forgot-password button/link opens flow | Click forgot-password link/button from login page | User is routed to/reset flow opens without UI errors |
| FE-AUTH-13 | Reset Password | Mismatched confirm password | Enter new password + different confirm password, submit | Clear mismatch validation; submit blocked |
| FE-AUTH-14 | Reset Password | Successful password reset | Enter compliant matching passwords and submit valid reset token | Success message shown; user can log in with new password |
| FE-AUTH-15 | Session | Logout button behavior | Log in, click logout button/menu action | Session/token cleared; user redirected to login |
| FE-AUTH-16 | Session Guard | Protected route without auth | Open protected route URL directly in new tab while logged out | Redirect to login or unauthorized page |
| FE-AUTH-17 | Role Access | Non-admin opening admin route/button | Log in as non-admin; attempt admin navigation/button | Button hidden/disabled or access denied safely |
| FE-AUTH-18 | Error Handling | Backend unavailable during login | Stop auth/backend service and submit login | User-friendly network/server error shown; app remains usable |
| FE-AUTH-19 | Accessibility | Keyboard navigation for auth controls | Navigate login form with Tab/Shift+Tab/Enter/Space | Logical focus order; all primary controls reachable and actionable |
| FE-AUTH-20 | Accessibility | Screen-reader labels on form controls | Inspect email, password, show/hide, login, forgot-password controls | Inputs/buttons have clear accessible labels and error announcements |

Execution notes:
- Record PASS/FAIL/BLOCKED and evidence (screenshot, response payload, console errors) for each ID.
- If a case fails, link it to a defect ticket and include severity.
- Re-run FE-AUTH-01 through FE-AUTH-08 after any auth UI change.

### 10.6 Full System Function Testing Matrix

This matrix extends coverage beyond auth and validates frontend features, backend APIs, and end-to-end user workflows.

#### 10.6.1 Frontend Function Coverage

| ID | Route/Page | Buttons and Functions to Test | Steps | Expected Result |
|---|---|---|---|---|
| FE-SYS-01 | `/login` | Login submit, show/hide password, forgot password link | Enter valid and invalid inputs, click all actions | Valid submit logs in; invalid inputs blocked with messages; controls responsive |
| FE-SYS-02 | `/` or `/snnreal` | Dashboard filters/cards/navigation actions | Open dashboard, click model cards, switch views | Charts/cards update correctly; no blank panel or crash |
| FE-SYS-03 | `/streaming` | Start/stop stream, model switch, live indicators | Start stream, switch model, stop stream | Live updates arrive, switch works, stop halts stream cleanly |
| FE-SYS-04 | `/system` | Refresh/system summary cards and links | Load page and trigger refresh actions | KPI values load without JS errors; degraded states shown clearly |
| FE-SYS-05 | `/snn-alerts` | Alert list actions, detail open, dismiss actions | Open alerts list and perform available actions | Alert states update and reflect backend response |
| FE-SYS-06 | `/batch-upload` | Model selection, upload button, process button, threshold rerun | Upload valid CSV, process, tune threshold, rerun | Processing completes, result tabs populate, rerun reflects new threshold |
| FE-SYS-07 | `/investigations` | Filter/search/sort/open detail/escalate controls | Use filters and open an alert | List updates correctly and selected alert opens |
| FE-SYS-08 | `/investigations/:alertId` | Resolve fraud, false positive, escalate, back navigation | Perform each action on sample alert | Action persists and status changes correctly in list/detail |
| FE-SYS-09 | `/reports` and `/summary` | Report filters, open report, download report | Select report, open details, download PDF | Report data loads; download succeeds |
| FE-SYS-10 | `/audit` | Audit cards/tables/charts rendering | Open audit detail from reports | Metrics render correctly with no NaN/formatting issues |
| FE-SYS-11 | `/user-management` (admin) | Create user, update role, delete user | Run full CRUD as admin | CRUD operations succeed with proper confirmation/errors |
| FE-SYS-12 | `/profile` | Update profile fields and save/logout | Edit fields and save; logout | Profile updates persist; logout clears session |
| FE-SYS-13 | `/unauthorized` | Back/navigation buttons | Attempt restricted route as wrong role | User lands on unauthorized view with safe navigation |
| FE-SYS-14 | Global Layout | Sidebar links, topbar actions, theme switch | Navigate all links and toggle theme | Active route highlights, theme applies consistently across pages |

#### 10.6.2 Backend API Function Coverage

| ID | Endpoint Group | Scenario | Request | Expected Result |
|---|---|---|---|---|
| BE-SYS-01 | Core API | Service root and health | `GET /`, `GET /health` | 200, service metadata present, model/auth health visible |
| BE-SYS-02 | Models | List models | `GET /models` | All loaded models include threshold/config/performance fields |
| BE-SYS-03 | Models | Self-test each model | `POST /models/test` | 200 with prediction and fraud_score |
| BE-SYS-04 | Models | Reload each model | `POST /models/reload` | 200 and model remains available afterward |
| BE-SYS-05 | Auth | Login valid/invalid/throttled | `POST /auth/login` | Valid returns token; invalid gives 401; throttled gives 429 |
| BE-SYS-06 | Auth | Session identity and logout | `GET /auth/me`, `POST /auth/logout` | Authenticated me returns user; logout revokes session |
| BE-SYS-07 | Auth Admin | User lifecycle and role update | `GET /auth/users`, `POST /auth/users`, `PATCH /auth/users/{id}/role`, `DELETE /auth/users/{id}` | Admin-only access enforced; CRUD/role changes behave correctly |
| BE-SYS-08 | Batch | Process valid CSV | `POST /batch/process` | 200, stats/results returned, persisted summary available |
| BE-SYS-09 | Batch | Invalid schema and bad model handling | `POST /batch/process` invalid payload | Clear 4xx/5xx error without server crash |
| BE-SYS-10 | Batch | Threshold update (persist and non-persist) | `POST /batch/threshold` | Threshold changes reflected; persistence behavior matches flag |
| BE-SYS-11 | Alerts | Batch alert summaries and high-risk list | `GET /alerts/summary`, `GET /alerts/high-risk` | Correct aggregation, risk distribution, and explainability payload |
| BE-SYS-12 | Alerts | Explainability lookup | `GET /alerts/explain/{batch_id}/{transaction_id}` | Returns explainability for valid transaction, 404 for invalid |
| BE-SYS-13 | Live Alerts | Live alert queue and dismiss | `GET /alerts/live/high-risk`, `POST /alerts/live/dismiss` | Queue reflects status filter; dismiss updates status |
| BE-SYS-14 | Investigations | List and detail | `GET /investigations/alerts`, `GET /investigations/alerts/{alert_id}` | List/detail contracts are stable and complete |
| BE-SYS-15 | Investigations | Workflow actions | `POST /investigations/alerts/resolve-fraud`, `.../false-positive`, `.../escalate` | State transitions persist and return success payload |
| BE-SYS-16 | Reports | List and fetch report | `GET /reports`, `GET /reports/{report_id}` | Report payload returned from model reports or fallback source |
| BE-SYS-17 | Reports | Download/generate PDF | `GET /reports/{report_id}/download` | Existing PDF served or generated on demand |
| BE-SYS-18 | Reports Admin | Trigger hourly report generation | `POST /reports/trigger-hourly` | Admin-only endpoint works and returns per-hour results |
| BE-SYS-19 | System | System overview dashboard payload | `GET /system/overview` | KPI/services/models payload complete and internally consistent |

#### 10.6.3 End-to-End Workflow Coverage

| ID | Workflow | Steps | Expected Result |
|---|---|---|---|
| E2E-01 | Login to Dashboard | Login as analyst/admin and open dashboard | Session created, protected routes accessible by role |
| E2E-02 | Batch Fraud Analysis | Upload CSV, process batch, review results, adjust threshold, rerun | Fraud metrics update and are reflected in history/reports |
| E2E-03 | Alert Investigation Lifecycle | Generate/open alert, view detail, escalate/resolve/mark false positive | Alert status transitions persist across list/detail pages |
| E2E-04 | Reporting Lifecycle | Process batch, open reports, inspect audit detail, download PDF | Report and audit views align with batch statistics |
| E2E-05 | Admin User Management | Admin creates analyst, analyst logs in, admin changes role, user loses/gains access | Role-based route and API permissions enforced correctly |
| E2E-06 | Live Streaming Monitoring | Start streaming, detect high-risk transaction, verify it appears in investigation queue | Streaming output and investigations data stay synchronized |
| E2E-07 | Session Security | Login, logout, then retry protected APIs/pages with old token | Access denied after logout/session expiration |

#### 10.6.4 Full System Exit Criteria

- All `FE-SYS-*`, `BE-SYS-*`, and `E2E-*` tests are executed and documented.
- No P0/P1 defects remain open.
- No frontend runtime crashes on protected or public routes.
- No backend endpoint returns unexpected 5xx under valid input scenarios.
- Role-based access control behaves correctly for admin, analyst, and viewer roles.
- Generated reports and downloaded artifacts are readable and internally consistent.

Pass criteria:
- No React runtime crashes
- No blank screens on critical routes
- All primary actions complete with expected UI feedback

## 11. Cross-Day Validation and Drift Strategy

### 11.1 Key Questions

1. Is LSTM overfitted to historical training patterns?
2. Do models produce stable fraud rates across repeated runs?
3. Are feature distributions drifting over time (amount, geo, time, category)?

### 11.2 Phased Execution

Phase 1: Baseline establishment
- Validate `GET /models` metadata completeness
- Run each model 3 times on the same baseline file
- Execute threshold sweep for monotonicity checks

Phase 2: Cross-day comparison (Days 1-7)
- Repeat baseline daily
- Track variance in fraud_count and average fraud_score
- Compare behavior on larger or newer samples where available

Phase 3: LSTM diagnostics
- Sequence sensitivity testing using reordered datasets
- Score calibration and distribution checks
- Single-feature perturbation tests to detect fragile dependence

Phase 4: Drift detection
- Daily score histograms by model
- Threshold effectiveness trend tracking (precision, recall, FPR, FNR)

Acceptance target:
- Fraud-count variance standard deviation < 15% of mean for stable baselines

## 12. Baseline Runtime Results (2026-04-26)

Dataset: `fraudTest_mixed40.csv` (40 rows: 20 fraud + 20 legit)

| Model | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Autoencoder | 77.5% | 86.67% | 65.0% | 74.29% | 13 | 2 | 18 | 7 | Good balance |
| LSTM | 50.0% | 50.0% | 30.0% | 37.5% | 6 | 6 | 14 | 14 | Distribution shift suspected |
| SNN | 85.0% | 100.0% | 70.0% | 82.35% | 14 | 0 | 20 | 6 | Best after threshold fix |

Summary findings:
- SNN threshold clamping fix resolved zero-recall behavior.
- SNN achieved strongest performance on mixed40 in this baseline.
- LSTM showed large config-vs-runtime gap, likely due to data distribution shift.
- All three models remained operational for batch and metadata flows.

## 13. Current Pass Criteria Summary

- No critical startup errors
- `/health` reports all models loaded
- Smoke testing passes for all model types
- Threshold sweeps are monotonic
- Batch uploads succeed for representative sample files
- Cross-day baseline variance remains within tolerance
- Frontend critical routes and workflows are regression-free

## 14. Known Issues and Deferred Items

- LSTM runtime performance materially below training-config metrics on mixed40
- SNN customer-profile generation may timeout on very large datasets
- Frontend lint warnings exist and should be tracked (non-blocking)

## 15. Deliverables

1. Daily execution logs with timestamp and model-level metrics
2. 7-day summary report with variance and drift analysis
3. LSTM diagnostic audit (sequence sensitivity, calibration, perturbation)
4. Recommendations for retraining, threshold tuning, and additional data collection

## 16. Deployment Validation Gate

Before production rollout, require:
- `GET /health` confirms all models loaded
- `GET /models` returns complete metadata for autoencoder, lstm, and snn
- Cross-day validation reports PASS on current baseline
- No unresolved P0/P1 defects

Post-deploy checks:
- API contract sanity checks
- Batch processing sanity run
- WebSocket stream/model switch validation

## 17. Results Location

- `testing/results/` for markdown run logs
- `results/` and `results/batch/` for generated CSV/JSON artifacts
