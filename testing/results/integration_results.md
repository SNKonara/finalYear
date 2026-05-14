# API + Frontend Integration Results

## Run Info

- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)
- Branch/Commit: Not captured in this run

## API Cases

| Case ID | Result (PASS/FAIL/BLOCKED) | Evidence | Notes |
|---|---|---|---|
| API-01 | PASS | GET /health and GET /models returned 200 in test cycle | All 3 models visible |
| API-02 | PASS | POST /batch/process passed for all models with sample20 | 200 responses across sweeps |
| API-03 | PASS | /batch/download/{batch_id}?format=json|csv|pdf returned 200 for all formats | Content-Types verified: application/json, text/csv, application/pdf |
| API-04 | PASS | Threshold tuning affected fraud counts across all models | Sensitivity behaved as expected |
| API-05 | PASS | POST /batch/threshold persist=true returned 200; lstm threshold reflected in GET /models and overrides file | `C:\finalYear\saved_models\threshold_overrides.json` updated with `lstm: 0.4321` |
| API-06 | PASS | Invalid model threshold request returned 400 with explicit detail | Validation path works |
| API-07 | PASS | Malformed CSV produced handled error payload (status 500 with detail) | Error detail includes missing required columns |

## Frontend Regression Snapshot

- Build status: PASS (`npm run build` completed; dist assets generated).
- Lint status: FAIL (`npm run lint` -> 31 issues: 25 errors, 6 warnings).
- Frontend entry route availability: PASS (`http://localhost:5173/` returned 200, text/html).
- Backend routes used by frontend: PASS (`/models` and `/batch/history` returned 200).
- Reports page UX update: PASS (reduced crowding with responsive 2/3-pane layout, improved spacing, compact title scale, mobile report selector fallback).
- Batch upload workflow: Previously validated manually; no regression observed after recent UI fixes.
- Theme switching: Previously validated for batch/realtime styles; no new issue in this cycle.
- Real-time pages: UI revalidation still pending (not exercised end-to-end in browser automation).

## Defects and Follow-up

- Open defects: Frontend lint gate currently failing (31 issues) and should be triaged before release hardening.
- Risk level summary: Moderate. Runtime/build checks pass, but lint quality gate is red and full UI route walkthrough evidence remains partial.
- Next test cycle focus: Fix lint errors, then perform screenshot-backed route-by-route frontend walkthrough.

## Storage Mode Note

- Batch processing persistence is now mongodb-only in current backend behavior.
- Local JSON/CSV/PDF result file writes during `/batch/process` are disabled.
