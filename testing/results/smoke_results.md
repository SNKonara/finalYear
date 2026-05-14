# Smoke Test Results

## Run Info

- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)
- Branch/Commit: Not captured in this run
- Backend URL: `http://127.0.0.1:8000`
- Command:

```bash
python backend/scripts/smoke_api.py --base-url http://127.0.0.1:8000 --all-models
```

## Endpoint Results

| Endpoint | Status | Notes |
|---|---|---|
| GET /investigations/alerts | PASS | 200 OK, alerts=1 |
| POST /batch/threshold (autoencoder) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (autoencoder) | PASS | 200 OK, total=20, fraud_count=0 |
| POST /batch/threshold (lstm) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (lstm) | PASS | 200 OK, total=20, fraud_count=6 |
| POST /batch/threshold (snn) | PASS | 200 OK, threshold=0.5 |
| POST /batch/process (snn) | PASS | 200 OK, total=20, fraud_count=2 |

## Summary

- Overall: PASS / FAIL
- Overall: PASS
- Blocking issues: None observed in smoke scope.
- Follow-up actions: Run detailed model suites (AE/LSTM/SNN) and log results in per-model files.
