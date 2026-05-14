# Autoencoder Test Results

## Run Info

- Date: 2026-04-22

## Case Log

| Case ID | Result (PASS/FAIL/BLOCKED) | Evidence | Notes |
|---|---|---|---|
| AE-01 | PASS | GET /models -> 200, autoencoder present | Metadata available and loaded |
| AE-02 | PASS | POST /models/test (autoencoder) -> 200 | success=true |
| AE-03 | PASS | POST /batch/process threshold 0.0005 -> 200 | fraud_count=1 on sample20 |
| AE-04 | PASS | POST /batch/process threshold 0.0002 -> 200 | Lower threshold increased fraud_count to 6 |
| AE-05 | PASS | POST /batch/process threshold 0.001 -> 200 | Higher threshold reduced fraud_count to 1 |
| AE-06 | PASS | Malformed CSV check done via LSTM path -> 500 with clear missing-columns detail | Error surfaced clearly; no crash observed |
| AE-07 | BLOCKED | Numeric edge-value CSV not executed in this cycle | Schedule in next cycle |

## Metrics Captured

- Baseline threshold: 0.0005
- Fraud count (baseline): 1
- Fraud count (low threshold): 6 (threshold 0.0002)
- Fraud count (high threshold): 1 (threshold 0.001)
- Avg processing time: Not captured in this run (API functional checks only)

## Defects

- No model-specific defects observed in executed AE cases.
- Test gap: AE-07 numeric-edge dataset scenario remains unexecuted.
