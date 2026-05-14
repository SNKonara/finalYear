# LSTM Test Results

## Run Info

- Date: 2026-04-22
- Tester: GitHub Copilot (GPT-5.3-Codex)
- Branch/Commit: Not captured in this run

## Case Log

| Case ID | Result (PASS/FAIL/BLOCKED) | Evidence | Notes |
|---|---|---|---|
| LS-01 | PASS | GET /models -> 200, lstm present | Metadata available |
| LS-02 | PASS | POST /models/test (lstm) -> 200 | success=true |
| LS-03 | PASS | POST /batch/process threshold 0.5 -> 200 | fraud_count=6 |
| LS-04 | PASS | 3 threshold runs all returned 200 | No timeout or crash in this sweep |
| LS-05 | PASS | Threshold sweep: 0.15->15, 0.5->6, 0.85->1 | Monotonic sensitivity observed |
| LS-06 | BLOCKED | 1-2 row sequence file not executed separately | Add targeted small-file test next cycle |
| LS-07 | PASS | Malformed CSV -> 500 with missing-columns detail | Clear validation-style detail returned |

## Metrics Captured

- Baseline threshold: 0.5
- Fraud count per run: t=0.15 => 15, t=0.5 => 6, t=0.85 => 1
- Mean processing time (3 runs): Not captured in this run
- Error/timeout count: 0 for valid sample20 runs

## Defects

- No LSTM inference defects observed in executed cases.
- Observation: malformed CSV returns 500 with detailed missing-columns message (could be improved to 400 in future hardening).
