# SNN Test Results

## Run Info

- Date: 2026-04-22

## Case Log

| Case ID | Result (PASS/FAIL/BLOCKED) | Evidence | Notes |
|---|---|---|---|
| SN-01 | PASS | GET /models -> 200, snn present | Metadata available |
| SN-02 | PASS | POST /models/test (snn) -> 200 | success=true |
| SN-03 | PASS | POST /batch/process threshold 0.5 -> 200 | fraud_count=2 |
| SN-04 | PASS | Threshold sweep successful | 0.15->12, 0.5->2, 0.85->0 |
| SN-05 | BLOCKED | Personalized/mixed-customer dedicated dataset not executed | Needs customer-profile test dataset |
| SN-06 | BLOCKED | Explainability payload deep validation not executed | Add explicit response schema assertions next cycle |
| SN-07 | PASS | Threshold extremes (0.15, 0.85) produced stable outputs | No crash/overflow observed |

## Metrics Captured

- Baseline threshold: 0.5
- Fraud count (baseline): 2
- Fraud count (threshold low/high): low(0.15)=12, high(0.85)=0
- Any explainability payload issues: Not observed in executed batch sweep; dedicated check still pending (SN-06)

## Defects

- No SNN runtime defects observed in executed sweep.
- Coverage gaps: SN-05 and SN-06 remain pending.
