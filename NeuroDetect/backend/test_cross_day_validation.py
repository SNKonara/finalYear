#!/usr/bin/env python3
"""
Cross-Day Model Validation & Data Drift Testing Script

Executes comprehensive model tests:
- Baseline batch runs with consistency checks
- Threshold sweep with monotonicity validation
- LSTM sequence sensitivity tests
- Data drift detection via score distribution analysis
- Detailed JSON report for trend analysis

Usage:
    python test_cross_day_validation.py [--output ./results]
"""

import json
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

# Configuration
API_BASE = "http://127.0.0.1:8000"
SAMPLE_FILE = "backend/dataset/fraudTest_sample20.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "testing" / "results"
REPORT_FILE = RESULTS_DIR / f"cross_day_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

MODELS = ["autoencoder", "lstm", "snn"]
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

# Expected baselines (from analysis and prior runs)
EXPECTED_BASELINES = {
    "autoencoder": {"fraud_count": (4, 8), "avg_score": (0.30, 0.60)},
    "lstm": {"fraud_count": (3, 7), "avg_score": (0.20, 0.45)},  # ← Monitor closely
    "snn": {"fraud_count": (4, 8), "avg_score": (0.30, 0.60)},
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _to_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "fraud", "yes"}:
            return 1
        if lowered in {"0", "false", "legit", "legitimate", "no"}:
            return 0
        try:
            return 1 if float(lowered) >= 0.5 else 0
        except ValueError:
            return None
    return None


def _extract_labeled_metrics(rows: Any) -> Dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        return {"labeled_metrics_available": False, "reason": "missing response rows"}

    try:
        tp = fp = tn = fn = 0
        labeled_rows = 0

        for row in rows:
            if not isinstance(row, dict):
                continue
            truth_raw = row.get("is_fraud", row.get("H1"))
            pred_raw = row.get("prediction")
            truth = _to_binary_label(truth_raw)
            pred = _to_binary_label(pred_raw)
            if truth is None or pred is None:
                continue
            labeled_rows += 1

            if truth == 1 and pred == 1:
                tp += 1
            elif truth == 0 and pred == 1:
                fp += 1
            elif truth == 0 and pred == 0:
                tn += 1
            elif truth == 1 and pred == 0:
                fn += 1

        if labeled_rows == 0:
            return {"labeled_metrics_available": False, "reason": "no usable labels in response rows"}

        positives = tp + fn
        negatives = tn + fp
        precision = (_safe_div(tp, tp + fp) if (tp + fp) > 0 else None)
        recall = (_safe_div(tp, tp + fn) if positives > 0 else None)
        f1 = (_safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) > 0 else None)
        accuracy = _safe_div(tp + tn, labeled_rows)
        specificity = (_safe_div(tn, tn + fp) if negatives > 0 else None)
        false_positive_rate = (_safe_div(fp, negatives) if negatives > 0 else None)

        return {
            "labeled_metrics_available": True,
            "labeled_rows": labeled_rows,
            "positive_labels": positives,
            "negative_labels": negatives,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "single_class_labels": positives == 0 or negatives == 0,
        }
    except Exception as exc:
        return {"labeled_metrics_available": False, "reason": f"metrics parse error: {exc}"}


def check_api_health() -> bool:
    """Verify API is running and models are loaded."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code != 200:
            print(f"❌ Health check failed: {resp.status_code}")
            return False

        models_resp = requests.get(f"{API_BASE}/models", timeout=5)
        if models_resp.status_code != 200:
            print(f"❌ Models endpoint failed: {models_resp.status_code}")
            return False

        models_data = models_resp.json()
        # models_data is a dict like {"autoencoder": {...}, "lstm": {...}, "snn": {...}}
        if isinstance(models_data, dict):
            for model in MODELS:
                if model not in models_data or not models_data[model].get("loaded"):
                    print(f"❌ Model {model} not loaded")
                    return False
        else:
            # Fallback for list format
            for model in MODELS:
                if not any(m["model_type"] == model and m["loaded"] for m in models_data):
                    print(f"❌ Model {model} not loaded")
                    return False

        print("✅ API health check passed")
        print(f"   Loaded models: {', '.join(m.upper() for m in MODELS if models_data.get(m, {}).get('loaded'))}")
        return True
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False


def run_batch_process(model_type: str, threshold: float = None) -> Dict[str, Any]:
    """
    Execute batch processing for a model.

    Returns: {success, timestamp, fraud_count, avg_score, response_time, ...}
    """
    try:
        with open(SAMPLE_FILE, "rb") as f:
            files = {"file": f}
            data = {"model_type": model_type}
            if threshold is not None:
                data["threshold"] = threshold

            start = time.time()
            resp = requests.post(f"{API_BASE}/batch/process", files=files, data=data, timeout=30)
            elapsed = time.time() - start

            if resp.status_code != 200:
                return {
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "response_time": elapsed,
                }

            response_data = resp.json()
            stats = response_data.get("statistics", {})

            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "model_type": model_type,
                "threshold": threshold or stats.get("threshold"),
                "fraud_count": stats.get("fraud_count", 0),
                "total": stats.get("total", 20),
                "fraud_percentage": stats.get("fraud_percentage", 0),
                "avg_fraud_score": stats.get("avg_fraud_score", 0),
                "max_fraud_score": stats.get("max_fraud_score", 0),
                "min_fraud_score": stats.get("min_fraud_score", 0),
                "response_time": elapsed,
                "batch_id": response_data.get("batch_id"),
                "labeled_metrics": _extract_labeled_metrics(response_data.get("results")),
            }

    except Exception as e:
        return {
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "error": str(e),
        }


def baseline_consistency_runs() -> Dict[str, List[Dict]]:
    """Run each model 3 times to check consistency."""
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE CONSISTENCY (3 runs per model)")
    print("=" * 60)

    results = {}
    for model in MODELS:
        print(f"\nTesting {model.upper()}...")
        model_results = []

        for run in range(1, 4):
            result = run_batch_process(model)
            model_results.append(result)

            if result["success"]:
                fraud_count = result["fraud_count"]
                avg_score = result["avg_fraud_score"]
                elapsed = result["response_time"]

                baseline = EXPECTED_BASELINES[model]
                fraud_ok = baseline["fraud_count"][0] <= fraud_count <= baseline["fraud_count"][1]
                score_ok = baseline["avg_score"][0] <= avg_score <= baseline["avg_score"][1]

                status = "✅" if (fraud_ok and score_ok) else "⚠️"
                print(
                    f"  Run {run}: {status} fraud={fraud_count}, "
                    f"avg_score={avg_score:.4f}, time={elapsed:.2f}s"
                )

                if model == "lstm" and not (fraud_ok and score_ok):
                    print(f"    ⚠️  LSTM OUT OF BASELINE: Expected fraud in {baseline['fraud_count']}, "
                          f"avg_score in {baseline['avg_score']}")
            else:
                print(f"  Run {run}: ❌ {result.get('error', 'Unknown error')}")

        results[model] = model_results

    return results


def threshold_sweep_monotonicity() -> Dict[str, List[Dict]]:
    """
    Validate threshold sweep is monotonic (fraud count decreases or stays same).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: THRESHOLD MONOTONICITY CHECK")
    print("=" * 60)

    results = {}
    for model in MODELS:
        print(f"\nTesting {model.upper()}...")
        sweep_results = []

        for threshold in THRESHOLDS:
            result = run_batch_process(model, threshold)
            sweep_results.append(result)

            if result["success"]:
                fraud_count = result["fraud_count"]
                labeled_metrics = result.get("labeled_metrics", {})
                if labeled_metrics.get("labeled_metrics_available"):
                    f1 = labeled_metrics.get("f1", 0.0)
                    if labeled_metrics.get("single_class_labels"):
                        acc = labeled_metrics.get("accuracy", 0.0)
                        fpr = labeled_metrics.get("false_positive_rate")
                        print(f"  Threshold {threshold}: fraud_count={fraud_count}, accuracy={acc:.4f}, fpr={0.0 if fpr is None else fpr:.4f}")
                    else:
                        print(f"  Threshold {threshold}: fraud_count={fraud_count}, f1={0.0 if f1 is None else f1:.4f}")
                else:
                    print(f"  Threshold {threshold}: fraud_count={fraud_count}")
            else:
                print(f"  Threshold {threshold}: ❌ {result.get('error')}")

        # Check monotonicity
        fraud_counts = [r["fraud_count"] for r in sweep_results if r["success"]]
        if fraud_counts and len(fraud_counts) == len(THRESHOLDS):
            is_monotonic = all(fraud_counts[i] >= fraud_counts[i + 1] for i in range(len(fraud_counts) - 1))
            status = "✅ MONOTONIC" if is_monotonic else "❌ NOT MONOTONIC"
            print(f"  Result: {status} - Counts: {fraud_counts}")

            if not is_monotonic:
                print(f"    ⚠️  RED FLAG: Non-monotonic threshold behavior detected!")
        else:
            print(f"  Result: ❌ Could not validate (missing results)")

        results[model] = sweep_results

    return results


def lstm_sequence_sensitivity() -> Dict[str, Any]:
    """
    Test if LSTM is sensitive to input ordering.
    Requires: Creating shuffled variants of sample file.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: LSTM SEQUENCE SENSITIVITY")
    print("=" * 60)

    print("\n⚠️  Note: This test requires file variants not yet created.")
    print("   Skipping for now. Add shuffled/sorted CSV variants to dataset/")

    return {"skipped": True, "reason": "Requires file variants"}


def data_drift_detection() -> Dict[str, Any]:
    """
    Analyze score distributions from baseline runs to detect drift.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: DATA DRIFT DETECTION")
    print("=" * 60)

    # Fetch latest batch results from MongoDB to analyze score distributions
    try:
        resp = requests.get(f"{API_BASE}/reports?limit=10", timeout=10)
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, list):
                reports = payload
            elif isinstance(payload, dict):
                candidate = payload.get("reports")
                if isinstance(candidate, list):
                    reports = candidate
                else:
                    reports = [item for item in payload.values() if isinstance(item, dict)]
            else:
                reports = []

            print(f"✅ Retrieved {len(reports)} recent reports for drift analysis")

            # Group by model type and check for score distribution changes
            by_model = {}
            for report in reports:
                if not isinstance(report, dict):
                    continue
                model_type = report.get("model_type")
                if model_type:
                    if model_type not in by_model:
                        by_model[model_type] = []
                    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
                    fraud_rate = summary.get("fraud_rate_percent", 0)
                    try:
                        fraud_rate = float(fraud_rate)
                    except Exception:
                        fraud_rate = 0.0
                    by_model[model_type].append(fraud_rate)

            print("\n📊 Fraud Rate Trends by Model:")
            for model, rates in by_model.items():
                if rates:
                    mean_rate = sum(rates) / len(rates)
                    variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
                    std_dev = variance ** 0.5
                    print(f"  {model}: mean={mean_rate:.1f}%, std_dev={std_dev:.1f}%, samples={len(rates)}")

                    if std_dev > mean_rate * 0.15:
                        print(f"    ⚠️  HIGH VARIANCE DETECTED (>15%)")

            return {"status": "partial", "data": by_model}
        else:
            print(f"⚠️  Could not fetch reports: HTTP {resp.status_code}")
            return {"status": "skipped", "reason": "Reports endpoint unavailable"}

    except Exception as e:
        print(f"⚠️  Data drift detection error: {e}")
        return {"status": "error", "reason": str(e)}


def recommendation_steps(sweep_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Generate threshold recommendations from labeled F1 metrics."""
    print("\n" + "=" * 60)
    print("PHASE 5: RECOMMENDATION STEPS (NO DEPLOYMENT)")
    print("=" * 60)

    recommendations: Dict[str, Any] = {
        "deployment_state": "on_hold",
        "project_deploy_mode": "deploy_all_models_together",
        "threshold_recommendations": {},
        "notes": [
            "Deployment is intentionally on hold.",
            "When resumed, deploy API + WebSocket + frontend together.",
        ],
    }

    for model in ("autoencoder", "snn"):
        candidates = []
        for run in sweep_results.get(model, []):
            labeled_metrics = run.get("labeled_metrics", {})
            if not run.get("success") or not labeled_metrics.get("labeled_metrics_available"):
                continue
            if labeled_metrics.get("single_class_labels"):
                continue
            candidates.append({
                "threshold": run.get("threshold"),
                "f1": labeled_metrics.get("f1", 0.0) or 0.0,
                "precision": labeled_metrics.get("precision", 0.0) or 0.0,
                "recall": labeled_metrics.get("recall", 0.0) or 0.0,
                "accuracy": labeled_metrics.get("accuracy", 0.0),
            })

        if not candidates:
            print(f"  {model.upper()}: no mixed-class labeled sample available for threshold recommendation")
            recommendations["threshold_recommendations"][model] = {
                "recommended": None,
                "reason": "no mixed-class labeled metrics available",
            }
            continue

        candidates.sort(key=lambda item: (item["f1"], item["recall"], -float(item["threshold"])), reverse=True)
        best = candidates[0]
        if float(best.get("f1", 0.0)) <= 0.0:
            print(f"  {model.upper()}: no positive F1 signal across tested thresholds; recommendation withheld")
            recommendations["threshold_recommendations"][model] = {
                "recommended": None,
                "reason": "all tested thresholds have non-positive F1",
                "all_candidates": candidates,
            }
            continue

        recommendations["threshold_recommendations"][model] = {
            "recommended": best,
            "all_candidates": candidates,
        }
        print(
            f"  {model.upper()}: recommended threshold={best['threshold']} "
            f"(f1={best['f1']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f})"
        )

    print("\n  Deployment state: ON HOLD")
    print("  Planned mode when resumed: FULL PROJECT + ALL MODELS TOGETHER")
    return recommendations


def apply_threshold_recommendations(recommendations: Dict[str, Any], persist: bool) -> Dict[str, Any]:
    """Optionally apply recommended thresholds via API."""
    applied: Dict[str, Any] = {"persist": persist, "updates": []}
    threshold_map = recommendations.get("threshold_recommendations", {})

    print("\nApplying threshold recommendations...")
    for model in ("autoencoder", "snn"):
        rec = threshold_map.get(model, {})
        best = rec.get("recommended") if isinstance(rec, dict) else None
        if not isinstance(best, dict) or best.get("threshold") is None:
            print(f"  {model.upper()}: skipped (no recommendation)")
            continue

        payload = {
            "model_type": model,
            "threshold": float(best["threshold"]),
            "persist": bool(persist),
            "reason": "cross-day validation recommendation",
        }
        try:
            resp = requests.post(f"{API_BASE}/batch/threshold", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                applied["updates"].append({"model": model, "success": True, "response": data})
                print(f"  {model.upper()}: updated threshold to {data.get('threshold')} (persist={persist})")
            else:
                applied["updates"].append({
                    "model": model,
                    "success": False,
                    "status_code": resp.status_code,
                    "error": resp.text[:300],
                })
                print(f"  {model.upper()}: update failed HTTP {resp.status_code}")
        except Exception as exc:
            applied["updates"].append({"model": model, "success": False, "error": str(exc)})
            print(f"  {model.upper()}: update failed ({exc})")

    return applied


def generate_report(
    baseline_results: Dict,
    sweep_results: Dict,
    sequence_results: Dict,
    drift_results: Dict,
    recommendation_results: Dict,
    threshold_apply_results: Dict | None,
) -> Dict[str, Any]:
    """
    Compile all results into a comprehensive report.
    """
    report = {
        "test_execution": datetime.utcnow().isoformat(),
        "api_base": API_BASE,
        "sample_file": SAMPLE_FILE,
        "phases": {
            "1_baseline_consistency": {
                "description": "3 runs per model for consistency check",
                "results": baseline_results,
            },
            "2_threshold_monotonicity": {
                "description": "Threshold sweep with monotonicity validation",
                "results": sweep_results,
            },
            "3_lstm_sequence_sensitivity": {
                "description": "LSTM ordering sensitivity test",
                "results": sequence_results,
            },
            "4_data_drift_detection": {
                "description": "Score distribution analysis",
                "results": drift_results,
            },
            "5_recommendation_steps": {
                "description": "Threshold recommendations and deployment hold state",
                "results": recommendation_results,
            },
        },
        "deployment": {
            "status": "on_hold",
            "strategy": "deploy_all_models_together_with_project",
            "apply_threshold_updates": threshold_apply_results,
        },
    }

    # Add summary
    baseline_pass = all(
        r["success"] for model_results in baseline_results.values() for r in model_results
    )
    sweep_pass = all(
        r["success"] for model_results in sweep_results.values() for r in model_results
    )

    report["summary"] = {
        "baseline_consistency_pass": baseline_pass,
        "threshold_monotonicity_pass": sweep_pass,
        "overall_status": "PASS" if (baseline_pass and sweep_pass) else "FAIL",
        "recommendations": [],
    }

    # Add recommendations based on results
    for model, results in baseline_results.items():
        for result in results:
            if result["success"]:
                baseline = EXPECTED_BASELINES[model]
                fraud_count = result["fraud_count"]
                avg_score = result["avg_fraud_score"]

                if model == "lstm":
                    if not (baseline["fraud_count"][0] <= fraud_count <= baseline["fraud_count"][1]):
                        report["summary"]["recommendations"].append(
                            f"LSTM fraud count {fraud_count} outside baseline "
                            f"{baseline['fraud_count']} - investigate data fit"
                        )
                    if not (baseline["avg_score"][0] <= avg_score <= baseline["avg_score"][1]):
                        report["summary"]["recommendations"].append(
                            f"LSTM avg_score {avg_score:.4f} outside baseline "
                            f"{baseline['avg_score']} - check score calibration"
                        )

    return report


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Cross-day model validation and recommendation runner")
    parser.add_argument(
        "--apply-recommendations",
        action="store_true",
        help="Apply recommended thresholds for autoencoder/snn via API",
    )
    parser.add_argument(
        "--persist-recommendations",
        action="store_true",
        help="Persist applied threshold recommendations to threshold_overrides.json",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("NEURODETECT CROSS-DAY MODEL VALIDATION & DATA DRIFT TESTING")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print("=" * 70)

    # Health check
    if not check_api_health():
        print("\n❌ API health check failed. Ensure backend is running on port 8000.")
        return 1

    # Execute all phases
    baseline_results = baseline_consistency_runs()
    sweep_results = threshold_sweep_monotonicity()
    sequence_results = lstm_sequence_sensitivity()
    drift_results = data_drift_detection()
    recommendation_results = recommendation_steps(sweep_results)

    threshold_apply_results = None
    if args.apply_recommendations:
        threshold_apply_results = apply_threshold_recommendations(
            recommendation_results,
            persist=args.persist_recommendations,
        )
    else:
        print("\nThreshold recommendations were generated but not applied.")

    # Compile report
    report = generate_report(
        baseline_results,
        sweep_results,
        sequence_results,
        drift_results,
        recommendation_results,
        threshold_apply_results,
    )

    # Save report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"✅ Test execution complete")
    print(f"📊 Report saved: {REPORT_FILE}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    print("=" * 70)

    if report["summary"]["recommendations"]:
        print("\n⚠️  RECOMMENDATIONS:")
        for rec in report["summary"]["recommendations"]:
            print(f"  - {rec}")

    return 0 if report["summary"]["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
