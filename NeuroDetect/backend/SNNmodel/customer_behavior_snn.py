"""
Customer-Centric SNN Fraud/Anomaly Training

Improvements:
- Uses the same preprocessing feature set as other project models (24 features)
- Uses the same SNN classifier architecture as previous snnfull model
- Adds validation-based early stopping
- Keeps customer-level behavior profiling and anomaly explanations
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:
    raise ImportError("snntorch is required. Install with: pip install snntorch") from exc


torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@dataclass
class CustomerProfile:
    cc_num: str
    transaction_count: int
    normal_count: int
    fraud_prob_mean: float
    fraud_prob_std: float
    fraud_prob_threshold: float
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]


class DataEngineer:
    """Same preprocessing style and feature set as other models."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def load_and_engineer_features(self, sample_size: int | None = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        if "cc_num" not in df.columns:
            raise ValueError("Input CSV must include `cc_num`")
        if "is_fraud" not in df.columns:
            raise ValueError("Input CSV must include `is_fraud`")

        df["cc_num"] = df["cc_num"].astype(str)
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df = df.dropna(subset=["trans_date_trans_time"]).copy()

        # Time features
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["day_of_month"] = df["trans_date_trans_time"].dt.day
        df["month"] = df["trans_date_trans_time"].dt.month

        # Distance
        for col in ["lat", "long", "merch_lat", "merch_long", "amt", "city_pop"]:
            if col not in df.columns:
                df[col] = 0.0
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
        df["long"] = pd.to_numeric(df["long"], errors="coerce").fillna(0.0)
        df["merch_lat"] = pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0.0)
        df["merch_long"] = pd.to_numeric(df["merch_long"], errors="coerce").fillna(0.0)
        df["amt"] = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)
        df["city_pop"] = pd.to_numeric(df["city_pop"], errors="coerce").fillna(0.0)

        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
        )

        # Derived features
        df["log_amt"] = np.log1p(np.clip(df["amt"], a_min=0, a_max=None))
        df["amt_per_pop"] = df["amt"] / (df["city_pop"] + 1)

        # Cyclical time encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Category encoding (same top categories)
        top_categories = [
            "gas_transport",
            "grocery_pos",
            "home",
            "shopping_pos",
            "kids_pets",
            "shopping_net",
            "entertainment",
            "food_dining",
        ]
        if "category" not in df.columns:
            df["category"] = "unknown"

        for cat in top_categories:
            df[f"cat_{cat}"] = (df["category"] == cat).astype(int)

        other_cats = set(df["category"].dropna().unique()) - set(top_categories)
        df["cat_other"] = df["category"].isin(other_cats).astype(int)

        # Gender encoding
        if "gender" not in df.columns:
            df["gender"] = "U"
        df["gender_M"] = (df["gender"] == "M").astype(int)

        self.feature_names = [
            "amt",
            "lat",
            "long",
            "city_pop",
            "merch_lat",
            "merch_long",
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "distance",
            "log_amt",
            "amt_per_pop",
            "hour_sin",
            "hour_cos",
            "cat_food_dining",
            "cat_gas_transport",
            "cat_grocery_pos",
            "cat_home",
            "cat_kids_pets",
            "cat_other",
            "cat_shopping_net",
            "cat_shopping_pos",
            "gender_M",
        ]

        X = df[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        y = df["is_fraud"].astype(int).values
        return df.reset_index(drop=True), X, y

    def split_and_normalize(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, np.ndarray | pd.DataFrame]:
        idx = np.arange(len(df))

        idx_temp, idx_test = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=42,
        )
        val_ratio = val_size / (1 - test_size)
        idx_train, idx_val = train_test_split(
            idx_temp,
            test_size=val_ratio,
            stratify=y[idx_temp],
            random_state=42,
        )

        X_train = X[idx_train]
        X_val = X[idx_val]
        X_test = X[idx_test]

        X_train_norm = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_norm = self.scaler.transform(X_val).astype(np.float32)
        X_test_norm = self.scaler.transform(X_test).astype(np.float32)

        return {
            "train_df": df.iloc[idx_train].reset_index(drop=True),
            "val_df": df.iloc[idx_val].reset_index(drop=True),
            "test_df": df.iloc[idx_test].reset_index(drop=True),
            "X_train": X_train_norm,
            "X_val": X_val_norm,
            "X_test": X_test_norm,
            "y_train": y[idx_train],
            "y_val": y[idx_val],
            "y_test": y[idx_test],
        }


class SpikingFraudDetector(nn.Module):
    """Same architecture as previous snnfull model."""

    def __init__(
        self,
        input_size: int = 24,
        hidden_size: int = 64,
        output_size: int = 2,
        beta: float = 0.95,
        spike_grad=None,
    ):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        outputs = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            out = self.fc3(spk2)
            outputs.append(out)

        return torch.sum(torch.stack(outputs, dim=0), dim=0)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SNNTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = None

    @property
    def device_name(self) -> str:
        return str(self.device)

    def set_class_weights(self, y_train: np.ndarray, loss_type: str = "ce", focal_gamma: float = 2.0) -> None:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        if loss_type == "focal":
            self.criterion = FocalLoss(weight=class_weights_tensor, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    def train_epoch(self, loader: DataLoader, num_steps: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        for data, target in loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data, num_steps=num_steps)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_labels.extend(target.detach().cpu().numpy().tolist())

        return {
            "loss": total_loss / max(len(loader), 1),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, num_steps: int) -> Dict[str, object]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []

        for data, target in loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(data, num_steps=num_steps)
            loss = self.criterion(output, target)
            total_loss += loss.item()

            probs = torch.softmax(output, dim=1)[:, 1]
            pred = torch.argmax(output, dim=1)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_labels.extend(target.detach().cpu().numpy().tolist())

        metrics = {
            "loss": total_loss / max(len(loader), 1),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "labels": np.array(all_labels, dtype=np.int64),
            "preds": np.array(all_preds, dtype=np.int64),
            "probs": np.array(all_probs, dtype=np.float32),
        }
        try:
            metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
        except Exception:
            metrics["auc"] = 0.0
        return metrics

    def fit_with_early_stopping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_steps: int,
        epochs: int,
        patience: int,
    ) -> Dict[str, List[float]]:
        if self.criterion is None:
            raise RuntimeError("Call set_class_weights before training")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
        }

        best_val_f1 = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_m = self.train_epoch(train_loader, num_steps=num_steps)
            val_m = self.evaluate(val_loader, num_steps=num_steps)

            history["train_loss"].append(train_m["loss"])
            history["val_loss"].append(val_m["loss"])
            history["train_f1"].append(train_m["f1"])
            history["val_f1"].append(val_m["f1"])

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_m['loss']:.4f} F1: {train_m['f1']:.4f} | "
                f"Val Loss: {val_m['loss']:.4f} F1: {val_m['f1']:.4f} AUC: {val_m['auc']:.4f}"
            )

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
                print("  New best model")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping triggered (patience={patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history


@torch.no_grad()
def get_probabilities(trainer: SNNTrainer, loader: DataLoader, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    trainer.model.eval()
    all_probs = []
    all_labels = []
    for data, target in loader:
        data = data.to(trainer.device, non_blocking=True)
        output = trainer.model(data, num_steps=num_steps)
        probs = torch.softmax(output, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(target.numpy().tolist())
    return np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def optimize_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def build_customer_profiles(
    train_df: pd.DataFrame,
    X_train_scaled: np.ndarray,
    train_probs: np.ndarray,
    feature_names: List[str],
    min_customer_txn: int,
) -> Dict[str, CustomerProfile]:
    work = train_df.copy()
    work["_prob"] = train_probs

    profiles: Dict[str, CustomerProfile] = {}
    for cc_num, group in work.groupby("cc_num"):
        if len(group) < min_customer_txn:
            continue

        group_idx = group.index.values
        X_group = X_train_scaled[group_idx]

        normal_group = group[group["is_fraud"] == 0]
        if len(normal_group) < max(5, min_customer_txn // 2):
            normal_group = group

        normal_idx = normal_group.index.values
        X_normal = X_train_scaled[normal_idx]
        normal_probs = normal_group["_prob"].values

        prob_mean = float(np.mean(normal_probs))
        prob_std = float(np.std(normal_probs))
        prob_q = float(np.quantile(normal_probs, 0.995)) if len(normal_probs) > 1 else float(prob_mean + 3 * prob_std)
        prob_threshold = max(prob_q, prob_mean + 3.0 * prob_std)

        feat_mean = np.mean(X_normal, axis=0)
        feat_std = np.std(X_normal, axis=0)

        profiles[cc_num] = CustomerProfile(
            cc_num=cc_num,
            transaction_count=int(len(group)),
            normal_count=int(len(normal_group)),
            fraud_prob_mean=prob_mean,
            fraud_prob_std=prob_std,
            fraud_prob_threshold=float(prob_threshold),
            feature_mean={k: float(v) for k, v in zip(feature_names, feat_mean)},
            feature_std={k: float(v) for k, v in zip(feature_names, feat_std)},
        )

    return profiles


def explain_transaction(
    cc_num: str,
    feature_names: List[str],
    x_scaled: np.ndarray,
    profile: CustomerProfile | None,
    fraud_prob: float,
    decision_threshold: float,
    top_k: int = 3,
) -> List[Dict[str, float | str]]:
    if profile is None:
        return [
            {
                "feature": "unknown_customer",
                "feature_abs_deviation": 0.0,
                "customer_zscore": 0.0,
                "combined_impact": float(max(0.0, fraud_prob - decision_threshold)),
            }
        ]

    mean_vec = np.array([profile.feature_mean[f] for f in feature_names], dtype=np.float32)
    std_vec = np.array([profile.feature_std[f] for f in feature_names], dtype=np.float32)
    std_vec = np.where(std_vec < 1e-6, 1.0, std_vec)

    abs_dev = np.abs(x_scaled - mean_vec)
    z = abs_dev / std_vec
    contrib = abs_dev * (1.0 + z)

    idx_top = np.argsort(contrib)[::-1][:top_k]
    reasons = []
    for idx in idx_top:
        reasons.append(
            {
                "feature": feature_names[idx],
                "feature_abs_deviation": float(abs_dev[idx]),
                "customer_zscore": float(z[idx]),
                "combined_impact": float(contrib[idx]),
            }
        )

    return reasons


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"customer_snn_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    if args.device == "cuda" and args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use CUDA-enabled PyTorch or run with --device cpu")

    engineer = DataEngineer(args.data_path)
    df, X, y = engineer.load_and_engineer_features(sample_size=args.sample_size)

    customer_summary = (
        df.groupby("cc_num", as_index=False)
        .size()
        .rename(columns={"size": "transaction_count"})
        .sort_values("transaction_count", ascending=False)
    )
    customer_summary.to_csv(os.path.join(run_dir, "customer_transaction_counts.csv"), index=False)

    print(f"Unique customers: {customer_summary['cc_num'].nunique():,}")
    print(f"Transactions: {len(df):,} | Fraud rate: {y.mean() * 100:.2f}%")

    split = engineer.split_and_normalize(df=df, X=X, y=y, test_size=args.test_size, val_size=args.val_size)

    train_ds = TensorDataset(torch.FloatTensor(split["X_train"]), torch.LongTensor(split["y_train"]))
    val_ds = TensorDataset(torch.FloatTensor(split["X_val"]), torch.LongTensor(split["y_val"]))
    test_ds = TensorDataset(torch.FloatTensor(split["X_test"]), torch.LongTensor(split["y_test"]))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = SpikingFraudDetector(
        input_size=24,
        hidden_size=args.hidden_size,
        output_size=2,
        beta=args.beta,
    )
    trainer = SNNTrainer(model=model, learning_rate=args.lr, device=args.device)
    trainer.set_class_weights(split["y_train"], loss_type=args.loss, focal_gamma=args.focal_gamma)

    print(f"Using device: {trainer.device_name}")
    if trainer.device_name == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")

    history = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        num_steps=args.time_steps,
        epochs=args.epochs,
        patience=args.patience,
    )

    val_probs, val_labels = get_probabilities(trainer, val_loader, args.time_steps)
    optimal_threshold = optimize_threshold(val_labels, val_probs)

    test_probs, test_labels = get_probabilities(trainer, test_loader, args.time_steps)
    test_preds = (test_probs >= optimal_threshold).astype(int)

    test_metrics = {
        "accuracy": float(np.mean(test_preds == test_labels)),
        "precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "auc": float(roc_auc_score(test_labels, test_probs)),
        "optimal_threshold": float(optimal_threshold),
    }

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Build customer behavior profiles from training split
    train_probs, _ = get_probabilities(trainer, train_loader, args.time_steps)
    customer_profiles = build_customer_profiles(
        train_df=split["train_df"],
        X_train_scaled=split["X_train"],
        train_probs=train_probs,
        feature_names=engineer.feature_names,
        min_customer_txn=args.min_customer_txn,
    )

    known_thresholds = np.array([v.fraud_prob_threshold for v in customer_profiles.values()], dtype=np.float32)
    if len(known_thresholds) == 0:
        global_threshold = float(optimal_threshold)
    else:
        global_threshold = float(np.quantile(known_thresholds, args.global_threshold_quantile))

    # Customer-aware scoring for test split
    scored_rows = []
    for i, row in split["test_df"].iterrows():
        cc_num = str(row["cc_num"])
        profile = customer_profiles.get(cc_num)

        if profile is None and args.unknown_customer_policy == "skip":
            continue

        if profile is None:
            decision_threshold = max(float(optimal_threshold), global_threshold)
            known_customer = 0
        else:
            decision_threshold = max(float(optimal_threshold), float(profile.fraud_prob_threshold))
            known_customer = 1

        decision_threshold = float(decision_threshold * args.threshold_scale)

        prob = float(test_probs[i])
        is_anomaly = int(prob >= decision_threshold)

        reasons = explain_transaction(
            cc_num=cc_num,
            feature_names=engineer.feature_names,
            x_scaled=split["X_test"][i],
            profile=profile,
            fraud_prob=prob,
            decision_threshold=decision_threshold,
            top_k=args.explain_top_k,
        )

        scored_rows.append(
            {
                "cc_num": cc_num,
                "trans_date_trans_time": str(row["trans_date_trans_time"]),
                "is_fraud": int(row["is_fraud"]),
                "fraud_probability": prob,
                "decision_threshold": decision_threshold,
                "known_customer": known_customer,
                "is_anomaly": is_anomaly,
                "anomaly_reasons": json.dumps(reasons),
            }
        )

    scored_df = pd.DataFrame(scored_rows)
    scored_df.to_csv(os.path.join(run_dir, "scored_transactions.csv"), index=False)

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "input_size": 24,
            "hidden_size": args.hidden_size,
            "output_size": 2,
            "beta": args.beta,
            "time_steps": args.time_steps,
            "feature_names": engineer.feature_names,
            "optimal_threshold": float(optimal_threshold),
        },
        os.path.join(run_dir, "snn_customer_classifier.pth"),
    )

    joblib.dump(engineer.scaler, os.path.join(run_dir, "scaler.pkl"))

    with open(os.path.join(run_dir, "customer_profiles.json"), "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in customer_profiles.items()}, f, indent=2)

    with open(os.path.join(run_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": engineer.feature_names, "num_features": 24}, f, indent=2)

    with open(os.path.join(run_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metadata = {
        "data_path": args.data_path,
        "run_dir": run_dir,
        "total_transactions": int(len(df)),
        "unique_customers": int(customer_summary["cc_num"].nunique()),
        "train_samples": int(len(split["train_df"])),
        "val_samples": int(len(split["val_df"])),
        "test_samples": int(len(split["test_df"])),
        "requested_device": args.device,
        "actual_device": trainer.device_name,
        "gpu_name": torch.cuda.get_device_name(0) if trainer.device_name == "cuda" else None,
        "epochs": args.epochs,
        "patience": args.patience,
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "test_metrics": test_metrics,
        "threshold_policy": {
            "unknown_customer_policy": args.unknown_customer_policy,
            "global_threshold_quantile": args.global_threshold_quantile,
            "global_threshold": global_threshold,
            "threshold_scale": args.threshold_scale,
        },
    }

    with open(os.path.join(run_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run complete: {run_dir}")
    return {"run_dir": run_dir, "test_metrics": test_metrics}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train improved customer-centric SNN model")
    parser.add_argument("--data-path", type=str, default="C:/finalYear/NeuroDetect/backend/dataset/fraudTrain.csv")
    parser.add_argument("--output-dir", type=str, default="C:/finalYear/NeuroDetect/backend/snn_models")
    parser.add_argument("--sample-size", type=int, default=None)

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--min-customer-txn", type=int, default=15)
    parser.add_argument("--unknown-customer-policy", type=str, default="global", choices=["global", "skip"])
    parser.add_argument("--global-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--explain-top-k", type=int, default=3)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
