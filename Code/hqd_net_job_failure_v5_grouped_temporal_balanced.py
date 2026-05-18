# -*- coding: utf-8 -*-
"""
HQD-Net v5: Grouped-Temporal Leakage-Safe Balanced-Training Hard-Example Quantum-Dependency Network
for Alibaba Cluster Trace v2018 job failure prediction.

Major fixes over v2:
  1) Removes all post-execution predictors: end_time-derived durations, runtime spans,
     failed instance counts, task/instance failure rates, and status-derived features.
  2) Uses train-only preprocessing through sklearn Pipeline.
  3) Uses threshold tuning on training data only to improve failure-focused F1/recall.
  4) Uses causal/previous-time machine pressure approximation through merge-asof.
  5) Saves CSV results, ablation CSVs, figures, trained models, feature groups, and README.

Run:
python hqd_net_job_failure_v5_grouped_temporal_balanced.py ^
  --data-dir "D:\\other\\ALIBABAQUATUM\\Dataset" ^
  --out-dir "D:\\other\\ALIBABAQUATUM\\Results_HQDNet_v4" ^
  --sample-sizes 5000 10000 20000 25000 ^
  --max-batch-task-rows 2000000 ^
  --max-batch-instance-rows 2000000 ^
  --max-machine-usage-rows 3000000 ^
  --run-ablations
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

BATCH_TASK_COLS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem"
]
BATCH_INSTANCE_COLS = [
    "instance_name", "task_name", "job_name", "task_type", "status",
    "start_time", "end_time", "machine_id", "seq_no", "total_seq_no",
    "cpu_avg", "cpu_max", "mem_avg", "mem_max"
]
MACHINE_USAGE_COLS = [
    "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
    "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"
]

LEAKAGE_FEATURES = [
    "end_time", "task_end_max", "last_end_time", "task_duration_mean", "task_duration_max",
    "inst_duration_mean", "inst_duration_max", "inst_duration_std", "runtime_span", "job_span",
    "failed_instance_count", "instance_fail_rate", "task_fail_rate", "prev_runtime_mean_200",
    "status", "task_failure", "instance_failure", "cpu_avg_mean", "cpu_avg_max", "cpu_max_mean",
    "cpu_max_max", "mem_avg_mean", "mem_avg_max", "mem_max_mean", "mem_max_max",
    "cpu_spike_mean", "mem_spike_mean"
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_schema(path: Path, columns: List[str], max_rows: Optional[int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    log(f"[data] reading {path} ...")
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    # Some files contain a real header, others may be headerless. Some contain repeated header rows.
    if list(df.columns) != columns:
        raw = pd.read_csv(path, nrows=max_rows, header=None, low_memory=False)
        if raw.shape[1] == len(columns):
            raw.columns = columns
            df = raw
    # Remove repeated header rows inside files.
    first_col = columns[0]
    if first_col in df.columns:
        df = df[df[first_col].astype(str) != first_col]
    df = df.dropna(how="all")
    log(f"[data] {path.name} shape={df.shape}")
    return df


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_div(a, b):
    return a / (b.replace(0, np.nan) if hasattr(b, "replace") else (b if b != 0 else np.nan))


def status_to_failure(s) -> int:
    txt = str(s).strip().lower()
    if txt in {"terminated", "finished", "success", "completed"}:
        return 0
    # Failed, killed, error, evicted, lost, unknown are treated as failure-risk events.
    return 1


def parse_parent_indices(task_name: str) -> List[int]:
    if not isinstance(task_name, str):
        return []
    nums = re.findall(r"\d+", task_name)
    if len(nums) <= 1:
        return []
    return [int(x) for x in nums[1:]]


def parse_task_index(task_name: str) -> int:
    if not isinstance(task_name, str):
        return 0
    nums = re.findall(r"\d+", task_name)
    return int(nums[0]) if nums else 0


def build_task_dag_features(task_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-submission/job-structure features only. No status/end_time/duration predictors."""
    task_df = task_df.copy()
    task_df = to_num(task_df, ["instance_num", "task_type", "start_time", "plan_cpu", "plan_mem"])
    task_df["task_index"] = task_df["task_name"].map(parse_task_index)
    task_df["parent_list"] = task_df["task_name"].map(parse_parent_indices)
    task_df["parent_count"] = task_df["parent_list"].map(len)
    task_df["is_root_task"] = (task_df["parent_count"] == 0).astype(int)
    task_df["resource_product"] = task_df["plan_cpu"].fillna(0) * task_df["plan_mem"].fillna(0)

    g = task_df.groupby("job_name", observed=True)
    job = g.agg(
        task_count=("task_name", "count"),
        unique_task_count=("task_name", "nunique"),
        total_instances=("instance_num", "sum"),
        max_instances_per_task=("instance_num", "max"),
        mean_instances_per_task=("instance_num", "mean"),
        dag_depth=("parent_count", lambda s: float(np.nanmax(s) + 1) if len(s) else 1.0),
        fan_in_mean=("parent_count", "mean"),
        fan_in_max=("parent_count", "max"),
        root_task_ratio=("is_root_task", "mean"),
        plan_cpu_mean=("plan_cpu", "mean"),
        plan_cpu_max=("plan_cpu", "max"),
        plan_cpu_sum=("plan_cpu", "sum"),
        plan_mem_mean=("plan_mem", "mean"),
        plan_mem_max=("plan_mem", "max"),
        plan_mem_sum=("plan_mem", "sum"),
        resource_product_sum=("resource_product", "sum"),
        task_start_min=("start_time", "min"),
    ).reset_index()

    job["parallelism_proxy"] = safe_div(job["total_instances"].fillna(0), job["task_count"].fillna(0) + 1.0)
    job["dependency_density"] = safe_div(job["fan_in_mean"].fillna(0), job["task_count"].fillna(0) + 1.0)
    job["cpu_mem_ratio"] = safe_div(job["plan_cpu_mean"].fillna(0), job["plan_mem_mean"].fillna(0) + 1e-9)
    # Safe proxy based on graph and requested parallelism, not observed duration.
    job["critical_path_proxy"] = job["dag_depth"].fillna(0) * np.log1p(job["max_instances_per_task"].fillna(0))
    job["dag_complexity"] = (
        np.log1p(job["task_count"].fillna(0)) +
        np.log1p(job["total_instances"].fillna(0)) +
        job["dag_depth"].fillna(0) +
        job["fan_in_max"].fillna(0)
    )
    return job


def build_instance_label_and_context(inst_df: pd.DataFrame) -> pd.DataFrame:
    """Use instance table for labels and scheduling context only. No post-execution usage as predictors."""
    inst_df = inst_df.copy()
    inst_df = to_num(inst_df, ["start_time", "seq_no", "total_seq_no"])
    inst_df["instance_failure"] = inst_df["status"].map(status_to_failure).astype(int)
    g = inst_df.groupby("job_name", observed=True)
    job = g.agg(
        instance_count=("instance_name", "count"),
        label_failed_instance_count=("instance_failure", "sum"),
        first_start_time=("start_time", "min"),
        primary_machine_id=("machine_id", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        unique_machines=("machine_id", "nunique"),
        seq_no_mean=("seq_no", "mean"),
        total_seq_no_mean=("total_seq_no", "mean"),
    ).reset_index()
    job["job_failure"] = (job["label_failed_instance_count"] > 0).astype(int)
    # label_failed_instance_count is label support only and will be dropped from predictors.
    return job


def build_machine_usage_timeline(data_dir: Path, max_rows: Optional[int]) -> pd.DataFrame:
    path = data_dir / "machine_usage_bigger.csv"
    if not path.exists():
        path = data_dir / "machine_usage.csv"
    if not path.exists():
        log("[warn] no machine_usage file found; pressure features will be empty")
        return pd.DataFrame(columns=["machine_id", "time_stamp"])
    mu = read_csv_schema(path, MACHINE_USAGE_COLS, max_rows=max_rows)
    mu = to_num(mu, ["time_stamp", "cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"])
    mu = mu.dropna(subset=["machine_id", "time_stamp"]).sort_values(["machine_id", "time_stamp"])
    # Causal rolling features: only previous and current observed machine measurements before job starts.
    for c in ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]:
        mu[c] = pd.to_numeric(mu[c], errors="coerce")
    mu["machine_cpu_mean"] = mu.groupby("machine_id")["cpu_util_percent"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mu["machine_cpu_max"] = mu.groupby("machine_id")["cpu_util_percent"].transform(lambda s: s.rolling(20, min_periods=1).max())
    mu["machine_mem_mean"] = mu.groupby("machine_id")["mem_util_percent"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mu["machine_mem_max"] = mu.groupby("machine_id")["mem_util_percent"].transform(lambda s: s.rolling(20, min_periods=1).max())
    mu["machine_net_in_mean"] = mu.groupby("machine_id")["net_in"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mu["machine_net_out_mean"] = mu.groupby("machine_id")["net_out"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mu["machine_disk_mean"] = mu.groupby("machine_id")["disk_io_percent"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    mu["machine_pressure"] = (
        mu["machine_cpu_mean"].fillna(0) / 100.0 +
        mu["machine_mem_mean"].fillna(0) / 100.0 +
        mu["machine_disk_mean"].fillna(0) / 100.0
    ) / 3.0
    keep = ["machine_id", "time_stamp", "machine_cpu_mean", "machine_cpu_max", "machine_mem_mean",
            "machine_mem_max", "machine_net_in_mean", "machine_net_out_mean", "machine_disk_mean", "machine_pressure"]
    return mu[keep]


def causal_attach_machine_pressure(job_df: pd.DataFrame, machine_timeline: pd.DataFrame) -> pd.DataFrame:
    if machine_timeline.empty or "primary_machine_id" not in job_df.columns:
        return job_df
    out_parts = []
    jobs = job_df.rename(columns={"primary_machine_id": "machine_id"}).copy()
    jobs = jobs.dropna(subset=["machine_id", "first_start_time"])
    timeline_groups = {k: v.sort_values("time_stamp") for k, v in machine_timeline.groupby("machine_id", observed=True)}
    for mid, gj in jobs.groupby("machine_id", observed=True):
        gm = timeline_groups.get(mid)
        if gm is None or gm.empty:
            tmp = gj.copy()
            for c in ["machine_cpu_mean", "machine_cpu_max", "machine_mem_mean", "machine_mem_max", "machine_net_in_mean", "machine_net_out_mean", "machine_disk_mean", "machine_pressure"]:
                tmp[c] = np.nan
            out_parts.append(tmp)
            continue
        tmp = pd.merge_asof(
            gj.sort_values("first_start_time"),
            gm.sort_values("time_stamp"),
            left_on="first_start_time", right_on="time_stamp",
            direction="backward"
        )
        out_parts.append(tmp)
    if not out_parts:
        return job_df
    attached = pd.concat(out_parts, ignore_index=True)
    attached = attached.rename(columns={"machine_id": "primary_machine_id"})
    # Preserve jobs that had missing machine id.
    missing = job_df[job_df["primary_machine_id"].isna()].copy()
    if len(missing):
        attached = pd.concat([attached, missing], ignore_index=True)
    return attached


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("first_start_time").copy()
    t = pd.to_numeric(df["first_start_time"], errors="coerce").fillna(0)
    df["submit_hour"] = ((t // 3600) % 24).astype(float)
    df["submit_day"] = (t // (24 * 3600)).astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["submit_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["submit_hour"] / 24.0)
    # Previous-job features only. These are allowed because they use past labels after sorting by time.
    df["prev_failure_rate_200"] = df["job_failure"].shift(1).rolling(200, min_periods=20).mean()
    df["prev_arrival_gap_200"] = t.diff().shift(1).rolling(200, min_periods=20).mean()
    df["local_arrival_density_200"] = 1.0 / df["prev_arrival_gap_200"].replace(0, np.nan)
    return df


def minmax_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=s.index)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    x = x.clip(lo, hi)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)


def col_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def add_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    components = {
        "cx_dag": col_or_zero(df, "dag_complexity"),
        "cx_pressure": col_or_zero(df, "machine_pressure"),
        "cx_repetition": col_or_zero(df, "total_instances"),
        "cx_parallel": col_or_zero(df, "parallelism_proxy"),
        "cx_resource": col_or_zero(df, "resource_product_sum"),
    }
    for k, v in components.items():
        df[k] = minmax_series(v)
    df["complexity_score"] = (
        0.30 * df["cx_dag"] +
        0.25 * df["cx_pressure"] +
        0.15 * df["cx_repetition"] +
        0.15 * df["cx_parallel"] +
        0.15 * df["cx_resource"]
    )
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_FEATURES if c in df.columns], errors="ignore")


def build_dataset(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    bt = read_csv_schema(data_dir / "batch_task.csv", BATCH_TASK_COLS, args.max_batch_task_rows)
    bi = read_csv_schema(data_dir / "batch_instance.csv", BATCH_INSTANCE_COLS, args.max_batch_instance_rows)

    task_feat = build_task_dag_features(bt)
    label_context = build_instance_label_and_context(bi)
    df = label_context.merge(task_feat, on="job_name", how="left")

    if args.use_machine_pressure:
        machine_timeline = build_machine_usage_timeline(data_dir, args.max_machine_usage_rows)
        df = causal_attach_machine_pressure(df, machine_timeline)

    df = add_temporal_features(df)
    df = add_complexity_score(df)
    df = drop_leakage_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"[data] engineered leakage-safe dataset shape={df.shape}, failure_rate={df['job_failure'].mean():.4f}")
    return df


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "resource": [
            "plan_cpu_mean", "plan_cpu_max", "plan_cpu_sum", "plan_mem_mean", "plan_mem_max", "plan_mem_sum",
            "cpu_mem_ratio", "resource_product_sum"
        ],
        "dag": [
            "task_count", "unique_task_count", "total_instances", "max_instances_per_task", "mean_instances_per_task",
            "dag_depth", "fan_in_mean", "fan_in_max", "root_task_ratio", "parallelism_proxy",
            "dependency_density", "critical_path_proxy", "dag_complexity"
        ],
        "scheduling": ["instance_count", "unique_machines", "seq_no_mean", "total_seq_no_mean"],
        "pressure": [
            "machine_cpu_mean", "machine_cpu_max", "machine_mem_mean", "machine_mem_max",
            "machine_net_in_mean", "machine_net_out_mean", "machine_disk_mean", "machine_pressure"
        ],
        "temporal": [
            "submit_hour", "submit_day", "sin_hour", "cos_hour", "prev_failure_rate_200",
            "prev_arrival_gap_200", "local_arrival_density_200", "first_start_time"
        ],
        "complexity": ["complexity_score", "cx_dag", "cx_pressure", "cx_repetition", "cx_parallel", "cx_resource"],
    }
    groups = {k: [c for c in v if c in df.columns and c not in LEAKAGE_FEATURES] for k, v in groups.items()}
    groups["all"] = sorted(set(sum(groups.values(), [])))
    groups["no_complexity"] = [c for c in groups["all"] if c not in groups["complexity"]]
    return groups


class QuantumStructuralEncoder(BaseEstimator, TransformerMixin):
    """CPU-safe quantum-inspired structural encoder using interference-style projections."""
    def __init__(self, n_components: int = 16, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0, 1, size=(X.shape[1], self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = X @ self.W_ + self.b_
        return np.hstack([np.cos(Z), np.sin(Z), np.cos(Z) * np.sin(Z)])


def make_preprocessor(features: List[str], scaler: str = "standard") -> ColumnTransformer:
    scale = StandardScaler() if scaler == "standard" else MinMaxScaler()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scale)]), features)
    ], remainder="drop")


def make_model(name: str, pos_weight: float = 1.0):
    if name == "LR":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        )
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
                             learning_rate_init=1e-3, max_iter=250, random_state=RANDOM_STATE,
                             early_stopping=True)
    if name == "XGB" and HAS_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
            scale_pos_weight=max(1.0, pos_weight)
        )
    if name == "LGBM" and HAS_LGBM:
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE,
                              n_jobs=-1, verbose=-1, class_weight="balanced")
    raise ValueError(f"Model unavailable or unknown: {name}")


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X)


def tune_threshold(y_true, scores) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    qs = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 80)))
    best_t, best_f1 = 0.5, -1
    for t in qs:
        pred = (scores >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def metrics_dict(y_true, score, pred, train_time, infer_time, model_name, sample_size, route="all", threshold=0.5) -> Dict[str, object]:
    out = {
        "model": model_name, "sample_size": sample_size, "route": route, "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "train_time_sec": train_time, "inference_time_sec": infer_time,
        "latency_ms_per_sample": (infer_time / max(1, len(y_true))) * 1000,
    }
    try: out["roc_auc"] = roc_auc_score(y_true, score)
    except Exception: out["roc_auc"] = np.nan
    try: out["pr_auc"] = average_precision_score(y_true, score)
    except Exception: out["pr_auc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out



def sample_dataset(df: pd.DataFrame, n: int, sampling_mode: str = "train_balanced") -> pd.DataFrame:
    """Build an evaluation pool WITHOUT balancing across train/test.

    Important v5 rule:
      - We do not balance before splitting.
      - The evaluation pool keeps the natural temporal distribution.
      - Balancing is applied only to the training split.

    This prevents duplicated rare failure templates from being copied into both train and test.
    """
    data = df.dropna(subset=["job_failure"]).copy()
    data["job_failure"] = data["job_failure"].astype(int)
    if len(data) == 0:
        raise RuntimeError("No valid labels found in job_failure.")
    if "first_start_time" in data.columns:
        data = data.sort_values("first_start_time")
    if n >= len(data):
        return data.reset_index(drop=True).copy()
    # temporal-prefix sampling keeps the setting deployment-like and reproducible.
    return data.head(n).reset_index(drop=True).copy()


def balance_training_frame(X: pd.DataFrame, y: pd.Series, mode: str = "train_balanced") -> Tuple[pd.DataFrame, pd.Series]:
    """Balance only the training partition.

    mode options:
      - train_balanced: 50/50 train data through controlled resampling.
      - class_weight_only or imbalanced: no resampling.
    """
    if mode in ["imbalanced", "class_weight_only", "none"]:
        return X.copy(), y.astype(int).copy()
    frame = X.copy()
    frame["job_failure"] = y.astype(int).values
    pos = frame[frame["job_failure"] == 1]
    neg = frame[frame["job_failure"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return X.copy(), y.astype(int).copy()
    # Use equal class counts. Cap to avoid exploding memory when positives are very rare.
    target = max(len(pos), min(len(neg), len(pos) * 4))
    target = max(target, min(len(neg), len(pos)))
    pos_b = pos.sample(n=target, replace=len(pos) < target, random_state=RANDOM_STATE + 1)
    neg_b = neg.sample(n=target, replace=len(neg) < target, random_state=RANDOM_STATE + 2)
    out = pd.concat([pos_b, neg_b], axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return out.drop(columns=["job_failure"]), out["job_failure"].astype(int)


def sample_stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return sample_dataset(df, n, sampling_mode="train_balanced")


def split_train_val_test_frame(data: pd.DataFrame, features: List[str], split_mode: str = "grouped_temporal"):
    """Return train/val/test frames with natural validation/test distribution.

    grouped_temporal protocol:
      1. sort by first_start_time;
      2. use first 70% for train, next 10% for validation, last 20% for test;
      3. keep job_name rows unique by construction;
      4. fall back to stratified IID only if a split has one class.
    """
    data = data.dropna(subset=["job_failure"]).copy()
    data["job_failure"] = data["job_failure"].astype(int)
    if split_mode in ["grouped_temporal", "temporal"] and "first_start_time" in data.columns:
        data = data.sort_values(["first_start_time", "job_name"] if "job_name" in data.columns else ["first_start_time"]).reset_index(drop=True)
        n = len(data)
        c1 = int(0.70 * n)
        c2 = int(0.80 * n)
        train = data.iloc[:c1].copy()
        val = data.iloc[c1:c2].copy()
        test = data.iloc[c2:].copy()
        ok = (train["job_failure"].nunique() == 2 and val["job_failure"].nunique() == 2 and test["job_failure"].nunique() == 2)
        if ok:
            return train, val, test
        log("[warn] temporal split had a single-class partition; using stratified fallback for this sample size.")
    # fallback for small samples or degenerate temporal windows
    tr, test = train_test_split(data, test_size=0.20, stratify=data["job_failure"], random_state=RANDOM_STATE)
    train, val = train_test_split(tr, test_size=0.125, stratify=tr["job_failure"], random_state=RANDOM_STATE)  # 70/10/20
    return train.copy(), val.copy(), test.copy()


def split_data(data: pd.DataFrame, features: List[str], split_mode: str):
    train, val, test = split_train_val_test_frame(data, features, split_mode)
    # Backward-compatible: combine train+val for older calls that expect train/test.
    train2 = pd.concat([train, val], axis=0)
    return train2[features], train2["job_failure"].astype(int), test[features], test["job_failure"].astype(int)


def split_train_val_test(data: pd.DataFrame, features: List[str], split_mode: str):
    train, val, test = split_train_val_test_frame(data, features, split_mode)
    return (train[features], train["job_failure"].astype(int),
            val[features], val["job_failure"].astype(int),
            test[features], test["job_failure"].astype(int))


def train_single_model(df: pd.DataFrame, features: List[str], model_name: str, sample_size: int, split_mode: str = "grouped_temporal", scaler: str = "standard", sampling_mode: str = "train_balanced"):
    data = sample_dataset(df, sample_size, sampling_mode=sampling_mode)
    X_train_raw, y_train_raw, X_val, y_val, X_test, y_test = split_train_val_test(data, features, split_mode)
    X_train, y_train = balance_training_frame(X_train_raw, y_train_raw, sampling_mode)
    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))
    pos_weight = neg / pos
    pipe = Pipeline([("prep", make_preprocessor(features, scaler)), ("clf", make_model(model_name, pos_weight))])
    t0 = time.perf_counter(); pipe.fit(X_train, y_train); train_time = time.perf_counter() - t0
    val_scores = predict_scores(pipe, X_val)
    threshold = tune_threshold(y_val, val_scores)
    t1 = time.perf_counter(); score = predict_scores(pipe, X_test); infer_time = time.perf_counter() - t1
    pred = (score >= threshold).astype(int)
    m = metrics_dict(y_test, score, pred, train_time, infer_time, model_name, sample_size, route=split_mode, threshold=threshold)
    m["train_positive_ratio_raw"] = float(np.mean(y_train_raw)) if len(y_train_raw) else np.nan
    m["train_positive_ratio_used"] = float(np.mean(y_train)) if len(y_train) else np.nan
    m["val_positive_ratio"] = float(np.mean(y_val)) if len(y_val) else np.nan
    m["test_positive_ratio"] = float(np.mean(y_test)) if len(y_test) else np.nan
    m["train_rows_used"] = int(len(y_train)); m["val_rows"] = int(len(y_val)); m["test_rows"] = int(len(y_test))
    return m, pipe, (X_train, y_train, X_test, y_test)


class HQDNet:
    """Hard-example routed classifier.

    Stage 1 trains a classical ensemble (RF + XGB/HGB + LGBM when available).
    Stage 2 estimates hard examples by ensemble disagreement and structural complexity.
    Stage 3 trains a quantum-inspired structural encoder only on those hard samples.
    Final prediction fuses classical ensemble and quantum-specialist scores only for routed samples.
    """
    def __init__(self, features: List[str], structural_features: List[str], tau_quantile: float = 0.80,
                 hard_quantile: float = 0.70, quantum_name: str = "MLP"):
        self.features = features
        self.structural_features = [c for c in structural_features if c in features]
        self.tau_quantile = tau_quantile
        self.hard_quantile = hard_quantile
        self.quantum_name = quantum_name

    def _ensemble_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        scores = []
        for pipe in self.ensemble_:
            scores.append(predict_scores(pipe, X[self.features]))
        S = np.vstack(scores).T
        return S.mean(axis=1), S.var(axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        neg, pos = max(1, int((y == 0).sum())), max(1, int((y == 1).sum()))
        pos_weight = neg / pos
        names = ["RF"]
        names.append("XGB" if HAS_XGB else "HGB")
        if HAS_LGBM:
            names.append("LGBM")
        self.ensemble_ = []
        for name in names:
            pipe = Pipeline([
                ("prep", make_preprocessor(self.features, "standard")),
                ("clf", make_model(name, pos_weight)),
            ])
            pipe.fit(X[self.features], y)
            self.ensemble_.append(pipe)
        p_ens, disagreement = self._ensemble_score(X)
        self.classical_threshold_ = tune_threshold(y, p_ens)

        cx = X["complexity_score"].fillna(0).values if "complexity_score" in X.columns else np.zeros(len(X))
        self.cx_tau_ = float(np.nanquantile(cx, self.tau_quantile)) if len(cx) else 0.0
        self.u_tau_ = float(np.nanquantile(disagreement, self.hard_quantile)) if len(disagreement) else 0.0
        hard_mask = (cx >= self.cx_tau_) & (disagreement >= self.u_tau_)

        # Guarantee enough positive hard examples. If too few, add the most uncertain positives and negatives.
        min_hard = max(40, int(0.08 * len(X)))
        if hard_mask.sum() < min_hard or y.loc[hard_mask].nunique() < 2:
            rank_score = pd.Series(disagreement, index=X.index).rank(pct=True).values + pd.Series(cx, index=X.index).rank(pct=True).values
            cutoff = np.nanquantile(rank_score, 0.70)
            hard_mask = rank_score >= cutoff
        if y.loc[hard_mask].nunique() < 2:
            hard_mask = np.ones(len(X), dtype=bool)

        self.route_train_ratio_ = float(np.mean(hard_mask))
        q_features = self.structural_features if self.structural_features else self.features
        self.q_features_ = q_features
        self.quantum_pipe_ = Pipeline([
            ("prep", make_preprocessor(q_features, "minmax")),
            ("qenc", QuantumStructuralEncoder(n_components=24, random_state=RANDOM_STATE)),
            ("clf", make_model(self.quantum_name, pos_weight)),
        ])
        self.quantum_pipe_.fit(X.loc[hard_mask, q_features], y.loc[hard_mask])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pc, disagreement = self._ensemble_score(X)
        cx = X["complexity_score"].fillna(0).values if "complexity_score" in X.columns else np.zeros(len(X))
        route_mask = (cx >= self.cx_tau_) & (disagreement >= self.u_tau_)
        score = pc.copy()
        if route_mask.any():
            pq = predict_scores(self.quantum_pipe_, X.loc[route_mask, self.q_features_])
            # Hard routed samples receive stronger quantum-specialist contribution.
            score[route_mask] = 0.50 * pc[route_mask] + 0.50 * pq
        self.last_route_ratio_ = float(np.mean(route_mask))
        return np.vstack([1 - score, score]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.classical_threshold_).astype(int)

def train_hqdnet(df: pd.DataFrame, features: List[str], groups: Dict[str, List[str]], sample_size: int, tau_quantile: float, split_mode: str = "grouped_temporal", sampling_mode: str = "train_balanced"):
    data = sample_dataset(df, sample_size, sampling_mode=sampling_mode)
    X_train_raw, y_train_raw, X_val, y_val, X_test, y_test = split_train_val_test(data, features, split_mode)
    X_train, y_train = balance_training_frame(X_train_raw, y_train_raw, sampling_mode)
    structural = sorted(set(groups.get("dag", []) + groups.get("pressure", []) + groups.get("complexity", [])))
    model = HQDNet(features=features, structural_features=structural, tau_quantile=tau_quantile, hard_quantile=0.70)
    t0 = time.perf_counter(); model.fit(X_train, y_train); train_time = time.perf_counter() - t0
    # Tune final decision threshold on natural validation, not balanced train.
    val_score = model.predict_proba(X_val)[:, 1]
    model.classical_threshold_ = tune_threshold(y_val, val_score)
    t1 = time.perf_counter(); score = model.predict_proba(X_test)[:, 1]; infer_time = time.perf_counter() - t1
    pred = (score >= model.classical_threshold_).astype(int)
    m = metrics_dict(y_test, score, pred, train_time, infer_time, f"HQD-Net(tau={tau_quantile:.2f})", sample_size, route=split_mode, threshold=model.classical_threshold_)
    m["quantum_route_train_ratio"] = model.route_train_ratio_
    m["quantum_route_test_ratio"] = getattr(model, "last_route_ratio_", np.nan)
    m["tau_threshold"] = getattr(model, "cx_tau_", np.nan)
    m["uncertainty_threshold"] = getattr(model, "u_tau_", np.nan)
    m["train_positive_ratio_raw"] = float(np.mean(y_train_raw)) if len(y_train_raw) else np.nan
    m["train_positive_ratio_used"] = float(np.mean(y_train)) if len(y_train) else np.nan
    m["val_positive_ratio"] = float(np.mean(y_val)) if len(y_val) else np.nan
    m["test_positive_ratio"] = float(np.mean(y_test)) if len(y_test) else np.nan
    m["train_rows_used"] = int(len(y_train)); m["val_rows"] = int(len(y_val)); m["test_rows"] = int(len(y_test))
    return m, model


def plot_metric_bars(metrics_df: pd.DataFrame, out_path: Path, title: str, sample_size: Optional[int] = None):
    df = metrics_df.copy()
    if sample_size is not None:
        df = df[df["sample_size"] == sample_size]
    if df.empty: return
    df = df.sort_values("pr_auc", ascending=False)
    labels = df["model"].astype(str).tolist(); x = np.arange(len(labels)); width = 0.16
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, col in enumerate(["balanced_accuracy", "f1", "recall", "roc_auc", "pr_auc"]):
        ax.bar(x + (i - 2) * width, df[col].astype(float).values, width, label=col.upper())
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_xlabel("Model"); ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08)); ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=600); plt.close(fig)


def plot_lines(df: pd.DataFrame, out_path: Path, x_col: str, y_col: str, hue_col: str, title: str, ylabel: str):
    if df.empty: return
    fig, ax = plt.subplots(figsize=(16, 9))
    for name, g in df.groupby(hue_col):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", linewidth=2, label=str(name))
    ax.set_xlabel(x_col); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(out_path, dpi=600); plt.close(fig)


def plot_bar(df: pd.DataFrame, out_path: Path, x_col: str, y_col: str, title: str):
    if df.empty: return
    g = df.sort_values(y_col, ascending=False)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(g[x_col].astype(str), g[y_col].astype(float))
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(title)
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=600); plt.close(fig)


def save_feature_importance(model, features: List[str], out_csv: Path, out_fig: Path):
    try:
        clf = model.named_steps.get("clf") if isinstance(model, Pipeline) else None
        if clf is None: return
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).ravel()
        else:
            return
        if len(imp) != len(features): return
        fi = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
        fi.to_csv(out_csv, index=False)
        top = fi.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(16, 9)); ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Importance"); ax.set_title("Top Leakage-Safe Feature Importance")
        fig.tight_layout(); fig.savefig(out_fig, dpi=600); plt.close(fig)
    except Exception as e:
        log(f"[warn] feature importance skipped: {e}")


def run_ablations(df: pd.DataFrame, features: List[str], groups: Dict[str, List[str]], csv_dir: Path, fig_dir: Path, model_dir: Path, n: int, sampling_mode: str):
    ab_model = "XGB" if HAS_XGB else "HGB"
    log("[ablation] feature blocks")
    rows = []
    feature_sets = {
        "resource": groups["resource"], "dag": groups["dag"], "scheduling": groups["scheduling"],
        "pressure": groups["pressure"], "temporal": groups["temporal"],
        "complexity": groups["complexity"], "dag+pressure": sorted(set(groups["dag"] + groups["pressure"])),
        "all": groups["all"], "no_complexity": groups["no_complexity"]
    }
    for name, fs in feature_sets.items():
        if not fs: continue
        try:
            m, _, _ = train_single_model(df, fs, ab_model, n, split_mode="grouped_temporal", sampling_mode=sampling_mode)
            m["feature_set"] = name; rows.append(m)
        except Exception as e: log(f"[warn] feature block {name}: {e}")
    feat = pd.DataFrame(rows); feat.to_csv(csv_dir / "ablation_feature_blocks.csv", index=False)
    if not feat.empty: plot_bar(feat, fig_dir / "fig_ablation_feature_blocks.png", "feature_set", "pr_auc", "Feature Block Ablation by PR-AUC")

    log("[ablation] routing threshold")
    rows = []
    for tq in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        try:
            m, model = train_hqdnet(df, features, groups, n, tau_quantile=tq, sampling_mode=sampling_mode)
            m["tau_quantile"] = tq; rows.append(m)
            joblib.dump(model, model_dir / f"HQDNet_tau{tq}_n{n}.joblib")
        except Exception as e: log(f"[warn] tau {tq}: {e}")
    tau = pd.DataFrame(rows); tau.to_csv(csv_dir / "ablation_routing_threshold.csv", index=False)
    if not tau.empty: plot_lines(tau, fig_dir / "fig_ablation_routing_threshold.png", "tau_quantile", "pr_auc", "model", "Routing Threshold Ablation", "PR-AUC")

    log("[ablation] temporal drift")
    rows = []
    for split in ["grouped_temporal", "iid"]:
        for name in ["RF", ab_model]:
            try:
                m, _, _ = train_single_model(df, features, name, n, split_mode=split, sampling_mode=sampling_mode)
                m["split"] = split; rows.append(m)
            except Exception as e: log(f"[warn] drift {name}-{split}: {e}")
        try:
            m, _ = train_hqdnet(df, features, groups, n, tau_quantile=0.80, split_mode=split, sampling_mode=sampling_mode)
            m["split"] = split; rows.append(m)
        except Exception as e: log(f"[warn] drift HQD-Net-{split}: {e}")
    drift = pd.DataFrame(rows); drift.to_csv(csv_dir / "ablation_temporal_drift.csv", index=False)
    if not drift.empty: plot_lines(drift, fig_dir / "fig_ablation_temporal_drift.png", "split", "pr_auc", "model", "IID vs Temporal Drift Evaluation", "PR-AUC")

    log("[ablation] feature noise robustness")
    rows = []; base = sample_dataset(df, n, sampling_mode=sampling_mode)
    for sigma in [0.00, 0.01, 0.03, 0.05, 0.10]:
        noisy = base.copy(); rng = np.random.default_rng(RANDOM_STATE)
        if sigma > 0:
            for c in features:
                std = pd.to_numeric(noisy[c], errors="coerce").std()
                noisy[c] = pd.to_numeric(noisy[c], errors="coerce") + rng.normal(0, sigma * (std if pd.notna(std) and std > 0 else 1), len(noisy))
        for name in ["RF", ab_model]:
            try:
                m, _, _ = train_single_model(noisy, features, name, min(n, len(noisy)), split_mode="grouped_temporal", sampling_mode=sampling_mode)
                m["feature_noise_sigma"] = sigma; rows.append(m)
            except Exception as e: log(f"[warn] noise {name}-{sigma}: {e}")
    noise = pd.DataFrame(rows); noise.to_csv(csv_dir / "ablation_noise_robustness.csv", index=False)
    if not noise.empty: plot_lines(noise, fig_dir / "fig_ablation_noise_robustness.png", "feature_noise_sigma", "pr_auc", "model", "Noise Robustness Ablation", "PR-AUC")


def run_main(df: pd.DataFrame, out_dir: Path, sample_sizes: List[int], run_abl: bool, sampling_mode: str):
    fig_dir = ensure_dir(out_dir / "figures"); csv_dir = ensure_dir(out_dir / "csv_results")
    model_dir = ensure_dir(out_dir / "trained_models"); proc_dir = ensure_dir(out_dir / "processed_data")
    groups = get_feature_groups(df); features = groups["all"]
    if not features:
        raise RuntimeError("No valid leakage-safe features were detected.")
    class_counts = df["job_failure"].value_counts().rename_axis("class").reset_index(name="count")
    class_counts.to_csv(csv_dir / "raw_class_distribution.csv", index=False)
    df[["job_name", "job_failure"] + features].to_csv(proc_dir / "engineered_job_dataset_leakage_safe.csv", index=False)
    (out_dir / "feature_groups_leakage_safe.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (out_dir / "removed_leakage_features.json").write_text(json.dumps(LEAKAGE_FEATURES, indent=2), encoding="utf-8")

    base_models = ["LR", "RF", "HGB", "MLP"]
    if HAS_XGB: base_models.append("XGB")
    if HAS_LGBM: base_models.append("LGBM")

    rows = []
    for n in sample_sizes:
        log(f"\n[exp] sample_size={n}")
        for name in base_models:
            try:
                log(f"[train] {name} n={n}")
                m, model, _ = train_single_model(df, features, name, n, split_mode="grouped_temporal", sampling_mode=sampling_mode)
                rows.append(m); joblib.dump(model, model_dir / f"{name}_n{n}_leakage_safe.joblib")
                if n == max(sample_sizes) and name in ["LR", "RF", "XGB", "LGBM"]:
                    save_feature_importance(model, features, csv_dir / f"feature_importance_{name}_n{n}.csv", fig_dir / f"feature_importance_{name}_n{n}.png")
            except Exception as e: log(f"[error] {name} n={n}: {e}")
        try:
            log(f"[train] HQD-Net n={n}")
            m, model = train_hqdnet(df, features, groups, n, tau_quantile=0.80, sampling_mode=sampling_mode)
            rows.append(m); joblib.dump(model, model_dir / f"HQDNet_tau0.80_n{n}_leakage_safe.joblib")
        except Exception as e: log(f"[error] HQD-Net n={n}: {e}")

    metrics = pd.DataFrame(rows); metrics.to_csv(csv_dir / "main_metrics_all_sample_sizes.csv", index=False)
    plot_lines(metrics, fig_dir / "fig_scalability_pr_auc.png", "sample_size", "pr_auc", "model", "Grouped-Temporal Leakage-Safe Scalability Evaluation", "PR-AUC")
    plot_lines(metrics, fig_dir / "fig_scalability_f1.png", "sample_size", "f1", "model", "Grouped-Temporal Leakage-Safe F1 Scalability Evaluation", "F1")
    plot_lines(metrics, fig_dir / "fig_latency_ms_per_sample.png", "sample_size", "latency_ms_per_sample", "model", "Deployment Latency Evaluation", "Latency (ms/sample)")
    for n in sample_sizes:
        plot_metric_bars(metrics, fig_dir / f"fig_model_metrics_n{n}.png", f"Grouped-Temporal Leakage-Safe Predictive Performance at n={n}", sample_size=n)
    if run_abl:
        run_ablations(df, features, groups, csv_dir, fig_dir, model_dir, max(sample_sizes), sampling_mode)
    best = metrics.sort_values(["pr_auc", "f1"], ascending=False).head(12) if not metrics.empty else pd.DataFrame()
    readme = [
        "# HQD-Net v4 Grouped-Temporal Leakage-Safe Results Summary", "",
        f"Sampling mode: `{sampling_mode}`. v5 balances only the training split and evaluates on natural grouped-temporal validation/test partitions. It removes post-execution leakage features and reports failure-focused metrics.", "",
        "## Removed Leakage Features", "", "```", json.dumps(LEAKAGE_FEATURES, indent=2), "```", "",
        "## Top Models by PR-AUC", "", best.to_markdown(index=False) if not best.empty else "No results generated.", "",
        "## Key Files", "",
        "- `csv_results/raw_class_distribution.csv`", "- `csv_results/main_metrics_all_sample_sizes.csv`", "- `csv_results/ablation_feature_blocks.csv`", "- `csv_results/ablation_temporal_drift.csv`", "- `csv_results/ablation_routing_threshold.csv`", "- `csv_results/ablation_noise_robustness.csv`", "- `figures/`", "- `trained_models/`"
    ]
    (out_dir / "README_RESULTS.md").write_text("\n".join(readme), encoding="utf-8")
    log(f"[done] results saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="HQD-Net v4 balanced leakage-safe Alibaba job failure prediction")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[5000, 10000, 20000, 25000])
    p.add_argument("--max-batch-task-rows", type=int, default=2000000)
    p.add_argument("--max-batch-instance-rows", type=int, default=2000000)
    p.add_argument("--max-machine-usage-rows", type=int, default=3000000)
    p.add_argument("--use-machine-pressure", action="store_true", help="Attach causal machine pressure features. Slower but stronger.")
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--reuse-engineered", action="store_true")
    p.add_argument("--sampling-mode", choices=["train_balanced", "class_weight_only", "imbalanced"], default="train_balanced", help="v5 default balances only the training split; validation/test remain natural and time-held-out.")
    return p.parse_args()


def main():
    args = parse_args(); out_dir = ensure_dir(Path(args.out_dir)); proc = ensure_dir(out_dir / "processed_data") / "engineered_job_dataset_full_leakage_safe.csv"
    if args.reuse_engineered and proc.exists():
        log(f"[data] reusing {proc}"); df = pd.read_csv(proc)
    else:
        df = build_dataset(args); df.to_csv(proc, index=False); log(f"[data] saved engineered dataset to {proc}")
    df = df[df["job_failure"].notna()].copy()
    run_main(df, out_dir, args.sample_sizes, args.run_ablations, args.sampling_mode)


if __name__ == "__main__":
    main()
