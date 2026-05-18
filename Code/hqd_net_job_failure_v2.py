# -*- coding: utf-8 -*-
"""
HQD-Net v2: Hierarchical Quantum-Dependency Network for Alibaba v2018 Job Failure Prediction

Outputs:
  - cleaned/engineered dataset CSV
  - metrics CSVs
  - ablation CSVs
  - publication-style figures
  - trained models (.joblib)

Designed for:
  D:\other\ALIBABAQUATUM\Dataset
  D:\other\ALIBABAQUATUM\Results_HQDNet_v2

Run example:
python hqd_net_job_failure_v2.py ^
  --data-dir "D:\other\ALIBABAQUATUM\Dataset" ^
  --out-dir "D:\other\ALIBABAQUATUM\Results_HQDNet_v2" ^
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
import os
import re
import time
import warnings
from dataclasses import dataclass
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

# Optional Qiskit. The code works without it using a quantum-inspired structural encoder.
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

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
CONTAINER_USAGE_COLS = [
    "container_id", "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
    "cpi", "mem_gps", "mpki", "net_in", "net_out", "disk_io_percent"
]
MACHINE_META_COLS = [
    "machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
    "cpu_num", "mem_size", "status"
]
CONTAINER_META_COLS = [
    "container_id", "machine_id", "time_stamp", "app_du", "status",
    "cpu_request", "cpu_limit", "mem_size"
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_schema(path: Path, columns: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    """Robust Alibaba CSV reader. Handles normal header, duplicated header rows, and headerless rows."""
    log(f"[data] reading {path} ...")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)

    # If columns don't match expected, reread as headerless.
    expected_set = set(columns)
    current_set = set(map(str, df.columns))
    if len(expected_set.intersection(current_set)) < max(2, len(columns) // 3):
        df = pd.read_csv(path, nrows=max_rows, header=None, names=columns, low_memory=False)
    else:
        # Ensure canonical order where possible.
        rename = {c: str(c).strip() for c in df.columns}
        df = df.rename(columns=rename)
        if list(df.columns[: len(columns)]) != columns and len(df.columns) == len(columns):
            df.columns = columns

    # Drop duplicated header lines present inside large CSVs.
    first_col = columns[0]
    if first_col in df.columns:
        df = df[df[first_col].astype(str) != first_col].copy()

    log(f"[data] {path.name} shape={df.shape}")
    return df


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_task_index(task_name: object) -> int:
    s = str(task_name)
    nums = re.findall(r"\d+", s)
    if not nums:
        return 1
    try:
        return int(nums[0])
    except Exception:
        return 1


def parse_parent_indices(task_name: object) -> List[int]:
    s = str(task_name)
    nums = re.findall(r"\d+", s)
    if len(nums) <= 1:
        return []
    out = []
    for z in nums[1:]:
        try:
            out.append(int(z))
        except Exception:
            pass
    return out


def status_to_failure(status: object) -> int:
    s = str(status).lower().strip()
    success_tokens = ["terminated", "finished", "success", "completed", "finish"]
    fail_tokens = ["failed", "fail", "killed", "evicted", "error", "lost", "terminated with failure"]
    if any(t in s for t in fail_tokens):
        return 1
    if any(t in s for t in success_tokens):
        return 0
    return 0


def safe_div(a, b):
    return a / (b + 1e-9)


def build_machine_usage_features(data_dir: Path, max_rows: Optional[int]) -> pd.DataFrame:
    candidates = [data_dir / "machine_usage_bigger.csv", data_dir / "machine_usage.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        log("[warn] machine_usage file not found; using empty pressure features")
        return pd.DataFrame(columns=["machine_id"])
    mu = read_csv_schema(path, MACHINE_USAGE_COLS, max_rows=max_rows)
    mu = to_num(mu, ["time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"])
    for c in ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]:
        mu[c] = mu[c].replace([np.inf, -np.inf], np.nan)
    agg = mu.groupby("machine_id", observed=True).agg(
        machine_cpu_mean=("cpu_util_percent", "mean"),
        machine_cpu_std=("cpu_util_percent", "std"),
        machine_cpu_max=("cpu_util_percent", "max"),
        machine_mem_mean=("mem_util_percent", "mean"),
        machine_mem_std=("mem_util_percent", "std"),
        machine_mem_max=("mem_util_percent", "max"),
        machine_net_in_mean=("net_in", "mean"),
        machine_net_out_mean=("net_out", "mean"),
        machine_disk_mean=("disk_io_percent", "mean"),
        machine_records=("time_stamp", "count"),
    ).reset_index()
    agg["machine_pressure"] = (
        agg["machine_cpu_mean"].fillna(0) / 100.0 +
        agg["machine_mem_mean"].fillna(0) / 100.0 +
        agg["machine_disk_mean"].fillna(0) / 100.0
    ) / 3.0
    return agg


def build_container_usage_features(data_dir: Path, max_rows: Optional[int]) -> pd.DataFrame:
    path = data_dir / "container_usage.csv"
    if not path.exists():
        log("[warn] container_usage file not found; using empty container pressure features")
        return pd.DataFrame(columns=["machine_id"])
    cu = read_csv_schema(path, CONTAINER_USAGE_COLS, max_rows=max_rows)
    cu = to_num(cu, ["time_stamp", "cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent", "cpi", "mpki"])
    agg = cu.groupby("machine_id", observed=True).agg(
        container_cpu_mean=("cpu_util_percent", "mean"),
        container_cpu_std=("cpu_util_percent", "std"),
        container_mem_mean=("mem_util_percent", "mean"),
        container_mem_std=("mem_util_percent", "std"),
        container_disk_mean=("disk_io_percent", "mean"),
        container_count=("container_id", "nunique"),
    ).reset_index()
    agg["container_pressure"] = (
        agg["container_cpu_mean"].fillna(0) / 100.0 +
        agg["container_mem_mean"].fillna(0) / 100.0 +
        agg["container_disk_mean"].fillna(0) / 100.0
    ) / 3.0
    return agg


def build_task_dag_features(task_df: pd.DataFrame) -> pd.DataFrame:
    task_df = task_df.copy()
    task_df = to_num(task_df, ["instance_num", "task_type", "start_time", "end_time", "plan_cpu", "plan_mem"])
    task_df["task_duration"] = (task_df["end_time"] - task_df["start_time"]).clip(lower=0)
    task_df["task_index"] = task_df["task_name"].map(parse_task_index)
    task_df["parent_list"] = task_df["task_name"].map(parse_parent_indices)
    task_df["parent_count"] = task_df["parent_list"].map(len)
    task_df["is_root_task"] = (task_df["parent_count"] == 0).astype(int)
    task_df["task_failure"] = task_df["status"].map(status_to_failure).astype(int)

    # Approximate DAG depth by number of numeric dependency tokens + task index rank proxy.
    task_df["local_depth_proxy"] = task_df["parent_count"] + 1
    task_df["resource_product"] = task_df["plan_cpu"].fillna(0) * task_df["plan_mem"].fillna(0)

    g = task_df.groupby("job_name", observed=True)
    job = g.agg(
        task_count=("task_name", "count"),
        unique_task_count=("task_name", "nunique"),
        total_instances=("instance_num", "sum"),
        max_instances_per_task=("instance_num", "max"),
        mean_instances_per_task=("instance_num", "mean"),
        dag_depth=("local_depth_proxy", "max"),
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
        task_end_max=("end_time", "max"),
        task_duration_mean=("task_duration", "mean"),
        task_duration_max=("task_duration", "max"),
        task_fail_rate=("task_failure", "mean"),
    ).reset_index()

    job["job_span"] = (job["task_end_max"] - job["task_start_min"]).clip(lower=0)
    job["parallelism_proxy"] = safe_div(job["total_instances"], job["job_span"] + 1.0)
    job["dependency_density"] = safe_div(job["fan_in_mean"], job["task_count"])
    job["cpu_mem_ratio"] = safe_div(job["plan_cpu_mean"], job["plan_mem_mean"])
    job["critical_path_proxy"] = job["dag_depth"] * job["task_duration_max"].fillna(0)
    job["dag_complexity"] = (
        np.log1p(job["task_count"].fillna(0)) +
        np.log1p(job["total_instances"].fillna(0)) +
        job["dag_depth"].fillna(0) +
        job["fan_in_max"].fillna(0)
    )
    return job


def build_instance_features(inst_df: pd.DataFrame) -> pd.DataFrame:
    inst_df = inst_df.copy()
    inst_df = to_num(inst_df, ["task_type", "start_time", "end_time", "seq_no", "total_seq_no", "cpu_avg", "cpu_max", "mem_avg", "mem_max"])
    inst_df["inst_duration"] = (inst_df["end_time"] - inst_df["start_time"]).clip(lower=0)
    inst_df["instance_failure"] = inst_df["status"].map(status_to_failure).astype(int)
    inst_df["cpu_spike"] = inst_df["cpu_max"].fillna(0) - inst_df["cpu_avg"].fillna(0)
    inst_df["mem_spike"] = inst_df["mem_max"].fillna(0) - inst_df["mem_avg"].fillna(0)

    g = inst_df.groupby("job_name", observed=True)
    job = g.agg(
        instance_count=("instance_name", "count"),
        failed_instance_count=("instance_failure", "sum"),
        instance_fail_rate=("instance_failure", "mean"),
        inst_duration_mean=("inst_duration", "mean"),
        inst_duration_max=("inst_duration", "max"),
        inst_duration_std=("inst_duration", "std"),
        cpu_avg_mean=("cpu_avg", "mean"),
        cpu_avg_max=("cpu_avg", "max"),
        cpu_max_mean=("cpu_max", "mean"),
        cpu_max_max=("cpu_max", "max"),
        mem_avg_mean=("mem_avg", "mean"),
        mem_avg_max=("mem_avg", "max"),
        mem_max_mean=("mem_max", "mean"),
        mem_max_max=("mem_max", "max"),
        cpu_spike_mean=("cpu_spike", "mean"),
        mem_spike_mean=("mem_spike", "mean"),
        first_start_time=("start_time", "min"),
        last_end_time=("end_time", "max"),
        primary_machine_id=("machine_id", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        unique_machines=("machine_id", "nunique"),
    ).reset_index()
    # Label: job fails if any instance has failed/killed/error status.
    job["job_failure"] = (job["failed_instance_count"] > 0).astype(int)
    job["runtime_span"] = (job["last_end_time"] - job["first_start_time"]).clip(lower=0)
    return job


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("first_start_time").copy()
    t = pd.to_numeric(df["first_start_time"], errors="coerce").fillna(0)
    df["submit_hour"] = ((t // 3600) % 24).astype(float)
    df["submit_day"] = (t // (24 * 3600)).astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["submit_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["submit_hour"] / 24.0)
    # Leakage-light rolling feature: previous jobs only.
    df["prev_failure_rate_200"] = df["job_failure"].shift(1).rolling(200, min_periods=20).mean()
    df["prev_runtime_mean_200"] = df["runtime_span"].shift(1).rolling(200, min_periods=20).mean()
    df["local_arrival_density_200"] = 1.0 / (t.diff().rolling(200, min_periods=20).mean().replace(0, np.nan))
    return df


def minmax_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    x = x.clip(lo, hi)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)


def add_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    components = {
        "cx_dag": df.get("dag_complexity", 0),
        "cx_pressure":
(
    df["machine_pressure"].fillna(0)
    if "machine_pressure" in df.columns
    else 0
)
+
(
    df["container_pressure"].fillna(0)
    if "container_pressure" in df.columns
    else 0
),
        "cx_retry": df.get("failed_instance_count", 0),
        "cx_parallel": df.get("parallelism_proxy", 0),
        "cx_span": df.get("runtime_span", 0),
    }
    for k, v in components.items():
        df[k] = minmax_series(pd.Series(v, index=df.index))
    df["complexity_score"] = (
        0.30 * df["cx_dag"] +
        0.25 * df["cx_pressure"] +
        0.15 * df["cx_retry"] +
        0.15 * df["cx_parallel"] +
        0.15 * df["cx_span"]
    )
    return df


def build_dataset(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir)

    bt = read_csv_schema(data_dir / "batch_task.csv", BATCH_TASK_COLS, args.max_batch_task_rows)
    bi = read_csv_schema(data_dir / "batch_instance.csv", BATCH_INSTANCE_COLS, args.max_batch_instance_rows)

    task_feat = build_task_dag_features(bt)
    inst_feat = build_instance_features(bi)
    df = inst_feat.merge(task_feat, on="job_name", how="left")

    machine_feat = build_machine_usage_features(data_dir, args.max_machine_usage_rows)
    if not machine_feat.empty:
        df = df.merge(machine_feat, left_on="primary_machine_id", right_on="machine_id", how="left")

    if args.use_container_usage:
        cont_feat = build_container_usage_features(data_dir, args.max_container_usage_rows)
        if not cont_feat.empty:
            df = df.merge(cont_feat, left_on="primary_machine_id", right_on="machine_id", how="left", suffixes=("", "_container_merge"))

    df = add_temporal_features(df)
    df = add_complexity_score(df)

    # Drop non-predictor identifiers except useful routing label kept separately.
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"[data] engineered dataset shape={df.shape}, failure_rate={df['job_failure'].mean():.4f}")
    return df


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "resource": [
            "plan_cpu_mean", "plan_cpu_max", "plan_cpu_sum", "plan_mem_mean", "plan_mem_max", "plan_mem_sum",
            "cpu_mem_ratio", "resource_product_sum", "cpu_avg_mean", "cpu_avg_max", "cpu_max_mean", "cpu_max_max",
            "mem_avg_mean", "mem_avg_max", "mem_max_mean", "mem_max_max", "cpu_spike_mean", "mem_spike_mean"
        ],
        "dag": [
            "task_count", "unique_task_count", "total_instances", "max_instances_per_task", "mean_instances_per_task",
            "dag_depth", "fan_in_mean", "fan_in_max", "root_task_ratio", "parallelism_proxy",
            "dependency_density", "critical_path_proxy", "dag_complexity"
        ],
        "runtime": [
            "instance_count", "inst_duration_mean", "inst_duration_max", "inst_duration_std", "runtime_span",
            "job_span", "task_duration_mean", "task_duration_max"
        ],
        "pressure": [
            "machine_cpu_mean", "machine_cpu_std", "machine_cpu_max", "machine_mem_mean", "machine_mem_std",
            "machine_mem_max", "machine_net_in_mean", "machine_net_out_mean", "machine_disk_mean", "machine_pressure",
            "container_cpu_mean", "container_cpu_std", "container_mem_mean", "container_mem_std", "container_disk_mean",
            "container_count", "container_pressure", "unique_machines"
        ],
        "temporal": [
            "submit_hour", "submit_day", "sin_hour", "cos_hour", "prev_failure_rate_200",
            "prev_runtime_mean_200", "local_arrival_density_200", "first_start_time"
        ],
        "complexity": ["complexity_score", "cx_dag", "cx_pressure", "cx_retry", "cx_parallel", "cx_span"],
    }
    # Keep only columns present.
    groups = {k: [c for c in v if c in df.columns] for k, v in groups.items()}
    groups["all"] = sorted(set(sum(groups.values(), [])))
    groups["no_quantum_complexity"] = [c for c in groups["all"] if c not in groups["complexity"]]
    return groups


class QuantumStructuralEncoder(BaseEstimator, TransformerMixin):
    """
    Lightweight quantum-inspired encoder. It is CPU-safe and avoids full O(n^2) QSVM.
    It maps structural features through trigonometric interference terms similar to compact feature-map projections.
    """
    def __init__(self, n_components: int = 16, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.W_ = rng.normal(0, 1, size=(n_features, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=float) @ self.W_ + self.b_
        # Interference-style features.
        return np.hstack([np.cos(Z), np.sin(Z), np.cos(Z) * np.sin(Z)])


def make_preprocessor(features: List[str], scaler: str = "standard") -> ColumnTransformer:
    scale = StandardScaler() if scaler == "standard" else MinMaxScaler()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scale),
    ])
    return ColumnTransformer([("num", numeric_pipe, features)], remainder="drop")


def make_model(name: str, n_features: int = 1):
    if name == "LR":
        return LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None, random_state=RANDOM_STATE)
    if name == "RF":
        return RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=250, random_state=RANDOM_STATE, early_stopping=True)
    if name == "XGB" and HAS_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "LGBM" and HAS_LGBM:
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    raise ValueError(f"Model unavailable or unknown: {name}")


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X)


def metrics_dict(y_true, score, pred, train_time, infer_time, model_name, sample_size, route="all") -> Dict[str, object]:
    out = {
        "model": model_name,
        "sample_size": sample_size,
        "route": route,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "latency_ms_per_sample": (infer_time / max(1, len(y_true))) * 1000,
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, score)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, score)
    except Exception:
        out["pr_auc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out


def sample_balanced_or_stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1, random_state=RANDOM_STATE).copy()
    # Preserve class distribution by default. If failure class is tiny, stratification still works.
    return df.groupby("job_failure", group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(round(n * len(x) / len(df)))), random_state=RANDOM_STATE)
    ).sample(frac=1, random_state=RANDOM_STATE).head(n).copy()


def train_single_model(df: pd.DataFrame, features: List[str], model_name: str, sample_size: int, out_dir: Path, split_mode: str = "iid", scaler: str = "standard") -> Tuple[Dict[str, object], object, Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    data = sample_balanced_or_stratified(df, sample_size)
    X = data[features].copy()
    y = data["job_failure"].astype(int)

    if split_mode == "temporal" and "first_start_time" in data.columns:
        data_sorted = data.sort_values("first_start_time")
        X = data_sorted[features].copy()
        y = data_sorted["job_failure"].astype(int)
        cut = int(0.8 * len(data_sorted))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

    pipe = Pipeline([
        ("prep", make_preprocessor(features, scaler=scaler)),
        ("clf", make_model(model_name, len(features))),
    ])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    score = predict_scores(pipe, X_test)
    pred = (score >= 0.5).astype(int)
    infer_time = time.perf_counter() - t1
    m = metrics_dict(y_test, score, pred, train_time, infer_time, model_name, sample_size, route=split_mode)
    return m, pipe, (X_train, y_train, X_test, y_test)


class HQDNet:
    """Complexity-routed hybrid model: classical model for normal jobs, quantum-inspired branch for complex jobs."""
    def __init__(self, features: List[str], structural_features: List[str], tau_quantile: float = 0.80, classical_name: str = "XGB", quantum_name: str = "MLP", scaler: str = "standard"):
        self.features = features
        self.structural_features = [c for c in structural_features if c in features]
        self.tau_quantile = tau_quantile
        self.classical_name = classical_name if (classical_name != "XGB" or HAS_XGB) else "HGB"
        self.quantum_name = quantum_name
        self.scaler = scaler

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tau_ = float(np.nanquantile(X["complexity_score"].fillna(0), self.tau_quantile)) if "complexity_score" in X.columns else 0.8
        self.classical_pipe_ = Pipeline([
            ("prep", make_preprocessor(self.features, self.scaler)),
            ("clf", make_model(self.classical_name, len(self.features))),
        ])
        self.classical_pipe_.fit(X[self.features], y)

        q_features = self.structural_features if self.structural_features else self.features
        self.q_features_ = q_features
        self.quantum_pipe_ = Pipeline([
            ("prep", make_preprocessor(q_features, "minmax")),
            ("qenc", QuantumStructuralEncoder(n_components=16, random_state=RANDOM_STATE)),
            ("clf", make_model(self.quantum_name, len(q_features))),
        ])
        route_mask = X["complexity_score"].fillna(0).values >= self.tau_
        # Ensure enough data for the quantum branch.
        if route_mask.sum() < max(50, int(0.05 * len(X))):
            route_mask = X["complexity_score"].fillna(0).rank(pct=True).values >= 0.70
        self.route_train_ratio_ = float(route_mask.mean())
        self.quantum_pipe_.fit(X.loc[route_mask, q_features], y.loc[route_mask])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pc = predict_scores(self.classical_pipe_, X[self.features])
        pq = np.zeros_like(pc, dtype=float)
        route_mask = X["complexity_score"].fillna(0).values >= self.tau_ if "complexity_score" in X.columns else np.zeros(len(X), dtype=bool)
        if route_mask.any():
            pq[route_mask] = predict_scores(self.quantum_pipe_, X.loc[route_mask, self.q_features_])
        # Adaptive fusion: complex route uses more quantum branch, normal route uses classical only.
        score = pc.copy()
        score[route_mask] = 0.60 * pq[route_mask] + 0.40 * pc[route_mask]
        return np.vstack([1 - score, score]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_hqdnet(df: pd.DataFrame, features: List[str], groups: Dict[str, List[str]], sample_size: int, tau_quantile: float, out_dir: Path, split_mode: str = "iid") -> Tuple[Dict[str, object], HQDNet]:
    data = sample_balanced_or_stratified(df, sample_size)
    if split_mode == "temporal" and "first_start_time" in data.columns:
        data = data.sort_values("first_start_time")
        cut = int(0.8 * len(data))
        train, test = data.iloc[:cut], data.iloc[cut:]
    else:
        train, test = train_test_split(data, test_size=0.2, stratify=data["job_failure"], random_state=RANDOM_STATE)

    y_train = train["job_failure"].astype(int)
    y_test = test["job_failure"].astype(int)
    structural = sorted(set(groups.get("dag", []) + groups.get("pressure", []) + groups.get("complexity", [])))
    model = HQDNet(features=features, structural_features=structural, tau_quantile=tau_quantile)
    t0 = time.perf_counter()
    model.fit(train[features], y_train)
    train_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    score = model.predict_proba(test[features])[:, 1]
    pred = (score >= 0.5).astype(int)
    infer_time = time.perf_counter() - t1
    m = metrics_dict(y_test, score, pred, train_time, infer_time, f"HQD-Net(tau={tau_quantile:.2f})", sample_size, route=split_mode)
    m["quantum_route_train_ratio"] = model.route_train_ratio_
    m["tau_threshold"] = model.tau_
    return m, model


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path, title: str, sample_size: Optional[int] = None):
    df = metrics_df.copy()
    if sample_size is not None:
        df = df[df["sample_size"] == sample_size]
    if df.empty:
        return
    df = df.sort_values("f1", ascending=False)
    labels = df["model"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, col in enumerate(["accuracy", "f1", "roc_auc", "pr_auc"]):
        ax.bar(x + (i - 1.5) * width, df[col].astype(float).values, width, label=col.upper())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_scalability(metrics_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    for model, g in metrics_df.groupby("model"):
        g = g.sort_values("sample_size")
        ax.plot(g["sample_size"], g["f1"], marker="o", linewidth=2, label=str(model))
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("F1 Score")
    ax.set_title("Scalability Evaluation Across Sample Sizes")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_latency(metrics_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    for model, g in metrics_df.groupby("model"):
        g = g.sort_values("sample_size")
        ax.plot(g["sample_size"], g["latency_ms_per_sample"], marker="s", linewidth=2, label=str(model))
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Latency (ms/sample)")
    ax.set_title("Deployment Latency Evaluation")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame, out_path: Path, x_col: str, y_col: str, hue_col: Optional[str], title: str):
    fig, ax = plt.subplots(figsize=(16, 9))
    if hue_col and hue_col in ablation_df.columns:
        for name, g in ablation_df.groupby(hue_col):
            g = g.sort_values(x_col)
            ax.plot(g[x_col].astype(str), g[y_col], marker="o", linewidth=2, label=str(name))
        ax.legend(ncol=3)
    else:
        g = ablation_df.sort_values(x_col)
        ax.bar(g[x_col].astype(str), g[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def save_feature_importance(model, features: List[str], out_csv: Path, out_fig: Path):
    try:
        clf = model.named_steps.get("clf") if isinstance(model, Pipeline) else None
        imp = None
        if clf is not None and hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif clf is not None and hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).ravel()
        if imp is None or len(imp) != len(features):
            return
        fi = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
        fi.to_csv(out_csv, index=False)
        top = fi.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importance")
        fig.tight_layout()
        fig.savefig(out_fig, dpi=600)
        plt.close(fig)
    except Exception as e:
        log(f"[warn] feature importance skipped: {e}")


def run_main_experiments(df: pd.DataFrame, out_dir: Path, sample_sizes: List[int], run_ablations: bool):
    fig_dir = ensure_dir(out_dir / "figures")
    model_dir = ensure_dir(out_dir / "trained_models")
    csv_dir = ensure_dir(out_dir / "csv_results")
    data_dir = ensure_dir(out_dir / "processed_data")

    groups = get_feature_groups(df)
    features = groups["all"]
    df[["job_name", "job_failure"] + features].to_csv(data_dir / "engineered_job_dataset.csv", index=False)
    with open(out_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)

    base_models = ["LR", "RF", "HGB", "MLP"]
    if HAS_XGB:
        base_models.append("XGB")
    if HAS_LGBM:
        base_models.append("LGBM")

    all_metrics = []
    for n in sample_sizes:
        log(f"\n[exp] sample_size={n}")
        for model_name in base_models:
            try:
                log(f"[train] {model_name} n={n}")
                m, model, _ = train_single_model(df, features, model_name, n, out_dir, split_mode="iid")
                all_metrics.append(m)
                joblib.dump(model, model_dir / f"{model_name}_n{n}.joblib")
                if n == max(sample_sizes) and model_name in ["RF", "XGB", "LGBM", "HGB"]:
                    save_feature_importance(model, features, csv_dir / f"feature_importance_{model_name}_n{n}.csv", fig_dir / f"feature_importance_{model_name}_n{n}.png")
            except Exception as e:
                log(f"[error] {model_name} n={n}: {e}")

        for tq in [0.80]:
            try:
                log(f"[train] HQD-Net n={n}")
                m, hqd = train_hqdnet(df, features, groups, n, tau_quantile=tq, out_dir=out_dir)
                all_metrics.append(m)
                joblib.dump(hqd, model_dir / f"HQDNet_tau{tq}_n{n}.joblib")
            except Exception as e:
                log(f"[error] HQD-Net n={n}: {e}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(csv_dir / "main_metrics_all_sample_sizes.csv", index=False)
    plot_scalability(metrics_df, fig_dir / "fig_scalability_f1.png")
    plot_latency(metrics_df, fig_dir / "fig_latency_ms_per_sample.png")
    for n in sample_sizes:
        plot_metrics_bar(metrics_df, fig_dir / f"fig_model_metrics_n{n}.png", f"Main Predictive Performance at n={n}", sample_size=n)

    if run_ablations:
        run_all_ablations(df, features, groups, csv_dir, fig_dir, model_dir, max(sample_sizes))

    # Summary markdown
    best = metrics_df.sort_values("f1", ascending=False).head(10)
    readme = ["# HQD-Net v2 Results Summary", "", "## Top Models by F1", "", best.to_markdown(index=False)]
    (out_dir / "README_RESULTS.md").write_text("\n".join(readme), encoding="utf-8")
    log(f"[done] results saved to {out_dir}")


def run_all_ablations(df: pd.DataFrame, features: List[str], groups: Dict[str, List[str]], csv_dir: Path, fig_dir: Path, model_dir: Path, n: int):
    log("\n[ablation] running feature-block ablations")
    ab_rows = []
    feature_sets = {
        "resource": groups["resource"],
        "dag": groups["dag"],
        "runtime": groups["runtime"],
        "pressure": groups["pressure"],
        "temporal": groups["temporal"],
        "dag+pressure": sorted(set(groups["dag"] + groups["pressure"])),
        "all": groups["all"],
        "no_complexity": groups["no_quantum_complexity"],
    }
    ab_model = "XGB" if HAS_XGB else "HGB"
    for fs_name, fs in feature_sets.items():
        if len(fs) == 0:
            continue
        try:
            m, model, _ = train_single_model(df, fs, ab_model, n, csv_dir, split_mode="iid")
            m["feature_set"] = fs_name
            ab_rows.append(m)
        except Exception as e:
            log(f"[warn] feature ablation {fs_name} skipped: {e}")
    feat_ab = pd.DataFrame(ab_rows)
    feat_ab.to_csv(csv_dir / "ablation_feature_blocks.csv", index=False)
    if not feat_ab.empty:
        plot_ablation(feat_ab, fig_dir / "fig_ablation_feature_blocks.png", "feature_set", "f1", None, "Feature-Block Ablation")

    log("[ablation] running routing threshold ablations")
    tau_rows = []
    for tq in [0.60, 0.70, 0.80, 0.90, 0.95]:
        try:
            m, model = train_hqdnet(df, features, groups, n, tau_quantile=tq, out_dir=csv_dir)
            m["tau_quantile"] = tq
            tau_rows.append(m)
            joblib.dump(model, model_dir / f"HQDNet_ablation_tau{tq}_n{n}.joblib")
        except Exception as e:
            log(f"[warn] tau ablation {tq} skipped: {e}")
    tau_ab = pd.DataFrame(tau_rows)
    tau_ab.to_csv(csv_dir / "ablation_routing_threshold.csv", index=False)
    if not tau_ab.empty:
        plot_ablation(tau_ab, fig_dir / "fig_ablation_routing_threshold.png", "tau_quantile", "f1", None, "Complexity Routing Threshold Ablation")

    log("[ablation] running temporal-vs-iid drift test")
    drift_rows = []
    for split in ["iid", "temporal"]:
        for model_name in (["RF"] + (["XGB"] if HAS_XGB else ["HGB"])):
            try:
                m, model, _ = train_single_model(df, features, model_name, n, csv_dir, split_mode=split)
                m["split"] = split
                drift_rows.append(m)
            except Exception as e:
                log(f"[warn] drift {model_name}-{split} skipped: {e}")
        try:
            m, model = train_hqdnet(df, features, groups, n, tau_quantile=0.80, out_dir=csv_dir, split_mode=split)
            m["split"] = split
            drift_rows.append(m)
        except Exception as e:
            log(f"[warn] drift HQD-Net-{split} skipped: {e}")
    drift_ab = pd.DataFrame(drift_rows)
    drift_ab.to_csv(csv_dir / "ablation_temporal_drift.csv", index=False)
    if not drift_ab.empty:
        plot_ablation(drift_ab, fig_dir / "fig_ablation_temporal_drift.png", "split", "f1", "model", "IID vs Temporal Drift Evaluation")

    log("[ablation] running noise robustness")
    noise_rows = []
    base = sample_balanced_or_stratified(df, n).copy()
    numeric_features = features
    for noise in [0.00, 0.01, 0.03, 0.05]:
        noisy = base.copy()
        if noise > 0:
            rng = np.random.default_rng(RANDOM_STATE)
            for c in numeric_features:
                if c in noisy.columns:
                    std = pd.to_numeric(noisy[c], errors="coerce").std()
                    noisy[c] = pd.to_numeric(noisy[c], errors="coerce") + rng.normal(0, noise * (std if std and not np.isnan(std) else 1), size=len(noisy))
        for model_name in (["RF"] + (["XGB"] if HAS_XGB else ["HGB"])):
            try:
                m, _, _ = train_single_model(noisy, features, model_name, min(n, len(noisy)), csv_dir, split_mode="iid")
                m["feature_noise_sigma"] = noise
                noise_rows.append(m)
            except Exception as e:
                log(f"[warn] noise {model_name}-{noise} skipped: {e}")
    noise_ab = pd.DataFrame(noise_rows)
    noise_ab.to_csv(csv_dir / "ablation_noise_robustness.csv", index=False)
    if not noise_ab.empty:
        plot_ablation(noise_ab, fig_dir / "fig_ablation_noise_robustness.png", "feature_noise_sigma", "f1", "model", "Noise Robustness Ablation")


def parse_args():
    p = argparse.ArgumentParser(description="HQD-Net v2 Alibaba job failure prediction")
    p.add_argument("--data-dir", required=True, help="Dataset folder")
    p.add_argument("--out-dir", required=True, help="Output folder")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=[5000, 10000, 20000, 25000])
    p.add_argument("--max-batch-task-rows", type=int, default=2000000)
    p.add_argument("--max-batch-instance-rows", type=int, default=2000000)
    p.add_argument("--max-machine-usage-rows", type=int, default=3000000)
    p.add_argument("--max-container-usage-rows", type=int, default=1000000)
    p.add_argument("--use-container-usage", action="store_true", help="Include container_usage aggregation. Slower.")
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--reuse-engineered", action="store_true", help="Reuse processed_data/engineered_job_dataset_full.csv if present")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    processed = ensure_dir(out_dir / "processed_data") / "engineered_job_dataset_full.csv"

    if args.reuse_engineered and processed.exists():
        log(f"[data] reusing {processed}")
        df = pd.read_csv(processed)
    else:
        df = build_dataset(args)
        df.to_csv(processed, index=False)
        log(f"[data] saved engineered full dataset to {processed}")

    # Remove rows without labels and impossible all-null records.
    df = df[df["job_failure"].notna()].copy()
    run_main_experiments(df, out_dir, args.sample_sizes, args.run_ablations)


if __name__ == "__main__":
    main()
