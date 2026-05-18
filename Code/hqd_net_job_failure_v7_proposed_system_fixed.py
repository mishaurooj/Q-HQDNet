#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HQD-Net-v7: Leakage-Safe Proposed Hybrid System for Alibaba Cloud Job Failure Prediction
=====================================================================================

This script is designed for IEEE-style experiments where the proposed model must be
reported as a complete system rather than a weak isolated branch.

Core design:
1. Leakage-safe feature engineering: removes post-execution and label-derived leakage.
2. Grouped-temporal evaluation: train/val/test ordered by first_start_time; no random test leakage.
3. Train-only balancing: validation and test keep natural future distribution.
4. Strong classical baselines: LR, RF, HGB, XGB, LGBM.
5. Proposed HQD-Net-v7:
   - LGBM deployment backbone.
   - XGB auxiliary learner.
   - graph/quantum-inspired structural specialist.
   - hard-example routing using ensemble uncertainty + dependency complexity.
   - validation-tuned fusion weights and decision threshold.
6. Saves CSVs, figures, trained models, feature importances, ablations, and README.

Expected input files in --data-dir:
- batch_task.csv
- batch_instance.csv
- machine_usage_bigger.csv or machine_usage.csv optional

Run example:
python hqd_net_job_failure_v7_proposed_system.py ^
  --data-dir "D:\\other\\ALIBABAQUATUM\\Dataset" ^
  --out-dir "D:\\other\\ALIBABAQUATUM\\Results_HQDNet_v7" ^
  --sample-sizes 5000 10000 20000 25000 50000 100000 ^
  --max-batch-task-rows 5000000 ^
  --max-batch-instance-rows 5000000 ^
  --max-machine-usage-rows 5000000 ^
  --use-machine-pressure ^
  --sampling-mode train_balanced ^
  --run-ablations
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

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

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BATCH_TASK_COLUMNS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem"
]
BATCH_INSTANCE_COLUMNS = [
    "instance_name", "task_name", "job_name", "task_type", "status", "start_time",
    "end_time", "machine_id", "seq_no", "total_seq_no", "cpu_avg", "cpu_max",
    "mem_avg", "mem_max"
]
MACHINE_USAGE_COLUMNS = [
    "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps",
    "mkpi", "net_in", "net_out", "disk_io_percent"
]

LEAKAGE_FEATURES = [
    "end_time", "task_end_max", "last_end_time", "task_duration_mean",
    "task_duration_max", "inst_duration_mean", "inst_duration_max",
    "inst_duration_std", "runtime_span", "job_span", "failed_instance_count",
    "instance_fail_rate", "task_fail_rate", "prev_runtime_mean_200", "status",
    "task_failure", "instance_failure", "cpu_avg_mean", "cpu_avg_max",
    "cpu_max_mean", "cpu_max_max", "mem_avg_mean", "mem_avg_max", "mem_max_mean",
    "mem_max_max", "cpu_spike_mean", "mem_spike_mean", "target", "label"
]

META_COLUMNS = ["job_name", "first_start_time", "dag_template", "target"]

# --------------------------------------------------------------------------------------
# IO utilities
# --------------------------------------------------------------------------------------

def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "csv": out_dir / "csv_results",
        "fig": out_dir / "figures",
        "models": out_dir / "trained_models",
        "data": out_dir / "processed_data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def read_csv_robust(path: Path, columns: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[data] reading {path} ...")
    # Many Alibaba files contain header row. We read normally first.
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    # If names are wrong or numeric, force known names.
    if len(df.columns) != len(columns) or any(str(c).startswith("Unnamed") for c in df.columns):
        df = pd.read_csv(path, nrows=max_rows, header=None, names=columns, low_memory=False)
    else:
        # If first row was read as columns but columns are exact or near exact, keep.
        # If the first data row repeats column names, remove below.
        df.columns = [str(c).strip() for c in df.columns]
        if set(columns).issubset(set(df.columns)):
            df = df[columns]
        elif len(df.columns) == len(columns):
            df.columns = columns
        else:
            df = pd.read_csv(path, nrows=max_rows, header=None, names=columns, low_memory=False)
    # Remove repeated header rows inside file.
    for c in columns:
        if c in df.columns:
            df = df[df[c].astype(str) != c]
    print(f"[data] {path.name} shape={df.shape}")
    return df.reset_index(drop=True)


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------

def parse_task_indices(task_name: Any) -> Tuple[Optional[int], List[int]]:
    """Parse Alibaba DAG-style task names, e.g., J4_2_3 -> node 4, parents [2,3]."""
    s = str(task_name)
    nums = re.findall(r"\d+", s)
    if not nums:
        return None, []
    node = int(nums[0])
    parents = [int(x) for x in nums[1:]]
    return node, parents


def task_template(task_name: Any) -> str:
    s = str(task_name)
    nums = re.findall(r"\d+", s)
    if not nums:
        return "NA"
    prefix = re.sub(r"\d+", "N", s[:1] if s else "X")
    return f"{prefix}_{len(nums)}_{'_'.join(['p'] * max(0, len(nums)-1))}"


def build_dag_features(task_df: pd.DataFrame) -> pd.DataFrame:
    df = task_df.copy()
    df = to_numeric(df, ["instance_num", "task_type", "start_time", "end_time", "plan_cpu", "plan_mem"])
    df["task_node"] = None
    df["fan_in"] = 0
    df["root_flag"] = 0
    df["task_template_unit"] = df["task_name"].map(task_template)

    nodes = []
    fanins = []
    roots = []
    for val in df["task_name"].values:
        node, parents = parse_task_indices(val)
        nodes.append(node if node is not None else -1)
        fanins.append(len(parents))
        roots.append(1 if len(parents) == 0 else 0)
    df["task_node"] = nodes
    df["fan_in"] = fanins
    df["root_flag"] = roots

    # Target at job level: any task not Terminated is failure. Conservative binary label.
    df["task_failed_label"] = (~df["status"].astype(str).str.lower().eq("terminated")).astype(int)
    df["resource_product"] = df["plan_cpu"].fillna(0) * df["plan_mem"].fillna(0)
    df["cpu_mem_ratio"] = df["plan_cpu"] / (df["plan_mem"].replace(0, np.nan))

    g = df.groupby("job_name", sort=False)
    out = g.agg(
        first_start_time=("start_time", "min"),
        task_count=("task_name", "count"),
        unique_task_count=("task_name", "nunique"),
        total_instances=("instance_num", "sum"),
        max_instances_per_task=("instance_num", "max"),
        mean_instances_per_task=("instance_num", "mean"),
        dag_depth=("task_node", "max"),
        fan_in_mean=("fan_in", "mean"),
        fan_in_max=("fan_in", "max"),
        root_task_ratio=("root_flag", "mean"),
        plan_cpu_mean=("plan_cpu", "mean"),
        plan_cpu_max=("plan_cpu", "max"),
        plan_cpu_sum=("plan_cpu", "sum"),
        plan_mem_mean=("plan_mem", "mean"),
        plan_mem_max=("plan_mem", "max"),
        plan_mem_sum=("plan_mem", "sum"),
        resource_product_sum=("resource_product", "sum"),
        cpu_mem_ratio=("cpu_mem_ratio", "mean"),
        target=("task_failed_label", "max"),
    ).reset_index()

    out["parallelism_proxy"] = out["total_instances"] / (out["unique_task_count"].replace(0, np.nan))
    out["dependency_density"] = out["fan_in_mean"] / (out["unique_task_count"].replace(0, np.nan))
    out["critical_path_proxy"] = out["dag_depth"] * np.log1p(out["max_instances_per_task"].fillna(0))
    out["dag_complexity"] = (
        np.log1p(out["unique_task_count"].fillna(0))
        + np.log1p(out["dag_depth"].fillna(0))
        + np.log1p(out["fan_in_max"].fillna(0))
        + np.log1p(out["parallelism_proxy"].fillna(0))
    )

    templ = df.groupby("job_name")["task_template_unit"].apply(lambda s: "|".join(sorted(s.astype(str).unique())[:50])).reset_index()
    templ.columns = ["job_name", "dag_template"]
    out = out.merge(templ, on="job_name", how="left")
    return out


def build_instance_features(inst_df: pd.DataFrame) -> pd.DataFrame:
    df = inst_df.copy()
    df = to_numeric(df, ["start_time", "end_time", "seq_no", "total_seq_no"])
    g = df.groupby("job_name", sort=False)
    out = g.agg(
        instance_count=("instance_name", "count"),
        unique_machines=("machine_id", "nunique"),
        seq_no_mean=("seq_no", "mean"),
        total_seq_no_mean=("total_seq_no", "mean"),
        first_machine_id=("machine_id", "first"),
    ).reset_index()
    # Do not use instance status, duration, cpu/mem observed usage: post-execution leakage.
    return out


def build_machine_pressure(machine_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if machine_df is None or machine_df.empty:
        return None
    df = machine_df.copy()
    df = to_numeric(df, ["time_stamp", "cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"])
    # Coarse machine-level aggregate. This is not exact time-window pressure, but safe if interpreted as historical baseline.
    g = df.groupby("machine_id", sort=False)
    out = g.agg(
        machine_cpu_mean=("cpu_util_percent", "mean"),
        machine_cpu_max=("cpu_util_percent", "max"),
        machine_mem_mean=("mem_util_percent", "mean"),
        machine_mem_max=("mem_util_percent", "max"),
        machine_net_in_mean=("net_in", "mean"),
        machine_net_out_mean=("net_out", "mean"),
        machine_disk_mean=("disk_io_percent", "mean"),
    ).reset_index()
    out["machine_pressure"] = (
        out["machine_cpu_mean"].fillna(0)/100.0
        + out["machine_mem_mean"].fillna(0)/100.0
        + out["machine_disk_mean"].fillna(0)/100.0
    ) / 3.0
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("first_start_time").reset_index(drop=True)
    t = pd.to_numeric(df["first_start_time"], errors="coerce").fillna(0)
    seconds_per_day = 24 * 3600
    df["submit_hour"] = ((t // 3600) % 24).astype(float)
    df["submit_day"] = (t // seconds_per_day).astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["submit_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["submit_hour"] / 24.0)
    # Historical only rolling features. Shift before rolling to prevent current label leakage.
    df["prev_failure_rate_200"] = df["target"].shift(1).rolling(200, min_periods=10).mean().fillna(0)
    gaps = df["first_start_time"].diff().fillna(0).clip(lower=0)
    df["prev_arrival_gap_200"] = gaps.shift(1).rolling(200, min_periods=10).mean().fillna(gaps.median())
    df["local_arrival_density_200"] = 1.0 / (df["prev_arrival_gap_200"].fillna(0) + 1.0)
    return df


def robust_minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    lo, hi = np.nanpercentile(x, [1, 99]) if len(x) else (0, 1)
    if hi <= lo:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def add_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cx_dag"] = robust_minmax(df.get("dag_complexity", pd.Series(0, index=df.index)))
    df["cx_pressure"] = robust_minmax(df.get("machine_pressure", pd.Series(0, index=df.index)))
    df["cx_repetition"] = robust_minmax(df.get("total_seq_no_mean", pd.Series(0, index=df.index)))
    df["cx_parallel"] = robust_minmax(df.get("parallelism_proxy", pd.Series(0, index=df.index)))
    df["cx_resource"] = robust_minmax(df.get("resource_product_sum", pd.Series(0, index=df.index)))
    df["complexity_score"] = (
        0.30 * df["cx_dag"] +
        0.20 * df["cx_pressure"] +
        0.15 * df["cx_repetition"] +
        0.20 * df["cx_parallel"] +
        0.15 * df["cx_resource"]
    )
    return df


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_FEATURES if c in df.columns and c not in ["target"]], errors="ignore")


def build_dataset(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    task = read_csv_robust(data_dir / "batch_task.csv", BATCH_TASK_COLUMNS, args.max_batch_task_rows)
    inst = read_csv_robust(data_dir / "batch_instance.csv", BATCH_INSTANCE_COLUMNS, args.max_batch_instance_rows)

    job_df = build_dag_features(task)
    inst_feat = build_instance_features(inst)
    job_df = job_df.merge(inst_feat, on="job_name", how="left")

    if args.use_machine_pressure:
        mpath = data_dir / "machine_usage_bigger.csv"
        if not mpath.exists():
            mpath = data_dir / "machine_usage.csv"
        if mpath.exists():
            mach = read_csv_robust(mpath, MACHINE_USAGE_COLUMNS, args.max_machine_usage_rows)
            mfeat = build_machine_pressure(mach)
            if mfeat is not None:
                job_df = job_df.merge(mfeat, left_on="first_machine_id", right_on="machine_id", how="left")
                job_df = job_df.drop(columns=["machine_id"], errors="ignore")

    job_df = add_temporal_features(job_df)
    job_df = add_complexity_features(job_df)
    job_df = remove_leakage_features(job_df)

    # Drop raw object columns except metadata. Keep dag_template for grouping but not training.
    job_df["target"] = pd.to_numeric(job_df["target"], errors="coerce").fillna(0).astype(int)
    job_df = job_df.replace([np.inf, -np.inf], np.nan)
    print(f"[data] built job-level dataset shape={job_df.shape}; failure_ratio={job_df['target'].mean():.4f}")
    return job_df

# --------------------------------------------------------------------------------------
# Feature groups and splits
# --------------------------------------------------------------------------------------

def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "resource": ["plan_cpu_mean", "plan_cpu_max", "plan_cpu_sum", "plan_mem_mean", "plan_mem_max", "plan_mem_sum", "cpu_mem_ratio", "resource_product_sum"],
        "dag": ["task_count", "unique_task_count", "total_instances", "max_instances_per_task", "mean_instances_per_task", "dag_depth", "fan_in_mean", "fan_in_max", "root_task_ratio", "parallelism_proxy", "dependency_density", "critical_path_proxy", "dag_complexity"],
        "scheduling": ["instance_count", "unique_machines", "seq_no_mean", "total_seq_no_mean"],
        "pressure": ["machine_cpu_mean", "machine_cpu_max", "machine_mem_mean", "machine_mem_max", "machine_net_in_mean", "machine_net_out_mean", "machine_disk_mean", "machine_pressure"],
        "temporal": ["submit_hour", "submit_day", "sin_hour", "cos_hour", "prev_failure_rate_200", "prev_arrival_gap_200", "local_arrival_density_200", "first_start_time"],
        "complexity": ["complexity_score", "cx_dag", "cx_pressure", "cx_repetition", "cx_parallel", "cx_resource"],
    }
    groups = {k: [c for c in v if c in df.columns] for k, v in groups.items()}
    all_features = sorted(set(sum(groups.values(), [])))
    groups["all"] = all_features
    groups["no_complexity"] = [c for c in all_features if c not in groups.get("complexity", [])]
    return groups


def sample_by_time(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.sort_values("first_start_time").reset_index(drop=True)
    if n is None or n <= 0 or n >= len(df):
        return df.copy()
    # Keep earliest-to-latest window up to n for temporal consistency. Sample across full range if requested.
    idx = np.linspace(0, len(df)-1, n).astype(int)
    return df.iloc[idx].drop_duplicates("job_name").sort_values("first_start_time").reset_index(drop=True)


def grouped_temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("first_start_time").drop_duplicates("job_name").reset_index(drop=True)
    n = len(df)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    # Remove accidental overlap by job name and template from val/test if present.
    train_jobs = set(train["job_name"].astype(str))
    val = val[~val["job_name"].astype(str).isin(train_jobs)].copy()
    test = test[~test["job_name"].astype(str).isin(train_jobs)].copy()
    return train, val, test


def balance_train(train: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode != "train_balanced":
        return train.copy()
    pos = train[train["target"] == 1]
    neg = train[train["target"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return train.copy()
    n = min(len(pos), len(neg))
    pos_s = resample(pos, replace=False if len(pos) >= n else True, n_samples=n, random_state=RANDOM_STATE)
    neg_s = resample(neg, replace=False if len(neg) >= n else True, n_samples=n, random_state=RANDOM_STATE)
    return pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# --------------------------------------------------------------------------------------
# Models and metrics
# --------------------------------------------------------------------------------------

def make_preprocessor(features: List[str]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_model(name: str):
    if name == "LR":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=None)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, l2_regularization=0.05, random_state=RANDOM_STATE)
    if name == "XGB":
        if not HAS_XGB:
            return None
        return XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.5, reg_alpha=0.05, eval_metric="logloss", random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1
        )
    if name == "LGBM":
        if not HAS_LGBM:
            return None
        return LGBMClassifier(
            n_estimators=450, max_depth=-1, num_leaves=31, learning_rate=0.035,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, class_weight=None,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    return None


def fit_pipeline(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Pipeline]:
    model = make_model(name)
    if model is None:
        return None
    pipe = Pipeline([
        ("prep", make_preprocessor(list(X_train.columns))),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def predict_proba_safe(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    dec = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-dec))


def best_threshold(y_val: np.ndarray, p_val: np.ndarray, objective: str = "f1") -> float:
    if len(np.unique(y_val)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5
    if objective == "mcc":
        best_t, best_s = 0.5, -999
        for t in np.unique(np.quantile(p_val, np.linspace(0.02, 0.98, 80))):
            pred = (p_val >= t).astype(int)
            s = matthews_corrcoef(y_val, pred) if len(np.unique(pred)) > 1 else -1
            if s > best_s:
                best_s, best_t = s, float(t)
        return best_t
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx])


def eval_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "roc_auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return out

# --------------------------------------------------------------------------------------
# Quantum-inspired structural specialist and HQD-Net-v7
# --------------------------------------------------------------------------------------

class QuantumInspiredStructuralEncoder(BaseEstimator, TransformerMixin):
    """Random Fourier feature map for dependency/complexity features.

    This acts as a scalable quantum-inspired structural representation. It does not
    claim hardware quantum advantage; it creates high-dimensional sinusoidal structural
    interactions over DAG, pressure, and complexity descriptors.
    """
    def __init__(self, n_components: int = 64, gamma: float = 0.7, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        self.W_ = rng.normal(0, math.sqrt(2 * self.gamma), size=(X.shape[1], self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.b_)
        return Z


def make_structural_features(X: pd.DataFrame) -> List[str]:
    keys = [
        "dag_depth", "fan_in_mean", "fan_in_max", "parallelism_proxy", "dependency_density",
        "critical_path_proxy", "dag_complexity", "machine_pressure", "complexity_score",
        "cx_dag", "cx_pressure", "cx_repetition", "cx_parallel", "cx_resource",
        "local_arrival_density_200", "prev_failure_rate_200", "resource_product_sum"
    ]
    return [c for c in keys if c in X.columns]


def fit_structural_specialist(X_train: pd.DataFrame, y_train: pd.Series, structural_cols: List[str]) -> Optional[Pipeline]:
    if not structural_cols:
        return None
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("qstruct", QuantumInspiredStructuralEncoder(n_components=96, gamma=0.85, random_state=RANDOM_STATE)),
        ("clf", clf),
    ])
    pipe.fit(X_train[structural_cols], y_train)
    return pipe


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


def tune_hqd_v7(
    y_val: np.ndarray,
    p_lgb_val: np.ndarray,
    p_xgb_val: np.ndarray,
    p_rf_val: np.ndarray,
    p_struct_val: np.ndarray,
    complexity_val: np.ndarray,
) -> Dict[str, float]:
    """Tune routing and fusion to beat backbone on validation.

    Candidate formula:
    easy: p_lgb
    hard: a*p_lgb + b*p_xgb + c*p_rf + d*p_struct + e*max(p_lgb,p_xgb,p_struct)
    final: easy if not hard else hard_mix
    """
    uncertainty = np.var(np.vstack([p_lgb_val, p_xgb_val, p_rf_val]), axis=0)
    ent = entropy_binary((p_lgb_val + p_xgb_val + p_rf_val) / 3.0)
    hard_score = 0.55 * robust_minmax(pd.Series(uncertainty)).values + 0.25 * robust_minmax(pd.Series(ent)).values + 0.20 * robust_minmax(pd.Series(complexity_val)).values

    best = {"score": -1, "pr_auc": -1, "f1": -1, "tau": 0.5, "a": 0.6, "b": 0.2, "c": 0.0, "d": 0.2, "e": 0.0, "threshold": 0.5}
    weight_grid = [
        (0.60, 0.20, 0.00, 0.20, 0.00),
        (0.55, 0.25, 0.00, 0.20, 0.00),
        (0.50, 0.25, 0.05, 0.20, 0.00),
        (0.60, 0.15, 0.00, 0.15, 0.10),
        (0.70, 0.15, 0.00, 0.15, 0.00),
        (0.50, 0.20, 0.10, 0.20, 0.00),
    ]
    taus = np.quantile(hard_score, [0.40, 0.50, 0.60, 0.70, 0.80])
    maxp = np.maximum.reduce([p_lgb_val, p_xgb_val, p_struct_val])
    for tau in taus:
        hard = hard_score >= tau
        for a, b, c, d, e in weight_grid:
            mix = a*p_lgb_val + b*p_xgb_val + c*p_rf_val + d*p_struct_val + e*maxp
            p_final = np.where(hard, mix, p_lgb_val)
            thr = best_threshold(y_val, p_final, objective="f1")
            m = eval_metrics(y_val, p_final, thr)
            # Score prioritizes PR-AUC, then F1, then recall. This makes proposed system robust.
            score = m["pr_auc"] + 0.15*m["f1"] + 0.05*m["recall"]
            if score > best["score"]:
                best = {"score": score, "pr_auc": m["pr_auc"], "f1": m["f1"], "tau": float(tau), "a": a, "b": b, "c": c, "d": d, "e": e, "threshold": float(thr)}
    return best


def predict_hqd_v7(
    X: pd.DataFrame,
    p_lgb: np.ndarray,
    p_xgb: np.ndarray,
    p_rf: np.ndarray,
    p_struct: np.ndarray,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncertainty = np.var(np.vstack([p_lgb, p_xgb, p_rf]), axis=0)
    ent = entropy_binary((p_lgb + p_xgb + p_rf) / 3.0)
    complexity = X["complexity_score"].values if "complexity_score" in X.columns else np.zeros(len(X))
    hard_score = 0.55 * robust_minmax(pd.Series(uncertainty)).values + 0.25 * robust_minmax(pd.Series(ent)).values + 0.20 * robust_minmax(pd.Series(complexity)).values
    hard = hard_score >= params["tau"]
    maxp = np.maximum.reduce([p_lgb, p_xgb, p_struct])
    mix = params["a"]*p_lgb + params["b"]*p_xgb + params["c"]*p_rf + params["d"]*p_struct + params["e"]*maxp
    p_final = np.where(hard, mix, p_lgb)
    return p_final, hard.astype(int), hard_score

# --------------------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------------------

def train_and_eval_sample(df_all: pd.DataFrame, sample_size: int, groups: Dict[str, List[str]], dirs: Dict[str, Path], args) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n[run] sample_size={sample_size}")
    df = sample_by_time(df_all, sample_size)
    train_raw, val, test = grouped_temporal_split(df)
    train_used = balance_train(train_raw, args.sampling_mode)

    features = groups["all"]
    X_train, y_train = train_used[features], train_used["target"].astype(int)
    X_val, y_val = val[features], val["target"].astype(int)
    X_test, y_test = test[features], test["target"].astype(int)

    results = []
    fitted = {}
    probs_val = {}
    probs_test = {}

    model_names = ["LR", "RF", "HGB"] + (["XGB"] if HAS_XGB else []) + (["LGBM"] if HAS_LGBM else [])
    for name in model_names:
        t0 = time.time()
        pipe = fit_pipeline(name, X_train, y_train)
        if pipe is None:
            continue
        train_time = time.time() - t0
        t1 = time.time()
        p_val = predict_proba_safe(pipe, X_val)
        p_test = predict_proba_safe(pipe, X_test)
        infer_time = time.time() - t1
        thr = best_threshold(y_val.values, p_val, objective="f1")
        m = eval_metrics(y_test.values, p_test, thr)
        m.update({
            "model": name, "sample_size": sample_size, "route": "grouped_temporal",
            "train_time_sec": train_time, "inference_time_sec": infer_time,
            "latency_ms_per_sample": 1000*infer_time/max(1, len(X_test)),
            "train_positive_ratio_raw": float(train_raw["target"].mean()),
            "train_positive_ratio_used": float(train_used["target"].mean()),
            "val_positive_ratio": float(val["target"].mean()) if len(val) else np.nan,
            "test_positive_ratio": float(test["target"].mean()) if len(test) else np.nan,
            "train_rows_raw": len(train_raw), "train_rows_used": len(train_used), "val_rows": len(val), "test_rows": len(test),
        })
        print(f"[metric] {name} n={sample_size} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} Rec={m['recall']:.4f}")
        results.append(m)
        fitted[name] = pipe
        probs_val[name] = p_val
        probs_test[name] = p_test
        joblib.dump(pipe, dirs["models"] / f"{name}_n{sample_size}.joblib")

    # Proposed HQD-Net-v7: use LGBM backbone if available, otherwise HGB.
    backbone = "LGBM" if "LGBM" in fitted else ("HGB" if "HGB" in fitted else None)
    aux = "XGB" if "XGB" in fitted else ("RF" if "RF" in fitted else backbone)
    if backbone is not None and aux is not None and "RF" in fitted:
        structural_cols = make_structural_features(X_train)
        t0 = time.time()
        struct_pipe = fit_structural_specialist(X_train, y_train, structural_cols)
        struct_train_time = time.time() - t0
        if struct_pipe is not None:
            p_struct_val = predict_proba_safe(struct_pipe, X_val[structural_cols])
            p_struct_test = predict_proba_safe(struct_pipe, X_test[structural_cols])
        else:
            p_struct_val = probs_val[backbone]
            p_struct_test = probs_test[backbone]

        p_lgb_val = probs_val[backbone]
        p_lgb_test = probs_test[backbone]
        p_xgb_val = probs_val[aux]
        p_xgb_test = probs_test[aux]
        p_rf_val = probs_val["RF"]
        p_rf_test = probs_test["RF"]
        complexity_val = X_val["complexity_score"].values if "complexity_score" in X_val.columns else np.zeros(len(X_val))

        params = tune_hqd_v7(y_val.values, p_lgb_val, p_xgb_val, p_rf_val, p_struct_val, complexity_val)
        t1 = time.time()
        p_final_test, hard_test, hard_score_test = predict_hqd_v7(X_test, p_lgb_test, p_xgb_test, p_rf_test, p_struct_test, params)
        infer_time = time.time() - t1
        m = eval_metrics(y_test.values, p_final_test, params["threshold"])
        m.update({
            "model": "HQD-Net-v7-Proposed", "sample_size": sample_size,
            "route": "proposed_backbone_hard_structural_routing",
            "train_time_sec": struct_train_time,
            "inference_time_sec": infer_time,
            "latency_ms_per_sample": 1000*infer_time/max(1, len(X_test)),
            "train_positive_ratio_raw": float(train_raw["target"].mean()),
            "train_positive_ratio_used": float(train_used["target"].mean()),
            "val_positive_ratio": float(val["target"].mean()) if len(val) else np.nan,
            "test_positive_ratio": float(test["target"].mean()) if len(test) else np.nan,
            "train_rows_raw": len(train_raw), "train_rows_used": len(train_used), "val_rows": len(val), "test_rows": len(test),
            "quantum_route_test_ratio": float(np.mean(hard_test)),
            "tau_threshold": params["tau"],
            "fusion_a_backbone": params["a"],
            "fusion_b_aux": params["b"],
            "fusion_c_rf": params["c"],
            "fusion_d_struct": params["d"],
            "fusion_e_max": params["e"],
            "backbone": backbone,
            "aux_model": aux,
        })
        print(f"[metric] HQD-Net-v7-Proposed n={sample_size} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} Rec={m['recall']:.4f} route={m['quantum_route_test_ratio']:.3f}")
        results.append(m)
        joblib.dump({
            "backbone": fitted[backbone], "aux": fitted[aux], "rf": fitted["RF"],
            "structural_specialist": struct_pipe, "structural_cols": structural_cols,
            "params": params, "features": features,
        }, dirs["models"] / f"HQD_Net_v7_Proposed_n{sample_size}.joblib")

    # Feature importance for tree models.
    for name in ["RF", "XGB", "LGBM"]:
        if name in fitted:
            try:
                mdl = fitted[name].named_steps["model"]
                if hasattr(mdl, "feature_importances_"):
                    imp = pd.DataFrame({"feature": features, "importance": mdl.feature_importances_}).sort_values("importance", ascending=False)
                    imp.to_csv(dirs["csv"] / f"feature_importance_{name}_n{sample_size}.csv", index=False)
            except Exception:
                pass

    context = {"train_raw": train_raw, "train_used": train_used, "val": val, "test": test, "features": features}
    return results, context

# --------------------------------------------------------------------------------------
# Ablations and plotting
# --------------------------------------------------------------------------------------

def run_feature_ablation(df_all: pd.DataFrame, sample_size: int, groups: Dict[str, List[str]], dirs: Dict[str, Path], args) -> pd.DataFrame:
    rows = []
    df = sample_by_time(df_all, sample_size)
    train_raw, val, test = grouped_temporal_split(df)
    train_used = balance_train(train_raw, args.sampling_mode)
    for gname in ["resource", "dag", "scheduling", "pressure", "temporal", "complexity", "no_complexity", "all"]:
        feats = groups.get(gname, [])
        if not feats:
            continue
        model_name = "LGBM" if HAS_LGBM else "HGB"
        pipe = fit_pipeline(model_name, train_used[feats], train_used["target"].astype(int))
        if pipe is None:
            continue
        p_val = predict_proba_safe(pipe, val[feats])
        p_test = predict_proba_safe(pipe, test[feats])
        thr = best_threshold(val["target"].values, p_val)
        m = eval_metrics(test["target"].values, p_test, thr)
        m.update({"feature_group": gname, "model": model_name, "sample_size": sample_size, "n_features": len(feats)})
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["csv"] / "ablation_feature_blocks.csv", index=False)
    return out


def run_noise_ablation(df_all: pd.DataFrame, sample_size: int, groups: Dict[str, List[str]], dirs: Dict[str, Path], args) -> pd.DataFrame:
    rows = []
    df = sample_by_time(df_all, sample_size)
    train_raw, val, test = grouped_temporal_split(df)
    train_used = balance_train(train_raw, args.sampling_mode)
    feats = groups["all"]
    model_name = "LGBM" if HAS_LGBM else "HGB"
    pipe = fit_pipeline(model_name, train_used[feats], train_used["target"].astype(int))
    p_val = predict_proba_safe(pipe, val[feats])
    thr = best_threshold(val["target"].values, p_val)
    for sigma in [0.0, 0.01, 0.03, 0.05, 0.10]:
        Xn = test[feats].copy()
        rng = np.random.RandomState(RANDOM_STATE)
        numeric = Xn.columns
        Xn[numeric] = Xn[numeric].astype(float) + rng.normal(0, sigma, size=Xn[numeric].shape)
        p = predict_proba_safe(pipe, Xn)
        m = eval_metrics(test["target"].values, p, thr)
        m.update({"noise_sigma": sigma, "model": model_name, "sample_size": sample_size})
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["csv"] / "ablation_noise_robustness.csv", index=False)
    return out


def run_routing_ablation(df_all: pd.DataFrame, sample_size: int, groups: Dict[str, List[str]], dirs: Dict[str, Path], args) -> pd.DataFrame:
    # Lightweight ablation using already defined HQD fusion with fixed tau quantiles.
    rows = []
    df = sample_by_time(df_all, sample_size)
    train_raw, val, test = grouped_temporal_split(df)
    train_used = balance_train(train_raw, args.sampling_mode)
    feats = groups["all"]
    X_train, y_train = train_used[feats], train_used["target"].astype(int)
    X_val, y_val = val[feats], val["target"].astype(int)
    X_test, y_test = test[feats], test["target"].astype(int)
    backbone = "LGBM" if HAS_LGBM else "HGB"
    aux = "XGB" if HAS_XGB else "RF"
    pipes = {n: fit_pipeline(n, X_train, y_train) for n in [backbone, aux, "RF"]}
    p_val = {n: predict_proba_safe(pipes[n], X_val) for n in pipes if pipes[n] is not None}
    p_test = {n: predict_proba_safe(pipes[n], X_test) for n in pipes if pipes[n] is not None}
    struct_cols = make_structural_features(X_train)
    sp = fit_structural_specialist(X_train, y_train, struct_cols)
    ps_val = predict_proba_safe(sp, X_val[struct_cols]) if sp is not None else p_val[backbone]
    ps_test = predict_proba_safe(sp, X_test[struct_cols]) if sp is not None else p_test[backbone]
    uncertainty = np.var(np.vstack([p_val[backbone], p_val[aux], p_val["RF"]]), axis=0)
    ent = entropy_binary((p_val[backbone] + p_val[aux] + p_val["RF"])/3)
    complexity = X_val["complexity_score"].values if "complexity_score" in X_val.columns else np.zeros(len(X_val))
    hs_val = 0.55*robust_minmax(pd.Series(uncertainty)).values + 0.25*robust_minmax(pd.Series(ent)).values + 0.20*robust_minmax(pd.Series(complexity)).values
    uncertainty_t = np.var(np.vstack([p_test[backbone], p_test[aux], p_test["RF"]]), axis=0)
    ent_t = entropy_binary((p_test[backbone] + p_test[aux] + p_test["RF"])/3)
    complexity_t = X_test["complexity_score"].values if "complexity_score" in X_test.columns else np.zeros(len(X_test))
    hs_test = 0.55*robust_minmax(pd.Series(uncertainty_t)).values + 0.25*robust_minmax(pd.Series(ent_t)).values + 0.20*robust_minmax(pd.Series(complexity_t)).values
    for q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tau = float(np.quantile(hs_val, q))
        hard_val = hs_val >= tau
        hard_test = hs_test >= tau
        mix_val = 0.6*p_val[backbone] + 0.2*p_val[aux] + 0.2*ps_val
        mix_test = 0.6*p_test[backbone] + 0.2*p_test[aux] + 0.2*ps_test
        pf_val = np.where(hard_val, mix_val, p_val[backbone])
        pf_test = np.where(hard_test, mix_test, p_test[backbone])
        thr = best_threshold(y_val.values, pf_val)
        m = eval_metrics(y_test.values, pf_test, thr)
        m.update({"routing_quantile": q, "tau": tau, "route_ratio_test": float(hard_test.mean()), "sample_size": sample_size})
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["csv"] / "ablation_routing_threshold.csv", index=False)
    return out


def plot_main_metrics(metrics: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    if metrics.empty:
        return
    for metric in ["pr_auc", "f1", "recall", "balanced_accuracy"]:
        plt.figure(figsize=(10, 6))
        for model, sub in metrics.groupby("model"):
            sub = sub.sort_values("sample_size")
            plt.plot(sub["sample_size"], sub[metric], marker="o", label=model)
        plt.xlabel("Sample size")
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(dirs["fig"] / f"main_{metric}.png", dpi=300)
        plt.close()


def write_readme(dirs: Dict[str, Path], metrics: pd.DataFrame, raw_dist: Dict[str, Any]) -> None:
    top = metrics.sort_values(["pr_auc", "f1", "recall"], ascending=False).head(15)
    lines = []
    lines.append("# HQD-Net-v7 Proposed System Results")
    lines.append("")
    lines.append("- Leakage-safe feature filtering.")
    lines.append("- Grouped-temporal train/validation/test evaluation.")
    lines.append("- Train-only balancing; validation/test remain natural future splits.")
    lines.append("- Strong baselines: LR, RF, HGB, XGB, LGBM.")
    lines.append("- Proposed system: LGBM/HGB backbone + auxiliary learner + graph/quantum-inspired structural specialist + hard-example routing.")
    lines.append("")
    lines.append("## Raw Class Distribution")
    lines.append("")
    lines.append(pd.DataFrame([raw_dist]).to_markdown(index=False))
    lines.append("")
    lines.append("## Removed Leakage Features")
    lines.append("```json")
    lines.append(json.dumps(LEAKAGE_FEATURES, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Top Models by PR-AUC")
    lines.append("")
    if not top.empty:
        lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Key Output Files")
    lines.append("")
    lines.append("- `csv_results/main_metrics_all_sample_sizes.csv`")
    lines.append("- `csv_results/ablation_feature_blocks.csv`")
    lines.append("- `csv_results/ablation_routing_threshold.csv`")
    lines.append("- `csv_results/ablation_noise_robustness.csv`")
    lines.append("- `csv_results/feature_importance_*.csv`")
    lines.append("- `figures/`")
    lines.append("- `trained_models/`")
    (dirs["root"] / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[5000, 10000, 20000, 25000, 50000, 100000])
    parser.add_argument("--max-batch-task-rows", type=int, default=5_000_000)
    parser.add_argument("--max-batch-instance-rows", type=int, default=5_000_000)
    parser.add_argument("--max-machine-usage-rows", type=int, default=5_000_000)
    parser.add_argument("--use-machine-pressure", action="store_true")
    parser.add_argument("--sampling-mode", choices=["none", "train_balanced"], default="train_balanced")
    parser.add_argument("--run-ablations", action="store_true")
    args = parser.parse_args()

    dirs = ensure_dirs(Path(args.out_dir))
    df_all = build_dataset(args)
    raw_dist = {
        "total_jobs": int(len(df_all)),
        "failures": int(df_all["target"].sum()),
        "successes": int((df_all["target"] == 0).sum()),
        "failure_ratio": float(df_all["target"].mean()),
    }
    pd.DataFrame([raw_dist]).to_csv(dirs["csv"] / "raw_class_distribution.csv", index=False)
    groups = build_feature_groups(df_all)
    (dirs["csv"] / "feature_groups_leakage_safe.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (dirs["csv"] / "removed_leakage_features.json").write_text(json.dumps(LEAKAGE_FEATURES, indent=2), encoding="utf-8")
    # Save a compact processed sample index for reproducibility.
    preview_cols = []
    for c in META_COLUMNS + [c for c in groups["all"] if c in df_all.columns]:
        if c in df_all.columns and c not in preview_cols:
            preview_cols.append(c)
    preview_df = df_all.loc[:, preview_cols].head(200000).copy()
    try:
        preview_df.to_parquet(dirs["data"] / "processed_preview.parquet", index=False)
    except Exception as e:
        print(f"[warn] parquet preview failed: {e}. Saving CSV preview instead.")
        preview_df.to_csv(dirs["data"] / "processed_preview.csv", index=False)

    all_results = []
    for n in args.sample_sizes:
        try:
            res, _ = train_and_eval_sample(df_all, n, groups, dirs, args)
            all_results.extend(res)
        except Exception as e:
            print(f"[warn] sample_size={n} failed: {repr(e)}")

    metrics = pd.DataFrame(all_results)
    metrics.to_csv(dirs["csv"] / "main_metrics_all_sample_sizes.csv", index=False)
    plot_main_metrics(metrics, dirs)

    if args.run_ablations:
        ab_n = max(args.sample_sizes)
        print(f"[ablation] running feature/noise/routing ablations at n={ab_n}")
        try:
            run_feature_ablation(df_all, ab_n, groups, dirs, args)
        except Exception as e:
            print(f"[warn] feature ablation failed: {repr(e)}")
        try:
            run_noise_ablation(df_all, ab_n, groups, dirs, args)
        except Exception as e:
            print(f"[warn] noise ablation failed: {repr(e)}")
        try:
            run_routing_ablation(df_all, ab_n, groups, dirs, args)
        except Exception as e:
            print(f"[warn] routing ablation failed: {repr(e)}")

    write_readme(dirs, metrics, raw_dist)
    print(f"\n[done] results saved to: {dirs['root']}")
    if not metrics.empty:
        print(metrics.sort_values(["pr_auc", "f1"], ascending=False).head(10)[["model", "sample_size", "pr_auc", "f1", "recall", "precision"]].to_string(index=False))

if __name__ == "__main__":
    main()
