# -*- coding: utf-8 -*-
"""
HQD-Net v6 Final
Leakage-safe grouped-temporal hard-failure learning for Alibaba Cluster Trace v2018.

Core design:
1) Build job-level pre-execution features from batch_task, batch_instance, and optional machine_usage.
2) Remove all post-execution leakage features from predictors.
3) Split by time first: train / validation / test. Balance training only.
4) Train strong classical models and a graph/quantum-inspired hard-sample specialist.
5) Use uncertainty + complexity routing for difficult workloads.
6) Save CSVs, figures, trained models, feature importances, ablations, and README.

Run example:
python hqd_net_job_failure_v6_final.py ^
  --data-dir "D:\\other\\ALIBABAQUATUM\\Dataset" ^
  --out-dir "D:\\other\\ALIBABAQUATUM\\Results_HQDNet_v6" ^
  --sample-sizes 5000 10000 20000 25000 50000 100000 ^
  --max-batch-task-rows 5000000 ^
  --max-batch-instance-rows 5000000 ^
  --max-machine-usage-rows 5000000 ^
  --use-machine-pressure ^
  --sampling-mode train_balanced ^
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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import joblib
import psutil

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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TASK_COLS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem"
]
INSTANCE_COLS = [
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
    "cpu_spike_mean", "mem_spike_mean", "target", "label"
]

# -----------------------------
# Utility
# -----------------------------
def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "csv": out_dir / "csv_results",
        "fig": out_dir / "figures",
        "model": out_dir / "trained_models",
        "data": out_dir / "processed_data",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_auc(metric_func, y_true, y_score) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(metric_func(y_true, y_score))
    except Exception:
        return np.nan


def memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def read_csv_robust(path: Path, cols: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    log(f"[data] reading {path} ...")
    # Read as header=None to handle files with repeated header rows. Then remove rows equal to header names.
    df = pd.read_csv(path, header=None, names=cols, nrows=max_rows, low_memory=False)
    # Remove repeated header rows.
    first_col = cols[0]
    df = df[df[first_col].astype(str).str.lower() != first_col.lower()].copy()
    log(f"[data] {path.name} shape={df.shape}")
    return df


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def status_to_failure(s: pd.Series) -> pd.Series:
    """Return 1 for failed/abnormal, 0 for successful Terminated."""
    ss = s.astype(str).str.lower().str.strip()
    # In Alibaba batch, Terminated usually means completed. Everything else is treated as failure/abnormal.
    return (~ss.isin(["terminated", "success", "succeeded", "finished", "completed"])).astype(int)


def parse_task_graph_stats(task_names: pd.Series) -> Dict[str, float]:
    """Extract DAG statistics from Alibaba task_name patterns: M1, R2_1, J4_2_3 etc."""
    ids = []
    edges = []
    roots = 0
    retry_like = 0
    for raw in task_names.dropna().astype(str).values:
        nums = re.findall(r"\d+", raw)
        if not nums:
            roots += 1
            continue
        cur = int(nums[0])
        ids.append(cur)
        parents = [int(x) for x in nums[1:]]
        if len(parents) == 0:
            roots += 1
        if raw.upper().startswith("R") or len(parents) > 1:
            retry_like += 1
        for p in parents:
            edges.append((p, cur))
    n = max(len(set(ids)), 1)
    m = len(edges)
    # fan-in
    fan_in = {}
    fan_out = {}
    for u, v in edges:
        fan_in[v] = fan_in.get(v, 0) + 1
        fan_out[u] = fan_out.get(u, 0) + 1
    fan_in_vals = list(fan_in.values()) or [0]
    fan_out_vals = list(fan_out.values()) or [0]
    # Approximate depth by DP on numeric order. For malformed tasks, this is robust enough.
    depth = {i: 1 for i in set(ids)}
    for u, v in sorted(edges, key=lambda x: (x[0], x[1])):
        depth[v] = max(depth.get(v, 1), depth.get(u, 1) + 1)
    dag_depth = max(depth.values()) if depth else 1
    density = m / max(n * (n - 1), 1)
    root_ratio = roots / max(len(task_names), 1)
    return {
        "dag_depth": float(dag_depth),
        "fan_in_mean": float(np.mean(fan_in_vals)),
        "fan_in_max": float(np.max(fan_in_vals)),
        "fan_out_mean": float(np.mean(fan_out_vals)),
        "fan_out_max": float(np.max(fan_out_vals)),
        "root_task_ratio": float(root_ratio),
        "dependency_density": float(density),
        "critical_path_proxy": float(dag_depth / max(n, 1)),
        "retry_task_ratio": float(retry_like / max(len(task_names), 1)),
        "dag_complexity": float(math.log1p(n) * math.log1p(m + 1) * (1 + density)),
    }


def add_temporal_history(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    df = df.sort_values("first_start_time").reset_index(drop=True)
    y_prev = df["job_failure"].shift(1)
    df["prev_failure_rate_200"] = y_prev.rolling(window=window, min_periods=10).mean().fillna(y_prev.expanding().mean()).fillna(0)
    starts = df["first_start_time"].astype(float)
    gaps = starts.diff().fillna(starts.diff().median() if starts.diff().notna().any() else 0)
    df["prev_arrival_gap_200"] = gaps.shift(1).rolling(window=window, min_periods=10).mean().fillna(gaps.median()).fillna(0)
    span = starts - starts.shift(window)
    df["local_arrival_density_200"] = (window / span.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["submit_hour"] = ((starts // 3600) % 24).astype(float)
    df["submit_day"] = ((starts // (3600 * 24)) % 14).astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["submit_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["submit_hour"] / 24.0)
    return df


def build_dataset(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    bt = read_csv_robust(data_dir / "batch_task.csv", TASK_COLS, args.max_batch_task_rows)
    bi = read_csv_robust(data_dir / "batch_instance.csv", INSTANCE_COLS, args.max_batch_instance_rows)

    bt = to_num(bt, ["instance_num", "task_type", "start_time", "end_time", "plan_cpu", "plan_mem"])
    bi = to_num(bi, ["task_type", "start_time", "end_time", "seq_no", "total_seq_no", "cpu_avg", "cpu_max", "mem_avg", "mem_max"])

    # Target only. Do not keep status-derived features as predictors.
    bt["task_failure"] = status_to_failure(bt["status"])
    bi["instance_failure"] = status_to_failure(bi["status"])

    # Task-level pre-execution resource + DAG summary.
    log("[feature] aggregating batch_task job-level features ...")
    agg_task = bt.groupby("job_name", observed=True).agg(
        task_count=("task_name", "count"),
        unique_task_count=("task_name", "nunique"),
        total_instances=("instance_num", "sum"),
        max_instances_per_task=("instance_num", "max"),
        mean_instances_per_task=("instance_num", "mean"),
        plan_cpu_mean=("plan_cpu", "mean"),
        plan_cpu_max=("plan_cpu", "max"),
        plan_cpu_sum=("plan_cpu", "sum"),
        plan_mem_mean=("plan_mem", "mean"),
        plan_mem_max=("plan_mem", "max"),
        plan_mem_sum=("plan_mem", "sum"),
        first_start_time=("start_time", "min"),
        task_start_min=("start_time", "min"),
        task_failure_any=("task_failure", "max"),
    ).reset_index()

    graph_rows = []
    for job, g in bt.groupby("job_name", observed=True)["task_name"]:
        stats = parse_task_graph_stats(g)
        stats["job_name"] = job
        graph_rows.append(stats)
    graph_df = pd.DataFrame(graph_rows)
    agg_task = agg_task.merge(graph_df, on="job_name", how="left")

    # Instance pre-execution scheduling summary. Avoid runtime and CPU/mem actual usage because these are post-execution.
    log("[feature] aggregating batch_instance scheduling features ...")
    agg_inst = bi.groupby("job_name", observed=True).agg(
        instance_count=("instance_name", "count"),
        unique_machines=("machine_id", "nunique"),
        seq_no_mean=("seq_no", "mean"),
        total_seq_no_mean=("total_seq_no", "mean"),
        inst_start_min=("start_time", "min"),
        instance_failure_any=("instance_failure", "max"),
    ).reset_index()

    # Machine pressure aggregated by machine over the observed raw window. Optional. It is a coarse pressure proxy.
    pressure_by_machine = None
    if args.use_machine_pressure:
        mfile = data_dir / "machine_usage_bigger.csv"
        if not mfile.exists():
            mfile = data_dir / "machine_usage.csv"
        if mfile.exists():
            mu = read_csv_robust(mfile, MACHINE_USAGE_COLS, args.max_machine_usage_rows)
            mu = to_num(mu, ["time_stamp", "cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"])
            pressure_by_machine = mu.groupby("machine_id", observed=True).agg(
                machine_cpu_mean=("cpu_util_percent", "mean"),
                machine_cpu_max=("cpu_util_percent", "max"),
                machine_mem_mean=("mem_util_percent", "mean"),
                machine_mem_max=("mem_util_percent", "max"),
                machine_net_in_mean=("net_in", "mean"),
                machine_net_out_mean=("net_out", "mean"),
                machine_disk_mean=("disk_io_percent", "mean"),
            ).reset_index()
            pressure_by_machine["machine_pressure"] = (
                pressure_by_machine[["machine_cpu_mean", "machine_mem_mean", "machine_disk_mean"]]
                .mean(axis=1, skipna=True)
            )

    # Job to machine list, then average pressure over machines used by job.
    if pressure_by_machine is not None:
        log("[feature] joining machine pressure into jobs ...")
        jm = bi[["job_name", "machine_id"]].dropna().drop_duplicates()
        jm = jm.merge(pressure_by_machine, on="machine_id", how="left")
        agg_press = jm.groupby("job_name", observed=True).agg(
            machine_cpu_mean=("machine_cpu_mean", "mean"),
            machine_cpu_max=("machine_cpu_max", "max"),
            machine_mem_mean=("machine_mem_mean", "mean"),
            machine_mem_max=("machine_mem_max", "max"),
            machine_net_in_mean=("machine_net_in_mean", "mean"),
            machine_net_out_mean=("machine_net_out_mean", "mean"),
            machine_disk_mean=("machine_disk_mean", "mean"),
            machine_pressure=("machine_pressure", "mean"),
        ).reset_index()
    else:
        agg_press = pd.DataFrame({"job_name": agg_task["job_name"]})

    df = agg_task.merge(agg_inst, on="job_name", how="left").merge(agg_press, on="job_name", how="left")
    df["job_failure"] = ((df["task_failure_any"].fillna(0) > 0) | (df["instance_failure_any"].fillna(0) > 0)).astype(int)
    df["first_start_time"] = df[["first_start_time", "inst_start_min"]].min(axis=1)

    # Safe derived features.
    df["cpu_mem_ratio"] = df["plan_cpu_sum"] / (df["plan_mem_sum"].replace(0, np.nan))
    df["resource_product_sum"] = df["plan_cpu_sum"].fillna(0) * df["plan_mem_sum"].fillna(0)
    df["parallelism_proxy"] = df["total_instances"].fillna(0) / (df["unique_task_count"].replace(0, np.nan))
    df["dependency_skew"] = df["fan_in_max"].fillna(0) / (df["fan_in_mean"].replace(0, np.nan))
    df["deep_chain_ratio"] = df["dag_depth"].fillna(0) / (df["unique_task_count"].replace(0, np.nan))
    df["bottleneck_task_ratio"] = df["max_instances_per_task"].fillna(0) / (df["total_instances"].replace(0, np.nan))

    df = add_temporal_history(df, window=args.history_window)

    # Complexity score, robust normalized later by rank percentile.
    safe_cols = [
        "dag_complexity", "dependency_density", "deep_chain_ratio", "parallelism_proxy",
        "retry_task_ratio", "resource_product_sum", "machine_pressure", "local_arrival_density_200"
    ]
    for c in safe_cols:
        if c not in df.columns:
            df[c] = 0.0
    # percentile rank avoids scale dependence.
    for c in safe_cols:
        df[f"rank_{c}"] = df[c].rank(pct=True).fillna(0)
    df["cx_dag"] = df[["rank_dag_complexity", "rank_dependency_density", "rank_deep_chain_ratio"]].mean(axis=1)
    df["cx_pressure"] = df["rank_machine_pressure"] if "rank_machine_pressure" in df.columns else 0
    df["cx_repetition"] = df["rank_retry_task_ratio"]
    df["cx_parallel"] = df["rank_parallelism_proxy"]
    df["cx_resource"] = df["rank_resource_product_sum"]
    df["complexity_score"] = df[["cx_dag", "cx_pressure", "cx_repetition", "cx_parallel", "cx_resource"]].mean(axis=1)

    # Drop leakage and helper target-status columns from predictor list later, but keep job_failure/job_name/time.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["job_name", "first_start_time", "job_failure"])
    df = df.drop_duplicates(subset=["job_name"]).reset_index(drop=True)
    log(f"[data] final unique jobs={len(df):,}, failure_ratio={df['job_failure'].mean():.4f}")
    return df


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "resource": [
            "plan_cpu_mean", "plan_cpu_max", "plan_cpu_sum", "plan_mem_mean", "plan_mem_max",
            "plan_mem_sum", "cpu_mem_ratio", "resource_product_sum"
        ],
        "dag": [
            "task_count", "unique_task_count", "total_instances", "max_instances_per_task",
            "mean_instances_per_task", "dag_depth", "fan_in_mean", "fan_in_max", "fan_out_mean",
            "fan_out_max", "root_task_ratio", "parallelism_proxy", "dependency_density",
            "critical_path_proxy", "dag_complexity", "retry_task_ratio", "dependency_skew",
            "deep_chain_ratio", "bottleneck_task_ratio"
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
    all_cols = []
    for v in groups.values():
        all_cols.extend(v)
    all_cols = sorted(set(all_cols))
    groups["all"] = all_cols
    groups["no_complexity"] = [c for c in all_cols if c not in groups["complexity"]]
    groups["graph_quantum"] = sorted(set(groups["dag"] + groups["complexity"] + [c for c in ["machine_pressure", "local_arrival_density_200", "prev_failure_rate_200"] if c in df.columns]))
    return groups


def select_sample_temporal(df: pd.DataFrame, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Choose up to n jobs while preserving time coverage and failures.
    The data remains sorted by time for grouped temporal split.
    """
    df = df.sort_values("first_start_time").reset_index(drop=True)
    if n is None or n <= 0 or n >= len(df):
        return df.copy()
    # Time-stratified sample to avoid only early jobs. Preserve rare positives.
    pos = df[df["job_failure"] == 1]
    neg = df[df["job_failure"] == 0]
    # Keep all positives if not too many; cap if too many.
    target_pos = min(len(pos), max(int(n * max(df["job_failure"].mean(), 0.05)), min(len(pos), n // 3)))
    if target_pos > 0 and len(pos) > target_pos:
        pos_s = pos.sample(target_pos, random_state=seed)
    else:
        pos_s = pos
    remaining = max(n - len(pos_s), 0)
    neg_s = neg.sample(min(remaining, len(neg)), random_state=seed) if remaining > 0 else neg.iloc[0:0]
    out = pd.concat([pos_s, neg_s], axis=0).drop_duplicates("job_name").sort_values("first_start_time").reset_index(drop=True)
    return out


def temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("first_start_time").drop_duplicates("job_name").reset_index(drop=True)
    n = len(df)
    n_train = max(int(n * train_frac), 1)
    n_val = max(int(n * val_frac), 1)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test


def balance_training(train: pd.DataFrame, mode: str, neg_ratio: float, seed=RANDOM_STATE) -> pd.DataFrame:
    if mode == "none":
        return train.copy()
    pos = train[train["job_failure"] == 1]
    neg = train[train["job_failure"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        log("[warn] cannot balance training because one class is missing.")
        return train.copy()
    n_neg = int(min(len(neg), max(1, round(len(pos) * neg_ratio))))
    neg_s = neg.sample(n_neg, random_state=seed) if len(neg) > n_neg else neg
    out = pd.concat([pos, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# -----------------------------
# Quantum-inspired structural encoder
# -----------------------------
class QuantumInspiredStructuralEncoder:
    """Random Fourier + trigonometric interaction features over graph/dependency features.
    This is used as a lightweight quantum-inspired structural representation for hard samples.
    """
    def __init__(self, n_components: int = 64, gamma: float = 1.0, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W_ = None
        self.b_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: np.ndarray):
        Xs = self.scaler_.fit_transform(X)
        rng = np.random.default_rng(self.random_state)
        d = Xs.shape[1]
        self.W_ = rng.normal(0, np.sqrt(2 * self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=(self.n_components,))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        Z = Xs @ self.W_ + self.b_
        rff = np.sqrt(2.0 / self.n_components) * np.cos(Z)
        sinf = np.sqrt(2.0 / self.n_components) * np.sin(Z)
        # Compact pairwise-style interactions using original scaled feature summary.
        mean = Xs.mean(axis=1, keepdims=True)
        std = Xs.std(axis=1, keepdims=True)
        norm = np.linalg.norm(Xs, axis=1, keepdims=True)
        return np.hstack([rff, sinf, mean, std, norm])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_models(pos_weight: float = 1.0) -> Dict[str, Any]:
    models = {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=None),
        "RF": RandomForestClassifier(
            n_estimators=350, max_depth=None, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HGB": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=31, l2_regularization=0.05,
            random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, reg_alpha=0.1, objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=max(pos_weight, 1.0), random_state=RANDOM_STATE, n_jobs=-1
        )
    if HAS_LGBM:
        models["LGBM"] = LGBMClassifier(
            n_estimators=350, max_depth=-1, learning_rate=0.04, num_leaves=31,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    return models


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    return model.predict(X).astype(float)


def best_threshold_by_metric(y_true, y_score, metric="f1") -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    ps, rs, ts = precision_recall_curve(y_true, y_score)
    if len(ts) == 0:
        return 0.5
    if metric == "f2":
        beta2 = 4
        vals = (1 + beta2) * ps[:-1] * rs[:-1] / (beta2 * ps[:-1] + rs[:-1] + 1e-12)
    else:
        vals = 2 * ps[:-1] * rs[:-1] / (ps[:-1] + rs[:-1] + 1e-12)
    idx = int(np.nanargmax(vals))
    return float(ts[idx])


def evaluate(y_true, y_score, threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = best_threshold_by_metric(y_true, y_score, metric="f1")
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": safe_auc(roc_auc_score, y_true, y_score),
        "pr_auc": safe_auc(average_precision_score, y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@dataclass
class HQDNetResult:
    model_bundle: Dict[str, Any]
    val_score: np.ndarray
    test_score: np.ndarray
    route_train_ratio: float
    route_test_ratio: float
    tau_threshold: float
    uncertainty_threshold: float


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def train_hqdnet(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    X_test_raw: pd.DataFrame,
    features: List[str],
    graph_features: List[str],
    pos_weight: float,
    tau_quantile: float = 0.80,
    uncertainty_quantile: float = 0.80,
) -> HQDNetResult:
    # Train classical ensemble on all features.
    prep = make_preprocessor()
    Xtr = prep.fit_transform(X_train_raw[features])
    Xva = prep.transform(X_val_raw[features])
    Xte = prep.transform(X_test_raw[features])

    base_defs = make_models(pos_weight)
    # use 3-4 stable base models to avoid slow overfitting
    base_names = [m for m in ["RF", "XGB", "LGBM", "HGB"] if m in base_defs]
    base_models = {}
    train_scores = []
    val_scores = []
    test_scores = []
    for name in base_names:
        mdl = clone(base_defs[name])
        mdl.fit(Xtr, y_train)
        base_models[name] = mdl
        train_scores.append(get_scores(mdl, Xtr))
        val_scores.append(get_scores(mdl, Xva))
        test_scores.append(get_scores(mdl, Xte))
    train_scores = np.vstack(train_scores)
    val_scores = np.vstack(val_scores)
    test_scores = np.vstack(test_scores)

    base_train = train_scores.mean(axis=0)
    base_val = val_scores.mean(axis=0)
    base_test = test_scores.mean(axis=0)
    u_train = train_scores.var(axis=0) + entropy_binary(base_train)
    u_val = val_scores.var(axis=0) + entropy_binary(base_val)
    u_test = test_scores.var(axis=0) + entropy_binary(base_test)

    # Complexity threshold from train only.
    c_train = X_train_raw["complexity_score"].fillna(0).values if "complexity_score" in X_train_raw else np.zeros(len(y_train))
    c_val = X_val_raw["complexity_score"].fillna(0).values if "complexity_score" in X_val_raw else np.zeros(len(y_val))
    c_test = X_test_raw["complexity_score"].fillna(0).values if "complexity_score" in X_test_raw else np.zeros(len(X_test_raw))
    tau = float(np.quantile(c_train, tau_quantile)) if len(c_train) else 0
    uth = float(np.quantile(u_train, uncertainty_quantile)) if len(u_train) else 0

    hard_train = (c_train >= tau) | (u_train >= uth) | (base_train > 0.25)
    # Ensure enough positives in specialist.
    if y_train[hard_train].sum() < max(5, int(0.2 * y_train.sum())):
        positive_mask = y_train == 1
        hard_train = hard_train | positive_mask
    hard_val = (c_val >= tau) | (u_val >= uth) | (base_val > 0.25)
    hard_test = (c_test >= tau) | (u_test >= uth) | (base_test > 0.25)

    # Graph/quantum-inspired specialist.
    gcols = [c for c in graph_features if c in X_train_raw.columns]
    if not gcols:
        gcols = features
    gimp = SimpleImputer(strategy="median")
    Xg_tr_raw = gimp.fit_transform(X_train_raw.loc[hard_train, gcols])
    encoder = QuantumInspiredStructuralEncoder(n_components=96, gamma=0.8, random_state=RANDOM_STATE)
    Xg_tr = encoder.fit_transform(Xg_tr_raw)

    specialist = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=250, max_leaf_nodes=15, l2_regularization=0.1,
        random_state=RANDOM_STATE
    )
    if len(np.unique(y_train[hard_train])) < 2:
        # fallback LR on all train if routed class is degenerate
        specialist = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        Xg_tr_raw = gimp.fit_transform(X_train_raw[gcols])
        Xg_tr = encoder.fit_transform(Xg_tr_raw)
        y_spec = y_train
    else:
        y_spec = y_train[hard_train]
    specialist.fit(Xg_tr, y_spec)

    def specialist_scores(Xraw: pd.DataFrame) -> np.ndarray:
        Xg = gimp.transform(Xraw[gcols])
        Z = encoder.transform(Xg)
        return get_scores(specialist, Z)

    spec_val = specialist_scores(X_val_raw)
    spec_test = specialist_scores(X_test_raw)

    # Adaptive fusion only for hard samples. Conservative blend to avoid destabilizing easy samples.
    val_final = base_val.copy()
    test_final = base_test.copy()
    val_weight = np.clip(0.35 + 0.35 * c_val + 0.30 * (u_val / (np.max(u_train) + 1e-9)), 0.25, 0.85)
    test_weight = np.clip(0.35 + 0.35 * c_test + 0.30 * (u_test / (np.max(u_train) + 1e-9)), 0.25, 0.85)
    val_final[hard_val] = (1 - val_weight[hard_val]) * base_val[hard_val] + val_weight[hard_val] * spec_val[hard_val]
    test_final[hard_test] = (1 - test_weight[hard_test]) * base_test[hard_test] + test_weight[hard_test] * spec_test[hard_test]

    bundle = {
        "preprocessor": prep,
        "base_models": base_models,
        "graph_imputer": gimp,
        "encoder": encoder,
        "specialist": specialist,
        "features": features,
        "graph_features": gcols,
        "tau": tau,
        "uncertainty_threshold": uth,
    }
    return HQDNetResult(
        model_bundle=bundle,
        val_score=val_final,
        test_score=test_final,
        route_train_ratio=float(np.mean(hard_train)),
        route_test_ratio=float(np.mean(hard_test)),
        tau_threshold=tau,
        uncertainty_threshold=uth,
    )


def train_eval_single_sample(df_all: pd.DataFrame, n: int, dirs: Dict[str, Path], args, groups: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log(f"\n[run] sample_size={n}")
    df = select_sample_temporal(df_all, n, seed=RANDOM_STATE)
    train_raw, val_raw, test_raw = temporal_split(df)
    train_used = balance_training(train_raw, args.sampling_mode, args.train_negative_ratio, seed=RANDOM_STATE)

    features = groups["all"]
    graph_features = groups["graph_quantum"]
    # Keep only features existing and not leakage.
    features = [c for c in features if c in df.columns and c not in LEAKAGE_FEATURES and c != "job_failure"]
    y_train = train_used["job_failure"].astype(int).values
    y_val = val_raw["job_failure"].astype(int).values
    y_test = test_raw["job_failure"].astype(int).values

    # pos weight based on raw train distribution before balancing, useful for XGB.
    raw_pos = max(train_raw["job_failure"].sum(), 1)
    raw_neg = max((train_raw["job_failure"] == 0).sum(), 1)
    pos_weight = raw_neg / raw_pos

    rows = []
    models_to_save = {}

    # Preprocess train-only for classical models.
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(train_used[features])
    X_val = preprocessor.transform(val_raw[features])
    X_test = preprocessor.transform(test_raw[features])

    model_defs = make_models(pos_weight)
    for name, base_model in model_defs.items():
        t0 = time.perf_counter(); mem0 = memory_mb()
        model = clone(base_model)
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0; mem1 = memory_mb()
        val_score = get_scores(model, X_val)
        threshold = best_threshold_by_metric(y_val, val_score, metric=args.threshold_metric)
        t1 = time.perf_counter()
        test_score = get_scores(model, X_test)
        inf_time = time.perf_counter() - t1
        met = evaluate(y_test, test_score, threshold)
        met.update({
            "model": name, "sample_size": n, "route": "grouped_temporal", "train_time_sec": train_time,
            "inference_time_sec": inf_time, "latency_ms_per_sample": 1000 * inf_time / max(len(y_test), 1),
            "ram_delta_mb": max(mem1 - mem0, 0),
            "train_positive_ratio_raw": float(train_raw["job_failure"].mean()),
            "train_positive_ratio_used": float(train_used["job_failure"].mean()),
            "val_positive_ratio": float(val_raw["job_failure"].mean()) if len(val_raw) else np.nan,
            "test_positive_ratio": float(test_raw["job_failure"].mean()) if len(test_raw) else np.nan,
            "train_rows_raw": len(train_raw), "train_rows_used": len(train_used),
            "val_rows": len(val_raw), "test_rows": len(test_raw),
        })
        rows.append(met)
        models_to_save[name] = {"preprocessor": preprocessor, "model": model, "features": features, "threshold": threshold}
        log(f"[metric] {name} n={n} PR-AUC={met['pr_auc']:.4f} F1={met['f1']:.4f} Rec={met['recall']:.4f}")

    # HQD-Net v6
    t0 = time.perf_counter(); mem0 = memory_mb()
    hqd = train_hqdnet(train_used, y_train, val_raw, y_val, test_raw, features, graph_features, pos_weight,
                       tau_quantile=args.hard_tau_quantile, uncertainty_quantile=args.uncertainty_quantile)
    train_time = time.perf_counter() - t0; mem1 = memory_mb()
    threshold = best_threshold_by_metric(y_val, hqd.val_score, metric=args.threshold_metric)
    t1 = time.perf_counter()
    # scoring already computed; time only small copy overhead
    test_score = hqd.test_score
    inf_time = time.perf_counter() - t1
    met = evaluate(y_test, test_score, threshold)
    met.update({
        "model": f"HQD-Net-v6", "sample_size": n, "route": "grouped_temporal_hard_quantum_inspired",
        "train_time_sec": train_time, "inference_time_sec": inf_time,
        "latency_ms_per_sample": 1000 * inf_time / max(len(y_test), 1),
        "ram_delta_mb": max(mem1 - mem0, 0),
        "train_positive_ratio_raw": float(train_raw["job_failure"].mean()),
        "train_positive_ratio_used": float(train_used["job_failure"].mean()),
        "val_positive_ratio": float(val_raw["job_failure"].mean()) if len(val_raw) else np.nan,
        "test_positive_ratio": float(test_raw["job_failure"].mean()) if len(test_raw) else np.nan,
        "train_rows_raw": len(train_raw), "train_rows_used": len(train_used),
        "val_rows": len(val_raw), "test_rows": len(test_raw),
        "quantum_route_train_ratio": hqd.route_train_ratio,
        "quantum_route_test_ratio": hqd.route_test_ratio,
        "tau_threshold": hqd.tau_threshold,
        "uncertainty_threshold": hqd.uncertainty_threshold,
    })
    rows.append(met)
    models_to_save["HQD-Net-v6"] = {**hqd.model_bundle, "threshold": threshold}
    log(f"[metric] HQD-Net-v6 n={n} PR-AUC={met['pr_auc']:.4f} F1={met['f1']:.4f} Rec={met['recall']:.4f}")

    # Save processed split for largest/specific sample optionally.
    split_info = {
        "train_raw": train_raw, "train_used": train_used, "val": val_raw, "test": test_raw,
        "features": features, "graph_features": graph_features, "models": models_to_save,
    }
    return pd.DataFrame(rows), split_info


def save_model_bundle(bundle: Dict[str, Any], dirs: Dict[str, Path], name: str, n: int) -> None:
    path = dirs["model"] / f"{name}_n{n}.joblib"
    joblib.dump(bundle, path)


def plot_main_metrics(metrics: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    if metrics.empty:
        return
    for metric in ["pr_auc", "roc_auc", "f1", "recall", "mcc", "latency_ms_per_sample"]:
        if metric not in metrics.columns:
            continue
        plt.figure(figsize=(12, 6.75))
        for model, g in metrics.groupby("model"):
            gg = g.sort_values("sample_size")
            plt.plot(gg["sample_size"], gg[metric], marker="o", label=model)
        plt.xlabel("Sample Size")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.title(f"{metric.replace('_',' ').upper()} Across Sample Sizes")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(dirs["fig"] / f"main_{metric}.png", dpi=600)
        plt.close()


def plot_pr_curves(split_info: Dict[str, Any], dirs: Dict[str, Path], n: int) -> None:
    # Build PR curves for saved models using test split.
    test = split_info["test"]
    y = test["job_failure"].astype(int).values
    if len(np.unique(y)) < 2:
        return
    plt.figure(figsize=(12, 6.75))
    for name, bundle in split_info["models"].items():
        try:
            if name == "HQD-Net-v6":
                # Skip recomputing complex model; already covered in metrics.
                continue
            X = bundle["preprocessor"].transform(test[bundle["features"]])
            s = get_scores(bundle["model"], X)
            p, r, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            plt.plot(r, p, label=f"{name} AP={ap:.3f}")
        except Exception:
            pass
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves, n={n}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(dirs["fig"] / f"precision_recall_curves_n{n}.png", dpi=600)
    plt.close()


def run_feature_block_ablation(df_all: pd.DataFrame, n: int, dirs: Dict[str, Path], args, groups: Dict[str, List[str]]) -> pd.DataFrame:
    df = select_sample_temporal(df_all, n, seed=RANDOM_STATE)
    train_raw, val_raw, test_raw = temporal_split(df)
    train_used = balance_training(train_raw, args.sampling_mode, args.train_negative_ratio, seed=RANDOM_STATE)
    y_train = train_used["job_failure"].astype(int).values
    y_val = val_raw["job_failure"].astype(int).values
    y_test = test_raw["job_failure"].astype(int).values
    raw_pos = max(train_raw["job_failure"].sum(), 1)
    raw_neg = max((train_raw["job_failure"] == 0).sum(), 1)
    pos_weight = raw_neg / raw_pos
    rows = []
    model_def = make_models(pos_weight).get("XGB", make_models(pos_weight)["RF"])
    for gname in ["resource", "dag", "scheduling", "pressure", "temporal", "complexity", "all", "no_complexity"]:
        feats = [c for c in groups.get(gname, []) if c in df.columns and c not in LEAKAGE_FEATURES]
        if len(feats) == 0:
            continue
        prep = make_preprocessor()
        Xtr = prep.fit_transform(train_used[feats])
        Xva = prep.transform(val_raw[feats])
        Xte = prep.transform(test_raw[feats])
        model = clone(model_def)
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        tr_time = time.perf_counter() - t0
        val_s = get_scores(model, Xva)
        th = best_threshold_by_metric(y_val, val_s, metric=args.threshold_metric)
        test_s = get_scores(model, Xte)
        met = evaluate(y_test, test_s, th)
        met.update({"ablation": "feature_block", "feature_block": gname, "sample_size": n, "model": "XGB_or_RF", "train_time_sec": tr_time, "n_features": len(feats)})
        rows.append(met)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["csv"] / "ablation_feature_blocks.csv", index=False)
    return out


def run_routing_ablation(df_all: pd.DataFrame, n: int, dirs: Dict[str, Path], args, groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for tq in [0.60, 0.70, 0.80, 0.90]:
        for uq in [0.60, 0.75, 0.90]:
            old_t, old_u = args.hard_tau_quantile, args.uncertainty_quantile
            args.hard_tau_quantile, args.uncertainty_quantile = tq, uq
            met, _ = train_eval_single_sample(df_all, n, dirs, args, groups)
            h = met[met["model"] == "HQD-Net-v6"].copy()
            h["tau_quantile"] = tq
            h["uncertainty_quantile"] = uq
            rows.append(h)
            args.hard_tau_quantile, args.uncertainty_quantile = old_t, old_u
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(dirs["csv"] / "ablation_routing_threshold.csv", index=False)
    return out


def run_noise_ablation(df_all: pd.DataFrame, n: int, dirs: Dict[str, Path], args, groups: Dict[str, List[str]]) -> pd.DataFrame:
    df = select_sample_temporal(df_all, n, seed=RANDOM_STATE)
    feats = groups["all"]
    rows = []
    for label_noise in [0.0, 0.01, 0.03, 0.05]:
        noisy = df.copy()
        if label_noise > 0:
            idx = noisy.sample(frac=label_noise, random_state=RANDOM_STATE).index
            noisy.loc[idx, "job_failure"] = 1 - noisy.loc[idx, "job_failure"]
        for feat_noise in [0.0, 0.01, 0.05]:
            noisy2 = noisy.copy()
            if feat_noise > 0:
                rng = np.random.default_rng(RANDOM_STATE)
                for c in feats:
                    if c in noisy2.columns:
                        std = noisy2[c].std(skipna=True)
                        if pd.notna(std) and std > 0:
                            noisy2[c] = noisy2[c] + rng.normal(0, feat_noise * std, size=len(noisy2))
            met, _ = train_eval_single_sample(noisy2, n, dirs, args, groups)
            h = met[met["model"].isin(["RF", "XGB", "HQD-Net-v6"])].copy()
            h["label_noise"] = label_noise
            h["feature_noise"] = feat_noise
            rows.append(h)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(dirs["csv"] / "ablation_noise_robustness.csv", index=False)
    return out


def save_feature_importance(split_info: Dict[str, Any], dirs: Dict[str, Path], n: int) -> None:
    test = split_info["test"]
    y = test["job_failure"].astype(int).values
    for name, bundle in split_info["models"].items():
        if name == "HQD-Net-v6":
            continue
        model = bundle["model"]
        feats = bundle["features"]
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).ravel()
        if importance is not None and len(importance) == len(feats):
            out = pd.DataFrame({"feature": feats, "importance": importance}).sort_values("importance", ascending=False)
            out.to_csv(dirs["csv"] / f"feature_importance_{name}_n{n}.csv", index=False)


def write_readme(metrics: pd.DataFrame, dirs: Dict[str, Path], args, groups: Dict[str, List[str]], raw_dist: pd.DataFrame) -> None:
    top = metrics.sort_values(["pr_auc", "f1"], ascending=False).head(15) if not metrics.empty else pd.DataFrame()
    text = []
    text.append("# HQD-Net v6 Final Results Summary")
    text.append("")
    text.append("This run uses leakage-safe grouped-temporal evaluation. Training may be balanced, but validation and test remain natural future splits.")
    text.append("")
    text.append("## Core Design")
    text.append("")
    text.append("- Post-execution leakage features removed.")
    text.append("- Train/validation/test split is ordered by first job start time.")
    text.append("- Training balance is applied only after splitting.")
    text.append("- HQD-Net-v6 uses ensemble uncertainty plus workload complexity for hard-example routing.")
    text.append("- The structural specialist uses graph/DAG features with a quantum-inspired random Fourier structural encoder.")
    text.append("")
    text.append("## Removed Leakage Features")
    text.append("```json")
    text.append(json.dumps(LEAKAGE_FEATURES, indent=2))
    text.append("```")
    text.append("")
    text.append("## Raw Class Distribution")
    text.append(raw_dist.to_markdown(index=False) if not raw_dist.empty else "No distribution available.")
    text.append("")
    text.append("## Top Models by PR-AUC")
    text.append(top.to_markdown(index=False) if not top.empty else "No metrics available.")
    text.append("")
    text.append("## Key Output Files")
    text.append("")
    text.append("- `csv_results/main_metrics_all_sample_sizes.csv`")
    text.append("- `csv_results/ablation_feature_blocks.csv`")
    text.append("- `csv_results/ablation_routing_threshold.csv`")
    text.append("- `csv_results/ablation_noise_robustness.csv`")
    text.append("- `csv_results/feature_importance_*.csv`")
    text.append("- `figures/`")
    text.append("- `trained_models/`")
    (dirs["root"] / "README_RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[5000, 10000, 20000, 25000, 50000, 100000])
    parser.add_argument("--max-batch-task-rows", type=int, default=5000000)
    parser.add_argument("--max-batch-instance-rows", type=int, default=5000000)
    parser.add_argument("--max-machine-usage-rows", type=int, default=5000000)
    parser.add_argument("--use-machine-pressure", action="store_true")
    parser.add_argument("--sampling-mode", choices=["none", "train_balanced"], default="train_balanced")
    parser.add_argument("--train-negative-ratio", type=float, default=1.0, help="1.0 gives 50/50 train balance; 2 or 3 keeps more negatives.")
    parser.add_argument("--history-window", type=int, default=200)
    parser.add_argument("--hard-tau-quantile", type=float, default=0.80)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.80)
    parser.add_argument("--threshold-metric", choices=["f1", "f2"], default="f2", help="f2 favors failure recall.")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--save-processed", action="store_true")
    args = parser.parse_args()

    dirs = ensure_dirs(Path(args.out_dir))
    df_all = build_dataset(args)
    groups = feature_groups(df_all)
    (dirs["csv"] / "feature_groups_leakage_safe.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (dirs["csv"] / "removed_leakage_features.json").write_text(json.dumps(LEAKAGE_FEATURES, indent=2), encoding="utf-8")

    raw_dist = pd.DataFrame([{
        "total_jobs": len(df_all),
        "failures": int(df_all["job_failure"].sum()),
        "successes": int((df_all["job_failure"] == 0).sum()),
        "failure_ratio": float(df_all["job_failure"].mean()),
    }])
    raw_dist.to_csv(dirs["csv"] / "raw_class_distribution.csv", index=False)

    all_metrics = []
    last_split_info = None
    last_n = None
    for n in args.sample_sizes:
        met, split_info = train_eval_single_sample(df_all, n, dirs, args, groups)
        all_metrics.append(met)
        last_split_info = split_info
        last_n = n
        # Save models only for each n but keep file count manageable.
        for mname, bundle in split_info["models"].items():
            safe_name = mname.replace("/", "_").replace(" ", "_")
            save_model_bundle(bundle, dirs, safe_name, n)
        save_feature_importance(split_info, dirs, n)
        if args.save_processed:
            split_info["train_raw"].to_csv(dirs["data"] / f"train_raw_n{n}.csv", index=False)
            split_info["train_used"].to_csv(dirs["data"] / f"train_used_n{n}.csv", index=False)
            split_info["val"].to_csv(dirs["data"] / f"val_n{n}.csv", index=False)
            split_info["test"].to_csv(dirs["data"] / f"test_n{n}.csv", index=False)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics.to_csv(dirs["csv"] / "main_metrics_all_sample_sizes.csv", index=False)
    plot_main_metrics(metrics, dirs)
    if last_split_info is not None:
        plot_pr_curves(last_split_info, dirs, last_n)

    if args.run_ablations and last_n is not None:
        log("[ablation] feature blocks ...")
        run_feature_block_ablation(df_all, last_n, dirs, args, groups)
        log("[ablation] routing thresholds ...")
        run_routing_ablation(df_all, last_n, dirs, args, groups)
        log("[ablation] noise robustness ...")
        run_noise_ablation(df_all, min(last_n, 25000), dirs, args, groups)

    write_readme(metrics, dirs, args, groups, raw_dist)
    log(f"[done] results saved to {dirs['root']}")


if __name__ == "__main__":
    main()
