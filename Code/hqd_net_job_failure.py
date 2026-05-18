# -*- coding: utf-8 -*-
"""
HQD-Net: Complexity-Aware Hybrid Quantum-Classical Classifier
for Alibaba Cluster Job Failure Prediction

Scalability setting: 5k, 10k, 20k, 25k samples.

This script is designed to be CPU-safe and Qiskit-free by default. It uses a
lightweight circuit-inspired quantum structural similarity encoder (QSSE) with
an SVC precomputed kernel only on the routed complex subset.

Author: generated for research prototyping
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

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

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def now_s() -> float:
    return time.perf_counter()


def rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def safe_numeric(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_string(df: pd.DataFrame, col: Optional[str], default: str = "unknown") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].astype(str).fillna(default)


def infer_success_label(status: pd.Series) -> pd.Series:
    """Infer binary success label from Alibaba-like status/event fields."""
    s = status.astype(str).str.lower().str.strip()
    success_tokens = ["success", "succeeded", "finished", "finish", "terminated", "completed", "complete", "done"]
    fail_tokens = ["fail", "failed", "killed", "evict", "lost", "error", "timeout", "cancel", "interrupt"]
    y = pd.Series(np.nan, index=status.index)
    for tok in success_tokens:
        y[s.str.contains(tok, na=False)] = 1
    for tok in fail_tokens:
        y[s.str.contains(tok, na=False)] = 0
    return y


def parse_task_dependencies(task_name: str) -> Tuple[int, List[int]]:
    """Parse Alibaba task_name format such as M5_3_4 -> task id 5 depends on 3 and 4."""
    if pd.isna(task_name):
        return -1, []
    text = str(task_name)
    nums = re.findall(r"\d+", text)
    if not nums:
        return -1, []
    node = int(nums[0])
    deps = [int(x) for x in nums[1:]]
    return node, deps


def compute_dag_features(task_df: pd.DataFrame, job_col: str, task_col: str) -> pd.DataFrame:
    """Compute job-level DAG features from batch_task.csv."""
    rows = []
    small = task_df[[job_col, task_col]].dropna().copy()
    for job, grp in small.groupby(job_col, sort=False):
        nodes = set()
        edges = []
        dep_counts = []
        for tn in grp[task_col].astype(str):
            node, deps = parse_task_dependencies(tn)
            if node >= 0:
                nodes.add(node)
                dep_counts.append(len(deps))
                for d in deps:
                    nodes.add(d)
                    edges.append((d, node))
        if not nodes:
            rows.append((job, 1, 0, 0, 0, 1, 0, 0))
            continue
        children = {n: [] for n in nodes}
        indeg = {n: 0 for n in nodes}
        for u, v in edges:
            children.setdefault(u, []).append(v)
            indeg[v] = indeg.get(v, 0) + 1
            indeg.setdefault(u, 0)
        roots = [n for n, d in indeg.items() if d == 0]
        # Approximate longest path depth using Kahn traversal.
        depth = {n: 1 for n in nodes}
        q = list(roots)
        seen = 0
        while q:
            u = q.pop(0)
            seen += 1
            for v in children.get(u, []):
                depth[v] = max(depth.get(v, 1), depth[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        dag_depth = max(depth.values()) if depth else 1
        fanouts = [len(children.get(n, [])) for n in nodes]
        rows.append((
            job,
            len(nodes),
            len(edges),
            dag_depth,
            float(np.mean(fanouts)) if fanouts else 0.0,
            float(np.max(fanouts)) if fanouts else 0.0,
            float(np.mean(dep_counts)) if dep_counts else 0.0,
            len(roots),
        ))
    out = pd.DataFrame(rows, columns=[
        job_col, "dag_num_tasks", "dag_num_edges", "dag_depth", "dag_fanout_mean",
        "dag_fanout_max", "dag_dep_mean", "dag_num_roots"
    ])
    return out

# -----------------------------------------------------------------------------
# Data construction
# -----------------------------------------------------------------------------

def load_csv_head(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(path, nrows=nrows, low_memory=False))


def build_dataset(data_dir: Path, max_raw_rows: int = 2_000_000) -> pd.DataFrame:
    """Build a job-instance learning table from Alibaba v2018 CSV files."""
    bi_path = data_dir / "batch_instance.csv"
    bt_path = data_dir / "batch_task.csv"
    if not bi_path.exists():
        raise FileNotFoundError(f"Missing required file: {bi_path}")
    if not bt_path.exists():
        raise FileNotFoundError(f"Missing required file: {bt_path}")

    print(f"[data] reading {bi_path} ...")
    bi = load_csv_head(bi_path, nrows=max_raw_rows)
    print(f"[data] batch_instance shape={bi.shape}")

    print(f"[data] reading {bt_path} ...")
    bt = load_csv_head(bt_path, nrows=max_raw_rows)
    print(f"[data] batch_task shape={bt.shape}")

    cols_bi = bi.columns
    cols_bt = bt.columns
    job_col_bi = first_existing(cols_bi, ["job_name", "job", "job_id", "jobid"])
    task_col_bi = first_existing(cols_bi, ["task_name", "task", "task_id", "taskid"])
    status_col = first_existing(cols_bi, ["status", "event", "event_type", "instance_status", "final_status"])
    job_col_bt = first_existing(cols_bt, ["job_name", "job", "job_id", "jobid"])
    task_col_bt = first_existing(cols_bt, ["task_name", "task", "task_id", "taskid"])

    if status_col is None:
        raise ValueError("Could not find a status/event column in batch_instance.csv. Please rename it to 'status'.")
    if job_col_bi is None or task_col_bi is None or job_col_bt is None or task_col_bt is None:
        raise ValueError("Could not find job/task columns. Expected job_name and task_name or similar names.")

    # Labels.
    y = infer_success_label(bi[status_col])
    bi = bi.loc[y.notna()].copy()
    bi["event_success"] = y.loc[y.notna()].astype(int).values

    # Core numeric columns.
    plan_cpu_col = first_existing(cols_bi, ["plan_cpu", "plancpu", "cpu", "request_cpu", "cpu_request"])
    plan_mem_col = first_existing(cols_bi, ["plan_mem", "planmem", "mem", "request_mem", "memory_request"])
    start_col = first_existing(cols_bi, ["start_time", "start", "start_timestamp", "time_start"])
    end_col = first_existing(cols_bi, ["end_time", "end", "end_timestamp", "time_end"])
    submit_col = first_existing(cols_bi, ["submit_time", "submit", "created_time", "create_time"])
    machine_col = first_existing(cols_bi, ["machine_id", "machine", "host", "server_id"])

    out = pd.DataFrame(index=bi.index)
    out["job_name"] = safe_string(bi, job_col_bi)
    out["task_name"] = safe_string(bi, task_col_bi)
    out["event_success"] = bi["event_success"].astype(int)
    out["plan_cpu"] = safe_numeric(bi, plan_cpu_col)
    out["plan_mem"] = safe_numeric(bi, plan_mem_col)
    out["cpu_mem_ratio"] = out["plan_cpu"] / (out["plan_mem"].replace(0, np.nan))
    out["start_time"] = safe_numeric(bi, start_col)
    out["end_time"] = safe_numeric(bi, end_col)
    out["submit_time"] = safe_numeric(bi, submit_col)
    out["runtime"] = (out["end_time"] - out["start_time"]).clip(lower=0)
    out["queue_delay"] = (out["start_time"] - out["submit_time"]).clip(lower=0)
    out["machine_id"] = safe_string(bi, machine_col)

    # Optional scheduling/priority fields.
    for new_col, aliases in {
        "scheduling_class": ["scheduling_class", "schedulingclass", "schedule_class", "class"],
        "priority": ["priority", "job_priority", "task_priority"],
        "instance_num": ["instance_num", "inst_num", "num_instances"],
    }.items():
        c = first_existing(cols_bi, aliases)
        if c is not None:
            out[new_col] = bi[c]

    # DAG features.
    print("[data] computing DAG features from batch_task.csv ...")
    dag = compute_dag_features(bt, job_col_bt, task_col_bt)
    dag = dag.rename(columns={job_col_bt: "job_name"})
    out = out.merge(dag, on="job_name", how="left")

    # Fill structural features if missing.
    dag_cols = [c for c in out.columns if c.startswith("dag_")]
    out[dag_cols] = out[dag_cols].fillna(0)

    # Simple rolling historical failure proxy using time order only inside raw table.
    out = out.sort_values("start_time").reset_index(drop=True)
    out["global_prev_failure_rate"] = (1 - out["event_success"]).expanding().mean().shift(1).fillna(0.0)
    out["job_prev_failure_rate"] = out.groupby("job_name")["event_success"].transform(lambda s: (1 - s).expanding().mean().shift(1)).fillna(0.0)

    # Cleanup infinities.
    out = out.replace([np.inf, -np.inf], np.nan)
    print(f"[data] final constructed dataset shape={out.shape}, positive_rate={out['event_success'].mean():.3f}")
    return out

# -----------------------------------------------------------------------------
# Quantum structural similarity classifier
# -----------------------------------------------------------------------------

class QuantumStructuralFeatureMap(BaseEstimator, TransformerMixin):
    """Lightweight circuit-inspired feature map.

    This is not a hardware quantum run. It approximates a small feature embedding
    that mimics rotation and pairwise entanglement terms, allowing CPU-safe
    experiments without Qiskit installation failures.
    """
    def __init__(self, reps: int = 2, max_pairs: int = 64):
        self.reps = reps
        self.max_pairs = max_pairs

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        pairs = []
        for i in range(self.n_features_in_):
            for j in range(i + 1, self.n_features_in_):
                pairs.append((i, j))
        self.pairs_ = pairs[: self.max_pairs]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = []
        # Rotation-like terms.
        for r in range(1, self.reps + 1):
            feats.append(np.sin(r * np.pi * X))
            feats.append(np.cos(r * np.pi * X))
        # Entanglement-like pair products.
        if getattr(self, "pairs_", None):
            pair_terms = []
            for i, j in self.pairs_:
                pair_terms.append(np.cos(np.pi * (X[:, i] * X[:, j]))[:, None])
                pair_terms.append(np.sin(np.pi * (X[:, i] + X[:, j]))[:, None])
            feats.append(np.hstack(pair_terms))
        Z = np.hstack(feats)
        norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z / norm


def fidelity_kernel(Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    K = np.dot(Z1, Z2.T)
    return np.clip(K * K, 0.0, 1.0)


@dataclass
class HQDNetConfig:
    route_threshold: float = 0.65
    q_train_ratio: float = 0.20
    q_max_train: int = 3000
    quantum_reps: int = 2
    alpha_min: float = 0.20
    alpha_max: float = 0.80


class HQDNet:
    """Complexity-aware hybrid quantum-classical classifier."""

    def __init__(self, config: HQDNetConfig):
        self.config = config
        self.classical_model = None
        self.complexity_cols = None
        self.complexity_scaler = None
        self.q_map = None
        self.q_svc = None
        self.q_X_train_scaled = None
        self.q_Z_train = None
        self.preprocessor = None

    def _make_classical(self):
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    def _complexity_score(self, raw_df: pd.DataFrame, entropy: Optional[np.ndarray] = None) -> np.ndarray:
        Xc = raw_df[self.complexity_cols].fillna(0.0).astype(float).values
        Xs = self.complexity_scaler.transform(Xc)
        score = Xs.mean(axis=1)
        if entropy is not None:
            score = 0.75 * score + 0.25 * entropy
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _entropy_from_prob(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-8, 1 - 1e-8)
        e = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return e  # already max 1 for binary

    def fit(self, X_proc: np.ndarray, raw_train: pd.DataFrame, y: np.ndarray):
        self.classical_model = self._make_classical()
        self.classical_model.fit(X_proc, y)

        # Complexity columns must be interpretable structural/operational features.
        preferred = [
            "dag_depth", "dag_fanout_mean", "dag_fanout_max", "dag_num_edges",
            "dag_dep_mean", "runtime", "queue_delay", "plan_cpu", "plan_mem",
            "global_prev_failure_rate", "job_prev_failure_rate",
        ]
        self.complexity_cols = [c for c in preferred if c in raw_train.columns]
        if not self.complexity_cols:
            self.complexity_cols = [c for c in raw_train.columns if c not in ["event_success", "job_name", "task_name"]][:5]
        self.complexity_scaler = MinMaxScaler()
        self.complexity_scaler.fit(raw_train[self.complexity_cols].fillna(0.0).astype(float).values)

        pc = self.predict_classical_proba(X_proc)
        entropy = self._entropy_from_prob(pc)
        cscore = self._complexity_score(raw_train, entropy)
        n_q = int(max(100, min(self.config.q_max_train, math.ceil(self.config.q_train_ratio * len(y)))))
        n_q = min(n_q, len(y))
        q_idx = np.argsort(cscore)[-n_q:]

        # Ensure both classes are present.
        if len(np.unique(y[q_idx])) < 2:
            q_idx = np.arange(len(y))

        self.q_map = QuantumStructuralFeatureMap(reps=self.config.quantum_reps)
        self.q_X_train_scaled = X_proc[q_idx]
        self.q_Z_train = self.q_map.fit_transform(self.q_X_train_scaled)
        Kq = fidelity_kernel(self.q_Z_train, self.q_Z_train)
        self.q_svc = SVC(kernel="precomputed", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        self.q_svc.fit(Kq, y[q_idx])
        self.q_train_indices_ = q_idx
        return self

    def predict_classical_proba(self, X_proc: np.ndarray) -> np.ndarray:
        if hasattr(self.classical_model, "predict_proba"):
            return self.classical_model.predict_proba(X_proc)[:, 1]
        s = self.classical_model.decision_function(X_proc)
        return 1.0 / (1.0 + np.exp(-s))

    def predict_quantum_proba(self, X_proc: np.ndarray) -> np.ndarray:
        Z = self.q_map.transform(X_proc)
        K = fidelity_kernel(Z, self.q_Z_train)
        return self.q_svc.predict_proba(K)[:, 1]

    def predict_proba(self, X_proc: np.ndarray, raw_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        pc = self.predict_classical_proba(X_proc)
        entropy = self._entropy_from_prob(pc)
        cscore = self._complexity_score(raw_df, entropy)
        route_q = cscore >= self.config.route_threshold

        p = pc.copy()
        routed_count = int(route_q.sum())
        if routed_count > 0:
            pq = self.predict_quantum_proba(X_proc[route_q])
            alpha = self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * cscore[route_q]
            alpha = np.clip(alpha, self.config.alpha_min, self.config.alpha_max)
            p[route_q] = (1.0 - alpha) * pc[route_q] + alpha * pq

        meta = {
            "quantum_route_ratio": float(route_q.mean()),
            "complexity_mean": float(np.mean(cscore)),
            "complexity_p90": float(np.quantile(cscore, 0.90)),
        }
        return p, meta

    def predict(self, X_proc: np.ndarray, raw_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        p, meta = self.predict_proba(X_proc, raw_df)
        return (p >= 0.5).astype(int), meta

# -----------------------------------------------------------------------------
# Preprocessing and evaluation
# -----------------------------------------------------------------------------

def make_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        if c in ["job_name", "task_name"]:
            continue
        if df[c].dtype == object:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")


def metrics_dict(y_true: np.ndarray, prob: np.ndarray, pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    if pred is None:
        pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = average_precision_score(y_true, prob)
    except Exception:
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out


def get_model_suite() -> Dict[str, object]:
    models = {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "HGB": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=80, early_stopping=True, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.04, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def evaluate_single_size(df: pd.DataFrame, sample_n: int, out_dir: Path, temporal_split: bool = True) -> pd.DataFrame:
    print(f"\n[run] sample size={sample_n}")
    if sample_n < len(df):
        df_s = df.groupby("event_success", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_n * len(x) / len(df)))), random_state=RANDOM_STATE)
        )
        # exact fill if group proportion rounding made sample short
        if len(df_s) < sample_n:
            extra = df.drop(df_s.index).sample(sample_n - len(df_s), random_state=RANDOM_STATE)
            df_s = pd.concat([df_s, extra], axis=0)
        df_s = df_s.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_s = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    y = df_s["event_success"].astype(int).values
    drop_cols = ["event_success"]
    feature_cols = [c for c in df_s.columns if c not in drop_cols]
    # prevent IDs from exploding dimensionality but keep job/task names for DAG already captured
    feature_cols = [c for c in feature_cols if c not in ["job_name", "task_name"]]

    if temporal_split and "start_time" in df_s.columns:
        df_s = df_s.sort_values("start_time").reset_index(drop=True)
        y = df_s["event_success"].astype(int).values
        n_train = int(0.70 * len(df_s))
        n_val = int(0.10 * len(df_s))
        train_idx = np.arange(0, n_train + n_val)  # use train+val for model training in benchmark
        test_idx = np.arange(n_train + n_val, len(df_s))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df_s)), test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

    raw_train = df_s.iloc[train_idx].copy()
    raw_test = df_s.iloc[test_idx].copy()
    y_train = raw_train["event_success"].astype(int).values
    y_test = raw_test["event_success"].astype(int).values

    pre = make_preprocessor(raw_train, feature_cols)
    X_train = pre.fit_transform(raw_train[feature_cols])
    X_test = pre.transform(raw_test[feature_cols])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    rows = []
    baseline_lr_time = None
    baseline_lr_mem = None

    for name, model in get_model_suite().items():
        gc.collect()
        mem0 = rss_mb()
        t0 = now_s()
        model.fit(X_train, y_train)
        train_time = now_s() - t0
        mem1 = rss_mb()

        t1 = now_s()
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            s = model.decision_function(X_test)
            prob = 1.0 / (1.0 + np.exp(-s))
        infer_time = now_s() - t1
        pred = (prob >= 0.5).astype(int)
        m = metrics_dict(y_test, prob, pred)
        if name == "LR":
            baseline_lr_time = max(train_time, 1e-9)
            baseline_lr_mem = max(mem1 - mem0, 1e-9)
        row = {
            "sample_n": sample_n,
            "split": "temporal" if temporal_split else "iid",
            "model": name,
            **m,
            "train_time_s": train_time,
            "inference_time_s": infer_time,
            "latency_ms_per_sample": 1000 * infer_time / max(len(y_test), 1),
            "throughput_samples_s": len(y_test) / max(infer_time, 1e-9),
            "ram_delta_mb": mem1 - mem0,
            "quantum_route_ratio": 0.0,
            "complexity_mean": np.nan,
            "complexity_p90": np.nan,
        }
        rows.append(row)
        print(f"  {name:8s} Acc={m['accuracy']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} time={train_time:.2f}s")

    # HQD-Net proposed.
    gc.collect()
    mem0 = rss_mb()
    cfg = HQDNetConfig(route_threshold=0.65, q_train_ratio=0.20, q_max_train=3000, quantum_reps=2)
    hqd = HQDNet(cfg)
    t0 = now_s()
    hqd.fit(X_train, raw_train, y_train)
    train_time = now_s() - t0
    mem1 = rss_mb()
    t1 = now_s()
    prob, meta = hqd.predict_proba(X_test, raw_test)
    infer_time = now_s() - t1
    pred = (prob >= 0.5).astype(int)
    m = metrics_dict(y_test, prob, pred)
    row = {
        "sample_n": sample_n,
        "split": "temporal" if temporal_split else "iid",
        "model": "HQD-Net",
        **m,
        "train_time_s": train_time,
        "inference_time_s": infer_time,
        "latency_ms_per_sample": 1000 * infer_time / max(len(y_test), 1),
        "throughput_samples_s": len(y_test) / max(infer_time, 1e-9),
        "ram_delta_mb": mem1 - mem0,
        **meta,
    }
    rows.append(row)
    print(f"  {'HQD-Net':8s} Acc={m['accuracy']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} routeQ={meta['quantum_route_ratio']:.2f} time={train_time:.2f}s")

    res = pd.DataFrame(rows)
    if baseline_lr_time is None:
        baseline_lr_time = max(res["train_time_s"].min(), 1e-9)
    if baseline_lr_mem is None or not np.isfinite(baseline_lr_mem):
        baseline_lr_mem = 1.0
    res["training_efficiency_index"] = (
        res["train_time_s"] / max(baseline_lr_time, 1e-9)
        + 0.05 * res["ram_delta_mb"].clip(lower=0) / max(abs(baseline_lr_mem), 1e-9)
    )
    return res

# -----------------------------------------------------------------------------
# Ablations
# -----------------------------------------------------------------------------

def run_hqd_threshold_ablation(df: pd.DataFrame, sample_n: int, out_dir: Path) -> pd.DataFrame:
    print("\n[ablation] HQD-Net routing threshold sweep")
    df_s = df.sample(min(sample_n, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    y = df_s["event_success"].astype(int).values
    feature_cols = [c for c in df_s.columns if c not in ["event_success", "job_name", "task_name"]]
    train_idx, test_idx = train_test_split(np.arange(len(df_s)), test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    raw_train = df_s.iloc[train_idx].copy()
    raw_test = df_s.iloc[test_idx].copy()
    y_train = raw_train["event_success"].astype(int).values
    y_test = raw_test["event_success"].astype(int).values
    pre = make_preprocessor(raw_train, feature_cols)
    X_train = pre.fit_transform(raw_train[feature_cols])
    X_test = pre.transform(raw_test[feature_cols])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    rows = []
    for th in [0.45, 0.55, 0.65, 0.75, 0.85]:
        cfg = HQDNetConfig(route_threshold=th, q_train_ratio=0.20, q_max_train=3000, quantum_reps=2)
        model = HQDNet(cfg)
        t0 = now_s()
        model.fit(X_train, raw_train, y_train)
        train_time = now_s() - t0
        prob, meta = model.predict_proba(X_test, raw_test)
        m = metrics_dict(y_test, prob)
        rows.append({"threshold": th, **m, **meta, "train_time_s": train_time})
        print(f"  th={th:.2f} Acc={m['accuracy']:.3f} F1={m['f1']:.3f} routeQ={meta['quantum_route_ratio']:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "ablation_threshold_hqd.csv", index=False)
    return out


def run_feature_block_ablation(df: pd.DataFrame, sample_n: int, out_dir: Path) -> pd.DataFrame:
    print("\n[ablation] feature block ablation using XGBoost/RF fallback")
    blocks = {
        "resource": ["plan_cpu", "plan_mem", "cpu_mem_ratio"],
        "dag": ["dag_num_tasks", "dag_num_edges", "dag_depth", "dag_fanout_mean", "dag_fanout_max", "dag_dep_mean", "dag_num_roots"],
        "temporal": ["start_time", "end_time", "submit_time", "runtime", "queue_delay"],
        "history": ["global_prev_failure_rate", "job_prev_failure_rate"],
        "all": [c for c in df.columns if c not in ["event_success", "job_name", "task_name"]],
    }
    rows = []
    df_s = df.sample(min(sample_n, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    y = df_s["event_success"].astype(int).values
    for bname, cols in blocks.items():
        cols = [c for c in cols if c in df_s.columns]
        if not cols:
            continue
        tr, te = train_test_split(np.arange(len(df_s)), test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        raw_train = df_s.iloc[tr].copy()
        raw_test = df_s.iloc[te].copy()
        pre = make_preprocessor(raw_train, cols)
        Xtr = pre.fit_transform(raw_train[cols])
        Xte = pre.transform(raw_test[cols])
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray().astype(np.float32)
            Xte = Xte.toarray().astype(np.float32)
        model = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE) if HAS_XGB else RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample")
        t0 = now_s()
        model.fit(Xtr, raw_train["event_success"].astype(int).values)
        train_time = now_s() - t0
        prob = model.predict_proba(Xte)[:, 1]
        m = metrics_dict(raw_test["event_success"].astype(int).values, prob)
        rows.append({"feature_block": bname, **m, "train_time_s": train_time, "num_features_raw": len(cols)})
        print(f"  {bname:10s} Acc={m['accuracy']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "ablation_feature_blocks.csv", index=False)
    return out

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # Accuracy/F1 scalability plot.
    for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]:
        fig, ax = plt.subplots(figsize=(16, 9))
        for model, g in results.groupby("model"):
            g = g.sort_values("sample_n")
            ax.plot(g["sample_n"], g[metric], marker="o", linewidth=2.2, label=model)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"Scalability Trend: {metric.replace('_', ' ').upper()}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_scalability_{metric}.png", dpi=600)
        plt.close(fig)

    # Deployment trade-off plot.
    fig, ax = plt.subplots(figsize=(16, 9))
    latest = results[results["sample_n"] == results["sample_n"].max()].copy()
    ax.scatter(latest["latency_ms_per_sample"], latest["f1"], s=180)
    for _, r in latest.iterrows():
        ax.text(r["latency_ms_per_sample"], r["f1"] + 0.003, r["model"], ha="center")
    ax.set_xlabel("Latency per Sample (ms)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Deployment Trade-off at Largest Sample Size")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_deployment_latency_f1.png", dpi=600)
    plt.close(fig)

    # Quantum route ratio.
    hqd = results[results["model"] == "HQD-Net"].sort_values("sample_n")
    if not hqd.empty:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.bar(hqd["sample_n"].astype(str), hqd["quantum_route_ratio"])
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Quantum Route Ratio")
        ax.set_title("Selective Quantum Utilization in HQD-Net")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_hqd_quantum_route_ratio.png", dpi=600)
        plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HQD-Net Alibaba job failure prediction study")
    p.add_argument("--data-dir", type=str, required=True, help="Folder containing Alibaba CSV files")
    p.add_argument("--out-dir", type=str, default="results_hqd_net", help="Output directory")
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[5000, 10000, 20000, 25000])
    p.add_argument("--max-raw-rows", type=int, default=2_000_000, help="Rows to read from large CSVs for prototyping")
    p.add_argument("--iid", action="store_true", help="Use IID split instead of temporal split")
    p.add_argument("--run-ablations", action="store_true", help="Run feature/routing ablations")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    df = build_dataset(data_dir, max_raw_rows=args.max_raw_rows)
    df.to_csv(out_dir / "constructed_learning_table_preview.csv", index=False)

    all_results = []
    for n in args.sample_sizes:
        res = evaluate_single_size(df, n, out_dir, temporal_split=not args.iid)
        all_results.append(res)
        pd.concat(all_results, ignore_index=True).to_csv(out_dir / "results_main_scalability.csv", index=False)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(out_dir / "results_main_scalability.csv", index=False)
    plot_results(results, out_dir)

    if args.run_ablations:
        run_hqd_threshold_ablation(df, min(max(args.sample_sizes), 25000), out_dir)
        run_feature_block_ablation(df, min(max(args.sample_sizes), 25000), out_dir)

    print("\n[done] outputs written to:", out_dir.resolve())
    print("Main CSV:", out_dir / "results_main_scalability.csv")


if __name__ == "__main__":
    main()
