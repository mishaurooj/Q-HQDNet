
<!-- README generated for the Q-HQDNet / NEBULA-HQD job-failure repository. -->
<div align="center">

# 🌌 Q-HQDNet: Evolutionary Hybrid Quantum-Dependency Network for Cloud Job Failure Prediction

### A versioned research codebase for Alibaba Cluster Trace v2018 experiments

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-02569B?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Optional-FF6600?style=for-the-badge)
![Status](https://img.shields.io/badge/Research-Codebase-16A34A?style=for-the-badge)

</div>

---

> **Repository purpose:** This repository contains a full experimental trail from **V1 to V7** for job failure prediction on Alibaba Cluster Trace v2018. Each version fixes a concrete weakness in the earlier script: leakage, poor failure recall, imbalance, random split optimism, weak routing, limited ablations, and incomplete proposed-system framing.

> **Recommended final script:** Use `hqd_net_job_failure_v7_proposed_system_fixed.py` for paper-grade experiments. Use earlier versions only for ablation history, debugging, and explaining the research evolution.

---

## 📌 Quick Navigation

- [`hqd_net_job_failure.py`](#v1-complexity-aware-hybrid-quantum-classical-classifier) — V1: Complexity-Aware Hybrid Quantum-Classical Classifier
- [`hqd_net_job_failure_v2.py`](#v2-hierarchical-quantum-dependency-network) — V2: Hierarchical Quantum-Dependency Network
- [`hqd_net_job_failure_v3_leakage_safe.py`](#v3-leakage-safe-hqd-net) — V3: Leakage-Safe HQD-Net
- [`hqd_net_job_failure_v4_balanced_hardrouting.py`](#v4-balanced-hard-routing-hqd-net) — V4: Balanced Hard-Routing HQD-Net
- [`hqd_net_job_failure_v5_grouped_temporal_balanced.py`](#v5-grouped-temporal-balanced-hqd-net) — V5: Grouped Temporal Balanced HQD-Net
- [`hqd_net_job_failure_v6_final.py`](#v6-final-leakage-safe-hard-failure-hqd-net) — V6: Final Leakage-Safe Hard-Failure HQD-Net
- [`hqd_net_job_failure_v6_final_notk.py`](#v6notk-v6-variant-without-threshold-k-tuning) — V6-NoTK: V6 variant without threshold-k tuning
- [`hqd_net_job_failure_v7_proposed_system.py`](#v7-proposed-hybrid-system) — V7: Proposed Hybrid System
- [`hqd_net_job_failure_v7_proposed_system_fixed.py`](#v7fixed-fixed-proposed-hybrid-system) — V7-Fixed: Fixed Proposed Hybrid System
- [`inspect_alibaba_csvs.py`](#dataset-inspection-tool) — schema inspection utility
- [Environment setup](#environment-setup)
- [Dataset layout](#dataset-layout)
- [Version comparison table](#version-comparison-table)
- [Run commands](#run-commands)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)


## 🗂️ Repository Map

```text
Q-HQDNet/
└── Code/
    ├── hqd_net_job_failure.py
    ├── hqd_net_job_failure_v2.py
    ├── hqd_net_job_failure_v3_leakage_safe.py
    ├── hqd_net_job_failure_v4_balanced_hardrouting.py
    ├── hqd_net_job_failure_v5_grouped_temporal_balanced.py
    ├── hqd_net_job_failure_v6_final.py
    ├── hqd_net_job_failure_v6_final_notk.py
    ├── hqd_net_job_failure_v7_proposed_system.py
    ├── hqd_net_job_failure_v7_proposed_system_fixed.py
    ├── inspect_alibaba_csvs.py
    └── README.md
```


## 🧠 One-Page Summary

| Area | Final Choice | Reason |
|---|---|---|
| Main final file | `hqd_net_job_failure_v7_proposed_system_fixed.py` | Most complete proposed-system version with leakage-safe grouped-temporal evaluation and validation-tuned fusion. |
| Stable backup | `hqd_net_job_failure_v6_final.py` | Full final-style implementation with strong ablations and model saving. |
| Best diagnostic version | `hqd_net_job_failure_v5_grouped_temporal_balanced.py` | Good for showing why grouped temporal evaluation and train-only balancing matter. |
| Early baseline | `hqd_net_job_failure.py` | Useful only for explaining original idea and baseline constraints. |
| Data inspection | `inspect_alibaba_csvs.py` | Checks schemas and row counts before expensive training. |

<table>
<tr><th>Color</th><th>Meaning</th></tr>
<tr><td style="background:#dbeafe">Blue</td><td>Early architecture and first runnable prototype.</td></tr>
<tr><td style="background:#dcfce7">Green</td><td>Leakage-safe and more scientifically defensible design.</td></tr>
<tr><td style="background:#fef3c7">Yellow</td><td>Balanced training and hard routing improvements.</td></tr>
<tr><td style="background:#fee2e2">Red</td><td>Final experiments, ablations, and paper-ready reporting.</td></tr>
</table>

## ⚙️ Environment Setup

### Option A: Windows / Anaconda
```bash
conda create -n hqdnet python=3.10 -y
conda activate hqdnet
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm openpyxl
pip install xgboost lightgbm
```

### Option B: Plain pip / venv
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm openpyxl
pip install xgboost lightgbm
```

### Option C: Minimal CPU-safe setup
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib tqdm
```

### Optional packages
| Package | Needed for | Install |
|---|---|---|
| `xgboost` | XGB baseline / auxiliary learner | `pip install xgboost` |
| `lightgbm` | LGBM backbone in V7 | `pip install lightgbm` |
| `openpyxl` | Excel inspection and export | `pip install openpyxl` |
| `psutil` | Optional memory diagnostics if added | `pip install psutil` |

### Hardware guidance
| Experiment scale | RAM | CPU | Notes |
|---|---:|---:|---|
| 5k to 25k | 8 GB | 4 cores | Good for fast debugging. |
| 50k | 16 GB | 6 to 8 cores | Recommended for paper plots. |
| 100k | 32 GB | 8+ cores | Use row caps and avoid unnecessary large CSV joins. |
| Full Alibaba extraction | 64 GB+ | 12+ cores | Use staged preprocessing and reuse engineered data. |

## 🧾 Dataset Layout

The scripts assume the Alibaba Cluster Trace v2018 CSV files are stored inside one dataset directory.

```text
Dataset/
├── batch_task.csv
├── batch_instance.csv
├── machine_usage.csv
├── machine_meta.csv
├── container_usage.csv
└── container_meta.csv
```

### Core CSV roles
| File | Role in this project | Required? |
|---|---|---|
| `batch_task.csv` | Main job/task table. Provides task name, job name, requested CPU/memory, status, timing, DAG pattern. | Yes |
| `batch_instance.csv` | Instance-level context and failure aggregation. Used from V2 onward. | Recommended |
| `machine_usage.csv` | Causal machine pressure features. Used when `--use-machine-pressure` is enabled. | Optional |
| `container_usage.csv` | Optional container pressure context in early V2-style experiments. | Optional |
| `machine_meta.csv` | Machine metadata if you extend pressure and failure-domain studies. | Optional |
| `container_meta.csv` | Container metadata for richer workload context. | Optional |

### Header note
Alibaba v2018 files are often treated as headerless. The scripts include schema readers and robust column mapping. If your CSV has a header row, inspect it first with `inspect_alibaba_csvs.py`.

## 🌈 Version Comparison Table

<table>
<tr><th>Version</th><th>Script</th><th>Architecture</th><th>Main improvement</th><th>Main limitation</th><th>Recommended use</th></tr>
<tr><td style="background:#3b82f6;color:white"><b>V1</b></td><td><code>hqd_net_job_failure.py</code></td><td>Complexity-aware classical + quantum-inspired routed classifier</td><td>Introduces DAG-aware complexity routing and QSSE-style kernel specialist</td><td>Early split/evaluation design is weaker than later leakage-safe versions</td><td>Prototype and baseline history</td></tr>
<tr><td style="background:#06b6d4;color:white"><b>V2</b></td><td><code>hqd_net_job_failure_v2.py</code></td><td>Hierarchical Quantum-Dependency Network with richer dataset joins</td><td>Adds task DAG, instance context, optional machine/container usage, model saving, figures</td><td>Still needs stronger leakage controls and temporal evaluation</td><td>Feature-engineering reference</td></tr>
<tr><td style="background:#10b981;color:white"><b>V3</b></td><td><code>hqd_net_job_failure_v3_leakage_safe.py</code></td><td>Leakage-safe hierarchical model</td><td>Removes post-execution leakage and uses train-only preprocessing</td><td>May still use less strict grouping/balancing than V5+</td><td>Leakage-safe baseline</td></tr>
<tr><td style="background:#f59e0b;color:white"><b>V4</b></td><td><code>hqd_net_job_failure_v4_balanced_hardrouting.py</code></td><td>Balanced hard-example routing HQD-Net</td><td>Adds stronger balanced training and hard-route specialist</td><td>Still not the final grouped-temporal formulation</td><td>Hard-routing ablation</td></tr>
<tr><td style="background:#8b5cf6;color:white"><b>V5</b></td><td><code>hqd_net_job_failure_v5_grouped_temporal_balanced.py</code></td><td>Grouped temporal balanced-training HQD-Net</td><td>Uses grouped temporal train/val/test and train-only balancing</td><td>Less complete final reporting than V6/V7</td><td>Defensible temporal evaluation</td></tr>
<tr><td style="background:#ef4444;color:white"><b>V6</b></td><td><code>hqd_net_job_failure_v6_final.py</code></td><td>Final leakage-safe grouped-temporal hard-failure system</td><td>Adds model bundles, PR curves, feature importance, routing/noise/feature ablations</td><td>System still framed as HQD-Net final, not the polished proposed-system narrative</td><td>Stable paper-grade baseline</td></tr>
<tr><td style="background:#ec4899;color:white"><b>V6-NoTK</b></td><td><code>hqd_net_job_failure_v6_final_notk.py</code></td><td>V6 variant without threshold-k variant logic</td><td>Useful sanity branch for threshold-related comparisons</td><td>Not the main recommended final file</td><td>Sensitivity/checkpoint variant</td></tr>
<tr><td style="background:#22c55e;color:white"><b>V7</b></td><td><code>hqd_net_job_failure_v7_proposed_system.py</code></td><td>Full proposed hybrid system</td><td>Uses LGBM backbone, XGB auxiliary learner, structural specialist, validation-tuned fusion</td><td>Initial proposed version may contain small fixes handled in V7-Fixed</td><td>Paper proposed-system draft</td></tr>
<tr><td style="background:#14b8a6;color:white"><b>V7-Fixed</b></td><td><code>hqd_net_job_failure_v7_proposed_system_fixed.py</code></td><td>Fixed full proposed hybrid system</td><td>Most complete and recommended implementation</td><td>Most complex, needs clean environment and enough RAM</td><td>Final Q1-style experiments</td></tr>
</table>

## 🚀 Run Commands

Use these commands from the repository `Code/` directory. Replace paths with your local paths.

### Dataset inspection first
```bash
python inspect_alibaba_csvs.py
```

### Run V1: `hqd_net_job_failure.py`
```bash
python hqd_net_job_failure.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V1" --sample-sizes 5000 10000 20000 25000 --max-raw-rows 2000000 --run-ablations
```

### Run V2: `hqd_net_job_failure_v2.py`
```bash
python hqd_net_job_failure_v2.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V2" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --run-ablations
```

### Run V3: `hqd_net_job_failure_v3_leakage_safe.py`
```bash
python hqd_net_job_failure_v3_leakage_safe.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V3" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --run-ablations
```

### Run V4: `hqd_net_job_failure_v4_balanced_hardrouting.py`
```bash
python hqd_net_job_failure_v4_balanced_hardrouting.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V4" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```

### Run V5: `hqd_net_job_failure_v5_grouped_temporal_balanced.py`
```bash
python hqd_net_job_failure_v5_grouped_temporal_balanced.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V5" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```

### Run V6: `hqd_net_job_failure_v6_final.py`
```bash
python hqd_net_job_failure_v6_final.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V6" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```

### Run V6-NoTK: `hqd_net_job_failure_v6_final_notk.py`
```bash
python hqd_net_job_failure_v6_final_notk.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V6_NoTK" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```

### Run V7: `hqd_net_job_failure_v7_proposed_system.py`
```bash
python hqd_net_job_failure_v7_proposed_system.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V7" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```

### Run V7-Fixed: `hqd_net_job_failure_v7_proposed_system_fixed.py`
```bash
python hqd_net_job_failure_v7_proposed_system_fixed.py --data-dir "D:\other\ALIBABAQUATUM\Dataset" --out-dir "D:\other\ALIBABAQUATUM\Results_V7_Fixed" --sample-sizes 5000 10000 20000 25000 --max-batch-task-rows 2000000 --max-batch-instance-rows 2000000 --max-machine-usage-rows 3000000 --use-machine-pressure --sampling-mode train_balanced --run-ablations
```


## 🏗️ Detailed Version Architecture, Limitations, and Improvements

<a id="v1-complexity-aware-hybrid-quantum-classical-classifier"></a>

### <span style="color:#3b82f6">V1 — Complexity-Aware Hybrid Quantum-Classical Classifier</span>

**File:** `hqd_net_job_failure.py`  
**Lines in script:** `851`  
**Classes:** `QuantumStructuralFeatureMap, HQDNetConfig, HQDNet`

| Dimension | Description |
|---|---|
| Architecture role | Complexity-Aware Hybrid Quantum-Classical Classifier. |
| Best use | Prototype and baseline history. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-raw-rows, --iid, --run-ablations` |

#### Architecture
1. CSV ingestion from batch task data.
2. Feature normalization and simple complexity scoring.
3. Classical model suite including tree and linear models.
4. QuantumStructuralFeatureMap / QSSE style routed specialist.
5. Threshold/complexity ablations and plots.

#### Limitations
1. Early prototype with limited dataset joins.
2. Evaluation can be optimistic relative to temporal deployment.
3. Limited machine pressure and instance context.
4. Less strict leakage audit than V3 onward.

#### What the next version improves
1. V2 adds richer joins and clearer modular outputs.
2. V3 removes post-execution leakage.
3. V4/V5 improve imbalance and routing design.

#### Key functions/classes detected
- `now_s()`
- `rss_mb()`
- `safe_mkdir()`
- `normalize_columns()`
- `first_existing()`
- `safe_numeric()`
- `safe_string()`
- `infer_success_label()`
- `parse_task_dependencies()`
- `compute_dag_features()`
- `load_csv_head()`
- `build_dataset()`
- `fidelity_kernel()`
- `make_preprocessor()`
- `metrics_dict()`
- `get_model_suite()`
- `evaluate_single_size()`
- `run_hqd_threshold_ablation()`
- `run_feature_block_ablation()`
- `plot_results()`
- `parse_args()`
- `main()`
- `__init__()`
- `fit()`
- `transform()`
- `__init__()`
- `_make_classical()`
- `_complexity_score()`
- `_entropy_from_prob()`
- `fit()`
- `predict_classical_proba()`
- `predict_quantum_proba()`
- `predict_proba()`
- `predict()`

<a id="v2-hierarchical-quantum-dependency-network"></a>

### <span style="color:#06b6d4">V2 — Hierarchical Quantum-Dependency Network</span>

**File:** `hqd_net_job_failure_v2.py`  
**Lines in script:** `943`  
**Classes:** `QuantumStructuralEncoder, HQDNet`

| Dimension | Description |
|---|---|
| Architecture role | Hierarchical Quantum-Dependency Network. |
| Best use | Feature-engineering reference. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --max-container-usage-rows, --use-container-usage, --run-ablations, --reuse-engineered` |

#### Architecture
1. Richer Alibaba v2018 ingestion.
2. Task DAG feature builder.
3. Instance feature builder.
4. Machine/container usage hooks.
5. Hierarchical quantum-dependency network class.
6. Model artifacts and publication-style figures.

#### Limitations
1. Some engineered context can still require leakage auditing.
2. Temporal grouping not final.
3. Balancing strategy not yet as defensible as later versions.

#### What the next version improves
1. V3 makes leakage safety explicit.
2. V4 introduces balanced hard routing.
3. V5 adds grouped temporal split.

#### Key functions/classes detected
- `log()`
- `ensure_dir()`
- `read_csv_schema()`
- `to_num()`
- `parse_task_index()`
- `parse_parent_indices()`
- `status_to_failure()`
- `safe_div()`
- `build_machine_usage_features()`
- `build_container_usage_features()`
- `build_task_dag_features()`
- `build_instance_features()`
- `add_temporal_features()`
- `minmax_series()`
- `add_complexity_score()`
- `build_dataset()`
- `get_feature_groups()`
- `make_preprocessor()`
- `make_model()`
- `predict_scores()`
- `metrics_dict()`
- `sample_balanced_or_stratified()`
- `train_single_model()`
- `train_hqdnet()`
- `plot_metrics_bar()`
- `plot_scalability()`
- `plot_latency()`
- `plot_ablation()`
- `save_feature_importance()`
- `run_main_experiments()`
- `run_all_ablations()`
- `parse_args()`
- `main()`
- `__init__()`
- `fit()`
- ... plus 5 more functions.

<a id="v3-leakage-safe-hqd-net"></a>

### <span style="color:#10b981">V3 — Leakage-Safe HQD-Net</span>

**File:** `hqd_net_job_failure_v3_leakage_safe.py`  
**Lines in script:** `790`  
**Classes:** `QuantumStructuralEncoder, HQDNet`

| Dimension | Description |
|---|---|
| Architecture role | Leakage-Safe HQD-Net. |
| Best use | Leakage-safe baseline. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --run-ablations, --reuse-engineered` |

#### Architecture
1. Leakage-safe feature removal.
2. Train-only sklearn preprocessing pipeline.
3. Threshold tuning using training data.
4. Causal previous-time machine pressure approximation.
5. Ablation CSVs and figures.

#### Limitations
1. Hard-example routing still evolving.
2. Grouped temporal split and train-only balancing need strengthening.
3. Final proposed-system ensemble not yet introduced.

#### What the next version improves
1. V4 adds stronger hard routing and balancing.
2. V5 adds grouped temporal train/val/test.
3. V6 adds final artifacts and ablations.

#### Key functions/classes detected
- `log()`
- `ensure_dir()`
- `read_csv_schema()`
- `to_num()`
- `safe_div()`
- `status_to_failure()`
- `parse_parent_indices()`
- `parse_task_index()`
- `build_task_dag_features()`
- `build_instance_label_and_context()`
- `build_machine_usage_timeline()`
- `causal_attach_machine_pressure()`
- `add_temporal_features()`
- `minmax_series()`
- `col_or_zero()`
- `add_complexity_score()`
- `drop_leakage_columns()`
- `build_dataset()`
- `get_feature_groups()`
- `make_preprocessor()`
- `make_model()`
- `predict_scores()`
- `tune_threshold()`
- `metrics_dict()`
- `sample_stratified()`
- `split_data()`
- `train_single_model()`
- `train_hqdnet()`
- `plot_metric_bars()`
- `plot_lines()`
- `plot_bar()`
- `save_feature_importance()`
- `run_ablations()`
- `run_main()`
- `parse_args()`
- ... plus 8 more functions.

<a id="v4-balanced-hard-routing-hqd-net"></a>

### <span style="color:#f59e0b">V4 — Balanced Hard-Routing HQD-Net</span>

**File:** `hqd_net_job_failure_v4_balanced_hardrouting.py`  
**Lines in script:** `872`  
**Classes:** `QuantumStructuralEncoder, HQDNet`

| Dimension | Description |
|---|---|
| Architecture role | Balanced Hard-Routing HQD-Net. |
| Best use | Hard-routing ablation. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --run-ablations, --reuse-engineered, --sampling-mode` |

#### Architecture
1. Balanced leakage-safe HQD-Net.
2. Hard-example quantum-dependency routing.
3. Failure-focused threshold tuning.
4. Feature block ablations and routing diagnostics.
5. Optional machine pressure.

#### Limitations
1. Grouped temporal split not yet the mature default.
2. Final reporting is less complete than V6.
3. May still need careful sampling-mode documentation.

#### What the next version improves
1. V5 formalizes grouped temporal balanced training.
2. V6 adds final model bundles and robust outputs.

#### Key functions/classes detected
- `log()`
- `ensure_dir()`
- `read_csv_schema()`
- `to_num()`
- `safe_div()`
- `status_to_failure()`
- `parse_parent_indices()`
- `parse_task_index()`
- `build_task_dag_features()`
- `build_instance_label_and_context()`
- `build_machine_usage_timeline()`
- `causal_attach_machine_pressure()`
- `add_temporal_features()`
- `minmax_series()`
- `col_or_zero()`
- `add_complexity_score()`
- `drop_leakage_columns()`
- `build_dataset()`
- `get_feature_groups()`
- `make_preprocessor()`
- `make_model()`
- `predict_scores()`
- `tune_threshold()`
- `metrics_dict()`
- `sample_dataset()`
- `sample_stratified()`
- `split_data()`
- `split_train_val_test()`
- `train_single_model()`
- `train_hqdnet()`
- `plot_metric_bars()`
- `plot_lines()`
- `plot_bar()`
- `save_feature_importance()`
- `run_ablations()`
- ... plus 11 more functions.

<a id="v5-grouped-temporal-balanced-hqd-net"></a>

### <span style="color:#8b5cf6">V5 — Grouped Temporal Balanced HQD-Net</span>

**File:** `hqd_net_job_failure_v5_grouped_temporal_balanced.py`  
**Lines in script:** `913`  
**Classes:** `QuantumStructuralEncoder, HQDNet`

| Dimension | Description |
|---|---|
| Architecture role | Grouped Temporal Balanced HQD-Net. |
| Best use | Defensible temporal evaluation. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --run-ablations, --reuse-engineered, --sampling-mode` |

#### Architecture
1. Grouped temporal leakage-safe split.
2. Train-only balancing.
3. Hard-example specialist.
4. Causal machine pressure option.
5. Feature groups and ablation plots.

#### Limitations
1. Final paper narrative still less polished.
2. Model saving/reporting less complete than V6/V7.
3. Proposed-system fusion not final.

#### What the next version improves
1. V6 stabilizes final experiment package.
2. V7 reframes as full proposed hybrid system.

#### Key functions/classes detected
- `log()`
- `ensure_dir()`
- `read_csv_schema()`
- `to_num()`
- `safe_div()`
- `status_to_failure()`
- `parse_parent_indices()`
- `parse_task_index()`
- `build_task_dag_features()`
- `build_instance_label_and_context()`
- `build_machine_usage_timeline()`
- `causal_attach_machine_pressure()`
- `add_temporal_features()`
- `minmax_series()`
- `col_or_zero()`
- `add_complexity_score()`
- `drop_leakage_columns()`
- `build_dataset()`
- `get_feature_groups()`
- `make_preprocessor()`
- `make_model()`
- `predict_scores()`
- `tune_threshold()`
- `metrics_dict()`
- `sample_dataset()`
- `balance_training_frame()`
- `sample_stratified()`
- `split_train_val_test_frame()`
- `split_data()`
- `split_train_val_test()`
- `train_single_model()`
- `train_hqdnet()`
- `plot_metric_bars()`
- `plot_lines()`
- `plot_bar()`
- ... plus 13 more functions.

<a id="v6-final-leakage-safe-hard-failure-hqd-net"></a>

### <span style="color:#ef4444">V6 — Final Leakage-Safe Hard-Failure HQD-Net</span>

**File:** `hqd_net_job_failure_v6_final.py`  
**Lines in script:** `1051`  
**Classes:** `QuantumInspiredStructuralEncoder, HQDNetResult`

| Dimension | Description |
|---|---|
| Architecture role | Final Leakage-Safe Hard-Failure HQD-Net. |
| Best use | Stable paper-grade baseline. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --sampling-mode, --train-negative-ratio, --history-window, --hard-tau-quantile, --uncertainty-quantile, --threshold-metric, --run-ablations, --save-processed` |

#### Architecture
1. Job-level pre-execution feature builder.
2. Temporal train/validation/test split.
3. Balanced training only, natural validation/test.
4. Strong classical baselines.
5. Graph/quantum-inspired hard-sample specialist.
6. Uncertainty + complexity routing.
7. Model bundles, PR curves, feature importance, README.

#### Limitations
1. Proposed system is strong but still less explicitly framed around LGBM/XGB fusion than V7.
2. Complexity requires enough RAM for 50k/100k runs.
3. Optional dependencies can affect baseline availability.

#### What the next version improves
1. V7 uses LGBM backbone, XGB auxiliary learner, validation-tuned fusion weights.
2. V7-Fixed is recommended final.

#### Key functions/classes detected
- `ensure_dirs()`
- `log()`
- `safe_auc()`
- `memory_mb()`
- `read_csv_robust()`
- `to_num()`
- `status_to_failure()`
- `parse_task_graph_stats()`
- `add_temporal_history()`
- `build_dataset()`
- `feature_groups()`
- `select_sample_temporal()`
- `temporal_split()`
- `balance_training()`
- `make_preprocessor()`
- `make_models()`
- `get_scores()`
- `best_threshold_by_metric()`
- `evaluate()`
- `entropy_binary()`
- `train_hqdnet()`
- `train_eval_single_sample()`
- `save_model_bundle()`
- `plot_main_metrics()`
- `plot_pr_curves()`
- `run_feature_block_ablation()`
- `run_routing_ablation()`
- `run_noise_ablation()`
- `save_feature_importance()`
- `write_readme()`
- `main()`
- `__init__()`
- `fit()`
- `transform()`
- `fit_transform()`
- ... plus 1 more functions.

<a id="v6notk-v6-variant-without-threshold-k-tuning"></a>

### <span style="color:#ec4899">V6-NoTK — V6 variant without threshold-k tuning</span>

**File:** `hqd_net_job_failure_v6_final_notk.py`  
**Lines in script:** `1056`  
**Classes:** `QuantumInspiredStructuralEncoder, HQDNetResult`

| Dimension | Description |
|---|---|
| Architecture role | V6 variant without threshold-k tuning. |
| Best use | Sensitivity/checkpoint variant. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --sampling-mode, --train-negative-ratio, --history-window, --hard-tau-quantile, --uncertainty-quantile, --threshold-metric, --run-ablations, --save-processed` |

#### Architecture
1. Same broad structure as V6.
2. Variant intended for threshold/k-related comparison.
3. Keeps outputs and core training similar to V6.

#### Limitations
1. Not intended as the final result-producing file.
2. Mainly useful for sensitivity analysis.
3. Can confuse readers if reported as the proposed system.

#### What the next version improves
1. Use V6 for stable final baseline or V7-Fixed for final proposed system.

#### Key functions/classes detected
- `ensure_dirs()`
- `log()`
- `safe_auc()`
- `memory_mb()`
- `read_csv_robust()`
- `to_num()`
- `status_to_failure()`
- `parse_task_graph_stats()`
- `add_temporal_history()`
- `build_dataset()`
- `feature_groups()`
- `select_sample_temporal()`
- `temporal_split()`
- `balance_training()`
- `make_preprocessor()`
- `make_models()`
- `get_scores()`
- `best_threshold_by_metric()`
- `evaluate()`
- `entropy_binary()`
- `train_hqdnet()`
- `train_eval_single_sample()`
- `save_model_bundle()`
- `plot_main_metrics()`
- `plot_pr_curves()`
- `run_feature_block_ablation()`
- `run_routing_ablation()`
- `run_noise_ablation()`
- `save_feature_importance()`
- `write_readme()`
- `main()`
- `__init__()`
- `fit()`
- `transform()`
- `fit_transform()`
- ... plus 1 more functions.

<a id="v7-proposed-hybrid-system"></a>

### <span style="color:#22c55e">V7 — Proposed Hybrid System</span>

**File:** `hqd_net_job_failure_v7_proposed_system.py`  
**Lines in script:** `988`  
**Classes:** `QuantumInspiredStructuralEncoder`

| Dimension | Description |
|---|---|
| Architecture role | Proposed Hybrid System. |
| Best use | Paper proposed-system draft. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --sampling-mode, --run-ablations` |

#### Architecture
1. Leakage-safe feature engineering.
2. Grouped temporal evaluation.
3. Train-only balancing.
4. LR, RF, HGB, XGB, LGBM baselines.
5. LGBM deployment backbone.
6. XGB auxiliary learner.
7. Graph/quantum-inspired structural specialist.
8. Hard-example routing.
9. Validation-tuned fusion weights and decision threshold.
10. CSV outputs, figures, README.

#### Limitations
1. Initial proposed-system script may need small fixes.
2. Most complex script, so environment must be consistent.
3. Requires clear reporting to avoid confusing backbone vs specialist.

#### What the next version improves
1. V7-Fixed provides corrected final implementation.

#### Key functions/classes detected
- `ensure_dirs()`
- `read_csv_robust()`
- `to_numeric()`
- `parse_task_indices()`
- `task_template()`
- `build_dag_features()`
- `build_instance_features()`
- `build_machine_pressure()`
- `add_temporal_features()`
- `robust_minmax()`
- `add_complexity_features()`
- `remove_leakage_features()`
- `build_dataset()`
- `build_feature_groups()`
- `sample_by_time()`
- `grouped_temporal_split()`
- `balance_train()`
- `make_preprocessor()`
- `make_model()`
- `fit_pipeline()`
- `predict_proba_safe()`
- `best_threshold()`
- `eval_metrics()`
- `make_structural_features()`
- `fit_structural_specialist()`
- `entropy_binary()`
- `tune_hqd_v7()`
- `predict_hqd_v7()`
- `train_and_eval_sample()`
- `run_feature_ablation()`
- `run_noise_ablation()`
- `run_routing_ablation()`
- `plot_main_metrics()`
- `write_readme()`
- `main()`
- ... plus 3 more functions.

<a id="v7fixed-fixed-proposed-hybrid-system"></a>

### <span style="color:#14b8a6">V7-Fixed — Fixed Proposed Hybrid System</span>

**File:** `hqd_net_job_failure_v7_proposed_system_fixed.py`  
**Lines in script:** `993`  
**Classes:** `QuantumInspiredStructuralEncoder`

| Dimension | Description |
|---|---|
| Architecture role | Fixed Proposed Hybrid System. |
| Best use | Final Q1-style experiments. |
| CLI arguments detected | `--data-dir, --out-dir, --sample-sizes, --max-batch-task-rows, --max-batch-instance-rows, --max-machine-usage-rows, --use-machine-pressure, --sampling-mode, --run-ablations` |

#### Architecture
1. Final corrected proposed-system implementation.
2. Same architecture as V7 with fixes.
3. Best candidate for journal experiments.
4. Recommended for final figures/tables.

#### Limitations
1. Heaviest script.
2. Run small sample first before 100k.
3. Needs full dataset paths and optional dependency handling.

#### What the next version improves
1. This is the final branch; future improvements should be new V8 or journal extension branch.

#### Key functions/classes detected
- `ensure_dirs()`
- `read_csv_robust()`
- `to_numeric()`
- `parse_task_indices()`
- `task_template()`
- `build_dag_features()`
- `build_instance_features()`
- `build_machine_pressure()`
- `add_temporal_features()`
- `robust_minmax()`
- `add_complexity_features()`
- `remove_leakage_features()`
- `build_dataset()`
- `build_feature_groups()`
- `sample_by_time()`
- `grouped_temporal_split()`
- `balance_train()`
- `make_preprocessor()`
- `make_model()`
- `fit_pipeline()`
- `predict_proba_safe()`
- `best_threshold()`
- `eval_metrics()`
- `make_structural_features()`
- `fit_structural_specialist()`
- `entropy_binary()`
- `tune_hqd_v7()`
- `predict_hqd_v7()`
- `train_and_eval_sample()`
- `run_feature_ablation()`
- `run_noise_ablation()`
- `run_routing_ablation()`
- `plot_main_metrics()`
- `write_readme()`
- `main()`
- ... plus 3 more functions.


## 📊 Ablation and Result Files You Should Expect

| Output | Meaning | Used in paper section |
|---|---|---|
| `main_metrics.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `ablation_feature_blocks.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `ablation_routing.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `ablation_noise.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `feature_importance.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `processed_dataset.csv` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `model_bundle.joblib` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `main_metrics.png` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `scalability.png` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `latency.png` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `precision_recall_curves.png` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |
| `README_EXPERIMENT.md` | Generated artifact for metrics, figures, models, or experiment documentation. | Results / ablation / reproducibility. |

## ✅ Recommended Paper-Grade Workflow

| Step | Action | Detail |
|---|---|---|
| Step 1 | Inspect CSVs | Run `inspect_alibaba_csvs.py` and confirm row counts and schemas. |
| Step 2 | Smoke test | Run V7-Fixed on 5k samples without ablations. |
| Step 3 | Main sweep | Run V7-Fixed on 5k, 10k, 20k, 25k, 50k, and 100k. |
| Step 4 | Ablations | Enable `--run-ablations` and collect feature, noise, and routing ablations. |
| Step 5 | Baseline comparison | Use LR, RF, HGB, XGB, LGBM outputs as classical baselines. |
| Step 6 | Proposed model table | Report HQD-Net-v7/Fix as the proposed system. |
| Step 7 | Discussion | Explain leakage safety, temporal split, train-only balancing, and hard-route specialist. |
| Step 8 | Archive | Save all CLI commands, random seed, and output directories. |

## 🔍 Dataset Inspection Tool

`inspect_alibaba_csvs.py` is a small utility for checking Alibaba CSV files before running expensive experiments.

```bash
python inspect_alibaba_csvs.py
```

If paths are hardcoded inside the utility, edit these two variables first:

```python
DATA_DIR = Path(r"D:\other\ALIBABAQUATUM\Dataset")
OUT_MD = Path(r"D:\other\ALIBABAQUATUM\Dataset\README_DATASET_INSPECTION.md")
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| CSV header mismatch | Alibaba files can be headerless. Use schema hints or inspect first. |
| Very high accuracy but weak failure recall | Dataset is imbalanced. Use train-only balancing and report recall/F1/AP. |
| Results look too good | Check leakage. Remove post-execution columns and label-derived aggregates. |
| Memory crash at 100k | Reduce max row caps, disable optional pressure, or reuse engineered CSV. |
| XGBoost/LightGBM import error | Install optional dependency or let script skip unavailable model if supported. |
| Slow machine pressure join | Use `--max-machine-usage-rows` and causal merge-asof only. |
| Different results across runs | Set seed 42 and keep sample sizes / sampling mode identical. |
| Validation/test distribution changes | Train-only balancing should not be applied to validation/test. |
| No positive failures in sample | Increase raw row cap or use balanced sampling mode. |
| Figures missing | Check output directory permissions and matplotlib backend. |

## 🎨 Colorful Architecture Cards

<div style="border:2px solid #3b82f6; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#3b82f6;">V1: Complexity-Aware Hybrid Quantum-Classical Classifier</h3>
<p><b>Script:</b> <code>hqd_net_job_failure.py</code></p>
<p><b>Detected size:</b> 851 lines. <b>Core classes:</b> QuantumStructuralFeatureMap, HQDNetConfig, HQDNet.</p>
<ul>
<li>CSV ingestion from batch task data</li>
<li>Feature normalization and simple complexity scoring</li>
<li>Classical model suite including tree and linear models</li>
<li>QuantumStructuralFeatureMap / QSSE style routed specialist</li>
</ul>
</div>
<div style="border:2px solid #06b6d4; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#06b6d4;">V2: Hierarchical Quantum-Dependency Network</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v2.py</code></p>
<p><b>Detected size:</b> 943 lines. <b>Core classes:</b> QuantumStructuralEncoder, HQDNet.</p>
<ul>
<li>Richer Alibaba v2018 ingestion</li>
<li>Task DAG feature builder</li>
<li>Instance feature builder</li>
<li>Machine/container usage hooks</li>
</ul>
</div>
<div style="border:2px solid #10b981; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#10b981;">V3: Leakage-Safe HQD-Net</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v3_leakage_safe.py</code></p>
<p><b>Detected size:</b> 790 lines. <b>Core classes:</b> QuantumStructuralEncoder, HQDNet.</p>
<ul>
<li>Leakage-safe feature removal</li>
<li>Train-only sklearn preprocessing pipeline</li>
<li>Threshold tuning using training data</li>
<li>Causal previous-time machine pressure approximation</li>
</ul>
</div>
<div style="border:2px solid #f59e0b; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#f59e0b;">V4: Balanced Hard-Routing HQD-Net</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v4_balanced_hardrouting.py</code></p>
<p><b>Detected size:</b> 872 lines. <b>Core classes:</b> QuantumStructuralEncoder, HQDNet.</p>
<ul>
<li>Balanced leakage-safe HQD-Net</li>
<li>Hard-example quantum-dependency routing</li>
<li>Failure-focused threshold tuning</li>
<li>Feature block ablations and routing diagnostics</li>
</ul>
</div>
<div style="border:2px solid #8b5cf6; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#8b5cf6;">V5: Grouped Temporal Balanced HQD-Net</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v5_grouped_temporal_balanced.py</code></p>
<p><b>Detected size:</b> 913 lines. <b>Core classes:</b> QuantumStructuralEncoder, HQDNet.</p>
<ul>
<li>Grouped temporal leakage-safe split</li>
<li>Train-only balancing</li>
<li>Hard-example specialist</li>
<li>Causal machine pressure option</li>
</ul>
</div>
<div style="border:2px solid #ef4444; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#ef4444;">V6: Final Leakage-Safe Hard-Failure HQD-Net</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v6_final.py</code></p>
<p><b>Detected size:</b> 1051 lines. <b>Core classes:</b> QuantumInspiredStructuralEncoder, HQDNetResult.</p>
<ul>
<li>Job-level pre-execution feature builder</li>
<li>Temporal train/validation/test split</li>
<li>Balanced training only, natural validation/test</li>
<li>Strong classical baselines</li>
</ul>
</div>
<div style="border:2px solid #ec4899; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#ec4899;">V6-NoTK: V6 variant without threshold-k tuning</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v6_final_notk.py</code></p>
<p><b>Detected size:</b> 1056 lines. <b>Core classes:</b> QuantumInspiredStructuralEncoder, HQDNetResult.</p>
<ul>
<li>Same broad structure as V6</li>
<li>Variant intended for threshold/k-related comparison</li>
<li>Keeps outputs and core training similar to V6</li>
</ul>
</div>
<div style="border:2px solid #22c55e; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#22c55e;">V7: Proposed Hybrid System</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v7_proposed_system.py</code></p>
<p><b>Detected size:</b> 988 lines. <b>Core classes:</b> QuantumInspiredStructuralEncoder.</p>
<ul>
<li>Leakage-safe feature engineering</li>
<li>Grouped temporal evaluation</li>
<li>Train-only balancing</li>
<li>LR, RF, HGB, XGB, LGBM baselines</li>
</ul>
</div>
<div style="border:2px solid #14b8a6; border-radius:10px; padding:12px; margin:10px 0;">
<h3 style="color:#14b8a6;">V7-Fixed: Fixed Proposed Hybrid System</h3>
<p><b>Script:</b> <code>hqd_net_job_failure_v7_proposed_system_fixed.py</code></p>
<p><b>Detected size:</b> 993 lines. <b>Core classes:</b> QuantumInspiredStructuralEncoder.</p>
<ul>
<li>Final corrected proposed-system implementation</li>
<li>Same architecture as V7 with fixes</li>
<li>Best candidate for journal experiments</li>
<li>Recommended for final figures/tables</li>
</ul>
</div>

## 📚 Version-by-Version Operational Checklists

### V1 operational checklist for `hqd_net_job_failure.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V2 operational checklist for `hqd_net_job_failure_v2.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V3 operational checklist for `hqd_net_job_failure_v3_leakage_safe.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V4 operational checklist for `hqd_net_job_failure_v4_balanced_hardrouting.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V5 operational checklist for `hqd_net_job_failure_v5_grouped_temporal_balanced.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V6 operational checklist for `hqd_net_job_failure_v6_final.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V6-NoTK operational checklist for `hqd_net_job_failure_v6_final_notk.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V7 operational checklist for `hqd_net_job_failure_v7_proposed_system.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.

### V7-Fixed operational checklist for `hqd_net_job_failure_v7_proposed_system_fixed.py`

- [ ] 01. Confirm the dataset directory contains `batch_task.csv`.
- [ ] 02. Confirm `batch_instance.csv` exists if the version uses instance features.
- [ ] 03. Confirm optional `machine_usage.csv` exists before enabling machine pressure.
- [ ] 04. Run with a small 5k sample before larger sweeps.
- [ ] 05. Write outputs to a fresh result folder.
- [ ] 06. Keep `random_state` or seed fixed at 42 where available.
- [ ] 07. Check class distribution before reading accuracy.
- [ ] 08. Report F1, recall, ROC-AUC, AP, and fit time together.
- [ ] 09. Inspect generated CSV before creating paper tables.
- [ ] 10. Keep validation/test natural for leakage-safe versions.
- [ ] 11. Do not use post-execution features as predictors.
- [ ] 12. Save exact CLI command in experiment notes.
- [ ] 13. Back up result folder after each successful run.
- [ ] 14. Use the same sample sizes across baseline and proposed models.
- [ ] 15. For figures, confirm labels are readable in IEEE two-column format.
- [ ] 16. For ablations, run one controlled change at a time.
- [ ] 17. For paper discussion, explain what this version fixed.
- [ ] 18. For failure prediction, prioritize minority-class recall and AP.
- [ ] 19. For routing ablation, compare with specialist disabled.
- [ ] 20. For feature ablation, remove one block at a time.


## 📑 Paper Table Templates

### Template Table 1: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 2: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 3: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 4: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 5: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 6: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 7: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 8: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 9: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 10: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 11: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 12: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 13: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 14: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 15: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 16: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 17: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 18: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 19: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 20: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 21: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 22: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 23: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 24: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 25: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 26: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 27: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 28: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 29: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |

### Template Table 30: Experiment reporting block
| Field | Value to fill | Notes |
|---|---|---|
| Dataset | Alibaba Cluster Trace v2018 | State row caps and tables used. |
| Sample size | 5k / 10k / 20k / 25k / 50k / 100k | Keep consistent across methods. |
| Split | Grouped temporal train/val/test | Avoid random leakage. |
| Balancing | Train-only | Validation/test remain natural. |
| Baselines | LR, RF, HGB, XGB, LGBM | Use available package set. |
| Proposed | HQD-Net-v7-Fixed | Include routing and fusion details. |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Fit time | Avoid accuracy-only claims. |
| Ablation | Feature / routing / noise | One controlled change per row. |


## 🧩 Appendix: CLI Argument Reference

### `hqd_net_job_failure.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-raw-rows` | CLI option detected in script. | see script default |
| `--iid` | CLI option detected in script. | see script default |
| `--run-ablations` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v2.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--max-container-usage-rows` | CLI option detected in script. | see script default |
| `--use-container-usage` | CLI option detected in script. | see script default |
| `--run-ablations` | CLI option detected in script. | flag |
| `--reuse-engineered` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v3_leakage_safe.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--run-ablations` | CLI option detected in script. | flag |
| `--reuse-engineered` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v4_balanced_hardrouting.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--run-ablations` | CLI option detected in script. | flag |
| `--reuse-engineered` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |

### `hqd_net_job_failure_v5_grouped_temporal_balanced.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--run-ablations` | CLI option detected in script. | flag |
| `--reuse-engineered` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |

### `hqd_net_job_failure_v6_final.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |
| `--train-negative-ratio` | CLI option detected in script. | see script default |
| `--history-window` | CLI option detected in script. | see script default |
| `--hard-tau-quantile` | CLI option detected in script. | see script default |
| `--uncertainty-quantile` | CLI option detected in script. | see script default |
| `--threshold-metric` | CLI option detected in script. | see script default |
| `--run-ablations` | CLI option detected in script. | flag |
| `--save-processed` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v6_final_notk.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |
| `--train-negative-ratio` | CLI option detected in script. | see script default |
| `--history-window` | CLI option detected in script. | see script default |
| `--hard-tau-quantile` | CLI option detected in script. | see script default |
| `--uncertainty-quantile` | CLI option detected in script. | see script default |
| `--threshold-metric` | CLI option detected in script. | see script default |
| `--run-ablations` | CLI option detected in script. | flag |
| `--save-processed` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v7_proposed_system.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |
| `--run-ablations` | CLI option detected in script. | flag |

### `hqd_net_job_failure_v7_proposed_system_fixed.py` arguments
| Argument | Meaning | Typical value |
|---|---|---|
| `--data-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Dataset` |
| `--out-dir` | CLI option detected in script. | `D:\other\ALIBABAQUATUM\Results_HQDNet` |
| `--sample-sizes` | CLI option detected in script. | `5000 10000 20000 25000 50000 100000` |
| `--max-batch-task-rows` | CLI option detected in script. | `5000000` |
| `--max-batch-instance-rows` | CLI option detected in script. | `5000000` |
| `--max-machine-usage-rows` | CLI option detected in script. | `5000000` |
| `--use-machine-pressure` | CLI option detected in script. | flag |
| `--sampling-mode` | CLI option detected in script. | `train_balanced` |
| `--run-ablations` | CLI option detected in script. | flag |


## 🧪 Extended Notes for Reviewers and Users

1. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
2. Leakage-safe evaluation removes features that become available only after job completion.
3. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
4. Train-only balancing prevents validation and test metrics from being artificially stabilized.
5. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
6. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
7. Machine pressure features must be causal, using only historical or previous-time information.
8. DAG features should come from task-name structure and not from future execution outcomes.
9. Feature block ablations should remove full groups, not single correlated columns only.
10. Noise ablations test robustness and should not be mixed with model-selection tuning.
11. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
12. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
13. Use one output directory per run to avoid overwriting tables and plots.
14. Keep raw logs because reviewers may ask how many rows were used after cleaning.
15. When reporting improvements, compare against the strongest baseline, not only against V1.
16. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
17. Leakage-safe evaluation removes features that become available only after job completion.
18. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
19. Train-only balancing prevents validation and test metrics from being artificially stabilized.
20. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
21. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
22. Machine pressure features must be causal, using only historical or previous-time information.
23. DAG features should come from task-name structure and not from future execution outcomes.
24. Feature block ablations should remove full groups, not single correlated columns only.
25. Noise ablations test robustness and should not be mixed with model-selection tuning.
26. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
27. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
28. Use one output directory per run to avoid overwriting tables and plots.
29. Keep raw logs because reviewers may ask how many rows were used after cleaning.
30. When reporting improvements, compare against the strongest baseline, not only against V1.
31. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
32. Leakage-safe evaluation removes features that become available only after job completion.
33. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
34. Train-only balancing prevents validation and test metrics from being artificially stabilized.
35. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
36. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
37. Machine pressure features must be causal, using only historical or previous-time information.
38. DAG features should come from task-name structure and not from future execution outcomes.
39. Feature block ablations should remove full groups, not single correlated columns only.
40. Noise ablations test robustness and should not be mixed with model-selection tuning.
41. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
42. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
43. Use one output directory per run to avoid overwriting tables and plots.
44. Keep raw logs because reviewers may ask how many rows were used after cleaning.
45. When reporting improvements, compare against the strongest baseline, not only against V1.
46. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
47. Leakage-safe evaluation removes features that become available only after job completion.
48. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
49. Train-only balancing prevents validation and test metrics from being artificially stabilized.
50. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
51. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
52. Machine pressure features must be causal, using only historical or previous-time information.
53. DAG features should come from task-name structure and not from future execution outcomes.
54. Feature block ablations should remove full groups, not single correlated columns only.
55. Noise ablations test robustness and should not be mixed with model-selection tuning.
56. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
57. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
58. Use one output directory per run to avoid overwriting tables and plots.
59. Keep raw logs because reviewers may ask how many rows were used after cleaning.
60. When reporting improvements, compare against the strongest baseline, not only against V1.
61. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
62. Leakage-safe evaluation removes features that become available only after job completion.
63. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
64. Train-only balancing prevents validation and test metrics from being artificially stabilized.
65. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
66. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
67. Machine pressure features must be causal, using only historical or previous-time information.
68. DAG features should come from task-name structure and not from future execution outcomes.
69. Feature block ablations should remove full groups, not single correlated columns only.
70. Noise ablations test robustness and should not be mixed with model-selection tuning.
71. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
72. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
73. Use one output directory per run to avoid overwriting tables and plots.
74. Keep raw logs because reviewers may ask how many rows were used after cleaning.
75. When reporting improvements, compare against the strongest baseline, not only against V1.
76. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
77. Leakage-safe evaluation removes features that become available only after job completion.
78. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
79. Train-only balancing prevents validation and test metrics from being artificially stabilized.
80. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
81. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
82. Machine pressure features must be causal, using only historical or previous-time information.
83. DAG features should come from task-name structure and not from future execution outcomes.
84. Feature block ablations should remove full groups, not single correlated columns only.
85. Noise ablations test robustness and should not be mixed with model-selection tuning.
86. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
87. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
88. Use one output directory per run to avoid overwriting tables and plots.
89. Keep raw logs because reviewers may ask how many rows were used after cleaning.
90. When reporting improvements, compare against the strongest baseline, not only against V1.
91. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
92. Leakage-safe evaluation removes features that become available only after job completion.
93. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
94. Train-only balancing prevents validation and test metrics from being artificially stabilized.
95. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
96. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
97. Machine pressure features must be causal, using only historical or previous-time information.
98. DAG features should come from task-name structure and not from future execution outcomes.
99. Feature block ablations should remove full groups, not single correlated columns only.
100. Noise ablations test robustness and should not be mixed with model-selection tuning.
101. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
102. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
103. Use one output directory per run to avoid overwriting tables and plots.
104. Keep raw logs because reviewers may ask how many rows were used after cleaning.
105. When reporting improvements, compare against the strongest baseline, not only against V1.
106. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
107. Leakage-safe evaluation removes features that become available only after job completion.
108. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
109. Train-only balancing prevents validation and test metrics from being artificially stabilized.
110. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
111. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
112. Machine pressure features must be causal, using only historical or previous-time information.
113. DAG features should come from task-name structure and not from future execution outcomes.
114. Feature block ablations should remove full groups, not single correlated columns only.
115. Noise ablations test robustness and should not be mixed with model-selection tuning.
116. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
117. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
118. Use one output directory per run to avoid overwriting tables and plots.
119. Keep raw logs because reviewers may ask how many rows were used after cleaning.
120. When reporting improvements, compare against the strongest baseline, not only against V1.
121. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
122. Leakage-safe evaluation removes features that become available only after job completion.
123. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
124. Train-only balancing prevents validation and test metrics from being artificially stabilized.
125. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
126. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
127. Machine pressure features must be causal, using only historical or previous-time information.
128. DAG features should come from task-name structure and not from future execution outcomes.
129. Feature block ablations should remove full groups, not single correlated columns only.
130. Noise ablations test robustness and should not be mixed with model-selection tuning.
131. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
132. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
133. Use one output directory per run to avoid overwriting tables and plots.
134. Keep raw logs because reviewers may ask how many rows were used after cleaning.
135. When reporting improvements, compare against the strongest baseline, not only against V1.
136. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
137. Leakage-safe evaluation removes features that become available only after job completion.
138. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
139. Train-only balancing prevents validation and test metrics from being artificially stabilized.
140. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
141. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
142. Machine pressure features must be causal, using only historical or previous-time information.
143. DAG features should come from task-name structure and not from future execution outcomes.
144. Feature block ablations should remove full groups, not single correlated columns only.
145. Noise ablations test robustness and should not be mixed with model-selection tuning.
146. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
147. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
148. Use one output directory per run to avoid overwriting tables and plots.
149. Keep raw logs because reviewers may ask how many rows were used after cleaning.
150. When reporting improvements, compare against the strongest baseline, not only against V1.
151. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
152. Leakage-safe evaluation removes features that become available only after job completion.
153. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
154. Train-only balancing prevents validation and test metrics from being artificially stabilized.
155. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
156. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
157. Machine pressure features must be causal, using only historical or previous-time information.
158. DAG features should come from task-name structure and not from future execution outcomes.
159. Feature block ablations should remove full groups, not single correlated columns only.
160. Noise ablations test robustness and should not be mixed with model-selection tuning.
161. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
162. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
163. Use one output directory per run to avoid overwriting tables and plots.
164. Keep raw logs because reviewers may ask how many rows were used after cleaning.
165. When reporting improvements, compare against the strongest baseline, not only against V1.
166. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
167. Leakage-safe evaluation removes features that become available only after job completion.
168. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
169. Train-only balancing prevents validation and test metrics from being artificially stabilized.
170. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
171. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
172. Machine pressure features must be causal, using only historical or previous-time information.
173. DAG features should come from task-name structure and not from future execution outcomes.
174. Feature block ablations should remove full groups, not single correlated columns only.
175. Noise ablations test robustness and should not be mixed with model-selection tuning.
176. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
177. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
178. Use one output directory per run to avoid overwriting tables and plots.
179. Keep raw logs because reviewers may ask how many rows were used after cleaning.
180. When reporting improvements, compare against the strongest baseline, not only against V1.
181. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
182. Leakage-safe evaluation removes features that become available only after job completion.
183. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
184. Train-only balancing prevents validation and test metrics from being artificially stabilized.
185. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.
186. LGBM and XGB can be optional dependencies; document package versions in the final artifact.
187. Machine pressure features must be causal, using only historical or previous-time information.
188. DAG features should come from task-name structure and not from future execution outcomes.
189. Feature block ablations should remove full groups, not single correlated columns only.
190. Noise ablations test robustness and should not be mixed with model-selection tuning.
191. Routing ablations test whether uncertainty and complexity actually improve hard-case performance.
192. Use fixed sample sizes in all figures to avoid unfair runtime comparisons.
193. Use one output directory per run to avoid overwriting tables and plots.
194. Keep raw logs because reviewers may ask how many rows were used after cleaning.
195. When reporting improvements, compare against the strongest baseline, not only against V1.
196. Accuracy alone can be misleading under extreme class imbalance; always include recall, F1, ROC-AUC, and AP.
197. Leakage-safe evaluation removes features that become available only after job completion.
198. Grouped temporal splitting better approximates deployment because future jobs should not influence training features.
199. Train-only balancing prevents validation and test metrics from being artificially stabilized.
200. Hard-example routing should be treated as a specialist branch, not as a standalone replacement for strong classical backbones.

## 📝 Changelog Style Version History

### V1 changelog
- Added/kept script: `hqd_net_job_failure.py`.
- Architecture: CSV ingestion from batch task data.
- Architecture: Feature normalization and simple complexity scoring.
- Architecture: Classical model suite including tree and linear models.
- Architecture: QuantumStructuralFeatureMap / QSSE style routed specialist.
- Architecture: Threshold/complexity ablations and plots.
- Known limitation: Early prototype with limited dataset joins.
- Known limitation: Evaluation can be optimistic relative to temporal deployment.
- Known limitation: Limited machine pressure and instance context.
- Known limitation: Less strict leakage audit than V3 onward.
- Improvement path: V2 adds richer joins and clearer modular outputs.
- Improvement path: V3 removes post-execution leakage.
- Improvement path: V4/V5 improve imbalance and routing design.

### V2 changelog
- Added/kept script: `hqd_net_job_failure_v2.py`.
- Architecture: Richer Alibaba v2018 ingestion.
- Architecture: Task DAG feature builder.
- Architecture: Instance feature builder.
- Architecture: Machine/container usage hooks.
- Architecture: Hierarchical quantum-dependency network class.
- Architecture: Model artifacts and publication-style figures.
- Known limitation: Some engineered context can still require leakage auditing.
- Known limitation: Temporal grouping not final.
- Known limitation: Balancing strategy not yet as defensible as later versions.
- Improvement path: V3 makes leakage safety explicit.
- Improvement path: V4 introduces balanced hard routing.
- Improvement path: V5 adds grouped temporal split.

### V3 changelog
- Added/kept script: `hqd_net_job_failure_v3_leakage_safe.py`.
- Architecture: Leakage-safe feature removal.
- Architecture: Train-only sklearn preprocessing pipeline.
- Architecture: Threshold tuning using training data.
- Architecture: Causal previous-time machine pressure approximation.
- Architecture: Ablation CSVs and figures.
- Known limitation: Hard-example routing still evolving.
- Known limitation: Grouped temporal split and train-only balancing need strengthening.
- Known limitation: Final proposed-system ensemble not yet introduced.
- Improvement path: V4 adds stronger hard routing and balancing.
- Improvement path: V5 adds grouped temporal train/val/test.
- Improvement path: V6 adds final artifacts and ablations.

### V4 changelog
- Added/kept script: `hqd_net_job_failure_v4_balanced_hardrouting.py`.
- Architecture: Balanced leakage-safe HQD-Net.
- Architecture: Hard-example quantum-dependency routing.
- Architecture: Failure-focused threshold tuning.
- Architecture: Feature block ablations and routing diagnostics.
- Architecture: Optional machine pressure.
- Known limitation: Grouped temporal split not yet the mature default.
- Known limitation: Final reporting is less complete than V6.
- Known limitation: May still need careful sampling-mode documentation.
- Improvement path: V5 formalizes grouped temporal balanced training.
- Improvement path: V6 adds final model bundles and robust outputs.

### V5 changelog
- Added/kept script: `hqd_net_job_failure_v5_grouped_temporal_balanced.py`.
- Architecture: Grouped temporal leakage-safe split.
- Architecture: Train-only balancing.
- Architecture: Hard-example specialist.
- Architecture: Causal machine pressure option.
- Architecture: Feature groups and ablation plots.
- Known limitation: Final paper narrative still less polished.
- Known limitation: Model saving/reporting less complete than V6/V7.
- Known limitation: Proposed-system fusion not final.
- Improvement path: V6 stabilizes final experiment package.
- Improvement path: V7 reframes as full proposed hybrid system.

### V6 changelog
- Added/kept script: `hqd_net_job_failure_v6_final.py`.
- Architecture: Job-level pre-execution feature builder.
- Architecture: Temporal train/validation/test split.
- Architecture: Balanced training only, natural validation/test.
- Architecture: Strong classical baselines.
- Architecture: Graph/quantum-inspired hard-sample specialist.
- Architecture: Uncertainty + complexity routing.
- Architecture: Model bundles, PR curves, feature importance, README.
- Known limitation: Proposed system is strong but still less explicitly framed around LGBM/XGB fusion than V7.
- Known limitation: Complexity requires enough RAM for 50k/100k runs.
- Known limitation: Optional dependencies can affect baseline availability.
- Improvement path: V7 uses LGBM backbone, XGB auxiliary learner, validation-tuned fusion weights.
- Improvement path: V7-Fixed is recommended final.

### V6-NoTK changelog
- Added/kept script: `hqd_net_job_failure_v6_final_notk.py`.
- Architecture: Same broad structure as V6.
- Architecture: Variant intended for threshold/k-related comparison.
- Architecture: Keeps outputs and core training similar to V6.
- Known limitation: Not intended as the final result-producing file.
- Known limitation: Mainly useful for sensitivity analysis.
- Known limitation: Can confuse readers if reported as the proposed system.
- Improvement path: Use V6 for stable final baseline or V7-Fixed for final proposed system.

### V7 changelog
- Added/kept script: `hqd_net_job_failure_v7_proposed_system.py`.
- Architecture: Leakage-safe feature engineering.
- Architecture: Grouped temporal evaluation.
- Architecture: Train-only balancing.
- Architecture: LR, RF, HGB, XGB, LGBM baselines.
- Architecture: LGBM deployment backbone.
- Architecture: XGB auxiliary learner.
- Architecture: Graph/quantum-inspired structural specialist.
- Architecture: Hard-example routing.
- Architecture: Validation-tuned fusion weights and decision threshold.
- Architecture: CSV outputs, figures, README.
- Known limitation: Initial proposed-system script may need small fixes.
- Known limitation: Most complex script, so environment must be consistent.
- Known limitation: Requires clear reporting to avoid confusing backbone vs specialist.
- Improvement path: V7-Fixed provides corrected final implementation.

### V7-Fixed changelog
- Added/kept script: `hqd_net_job_failure_v7_proposed_system_fixed.py`.
- Architecture: Final corrected proposed-system implementation.
- Architecture: Same architecture as V7 with fixes.
- Architecture: Best candidate for journal experiments.
- Architecture: Recommended for final figures/tables.
- Known limitation: Heaviest script.
- Known limitation: Run small sample first before 100k.
- Known limitation: Needs full dataset paths and optional dependency handling.
- Improvement path: This is the final branch; future improvements should be new V8 or journal extension branch.


## 🏁 Final Recommendation

Use `hqd_net_job_failure_v7_proposed_system_fixed.py` for the final experiment package. Use `hqd_net_job_failure_v6_final.py` as a stable backup and use V1 to V5 to explain research evolution. For a paper, report the final proposed system, the strongest baselines, and ablations that isolate leakage safety, grouped temporal splitting, train-only balancing, feature blocks, routing, and noise robustness.

---

<div align="center">

**Q-HQDNet / NEBULA-HQD — prepared for reproducible cloud job-failure prediction experiments.**

</div>

## ❓ Extended FAQ

### FAQ 1: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 2: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 3: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 4: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 5: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 6: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 7: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 8: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 9: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 10: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 11: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 12: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 13: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 14: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 15: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 16: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 17: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 18: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 19: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 20: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 21: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 22: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 23: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 24: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 25: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 26: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 27: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 28: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 29: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 30: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 31: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 32: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 33: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 34: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 35: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 36: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 37: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 38: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 39: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 40: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 41: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 42: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 43: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 44: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 45: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 46: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 47: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 48: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 49: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 50: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 51: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 52: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 53: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 54: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 55: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 56: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 57: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 58: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 59: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 60: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 61: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 62: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 63: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 64: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 65: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 66: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 67: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 68: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 69: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 70: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 71: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 72: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 73: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 74: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 75: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 76: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 77: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 78: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 79: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 80: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 81: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 82: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 83: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 84: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 85: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 86: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 87: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 88: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 89: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 90: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 91: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 92: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 93: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 94: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 95: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 96: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 97: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 98: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 99: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 100: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 101: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 102: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 103: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 104: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 105: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 106: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 107: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 108: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 109: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 110: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 111: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 112: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 113: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 114: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 115: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 116: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 117: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 118: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 119: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 120: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 121: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 122: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 123: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 124: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 125: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 126: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 127: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 128: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 129: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 130: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 131: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 132: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 133: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 134: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 135: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 136: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 137: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 138: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 139: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 140: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 141: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 142: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 143: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 144: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 145: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 146: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 147: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 148: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 149: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 150: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 151: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 152: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 153: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 154: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 155: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 156: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 157: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 158: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 159: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 160: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 161: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 162: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 163: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 164: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 165: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 166: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 167: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 168: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 169: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 170: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 171: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 172: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 173: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 174: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 175: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 176: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 177: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 178: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 179: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 180: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 181: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 182: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 183: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 184: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 185: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 186: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 187: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 188: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 189: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 190: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 191: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 192: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 193: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 194: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 195: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 196: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 197: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 198: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 199: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 200: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 201: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 202: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 203: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 204: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 205: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 206: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 207: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 208: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 209: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 210: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 211: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 212: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 213: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 214: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 215: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 216: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 217: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 218: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 219: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 220: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 221: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 222: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 223: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 224: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 225: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 226: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 227: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 228: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 229: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 230: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 231: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 232: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 233: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 234: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 235: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 236: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 237: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 238: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 239: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 240: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 241: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 242: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 243: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 244: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 245: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 246: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 247: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 248: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 249: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 250: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 251: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 252: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 253: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 254: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 255: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 256: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 257: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 258: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 259: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 260: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 261: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 262: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 263: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 264: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 265: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 266: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 267: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 268: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 269: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 270: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 271: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 272: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 273: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 274: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 275: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 276: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 277: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 278: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 279: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 280: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 281: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 282: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 283: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 284: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 285: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 286: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 287: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 288: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 289: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 290: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 291: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 292: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 293: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 294: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 295: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 296: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 297: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 298: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 299: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 300: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 301: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 302: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 303: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 304: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 305: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 306: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 307: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 308: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 309: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 310: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 311: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 312: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 313: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 314: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 315: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 316: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 317: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 318: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 319: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 320: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 321: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 322: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 323: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 324: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 325: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 326: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 327: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 328: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 329: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 330: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 331: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 332: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 333: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 334: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 335: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 336: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 337: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 338: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 339: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 340: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.

### FAQ 341: Which file should I run first?
Run `inspect_alibaba_csvs.py`, then run V7-Fixed on 5k samples.

### FAQ 342: Which version is final?
`hqd_net_job_failure_v7_proposed_system_fixed.py`.

### FAQ 343: Why keep older versions?
They document the research evolution and justify ablations.

### FAQ 344: Should validation/test be balanced?
No. Balance training only for leakage-safe deployment-style evaluation.

### FAQ 345: Why temporal split?
It prevents future workload information from leaking into training.

### FAQ 346: Why remove duration?
Duration can be post-execution depending on prediction timing; avoid it unless used only causally.

### FAQ 347: What is the specialist branch?
A graph/quantum-inspired structural encoder used mainly for hard examples.

### FAQ 348: What metrics matter most?
Recall, F1, ROC-AUC, AP, and fit/inference time.

### FAQ 349: What if LightGBM fails?
Install `lightgbm` or report the available baselines clearly.

### FAQ 350: What if memory fails?
Lower sample sizes or max row caps and reuse engineered data.
