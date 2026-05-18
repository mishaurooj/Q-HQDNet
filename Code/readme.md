
# NEBULA-JFP: Job Failure Prediction in Distributed Cloud Systems

**Framework:** NEBULA-JFP, Neural-Blend Unified Learning Architecture for Job Failure Prediction  
**Custom DNN:** NebulaNet-DNN  
**Quantum branch:** Q-Helix-SVM  
**Dataset:** Alibaba Cluster Trace v2018, `batch_task.csv`

This repository contains multiple implementation versions for job failure prediction using classical machine learning, lightweight deep learning, hybrid ensemble learning, and quantum-kernel learning. The code supports fast classical experiments and a separate quantum experiment for small balanced samples.

---

## 1. Problem Statement

Large-scale cloud clusters execute thousands to millions of batch jobs under dynamic scheduling, heterogeneous resource requests, and changing workload pressure. A job failure prediction model must identify likely failed or terminated jobs before execution completes so that schedulers can reduce wasted resources, improve reliability, and support proactive resubmission or migration.

The main difficulty is class imbalance. In `batch_task.csv`, failed or minority-status jobs may appear in very small proportions, for example about **1.5%** in a 1,000-row sample. A model can show high accuracy by predicting only the majority class, but such a model has poor recall and fails to detect the actual risky jobs. NEBULA-JFP addresses this by using balanced sampling, strong classical baselines, NebulaNet-DNN, ensemble fusion, and a separate quantum kernel branch for comparison with classical SVM.

---

## 2. Repository Layout

Use this structure for a clean submission or GitHub upload:

```text
NEBULA-JFP/
├── README.md
├── requirements.txt
├── data/
│   └── batch_task.csv
├── src/
│   ├── v1_rf_baseline.py
│   ├── v2_svm_baseline.py
│   ├── v3_nebulanet_dnn.py
│   ├── v4_rf_svm_dnn_ablation.py
│   ├── v5_stacking_nebula_jfp.py
│   ├── v6_enhanced_nebula_jfp.py
│   └── v7_qhelix_quantum.py
├── outputs/
│   ├── plots/
│   ├── trained_models/
│   └── ablation_results.xlsx
└── notebooks/
    ├── NEBULA_JFP_Classical.ipynb
    └── NEBULA_JFP_QHelix_Quantum.ipynb
```

If your files use different names, keep the same version meaning and rename the commands below accordingly.

---

## 3. Version History: V1 to V7

| Version | File | Architecture | Main Improvement | Limitation |
|---|---|---|---|---|
| V1 | `v1_rf_baseline.py` | Random Forest baseline | Establishes a strong non-neural classical baseline | Can hide minority-class failure due to imbalance; limited deep representation learning |
| V2 | `v2_svm_baseline.py` | Classical RBF-SVM | Adds a kernel baseline for fair comparison with quantum SVM | Slower than RF on larger samples; sensitive to `C`, `gamma`, and scaling |
| V3 | `v3_nebulanet_dnn.py` | NebulaNet-DNN | Adds lightweight deep nonlinear prediction | Needs tuning; may overfit small balanced samples |
| V4 | `v4_rf_svm_dnn_ablation.py` | RF + SVM + NebulaNet ablation | Compares each branch under the same split and metrics | Uses equal comparison only; no meta-learning fusion yet |
| V5 | `v5_stacking_nebula_jfp.py` | Stacking ensemble | Combines RF, SVM, and NebulaNet using logistic-regression meta learner | Higher training time; stacking can overfit if validation split is weak |
| V6 | `v6_enhanced_nebula_jfp.py` | Enhanced classical NEBULA-JFP | Adds feature engineering, balanced sampling, model saving, Excel logging, and plots | Classical-only; does not test quantum feature space |
| V7 | `v7_qhelix_quantum.py` | Q-Helix-SVM with Qiskit quantum kernel | Adds quantum-kernel branch using `PauliFeatureMap` or `ZZFeatureMap` and `PegasosQSVC` | Expensive for large samples; practical runs should use small balanced samples, e.g., 100 to 1,000 rows |

---

## 4. Architecture Summary

### 4.1 Classical NEBULA-JFP

The classical branch uses:

1. **Feature engineering** from `batch_task.csv`:
   - `duration = end_time - start_time`
   - `cpu_per_instance = plan_cpu / instance_num`
   - `mem_per_instance = plan_mem / instance_num`
   - `runtime_ratio = duration / (plan_cpu + 1)`
   - `log_duration = log1p(duration)`
   - `hour = start_time // 3600 % 24`

2. **Balanced real-sample selection**:
   - Downsample majority class.
   - Use equal number of `status=0` and `status=1` samples.
   - Avoid SMOTE for the quantum branch to keep all samples real.

3. **Models**:
   - Random Forest
   - Classical RBF-SVM
   - NebulaNet-DNN
   - Stacking ensemble

### 4.2 NebulaNet-DNN

NebulaNet-DNN is the custom deep branch of NEBULA-JFP. A recommended configuration is:

```python
hidden_layer_sizes = (512, 256, 128, 64)
activation = "relu"
solver = "adam"
alpha = 1e-5
batch_size = 128
learning_rate_init = 1e-4
max_iter = 200
early_stopping = True
n_iter_no_change = 20
```

Its purpose is to learn nonlinear relations between requested resources, execution timing, and failure status while remaining light enough for notebook-based experimentation.

### 4.3 Q-Helix-SVM

Q-Helix-SVM is the quantum branch of NEBULA-JFP. It uses:

- Qiskit `PauliFeatureMap` or `ZZFeatureMap`
- `FidelityQuantumKernel`
- `PegasosQSVC`
- Features scaled to `[0, pi]`
- Small balanced samples due to quantum kernel cost

Recommended practical setup:

```python
sample_size = 1000
train_size = 0.8
feature_range = (0, np.pi)
feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1, paulis=["X", "Z", "Y"], entanglement="full")
C = 200
num_steps = 5
```

---

## 5. Dataset Format

The expected `batch_task.csv` format has no header and uses the following columns:

```text
task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem
```

Status is converted to binary:

| Original status | Encoded label |
|---|---:|
| `Terminated`, `Failed`, `terminated`, `failed` | 0 |
| `Running`, `Waiting`, `running`, `waiting` | 1 |

You can modify this mapping if your paper defines failure differently. Keep the mapping identical across classical and quantum notebooks.

---

## 6. Environment Setup

### 6.1 Python Version

Recommended:

```bash
Python 3.10 or 3.11
```

Google Colab and Kaggle may use newer Python versions. If Qiskit runs successfully in your setup, keep that environment unchanged.

### 6.2 Classical Environment

Install:

```bash
pip install -U numpy pandas scikit-learn imbalanced-learn matplotlib openpyxl xlsxwriter joblib tqdm
```

### 6.3 Quantum Environment

Install:

```bash
pip install -U qiskit qiskit-aer qiskit-machine-learning
```

Test installation:

```python
from qiskit_aer import Aer
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC
print("Qiskit setup OK")
```

### 6.4 Example `requirements.txt`

```text
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
openpyxl
xlsxwriter
joblib
tqdm
qiskit
qiskit-aer
qiskit-machine-learning
```

---

## 7. How to Run Each Version

### V1: Random Forest Baseline

```bash
python src/v1_rf_baseline.py --data data/batch_task.csv --n-rows 10000 --out outputs/v1_rf
```

Use this to establish the first classical baseline.

### V2: Classical SVM Baseline

```bash
python src/v2_svm_baseline.py --data data/batch_task.csv --n-rows 10000 --out outputs/v2_svm
```

Use this as the direct classical comparison for Q-Helix-SVM.

### V3: NebulaNet-DNN

```bash
python src/v3_nebulanet_dnn.py --data data/batch_task.csv --n-rows 10000 --out outputs/v3_nebulanet
```

Use this to test the custom deep branch.

### V4: RF + SVM + NebulaNet Ablation

```bash
python src/v4_rf_svm_dnn_ablation.py --data data/batch_task.csv --sizes 10000 20000 30000 --out outputs/v4_ablation
```

Use this for ablation tables.

### V5: Stacking Ensemble

```bash
python src/v5_stacking_nebula_jfp.py --data data/batch_task.csv --sizes 10000 20000 30000 --out outputs/v5_stacking
```

Use this to evaluate learned fusion using a logistic-regression meta learner.

### V6: Enhanced Classical NEBULA-JFP

```bash
python src/v6_enhanced_nebula_jfp.py --data data/batch_task.csv --sizes 10000 20000 30000 40000 50000 --out outputs/v6_enhanced
```

Use this for the main classical journal results.

### V7: Q-Helix-SVM Quantum Branch

```bash
python src/v7_qhelix_quantum.py --data data/batch_task.csv --n-rows 1000 --out outputs/v7_qhelix
```

Use this for the quantum comparison. Keep the sample small because quantum-kernel models scale poorly with sample count.

---

## 8. Recommended Quantum Run Settings

For Colab with limited runtime:

| Setting | Recommended value |
|---|---:|
| Balanced total sample size | 100 to 1,000 |
| Train/test split | 80/20 |
| Feature scaling | `[0, pi]` |
| Feature map | `PauliFeatureMap` or `ZZFeatureMap` |
| Repetitions | 1 |
| Entanglement | `full` or `linear` |
| Pegasos steps | 5 to 20 |
| C | 100 to 300 |

Start with:

```python
sample_size = 1000
tau = 5
C = 200
```

If runtime is too high, reduce `sample_size` to 200 or 500.

---

## 9. Balanced Sampling Logic

Use real balanced sampling rather than SMOTE for Q-Helix-SVM:

```python
df0 = df[df["status"] == 0]
df1 = df[df["status"] == 1]
half = min(len(df0), len(df1), n_rows // 2)
df0 = df0.sample(n=half, random_state=42)
df1 = df1.sample(n=half, random_state=42)
df_balanced = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
```

This prevents inflated performance from synthetic minority samples and gives a cleaner quantum comparison.

---

## 10. Metrics Reported

Each version should report:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correctness of predicted positive class |
| Recall | Ability to detect positive class |
| F1 score | Balance between precision and recall |
| ROC-AUC | Ranking quality when probabilities are available |
| Average precision | Precision-recall quality for imbalanced data |
| Balanced accuracy | Robustness under class imbalance |
| MCC | Strong single-score metric for imbalanced binary classification |
| Confusion matrix | TP, TN, FP, FN breakdown |
| Fit time | Training cost |
| Prediction time | Inference cost |

For imbalanced raw data, do not rely on accuracy alone.

---

## 11. Known Limitations

### Dataset limitations

- Raw `batch_task.csv` may be extremely imbalanced.
- Some status labels may reflect scheduling state rather than final job failure.
- `batch_task.csv` alone does not include full machine pressure unless merged with other Alibaba tables.

### Classical model limitations

- RF can achieve high accuracy while still missing minority failures.
- SVM becomes slow as data grows.
- NebulaNet-DNN requires careful early stopping and scaling.
- Stacking increases runtime and may overfit if validation data is too small.

### Quantum model limitations

- Quantum kernel computation is expensive.
- Q-Helix-SVM is practical only on small balanced subsets in Colab.
- More qubits are required as feature count increases.
- Current runs use simulator-based quantum kernels, not real quantum hardware.
- PegasosQSVC supports binary classification only.

---

## 12. Suggested Paper Contribution Mapping

| Contribution | Supported by version |
|---|---|
| Strong classical baseline | V1, V2 |
| Custom deep model | V3 |
| Multi-model ablation | V4 |
| Hybrid learned ensemble | V5 |
| Main scalable classical framework | V6 |
| Quantum-kernel comparison | V7 |

Recommended journal framing:

> NEBULA-JFP first establishes robust classical baselines through RF, RBF-SVM, and NebulaNet-DNN. It then introduces stacking-based fusion for scalable classical prediction and evaluates Q-Helix-SVM as a quantum-kernel counterpart to classical SVM on small balanced samples. This design separates practical deployment performance from quantum feature-space analysis.

---

## 13. Reproducibility Checklist

Before reporting results, confirm:

- [ ] Same target mapping across all versions.
- [ ] Same feature columns across classical SVM and Q-Helix-SVM.
- [ ] Balanced sampling is reported clearly.
- [ ] Random seed is fixed, preferably `42`.
- [ ] Train/test split is stratified and reproducible.
- [ ] Accuracy is not reported alone.
- [ ] Confusion matrix is included.
- [ ] Quantum sample size is stated separately from classical sample size.
- [ ] Runtime and hardware are reported.
- [ ] Excel results and plots are saved.

---

## 14. Recommended Output Files

Each run should save:

```text
outputs/
├── ablation_results.xlsx
├── quantum_results.xlsx
├── plots/
│   ├── ROC.png
│   ├── PR.png
│   └── confusion_matrix.png
└── trained_models/
    ├── RF_model.joblib
    ├── SVM_model.joblib
    ├── NebulaNet_model.joblib
    └── Stacking_model.joblib
```

Quantum models may not always serialize cleanly across Qiskit versions. Save quantum metrics and circuit configuration even if model serialization is skipped.

---

## 15. Citation Note

When writing the paper, cite the Alibaba Cluster Trace v2018 dataset and Qiskit Machine Learning. Also cite your ICTC conference version as the preliminary study if allowed by the target journal.

---

## 16. Quick Start

Classical main run:

```bash
python src/v6_enhanced_nebula_jfp.py --data data/batch_task.csv --sizes 10000 20000 30000 40000 50000 --out outputs/v6_enhanced
```

Quantum quick run:

```bash
python src/v7_qhelix_quantum.py --data data/batch_task.csv --n-rows 1000 --out outputs/v7_qhelix
```

Then compare:

- Classical RBF-SVM vs Q-Helix-SVM
- NebulaNet-DNN vs RF
- Stacking ensemble vs individual models
- Runtime cost vs predictive gain

