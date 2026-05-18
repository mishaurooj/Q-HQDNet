
# HQD-Net v3 Leakage-Safe Results Summary

This run removes post-execution leakage features and reports failure-focused metrics.

## Removed Leakage Features

```
[
  "end_time",
  "task_end_max",
  "last_end_time",
  "task_duration_mean",
  "task_duration_max",
  "inst_duration_mean",
  "inst_duration_max",
  "inst_duration_std",
  "runtime_span",
  "job_span",
  "failed_instance_count",
  "instance_fail_rate",
  "task_fail_rate",
  "prev_runtime_mean_200",
  "status",
  "task_failure",
  "instance_failure",
  "cpu_avg_mean",
  "cpu_avg_max",
  "cpu_max_mean",
  "cpu_max_max",
  "mem_avg_mean",
  "mem_avg_max",
  "mem_max_mean",
  "mem_max_max",
  "cpu_spike_mean",
  "mem_spike_mean"
]
```

## Top Models by PR-AUC

| model             |   sample_size | route   |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   train_time_sec |   inference_time_sec |   latency_ms_per_sample |   roc_auc |   pr_auc |   tn |   fp |   fn |   tp |   quantum_route_train_ratio |   tau_threshold |
|:------------------|--------------:|:--------|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|-----------------:|---------------------:|------------------------:|----------:|---------:|-----:|-----:|-----:|-----:|----------------------------:|----------------:|
| XGB               |          5000 | iid     |    0.779533 |   0.967    |            0.770095 |    0.513514 | 0.558824 | 0.535211 | 0.518633 |         0.824012 |            0.0055773 |              0.0055773  |  0.850003 | 0.521357 |  948 |   18 |   15 |   19 |                   nan       |       nan       |
| HGB               |         10000 | iid     |    0.514734 |   0.973818 |            0.650669 |    0.777778 | 0.304348 | 0.4375   | 0.476726 |         4.38864  |            0.0391025 |              0.0284382  |  0.87583  | 0.48683  | 1325 |    4 |   32 |   14 |                   nan       |       nan       |
| HGB               |         20000 | iid     |    0.514734 |   0.973818 |            0.650669 |    0.777778 | 0.304348 | 0.4375   | 0.476726 |         4.49107  |            0.0435078 |              0.031642   |  0.87583  | 0.48683  | 1325 |    4 |   32 |   14 |                   nan       |       nan       |
| HGB               |         25000 | iid     |    0.514734 |   0.973818 |            0.650669 |    0.777778 | 0.304348 | 0.4375   | 0.476726 |         4.44042  |            0.038329  |              0.0278756  |  0.87583  | 0.48683  | 1325 |    4 |   32 |   14 |                   nan       |       nan       |
| XGB               |         10000 | iid     |    0.87393  |   0.971636 |            0.702007 |    0.612903 | 0.413043 | 0.493506 | 0.489393 |         0.712356 |            0.0051226 |              0.00372553 |  0.869802 | 0.4851   | 1317 |   12 |   27 |   19 |                   nan       |       nan       |
| XGB               |         20000 | iid     |    0.87393  |   0.971636 |            0.702007 |    0.612903 | 0.413043 | 0.493506 | 0.489393 |         0.68307  |            0.0051336 |              0.00373353 |  0.869802 | 0.4851   | 1317 |   12 |   27 |   19 |                   nan       |       nan       |
| XGB               |         25000 | iid     |    0.87393  |   0.971636 |            0.702007 |    0.612903 | 0.413043 | 0.493506 | 0.489393 |         0.705051 |            0.0058289 |              0.0042392  |  0.869802 | 0.4851   | 1317 |   12 |   27 |   19 |                   nan       |       nan       |
| RF                |          5000 | iid     |    0.443727 |   0.958    |            0.765437 |    0.413043 | 0.558824 | 0.475    | 0.459268 |         0.680454 |            0.0577197 |              0.0577197  |  0.868332 | 0.472226 |  939 |   27 |   15 |   19 |                   nan       |       nan       |
| HQD-Net(tau=0.80) |          5000 | iid     |    0.779533 |   0.966    |            0.727013 |    0.5      | 0.470588 | 0.484848 | 0.467515 |         1.02025  |            0.0119206 |              0.0119206  |  0.849699 | 0.465361 |  950 |   16 |   18 |   16 |                     0.20025 |         0.16467 |
| LGBM              |         10000 | iid     |    0.933023 |   0.970909 |            0.628177 |    0.666667 | 0.26087  | 0.375    | 0.405562 |         2.66715  |            0.0087356 |              0.00635316 |  0.877114 | 0.455537 | 1323 |    6 |   34 |   12 |                   nan       |       nan       |
| LGBM              |         20000 | iid     |    0.933023 |   0.970909 |            0.628177 |    0.666667 | 0.26087  | 0.375    | 0.405562 |         2.75658  |            0.0083762 |              0.00609178 |  0.877114 | 0.455537 | 1323 |    6 |   34 |   12 |                   nan       |       nan       |
| LGBM              |         25000 | iid     |    0.933023 |   0.970909 |            0.628177 |    0.666667 | 0.26087  | 0.375    | 0.405562 |         2.61288  |            0.0079398 |              0.0057744  |  0.877114 | 0.455537 | 1323 |    6 |   34 |   12 |                   nan       |       nan       |

## Key Files

- `csv_results/main_metrics_all_sample_sizes.csv`
- `csv_results/ablation_feature_blocks.csv`
- `csv_results/ablation_temporal_drift.csv`
- `csv_results/ablation_routing_threshold.csv`
- `csv_results/ablation_noise_robustness.csv`
- `figures/`
- `trained_models/`
