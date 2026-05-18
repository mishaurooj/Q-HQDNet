
# HQD-Net-v7 Proposed System Results

This run reports a full proposed system, not a standalone weak quantum branch. HQD-Net-v7 uses the strongest classical backbone with hard-example structural correction and validation-tuned fusion.

## Core Design

- Leakage-safe feature filtering.
- Grouped-temporal train/validation/test evaluation.
- Train-only balancing; validation/test remain natural future splits.
- Strong baselines: LR, RF, HGB, XGB, LGBM.
- Proposed system: LGBM/HGB backbone + auxiliary learner + graph/quantum-inspired structural specialist + hard-example routing.

## Raw Class Distribution

|   total_jobs |   failures |   successes |   failure_ratio |
|-------------:|-----------:|------------:|----------------:|
|  1.46589e+06 |      42853 | 1.42304e+06 |       0.0292335 |

## Removed Leakage Features
```json
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
  "mem_spike_mean",
  "target",
  "label"
]
```

## Top Models by PR-AUC

|   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   roc_auc |   pr_auc |    tn |   fp |   fn |   tp | model               |   sample_size | route                                     |   train_time_sec |   inference_time_sec |   latency_ms_per_sample |   train_positive_ratio_raw |   train_positive_ratio_used |   val_positive_ratio |   test_positive_ratio |   train_rows_raw |   train_rows_used |   val_rows |   test_rows |   quantum_route_test_ratio |   tau_threshold |   fusion_a_backbone |   fusion_b_aux |   fusion_c_rf |   fusion_d_struct |   fusion_e_max | backbone   | aux_model   |
|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|----------:|---------:|------:|-----:|-----:|-----:|:--------------------|--------------:|:------------------------------------------|-----------------:|---------------------:|------------------------:|---------------------------:|----------------------------:|---------------------:|----------------------:|-----------------:|------------------:|-----------:|------------:|---------------------------:|----------------:|--------------------:|---------------:|--------------:|------------------:|---------------:|:-----------|:------------|
|    0.978301 |     0.9812 |            0.868927 |    0.791878 | 0.746411 | 0.768473 | 0.759044 |  0.969626 | 0.850719 |  4750 |   41 |   53 |  156 | HGB                 |         25000 | grouped_temporal                          |        3.03406   |           0.0624406  |             0.0124881   |                  0.0284571 |                         0.5 |               0.0176 |                0.0418 |            17500 |               996 |       2500 |        5000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.943758 |     0.9838 |            0.849692 |    0.885542 | 0.703349 | 0.784    | 0.781258 |  0.973401 | 0.839218 |  4772 |   19 |   62 |  147 | XGB                 |         25000 | grouped_temporal                          |        0.567177  |           0.0130219  |             0.00260439  |                  0.0284571 |                         0.5 |               0.0176 |                0.0418 |            17500 |               996 |       2500 |        5000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.964873 |     0.9812 |            0.850623 |    0.81768  | 0.708134 | 0.758974 | 0.751344 |  0.970998 | 0.791474 |  4758 |   33 |   61 |  148 | HQD-Net-v7-Proposed |         25000 | proposed_backbone_hard_structural_routing |        0.0158966 |           0.00600004 |             0.00120001  |                  0.0284571 |                         0.5 |               0.0176 |                0.0418 |            17500 |               996 |       2500 |        5000 |                    0.4584  |        0.141422 |                0.6  |           0.2  |           0   |               0.2 |              0 | LGBM       | XGB         |
|    0.866059 |     0.9782 |            0.926849 |    0.689394 | 0.870813 | 0.769556 | 0.763981 |  0.972424 | 0.790206 |  4709 |   82 |   27 |  182 | RF                  |         25000 | grouped_temporal                          |        0.458471  |           0.149494   |             0.0298988   |                  0.0284571 |                         0.5 |               0.0176 |                0.0418 |            17500 |               996 |       2500 |        5000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.98324  |     0.9776 |            0.805273 |    0.801242 | 0.617225 | 0.697297 | 0.692171 |  0.969265 | 0.772645 |  4759 |   32 |   80 |  129 | LGBM                |         25000 | grouped_temporal                          |        2.17594   |           0.028918   |             0.00578361  |                  0.0284571 |                         0.5 |               0.0176 |                0.0418 |            17500 |               996 |       2500 |        5000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.913967 |     0.9762 |            0.745979 |    0.82766  | 0.496173 | 0.620415 | 0.630257 |  0.956684 | 0.7373   | 19135 |   81 |  395 |  389 | HQD-Net-v7-Proposed |        100000 | proposed_backbone_hard_structural_routing |        0.0277994 |           0.00700092 |             0.000350046 |                  0.0271143 |                         0.5 |               0.0198 |                0.0392 |            70000 |              3796 |      10000 |       20000 |                    0.44395 |        0.230886 |                0.6  |           0.2  |           0   |               0.2 |              0 | LGBM       | XGB         |
|    0.964003 |     0.97   |            0.625    |    1        | 0.25     | 0.4      | 0.492366 |  0.964089 | 0.733491 |  1920 |    0 |   60 |   20 | HQD-Net-v7-Proposed |         10000 | proposed_backbone_hard_structural_routing |        0.011744  |           0          |             0           |                  0.028     |                         0.5 |               0.01   |                0.04   |             7000 |               392 |       1000 |        2000 |                    0.4655  |        0.194022 |                0.5  |           0.2  |           0.1 |               0.2 |              0 | LGBM       | XGB         |
|    0.921127 |     0.9698 |            0.731178 |    0.701439 | 0.471014 | 0.563584 | 0.560267 |  0.966596 | 0.698912 |  9503 |   83 |  219 |  195 | HQD-Net-v7-Proposed |         50000 | proposed_backbone_hard_structural_routing |        0.0317023 |           0          |             0           |                  0.0275714 |                         0.5 |               0.0172 |                0.0414 |            35000 |              1930 |       5000 |       10000 |                    0.4637  |        0.145463 |                0.55 |           0.25 |           0   |               0.2 |              0 | LGBM       | XGB         |
|    0.975226 |     0.975  |            0.81322  |    0.657143 | 0.638889 | 0.647887 | 0.634998 |  0.955826 | 0.688802 |   952 |   12 |   13 |   23 | HGB                 |          5000 | grouped_temporal                          |        1.99049   |           0.0260503  |             0.0260503   |                  0.0325714 |                         0.5 |               0.018  |                0.036  |             3500 |               228 |        500 |        1000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.987787 |     0.9675 |            0.566667 |    1        | 0.133333 | 0.235294 | 0.359135 |  0.955396 | 0.684146 |  3850 |    0 |  130 |   20 | HQD-Net-v7-Proposed |         20000 | proposed_backbone_hard_structural_routing |        0         |           0          |             0           |                  0.0247857 |                         0.5 |               0.0185 |                0.0375 |            14000 |               694 |       2000 |        4000 |                    0.459   |        0.18582  |                0.55 |           0.25 |           0   |               0.2 |              0 | LGBM       | XGB         |
|    0.85494  |     0.966  |            0.888774 |    0.517857 | 0.805556 | 0.630435 | 0.629994 |  0.965523 | 0.654433 |   937 |   27 |    7 |   29 | HQD-Net-v7-Proposed |          5000 | proposed_backbone_hard_structural_routing |        0.0048902 |           0.00299811 |             0.00299811  |                  0.0325714 |                         0.5 |               0.018  |                0.036  |             3500 |               228 |        500 |        1000 |                    0.348   |        0.284228 |                0.5  |           0.2  |           0.1 |               0.2 |              0 | LGBM       | XGB         |
|    0.911861 |     0.966  |            0.915514 |    0.516667 | 0.861111 | 0.645833 | 0.651877 |  0.965739 | 0.652878 |   935 |   29 |    5 |   31 | XGB                 |          5000 | grouped_temporal                          |        0.418162  |           0          |             0           |                  0.0325714 |                         0.5 |               0.018  |                0.036  |             3500 |               228 |        500 |        1000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.827044 |     0.978  |            0.801406 |    0.733333 | 0.611111 | 0.666667 | 0.658302 |  0.97058  | 0.648439 |   956 |    8 |   14 |   22 | RF                  |          5000 | grouped_temporal                          |        0.271681  |           0.0782695  |             0.0782695   |                  0.0325714 |                         0.5 |               0.018  |                0.036  |             3500 |               228 |        500 |        1000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.958427 |     0.963  |            0.860477 |    0.490909 | 0.75     | 0.593407 | 0.589115 |  0.964485 | 0.638429 |   936 |   28 |    9 |   27 | LGBM                |          5000 | grouped_temporal                          |        0.458436  |           0.00500011 |             0.00500011  |                  0.0325714 |                         0.5 |               0.018  |                0.036  |             3500 |               228 |        500 |        1000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |
|    0.9505   |     0.9646 |            0.596731 |    0.794118 | 0.195652 | 0.313953 | 0.383566 |  0.96458  | 0.616773 |  9565 |   21 |  333 |   81 | XGB                 |         50000 | grouped_temporal                          |        0.621565  |           0.0161998  |             0.00161998  |                  0.0275714 |                         0.5 |               0.0172 |                0.0414 |            35000 |              1930 |       5000 |       10000 |                  nan       |      nan        |              nan    |         nan    |         nan   |             nan   |            nan | nan        | nan         |

## Key Output Files

- `csv_results/main_metrics_all_sample_sizes.csv`
- `csv_results/ablation_feature_blocks.csv`
- `csv_results/ablation_routing_threshold.csv`
- `csv_results/ablation_noise_robustness.csv`
- `csv_results/feature_importance_*.csv`
- `figures/`
- `trained_models/`
