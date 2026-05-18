
# HQD-Net v4 Grouped-Temporal Leakage-Safe Results Summary

Sampling mode: `train_balanced`. v5 balances only the training split and evaluates on natural grouped-temporal validation/test partitions. It removes post-execution leakage features and reports failure-focused metrics.

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

| model             |   sample_size | route            |   threshold |   accuracy |   balanced_accuracy |   precision |    recall |        f1 |      mcc |   train_time_sec |   inference_time_sec |   latency_ms_per_sample |   roc_auc |   pr_auc |   tn |   fp |   fn |   tp |   train_positive_ratio_raw |   train_positive_ratio_used |   val_positive_ratio |   test_positive_ratio |   train_rows_used |   val_rows |   test_rows |   quantum_route_train_ratio |   quantum_route_test_ratio |   tau_threshold |   uncertainty_threshold |
|:------------------|--------------:|:-----------------|------------:|-----------:|--------------------:|------------:|----------:|----------:|---------:|-----------------:|---------------------:|------------------------:|----------:|---------:|-----:|-----:|-----:|-----:|---------------------------:|----------------------------:|---------------------:|----------------------:|------------------:|-----------:|------------:|----------------------------:|---------------------------:|----------------:|------------------------:|
| RF                |          5000 | grouped_temporal |   0.298262  |   0.692    |            0.754389 |   0.0285714 | 0.818182  | 0.0552147 | 0.114241 |         0.314808 |            0.0395486 |              0.0395486  |  0.879309 | 0.284428 |  683 |  306 |    2 |    9 |                  0.0445714 |                         0.5 |            0.002     |             0.011     |              1248 |        500 |        1000 |                  nan        |                 nan        |      nan        |            nan          |
| XGB               |          5000 | grouped_temporal |   0.575301  |   0.753    |            0.740279 |   0.031746  | 0.727273  | 0.0608365 | 0.115449 |         0.474256 |            0.0030915 |              0.0030915  |  0.870347 | 0.283768 |  745 |  244 |    3 |    8 |                  0.0445714 |                         0.5 |            0.002     |             0.011     |              1248 |        500 |        1000 |                  nan        |                 nan        |      nan        |            nan          |
| HQD-Net(tau=0.80) |          5000 | grouped_temporal |   0.324143  |   0.673    |            0.744784 |   0.0269461 | 0.818182  | 0.0521739 | 0.108267 |         2.28365  |            0.0494224 |              0.0494224  |  0.880963 | 0.267213 |  664 |  325 |    2 |    9 |                  0.0445714 |                         0.5 |            0.002     |             0.011     |              1248 |        500 |        1000 |                    0.300481 |                   0.165    |        0.176549 |              0.00246048 |
| LGBM              |          5000 | grouped_temporal |   0.109432  |   0.694    |            0.7554   |   0.028754  | 0.818182  | 0.0555556 | 0.114893 |         1.43233  |            0.0034136 |              0.0034136  |  0.848791 | 0.25978  |  685 |  304 |    2 |    9 |                  0.0445714 |                         0.5 |            0.002     |             0.011     |              1248 |        500 |        1000 |                  nan        |                 nan        |      nan        |            nan          |
| HGB               |          5000 | grouped_temporal |   0.0259638 |   0.72     |            0.723596 |   0.0280702 | 0.727273  | 0.0540541 | 0.103327 |         4.09807  |            0.0214485 |              0.0214485  |  0.803842 | 0.234438 |  712 |  277 |    3 |    8 |                  0.0445714 |                         0.5 |            0.002     |             0.011     |              1248 |        500 |        1000 |                  nan        |                 nan        |      nan        |            nan          |
| HQD-Net(tau=0.80) |         10000 | grouped_temporal |   0.531828  |   0.955636 |            0.532874 |   0.266667  | 0.0740741 | 0.115942  | 0.122945 |         2.13715  |            0.0542389 |              0.0394465  |  0.672267 | 0.139047 | 1310 |   11 |   50 |    4 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                    0.301958 |                   0.114909 |        0.166162 |              0.00267624 |
| HQD-Net(tau=0.80) |         20000 | grouped_temporal |   0.531828  |   0.955636 |            0.532874 |   0.266667  | 0.0740741 | 0.115942  | 0.122945 |         2.57002  |            0.250408  |              0.182115   |  0.672267 | 0.139047 | 1310 |   11 |   50 |    4 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                    0.301958 |                   0.114909 |        0.166162 |              0.00267624 |
| HQD-Net(tau=0.80) |         25000 | grouped_temporal |   0.531828  |   0.955636 |            0.532874 |   0.266667  | 0.0740741 | 0.115942  | 0.122945 |         2.87249  |            0.0672167 |              0.0488849  |  0.672267 | 0.139047 | 1310 |   11 |   50 |    4 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                    0.301958 |                   0.114909 |        0.166162 |              0.00267624 |
| XGB               |         10000 | grouped_temporal |   0.712315  |   0.957091 |            0.52475  |   0.272727  | 0.0555556 | 0.0923077 | 0.107931 |         0.380581 |            0.0034746 |              0.00252698 |  0.653812 | 0.138426 | 1313 |    8 |   51 |    3 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                  nan        |                 nan        |      nan        |            nan          |
| XGB               |         20000 | grouped_temporal |   0.712315  |   0.957091 |            0.52475  |   0.272727  | 0.0555556 | 0.0923077 | 0.107931 |         0.377996 |            0.0033444 |              0.00243229 |  0.653812 | 0.138426 | 1313 |    8 |   51 |    3 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                  nan        |                 nan        |      nan        |            nan          |
| XGB               |         25000 | grouped_temporal |   0.712315  |   0.957091 |            0.52475  |   0.272727  | 0.0555556 | 0.0923077 | 0.107931 |         0.568999 |            0.0035773 |              0.00260167 |  0.653812 | 0.138426 | 1313 |    8 |   51 |    3 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                  nan        |                 nan        |      nan        |            nan          |
| RF                |         10000 | grouped_temporal |   0.562153  |   0.958545 |            0.543268 |   0.384615  | 0.0925926 | 0.149254  | 0.173695 |         0.319039 |            0.0350524 |              0.0254927  |  0.666639 | 0.137232 | 1313 |    8 |   49 |    5 |                  0.0345043 |                         0.5 |            0.0174419 |             0.0392727 |              1328 |        688 |        1375 |                  nan        |                 nan        |      nan        |            nan          |

## Key Files

- `csv_results/raw_class_distribution.csv`
- `csv_results/main_metrics_all_sample_sizes.csv`
- `csv_results/ablation_feature_blocks.csv`
- `csv_results/ablation_temporal_drift.csv`
- `csv_results/ablation_routing_threshold.csv`
- `csv_results/ablation_noise_robustness.csv`
- `figures/`
- `trained_models/`
