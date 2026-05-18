
# HQD-Net v4 Balanced Leakage-Safe Results Summary

Sampling mode: `balanced`. This run removes post-execution leakage features and reports failure-focused metrics.

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

| model             |   sample_size | route   |   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   train_time_sec |   inference_time_sec |   latency_ms_per_sample |   roc_auc |   pr_auc |   tn |   fp |   fn |   tp |   quantum_route_train_ratio |   tau_threshold |   uncertainty_threshold |
|:------------------|--------------:|:--------|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|-----------------:|---------------------:|------------------------:|----------:|---------:|-----:|-----:|-----:|-----:|----------------------------:|----------------:|------------------------:|
| RF                |         25000 | iid     |    0.936056 |    0.9908  |             0.9908  |    0.99878  |   0.9828 | 0.990726 | 0.981726 |          2.29664 |            8.51604   |              1.70321    |  0.999888 | 0.99988  | 2497 |    3 |   43 | 2457 |                  nan        |      nan        |            nan          |
| RF                |         20000 | iid     |    0.925514 |    0.99475 |             0.99475 |    0.998489 |   0.991  | 0.99473  | 0.989528 |          1.21073 |            0.0992108 |              0.0248027  |  0.999832 | 0.99982  | 1997 |    3 |   18 | 1982 |                  nan        |      nan        |            nan          |
| LGBM              |         20000 | iid     |    0.376872 |    0.991   |             0.991   |    0.982318 |   1      | 0.99108  | 0.982159 |          3.41028 |            0.0187727 |              0.00469318 |  0.999626 | 0.999568 | 1964 |   36 |    0 | 2000 |                  nan        |      nan        |            nan          |
| HQD-Net(tau=0.80) |         20000 | iid     |    0.789954 |    0.9905  |             0.9905  |    0.99696  |   0.984  | 0.990438 | 0.981083 |          9.27294 |            0.106533  |              0.0266332  |  0.999483 | 0.999395 | 1994 |    6 |   32 | 1968 |                    0.300375 |        0.177246 |              0.00184302 |
| HQD-Net(tau=0.80) |         25000 | iid     |    0.48979  |    0.99    |             0.99    |    0.980392 |   1      | 0.990099 | 0.980196 |          6.6953  |            0.0773599 |              0.015472   |  0.999439 | 0.999336 | 2450 |   50 |    0 | 2500 |                    0.30005  |        0.177246 |              0.00166749 |
| HGB               |         25000 | iid     |    0.437634 |    0.991   |             0.991   |    0.982318 |   1      | 0.99108  | 0.982159 |          8.23449 |            0.0681384 |              0.0136277  |  0.999416 | 0.99931  | 2455 |   45 |    0 | 2500 |                  nan        |      nan        |            nan          |
| LGBM              |         25000 | iid     |    0.426117 |    0.9918  |             0.9918  |    0.983865 |   1      | 0.991867 | 0.983732 |          5.90116 |            0.0679314 |              0.0135863  |  0.999373 | 0.999282 | 2459 |   41 |    0 | 2500 |                  nan        |      nan        |            nan          |
| HGB               |         20000 | iid     |    0.453783 |    0.98925 |             0.98925 |    0.978953 |   1      | 0.989364 | 0.978726 |          5.81334 |            0.0614065 |              0.0153516  |  0.999246 | 0.99914  | 1957 |   43 |    0 | 2000 |                  nan        |      nan        |            nan          |
| RF                |         10000 | iid     |    0.862113 |    0.9895  |             0.9895  |    0.992951 |   0.986  | 0.989463 | 0.979024 |          1.03959 |            0.0791511 |              0.0395755  |  0.998556 | 0.998139 |  993 |    7 |   14 |  986 |                  nan        |      nan        |            nan          |
| LGBM              |         10000 | iid     |    0.391792 |    0.984   |             0.984   |    0.968992 |   1      | 0.984252 | 0.968496 |          3.1379  |            0.0209529 |              0.0104764  |  0.998141 | 0.997887 |  968 |   32 |    0 | 1000 |                  nan        |      nan        |            nan          |
| HGB               |         10000 | iid     |    0.453597 |    0.985   |             0.985   |    0.970874 |   1      | 0.985222 | 0.970437 |          5.44135 |            0.0489421 |              0.0244711  |  0.997619 | 0.997144 |  970 |   30 |    0 | 1000 |                  nan        |      nan        |            nan          |
| HQD-Net(tau=0.80) |         10000 | iid     |    0.527113 |    0.985   |             0.985   |    0.970874 |   1      | 0.985222 | 0.970437 |          6.92661 |            0.122219  |              0.0611096  |  0.996945 | 0.996499 |  970 |   30 |    0 | 1000 |                    0.300125 |        0.177246 |              0.00152833 |

## Key Files

- `csv_results/raw_class_distribution.csv`
- `csv_results/main_metrics_all_sample_sizes.csv`
- `csv_results/ablation_feature_blocks.csv`
- `csv_results/ablation_temporal_drift.csv`
- `csv_results/ablation_routing_threshold.csv`
- `csv_results/ablation_noise_robustness.csv`
- `figures/`
- `trained_models/`
