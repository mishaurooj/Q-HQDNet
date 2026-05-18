# HQD-Net v6 Final Results Summary

This run uses leakage-safe grouped-temporal evaluation. Training may be balanced, but validation and test remain natural future splits.

## Core Design

- Post-execution leakage features removed.
- Train/validation/test split is ordered by first job start time.
- Training balance is applied only after splitting.
- HQD-Net-v6 uses ensemble uncertainty plus workload complexity for hard-example routing.
- The structural specialist uses graph/DAG features with a quantum-inspired random Fourier structural encoder.

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

## Raw Class Distribution
|   total_jobs |   failures |   successes |   failure_ratio |
|-------------:|-----------:|------------:|----------------:|
|  1.46589e+06 |      42909 | 1.42298e+06 |       0.0292717 |

## Top Models by PR-AUC
|   threshold |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |      mcc |   roc_auc |   pr_auc |    tn |   fp |   fn |   tp | model      |   sample_size | route                                  |   train_time_sec |   inference_time_sec |   latency_ms_per_sample |   ram_delta_mb |   train_positive_ratio_raw |   train_positive_ratio_used |   val_positive_ratio |   test_positive_ratio |   train_rows_raw |   train_rows_used |   val_rows |   test_rows |   quantum_route_train_ratio |   quantum_route_test_ratio |   tau_threshold |   uncertainty_threshold |
|------------:|-----------:|--------------------:|------------:|---------:|---------:|---------:|----------:|---------:|------:|-----:|-----:|-----:|:-----------|--------------:|:---------------------------------------|-----------------:|---------------------:|------------------------:|---------------:|---------------------------:|----------------------------:|---------------------:|----------------------:|-----------------:|------------------:|-----------:|------------:|----------------------------:|---------------------------:|----------------:|------------------------:|
|    0.232829 |    0.88455 |            0.895466 |    0.795747 | 0.943618 | 0.863397 | 0.773007 |  0.974856 | 0.968258 | 10394 | 1873 |  436 | 7297 | LGBM       |        100000 | grouped_temporal                       |         2.89378  |            0.014686  |             0.0007343   |       14.2656  |                   0.316671 |                         0.5 |               0.3433 |               0.38665 |            70000 |             44334 |      10000 |       20000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.263578 |    0.8818  |            0.893224 |    0.791003 | 0.943618 | 0.860597 | 0.768284 |  0.97371  | 0.966675 | 10339 | 1928 |  436 | 7297 | HGB        |        100000 | grouped_temporal                       |         4.8931   |            0.0735236 |             0.00367618  |        3.53125 |                   0.316671 |                         0.5 |               0.3433 |               0.38665 |            70000 |             44334 |      10000 |       20000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.41124  |    0.86725 |            0.882606 |    0.763929 | 0.950343 | 0.847001 | 0.745828 |  0.972815 | 0.966672 |  9996 | 2271 |  384 | 7349 | XGB        |        100000 | grouped_temporal                       |         0.770376 |            0.0048371 |             0.000241855 |       10.6523  |                   0.316671 |                         0.5 |               0.3433 |               0.38665 |            70000 |             44334 |      10000 |       20000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.246963 |    0.8744  |            0.886451 |    0.77965  | 0.938622 | 0.851782 | 0.754077 |  0.972369 | 0.964743 |  5135 | 1020 |  236 | 3609 | HGB        |         50000 | grouped_temporal                       |         3.47836  |            0.0341829 |             0.00341829  |        8.21875 |                   0.317314 |                         0.5 |               0.343  |               0.3845  |            35000 |             22212 |       5000 |       10000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.205257 |    0.8695  |            0.883545 |    0.768954 | 0.944343 | 0.847671 | 0.747498 |  0.97143  | 0.964149 |  5064 | 1091 |  214 | 3631 | LGBM       |         50000 | grouped_temporal                       |         2.36841  |            0.0065856 |             0.00065856  |       14.9219  |                   0.317314 |                         0.5 |               0.343  |               0.3845  |            35000 |             22212 |       5000 |       10000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.426919 |    0.876   |            0.888288 |    0.781014 | 0.941482 | 0.853774 | 0.757594 |  0.9707   | 0.963213 |  5140 | 1015 |  225 | 3620 | XGB        |         50000 | grouped_temporal                       |         0.548847 |            0.0022249 |             0.00022249  |       17.8477  |                   0.317314 |                         0.5 |               0.343  |               0.3845  |            35000 |             22212 |       5000 |       10000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.433832 |    0.86055 |            0.877718 |    0.752193 | 0.953446 | 0.840947 | 0.735912 |  0.970835 | 0.96005  |  9838 | 2429 |  360 | 7373 | HQD-Net-v6 |        100000 | grouped_temporal_hard_quantum_inspired |        23.896    |            2.93e-05  |             1.465e-06   |      174.621   |                   0.316671 |                         0.5 |               0.3433 |               0.38665 |            70000 |             44334 |      10000 |       20000 |                    0.651847 |                     0.5975 |        0.521747 |                0.473525 |
|    0.473386 |    0.8748  |            0.886056 |    0.78306  | 0.93595  | 0.852706 | 0.754298 |  0.966808 | 0.959017 |  2562 |  502 |  124 | 1812 | XGB        |         25000 | grouped_temporal                       |         0.509601 |            0.0046874 |             0.00093748  |        0       |                   0.315943 |                         0.5 |               0.3472 |               0.3872  |            17500 |             11058 |       2500 |        5000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.27033  |    0.867   |            0.879787 |    0.769851 | 0.936467 | 0.845024 | 0.74124  |  0.965842 | 0.958411 |  2522 |  542 |  123 | 1813 | LGBM       |         25000 | grouped_temporal                       |         2.18139  |            0.0038357 |             0.00076714  |       14.8906  |                   0.315943 |                         0.5 |               0.3472 |               0.3872  |            17500 |             11058 |       2500 |        5000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.377138 |    0.8402  |            0.862573 |    0.718963 | 0.959428 | 0.82197  | 0.705777 |  0.969202 | 0.958357 |  4713 | 1442 |  156 | 3689 | HQD-Net-v6 |         50000 | grouped_temporal_hard_quantum_inspired |        11.0609   |            2.98e-05  |             2.98e-06    |       85.5703  |                   0.317314 |                         0.5 |               0.343  |               0.3845  |            35000 |             22212 |       5000 |       10000 |                    0.65226  |                     0.5897 |        0.520826 |                0.467695 |
|    0.503516 |    0.8645  |            0.877372 |    0.767036 | 0.934965 | 0.842716 | 0.736707 |  0.965222 | 0.956908 |  2006 |  441 |  101 | 1452 | XGB        |         20000 | grouped_temporal                       |         0.474072 |            0.0010553 |             0.000263825 |        4.89844 |                   0.315714 |                         0.5 |               0.3465 |               0.38825 |            14000 |              8840 |       2000 |        4000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.398558 |    0.8754  |            0.884549 |    0.789335 | 0.925103 | 0.851843 | 0.75249  |  0.964531 | 0.956532 |  2586 |  478 |  145 | 1791 | HGB        |         25000 | grouped_temporal                       |         5.68396  |            0.0173794 |             0.00347588  |        6.99219 |                   0.315943 |                         0.5 |               0.3472 |               0.3872  |            17500 |             11058 |       2500 |        5000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.388056 |    0.8525  |            0.868623 |    0.745789 | 0.94076  | 0.832005 | 0.718747 |  0.96517  | 0.955776 |  1949 |  498 |   92 | 1461 | HGB        |         20000 | grouped_temporal                       |        13.1843   |            0.0283237 |             0.00708093  |        6.91406 |                   0.315714 |                         0.5 |               0.3465 |               0.38825 |            14000 |              8840 |       2000 |        4000 |                  nan        |                   nan      |      nan        |              nan        |
|    0.474081 |    0.856   |            0.872048 |    0.749589 | 0.943182 | 0.835316 | 0.725151 |  0.963439 | 0.950356 |  2454 |  610 |  110 | 1826 | HQD-Net-v6 |         25000 | grouped_temporal_hard_quantum_inspired |         8.50767  |            3.24e-05  |             6.48e-06    |       18.582   |                   0.315943 |                         0.5 |               0.3472 |               0.3872  |            17500 |             11058 |       2500 |        5000 |                    0.65274  |                     0.6132 |        0.522072 |                0.4568   |
|    0.404463 |    0.869   |            0.881463 |    0.769651 | 0.93254  | 0.843301 | 0.74249  |  0.961906 | 0.949262 |  1033 |  211 |   51 |  705 | XGB        |         10000 | grouped_temporal                       |         0.49558  |            0.0008935 |             0.00044675  |        0       |                   0.314571 |                         0.5 |               0.375  |               0.378   |             7000 |              4404 |       1000 |        2000 |                  nan        |                   nan      |      nan        |              nan        |

## Key Output Files

- `csv_results/main_metrics_all_sample_sizes.csv`
- `csv_results/ablation_feature_blocks.csv`
- `csv_results/ablation_routing_threshold.csv`
- `csv_results/ablation_noise_robustness.csv`
- `csv_results/feature_importance_*.csv`
- `figures/`
- `trained_models/`
