# Records of the paper

One JSON object per line and run, one file per experiment and kind of run. `experiments/make_figures.py` reads exactly
these files.

| File | Written by (`experiments/bench/`) | Content |
|---|---|---|
| `exp00_throughput.jsonl` | `throughput.cpp` (`bin/throughput`) | Section 6.1 (Implementation) and Section 5.3: throughput of the matrix multiplications and of full-dimensional distances and the read bandwidth of one socket, one line per dataset shape and kind |
| `exp01_main.jsonl` | `run_methods.py` | Exp-1, Exp-3: the seven configurations of the three systems, SketchCD, SketchCD-Fast; three seeds (FaissKMeans: one) |
| `exp01_extra_baselines.jsonl` | `run_extra_baselines.py` | Exp-1, Exp-3, Exp-12: PCA-FaissKMeans, BatchedHartigan, CDKM |
| `exp01_centroid_gap.jsonl` | `centroid_gap.py` | Exp-1, Metrics of Section 6.1: for every saved build, its WCSS and the part that the means of its lists would remove |
| `exp02_tradeoff_sketchcd.jsonl` | `run_methods.py` | Exp-2: SketchCD with 0 to 16 sketch sweeps and 0 to 6 verified sweeps |
| `exp02_tradeoff_baselines.jsonl` | `run_baseline_curves.py` | Exp-2: the competitors at several budgets |
| `exp04_qps.jsonl` | `measure_qps.py` | Exp-4: queries per second and recall at increasing numbers of probes |
| `exp05_scale_k.jsonl`, `exp05_scale_n.jsonl`, `exp05_scale_d.jsonl` | `run_methods.py` | Exp-5: number of lists, vectors, dimensions |
| `exp05_threads.jsonl` | `run_sketchcd_configs.py` | Exp-5: 1 to 20 threads, best of three runs on an idle machine |
| `exp05_bigann100m.jsonl`, `exp05_bigann100m_stopped.txt` | `run_100m.py` | Exp-5: BIGANN-100M; the second file lists the methods that the watchdog stopped |
| `exp06_memory.jsonl` | `measure_memory.py` | Exp-6 |
| `exp07_sketch_dim.jsonl` | `run_sketchcd_configs.py` | Exp-7: seed 42, best of two runs |
| `exp08_sensitivity_quality.jsonl` | `run_sketchcd_configs.py` | Exp-8: three seeds, used for WCSS and recall |
| `exp08_sensitivity_times.jsonl` | `run_sketchcd_configs.py` | Exp-8: seed 42, best of three runs on an idle machine, used for the build times |
| `exp09_nc_rule.jsonl` | `run_methods.py` | Exp-9: SketchCD-NC (`SketchCD@weighted=0`) at the default K |
| `exp09_list_size_sweep.jsonl` | `run_methods.py` | Exp-9: both rules with K from sqrt(n) to 64 sqrt(n) |
| `exp09_list_sizes.jsonl` | `list_sizes.py` | Exp-9: list-size quantiles of the saved indexes |
| `exp09_model_params.jsonl` | `model_params.py` | Section 4.2: sigma^2 / Delta^2 on the reference partitions |
| `exp10_exactness.jsonl` | `check_exactness.py` | Exp-10: seed 42, best of two runs for the times |
| `exp11_ablation.jsonl` | `run_sketchcd_configs.py` | Exp-11: best of three runs on an idle machine |

Notes on the records:

- `dataset` is a name of `experiments/prepare_data.py`. A suffix `-k<m>` means `K = m * sqrt(n)` instead of `4 * sqrt(n)`.
- `SketchCD@key=value,...` is SketchCD with these options; `SketchCD@weighted=0` is SketchCD-NC.
- Two files differ from the raw output of the runs, and no value was changed in either: in `exp11_ablation.jsonl`
  the default configuration is named `default` (the raw records called it `skv`), and `exp09_model_params.jsonl`
  keeps only the fields that `model_params.py` writes.
- `exp01_extra_baselines.jsonl` holds three BatchedHartigan runs on DEEP-10M without a single size-aware move
  (`parts.moves` is all zero). They are Lloyd iterations, and `make_figures.py` leaves them out.
- `exp09_nc_rule.jsonl` has no record for GIST. For GIST, `make_figures.py` takes SketchCD-NC from the `unweighted`
  record of `exp11_ablation.jsonl`, the same configuration and seed; builds are deterministic, so the lists are the same.
  The block `exp9` of `reproduce.sh` writes the GIST record as well.
- Records that the paper does not use: two runs `SketchCD@candk=256,fcandk=512` in `exp01_main.jsonl`; the
  single-seed CDKM runs on FiQA, CIFAR-10, and CIFAR-100 in `exp01_extra_baselines.jsonl`; and in
  `exp02_tradeoff_baselines.jsonl` an earlier run of FiQA, DBpedia-3S, CIFAR-10, and CIFAR-100 that evaluated only
  two scan budgets. Where a configuration has two records, `make_figures.py` uses the later one.
- The row "no pruning" of the ablation table takes its build time from `exp10_exactness.jsonl` (`total_all_s`).
