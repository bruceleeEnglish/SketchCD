# SketchCD

Code and raw results of the paper **"Highly Efficient Partitioned Vector Index with Sketched Coordinate Descent"**.

SketchCD builds the lists of an inverted-file (IVF) index. It decides with the size-aware rule of coordinate descent
in a compact sketch, certifies the decisions in full dimension, and runs coordinate descent in parallel over cells.

```
.
├── build.sh                  builds the binaries into bin/
├── src/
│   └── sketchcd.cpp          SketchCD
├── baselines/
│   ├── cdkm.cpp              exact coordinate descent (the CDKM baseline of Exp-12)
│   ├── batched_hartigan/     the BatchedHartigan baseline, a modified copy of SuperKMeans
│   ├── e_means/              the e-means baseline: CPU path of its reference implementation, unchanged
│   └── README.md             the competitors of the paper and where each comes from
├── experiments/
│   ├── prepare_data.py       downloads and converts the datasets
│   ├── reproduce.sh          the commands of every experiment, one block per experiment
│   ├── make_figures.py       turns the records into the tables and figures of the paper
│   ├── verify_theory.py      numerical checks of the formal statements of Sections 3 to 5
│   └── bench/                the benchmark scripts that reproduce.sh calls
└── results/                  the raw records behind every table and figure, one file per experiment
```

## 1. Build

```bash
bash build.sh sketchcd        # bin/sketchcd; needs g++ (C++17, OpenMP) and OpenBLAS
bash build.sh                 # also bin/cdkm and bin/batched_hartigan (cmake and Eigen for the last one; the paper used Eigen 5.0.0)
bash build.sh throughput      # bin/throughput, the matrix-multiplication and distance throughput of the machine
```

`OPENBLAS=/path/to/libopenblas.so` selects the OpenBLAS library. Use the same one for all three binaries.

## 2. Run SketchCD on your own vectors

```bash
bin/sketchcd base.fvecs 4000 out      # vectors in fvecs format, K = 4000 lists
```

This writes `out.centroids.fvecs` (the K centroids) and `out.labels.u32` (the list of every vector, 32-bit unsigned
integers) and prints one JSON line with the time of every stage. With `-` in place of `out` nothing is written.
Options are `key=value` pairs, listed at the head of `src/sketchcd.cpp`; the defaults are the configuration of the
paper. `OMP_NUM_THREADS` sets the number of threads.

## 3. Reproduce the paper

The benchmark scripts are Python 3.10 with `faiss` (1.14.3 in the paper), `numpy`, and `scipy`; the competitors are
listed in `baselines/README.md`. `make_figures.py` needs Python 3.12 or newer and `matplotlib`.

```bash
python experiments/prepare_data.py fiqa        # one dataset into data/fiqa/; "all" for everything
bash experiments/reproduce.sh                  # lists the blocks
bash experiments/reproduce.sh exp1             # runs one block; records go to runs/
SKETCHCD_RESULTS=runs python experiments/make_figures.py     # tables and figures from the new records
```

Without `SKETCHCD_RESULTS`, `make_figures.py` rebuilds the tables and figures of the paper from `results/`. The output
goes to `paper_output/`. `python experiments/verify_theory.py` needs only NumPy.

| Experiment | Block | Records in `results/` | Table or figure |
|---|---|---|---|
| Machine throughput | `exp0` | `exp00_throughput` | Sections 4.4, 5.3, and 6.1 (text) |
| Exp-1 build time and quality | `exp1` | `exp01_main`, `exp01_extra_baselines`, `exp01_centroid_gap` | Tables 3 and 6, Figures 1, 2, and 5 |
| Exp-2 time to equal quality | `exp2` | `exp02_tradeoff_sketchcd`, `exp02_tradeoff_baselines` | Figure 6 |
| Exp-3 Recall@k | `exp1` | `exp01_main`, `exp01_extra_baselines` | Table 4 |
| Exp-4 query throughput | `exp4` | `exp04_qps` | Figure 7 |
| Exp-5 scalability | `exp5` | `exp05_scale_k`, `exp05_scale_n`, `exp05_scale_d`, `exp05_threads`, `exp05_bigann100m` | Figure 8 (its 10M and 64x64 points come from `exp01_main`), Table 5 |
| Exp-6 memory | `exp6` | `exp06_memory` | (text) |
| Exp-7 sketch dimension | `exp7` | `exp07_sketch_dim` | Figure 9 |
| Exp-8 sensitivity | `exp8` | `exp08_sensitivity_quality`, `exp08_sensitivity_times` | Figure 10 |
| Exp-9 decision rule | `exp9` | `exp09_nc_rule`, `exp09_list_size_sweep`, `exp09_list_sizes`, `exp09_model_params` | Table 7, Figure 11 |
| Exp-10 certified pruning | `exp10` | `exp10_exactness` | Table 8 |
| Exp-11 ablation | `exp11` | `exp11_ablation` | Table 9 |
| Exp-12 exact coordinate descent | `exp1` | `exp01_extra_baselines` | Table 10 |

The paper uses ten datasets (Table 2), the first 100 million vectors of BIGANN for the scale experiment, and seven
sets derived from Tiny-ImageNet and DEEP-10M for the scalability experiment. The docstring of
`experiments/prepare_data.py` lists the source of each. The paper's runs read GIST from ann-benchmarks'
`gist-960-euclidean.hdf5` and used its ground truth; the exact recomputation in `prepare_data.py` changes the neighbor
sets of 14 of the 1000 queries, all through exactly tied duplicate vectors.

## Notes on reproducing the numbers

- **Machine.** Two Intel Xeon Gold 5218R (20 cores each, AVX-512), 128 GB of memory. Every run uses the 20 cores of one
  NUMA node, pinned with `taskset`. The benchmark scripts therefore need Linux and read `/sys/devices/system/node`.
- **Quality is deterministic, time is not.** Given a seed, SketchCD returns the same lists and centroids in practice (the sample mean of the PCA adds the
  per-thread partial sums in arbitrary order, so its last bits may in principle differ between runs). Build times
  depend on the machine and on what else runs on it; the times of the ablation, thread, and sensitivity experiments in
  the paper are the best of three runs on an otherwise idle machine.
- **One OpenBLAS for all three binaries.** The paper links them against OpenBLAS 0.3.21. Linking the BatchedHartigan
  driver against the OpenBLAS that ships inside the `superkmeans` wheel loads a second OpenMP runtime and makes it
  about three times slower, with the same output.
- **Reference partition.** The WCSS excess is relative to 30 iterations of BatchedHartigan, cached in `tmp/`. The
  size-aware pass of that code needs `d >= 128`, so on DEEP-10M (`d = 96`) the reference is 30 Lloyd iterations and
  BatchedHartigan is not reported. `results/exp01_extra_baselines.jsonl` still holds the three DEEP-10M runs; they
  make no size-aware move, and `make_figures.py` leaves them out.
- **WCSS excess.** Every method is evaluated as it is returned: its WCSS uses the centroids that it outputs, by which
  an index routes its queries. SketchCD, e-means, PCA-FaissKMeans, and CDKM return the exact means of their lists. The
  sampled, hierarchical, and quantized configurations of SuperKMeans do not, and `results/exp01_centroid_gap.jsonl`
  records how much of their excess the means of the lists would remove (Metrics, Section 6.1).
- **Parameters.** All datasets use the default configuration. It was tuned on two collections that are not part of
  the evaluation (DBpedia-OpenAI and MXBAI); they are not needed to reproduce the paper.

## License

The code of this repository is released under the MIT license (`LICENSE`). Two folders hold third-party code under
the MIT licenses of their authors, each with its `LICENSE` and a `NOTICE.md` that names the source and the changes:
`baselines/batched_hartigan/` is a modified copy of SuperKMeans, and `baselines/e_means/` is an unchanged copy of the
CPU path of e-means from the FAISS repository.
