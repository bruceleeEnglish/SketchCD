# Baselines

`experiments/bench/` runs every competitor of the paper. CDKM, BatchedHartigan, and e-means are part of this
repository; SuperKMeans and FAISS are installed with pip.

| Method of the paper | Source | How the benchmark scripts find it |
|---|---|---|
| CDKM | `cdkm.cpp` | `bash build.sh cdkm` builds `bin/cdkm` |
| BatchedHartigan | `batched_hartigan/` | `bash build.sh hartigan` builds `bin/batched_hartigan` |
| FullSuperKMeans, SampleSuperKMeans, HierSuperKMeans, QuantSuperKMeans | SuperKMeans 0.2.0 | `pip install superkmeans==0.2.0` |
| EMeans-2, EMeans-5 | `e_means/`: the CPU path (`NumpyEmeans`) of the reference implementation, unchanged | imported from `baselines/`; set `EMEANS_PATH` to another folder that contains an `e_means` package to use a different copy |
| FaissKMeans, PCA-FaissKMeans | FAISS 1.14.3 | `pip install faiss-cpu==1.14.3` |

PCA-FaissKMeans is implemented in `experiments/bench/run_extra_baselines.py`.

## BatchedHartigan

`batched_hartigan/` is a copy of the SuperKMeans headers with the size-aware rule added to the pruned pass, and one
driver, `examples/batched_hartigan_bench.cpp`; `NOTICE.md` lists the changes. `build.sh` configures it with cmake:

```bash
OPENBLAS=/path/to/libopenblas.so bash build.sh hartigan
```

Eigen is found with `find_package(Eigen3)`; otherwise pass `EIGEN=/path/to/eigen`. Link it against the same OpenBLAS as
`bin/sketchcd`.

With 30 instead of 10 iterations the same binary computes the reference partition of the WCSS excess
(`Ctx.reference` in `experiments/bench/common.py`).

## e-means

`e_means/` holds the four files of the CPU path of the reference implementation, copied without changes from the FAISS
repository (`contrib/e_means`); `NOTICE.md` there gives the source and the license. `NumpyEmeans` needs `faiss` and
`numpy`. The GPU classes of the package are not included.
