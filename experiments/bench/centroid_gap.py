"""How much of a method's WCSS is due to returned centroids that are not the means of the returned lists (Exp-1).

The paper evaluates every method as it is returned: the WCSS excess uses the centroids that the method outputs, because
an index routes its queries by them. SketchCD, e-means, PCA-FaissKMeans, and CDKM return the exact means of their lists.
A method that trains on a sample, on meso-clusters, or on quantized codes returns centroids that differ from the means of
its final lists. For every saved build (<method>_s<seed>_{C,L}.npy, written by run_methods.py and run_extra_baselines.py)
this script records

  wcss   sum over the vectors of |x - c|^2 with the returned centroid c of the vector's list (the `wcss` of the records)
  gap    sum over the lists of b_k |c_k - mean_k|^2, which is exactly wcss minus the WCSS of the same lists with their means

so 100 * gap / Jref is the part of the WCSS excess, in points, that the means of the lists would remove.

Usage: python centroid_gap.py --out runs/exp01_centroid_gap.jsonl
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp

from common import PAPER_DATASETS, RUNS, SAVE, append_record, dataset_files, fvecs_shape

METHODS = ["FaissKMeans", "PCA-FaissKMeans", "EMeans-2", "EMeans-5", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans",
           "QuantSuperKMeans", "BatchedHartigan", "SketchCD-Fast", "SketchCD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--save", default=SAVE)
    ap.add_argument("--out", default=f"{RUNS}/exp01_centroid_gap.jsonl")
    a = ap.parse_args()
    for name in a.datasets:
        base = dataset_files(name)[0]
        N, d = fvecs_shape(base)
        X = np.memmap(base, dtype=np.float32, mode="r", shape=(N, d + 1))
        builds = []  # [method, seed, centroids, labels, sums of the lists, wcss]
        for m in a.methods:
            for seed in a.seeds:
                fc, fl = (f"{a.save}/{name}/{m}_s{seed}_{x}.npy" for x in "CL")
                if os.path.exists(fc) and os.path.exists(fl):
                    C = np.load(fc)
                    builds.append([m, seed, C, np.load(fl).astype(np.int64), np.zeros(C.shape, dtype=np.float64), 0.0])
        rows = 65536 if d <= 4096 else 16384
        for s in range(0, N, rows):
            x = np.ascontiguousarray(X[s:s + rows, 1:])
            for b in builds:
                C, lab = b[2], b[3][s:s + rows]
                onehot = sp.csr_matrix((np.ones(len(lab), dtype=np.float32), (lab, np.arange(len(lab)))), shape=(C.shape[0], len(lab)))
                b[4] += onehot @ x
                diff = x - C[lab]
                b[5] += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
        for m, seed, C, lab, sums, wcss in builds:
            size = np.bincount(lab, minlength=C.shape[0]).astype(np.float64)
            mean = sums / np.maximum(size, 1)[:, None]
            gap = float((size * ((C.astype(np.float64) - mean) ** 2).sum(axis=1)).sum())
            append_record(a.out, dict(dataset=name, method=m, seed=seed, N=N, d=d, K=int(C.shape[0]), wcss=wcss, gap=gap,
                                      empty=int((size == 0).sum())))
            print(f"{name:13s} {m:18s} s{seed} wcss={wcss:.6e} gap={gap:.3e} ({100 * gap / wcss:.3f}% of the WCSS)", flush=True)
        del X, builds


if __name__ == "__main__":
    main()
