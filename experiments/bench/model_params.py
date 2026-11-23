"""The ratio b* = sigma^2 / Delta^2 of the bias analysis (Section 4.2), measured on the reference partition of a dataset.

  sigma^2   mean squared distance of a vector to its centroid
  Delta^2   mean squared distance of a centroid to the nearest other centroid
By Corollary 4.4 a young list with fewer than b* members has, in expectation, the higher nearest-centroid cost for the
vectors of its own population.

Usage: python model_params.py --datasets gist fiqa ... --out runs/exp09_model_params.jsonl
"""
import argparse
import math

import faiss
import numpy as np

from common import PAPER_DATASETS, RUNS, Ctx, append_record


def model_params(ctx, C):
    X, N, d = ctx.X, ctx.N, ctx.d
    K = C.shape[0]
    index = faiss.IndexFlatL2(d)
    index.add(C)
    D2, _ = index.search(X, 2)
    sigma2 = float(D2[:, 0].astype(np.float64).mean())
    dC, _ = index.search(C, 2)
    delta_nn2 = float(dC[:, 1].mean())
    return {"K": K, "b": N / K, "sigma2": sigma2, "delta_nn2": delta_nn2, "bstar": sigma2 / delta_nn2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS)
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--out", default=f"{RUNS}/exp09_model_params.jsonl")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 100)
        K = int(round(4 * math.sqrt(ctx.N)))
        mp = model_params(ctx, ctx.reference(K))
        append_record(a.out, dict(dataset=name, exp="params", **mp))
        print(f"{name:13s} K={K} sigma2={mp['sigma2']:.4g} delta_nn2={mp['delta_nn2']:.4g} b*={mp['bstar']:.2f}", flush=True)
        del ctx


if __name__ == "__main__":
    main()
