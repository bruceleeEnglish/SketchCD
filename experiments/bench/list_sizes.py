"""List-size distributions of the saved indexes (seed 42).

For every dataset and method: sizes of the K lists relative to the mean size n/K, summarized by quantiles, the share
of tiny lists (< 0.1 of the mean), empty lists, and the coefficient of variation. Reads the lists saved by
run_methods.py (<method>_s42_L.npy) and the centroids (<method>_s42_C.npy, for K).

Usage: python list_sizes.py --out runs/exp09_list_sizes.jsonl
"""
import argparse
import json
import os

import numpy as np

from common import PAPER_DATASETS, RUNS, SAVE

METHODS = ["SketchCD", "SketchCD@weighted=0", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans",
           "EMeans-2", "EMeans-5", "FaissKMeans"]
QS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", nargs="+", default=[SAVE], help="folders with the saved builds; the first match wins")
    ap.add_argument("--out", default=f"{RUNS}/exp09_list_sizes.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for ds in PAPER_DATASETS:
            for m in METHODS:
                src = next((d for d in a.save if os.path.exists(f"{d}/{ds}/{m}_s42_L.npy")), None)
                if src is None:
                    print(f"{ds:13s} {m:20s} missing", flush=True)
                    continue
                L = np.load(f"{src}/{ds}/{m}_s42_L.npy").astype(np.int64)
                K = np.load(f"{src}/{ds}/{m}_s42_C.npy", mmap_mode="r").shape[0]
                sizes = np.bincount(L, minlength=K).astype(np.float64)
                rel = sizes / sizes.mean()
                rec = dict(dataset=ds, method=m, source=os.path.basename(src), n=int(L.size), K=int(K), q=QS,
                           rel_quantiles=[float(np.quantile(rel, q)) for q in QS], tiny=float((rel < 0.1).mean()),
                           empty=int((sizes == 0).sum()), cv=float(rel.std()))
                f.write(json.dumps(rec) + "\n")
                print(f"{ds:13s} {m:20s} K={K:6d} tiny={100 * rec['tiny']:5.2f}% cv={rec['cv']:.3f}", flush=True)


if __name__ == "__main__":
    main()
