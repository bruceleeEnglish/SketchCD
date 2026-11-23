"""Exactness of certified verification (Exp-10): for every dataset, the verified stages with pruning (prune=1) against
the same stages evaluating every candidate in full dimension (prune=0). The sketch extension is fixed to the value that
the default pilot chooses on that dataset, so both runs have the same sketch and the same candidate lists.

Recorded: labels that differ, max |centroid difference|, full-dimensional distances per vector in the verified sweeps
and in the final relabeling, and the time of those two stages (best of --reps).

Usage: python check_exactness.py --datasets gist fiqa treccovid dbp3s cifar10 cifar100 svhn tinyimagenet --out runs/exp10_exactness.jsonl
"""
import argparse
import math

import numpy as np

from common import RUNS, Ctx, append_record


def stage_s(r):
    return (r["xsweep_ms"] + r["fnn_ms"]) / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"{RUNS}/exp10_exactness.jsonl")
    a = ap.parse_args()
    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 10)
        N, d = ctx.N, ctx.d
        K = int(round(4 * math.sqrt(N)))
        r_def, _, _ = ctx.sketchcd(K, seed=a.seed)
        E = int(r_def["ext"])
        best = {}
        for prune in (1, 0):
            for _ in range(a.reps):
                r, C, lab = ctx.sketchcd(K, seed=a.seed, prune=prune, ext=E)
                if prune not in best or stage_s(r) < stage_s(best[prune][0]):
                    best[prune] = (r, C, lab)
        (r1, C1, L1), (r0, C0, L0) = best[1], best[0]
        rec = dict(dataset=name, N=N, d=d, K=K, seed=a.seed, l=int(r1["l"]), ext=E, pilot=r_def["pilot_evals"],
                   labels_differ=int((L1 != L0).sum()), max_cdiff=float(np.abs(C1 - C0).max()),
                   xev_pruned=r1["xevals_per_point"], fev_pruned=r1["fevals_per_point"],
                   xev_all=r0["xevals_per_point"], fev_all=r0["fevals_per_point"],
                   stage_pruned_s=stage_s(r1), stage_all_s=stage_s(r0), total_pruned_s=r1["total_ms"] / 1e3,
                   total_all_s=r0["total_ms"] / 1e3, total_default_s=r_def["total_ms"] / 1e3)
        print(f"{name:14s} [ext={E}] l={rec['l']} labels differ on {rec['labels_differ']} of {N}, max |centroid diff| "
              f"{rec['max_cdiff']:.1e} | verified stages: pruned {rec['stage_pruned_s']:.2f}s ({rec['xev_pruned']:.1f} / "
              f"{rec['fev_pruned']:.1f} per vector), all {rec['stage_all_s']:.2f}s ({rec['xev_all']:.0f} / {rec['fev_all']:.0f}), "
              f"{rec['stage_all_s'] / rec['stage_pruned_s']:.2f}x", flush=True)
        append_record(a.out, rec)
        del ctx


if __name__ == "__main__":
    main()
