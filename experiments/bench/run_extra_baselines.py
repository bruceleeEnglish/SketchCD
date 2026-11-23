"""Three further baselines in the harness, metrics, and record format of run_methods.py.

  PCA-FaissKMeans   FaissKMeans (10 iterations, default sample) on a principal-component sketch of the same construction
                    and dimension l as that of SketchCD (randomized subspace iteration on 16,384 vectors, NumPy); lists
                    assigned in the sketch, centroids = full-dimensional means of the lists. Defined for d > 256.
  BatchedHartigan   the size-aware rule in every pass after the first of FullSuperKMeans (all vectors, 10 iterations, no
                    early termination; the moves of a pass are committed with the live list sizes), then the library's
                    fast assignment. Our implementation in a fork of SuperKMeans (baselines/batched_hartigan); its
                    size-aware pass needs d >= 128. With 30 iterations the same binary computes the reference partition.
  CDKM              exact sequential coordinate descent (Hartigan's method / CDKM, Nie et al. 2022): full dimension, all K
                    lists, 10 passes or until no vector moves (bin/cdkm). CDKM@p: p passes.
                    --cdkm-curve: datasets with CDKM for every seed plus CDKM@1..3 for seed 42 (Exp-12);
                    --cdkm-single: datasets with CDKM for seed 42 only.

Usage: python run_extra_baselines.py --datasets gist fiqa ... --seeds 42 43 44 --out runs/exp01_extra_baselines.jsonl
"""
import argparse
import json
import math
import os
import subprocess
import time

import faiss
import numpy as np

from common import ANALYSIS_DATASETS, BATCHED_HARTIGAN_BIN, CDKM_BIN, RUNS, SAVE, Ctx, append_record, evaluate_lists, read_fvecs


def sketch_dim(d):
    """The sketch dimension of SketchCD (src/sketchcd.cpp): 64 * round(d / 768), clamped to [128, 384]; none for d <= 256."""
    if d <= 256:
        return None
    return int(min(384, max(128, 64 * math.floor(d / 768 + 0.5))))  # round half up, as llround does


def rand_pca(X, l, seed, nsample=16384, over=8, power=2):
    """Randomized subspace iteration on a sample, as in SketchCD: mean mu and the top-l principal directions (l x d)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], min(nsample, X.shape[0]), replace=False))
    S = X[idx].astype(np.float32)
    mu = S.mean(axis=0)
    S -= mu
    Om = rng.standard_normal((S.shape[1], l + over)).astype(np.float32)
    Y = S @ Om
    for _ in range(power):
        Y, _ = np.linalg.qr(Y)
        Y = S @ (S.T @ Y)
    Qy, _ = np.linalg.qr(Y)
    Bm = Qy.T @ S  # (l+over) x d
    _, _, Vt = np.linalg.svd(Bm, full_matrices=False)
    return mu, np.ascontiguousarray(Vt[:l])


def full_means(X, lab, K):
    """Full-dimensional sums and sizes of the lists (chunked sparse one-hot product)."""
    import scipy.sparse as sp
    n, d = X.shape
    Ssum = np.zeros((K, d), dtype=np.float64)
    for s in range(0, n, 262144):
        e = min(n, s + 262144)
        M = sp.csr_matrix((np.ones(e - s, dtype=np.float32), (lab[s:e], np.arange(e - s))), shape=(K, e - s))
        Ssum += M @ X[s:e]
    cnt = np.bincount(lab, minlength=K).astype(np.float64)
    return Ssum, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--methods", nargs="+", default=["PCA-FaissKMeans", "BatchedHartigan", "CDKM"])
    ap.add_argument("--cdkm-curve", nargs="*", default=ANALYSIS_DATASETS)
    ap.add_argument("--cdkm-single", nargs="*", default=["fiqa", "cifar10", "cifar100"])
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--save", default=SAVE)
    ap.add_argument("--out", default=f"{RUNS}/exp01_extra_baselines.jsonl")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)

    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 1000)
        X, N, d = ctx.X, ctx.N, ctx.d
        K = int(round(4.0 * math.sqrt(N)))
        big = N > 2_000_000
        Cref = ctx.reference(K)  # cached reference partition (30 iterations)
        Jref = evaluate_lists(ctx, Cref, faiss.knn(X, Cref, 1)[1].ravel(), nearest=False)["wcss"]
        os.makedirs(f"{a.save}/{name}", exist_ok=True)
        print(f"\n### {name}: N={N} d={d} K={K} Jref={Jref}", flush=True)

        def record(method, seed, build, parts, C, labels):
            C = np.ascontiguousarray(C, dtype=np.float32)
            labels = np.asarray(labels).astype(np.int32)
            np.save(f"{a.save}/{name}/{method}_s{seed}_C.npy", C)
            np.save(f"{a.save}/{name}/{method}_s{seed}_L.npy", labels)
            ev = evaluate_lists(ctx, C, labels, nearest=not big)
            ex = 100.0 * (ev["wcss"] - Jref) / Jref
            append_record(a.out, dict(dataset=name, N=N, d=d, K=K, method=method, seed=seed, build_s=build, parts=parts,
                                      Jref=Jref, excess=ex, **ev))
            r = lambda k, s="0.01": ev[f"R{k}_scan"][s]  # noqa: E731
            print(f"{method:18s} s{seed} build={build:8.2f}s excess={ex:6.2f}% R10={r(10):.4f}@1% R10={r(10, '0.05'):.4f}@5% "
                  f"| cv={ev['cv']:.3f} tiny={100 * ev['tiny']:.2f}% empty={ev['empty']}", flush=True)

        methods = [m for m in a.methods if not m.startswith("CDKM")]
        if "CDKM" in a.methods and name in a.cdkm_curve:
            methods += ["CDKM", "CDKM@1", "CDKM@2", "CDKM@3"]
        elif "CDKM" in a.methods and name in a.cdkm_single:
            methods += ["CDKM"]
        for seed in a.seeds:
            for m in methods:
                try:
                    if m.startswith("CDKM"):
                        if seed != 42 and ("@" in m or name not in a.cdkm_curve):
                            continue
                        passes = int(m.split("@")[1]) if "@" in m else a.passes
                        cmd = ["taskset", "-c", ctx.pin, CDKM_BIN, ctx.base, str(K), ctx.tmp, str(passes), str(seed)]
                        p = subprocess.run(cmd, env=ctx.env(), capture_output=True, text=True)
                        if p.returncode != 0:
                            raise RuntimeError(p.stderr[-1500:])
                        r = json.loads(p.stdout.strip().splitlines()[-1])
                        C = read_fvecs(ctx.tmp + ".centroids.fvecs")
                        lab = np.fromfile(ctx.tmp + ".labels.u32", dtype=np.uint32)
                        os.remove(ctx.tmp + ".centroids.fvecs")
                        os.remove(ctx.tmp + ".labels.u32")
                        record(m, seed, r["total_ms"] / 1000.0, r, C, lab)
                    elif m == "BatchedHartigan":
                        cmd = ["taskset", "-c", ctx.pin, BATCHED_HARTIGAN_BIN, ctx.base, str(K), "10", "1", "0", ctx.tmp]
                        p = subprocess.run(cmd, env=ctx.env({"SKM_SEED": str(seed), "SKM_LABELS": "1"}),
                                           capture_output=True, text=True)
                        if p.returncode != 0:
                            raise RuntimeError(p.stderr[-1500:])
                        kv = {ln.split(",")[0]: ln.split(",")[1] for ln in p.stdout.splitlines()
                              if ln.startswith(("train_ms,", "assign_ms,"))}
                        moves = [int(ln.split(",")[8]) for ln in p.stdout.splitlines()
                                 if ln.split(",")[0].isdigit() and len(ln.split(",")) >= 10]
                        train, assign = float(kv["train_ms"]) / 1000.0, float(kv["assign_ms"]) / 1000.0
                        C = read_fvecs(ctx.tmp + ".centroids.fvecs")
                        lab = np.fromfile(ctx.tmp + ".labels.ivecs", dtype=np.int32).reshape(-1, 2)[:, 1]
                        os.remove(ctx.tmp + ".centroids.fvecs")
                        os.remove(ctx.tmp + ".labels.ivecs")
                        if not any(moves):  # d < 128: the library ran Lloyd iterations, this is not BatchedHartigan
                            print(f"{m:18s} s{seed} NOTE: no size-aware move in any iteration (d={d} < 128)", flush=True)
                        record(m, seed, train + assign, dict(train_s=train, assign_s=assign, moves=moves), C, lab)
                    elif m == "PCA-FaissKMeans":
                        l = sketch_dim(d)
                        if l is None:
                            print(f"{m:18s} s{seed} SKIPPED: d={d} <= 256 (SketchCD uses no sketch)", flush=True)
                            continue
                        t0 = time.perf_counter()
                        mu, Qd = rand_pca(X, l, seed)
                        t1 = time.perf_counter()
                        Y = np.empty((N, l), dtype=np.float32)
                        QT = np.ascontiguousarray(Qd.T)
                        ch = max(4096, 2 ** 27 // d)  # 512 MB temporaries
                        for s in range(0, N, ch):
                            Y[s:s + ch] = (X[s:s + ch] - mu) @ QT
                        t2 = time.perf_counter()
                        km = faiss.Kmeans(l, K, niter=10, seed=seed, verbose=False)
                        km.train(Y)
                        t3 = time.perf_counter()
                        _, lab = faiss.knn(Y, km.centroids, 1)
                        lab = lab.ravel()
                        t4 = time.perf_counter()
                        Ssum, cnt = full_means(X, lab, K)
                        lifted = mu + km.centroids @ Qd  # only for lists that end empty
                        C = np.where(cnt[:, None] > 0, Ssum / np.maximum(cnt, 1)[:, None], lifted).astype(np.float32)
                        t5 = time.perf_counter()
                        del Y
                        record(m, seed, t5 - t0, dict(pca_s=t1 - t0, proj_s=t2 - t1, train_s=t3 - t2, assign_s=t4 - t3,
                                                      means_s=t5 - t4, l=l), C, lab)
                    else:
                        raise ValueError(m)
                except Exception as e:  # keep going; the log records the failure
                    print(f"{m:18s} s{seed} FAILED: {e!r}", flush=True)
        del ctx, X


if __name__ == "__main__":
    main()
