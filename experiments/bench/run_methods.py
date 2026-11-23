"""Main benchmark. For every dataset / method / seed: build, time, save the centroids and lists, evaluate.

Methods (names of the paper):
  SketchCD, SketchCD-Fast   ours (bin/sketchcd; Fast = 4 sketch sweeps, 1 verified sweep, 64/128 candidates)
  SketchCD@key=value,...    SketchCD with these options, e.g. SketchCD@weighted=0 (SketchCD-NC, the nearest-centroid
                            rule in every stage) or SketchCD@sweeps=12,xsweeps=4 (a point of the trade-off curve)
  FullSuperKMeans           SuperKMeans 0.2.0, flat, all vectors, 10 iterations, no early termination
  SampleSuperKMeans         SuperKMeans 0.2.0, flat default (30% sample, 10 iterations, early termination)
  HierSuperKMeans           SuperKMeans 0.2.0, hierarchical default (sqrt(K) meso-clusters, 3 + 5 iterations)
  QuantSuperKMeans          HierSuperKMeans on RaBitQ codes
  EMeans-2, EMeans-5        e-means (reference implementation, CPU path), 2 / 5 epochs; lists = labels of the last epoch
  FaissKMeans               FAISS k-means, 10 iterations, then exact assignment
Every SuperKMeans variant is timed with both assignment paths (exact assign; the library's fast path); the faster one
defines build_s and its lists are evaluated and saved. Every method runs once per seed.

Metrics: Recall@k (k = 1, 10, 30, 50, 100) at equal fractions of scanned vectors (0.1% ... 10%) and at fixed nprobe,
WCSS of the lists relative to the reference partition, list-size statistics.

Usage: python run_methods.py --datasets gist fiqa --seeds 42 43 44 --out runs/exp01_main.jsonl
       python run_methods.py --datasets gist --kmult 8 --no-ref --methods SketchCD SketchCD@weighted=0 --out runs/exp09_list_size_sweep.jsonl
"""
import argparse
import math
import os
import time

import faiss
import numpy as np

from common import RUNS, SAVE, Ctx, append_record, evaluate_lists, load_emeans

ALL = ["SketchCD", "SketchCD-Fast", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans",
       "EMeans-2", "EMeans-5", "FaissKMeans"]
PARTS = ("basis_ms", "proj_ms", "pass_ms", "sweep_ms", "cent_ms", "ext_ms", "xsweep_ms", "fnn_ms")
FAST = dict(sweeps=4, xsweeps=1, candk=64, fcandk=128)  # SketchCD-Fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--faiss-seeds", type=int, nargs="+", default=[42], help="seeds for FaissKMeans (the slowest of the seven)")
    ap.add_argument("--node", type=int, default=1, help="NUMA node whose cores run the builds")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--kmult", type=float, default=4.0, help="K = kmult * sqrt(n)")
    ap.add_argument("--no-ref", action="store_true", help="skip the reference partition (no WCSS excess)")
    ap.add_argument("--ref-big", action="store_true", help="also compute the reference for n > 2M")
    ap.add_argument("--save", default=SAVE)
    ap.add_argument("--out", default=f"{RUNS}/exp01_main.jsonl")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    need_skm = any(m.endswith("SuperKMeans") for m in a.methods)
    need_em = any(m.startswith("EMeans-") for m in a.methods)
    if need_skm:
        import superkmeans
    NumpyEmeans = load_emeans() if need_em else None

    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 1000)
        X, N, d = ctx.X, ctx.N, ctx.d
        K = int(round(a.kmult * math.sqrt(N)))
        big = N > 2_000_000
        Jref = None
        if not (a.no_ref or (big and not a.ref_big)):
            Cref = ctx.reference(K)
            Jref = evaluate_lists(ctx, Cref, faiss.knn(X, Cref, 1)[1].ravel(), nearest=False)["wcss"]
        tag_ds = name if a.kmult == 4.0 else f"{name}-k{a.kmult:g}"
        os.makedirs(f"{a.save}/{tag_ds}", exist_ok=True)
        print(f"\n### {name}: N={N} d={d} K={K} Jref={Jref}", flush=True)

        def record(method, seed, build, parts, C, labels):
            C = np.ascontiguousarray(C, dtype=np.float32)
            labels = np.asarray(labels).astype(np.int32)
            np.save(f"{a.save}/{tag_ds}/{method}_s{seed}_C.npy", C)
            np.save(f"{a.save}/{tag_ds}/{method}_s{seed}_L.npy", labels)
            ev = evaluate_lists(ctx, C, labels, nearest=not big)
            ex = 100.0 * (ev["wcss"] - Jref) / Jref if Jref else float("nan")
            append_record(a.out, dict(dataset=tag_ds, N=N, d=d, K=K, method=method, seed=seed, build_s=build, parts=parts,
                                      Jref=Jref, excess=ex, **ev))
            r = lambda k, s="0.01": ev[f"R{k}_scan"][s]  # noqa: E731
            print(f"{method:18s} s{seed} build={build:8.2f}s excess={ex:6.2f}% R1={r(1):.4f} R10={r(10):.4f} R50={r(50):.4f} "
                  f"R100={r(100):.4f} @1% | cv={ev['cv']:.3f} tiny={100 * ev['tiny']:.2f}% empty={ev['empty']}", flush=True)

        def skm(method, seed, quantized=False, **kw):
            km = superkmeans.SuperKMeans(n_clusters=K, dimensionality=d, n_threads=a.threads, seed=seed, **kw)
            t0 = time.perf_counter()
            C = km.train(X)
            t1 = time.perf_counter()
            lab_exact = np.asarray(km.assign(X, C))
            t2 = time.perf_counter()
            lab_fast = np.asarray(km.quantized_assign(X, C) if quantized else km.assign_training_points(X, C))
            t3 = time.perf_counter()
            train, exact, fast = t1 - t0, t2 - t1, t3 - t2
            use_fast = fast < exact
            parts = dict(train_s=train, exact_assign_s=exact, fast_assign_s=fast, path="fast" if use_fast else "exact",
                         agree=float((lab_fast == lab_exact).mean()))
            record(method, seed, train + min(exact, fast), parts, C, lab_fast if use_fast else lab_exact)
            del km

        for seed in a.seeds:
            for m in a.methods:
                if m in ("SketchCD", "SketchCD-Fast") or m.startswith("SketchCD@"):
                    if m.startswith("SketchCD@"):  # the default with these options
                        opts = dict(kv.split("=") for kv in m.split("@", 1)[1].split(","))
                    else:
                        opts = {} if m == "SketchCD" else FAST
                    r, C, lab = ctx.sketchcd(K, seed=seed, **opts)
                    parts = {k: r[k] for k in PARTS}
                    parts.update(l=r["l"], ext=r["ext"], pilot=r["pilot_evals"])
                    record(m, seed, r["total_ms"] / 1000.0, parts, C, lab)
                elif m == "FullSuperKMeans":
                    skm(m, seed, hierarchical=False, sampling_fraction=1.0, early_termination=False, iters=10)
                elif m == "SampleSuperKMeans":
                    skm(m, seed, hierarchical=False)
                elif m == "HierSuperKMeans":
                    skm(m, seed, hierarchical=True)
                elif m == "QuantSuperKMeans":
                    try:
                        skm(m, seed, quantized=True, hierarchical=True, quantizer="rabitq")
                    except ValueError as e:  # e.g. "RaBitQ requires dimensionality <= 4096"
                        print(f"{m:18s} s{seed} NOT SUPPORTED: {e}", flush=True)
                elif m.startswith("EMeans-"):
                    em = NumpyEmeans(d, K, nepochs=int(m.split("-")[1]), seed=seed)
                    t0 = time.perf_counter()
                    C, lab = em.train(X)
                    t1 = time.perf_counter()
                    record(m, seed, t1 - t0, dict(train_s=t1 - t0), C, lab)
                    del em
                elif m == "FaissKMeans":
                    if seed not in a.faiss_seeds:
                        continue
                    km = faiss.Kmeans(d, K, niter=10, seed=seed, verbose=False)
                    t0 = time.perf_counter()
                    km.train(X)
                    t1 = time.perf_counter()
                    _, lab = faiss.knn(X, km.centroids, 1)
                    t2 = time.perf_counter()
                    record(m, seed, t2 - t0, dict(train_s=t1 - t0, assign_s=t2 - t1), km.centroids, lab.ravel())
                    del km
                else:
                    raise ValueError(f"unknown method {m!r}")
        del ctx, X


if __name__ == "__main__":
    main()
