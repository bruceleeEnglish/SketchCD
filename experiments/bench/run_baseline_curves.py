"""Trade-off curves of the competitors (Exp-2): every system at several budgets, each with its fastest official way of
obtaining the lists of all vectors. The curve of SketchCD comes from run_methods.py (methods SketchCD@sweeps=..,xsweeps=..).

  SuperKMeans (pip 0.2.0)
    skm-full-itN      flat, all data, N iterations, no early termination        (FullSuperKMeans at N = 10)
    skm-flat-default  flat, library defaults (30% sample, 10 iterations)        (SampleSuperKMeans)
    skm-hier          hierarchical, library defaults;  skm-hier-rN: N refinement iterations   (HierSuperKMeans)
    skm-rabitq-hier   hierarchical on RaBitQ codes;    skm-rabitq-full: flat, all data, 10 iterations on RaBitQ codes
    Every configuration is recorded twice: "<name>" = train + exact assign (lists = exact nearest centroid) and
    "<name>+fa" = train + the library's fast path (assign_training_points, or quantized_assign for RaBitQ), lists =
    the labels that path returns. The figures use the faster of the two.
  e-means (reference implementation, CPU path)   emeans-Nep, lists = labels of the last epoch
  FAISS k-means                                  faiss-itN, train + exact assign

Usage: python run_baseline_curves.py --datasets gist fiqa --seeds 42 --out runs/exp02_tradeoff_baselines.jsonl
"""
import argparse
import math
import time

import faiss
import numpy as np

from common import RUNS, Ctx, append_record, load_emeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--what", nargs="+", default=["skm", "emeans", "faiss"])
    ap.add_argument("--skm-iters", type=int, nargs="+", default=[1, 2, 3, 5, 10, 25])
    ap.add_argument("--skm-refine", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--emeans-epochs", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--faiss-iters", type=int, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--standard-only", action="store_true", help="only the standard configurations")
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--kmult", type=float, default=4.0)
    ap.add_argument("--no-ref", action="store_true", help="skip the reference partition (no WCSS excess)")
    ap.add_argument("--big-flat", action="store_true", help="also run the flat configurations for n > 3M")
    ap.add_argument("--out", default=f"{RUNS}/exp02_tradeoff_baselines.jsonl")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    if "skm" in a.what:
        import superkmeans
    NumpyEmeans = load_emeans() if "emeans" in a.what else None

    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 1000)
        X, N, d = ctx.X, ctx.N, ctx.d
        K = int(round(a.kmult * math.sqrt(N)))
        big = N > 3_000_000
        Jref = None if (big or a.no_ref) else ctx.evaluate(ctx.reference(K))["wcss_nc"]
        print(f"\n### {name}: N={N} d={d} K={K}", flush=True)

        def log(method, seed, build, parts, ev, which):
            append_record(a.out, dict(dataset=name, N=N, d=d, K=K, method=method, seed=seed, build_s=build, parts=parts,
                                      which=which, Jref=Jref, **ev))
            J = ev["wcss_lab"] if which == "lab" else ev["wcss_nc"]
            ex = 100.0 * (J - Jref) / Jref if Jref else float("nan")
            print(f"{method:22s} s{seed} build={build:8.2f}s excess={ex:6.2f}% R10@1%={ev[f'R10@0.01_{which}']:.4f} "
                  f"R100@2%={ev[f'R100@0.02_{which}']:.4f} cv={ev[f'cv_{which}']:.3f} tiny={100 * ev[f'tiny_{which}']:.2f}% "
                  f"{parts}", flush=True)

        def skm(tag, seed, quantized=False, **kw):
            try:
                km = superkmeans.SuperKMeans(n_clusters=K, dimensionality=d, n_threads=a.threads, seed=seed, **kw)
                t0 = time.perf_counter()
                C = km.train(X)
            except ValueError as e:  # e.g. "RaBitQ requires dimensionality <= 4096"
                print(f"   {tag}: NOT SUPPORTED: {e}", flush=True)
                return
            t1 = time.perf_counter()
            lab_exact = np.asarray(km.assign(X, C))
            t2 = time.perf_counter()
            fast, t_fast = None, float("nan")
            try:
                t3 = time.perf_counter()
                fast = np.asarray(km.quantized_assign(X, C) if quantized else km.assign_training_points(X, C))
                t_fast = time.perf_counter() - t3
            except Exception as e:  # noqa: BLE001
                print(f"   {tag}: fast assignment failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            C = np.ascontiguousarray(C, dtype=np.float32)
            train, exact = t1 - t0, t2 - t1
            if fast is not None:
                ev = ctx.evaluate(C, fast.astype(np.int64))
                agree = float((fast == lab_exact).mean())
                log(tag, seed, train + exact, dict(train_s=train, assign_s=exact), ev, "nc")
                log(tag + "+fa", seed, train + t_fast, dict(train_s=train, assign_s=t_fast, agree=agree), ev, "lab")
            else:
                log(tag, seed, train + exact, dict(train_s=train, assign_s=exact), ctx.evaluate(C), "nc")
            del km, C

        for seed in a.seeds:
            if "skm" in a.what:
                skm("skm-hier", seed, hierarchical=None)
                skm("skm-rabitq-hier", seed, quantized=True, hierarchical=None, quantizer="rabitq")
                if not (big and not a.big_flat):  # the flat configurations take 6-15 minutes each at 10M
                    skm("skm-flat-default", seed, hierarchical=False)
                    for it in ([10] if a.standard_only else a.skm_iters):
                        skm(f"skm-full-it{it}", seed, hierarchical=False, sampling_fraction=1.0, early_termination=False, iters=it)
                    if not a.standard_only:
                        for r in a.skm_refine:
                            skm(f"skm-hier-r{r}", seed, hierarchical=True, iters_refinement=r)
                        skm("skm-rabitq-full", seed, quantized=True, hierarchical=False, sampling_fraction=1.0,
                            early_termination=False, iters=10, quantizer="rabitq")
            if "emeans" in a.what:
                for ne in ([2, 5] if a.standard_only else a.emeans_epochs):
                    em = NumpyEmeans(d, K, nepochs=ne, seed=seed)
                    t0 = time.perf_counter()
                    C, lab = em.train(X)
                    t1 = time.perf_counter()
                    C = np.ascontiguousarray(C, dtype=np.float32)
                    ev = ctx.evaluate(C, np.asarray(lab).astype(np.int64))
                    log(f"emeans-{ne:g}ep", seed, t1 - t0, dict(train_s=t1 - t0), ev, "lab")
                    del em, C, lab
            if "faiss" in a.what:
                for it in ([10] if a.standard_only else a.faiss_iters):
                    km = faiss.Kmeans(d, K, niter=it, seed=seed, verbose=False)
                    t0 = time.perf_counter()
                    km.train(X)
                    t1 = time.perf_counter()
                    idx = faiss.IndexFlatL2(d)
                    idx.add(km.centroids)
                    idx.search(X, 1)
                    t2 = time.perf_counter()
                    ev = ctx.evaluate(np.ascontiguousarray(km.centroids, dtype=np.float32))
                    log(f"faiss-it{it}", seed, t2 - t0, dict(train_s=t1 - t0, assign_s=t2 - t1), ev, "nc")
        del ctx, X


if __name__ == "__main__":
    main()
