"""Scale experiment (Exp-5): one method on BIGANN-100M, in its own process.

Every method uses the 20 threads of one socket (taskset, as in the main protocol); the memory policy interleaves pages
over both NUMA nodes, because 100M x 128 floats (51 GB) fill most of one node (64 GB) -- the same policy for every method.
The dataset is not loaded for SketchCD / BatchedHartigan (their binaries read the file); quality is evaluated afterwards
from a memory map. Metrics and record format as run_methods.py (no reference partition: excess is NaN, and the WCSS is
compared between methods). Driver: run_100m.py (watchdog for swap and time).

Usage: python run_100m_method.py --method SketchCD --seed 42 --out runs/exp05_bigann100m.jsonl
"""
import argparse
import ctypes
import math
import os
import subprocess
import time

import numpy as np

from common import BATCHED_HARTIGAN_BIN, RUNS, SAVE, SCALE_DATASET, SKETCHCD_BIN, TMP, append_record, dataset_files, node_cpus

PARTS = ("basis_ms", "proj_ms", "pass_ms", "sweep_ms", "cent_ms", "ext_ms", "xsweep_ms", "fnn_ms")


def interleave_memory():
    """set_mempolicy(MPOL_INTERLEAVE, nodes {0,1}); inherited by child processes (syscall 238 on x86_64)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    mask = ctypes.c_ulong(0b11)
    if libc.syscall(238, 3, ctypes.byref(mask), ctypes.c_ulong(64)) != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")


class LazyCtx:
    """What evaluate_lists needs: X (memory map or array), Q, gt."""

    def __init__(self, base, qpath, gpath, load):
        from common import read_fvecs, read_ivecs
        self.Q = read_fvecs(qpath, 1000)
        self.gt = read_ivecs(gpath, 1000)[:, :100]
        mm = np.memmap(base, dtype=np.float32, mode="r")
        d = int(mm[:1].view(np.int32)[0])
        self.N, self.d = mm.shape[0] // (d + 1), d
        self.X = read_fvecs(base) if load else mm.reshape(self.N, d + 1)[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=SCALE_DATASET)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--save", default=SAVE)
    ap.add_argument("--out", default=f"{RUNS}/exp05_bigann100m.jsonl")
    a = ap.parse_args()
    interleave_memory()
    os.environ.update(OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads))
    import faiss
    from common import evaluate_lists, load_emeans, read_fvecs
    faiss.omp_set_num_threads(a.threads)

    name, m, seed = a.dataset, a.method, a.seed
    base, qpath, gpath = dataset_files(name)
    pin = node_cpus(a.node)
    env = dict(os.environ)
    os.makedirs(TMP, exist_ok=True)
    tmp = f"{TMP}/s100_{name}_{os.getpid()}"
    binary = m.startswith("SketchCD") or m == "BatchedHartigan"
    ctx = LazyCtx(base, qpath, gpath, load=not binary)
    N, d = ctx.N, ctx.d
    K = int(round(4.0 * math.sqrt(N)))
    os.makedirs(f"{a.save}/{name}", exist_ok=True)
    print(f"### {name}: N={N} d={d} K={K} method={m} seed={seed}", flush=True)

    def run(cmd, extra=None):
        p = subprocess.run(["taskset", "-c", pin] + cmd, env=dict(env, **(extra or {})), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        return p.stdout

    if m.startswith("SketchCD"):
        import json
        opts = {} if m == "SketchCD" else dict(sweeps=4, xsweeps=1, candk=64, fcandk=128)
        out = run([SKETCHCD_BIN, base, str(K), tmp, f"seed={seed}"] + [f"{k}={v}" for k, v in opts.items()])
        r = json.loads(out.strip().splitlines()[-1])
        C = read_fvecs(tmp + ".centroids.fvecs")
        lab = np.fromfile(tmp + ".labels.u32", dtype=np.uint32)
        os.remove(tmp + ".centroids.fvecs")
        os.remove(tmp + ".labels.u32")
        build = r["total_ms"] / 1000.0
        parts = {k: r[k] for k in PARTS if k in r}
    elif m == "BatchedHartigan":
        out = run([BATCHED_HARTIGAN_BIN, base, str(K), "10", "1", "0", tmp], {"SKM_SEED": str(seed), "SKM_LABELS": "1"})
        kv = {ln.split(",")[0]: float(ln.split(",")[1]) for ln in out.splitlines() if ln.startswith(("train_ms,", "assign_ms,"))}
        C = read_fvecs(tmp + ".centroids.fvecs")
        lab = np.fromfile(tmp + ".labels.ivecs", dtype=np.int32).reshape(-1, 2)[:, 1]
        os.remove(tmp + ".centroids.fvecs")
        os.remove(tmp + ".labels.ivecs")
        build = (kv["train_ms"] + kv["assign_ms"]) / 1000.0
        parts = dict(train_s=kv["train_ms"] / 1000.0, assign_s=kv["assign_ms"] / 1000.0)
    else:
        os.sched_setaffinity(0, {int(c) for part in pin.split(",") for c in
                                 (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
        X = ctx.X
        if m.endswith("SuperKMeans"):
            import superkmeans
            kw = dict(FullSuperKMeans=dict(hierarchical=False, sampling_fraction=1.0, early_termination=False, iters=10),
                      SampleSuperKMeans=dict(hierarchical=False), HierSuperKMeans=dict(hierarchical=True),
                      QuantSuperKMeans=dict(hierarchical=True, quantizer="rabitq"))[m]
            km = superkmeans.SuperKMeans(n_clusters=K, dimensionality=d, n_threads=a.threads, seed=seed, **kw)
            t0 = time.perf_counter()
            C = km.train(X)
            t1 = time.perf_counter()
            # the library's fast path
            lab = np.asarray(km.quantized_assign(X, C) if m == "QuantSuperKMeans" else km.assign_training_points(X, C))
            t2 = time.perf_counter()
            build, parts = t2 - t0, dict(train_s=t1 - t0, fast_assign_s=t2 - t1, path="fast")
        elif m.startswith("EMeans-"):
            em = load_emeans()(d, K, nepochs=int(m.split("-")[1]), seed=seed)
            t0 = time.perf_counter()
            C, lab = em.train(X)
            build = time.perf_counter() - t0
            parts = dict(train_s=build)
        elif m == "FaissKMeans":
            km = faiss.Kmeans(d, K, niter=10, seed=seed, verbose=False)
            t0 = time.perf_counter()
            km.train(X)
            t1 = time.perf_counter()
            _, lab = faiss.knn(X, km.centroids, 1)
            t2 = time.perf_counter()
            C, lab, build, parts = km.centroids, lab.ravel(), t2 - t0, dict(train_s=t1 - t0, assign_s=t2 - t1)
        else:
            raise ValueError(m)

    C = np.ascontiguousarray(C, dtype=np.float32)
    lab = np.asarray(lab).astype(np.int32)
    np.save(f"{a.save}/{name}/{m}_s{seed}_C.npy", C)
    np.save(f"{a.save}/{name}/{m}_s{seed}_L.npy", lab)
    if binary:  # evaluate from the page cache / disk
        ctx.X = np.memmap(base, dtype=np.float32, mode="r").reshape(N, d + 1)[:, 1:]
    ev = evaluate_lists(ctx, C, lab, nearest=False)
    append_record(a.out, dict(dataset=name, N=N, d=d, K=K, method=m, seed=seed, build_s=build, parts=parts, Jref=None,
                              excess=float("nan"), **ev))
    r = lambda k, s: ev[f"R{k}_scan"][s]  # noqa: E731
    print(f"{m:18s} s{seed} build={build:8.2f}s wcss={ev['wcss']:.6e} R10={r(10, '0.001'):.4f}@0.1% "
          f"R10={r(10, '0.005'):.4f}@0.5% R100={r(100, '0.001'):.4f}@0.1% | cv={ev['cv']:.3f} tiny={100 * ev['tiny']:.2f}% "
          f"empty={ev['empty']}", flush=True)


if __name__ == "__main__":
    main()
