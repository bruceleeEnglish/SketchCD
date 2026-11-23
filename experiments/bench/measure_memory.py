"""Peak memory of each method (Exp-6), measured in a fresh process per (dataset, method).

Reported: extra = peak RSS - size of the input vectors (n * d * 4 bytes). Python baselines receive the vectors as a
NumPy array that is already loaded; SketchCD (a separate binary) reads them itself.

Usage: python measure_memory.py --datasets gist treccovid svhn --out runs/exp06_memory.jsonl
       python measure_memory.py child <dataset> <method>         (internal)
"""
import json
import math
import os
import resource
import subprocess
import sys

from common import RUNS, SKETCHCD_BIN, append_record, dataset_files, load_emeans, node_cpus

METHODS = ["SketchCD", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans", "EMeans-2", "FaissKMeans"]


def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def read_fvecs_lowmem(path):
    """Plain chunked reads into a preallocated array: the peak RSS of loading is the array itself (a memory map or
    np.fromfile would double it, and ru_maxrss would then report the loading peak instead of the method's)."""
    import numpy as np
    with open(path, "rb") as f:
        d = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    n = os.path.getsize(path) // (4 * (d + 1))
    X = np.empty((n, d), dtype=np.float32)
    rows = 8192
    buf = np.empty((rows, d + 1), dtype=np.float32)
    with open(path, "rb") as f:
        for s in range(0, n, rows):
            k = min(rows, n - s)
            f.readinto(buf[:k])
            X[s:s + k] = buf[:k, 1:]
    del buf
    return X


def child(name, method):
    import faiss
    faiss.omp_set_num_threads(20)
    X = read_fvecs_lowmem(dataset_files(name)[0])
    N, d = X.shape
    K = int(round(4 * math.sqrt(N)))
    after_load = rss_kb()
    if method.endswith("SuperKMeans"):
        import superkmeans
        kw = {"FullSuperKMeans": dict(hierarchical=False, sampling_fraction=1.0, early_termination=False, iters=10),
              "SampleSuperKMeans": dict(hierarchical=False), "HierSuperKMeans": dict(hierarchical=True),
              "QuantSuperKMeans": dict(hierarchical=True, quantizer="rabitq")}[method]
        km = superkmeans.SuperKMeans(n_clusters=K, dimensionality=d, n_threads=20, seed=42, **kw)
        C = km.train(X)
        km.quantized_assign(X, C) if method == "QuantSuperKMeans" else km.assign_training_points(X, C)
    elif method == "EMeans-2":
        load_emeans()(d, K, nepochs=2, seed=42).train(X)
    elif method == "FaissKMeans":
        km = faiss.Kmeans(d, K, niter=10, seed=42, verbose=False)
        km.train(X)
        faiss.knn(X, km.centroids, 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(peak_kb=peak, after_load_kb=after_load, data_kb=N * d * 4 // 1024)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--out", default=f"{RUNS}/exp06_memory.jsonl")
    a = ap.parse_args()
    cpus = node_cpus(a.node)
    env = dict(os.environ, OMP_NUM_THREADS="20", OPENBLAS_NUM_THREADS="20")
    for name in a.datasets:
        base = dataset_files(name)[0]
        with open(base, "rb") as f:
            d = int.from_bytes(f.read(4), "little")
        N = os.path.getsize(base) // (4 * (d + 1))
        K = int(round(4 * math.sqrt(N)))
        for m in a.methods:
            if m == "SketchCD":
                p = subprocess.Popen(["taskset", "-c", cpus, SKETCHCD_BIN, base, str(K), "-"], env=dict(env, OPENBLAS_NUM_THREADS="1"),
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                p.stdout.read()
                _, _, ru = os.wait4(p.pid, 0)
                r = dict(peak_kb=ru.ru_maxrss, after_load_kb=N * d * 4 // 1024, data_kb=N * d * 4 // 1024)
            else:
                o = subprocess.run(["taskset", "-c", cpus, sys.executable, os.path.abspath(__file__), "child", name, m], env=env,
                                   capture_output=True, text=True)
                lines = [ln for ln in o.stdout.splitlines() if ln.startswith("{")]
                if not lines:
                    print(name, m, "failed", o.stderr[-400:], flush=True)
                    continue
                r = json.loads(lines[-1])
            r.update(dataset=name, method=m, N=N, d=d, K=K, extra_gb=(r["peak_kb"] - r["data_kb"]) / 1024 ** 2,
                     data_gb=r["data_kb"] / 1024 ** 2)
            print(f"{name:8s} {m:18s} data {r['data_gb']:6.2f} GB  peak {r['peak_kb'] / 1024 ** 2:6.2f} GB  extra {r['extra_gb']:6.2f} GB", flush=True)
            append_record(a.out, r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
