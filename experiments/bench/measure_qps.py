"""Query throughput of the built indexes (Exp-4).

The centroids and lists of every method (saved by run_methods.py) are loaded into the same FAISS IndexIVFFlat, and 1000
queries are searched with k = 100 at increasing nprobe on one thread. For every nprobe we report
  qps    queries per second, best of --reps runs
  ndis   vectors scanned per query (FAISS indexIVF_stats.ndis; deterministic, independent of timing noise)
  R1/R10/R100 against the exact neighbours.
The base vectors are streamed from the fvecs file in chunks and dropped from the page cache, so only the index itself is
resident; --bind-node binds all allocations to one NUMA node.

Usage: taskset -c 0 python measure_qps.py --bind-node 0 --reps 2 --seed 42 --datasets gist fiqa \
           --methods SketchCD FullSuperKMeans HierSuperKMeans QuantSuperKMeans EMeans-5 FaissKMeans SampleSuperKMeans --out runs/exp04_qps.jsonl
"""
import argparse
import os
import time

import faiss
import numpy as np

from common import RUNS, SAVE, append_record, dataset_files, fvecs_shape, read_fvecs, read_ivecs

NPROBES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def bind_memory(node):
    """MPOL_BIND all future allocations of this process to one NUMA node (the kernel then reclaims page cache on that node
    instead of spilling to the other socket, which would disturb timings there)."""
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    mask = ctypes.c_ulong(1 << node)
    if libc.syscall(238, 2, ctypes.byref(mask), ctypes.c_ulong(65)) != 0:  # set_mempolicy(MPOL_BIND, &mask, maxnode)
        print("set_mempolicy failed, errno", ctypes.get_errno(), flush=True)


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=SAVE)
    ap.add_argument("--max-nprobe", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--bind-node", type=int, default=None)
    ap.add_argument("--out", default=f"{RUNS}/exp04_qps.jsonl")
    a = ap.parse_args()
    if a.bind_node is not None:
        bind_memory(a.bind_node)
    for name in a.datasets:
        base, qp, gp = dataset_files(name)
        drop_cache(base)
        N, d = fvecs_shape(base)
        Q = np.ascontiguousarray(read_fvecs(qp, 1000))
        gt = read_ivecs(gp, 1000)[:, :100]
        print(f"\n### {name}: N={N} d={d}", flush=True)
        for m in a.methods:
            fc, fl = f"{a.save}/{name}/{m}_s{a.seed}_C.npy", f"{a.save}/{name}/{m}_s{a.seed}_L.npy"
            if not os.path.exists(fc):
                print("missing", fc, flush=True)
                continue
            C = np.ascontiguousarray(np.load(fc), dtype=np.float32)
            L = np.ascontiguousarray(np.load(fl).astype(np.int64))
            assert L.shape[0] == N
            K = C.shape[0]
            faiss.omp_set_num_threads(20)
            quant = faiss.IndexFlatL2(d)
            quant.add(C)
            index = faiss.IndexIVFFlat(quant, d, K)
            index.is_trained = True
            fd = os.open(base, os.O_RDONLY)
            row = d + 1
            with open(base, "rb") as fh:
                for i in range(0, N, a.chunk):
                    j = min(N, i + a.chunk)
                    buf = np.fromfile(fh, dtype=np.float32, count=(j - i) * row).reshape(j - i, row)
                    xc = np.ascontiguousarray(buf[:, 1:])
                    index.add_core(j - i, faiss.swig_ptr(xc), None, faiss.swig_ptr(L[i:j]))
                    os.posix_fadvise(fd, 4 * i * row, 4 * (j - i) * row, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            assert index.ntotal == N
            drop_cache(base)  # the index holds the only resident copy of the data
            faiss.omp_set_num_threads(1)
            index.search(Q[:50], 100)  # warm-up
            rows = []
            for npb in NPROBES:
                if npb > a.max_nprobe or npb > K:
                    continue
                index.nprobe = npb
                best = 1e9
                for _ in range(a.reps):
                    faiss.cvar.indexIVF_stats.reset()
                    t0 = time.perf_counter()
                    _, I = index.search(Q, 100)
                    best = min(best, time.perf_counter() - t0)
                ndis = faiss.cvar.indexIVF_stats.ndis / len(Q)
                rec = {f"R{k}": float(np.mean([len(set(I[i, :k]) & set(gt[i, :k])) / k for i in range(len(Q))])) for k in (1, 10, 100)}
                rows.append(dict(nprobe=npb, qps=len(Q) / best, ndis=ndis, **rec))
                print(f"{m:18s} nprobe={npb:4d} QPS={len(Q) / best:9.1f} ndis={ndis:9.0f} R1={rec['R1']:.4f} R10={rec['R10']:.4f} "
                      f"R100={rec['R100']:.4f}", flush=True)
            append_record(a.out, dict(dataset=name, method=m, seed=a.seed, N=N, d=d, K=K, reps=a.reps, curve=rows))
            del index, quant


if __name__ == "__main__":
    main()
