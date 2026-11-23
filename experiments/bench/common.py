"""Shared pieces of the benchmark harness: locations, datasets, file formats, evaluation, runners of the binaries.

Locations (every one can be set through an environment variable of the same name):
  SKETCHCD_ROOT   the repository (default: two levels above this folder)
  SKETCHCD_DATA   datasets, one folder per dataset: <name>/<name>_base.fvecs, _query.fvecs, _gt.ivecs  (default: data/)
  SKETCHCD_RUNS   records written by the benchmark scripts, one JSON line per run                      (default: runs/)
  SKETCHCD_SAVE   centroids and lists of every build (<dataset>/<method>_s<seed>_{C,L}.npy)            (default: outputs/)
  SKETCHCD_TMP    scratch files and the cached reference partitions                                    (default: tmp/)
  SKETCHCD_BIN, CDKM_BIN, BATCHED_HARTIGAN_BIN   the three binaries (default: bin/, where build.sh puts them)
  EMEANS_PATH     folder that contains the e_means package (default: baselines/, which holds a copy of its CPU path)

All timed runs are pinned to the cores of one NUMA node with taskset, so the harness needs Linux.
"""
import json
import os
import subprocess
import sys

import faiss
import numpy as np

ROOT = os.environ.get("SKETCHCD_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("SKETCHCD_DATA", f"{ROOT}/data")
RUNS = os.environ.get("SKETCHCD_RUNS", f"{ROOT}/runs")
SAVE = os.environ.get("SKETCHCD_SAVE", f"{ROOT}/outputs")
TMP = os.environ.get("SKETCHCD_TMP", f"{ROOT}/tmp")
SKETCHCD_BIN = os.environ.get("SKETCHCD_BIN", f"{ROOT}/bin/sketchcd")
CDKM_BIN = os.environ.get("CDKM_BIN", f"{ROOT}/bin/cdkm")
BATCHED_HARTIGAN_BIN = os.environ.get("BATCHED_HARTIGAN_BIN", f"{ROOT}/bin/batched_hartigan")
EMEANS_PATH = os.environ.get("EMEANS_PATH", f"{ROOT}/baselines")

# The datasets of the paper (Table 2) and the sets derived from them for the scalability experiment (Exp-5).
PAPER_DATASETS = ["gist", "fiqa", "treccovid", "dbp3s", "cifar10", "cifar100", "svhn", "tinyimagenet", "bigann10m", "deep10m"]
SCALE_DATASET = "bigann100m"
DERIVED_DATASETS = ["tiny8", "tiny16", "tiny32", "deep500k", "deep1m", "deep2m", "deep5m"]
ANALYSIS_DATASETS = ["gist", "treccovid", "svhn"]  # one dataset of each kind (Exp-6 to Exp-8, Exp-11, Exp-12)
ALL_DATASETS = PAPER_DATASETS + [SCALE_DATASET] + DERIVED_DATASETS


def dataset_files(name):
    """(base, queries, ground truth) of a dataset prepared by experiments/prepare_data.py."""
    if name not in ALL_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(ALL_DATASETS)}")
    return tuple(f"{DATA}/{name}/{name}_{s}" for s in ("base.fvecs", "query.fvecs", "gt.ivecs"))


def node_cpus(node):
    """The cpu list of a NUMA node in the form that taskset takes."""
    return open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()


def load_emeans():
    """NumpyEmeans, the CPU path of the reference implementation of e-means (baselines/e_means, see NOTICE.md there)."""
    if EMEANS_PATH not in sys.path:
        sys.path.insert(0, EMEANS_PATH)
    try:
        from e_means import NumpyEmeans
    except ImportError as e:
        raise ImportError(f"the e_means package was not found under {EMEANS_PATH}; see baselines/README.md") from e
    return NumpyEmeans


# ------------------------------------------------------------------ file formats
def read_fvecs(path, limit=None):
    if os.path.getsize(path) > 8 * 2 ** 30:  # large file: copy in chunks from a memory map (no 2x peak)
        mm = np.memmap(path, dtype=np.int32, mode="r")
        d = int(mm[0])
        m = mm.reshape(-1, d + 1)
        n = min(limit, len(m)) if limit else len(m)
        out = np.empty((n, d), dtype=np.float32)
        for s in range(0, n, 262144):
            out[s:s + 262144] = m[s:min(n, s + 262144), 1:].view(np.float32)
        del m, mm
        return out
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    m = a.reshape(-1, d + 1)
    if limit:
        m = m[:limit]
    return np.ascontiguousarray(m[:, 1:]).view(np.float32)


def read_ivecs(path, limit=None):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    m = a.reshape(-1, d + 1)
    if limit:
        m = m[:limit]
    return np.ascontiguousarray(m[:, 1:]).astype(np.int64)


def fvecs_shape(path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    return os.path.getsize(path) // (4 * (d + 1)), d


def append_record(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ quality of an IVF index (records of the main benchmark)
KS = (1, 10, 30, 50, 100)
SCAN = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
NPROBE = (1, 2, 4, 8, 16, 32, 64, 128)


def evaluate_lists(ctx, C, labels, nearest=True):
    """WCSS of the lists, list-size statistics, and Recall@k at equal fractions of scanned vectors and at fixed nprobe.

    The lists are those of `labels` (every method is evaluated with its own lists). Probing follows the centroid
    distance; recall at a scan budget is interpolated between probes.
    """
    X, Q, gt = ctx.X, ctx.Q, ctx.gt
    N, d = X.shape
    K = C.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    index = faiss.IndexFlatL2(d)
    index.add(C)
    out = {}
    w = 0.0
    for s in range(0, N, 65536):
        diff = X[s:s + 65536] - C[labels[s:s + 65536]]
        w += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
    out["wcss"] = w
    if nearest:
        _, nn = index.search(X, 1)
        out["label_ne_nearest"] = float((labels != nn.ravel()).mean())
    sizes = np.bincount(labels, minlength=K)
    out["cv"] = float(sizes.std() / sizes.mean())
    out["empty"] = int((sizes == 0).sum())
    out["tiny"] = float((sizes < 0.1 * N / K).mean())
    out["max_over_mean"] = float(sizes.max() / sizes.mean())
    Pn = min(K, max(16, int(round(0.3 * K))))
    _, probe = index.search(Q, Pn)
    scan = np.cumsum(sizes[probe], axis=1).mean(axis=0) / N
    out["scan_at_nprobe"] = {str(p): float(scan[p - 1]) for p in NPROBE if p <= Pn}
    for k in KS:
        lists = labels[gt[:, :k]]
        rank = np.full(lists.shape, Pn, dtype=np.int32)
        for j in range(k):
            hit = probe == lists[:, j][:, None]
            rank[:, j] = np.where(hit.any(axis=1), hit.argmax(axis=1), Pn)
        rec = np.cumsum(np.bincount(rank.ravel(), minlength=Pn + 1)[:Pn]) / rank.size
        out[f"R{k}_scan"] = {str(s): float(np.interp(s, scan, rec)) for s in SCAN}
        out[f"R{k}_nprobe"] = {str(p): float(rec[p - 1]) for p in NPROBE if p <= Pn}
    return out


class Ctx:
    """One dataset in memory, and the runners of the three binaries (each run in a fresh, pinned process)."""

    def __init__(self, name, node, threads, nq):
        base, qp, gp = dataset_files(name)
        self.name, self.base = name, base
        self.X = read_fvecs(base)
        self.Q = read_fvecs(qp, nq)
        self.gt = read_ivecs(gp, nq)[:, :100]
        self.N, self.d = self.X.shape
        self.pin = node_cpus(node)
        self.threads = threads
        os.makedirs(TMP, exist_ok=True)
        self.tmp = f"{TMP}/run_{name}_{os.getpid()}"

    def env(self, extra=None):
        e = dict(os.environ, OMP_NUM_THREADS=str(self.threads), OPENBLAS_NUM_THREADS=str(self.threads))
        e.update(extra or {})
        return e

    # ---------- quality in the record format of the analysis experiments and of the trade-off curves ----------
    def evaluate(self, C, labels=None):
        """Quality of the lists given by `labels` (suffix _lab) and of the nearest-centroid lists of C (suffix _nc)."""
        X, Q, gt = self.X, self.Q, self.gt
        N, k = X.shape[0], C.shape[0]
        index = faiss.IndexFlatL2(C.shape[1])
        index.add(C)
        out = {}
        d2, nn = index.search(X, 1)
        nn = nn.ravel()
        out["wcss_nc"] = float(d2.sum(dtype=np.float64))
        np1 = max(1, round(0.01 * k))
        Pn = min(k, max(np1, int(round(0.15 * k))))
        _, probe = index.search(Q, Pn)
        modes = {"nc": nn}
        if labels is not None:
            modes["lab"] = labels.astype(np.int64)
            out["label_ne_nearest"] = float((labels != nn).mean())
            w = 0.0
            for s in range(0, N, 65536):
                diff = X[s:s + 65536] - C[labels[s:s + 65536]]
                w += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
            out["wcss_lab"] = w
        for tag, ivf in modes.items():
            sizes = np.bincount(ivf, minlength=k)
            out[f"cv_{tag}"] = float(sizes.std() / sizes.mean())
            out[f"empty_{tag}"] = int((sizes == 0).sum())
            out[f"tiny_{tag}"] = float((sizes < 0.1 * N / k).mean())
            scan = np.cumsum(sizes[probe], axis=1).mean(axis=0) / N
            for Kt in (10, 100):
                lists = ivf[gt[:, :Kt]]
                rank = np.full(lists.shape, Pn, dtype=np.int32)
                for j in range(Kt):
                    hit = probe == lists[:, j][:, None]
                    rank[:, j] = np.where(hit.any(axis=1), hit.argmax(axis=1), Pn)
                rec = np.cumsum(np.bincount(rank.ravel(), minlength=Pn + 1)[:Pn]) / rank.size
                for sfr in SCAN:  # scan budgets (fraction of vectors)
                    out[f"R{Kt}@{sfr}_{tag}"] = float(np.interp(sfr, scan, rec))
                out[f"scan@R{Kt}=.9_{tag}"] = float(np.interp(0.9, rec, scan)) if rec[-1] >= 0.9 else float("nan")
        return out

    # ---------- runners ----------
    def sketchcd(self, K, **opts):
        """One build with bin/sketchcd; returns (the JSON line of the binary, centroids, labels)."""
        args = ["taskset", "-c", self.pin, SKETCHCD_BIN, self.base, str(K), self.tmp] + [f"{k}={v}" for k, v in opts.items()]
        p = subprocess.run(args, env=self.env(), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        r = json.loads(p.stdout.strip().splitlines()[-1])
        C = read_fvecs(self.tmp + ".centroids.fvecs")
        lab = np.fromfile(self.tmp + ".labels.u32", dtype=np.uint32)
        os.remove(self.tmp + ".centroids.fvecs")
        os.remove(self.tmp + ".labels.u32")
        return r, C, lab

    def batched_hartigan(self, K, iters, size_aware=1, extra=None):
        """`iters` iterations of the SuperKMeans fork over all vectors; returns (times, centroids)."""
        cmd = ["taskset", "-c", self.pin, BATCHED_HARTIGAN_BIN, self.base, str(K), str(iters), str(size_aware), "0", self.tmp]
        p = subprocess.run(cmd, env=self.env(extra), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        r = {"iter_ms": []}
        for line in p.stdout.splitlines():
            parts = line.split(",")
            if parts[0] == "train_ms":
                r["train_ms"] = float(parts[1])
            elif parts[0].isdigit() and len(parts) >= 10:
                r["iter_ms"].append(float(parts[9]))
        C = read_fvecs(self.tmp + ".centroids.fvecs")
        os.remove(self.tmp + ".centroids.fvecs")
        return r, C

    def reference(self, K, iters=30):
        """Centroids of the reference partition of the WCSS excess: `iters` iterations of BatchedHartigan (cached).

        The size-aware pass of the fork needs d >= 128; below that the library runs Lloyd iterations (DEEP-10M).
        """
        path = f"{TMP}/ref_{self.name}_{K}_{iters}.npy"
        if os.path.exists(path):
            return np.load(path)
        _, C = self.batched_hartigan(K, iters, 1)
        np.save(path, C)
        return C
