"""Download and prepare the datasets of the paper. Everything is written under $SKETCHCD_DATA/<name>/ (default: data/):
<name>_base.fvecs, <name>_query.fvecs (1000 queries), <name>_gt.ivecs (exact L2 top-100, computed here), INFO.json.
Raw downloads are deleted after conversion.

  gist           GIST1M of the TEXMEX corpus, its 1000 public queries
  fiqa           FiQA corpus of BEIR, Cohere embed-english-v3 (1024-d)                    L2-normalised
  treccovid      TREC-COVID corpus of BEIR, Cohere embed-english-v3 (1024-d)              L2-normalised
  dbp3s          DBpedia entities, OpenAI text-embedding-3-small (1536-d, 100K)           L2-normalised
  cifar10        CIFAR-10 train + test, raw pixels (32x32x3)
  cifar100       CIFAR-100 train + test, raw pixels (32x32x3)
  svhn           SVHN cropped digits train + test + extra, raw pixels (32x32x3)
  tinyimagenet   Tiny-ImageNet train + validation, raw pixels (64x64x3)
  bigann10m      the first 10M vectors of BIGANN (uint8 -> float32), the first 1000 public queries
  deep10m        deep-image-96 of ANN-Benchmarks (10M vectors of DEEP1B, 10,000 of them held out as queries)   L2-normalised
  bigann100m     the first 100M vectors of BIGANN, the first 1000 public queries (scale experiment)
  tiny8, tiny16, tiny32             Tiny-ImageNet averaged over 8x8 / 4x4 / 2x2 pixel blocks (d = 192, 768, 3072)
  deep500k, deep1m, deep2m, deep5m  the first n vectors of deep10m, same queries

Queries: the dataset's own queries where they exist (gist, bigann*, deep10m); otherwise 1000 vectors held out of the
collection at random (seed 0) and removed from the base.

Usage: python prepare_data.py <name> [<name> ...]        python prepare_data.py all
Needs: faiss, numpy, pyarrow, pillow (image sets), h5py (deep10m), curl. HF_ENDPOINT selects a Hugging Face mirror.
"""
import json
import os
import subprocess
import sys
import tarfile
import time

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("SKETCHCD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SKETCHCD_DATA", f"{ROOT}/data")
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
NQ = 1000
faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "10")))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(url, dst):
    for _ in range(30):
        r = subprocess.run(["curl", "-q", "-sS", "-L", "-C", "-", "--speed-limit", "100000", "--speed-time", "60",
                            "--connect-timeout", "30", "-o", dst, url])
        if r.returncode == 0 and os.path.getsize(dst) > 0:
            return dst
        time.sleep(5)
    raise RuntimeError(f"download failed: {url}")


def hf_files(repo, prefix):
    import urllib.request
    with urllib.request.urlopen(f"{HF}/api/datasets/{repo}", timeout=60) as f:
        sib = [s["rfilename"] for s in json.load(f)["siblings"]]
    return sorted(s for s in sib if s.startswith(prefix) and s.endswith(".parquet"))


def write_rows(fh, x, normalize):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if normalize:
        nrm = np.sqrt(np.einsum("ij,ij->i", x, x, dtype=np.float64)).astype(np.float32)
        x = x / np.maximum(nrm, 1e-12)[:, None]
    m, d = x.shape
    rows = np.empty((m, d + 1), dtype=np.float32)
    rows[:, 0] = np.frombuffer(np.int32(d).tobytes(), dtype=np.float32)[0]
    rows[:, 1:] = x
    rows.tofile(fh)
    return m, d


def is_list(t):
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)


def parquet_to_fvecs(path, column, fh, normalize, limit=None):
    pf = pq.ParquetFile(path)
    sch = pf.schema_arrow
    if column not in sch.names:  # fall back to the first list-of-float column
        cands = [f.name for f in sch if is_list(f.type)]
        log(f"column {column!r} not found; using {cands[0]!r} of {cands}")
        column = cands[0]
    n, d = 0, None
    for batch in pf.iter_batches(batch_size=20000, columns=[column]):
        col = batch.column(0)
        m = len(col)
        if limit is not None and n + m > limit:
            col, m = col.slice(0, limit - n), limit - n
        flat = col.flatten()
        while is_list(flat.type):
            flat = flat.flatten()
        x = flat.to_numpy(zero_copy_only=False).reshape(m, -1)
        _, d = write_rows(fh, x, normalize)
        n += m
        if limit is not None and n >= limit:
            break
    return n, d


def fvecs_shape(path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    return os.path.getsize(path) // (4 * (d + 1)), d


def read_fvecs(path, n=None):
    N, d = fvecs_shape(path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(N, d + 1))
    return np.ascontiguousarray(mm[: (n or N), 1:])


def write_fvecs(path, X):
    with open(path, "wb") as fh:
        write_rows(fh, X, False)


def write_ivecs(path, A):
    A = np.ascontiguousarray(A, dtype=np.int32)
    out = np.empty((A.shape[0], A.shape[1] + 1), dtype=np.int32)
    out[:, 0] = A.shape[1]
    out[:, 1:] = A
    out.tofile(path)


def holdout(allpath, base, query):
    """Move NQ random rows (seed 0) of allpath into query; the rest becomes base."""
    N, d = fvecs_shape(allpath)
    rng = np.random.default_rng(0)
    qi = np.sort(rng.choice(N, NQ, replace=False))
    mm = np.memmap(allpath, dtype=np.float32, mode="r", shape=(N, d + 1))
    np.ascontiguousarray(mm[qi]).tofile(query)
    keep = np.ones(N, dtype=bool)
    keep[qi] = False
    with open(base, "wb") as fh:
        for s in range(0, N, 200000):
            e = min(N, s + 200000)
            np.ascontiguousarray(mm[s:e][keep[s:e]]).tofile(fh)
    del mm
    os.remove(allpath)


def finish(name):
    """Exact L2 top-100 ground truth of the first NQ queries, and INFO.json."""
    o = f"{OUT}/{name}"
    X = read_fvecs(f"{o}/{name}_base.fvecs")
    Q = read_fvecs(f"{o}/{name}_query.fvecs", NQ)
    nr = np.linalg.norm(X[:: max(1, len(X) // 20000)], axis=1)
    log(f"{name}: base {X.shape}, queries {Q.shape}, norms {nr.min():.3f}..{nr.max():.3f}; computing exact GT")
    _, I = faiss.knn(Q, X, 100)
    write_ivecs(f"{o}/{name}_gt.ivecs", I)
    with open(f"{o}/INFO.json", "w") as f:
        json.dump(dict(name=name, n=int(X.shape[0]), d=int(X.shape[1]), nq=int(Q.shape[0])), f)
    log(f"{name}: done")


# ------------------------------------------------------------------ sources
def src_texmex(name, url, member):
    """A TEXMEX archive (<member>/<member>_base.fvecs, <member>_query.fvecs): base and queries as they are."""
    o = f"{OUT}/{name}"
    raw = fetch(url, f"{o}/raw.tar.gz")
    with tarfile.open(raw) as tf:
        for part in ("base", "query"):
            src = tf.extractfile(f"{member}/{member}_{part}.fvecs")
            with open(f"{o}/{name}_{part}.fvecs", "wb") as fh:
                while chunk := src.read(1 << 24):
                    fh.write(chunk)
    os.remove(raw)


def src_hf(name, repo, prefix, column, normalize=True, limit=None):
    """Embeddings in the parquet files of a Hugging Face dataset; NQ random rows become the queries."""
    o = f"{OUT}/{name}"
    files = hf_files(repo, prefix)
    log(f"{name}: {len(files)} parquet files in {repo}/{prefix}")
    n = 0
    allp = f"{o}/all.fvecs"
    with open(allp, "wb") as fh:
        for i, fn in enumerate(files):
            raw = fetch(f"{HF}/datasets/{repo}/resolve/main/{fn}", f"{o}/raw.parquet")
            if i == 0:
                log(f"{name}: schema {pq.ParquetFile(raw).schema_arrow}")
            m, d = parquet_to_fvecs(raw, column, fh, normalize, None if limit is None else limit + NQ - n)
            n += m
            os.remove(raw)
            log(f"{name}: {fn} -> {n} vectors (d={d})")
            if limit is not None and n >= limit + NQ:
                break
    holdout(allp, f"{o}/{name}_base.fvecs", f"{o}/{name}_query.fvecs")


def src_hf_images(name, repo, files, limit=None):
    """Raw-pixel vectors (uint8 -> float32, no normalisation) from image parquet files; NQ random rows become the queries."""
    import io as _io
    from PIL import Image
    o = f"{OUT}/{name}"
    allp = f"{o}/all.fvecs"
    n = 0
    with open(allp, "wb") as fh:
        for fn in files:
            raw = fetch(f"{HF}/datasets/{repo}/resolve/main/{fn}", f"{o}/raw.parquet")
            pf = pq.ParquetFile(raw)
            icol = [f.name for f in pf.schema_arrow if pa.types.is_struct(f.type)][0]
            mode = None
            for batch in pf.iter_batches(batch_size=5000, columns=[icol]):
                rows = []
                for item in batch.column(0).to_pylist():
                    im = Image.open(_io.BytesIO(item["bytes"]))
                    if mode is None:
                        mode = "L" if im.mode in ("L", "1") else "RGB"
                    rows.append(np.asarray(im.convert(mode), dtype=np.float32).ravel())
                write_rows(fh, np.stack(rows), False)
                n += len(rows)
            os.remove(raw)
            log(f"{name}: {fn} -> {n} vectors (d={rows[0].size}, mode {mode})")
            if limit is not None and n >= limit:
                break
    holdout(allp, f"{o}/{name}_base.fvecs", f"{o}/{name}_query.fvecs")


def src_bin_prefix(name, base_url, query_url, n, dtype, normalize=False):
    """First n vectors of a big-ann-benchmarks .u8bin/.fbin file (HTTP range request), queries = first NQ of the public set."""
    o = f"{OUT}/{name}"
    item = np.dtype(dtype).itemsize
    hdr = f"{o}/hdr.bin"
    subprocess.run(["curl", "-q", "-sS", "-L", "-r", "0-7", "-o", hdr, base_url], check=True)
    N1, d = (int(v) for v in np.fromfile(hdr, dtype=np.int32, count=2))
    os.remove(hdr)
    raw = f"{o}/raw.bin"
    need = n * d * item
    for _ in range(30):  # resume until the prefix is complete
        have = os.path.getsize(raw) if os.path.exists(raw) else 0
        if have >= need:
            break
        with open(raw, "ab") as fh:
            subprocess.run(["curl", "-q", "-sS", "-L", "-r", f"{8 + have}-{8 + need - 1}", base_url], stdout=fh)
    assert os.path.getsize(raw) == need, (os.path.getsize(raw), need)
    log(f"{name}: {n} of {N1} vectors, d={d}, {dtype}")
    mm = np.memmap(raw, dtype=dtype, mode="r", shape=(n, d))
    with open(f"{o}/{name}_base.fvecs", "wb") as fh:
        for s in range(0, n, 500000):
            write_rows(fh, mm[s:s + 500000].astype(np.float32), normalize)
    del mm
    os.remove(raw)
    q = f"{o}/q.bin"
    fetch(query_url, q)
    nq, dq = (int(v) for v in np.fromfile(q, dtype=np.int32, count=2))
    Q = np.fromfile(q, dtype=dtype, offset=8, count=NQ * dq).reshape(NQ, dq).astype(np.float32)
    os.remove(q)
    with open(f"{o}/{name}_query.fvecs", "wb") as fh:
        write_rows(fh, Q, normalize)


def src_annb_hdf5(name, url, normalize):
    """ann-benchmarks HDF5 (train / test): base = train, queries = first NQ of test."""
    import h5py
    o = f"{OUT}/{name}"
    raw = fetch(url, f"{o}/raw.hdf5")
    with h5py.File(raw, "r") as h:
        tr, te = h["train"], h["test"]
        log(f"{name}: train {tr.shape}, test {te.shape}")
        with open(f"{o}/{name}_base.fvecs", "wb") as fh:
            for s in range(0, tr.shape[0], 500000):
                write_rows(fh, np.asarray(tr[s:s + 500000], dtype=np.float32), normalize)
        with open(f"{o}/{name}_query.fvecs", "wb") as fh:
            write_rows(fh, np.asarray(te[:NQ], dtype=np.float32), normalize)
    os.remove(raw)


# ------------------------------------------------------------------ sets derived for the scalability experiment
def pool(X, f):
    """Rows are 64x64x3 images (HWC, as written by src_hf_images); average over f x f blocks."""
    n = X.shape[0]
    s = 64 // f
    return X.reshape(n, s, f, s, f, 3).mean(axis=(2, 4), dtype=np.float32).reshape(n, -1)


def derive_tiny(name, f):
    src = f"{OUT}/tinyimagenet/tinyimagenet"
    o = f"{OUT}/{name}"
    N, d = fvecs_shape(src + "_base.fvecs")
    assert d == 12288, d
    mm = np.memmap(src + "_base.fvecs", dtype=np.float32, mode="r", shape=(N, d + 1))
    with open(f"{o}/{name}_base.fvecs.tmp", "wb") as fh:
        for s in range(0, N, 10000):
            write_rows(fh, pool(np.ascontiguousarray(mm[s:s + 10000, 1:]), f), False)
    del mm
    os.replace(f"{o}/{name}_base.fvecs.tmp", f"{o}/{name}_base.fvecs")
    write_fvecs(f"{o}/{name}_query.fvecs", pool(read_fvecs(src + "_query.fvecs"), f))


def derive_prefix(name, n):
    src = f"{OUT}/deep10m/deep10m"
    o = f"{OUT}/{name}"
    N, d = fvecs_shape(src + "_base.fvecs")
    mm = np.memmap(src + "_base.fvecs", dtype=np.float32, mode="r", shape=(N, d + 1))
    with open(f"{o}/{name}_base.fvecs.tmp", "wb") as fh:
        for s in range(0, n, 500000):
            np.ascontiguousarray(mm[s:min(n, s + 500000)]).tofile(fh)
    del mm
    os.replace(f"{o}/{name}_base.fvecs.tmp", f"{o}/{name}_base.fvecs")
    write_fvecs(f"{o}/{name}_query.fvecs", read_fvecs(src + "_query.fvecs"))


BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann"
COHERE = "CohereLabs/beir-embed-english-v3"

SOURCES = {
    "gist": lambda nm: src_texmex(nm, "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz", "gist"),
    "fiqa": lambda nm: src_hf(nm, COHERE, "fiqa/corpus/", "emb"),
    "treccovid": lambda nm: src_hf(nm, COHERE, "trec-covid/corpus/", "emb"),
    "dbp3s": lambda nm: src_hf(nm, "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K", "data/",
                               "text-embedding-3-small-1536-embedding"),
    "cifar10": lambda nm: src_hf_images(nm, "uoft-cs/cifar10", ["plain_text/train-00000-of-00001.parquet",
                                                                 "plain_text/test-00000-of-00001.parquet"]),
    "cifar100": lambda nm: src_hf_images(nm, "uoft-cs/cifar100", ["cifar100/train-00000-of-00001.parquet",
                                                                   "cifar100/test-00000-of-00001.parquet"]),
    "svhn": lambda nm: src_hf_images(nm, "ufldl-stanford/svhn", ["cropped_digits/train-00000-of-00001.parquet",
                                                                  "cropped_digits/test-00000-of-00001.parquet",
                                                                  "cropped_digits/extra-00000-of-00002.parquet",
                                                                  "cropped_digits/extra-00001-of-00002.parquet"]),
    "tinyimagenet": lambda nm: src_hf_images(nm, "zh-plus/tiny-imagenet", ["data/train-00000-of-00001-1359597a978bc4fa.parquet",
                                                                           "data/valid-00000-of-00001-70d52db3c749a935.parquet"]),
    "bigann10m": lambda nm: src_bin_prefix(nm, f"{BIGANN}/base.1B.u8bin", f"{BIGANN}/query.public.10K.u8bin", 10_000_000, np.uint8),
    "deep10m": lambda nm: src_annb_hdf5(nm, "http://ann-benchmarks.com/deep-image-96-angular.hdf5", True),
    "bigann100m": lambda nm: src_bin_prefix(nm, f"{BIGANN}/base.1B.u8bin", f"{BIGANN}/query.public.10K.u8bin", 100_000_000, np.uint8),
    "tiny8": lambda nm: derive_tiny(nm, 8),
    "tiny16": lambda nm: derive_tiny(nm, 4),
    "tiny32": lambda nm: derive_tiny(nm, 2),
    "deep500k": lambda nm: derive_prefix(nm, 500_000),
    "deep1m": lambda nm: derive_prefix(nm, 1_000_000),
    "deep2m": lambda nm: derive_prefix(nm, 2_000_000),
    "deep5m": lambda nm: derive_prefix(nm, 5_000_000),
}

if __name__ == "__main__":
    names = list(SOURCES) if sys.argv[1:] == ["all"] else sys.argv[1:]
    if not names:
        sys.exit(__doc__)
    for name in names:
        o = f"{OUT}/{name}"
        if os.path.exists(f"{o}/INFO.json"):
            log(f"{name}: exists")
            continue
        os.makedirs(o, exist_ok=True)
        try:
            if not os.path.exists(f"{o}/{name}_query.fvecs"):
                SOURCES[name](name)
            finish(name)
        except Exception as e:  # keep going with the next dataset
            log(f"{name}: FAILED {type(e).__name__}: {e}")
