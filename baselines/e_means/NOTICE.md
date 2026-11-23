# Notice

The four Python files of this folder are the CPU path of the reference implementation of **e-means**, taken without
changes from the FAISS repository (https://github.com/facebookresearch/faiss, directory `contrib/e_means`). They were
copied on 13 November 2026 and are byte-identical to the `main` branch of that repository on 21 November 2026.
Copyright (c) Meta Platforms, Inc. and affiliates, distributed under the MIT License in `LICENSE` (the license file of
the FAISS repository).

| File | Content |
|---|---|
| `e_means.py` | `NumpyEmeans`, the class that the paper runs |
| `base.py` | recipes and lifecycle shared by the implementations |
| `e_means_spherical.py` | normalisation for the spherical variant, imported by `e_means.py` on request |
| `__init__.py` | exports `NumpyEmeans` |

The paper calls `NumpyEmeans(d, K, nepochs=2, seed=...)` and `nepochs=5` (EMeans-2 and EMeans-5) with all other
arguments at their defaults, and takes the labels of the last epoch as the lists.

The GPU, quantized, distributed, and sharded classes of the package (`Emeans`, `QuantizedEmeans`, ...) are not part of
this copy. `__init__.py` imports them on first use, so asking for one of them raises an import error; they are in the
FAISS repository.

Reference: S. Bessa, P.-E. Mazare, G. Szilvasy, H. Touvron, D. Picard, M. Douze, H. Jegou. e-means: Outperforming
k-means (in One-Tenth the Epochs). arXiv:2611.09501, 2026.
