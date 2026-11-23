"""SketchCD under named configurations: the ablation (Exp-11), the sketch dimension (Exp-7), the sensitivity study
(Exp-8), and the thread scaling (Exp-5). Build time = everything needed to obtain the centroids AND the lists; recall at
equal scan volume. Each configuration is built --reps times and the fastest build is recorded.

  python run_sketchcd_configs.py --datasets gist treccovid svhn --seeds 42 --reps 3 \
      --cfg "default:" "unweighted:weighted=0" "no-sketch-sweeps:sweeps=0" --out runs/exp11_ablation.jsonl

A configuration is  name:key=value,key=value  (options of bin/sketchcd, see the head of sketchcd.cpp); the name is the
`method` of the record. Every run is pinned to one NUMA node.
"""
import argparse
import math

from common import RUNS, Ctx, append_record

PARTS = ("basis_ms", "proj_ms", "pass_ms", "sweep_ms", "cent_ms", "ext_ms", "xsweep_ms", "fnn_ms")


def parse_cfg(s):
    name, _, rest = s.partition(":")
    opts = {}
    for kv in rest.split(","):
        if kv.strip():
            k, _, v = kv.partition("=")
            opts[k.strip()] = v.strip()
    return name, opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--cfg", nargs="+", default=["default:"])
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--kmult", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-ref", action="store_true", help="skip the reference partition (no WCSS excess)")
    ap.add_argument("--out", default=f"{RUNS}/sketchcd_configs.jsonl")
    a = ap.parse_args()
    cfgs = [parse_cfg(c) for c in a.cfg]
    for name in a.datasets:
        ctx = Ctx(name, a.node, a.threads, 1000)
        N, d = ctx.N, ctx.d
        K = int(round(a.kmult * math.sqrt(N)))
        big = N > 3_000_000
        Jref = None if (big or a.no_ref) else ctx.evaluate(ctx.reference(K))["wcss_nc"]
        print(f"\n### {name}: N={N} d={d} K={K} (N/K={N / K:.0f}) threads={a.threads}", flush=True)
        for seed in a.seeds:
            for tag, opts in cfgs:
                best = None
                for _ in range(1 if big else a.reps):
                    r, C, lab = ctx.sketchcd(K, seed=seed, **opts)
                    if best is None or r["total_ms"] < best[0]["total_ms"]:
                        best = (r, C, lab)
                r, C, lab = best
                ev = ctx.evaluate(C, lab)
                ex = lambda w: 100.0 * (ev[w] - Jref) / Jref if Jref else float("nan")  # noqa: E731
                parts = " ".join(f"{r[k] / 1000:.2f}" for k in PARTS)
                print(f"{tag:22s} s{seed} total={r['total_ms'] / 1000:6.2f}s [{parts}] | lists: excess={ex('wcss_lab'):5.2f}% "
                      f"R10@1%={ev['R10@0.01_lab']:.4f} R100@2%={ev['R100@0.02_lab']:.4f} cv={ev['cv_lab']:.3f} "
                      f"tiny={100 * ev['tiny_lab']:.1f}% empty={ev['empty_lab']} lab!=nn={100 * ev['label_ne_nearest']:.2f}% | "
                      f"l={r['l']} pilot={r['pilot_evals']:.0f} xev/pt={r['xevals_per_point']:.1f} "
                      f"fev/pt={r['fevals_per_point']:.1f} moves={r['moves']}", flush=True)
                append_record(a.out, dict(dataset=name, N=N, d=d, K=K, method=tag, opts=opts, seed=seed,
                                          build_s=r["total_ms"] / 1000.0, run=r, Jref=Jref, which="lab", **ev))
                del C, lab
        del ctx


if __name__ == "__main__":
    main()
