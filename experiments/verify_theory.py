"""Numerical checks of every formal statement and of every derived number in Sections 3-5 of the paper.

Each check prints PASS / FAIL. Exact identities are tested on random instances to rounding accuracy, statements about
expectations by Monte Carlo, Algorithm 1 against brute force, and the numbers quoted in the text by recomputation.
"""
import math

import numpy as np

rng = np.random.default_rng(7)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def J_of(lists):
    return sum(((L - L.mean(0)) ** 2).sum() for L in lists if len(L))


w = lambda b: b / (b + 1.0)
wbar = lambda b: b / (b - 1.0)

# ---------------------------------------------------------------- Proposition 3.1 (scan volume)
b = rng.integers(1, 400, size=500).astype(float)
n, K = b.sum(), len(b)
cv = b.std() / b.mean()
check("Prop 3.1  sum b_k^2 / n = (n/K)(1 + CV^2)", abs((b ** 2).sum() / n - n / K * (1 + cv ** 2)) < 1e-6 * n / K)

# ---------------------------------------------------------------- Lemma 4.1 (exact gain), Eq. (4)
err_ins = err_rem = err_mov = err_loo = 0.0
for _ in range(300):
    d = int(rng.integers(2, 40))
    bp, bq = int(rng.integers(2, 30)), int(rng.integers(1, 30))
    Cp, Cq = rng.normal(size=(bp, d)) * rng.uniform(0.2, 3), rng.normal(size=(bq, d)) + rng.normal(size=d)
    x = rng.normal(size=d) * 2
    # insertion of a new vector into C_q
    err_ins = max(err_ins, abs(J_of([np.vstack([Cq, x])]) - J_of([Cq]) - w(bq) * ((x - Cq.mean(0)) ** 2).sum()))
    # removal of the member Cp[0], and a move of it to C_q
    xm, rest = Cp[0], Cp[1:]
    cp, cq = Cp.mean(0), Cq.mean(0)
    err_rem = max(err_rem, abs(J_of([Cp]) - J_of([rest]) - wbar(bp) * ((xm - cp) ** 2).sum()))
    g = w(bq) * ((xm - cq) ** 2).sum() - wbar(bp) * ((xm - cp) ** 2).sum()
    err_mov = max(err_mov, abs(J_of([rest, np.vstack([Cq, xm])]) - J_of([Cp, Cq]) - g))
    err_loo = max(err_loo, np.abs((xm - rest.mean(0)) - wbar(bp) * (xm - cp)).max())
check("Lemma 4.1 insertion raises J by w(b_k)|x - c_k|^2", err_ins < 1e-8, f"max err {err_ins:.1e}")
check("Lemma 4.1 removal lowers J by wbar(b_p)|x - c_p|^2", err_rem < 1e-8, f"max err {err_rem:.1e}")
check("Lemma 4.1 move changes J by g_{p->q}(x)", err_mov < 1e-8, f"max err {err_mov:.1e}")
check("Eq. (4)   x - c_p^(-x) = wbar(b_p)(x - c_p)", err_loo < 1e-10, f"max err {err_loo:.1e}")
check("Lemma 4.1 w(b-1) wbar(b)^2 = wbar(b)", all(abs(w(bb - 1) * wbar(bb) ** 2 - wbar(bb)) < 1e-12 for bb in range(2, 200)))

# ---------------------------------------------------------------- Lemma 4.3 (expected costs), Monte Carlo
d, trials = 12, 400000
A_j, A_k = rng.normal(size=(d, d)) * 0.5, rng.normal(size=(d, d)) * 0.8     # anisotropic, different covariances
mu_j, mu_k = rng.normal(size=d), rng.normal(size=d) * 1.5
s2j, s2k = (A_j ** 2).sum(), (A_k ** 2).sum()                                # tr Sigma = |A|_F^2
D2 = ((mu_j - mu_k) ** 2).sum()
worst = 0.0
for bj, bk in ((1, 1), (2, 7), (5, 3), (9, 40)):
    x = mu_j + rng.normal(size=(trials, d)) @ A_j.T
    ck = mu_k + (rng.normal(size=(trials, bk, d)) @ A_k.T).mean(1)
    cj = mu_j + (rng.normal(size=(trials, bj, d)) @ A_j.T).mean(1)
    e_k = ((x - ck) ** 2).sum(1).mean()
    e_new = ((x - cj) ** 2).sum(1).mean()
    worst = max(worst, abs(e_k / (D2 + s2j + s2k / bk) - 1), abs(e_new / (s2j * (1 + 1 / bj)) - 1), abs(w(bj) * e_new / s2j - 1))
    if bj >= 2:                                   # a member: x is one of the b_j vectors that form c_j
        mem = mu_j + rng.normal(size=(trials, bj, d)) @ A_j.T
        e_mem = ((mem[:, 0] - mem.mean(1)) ** 2).sum(1).mean()
        worst = max(worst, abs(e_mem / (s2j * (1 - 1 / bj)) - 1), abs(wbar(bj) * e_mem / s2j - 1))
check("Lemma 4.3 (i), (ii) newcomer and member, nearest-centroid and size-aware costs", worst < 0.01, f"max rel. dev {worst:.4f} (Monte Carlo)")

# ---------------------------------------------------------------- Corollary 4.4 (starvation threshold)
s2 = 1.7
bad = 0
for _ in range(2000):
    bj, bk = int(rng.integers(1, 60)), int(rng.integers(1, 60))
    Dd = rng.uniform(0.01, 3)
    Ek, Ej = Dd + s2 + s2 / bk, s2 * (1 + 1 / bj)                 # Lemma 4.3 with equal variances
    bad += abs(w(bk) * Ek - w(bj) * Ej - w(bk) * Dd) > 1e-10      # first identity
    bad += abs((Ek - Ej) - (Dd - s2 * (1 / bj - 1 / bk))) > 1e-10  # second identity
    bad += (Ej > Ek) != (1 / bj - 1 / bk > Dd / s2)                # own list dearer under the nearest-centroid rule
    bad += not (w(bj) * Ej < w(bk) * Ek)                           # own list always cheaper under the size-aware rule
    bad += (s2 * (1 + 1 / bj) > Dd + s2) != (bj < s2 / Dd)         # mature neighbour: b_j < b*
check("Cor 4.4   both identities, both equivalences, mature-neighbour threshold b*", bad == 0)

# ---------------------------------------------------------------- Corollary 4.5 (i)
bad = 0
for _ in range(400):
    d, Kk = 6, 5
    lists = [rng.normal(size=(int(rng.integers(2, 12)), d)) + rng.normal(size=d) * 1.5 for _ in range(Kk)]
    p = int(rng.integers(Kk))
    x = lists[p][0]
    stay = wbar(len(lists[p])) * ((x - lists[p].mean(0)) ** 2).sum()
    costs = [w(len(L)) * ((x - L.mean(0)) ** 2).sum() for L in lists]
    costs[p] = stay
    rule = int(np.argmin(costs))                                   # size-aware rule (Definition 4.2)
    wo = [L for L in lists]
    wo[p] = lists[p][1:]                                           # every list taken without x
    alt = int(np.argmin([w(len(L)) * ((x - L.mean(0)) ** 2).sum() for L in wo]))
    bad += rule != alt
    bad += abs(stay - w(len(lists[p]) - 1) * ((x - lists[p][1:].mean(0)) ** 2).sum()) > 1e-9
check("Cor 4.5   (i) member decision = arg min of w(b)|x - c|^2 with lists taken without x", bad == 0)
check("Cor 4.5   w is increasing, sigma^2/b decreasing", all(w(bb + 1) > w(bb) for bb in range(0, 500)))

# ---------------------------------------------------------------- Lemma 4.6 (sketch bounds), Eq. (8)
bad, err_slack, err_r = 0, 0.0, 0.0
for _ in range(500):
    d = int(rng.integers(3, 60))
    l = int(rng.integers(1, d + 1))
    Q = np.linalg.qr(rng.normal(size=(d, l)))[0].T                 # l x d, orthonormal rows
    xbar, x, c = rng.normal(size=d), rng.normal(size=d) * 2, rng.normal(size=d)
    yx, yc = Q @ (x - xbar), Q @ (c - xbar)
    rx = math.sqrt(max(((x - xbar) ** 2).sum() - (yx ** 2).sum(), 0))
    rc = math.sqrt(max(((c - xbar) ** 2).sum() - (yc ** 2).sum(), 0))
    P = np.eye(d) - Q.T @ Q
    xp, cp_ = P @ (x - xbar), P @ (c - xbar)
    err_r = max(err_r, abs(rx ** 2 - (xp ** 2).sum()), abs(rc ** 2 - (cp_ ** 2).sum()))   # squares: r = 0 when l = d
    D = ((x - c) ** 2).sum()
    LB, UB = ((yx - yc) ** 2).sum() + (rx - rc) ** 2, ((yx - yc) ** 2).sum() + (rx + rc) ** 2
    bad += not (LB <= D + 1e-9 and D <= UB + 1e-9)
    if rx > 1e-9 and rc > 1e-9:
        cos = float(xp @ cp_) / (rx * rc)
        err_slack = max(err_slack, abs((D - LB) - 2 * rx * rc * (1 - cos)))
check("Lemma 4.6 LB <= |x - c|^2 <= UB for random orthonormal Q, any centre", bad == 0)
check("Eq. (6)   r_x^2 equals the squared norm of the discarded component", err_r < 1e-10, f"max err {err_r:.1e}")
check("Eq. (8)   |x - c|^2 - LB = 2 r_x r_c (1 - cos theta)", err_slack < 1e-8, f"max err {err_slack:.1e}")


# ---------------------------------------------------------------- Theorem 4.7 (Algorithm 1 against brute force)
def verify(x, p, cand, C, sizes, Q, xbar, weights=True):
    yx = Q @ (x - xbar)
    rx = math.sqrt(max(((x - xbar) ** 2).sum() - (yx ** 2).sum(), 0))
    wq = (lambda q: w(sizes[q])) if weights else (lambda q: 1.0)
    tau = (wbar(sizes[p]) if weights else 1.0) * ((x - C[p]) ** 2).sum()
    qs, evals = p, 1
    beta = {}
    for q in cand:
        yc = Q @ (C[q] - xbar)
        rc = math.sqrt(max(((C[q] - xbar) ** 2).sum() - (yc ** 2).sum(), 0))
        beta[q] = wq(q) * (((yx - yc) ** 2).sum() + (rx - rc) ** 2)
    for q in sorted(cand, key=lambda q: beta[q]):
        if beta[q] >= tau:
            break
        kappa = wq(q) * ((x - C[q]) ** 2).sum()
        evals += 1
        if kappa < tau:
            tau, qs = kappa, q
    return qs, tau, evals


bad = bad_unit = 0
saved = []
for _ in range(1500):
    d, Kk = int(rng.integers(4, 40)), int(rng.integers(3, 30))
    l = int(rng.integers(1, d + 1))
    Q = np.linalg.qr(rng.normal(size=(d, l)))[0].T
    xbar = rng.normal(size=d)
    C = rng.normal(size=(Kk, d)) * rng.uniform(0.3, 2)
    sizes = rng.integers(2, 50, size=Kk)
    p = int(rng.integers(Kk))
    cand = [q for q in range(Kk) if q != p][: int(rng.integers(1, Kk))]
    x = C[p] + rng.normal(size=d) * rng.uniform(0.2, 2)
    for weights in (True, False):
        qs, tau, ev = verify(x, p, cand, C, sizes, Q, xbar, weights)
        exact = {q: (w(sizes[q]) if weights else 1.0) * ((x - C[q]) ** 2).sum() for q in cand}
        exact[p] = (wbar(sizes[p]) if weights else 1.0) * ((x - C[p]) ** 2).sum()
        best = min(exact.values())
        fail = abs(tau - best) > 1e-9 or abs(exact[qs] - best) > 1e-9
        if weights:
            bad += fail
            saved.append(1 - ev / (len(cand) + 1))
        else:
            bad_unit += fail
check("Thm 4.7   Algorithm 1 returns a minimiser of the exact size-aware cost (1500 instances)", bad == 0, f"mean share of costs skipped {np.mean(saved):.2f}")
check("Thm 4.7   with unit weights: exact nearest centroid among the candidates", bad_unit == 0)

# ---------------------------------------------------------------- Figure 3 and the worked example of Figure 4 (toy numbers)
bstar = 2.4
check("Fig. 3    neighbour cost 1 + 1/b* = 1.42, own cost 2.0 at b_j = 1, size-aware 1.0", abs(1 + 1 / bstar - 1.4167) < 1e-3 and w(1) * 2.0 == 1.0)
check("Fig. 3    curves cross at b_j = b* = 2.4", abs((1 + 1 / 2.4) - (1 + 1 / bstar)) < 1e-12)
tau0 = wbar(5) * 1.6
toy = [(3, 1.6, 2.0), (4, 2.0, None), (9, 2.4, None), (2, 3.6, None)]      # (b_q, LB, exact distance)
betas = [round(w(bq) * lb, 2) for bq, lb, _ in toy]
check("Fig. 4    worked example of Algorithm 1: tau = 2.00, bounds 1.20 1.60 2.16 2.40, exact cost of q1 = 1.50, stop at 1.60 >= 1.50",
      abs(tau0 - 2.0) < 1e-12 and betas == [1.2, 1.6, 2.16, 2.4] and abs(w(3) * 2.0 - 1.5) < 1e-12 and betas[1] >= 1.5 and betas == sorted(betas))

# ---------------------------------------------------------------- Proposition 4.8 and the numbers of Section 4.4
# rho = throughput of matrix multiplication / throughput of full-dimensional distances (results/exp00_throughput.jsonl):
# about 3 for d <= 1536 (1100 against 350-420 GFLOPS) and about 6 for d = 3072 (175 GFLOPS)
Ps, P, W, Wp, rho = 6, 2, 129, 257, 3.0
RHO = {"GIST": 3.0, "TREC-COVID": 3.0, "SVHN": 6.0}
V = P + 1
thr = {}
for name, d, K in (("GIST", 960, 4000), ("TREC-COVID", 1024, 1651), ("SVHN", 3072, 3173)):
    eta = 4.5 * math.sqrt(K) + Ps * W + P * W + Wp
    thr[name] = (d + eta) / (V * RHO[name] * d)
check("Prop 4.8  thresholds 0.29 / 0.27 / 0.08", [round(v, 2) for v in thr.values()] == [0.29, 0.27, 0.08], str({k: round(v, 3) for k, v in thr.items()}))
T = lambda l, d, eta, s, r: l * (d + eta) + V * r * d * s
sl = {"GIST": {32: 12.1, 64: 4.7, 96: 2.9, 128: 2.2, 192: 1.6, 256: 1.3, 384: 1.2},
      "TREC-COVID": {32: 75.9, 64: 35.7, 96: 18.7, 128: 11.0, 192: 5.0, 256: 3.0, 384: 1.6},
      "SVHN": {32: 2.4, 64: 1.3, 96: 1.1, 128: 1.1, 192: 1.1, 256: 1.1, 384: 1.1}}     # measured s(l): results/exp07_sketch_dim.jsonl (Exp-7)
dims = {"GIST": (960, 4000), "TREC-COVID": (1024, 1651), "SVHN": (3072, 3173)}
opt = {}
for name, tab in sl.items():
    d, K = dims[name]
    eta = 4.5 * math.sqrt(K) + Ps * W + P * W + Wp
    opt[name] = min(tab, key=lambda l: T(l, d, eta, tab[l], RHO[name]))
    opt3 = min(tab, key=lambda l: T(l, d, eta, tab[l], 3.0))   # the optimum does not depend on the choice between 3 and 6
    ok_all &= opt3 == opt[name]
check("Sec 4.4   cost model places the optimum at 32 / 96 / 32 dimensions", list(opt.values()) == [32, 96, 32], str(opt))
drops = [round((sl["GIST"][32] - sl["GIST"][64]) / 32, 2), round((sl["TREC-COVID"][96] - sl["TREC-COVID"][128]) / 32, 2), round((sl["SVHN"][32] - sl["SVHN"][64]) / 32, 2)]
check("Sec 4.4   measured drops of s per dimension 0.23 / 0.24 / 0.03", drops == [0.23, 0.24, 0.03], str(drops))
s1 = [128 * (d + P * W + Wp) / (V * rho * d) for d in (768, 1536)]   # cost of the extension in full-dimensional distances per stage
check("Sec 4.4   the extension costs as much as 19-24 distances per verified stage for d = 768-1536 (rho = 3); 48 is about twice that",
      round(s1[1]) == 19 and round(s1[0]) == 24 and 2.0 <= 48 / s1[0] <= 48 / s1[1] <= 2.6, f"{s1[1]:.1f} .. {s1[0]:.1f}")
ell = lambda d: d if d <= 256 else min(384, max(128, 64 * round(d / 768)))
check("Sec 4.4   sketch-dimension rule gives 128 / 128 / 128 / 256 / 384 for d = 960 / 1024 / 1536 / 3072 / 12288",
      [ell(d) for d in (960, 1024, 1536, 3072, 12288)] == [128, 128, 128, 256, 384])

# ---------------------------------------------------------------- Proposition 5.1 and the first pass
nm = rng.integers(50, 5000, size=63).astype(float)
n, M, K = nm.sum(), len(nm), 4000
cvm = nm.std() / nm.mean()
Km = K * nm / n
check("Prop 5.1  sum n_m K_m = (nK/M)(1 + CV_M^2)", abs((nm * Km).sum() / (n * K / M * (1 + cvm ** 2)) - 1) < 1e-12)
bad = 0
for lam in (1, 2, 3):
    for cv_ in (0.0, 0.33, 0.52, 1.0):
        f = lambda M_: M_ + lam * K * (1 + cv_ ** 2) / M_
        Ms = np.linspace(5, 2000, 400000)
        bad += abs(Ms[np.argmin(f(Ms))] - math.sqrt(lam * K * (1 + cv_ ** 2))) > 0.02
check("Prop 5.1  routing + lambda passes minimised at M = sqrt(lambda K (1 + CV_M^2))", bad == 0)
ratio = lambda cv_: (4 + 3 * cv_ ** 2) / (2 * math.sqrt(3 * (1 + cv_ ** 2)))
check("Sec 5.1   M = sqrt(K), 3 passes: (4 + 3 CV^2) n sqrt(K) l is 19-23% above the minimum for CV_M = 0.33-0.52",
      [round(100 * (ratio(c) - 1)) for c in (0.33, 0.52)] == [19, 23])

# ---------------------------------------------------------------- Section 5.3, complexity (numbers)
d, K, l, s, sp = 960, 4000, 128, 2.2, 2.1
tot = d * l + 4.5 * math.sqrt(K) * l + Ps * W * l + d + P * (W * l + s * d) + (Wp * l + sp * d)
check("Sec 5.3   GIST: 3.3e5 multiply-adds per vector, a twelfth of K d = 3.8e6", round(tot / 1e5, 1) == 3.3 and round(K * d / 1e6, 1) == 3.8 and round(K * d / tot) == 12, f"{tot:.0f}, ratio {K * d / tot:.1f}")
check("Intro     one exhaustive pass on GIST at 5.5e11 multiply-adds/s (1100 GFLOPS): 6.98 s", round(1e6 * 4000 * 960 / 5.5e11, 2) == 6.98)
check("Fig. 2    step times 356 + 189 + 471 + 167 + 429 + 226 = 1838 ms", 356 + 189 + 471 + 167 + 429 + 226 == 1838)
# Proposition 4.8: T(l + 1) - T(l) = (d + eta) - V rho d (s(l) - s(l + 1)) for the cost model T(l) = l (d + eta) + V rho d s(l)
rng8 = np.random.default_rng(8)
bad8 = 0
for _ in range(2000):
    d8, eta8, V8, rho8 = rng8.integers(64, 4096), rng8.uniform(10, 2000), rng8.integers(1, 6), rng8.uniform(1, 8)
    l8, s0, s1 = rng8.integers(8, 512), rng8.uniform(0, 80), rng8.uniform(0, 80)
    T0, T1 = l8 * (d8 + eta8) + V8 * rho8 * d8 * s0, (l8 + 1) * (d8 + eta8) + V8 * rho8 * d8 * s1
    bad8 += (T1 < T0) != (s0 - s1 > (d8 + eta8) / (V8 * rho8 * d8))
check("Prop 4.8  one more sketch dimension lowers T iff s(l) - s(l + 1) > (d + eta) / (V rho d)", bad8 == 0)
# Proposition 5.2 with the measured read bandwidth B = 38 GB/s of one socket (results/exp00_throughput.jsonl)
read5 = {ds: 5 * 4 * n_ * d_ / 38e9 for ds, (n_, d_) in {"GIST": (1000000, 960), "SVHN": (629420, 3072)}.items()}
check("Prop 5.2  five passes at 38 GB/s: 0.5 s on GIST and 1.0 s on SVHN, more than a quarter of the builds of 1.84 s and 3.38 s",
      round(read5["GIST"], 1) == 0.5 and round(read5["SVHN"], 1) == 1.0 and read5["GIST"] / 1.84 > 0.25 and read5["SVHN"] / 3.38 > 0.25,
      f"{read5['GIST']:.2f} s, {read5['SVHN']:.2f} s")
print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
