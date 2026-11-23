// sketchcd.cpp -- SketchCD: construction of the lists of a partitioned (IVF) vector index.
//
// The six steps of the paper (Figure 2, Algorithm 2):
//   Phase 1, sketch-space clustering with the size-aware rule
//     1. sketch   : Y = (X - mu) Q^T, Q (l x d) = top-l principal directions of a sample (randomized subspace
//                   iteration + Rayleigh-Ritz, coordinates ordered by variance), and the residual norm
//                   r_x = |(x - mu) - Q^T y| of every vector (one extra float per vector).
//     2. cells    : sqrt(K) meso-centroids from a sample, every vector routed to its cell; inside every cell an
//                   independent size-aware online pass, join cost b/(b+1)|y - c|^2, followed by `replays`
//                   leave-one-out replays with immediate updates. The cells are independent, so sequential
//                   coordinate descent runs in parallel over the cells (mode=two; mode=flat: one pass against all K
//                   centroids, the "no cells" variant of the ablation).
//     3. sweeps   : candidate-restricted sweeps in sketch space (own list + its candk nearest lists), centroids
//                   over-relaxed between sweeps.
//   Phase 2, full-dimensional certification
//     4. means    : full-dimensional means of the lists; a pilot measures how tight the bound is, and if it is loose
//                   the sketch is extended by further principal directions (used by the bound only).
//     5. xsweeps  : the same sweeps with EXACT full-dimensional decisions (Algorithm 1). For a vector x and a centroid c
//                       |x - c|^2 = |y - Qc|^2 + |x_perp - c_perp|^2   and   (r_x - r_c)^2 <= |x_perp - c_perp|^2,
//                   so  LB = |y - Qc|^2 + (r_x - r_c)^2  is a lower bound that costs l instead of d operations.
//                   Only candidates whose bound beats the best exact cost so far are evaluated in full dimension;
//                   the decisions are identical to those of a full-dimensional sweep over the same candidates.
//     6. fnn      : final exact relabelling to the nearest of fcandk candidate centroids; the returned centroids are
//                   the means of the returned lists (fupd).
//   For d <= 256 no sketch is used (l = d): the verified sweeps are then ordinary sweeps and run as part of step 3.
//
// Every matrix multiplication is a single-threaded BLAS call made by a worker on a cache-sized tile.
//
// build: bash build.sh sketchcd   (g++ -O3 -march=native -fopenmp, linked against OpenBLAS; the binary is bin/sketchcd)
// usage: sketchcd <base.fvecs> <K> <out_prefix|-> [key=value ...]
//   dp=0 ds=0 (auto) nsample=16384 power=2 over=8     sketch dimensions l (bound) and ls (decisions), subspace estimation
//   ext=-1 (auto: pilot, extend by 128 when more than pthr=48 candidates per point survive the bound; 0: never)
//   mode=auto|flat|two  meso=0 (0: sqrt(K))  kflat=0 (auto: flat when K <= kflat)  d1=0 (dims of the first pass, 0: all)
//   rho=0.25 bmin=256 bmax=16384 weighted=1 replays=2  online pass (geometric refresh) and replays
//   B=4096  fixed batch when rho=0; bmin applies to mode=flat, inside cells the minimum batch is max(16, min(256, K_m/8))
//   sweeps=6 xsweeps=2 omega=1.6 candk=128 nbr=2 gs=1  refinement (nbr: sweeps between neighbour-list rebuilds)
//   fnn=1 fcandk=256 fupd=1                            final relabelling
//   seed=42 limit=0   prune=1 (0: no pruning in the exact sweeps, for verification)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

extern "C" {
int sgemm_(const char*, const char*, int*, int*, int*, const float*, const float*, int*, const float*, int*,
           float*, float*, int*);
int dsyevd_(const char*, const char*, int*, double*, int*, double*, double*, int*, int*, int*, int*);
void openblas_set_num_threads(int);
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
static std::map<std::string, double> g_prof; // wall time of serial-level sections, printed as "prof"
static bool g_prune = true; // prune=0: the exact sweeps evaluate every candidate (to verify that pruning changes nothing)
#define PROF(name, stmt)              \
    {                                 \
        const auto _t = clk::now();   \
        stmt;                         \
        g_prof[name] += ms_since(_t); \
    }

// C (rb x ra, row-major) = B[:, :k] * A[:, :k]^T ; A is ra x lda row-major, B is rb x ldb row-major
static inline void gemm_rows(const float* A, int ra, int lda, const float* B, int rb, int ldb, int k, float* C) {
    const char ta = 'T', tb = 'N';
    float one = 1.0f, zero = 0.0f;
    int m = ra, n = rb, kk = k, ldc = ra;
    sgemm_(&ta, &tb, &m, &n, &kk, &one, A, &lda, B, &ldb, &zero, C, &ldc);
}

// squared Euclidean distance with four independent accumulators (a single accumulator is bound by the FMA latency)
static inline float sqdist(const float* a, const float* b, size_t dim) {
#if defined(__AVX512F__)
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    size_t t = 0;
    for (; t + 64 <= dim; t += 64) {
        const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + t), _mm512_loadu_ps(b + t));
        const __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + t + 16), _mm512_loadu_ps(b + t + 16));
        const __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + t + 32), _mm512_loadu_ps(b + t + 32));
        const __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + t + 48), _mm512_loadu_ps(b + t + 48));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
        s2 = _mm512_fmadd_ps(d2, d2, s2);
        s3 = _mm512_fmadd_ps(d3, d3, s3);
    }
    for (; t + 16 <= dim; t += 16) {
        const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + t), _mm512_loadu_ps(b + t));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    for (; t < dim; ++t) {
        const float df = a[t] - b[t];
        s += df * df;
    }
    return s;
#else
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t t = 0;
    for (; t + 8 <= dim; t += 8)
        for (int u = 0; u < 8; ++u) {
            const float df = a[t + u] - b[t + u];
            acc[u] += df * df;
        }
    float s = 0;
    for (int u = 0; u < 8; ++u) s += acc[u];
    for (; t < dim; ++t) {
        const float df = a[t] - b[t];
        s += df * df;
    }
    return s;
#endif
}

// eigen-decomposition of a symmetric n x n matrix (overwritten by the eigenvectors, eigenvalues ascending)
static void sym_eig(std::vector<double>& A, int n, std::vector<double>& w) {
    w.resize(n);
    int lda = n, info = 0, lwork = -1, liwork = -1, iq = 0;
    double wq = 0;
    dsyevd_("V", "U", &n, A.data(), &lda, w.data(), &wq, &lwork, &iq, &liwork, &info);
    lwork = (int)wq;
    liwork = iq;
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    dsyevd_("V", "U", &n, A.data(), &lda, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
}

// dot product, four accumulators. Measured on the Xeon Gold 5218R: 1.2-1.35x faster than the
// subtract + FMA kernel when both operands are in L1, but no faster (1.01-1.05x) in the access pattern of the exact
// sweep, where the centroid rows come from L2. It is used because the centred form below is well conditioned.
static inline float dotp(const float* a, const float* b, size_t dim) {
#if defined(__AVX512F__)
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    size_t t = 0;
    for (; t + 64 <= dim; t += 64) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + t), _mm512_loadu_ps(b + t), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + t + 16), _mm512_loadu_ps(b + t + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + t + 32), _mm512_loadu_ps(b + t + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + t + 48), _mm512_loadu_ps(b + t + 48), s3);
    }
    for (; t + 16 <= dim; t += 16) s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + t), _mm512_loadu_ps(b + t), s0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    for (; t < dim; ++t) s += a[t] * b[t];
    return s;
#else
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t t = 0;
    for (; t + 8 <= dim; t += 8)
        for (int u = 0; u < 8; ++u) acc[u] += a[t + u] * b[t + u];
    float s = 0;
    for (int u = 0; u < 8; ++u) s += acc[u];
    for (; t < dim; ++t) s += a[t] * b[t];
    return s;
#endif
}

static inline float sqnorm(const float* a, size_t dim) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (size_t t = 0; t < dim; ++t) s += a[t] * a[t];
    return s;
}

// ------------------------------------------------------------------------------------------------------------
// 1. principal subspace of a sample: randomized subspace iteration, rows of Q orthonormal and ordered by variance
// ------------------------------------------------------------------------------------------------------------
// rows of Q (L x d) <- an orthonormal basis of their span, via the eigen-decomposition of the L x L Gram matrix
static void orthonormalize(std::vector<float>& Q, int L, size_t d) {
    std::vector<double> G((size_t)L * L);
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j <= i; ++j) {
            const float *a = &Q[(size_t)i * d], *b = &Q[(size_t)j * d];
            double s = 0;
#pragma omp simd reduction(+ : s)
            for (size_t t = 0; t < d; ++t) s += (double)a[t] * b[t];
            G[(size_t)i * L + j] = G[(size_t)j * L + i] = s;
        }
    std::vector<double> w;
    sym_eig(G, L, w);
    const double floor_w = std::max(w[L - 1], 1e-300) * 1e-12;
    std::vector<float> Qn((size_t)L * d);
#pragma omp parallel
    {
        std::vector<double> acc(d);
#pragma omp for schedule(dynamic, 4)
        for (int r = 0; r < L; ++r) {
            std::fill(acc.begin(), acc.end(), 0.0);
            const double* u = &G[(size_t)r * L]; // eigenvector r (column r)
            for (int i = 0; i < L; ++i) {
                const double c = u[i];
                const float* q = &Q[(size_t)i * d];
#pragma omp simd
                for (size_t t = 0; t < d; ++t) acc[t] += c * q[t];
            }
            const double sc = 1.0 / std::sqrt(std::max(w[r], floor_w));
            float* o = &Qn[(size_t)(L - 1 - r) * d]; // descending eigenvalue order
            for (size_t t = 0; t < d; ++t) o[t] = (float)(acc[t] * sc);
        }
    }
    Q.swap(Qn);
}

struct Sketch {
    int l = 0;
    std::vector<float> Q, mu, qmu; // l x d, d, l
    double captured = -1.0, captured_ext = -1.0, orth_err = 0.0;
};

// l principal directions of a sample. lprev > 0: the sketch is extended; the sample is deflated by the lprev rows
// already in sk.Q (same centre sk.mu), and the l new rows, orthogonal to them, are appended.
static void pca_basis(const float* X, size_t n, size_t d, int l, size_t nsample, int power, int over,
                      std::mt19937& rng, Sketch& sk, int lprev = 0) {
    const int T = omp_get_max_threads();
    const size_t ns = std::min(n, nsample);
    auto tlap = clk::now();
    auto lap = [&](const char* name) {
        g_prof[name] += ms_since(tlap);
        tlap = clk::now();
    };
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t i = 0; i < ns; ++i) std::swap(idx[i], idx[i + rng() % (n - i)]);
    std::sort(idx.begin(), idx.begin() + ns);
    std::unique_ptr<float[]> Zown(new float[ns * d]);
    float* Z = Zown.get();
    std::vector<double> mean(d, 0.0);
#pragma omp parallel
    {
        std::vector<double> loc(d, 0.0);
#pragma omp for nowait
        for (size_t i = 0; i < ns; ++i) {
            memcpy(&Z[i * d], &X[idx[i] * d], d * 4);
            for (size_t t = 0; t < d; ++t) loc[t] += Z[i * d + t];
        }
#pragma omp critical
        for (size_t t = 0; t < d; ++t) mean[t] += loc[t];
    }
    if (lprev == 0) {
        sk.mu.resize(d);
        for (size_t t = 0; t < d; ++t) sk.mu[t] = (float)(mean[t] / (double)ns);
    }
    double tot = 0;
#pragma omp parallel for reduction(+ : tot)
    for (size_t i = 0; i < ns; ++i) {
        float* z = &Z[i * d];
        double s = 0;
        for (size_t t = 0; t < d; ++t) {
            z[t] -= sk.mu[t];
            s += (double)z[t] * z[t];
        }
        tot += s;
    }
    if (lprev > 0) { // deflate: Z <- Z - (Z Qp^T) Qp
        tot = 0;
#pragma omp parallel reduction(+ : tot)
        {
            std::vector<float> Pz(512 * (size_t)lprev);
#pragma omp for schedule(dynamic, 1)
            for (size_t s0 = 0; s0 < ns; s0 += 512) {
                const size_t nb = std::min<size_t>(512, ns - s0);
                gemm_rows(sk.Q.data(), lprev, (int)d, &Z[s0 * d], (int)nb, (int)d, (int)d, Pz.data());
                const char ta = 'N', tb = 'N';
                float mone = -1.0f, one = 1.0f;
                int m = (int)d, nn = (int)nb, kk = lprev, lda = (int)d, ldb = lprev, ldc = (int)d;
                sgemm_(&ta, &tb, &m, &nn, &kk, &mone, sk.Q.data(), &lda, Pz.data(), &ldb, &one, &Z[s0 * d], &ldc);
                double sq = 0;
                for (size_t i = 0; i < nb * d; ++i) sq += (double)Z[s0 * d + i] * Z[s0 * d + i];
                tot += sq;
            }
        }
    }
    lap("b.sample");
    const int L = (int)std::min<size_t>(std::min(d, ns), (size_t)l + over);
    std::normal_distribution<float> gauss;
    std::vector<float> Q((size_t)L * d);
    for (auto& v : Q) v = gauss(rng);
    lap("b.gauss");
    orthonormalize(Q, L, d);
    lap("b.orth");
    std::unique_ptr<float[]> Y1(new float[ns * (size_t)L]);
    auto mul_ZQt = [&]() { // Y1 = Z Q^T, one single-threaded GEMM per worker and tile
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t s = 0; s < ns; s += 512) {
            const size_t nb = std::min<size_t>(512, ns - s);
            gemm_rows(Q.data(), L, (int)d, &Z[s * d], (int)nb, (int)d, (int)d, &Y1[s * L]);
        }
    };
    const size_t wcol = std::max<size_t>(32, (d + (size_t)T - 1) / (size_t)T);
    for (int it = 0; it < power; ++it) {
        mul_ZQt();
        lap("b.ZQt");
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t c0 = 0; c0 < d; c0 += wcol) { // Q[:, c0:c0+w] = Y1^T Z[:, c0:c0+w]
            const char ta = 'N', tb = 'T';
            float one = 1.0f, zero = 0.0f;
            int m = (int)std::min(wcol, d - c0), nn = L, kk = (int)ns, lda = (int)d, ldb = L, ldc = (int)d;
            sgemm_(&ta, &tb, &m, &nn, &kk, &one, Z + c0, &lda, Y1.get(), &ldb, &zero, Q.data() + c0, &ldc);
        }
        lap("b.YtZ");
        orthonormalize(Q, L, d);
        lap("b.orth");
    }
    // Rayleigh-Ritz: rotate the basis to the principal directions of the sample inside the subspace
    mul_ZQt();
    lap("b.ZQt");
    std::vector<float> Hf((size_t)L * L);
    {
        const char ta = 'N', tb = 'T';
        float one = 1.0f, zero = 0.0f;
        int m = L, nn = L, kk = (int)ns, lda = L, ldb = L, ldc = L;
        sgemm_(&ta, &tb, &m, &nn, &kk, &one, Y1.get(), &lda, Y1.get(), &ldb, &zero, Hf.data(), &ldc);
    }
    std::vector<double> H((size_t)L * L), w;
    for (size_t i = 0; i < H.size(); ++i) H[i] = Hf[i];
    sym_eig(H, L, w);
    std::vector<float> Qn((size_t)l * d, 0.0f);
    double cap = 0;
#pragma omp parallel
    {
        std::vector<double> acc(d);
#pragma omp for schedule(dynamic, 4) reduction(+ : cap)
        for (int r = 0; r < l; ++r) {
            const int col = L - 1 - r; // descending
            cap += w[col];
            std::fill(acc.begin(), acc.end(), 0.0);
            const double* u = &H[(size_t)col * L];
            for (int i = 0; i < L; ++i) {
                const double c = u[i];
                const float* q = &Q[(size_t)i * d];
#pragma omp simd
                for (size_t t = 0; t < d; ++t) acc[t] += c * q[t];
            }
            float* o = &Qn[(size_t)r * d];
            for (size_t t = 0; t < d; ++t) o[t] = (float)acc[t];
        }
    }
    if (lprev == 0) {
        sk.captured = cap / tot;
        sk.Q.swap(Qn);
    } else { // remove what rounding left of the old directions, then append
        sk.captured_ext = cap / tot;
#pragma omp parallel
        {
            std::vector<double> acc(d);
#pragma omp for schedule(dynamic, 4)
            for (int r = 0; r < l; ++r) {
                float* q = &Qn[(size_t)r * d];
                for (size_t t = 0; t < d; ++t) acc[t] = q[t];
                for (int j = 0; j < lprev; ++j) {
                    const float* qp = &sk.Q[(size_t)j * d];
                    double c = 0;
#pragma omp simd reduction(+ : c)
                    for (size_t t = 0; t < d; ++t) c += (double)q[t] * qp[t];
#pragma omp simd
                    for (size_t t = 0; t < d; ++t) acc[t] -= c * qp[t];
                }
                for (size_t t = 0; t < d; ++t) q[t] = (float)acc[t];
            }
        }
        orthonormalize(Qn, l, d);
        sk.Q.insert(sk.Q.end(), Qn.begin(), Qn.end());
    }
    l += lprev;
    sk.l = l;
    lap("b.ritz");
    sk.qmu.assign(l, 0.0f);
    double err = 0;
    for (int r = 0; r < l; ++r) {
        double s = 0;
        for (size_t t = 0; t < d; ++t) s += (double)sk.Q[(size_t)r * d + t] * sk.mu[t];
        sk.qmu[r] = (float)s;
    }
#pragma omp parallel for reduction(max : err) schedule(dynamic, 4)
    for (int i = 0; i < l; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (size_t t = 0; t < d; ++t) s += (double)sk.Q[(size_t)i * d + t] * sk.Q[(size_t)j * d + t];
            err = std::max(err, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    sk.orth_err = err;
    lap("b.check");
}

// ------------------------------------------------------------------------------------------------------------
// 2. size-aware online pass over the stream ids[0..m) against Kl clusters seeded by the points seeds[0..Kl)
//    join cost  b/(b+1) |y - c|^2 on the first dq sketch coordinates; lab[id] = base + cluster.
//    par = false: runs on the calling thread (used inside a parallel loop over meso cells)
// ------------------------------------------------------------------------------------------------------------
static size_t online_pass(const float* Y, size_t l, int dq, const uint32_t* ids, size_t m, const uint32_t* seeds,
                          size_t Kl, uint32_t base, uint32_t* lab, bool weighted, double rho, size_t Bfixed,
                          size_t bmin, size_t bmax, bool par, int T, int replays = 0, size_t* replay_moves = nullptr) {
    if (Kl == 1) {
        for (size_t i = 0; i < m; ++i) lab[ids[i]] = base;
        return 0;
    }
    const int Tn = par ? T : 1;
    const size_t RBmax = 256, KT = 1024;
    std::vector<float> C(Kl * (size_t)dq), cn(Kl), wj(Kl, weighted ? 0.5f : 1.0f);
    std::vector<double> sums(Kl * (size_t)dq);
    std::vector<int64_t> cnt(Kl, 1); // every seed is a virtual first member of its cluster while streaming
    for (size_t k = 0; k < Kl; ++k) {
        const float* y = &Y[(size_t)seeds[k] * l];
        float nn = 0;
        for (int t = 0; t < dq; ++t) {
            sums[k * dq + t] = y[t];
            C[k * dq + t] = y[t];
            nn += y[t] * y[t];
        }
        cn[k] = nn;
    }
    std::vector<std::vector<float>> Ygs(Tn), Gs(Tn);
    for (int t = 0; t < Tn; ++t) {
        Ygs[t].resize(RBmax * (size_t)dq);
        Gs[t].resize(RBmax * std::min(KT, Kl));
    }
    const size_t maxB = std::min(m, std::max<size_t>(std::max(Bfixed, bmax), 1));
    std::vector<uint32_t> blk(maxB), order(maxB), cntb(Kl, 0), start(Kl), cursor(Kl), touched;
    touched.reserve(std::min(Kl, maxB));
    size_t refreshes = 0;
    for (size_t s = 0; s < m;) {
        size_t B = Bfixed;
        if (rho > 0) B = std::min(bmax, std::max(bmin, (size_t)(rho * (double)s)));
        B = std::min(B, maxB);
        const size_t e = std::min(m, s + B), nb = e - s;
        const size_t rb = par ? std::min(RBmax, std::max<size_t>(32, (nb + Tn - 1) / Tn)) : RBmax;
        const long ntile = (long)((nb + rb - 1) / rb);
#pragma omp parallel for if (par && ntile > 1) schedule(static, 1) num_threads(Tn)
        for (long tl = 0; tl < ntile; ++tl) {
            const int tid = par ? omp_get_thread_num() : 0;
            float *yg = Ygs[tid].data(), *G = Gs[tid].data();
            const size_t ts = s + (size_t)tl * rb, tn = std::min(rb, e - ts);
            float yn[256], bestv[256];
            uint32_t bestk[256];
            for (size_t i = 0; i < tn; ++i) {
                const float* y = &Y[(size_t)ids[ts + i] * l];
                memcpy(&yg[i * dq], y, dq * sizeof(float));
                yn[i] = sqnorm(y, dq);
                bestv[i] = 3.0e38f;
                bestk[i] = 0;
            }
            for (size_t k0 = 0; k0 < Kl; k0 += KT) {
                const size_t kt = std::min(KT, Kl - k0);
                gemm_rows(&C[k0 * dq], (int)kt, dq, yg, (int)tn, dq, dq, G);
                const float *cnk = &cn[k0], *wk = &wj[k0];
                for (size_t i = 0; i < tn; ++i) {
                    float* g = &G[i * kt];
                    const float x2 = yn[i];
                    float mn = 3.0e38f;
#pragma omp simd reduction(min : mn)
                    for (size_t k = 0; k < kt; ++k) {
                        const float v = (x2 + cnk[k] - 2.0f * g[k]) * wk[k];
                        g[k] = v;
                        mn = std::min(mn, v);
                    }
                    if (mn >= bestv[i]) continue;
                    for (size_t k = 0; k < kt; ++k)
                        if (g[k] == mn) {
                            bestv[i] = mn;
                            bestk[i] = (uint32_t)(k0 + k);
                            break;
                        }
                }
            }
            for (size_t i = 0; i < tn; ++i) blk[ts - s + i] = bestk[i];
        }
        // refresh the clusters that received points
        touched.clear();
        for (size_t i = 0; i < nb; ++i) {
            const uint32_t k = blk[i];
            lab[ids[s + i]] = base + k;
            if (cntb[k]++ == 0) touched.push_back(k);
        }
        uint32_t pos = 0;
        for (uint32_t k : touched) {
            start[k] = cursor[k] = pos;
            pos += cntb[k];
        }
        for (size_t i = 0; i < nb; ++i) order[cursor[blk[i]]++] = (uint32_t)i;
#pragma omp parallel for if (par && touched.size() >= 256) schedule(dynamic, 32) num_threads(Tn)
        for (size_t t = 0; t < touched.size(); ++t) {
            const size_t k = touched[t];
            double* sk = &sums[k * dq];
            for (uint32_t p = start[k]; p < start[k] + cntb[k]; ++p) {
                const float* y = &Y[(size_t)ids[s + order[p]] * l];
                for (int j = 0; j < dq; ++j) sk[j] += y[j];
            }
            cnt[k] += cntb[k];
            cntb[k] = 0;
            const double inv = 1.0 / (double)cnt[k];
            float nn = 0;
            for (int j = 0; j < dq; ++j) {
                const float v = (float)(sk[j] * inv);
                C[k * dq + j] = v;
                nn += v * v;
            }
            cn[k] = nn;
            if (weighted) wj[k] = (float)cnt[k] / (float)(cnt[k] + 1);
        }
        ++refreshes;
        s = e;
    }
    if (replays <= 0) return refreshes;

    // ---- replays: the stream is re-decided with the exact leave-one-out (Hartigan) test, in small blocks.
    //      Staying costs b/(b-1)|y - c|^2, joining b/(b+1)|y - c|^2; moves are committed sequentially with live
    //      sizes and sums, the centroids of the touched clusters are refreshed after every block.
    std::vector<float> ws(Kl, 1.0f);
    auto refresh = [&](size_t k) {
        const int64_t b = cnt[k];
        if (b > 0) {
            const double inv = 1.0 / (double)b;
            float nn = 0;
            for (int j = 0; j < dq; ++j) {
                const float v = (float)(sums[k * dq + j] * inv);
                C[k * dq + j] = v;
                nn += v * v;
            }
            cn[k] = nn;
        }
        wj[k] = weighted ? (float)b / (float)(b + 1) : 1.0f; // empty cluster: joining it is free
        ws[k] = (weighted && b > 1) ? (float)b / (float)(b - 1) : 1.0f;
    };
    for (size_t k = 0; k < Kl; ++k) { // remove the virtual seed members
        cnt[k] -= 1;
        const float* y = &Y[(size_t)seeds[k] * l];
        for (int t = 0; t < dq; ++t) sums[k * dq + t] = cnt[k] > 0 ? sums[k * dq + t] - y[t] : 0.0;
        refresh(k);
    }
    const size_t RB2 = std::min(maxB, std::max<size_t>(64, 4 * Kl));
    std::vector<float> dst(RB2), dbt(RB2);
    size_t moves = 0;
    for (int rp = 0; rp < replays; ++rp) {
        for (size_t s = 0; s < m;) {
            const size_t e = std::min(m, s + RB2), nb = e - s;
            const size_t rb = par ? std::min(RBmax, std::max<size_t>(32, (nb + Tn - 1) / Tn)) : RBmax;
            const long ntile = (long)((nb + rb - 1) / rb);
#pragma omp parallel for if (par && ntile > 1) schedule(static, 1) num_threads(Tn)
            for (long tl = 0; tl < ntile; ++tl) {
                const int tid = par ? omp_get_thread_num() : 0;
                float *yg = Ygs[tid].data(), *G = Gs[tid].data();
                const size_t ts = s + (size_t)tl * rb, tn = std::min(rb, e - ts);
                float yn[256], bestv[256], ds[256];
                uint32_t bestk[256], own[256];
                for (size_t i = 0; i < tn; ++i) {
                    const uint32_t id = ids[ts + i];
                    const float* y = &Y[(size_t)id * l];
                    memcpy(&yg[i * dq], y, dq * sizeof(float));
                    yn[i] = sqnorm(y, dq);
                    own[i] = lab[id] - base;
                    bestk[i] = own[i];
                    bestv[i] = 3.0e38f;
                    ds[i] = 0.0f;
                }
                for (size_t k0 = 0; k0 < Kl; k0 += KT) {
                    const size_t kt = std::min(KT, Kl - k0);
                    gemm_rows(&C[k0 * dq], (int)kt, dq, yg, (int)tn, dq, dq, G);
                    const float *cnk = &cn[k0], *wk = &wj[k0];
                    for (size_t i = 0; i < tn; ++i) {
                        float* g = &G[i * kt];
                        const float x2 = yn[i];
                        const size_t o = own[i];
                        const bool has_own = o >= k0 && o < k0 + kt;
                        const float dso = has_own ? std::max(x2 + cnk[o - k0] - 2.0f * g[o - k0], 0.0f) : 0.0f;
#pragma omp simd
                        for (size_t k = 0; k < kt; ++k) g[k] = (x2 + cnk[k] - 2.0f * g[k]) * wk[k];
                        if (has_own) { // the own cluster competes with its stay cost
                            g[o - k0] = dso * ws[o];
                            ds[i] = dso;
                        }
                        float mn = 3.0e38f;
#pragma omp simd reduction(min : mn)
                        for (size_t k = 0; k < kt; ++k) mn = std::min(mn, g[k]);
                        if (mn >= bestv[i]) continue;
                        for (size_t k = 0; k < kt; ++k)
                            if (g[k] == mn) {
                                bestv[i] = mn;
                                bestk[i] = (uint32_t)(k0 + k);
                                break;
                            }
                    }
                }
                for (size_t i = 0; i < tn; ++i) {
                    const size_t pos = ts - s + i;
                    blk[pos] = bestk[i];
                    dst[pos] = ds[i];
                    dbt[pos] = bestk[i] == own[i] ? ds[i] : (wj[bestk[i]] > 0.0f ? bestv[i] / wj[bestk[i]] : -1.0f);
                }
            }
            touched.clear();
            for (size_t i = 0; i < nb; ++i) {
                const uint32_t id = ids[s + i];
                const uint32_t p = lab[id] - base, q = blk[i];
                if (q == p) continue;
                const int64_t bp = cnt[p], bq = cnt[q];
                if (bp <= 1) continue; // never empty a cluster
                const float* y = &Y[(size_t)id * l];
                if (weighted) {
                    double dq2 = dbt[i];
                    if (dbt[i] < 0.0f) { // q was empty when the block was scored: use its live mean
                        dq2 = 0.0;
                        if (bq > 0)
                            for (int t = 0; t < dq; ++t) {
                                const double df = y[t] - sums[(size_t)q * dq + t] / (double)bq;
                                dq2 += df * df;
                            }
                    }
                    const double cq = (double)bq / (double)(bq + 1) * dq2;
                    const double cp = (double)bp / (double)(bp - 1) * dst[i];
                    if (!(cq < cp)) continue;
                }
                double *sp = &sums[(size_t)p * dq], *sq = &sums[(size_t)q * dq];
                for (int t = 0; t < dq; ++t) {
                    sp[t] -= y[t];
                    sq[t] += y[t];
                }
                --cnt[p];
                ++cnt[q];
                lab[id] = base + q;
                ++moves;
                if (cntb[p]++ == 0) touched.push_back(p);
                if (cntb[q]++ == 0) touched.push_back(q);
            }
            for (uint32_t k : touched) {
                refresh(k);
                cntb[k] = 0;
            }
            s = e;
        }
    }
    if (replay_moves) *replay_moves = moves;
    return refreshes;
}

// ------------------------------------------------------------------------------------------------------------
// cluster-grouped views
// ------------------------------------------------------------------------------------------------------------
static void build_groups(const uint32_t* lab, size_t n, size_t K, std::vector<uint32_t>& start,
                         std::vector<uint32_t>& members) {
    const int T = omp_get_max_threads();
    start.assign(K + 1, 0);
    members.resize(n);
    std::vector<std::vector<uint32_t>> hist(T, std::vector<uint32_t>(K, 0));
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const size_t i0 = n * (size_t)t / T, i1 = n * (size_t)(t + 1) / T;
        uint32_t* h = hist[t].data();
        for (size_t i = i0; i < i1; ++i) ++h[lab[i]];
    }
    uint32_t pos = 0;
    for (size_t k = 0; k < K; ++k) {
        start[k] = pos;
        for (int t = 0; t < T; ++t) {
            const uint32_t c = hist[t][k];
            hist[t][k] = pos;
            pos += c;
        }
    }
    start[K] = pos;
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const size_t i0 = n * (size_t)t / T, i1 = n * (size_t)(t + 1) / T;
        uint32_t* h = hist[t].data();
        for (size_t i = i0; i < i1; ++i) members[h[lab[i]]++] = (uint32_t)i;
    }
}

// means of the members (P: n x dim rows). Empty clusters keep the row of their seed point. sums optional (K x dim).
// (P: rows of pstride floats, the first dim are used; C: rows of cstride floats; sums: K x dim)
static void group_means(const float* P, size_t pstride, size_t dim, const std::vector<uint32_t>& start,
                        const std::vector<uint32_t>& members, size_t K, const uint32_t* seedpt, float* C,
                        size_t cstride, double* sums) {
#pragma omp parallel
    {
        std::vector<double> loc(sums ? 0 : dim);
#pragma omp for schedule(dynamic, 8)
        for (size_t k = 0; k < K; ++k) {
            double* acc = sums ? &sums[k * dim] : loc.data();
            for (size_t t = 0; t < dim; ++t) acc[t] = 0.0;
            const uint32_t m0 = start[k], m1 = start[k + 1];
            for (uint32_t m = m0; m < m1; ++m) {
                const float* x = &P[(size_t)members[m] * pstride];
#pragma omp simd
                for (size_t t = 0; t < dim; ++t) acc[t] += x[t];
            }
            float* c = &C[k * cstride];
            if (m1 > m0) {
                const double inv = 1.0 / (double)(m1 - m0);
                for (size_t t = 0; t < dim; ++t) c[t] = (float)(acc[t] * inv);
            } else {
                memcpy(c, &P[(size_t)seedpt[k] * pstride], dim * sizeof(float));
            }
        }
    }
}

// Kc nearest centroids of every centroid (rows of C: K x dim), self excluded
static void build_neigh(const float* C, size_t stride, const float* cn, size_t K, size_t dim, size_t Kc,
                        std::vector<uint32_t>& neigh) {
    neigh.resize(K * Kc);
    const size_t RB = 64, KT = 1024;
#pragma omp parallel
    {
        std::vector<float> G(RB * KT), Dm(RB * K), vals(K);
#pragma omp for schedule(dynamic, 1)
        for (size_t cs = 0; cs < K; cs += RB) {
            const size_t nb = std::min(RB, K - cs);
            for (size_t k0 = 0; k0 < K; k0 += KT) {
                const size_t kt = std::min(KT, K - k0);
                gemm_rows(&C[k0 * stride], (int)kt, (int)stride, &C[cs * stride], (int)nb, (int)stride, (int)dim, G.data());
                for (size_t i = 0; i < nb; ++i) {
                    const float* g = &G[i * kt];
                    float* o = &Dm[i * K + k0];
                    const float* cnk = &cn[k0];
#pragma omp simd
                    for (size_t k = 0; k < kt; ++k) o[k] = cnk[k] - 2.0f * g[k];
                }
            }
            for (size_t i = 0; i < nb; ++i) {
                float* g = &Dm[i * K];
                g[cs + i] = 3.0e38f;
                memcpy(vals.data(), g, K * sizeof(float));
                std::nth_element(vals.begin(), vals.begin() + (Kc - 1), vals.end());
                const float thr = vals[Kc - 1]; // the Kc-th smallest value
                uint32_t* out = &neigh[(cs + i) * Kc];
                size_t m = 0;
                for (size_t k = 0; k < K && m < Kc; ++k)
                    if (g[k] < thr) out[m++] = (uint32_t)k;
                for (size_t k = 0; k < K && m < Kc; ++k) // ties at the threshold
                    if (g[k] == thr) out[m++] = (uint32_t)k;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------
// 3 / 4. proposals of one candidate-restricted sweep over the clusters [kb, ke).
//   SKETCH : costs are sketch distances (X, C, rx, rc unused)
//   EXACT  : costs are full-dimensional distances; the sketch + residual-norm bound decides which
//            candidates have to be evaluated in full dimension
//   weighted      : Hartigan costs b/(b+1) d to join and b/(b-1) d to stay; otherwise plain nearest centroid
// ------------------------------------------------------------------------------------------------------------
struct SweepIn {
    const float *X = nullptr, *Y = nullptr, *y2 = nullptr, *rx = nullptr, *xc2 = nullptr; // points
    const float *C = nullptr, *Cs = nullptr, *cn = nullptr, *rc = nullptr, *cc2 = nullptr; // centroids
    const float *y2s = nullptr, *cns = nullptr; // squared norms over the first ls sketch coordinates
    const float* Y2 = nullptr;                  // sketch extension: coordinates ystride .. l of every point
    const float *Cc = nullptr, *kq = nullptr;   // centred centroids c - mu and |c - mu|^2 + 2 <mu, c - mu>
    size_t n = 0, d = 0, l = 0, ls = 0, ystride = 0, K = 0, Kc = 0;
};
enum { SKETCH = 0, EXACT = 2 };

static double sweep_proposals(const SweepIn& in, const std::vector<int64_t>& cnt, const std::vector<uint32_t>& gstart,
                              const std::vector<uint32_t>& gmem, const std::vector<uint32_t>& neigh, int mode,
                              bool weighted, uint32_t* prop, float* dstay, float* dbest, size_t kb, size_t ke,
                              size_t kstep = 1) {
    const bool exact = mode == EXACT;
    const size_t l = in.l, d = in.d, Kc = in.Kc, W = Kc + 1, RB = 256;
    const size_t lw = exact ? l : in.ls; // coordinates of the GEMM
    const float *y2w = exact ? in.y2 : in.y2s, *cnw = exact ? in.cn : in.cns;
    double evals = 0;
#pragma omp parallel reduction(+ : evals)
    {
        std::vector<float> Cp(W * lw), Yg(RB * lw), G(RB * W), cnl(W), wq(W), rcl(W), slc(W), lbv(W), kql(W);
        std::vector<uint32_t> sidx(W);
        double ev = 0;
#pragma omp for schedule(dynamic, 4)
        for (size_t pi = 0; pi < (ke - kb + kstep - 1) / kstep; ++pi) {
            const size_t p = kb + pi * kstep;
            const uint32_t m0 = gstart[p], m1 = gstart[p + 1];
            if (m0 == m1) continue;
            const uint32_t* nb = &neigh[p * Kc];
            for (size_t j = 0; j < W; ++j) {
                const size_t q = j ? nb[j - 1] : p;
                memcpy(&Cp[j * lw], &in.Cs[q * l], lw * sizeof(float));
                cnl[j] = cnw[q];
                wq[j] = weighted ? (float)cnt[q] / (float)(cnt[q] + 1) : 1.0f;
                if (exact) {
                    rcl[j] = in.rc[q];
                    slc[j] = 4.0e-6f * in.cc2[q];
                    kql[j] = in.kq[q];
                }
            }
            const int64_t bp = cnt[p];
            const float wstay = (weighted && bp > 1) ? (float)bp / (float)(bp - 1) : 1.0f;
            const bool frozen = weighted && bp <= 1; // leaving a singleton can never lower J
            for (uint32_t ms = m0; ms < m1; ms += RB) {
                const size_t tn = std::min<size_t>(RB, m1 - ms);
                for (size_t i = 0; i < tn; ++i) {
                    const size_t id = gmem[ms + i], l1 = std::min(lw, in.ystride);
                    memcpy(&Yg[i * lw], &in.Y[id * in.ystride], l1 * sizeof(float));
                    if (lw > l1) memcpy(&Yg[i * lw + l1], &in.Y2[id * (l - in.ystride)], (lw - l1) * sizeof(float));
                }
                gemm_rows(Cp.data(), (int)W, (int)lw, Yg.data(), (int)tn, (int)lw, (int)lw, G.data());
                for (size_t i = 0; i < tn; ++i) {
                    const uint32_t id = gmem[ms + i];
                    const float* g = &G[i * W];
                    const float yi2 = y2w[id];
                    if (!exact) {
                        const float dp = std::max(yi2 + cnl[0] - 2.0f * g[0], 0.0f);
                        dstay[id] = dp;
                        prop[id] = (uint32_t)p;
                        dbest[id] = dp;
                        if (frozen) continue;
                        float best = wstay * dp;
                        size_t bj = 0;
                        float bd = dp;
                        for (size_t j = 1; j < W; ++j) {
                            const float dq = std::max(yi2 + cnl[j] - 2.0f * g[j], 0.0f);
                            const float v = dq * wq[j];
                            if (v < best) {
                                best = v;
                                bj = j;
                                bd = dq;
                            }
                        }
                        if (bj) {
                            prop[id] = nb[bj - 1];
                            dbest[id] = bd;
                        }
                        continue;
                    }
                    // |x - c|^2 = |x - mu|^2 + |c - mu|^2 - 2 <x - mu, c - mu> : one dot product per evaluation
                    const float* x = &in.X[(size_t)id * d];
                    const float xc = in.xc2[id];
                    const float D0 = std::max(xc + kql[0] - 2.0f * dotp(x, &in.Cc[p * d], d), 0.0f);
                    ev += 1;
                    dstay[id] = D0;
                    prop[id] = (uint32_t)p;
                    dbest[id] = D0;
                    if (frozen) continue;
                    float tau = wstay * D0;
                    const float r = in.rx[id], slx = 4.0e-6f * in.xc2[id];
#pragma omp simd
                    for (size_t j = 1; j < W; ++j) {
                        const float S = std::max(yi2 + cnl[j] - 2.0f * g[j], 0.0f);
                        const float df = r - rcl[j];
                        lbv[j] = (S + df * df - slx - slc[j]) * wq[j];
                    }
                    size_t ns = 0;
                    for (size_t j = 1; j < W; ++j)
                        if (lbv[j] < tau || !g_prune) sidx[ns++] = (uint32_t)j;
                    // best-first: smallest bound first, stop when no bound beats the best exact cost so far
                    std::sort(sidx.begin(), sidx.begin() + ns, [&](uint32_t a, uint32_t b) { return lbv[a] < lbv[b]; });
                    for (size_t t = 0; t < ns; ++t) {
                        const size_t j = sidx[t];
                        if (g_prune && lbv[j] >= tau) break;
                        const float Dq = std::max(xc + kql[j] - 2.0f * dotp(x, &in.Cc[(size_t)nb[j - 1] * d], d), 0.0f);
                        ev += 1;
                        const float v = Dq * wq[j];
                        if (v < tau) {
                            tau = v;
                            prop[id] = nb[j - 1];
                            dbest[id] = Dq;
                        }
                    }
                }
            }
        }
        evals += ev;
    }
    return evals;
}

// sequential commit with live sizes (a cluster is never emptied). moved: (point, from, to).
// The points that propose a move are collected in parallel as compact records, so that the sequential part
// reads contiguous memory.
struct Want {
    uint32_t i, p, q;
    float ds, db;
};
static size_t commit(const uint32_t* ids, size_t n, uint32_t* lab, const uint32_t* prop, const float* dstay,
                     const float* dbest, std::vector<int64_t>& cnt, bool weighted, std::vector<uint32_t>* moved) {
    size_t moves = 0;
    if (moved) moved->clear();
    const int T = omp_get_max_threads();
    std::vector<std::vector<Want>> want(T);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const size_t u0 = n * (size_t)t / T, u1 = n * (size_t)(t + 1) / T;
        for (size_t u = u0; u < u1; ++u) {
            const uint32_t i = ids ? ids[u] : (uint32_t)u;
            if (prop[i] != lab[i]) want[t].push_back({i, lab[i], prop[i], dstay[i], dbest[i]});
        }
    }
    for (int t = 0; t < T; ++t)
        for (const Want& w : want[t]) {
            if (weighted) {
                const int64_t bp = cnt[w.p], bq = cnt[w.q];
                if (bp <= 1) continue;
                const double cq = (double)bq / (double)(bq + 1) * w.db;
                const double cp = (double)bp / (double)(bp - 1) * w.ds;
                if (!(cq < cp)) continue;
            }
            --cnt[w.p];
            ++cnt[w.q];
            lab[w.i] = w.q;
            ++moves;
            if (moved) {
                moved->push_back(w.i);
                moved->push_back(w.p);
                moved->push_back(w.q);
            }
        }
    return moves;
}

// sums[k] -= / += the rows of the moved points (moved: point, from, to), one worker per touched cluster.
// es: K + 1 zeros on entry and on exit.
static void apply_moves(const float* P, size_t pstride, size_t dim, const std::vector<uint32_t>& moved, double* sums,
                        std::vector<uint32_t>& es, std::vector<uint32_t>& ev, std::vector<uint32_t>& touched) {
    const size_t mv = moved.size() / 3;
    touched.clear();
    for (size_t e = 0; e < mv; ++e) {
        const uint32_t p = moved[3 * e + 1], q = moved[3 * e + 2];
        if (es[p]++ == 0) touched.push_back(p);
        if (es[q]++ == 0) touched.push_back(q);
    }
    std::vector<uint32_t> off(touched.size() + 1);
    uint32_t pos = 0;
    for (size_t t = 0; t < touched.size(); ++t) {
        const uint32_t k = touched[t];
        off[t] = pos;
        pos += es[k];
        es[k] = off[t];
    }
    off[touched.size()] = pos;
    ev.resize(2 * mv);
    for (size_t e = 0; e < mv; ++e) {
        ev[es[moved[3 * e + 1]]++] = (uint32_t)(2 * e);     // leave
        ev[es[moved[3 * e + 2]]++] = (uint32_t)(2 * e + 1); // join
    }
#pragma omp parallel for schedule(dynamic, 16) if (mv > 512)
    for (size_t t = 0; t < touched.size(); ++t) {
        double* a = &sums[(size_t)touched[t] * dim];
        for (uint32_t u = off[t]; u < off[t + 1]; ++u) {
            const float* x = &P[(size_t)moved[3 * (size_t)(ev[u] >> 1)] * pstride];
            if (ev[u] & 1) {
#pragma omp simd
                for (size_t j = 0; j < dim; ++j) a[j] += x[j];
            } else {
#pragma omp simd
                for (size_t j = 0; j < dim; ++j) a[j] -= x[j];
            }
        }
    }
    for (uint32_t k : touched) es[k] = 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: sketchcd <base.fvecs> <K> <out_prefix|-> [key=value ...]\n");
        return 2;
    }
    const std::string base = argv[1], prefix = argv[3];
    const size_t K = std::atoll(argv[2]);
    std::map<std::string, std::string> opt = {
        {"dp", "0"},     {"ds", "0"},          {"nsample", "16384"}, {"power", "2"},   {"over", "8"},     {"mode", "auto"}, {"meso", "0"},
        {"kflat", "0"},  {"d1", "0"},          {"rho", "0.25"},  {"bmin", "256"},   {"bmax", "16384"},
        {"weighted", "1"}, {"sweeps", "6"},    {"xsweeps", "2"}, {"fnn", "1"},      {"candk", "128"},
        {"seed", "42"},  {"limit", "0"},       {"B", "4096"},    {"replays", "2"},  {"gs", "1"},
        {"fcandk", "256"}, {"omega", "1.6"}, {"fupd", "1"}, {"nbr", "2"}, {"ext", "-1"}, {"pthr", "48"}, {"prune", "1"}};
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        const size_t p = a.find('=');
        if (p == std::string::npos || !opt.count(a.substr(0, p))) {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
        opt[a.substr(0, p)] = a.substr(p + 1);
    }
    auto geti = [&](const char* k) { return (long long)std::atoll(opt[k].c_str()); };
    const size_t nsample = geti("nsample"), bmin = geti("bmin"), bmax = geti("bmax"), limit = geti("limit"),
                 Bfixed = geti("B");
    const int power = (int)geti("power"), over = (int)geti("over");
    int sweeps = (int)geti("sweeps"), xsweeps = (int)geti("xsweeps");
    // over-relaxed centroid update after a sweep: c <- c_start + omega (mean - c_start). The output is always the
    // exact mean of the members.
    const float omega = (float)std::atof(opt["omega"].c_str());
    const bool fnn = geti("fnn") != 0, weighted = geti("weighted") != 0;
    const double rho = std::atof(opt["rho"].c_str());
    const int seed = (int)geti("seed"), replays = (int)geti("replays");
    size_t replay_moves = 0;
    const size_t Kc = std::min<size_t>(geti("candk"), K - 1);
    std::mt19937 rng(seed), rng_seeds(seed + 7919);
    const int T = omp_get_max_threads();
    openblas_set_num_threads(1);
    g_prune = geti("prune") != 0;
    // every sweep visits the clusters in nch chunks; moves are committed and the touched centroids refreshed
    // after every chunk (block Gauss-Seidel). gs=1 is the Jacobi sweep.
    const size_t nch = std::max<size_t>(1, std::min<size_t>((size_t)geti("gs"), K / (16 * (size_t)T)));

    // ---- load ----
    FILE* f = fopen(base.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", base.c_str());
        return 2;
    }
    int32_t dim = 0;
    if (fread(&dim, 4, 1, f) != 1) return 2;
    fseek(f, 0, SEEK_END);
    const size_t d = dim, rowb = 4 + d * 4;
    size_t n = (size_t)ftell(f) / rowb;
    if (limit && limit < n) n = limit;
    fseek(f, 0, SEEK_SET);
    std::unique_ptr<float[]> Xown(new float[n * d]);
    float* X = Xown.get();
    {
        const size_t chunk = 16384;
        std::vector<char> buf(chunk * rowb);
        for (size_t s = 0; s < n; s += chunk) {
            const size_t nb = std::min(chunk, n - s);
            if (fread(buf.data(), rowb, nb, f) != nb) return 2;
#pragma omp parallel for
            for (size_t i = 0; i < nb; ++i) memcpy(&X[(s + i) * d], &buf[i * rowb + 4], d * 4);
        }
    }
    fclose(f);
    if (K < 2 || K > n) {
        fprintf(stderr, "bad K\n");
        return 2;
    }

    const auto t_total = clk::now();
    // ---- 1. sketch ----
    // Sketch dimensions. The first pass and the sketch sweeps use the first ls coordinates, the exact bound all l.
    //   ds=0 : ls = d up to 256 dimensions (no sketch: the projection would not pay for itself), otherwise 128 per
    //          1536 input dimensions, between 128 and 384 (d = 3072: 256 is both faster and better than 128)
    //   dp=0 : l = ls at first. Before the exact stage a pilot on every 64th cluster (at least 32 clusters) measures
    //          how many candidates the bound leaves to verify; if there are too many (a slowly decaying spectrum), the
    //          sketch is extended by `ext` further principal directions, which only the bound uses. With the default
    //          sketch dimensions this never happens on the datasets of the paper (pilot estimates below 23 of 128).
    const size_t ls_auto =
        d <= 256 ? d : std::min<size_t>(384, std::max<size_t>(128, 64 * (size_t)std::llround((double)d / 768.0)));
    size_t ls = (size_t)geti("ds");
    if (ls == 0) ls = ls_auto;
    size_t l = (size_t)geti("dp");
    if (l == 0) l = ls;
    l = std::min(l, d);
    ls = std::min(ls, l);
    const bool reduced = l < d;
    Sketch sk;
    std::unique_ptr<float[]> Yown;
    const float* Y = X;
    std::unique_ptr<float[]> y2(new float[n]), y2s(new float[n]), rx, xc2;
    double basis_ms = 0, proj_ms = 0;
    if (reduced) {
        const auto t0 = clk::now();
        pca_basis(X, n, d, (int)l, nsample, power, over, rng, sk);
        basis_ms = ms_since(t0);
        const auto t1 = clk::now();
        Yown.reset(new float[n * l]);
        rx.reset(new float[n]);
        xc2.reset(new float[n]);
        float* Yw = Yown.get();
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t s = 0; s < n; s += 256) {
            const size_t nb = std::min<size_t>(256, n - s);
            gemm_rows(sk.Q.data(), (int)l, (int)d, &X[s * d], (int)nb, (int)d, (int)d, &Yw[s * l]);
            for (size_t i = 0; i < nb; ++i) {
                float* y = &Yw[(s + i) * l];
                float yy = 0;
#pragma omp simd reduction(+ : yy)
                for (size_t t = 0; t < l; ++t) {
                    y[t] -= sk.qmu[t];
                    yy += y[t] * y[t];
                }
                const float xx = sqdist(&X[(s + i) * d], sk.mu.data(), d);
                y2s[s + i] = sqnorm(y, ls);
                y2[s + i] = yy;
                xc2[s + i] = xx;
                rx[s + i] = std::sqrt(std::max(xx - yy, 0.0f));
            }
        }
        Y = Yw;
        proj_ms = ms_since(t1);
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) y2[i] = y2s[i] = sqnorm(&X[i * d], d);
        sweeps += xsweeps; // the sketch is the full space: every sweep is exact
        xsweeps = 0;
    }
    const double sketch_ms = basis_ms + proj_ms;

    // ---- 2. first pass ----
    const auto t_pass = clk::now();
    int dq = (int)geti("d1");
    if (dq <= 0 || dq > (int)l) dq = (int)ls;
    std::string mode = opt["mode"];
    if (mode == "auto") mode = K <= (size_t)geti("kflat") ? "flat" : "two";
    std::unique_ptr<uint32_t[]> lab(new uint32_t[n]);
    std::vector<uint32_t> seedpt(K);
    size_t refreshes = 0, M = 0;
    double meso_ms = 0, route_ms = 0, cell_ms = 0, cell_cv = 0;
    if (mode == "flat") {
        std::vector<uint32_t> ord(n);
        std::iota(ord.begin(), ord.end(), 0u);
        for (size_t i = 0; i < K; ++i) std::swap(ord[i], ord[i + rng_seeds() % (n - i)]);
        std::copy(ord.begin(), ord.begin() + K, seedpt.begin());
        std::iota(ord.begin(), ord.end(), 0u);
        std::shuffle(ord.begin(), ord.end(), rng_seeds);
        refreshes = online_pass(Y, l, dq, ord.data(), n, seedpt.data(), K, 0, lab.get(), weighted, rho, Bfixed, bmin,
                                bmax, true, T, replays, &replay_moves);
    } else {
        // meso centroids from a sample
        const auto tm = clk::now();
        M = geti("meso") > 0 ? (size_t)geti("meso") : (size_t)std::llround(std::sqrt((double)K));
        M = std::max<size_t>(2, std::min(M, K));
        const size_t ns2 = std::min(n, std::max<size_t>(32768, 256 * M));
        std::vector<uint32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0u);
        for (size_t i = 0; i < ns2; ++i) std::swap(ids[i], ids[i + rng_seeds() % (n - i)]);
        online_pass(Y, l, dq, ids.data(), ns2, ids.data() + (ns2 - M), M, 0, lab.get(), weighted, rho, Bfixed,
                    std::max<size_t>(16, std::min<size_t>(256, M / 8)), bmax, true, T);
        std::vector<float> Cm(M * (size_t)dq), cmn(M);
        {
            std::vector<double> acc(M * (size_t)dq, 0.0);
            std::vector<size_t> c(M, 0);
            for (size_t i = 0; i < ns2; ++i) {
                const uint32_t k = lab[ids[i]];
                const float* y = &Y[(size_t)ids[i] * l];
                for (int t = 0; t < dq; ++t) acc[k * dq + t] += y[t];
                ++c[k];
            }
            for (size_t k = 0; k < M; ++k) {
                const float* sd = &Y[(size_t)ids[ns2 - M + k] * l];
                for (int t = 0; t < dq; ++t)
                    Cm[k * dq + t] = c[k] ? (float)(acc[k * dq + t] / (double)c[k]) : sd[t];
                cmn[k] = sqnorm(&Cm[k * dq], dq);
            }
        }
        meso_ms = ms_since(tm);
        // route every point to its nearest meso centroid
        const auto tr = clk::now();
        std::unique_ptr<uint32_t[]> mlab(new uint32_t[n]);
#pragma omp parallel
        {
            std::vector<float> G(256 * M), yg(256 * (size_t)dq);
#pragma omp for schedule(dynamic, 4)
            for (size_t s = 0; s < n; s += 256) {
                const size_t nb = std::min<size_t>(256, n - s);
                const float* rows = &Y[s * l];
                int ld = (int)l;
                if ((size_t)dq != l) { // BLAS needs the leading dq columns as a contiguous block
                    for (size_t i = 0; i < nb; ++i) memcpy(&yg[i * dq], &Y[(s + i) * l], dq * sizeof(float));
                    rows = yg.data();
                    ld = dq;
                }
                gemm_rows(Cm.data(), (int)M, dq, rows, (int)nb, ld, dq, G.data());
                for (size_t i = 0; i < nb; ++i) {
                    const float* g = &G[i * M];
                    float mn = 3.0e38f;
                    uint32_t bk = 0;
                    for (size_t k = 0; k < M; ++k) {
                        const float v = cmn[k] - 2.0f * g[k];
                        if (v < mn) {
                            mn = v;
                            bk = (uint32_t)k;
                        }
                    }
                    mlab[s + i] = bk;
                }
            }
        }
        std::vector<uint32_t> cstart, cmem;
        build_groups(mlab.get(), n, M, cstart, cmem);
        route_ms = ms_since(tr);
        // clusters per cell proportional to the cell size (largest remainder), at least 1, at most the cell size
        const auto tc = clk::now();
        std::vector<size_t> Km(M, 0);
        {
            std::vector<std::pair<double, size_t>> frac;
            size_t used = 0, nonempty = 0;
            for (size_t m = 0; m < M; ++m) {
                const size_t nm = cstart[m + 1] - cstart[m];
                if (!nm) continue;
                ++nonempty;
                const double q = (double)K * (double)nm / (double)n;
                Km[m] = std::min(nm, std::max<size_t>(1, (size_t)q));
                used += Km[m];
                frac.push_back({q - std::floor(q), m});
            }
            std::sort(frac.begin(), frac.end(), [](auto& a, auto& b) { return a.first > b.first; });
            for (size_t r = 0; used < K; ++r) { // hand out the remainder
                const size_t m = frac[r % frac.size()].second;
                if (Km[m] < cstart[m + 1] - cstart[m]) {
                    ++Km[m];
                    ++used;
                }
            }
            for (size_t r = 0; used > K; ++r) { // the minimum of 1 can overshoot: take back from the largest
                size_t bm = 0;
                for (size_t m = 0; m < M; ++m)
                    if (Km[m] > Km[bm]) bm = m;
                --Km[bm];
                --used;
            }
            (void)nonempty;
        }
        std::vector<uint32_t> kbase(M + 1, 0), cells(M);
        double s1 = 0, s2 = 0;
        for (size_t m = 0; m < M; ++m) {
            kbase[m + 1] = kbase[m] + (uint32_t)Km[m];
            const double nm = (double)(cstart[m + 1] - cstart[m]);
            s1 += nm;
            s2 += nm * nm;
        }
        cell_cv = std::sqrt(std::max(s2 / M - (s1 / M) * (s1 / M), 0.0)) / (s1 / M);
        std::iota(cells.begin(), cells.end(), 0u);
        std::sort(cells.begin(), cells.end(), [&](uint32_t a, uint32_t b) {
            return (double)(cstart[a + 1] - cstart[a]) * Km[a] > (double)(cstart[b + 1] - cstart[b]) * Km[b];
        });
        size_t refs = 0, rmv = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : refs, rmv)
        for (size_t ci = 0; ci < M; ++ci) {
            const size_t m = cells[ci], nm = cstart[m + 1] - cstart[m], km = Km[m];
            if (!nm) continue;
            uint32_t* seg = &cmem[cstart[m]];
            std::mt19937 r2((uint32_t)(seed * 2654435761u + 97u * (uint32_t)m + 13u));
            std::shuffle(seg, seg + nm, r2);
            const uint32_t* sd = seg + (nm - km); // seeds: the last km points of the shuffled cell
            for (size_t j = 0; j < km; ++j) seedpt[kbase[m] + j] = sd[j];
            size_t mv = 0;
            refs += online_pass(Y, l, dq, seg, nm, sd, km, kbase[m], lab.get(), weighted, rho, Bfixed,
                                std::max<size_t>(16, std::min<size_t>(256, km / 8)), bmax, false, T, replays, &mv);
            rmv += mv;
        }
        refreshes = refs;
        replay_moves = rmv;
        cell_ms = ms_since(tc);
    }
    const double pass_ms = ms_since(t_pass);

    // ---- 3. candidate-restricted Hartigan sweeps in sketch space ----
    const auto t_sw = clk::now();
    std::vector<uint32_t> gstart, gmem, neigh;
    std::vector<int64_t> cnt(K);
    std::vector<float> Cs(K * l), cn(K), cns(K); // sketch centroids (rows of l floats), norms over l / ls coordinates
    std::unique_ptr<uint32_t[]> prop(new uint32_t[n]);
    std::unique_ptr<float[]> dstay(new float[n]), dbest(new float[n]);
    auto regroup = [&]() {
        build_groups(lab.get(), n, K, gstart, gmem);
        for (size_t k = 0; k < K; ++k) cnt[k] = gstart[k + 1] - gstart[k];
    };
    std::vector<double> ssum(sweeps > 0 ? K * ls : 0);
    auto sketch_centroids = [&]() { // the first ls coordinates only; the exact stage fills all l from the full means
        group_means(Y, l, ls, gstart, gmem, K, seedpt.data(), Cs.data(), l, ssum.empty() ? nullptr : ssum.data());
#pragma omp parallel for
        for (size_t k = 0; k < K; ++k) cns[k] = sqnorm(&Cs[k * l], ls);
    };
    std::vector<uint32_t> moved, touched, es(K + 1, 0), ev;
    PROF("s.regroup", regroup());
    size_t empty_after_pass = 0;
    for (size_t k = 0; k < K; ++k) empty_after_pass += cnt[k] == 0;
    PROF("s.means", sketch_centroids());
    SweepIn in;
    in.X = X, in.Y = Y, in.y2 = y2.get(), in.Cs = Cs.data(), in.cn = cn.data();
    in.y2s = y2s.get(), in.cns = cns.data();
    in.n = n, in.d = d, in.l = l, in.ls = ls, in.ystride = l, in.K = K, in.Kc = Kc;
    std::string moves_log;
    size_t sweep_moves = 0;
    const int nbr = std::max(1, (int)geti("nbr"));
    int since_build = nbr; // sweeps since the neighbour lists were built
    std::vector<float> Cs0;
    for (int sw = 0; sw < sweeps; ++sw) {
        if (omega != 1.0f) Cs0 = Cs;
        if (since_build >= nbr) {
            PROF("s.neigh", build_neigh(Cs.data(), l, cns.data(), K, ls, Kc, neigh));
            since_build = 0;
        }
        ++since_build;
        size_t mvs = 0;
        for (size_t c = 0; c < nch; ++c) {
            const size_t kb = K * c / nch, ke = K * (c + 1) / nch;
            PROF("s.prop", sweep_proposals(in, cnt, gstart, gmem, neigh, SKETCH, weighted, prop.get(), dstay.get(),
                                           dbest.get(), kb, ke));
            size_t mv = 0;
            PROF("s.commit", mv = commit(&gmem[gstart[kb]], gstart[ke] - gstart[kb], lab.get(), prop.get(), dstay.get(),
                                         dbest.get(), cnt, weighted, &moved));
            mvs += mv;
            if (!mv) continue;
            const auto tu = clk::now();
            apply_moves(Y, l, ls, moved, ssum.data(), es, ev, touched);
#pragma omp parallel for schedule(dynamic, 32) if (touched.size() > 256)
            for (size_t t = 0; t < touched.size(); ++t) {
                const size_t k = touched[t];
                if (cnt[k] <= 0) continue;
                const double inv = 1.0 / (double)cnt[k];
                float* c2 = &Cs[k * l];
                for (size_t j = 0; j < ls; ++j) c2[j] = (float)(ssum[k * ls + j] * inv);
                cns[k] = sqnorm(c2, ls);
            }
            g_prof["s.update"] += ms_since(tu);
        }
        sweep_moves += mvs;
        moves_log += (moves_log.empty() ? "" : ",") + std::to_string(mvs);
        PROF("s.regroup", regroup());
        if (omega != 1.0f) { // all centroids: relaxed between sweeps, the exact means after the last one
            const float om = sw + 1 < sweeps ? omega : 1.0f;
#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < K; ++k) {
                float *c2 = &Cs[k * l], *c0 = &Cs0[k * l];
                if (cnt[k] > 0)
                    for (size_t j = 0; j < ls; ++j) c2[j] = c0[j] + om * ((float)(ssum[k * ls + j] / (double)cnt[k]) - c0[j]);
                cns[k] = sqnorm(c2, ls);
            }
        }
    }
    const double sweep_ms = ms_since(t_sw);

    // ---- 4. full-dimensional centroids ----
    const auto t_c = clk::now();
    std::unique_ptr<float[]> Cown;
    std::unique_ptr<double[]> csum;
    const float* C = Cs.data(); // not reduced: the sketch centroids are the centroids
    std::vector<float> rc, cc2, kq;
    std::unique_ptr<float[]> Ccown; // centred centroids c - mu, for the dot-product form of the exact distance
    auto centre = [&](size_t k) {
        const float* cf = &Cown[k * d];
        float* cc = &Ccown[k * d];
        double md = 0;
        for (size_t j = 0; j < d; ++j) {
            cc[j] = cf[j] - sk.mu[j];
            md += (double)sk.mu[j] * cc[j];
        }
        kq[k] = cc2[k] + 2.0f * (float)md;
    };
    auto project_centroids = [&]() { // Cs = (C - mu) Q^T, residual norms of the centroids
        const float* Cf = Cown.get();
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t s = 0; s < K; s += 256) {
            const size_t nb = std::min<size_t>(256, K - s);
            gemm_rows(sk.Q.data(), (int)l, (int)d, &Cf[s * d], (int)nb, (int)d, (int)d, &Cs[s * l]);
            for (size_t i = 0; i < nb; ++i) {
                float* y = &Cs[(s + i) * l];
                for (size_t t = 0; t < l; ++t) y[t] -= sk.qmu[t];
                const float yy = sqnorm(y, l), xx = sqdist(&Cf[(s + i) * d], sk.mu.data(), d);
                cn[s + i] = yy;
                cc2[s + i] = xx;
                rc[s + i] = std::sqrt(std::max(xx - yy, 0.0f));
                centre(s + i);
            }
        }
    };
    if (reduced) {
        Cown.reset(new float[K * d]);
        Ccown.reset(new float[K * d]);
        csum.reset(new double[K * d]);
        rc.resize(K);
        cc2.resize(K);
        kq.resize(K);
        PROF("c.means", group_means(X, d, d, gstart, gmem, K, seedpt.data(), Cown.get(), d, csum.get()));
        PROF("c.project", project_centroids());
        C = Cown.get();
        in.C = C, in.rx = rx.get(), in.xc2 = xc2.get(), in.rc = rc.data(), in.cc2 = cc2.data();
        in.Cc = Ccown.get(), in.kq = kq.data();
    }
    const double cent_ms = ms_since(t_c);

    // after apply_moves on csum: new means of the touched clusters, their sketches and residual norms
    auto refresh_full = [&]() {
        float* Cw = Cown.get();
#pragma omp parallel for schedule(dynamic, 8) if (touched.size() > 64)
        for (size_t t = 0; t < touched.size(); ++t) {
            const size_t k = touched[t];
            if (cnt[k] <= 0) continue; // emptied (only possible with weighted=0): keep the old position
            const double inv = 1.0 / (double)cnt[k];
            float* cf = &Cw[k * d];
            for (size_t j = 0; j < d; ++j) cf[j] = (float)(csum[k * d + j] * inv);
            float* y = &Cs[k * l];
            gemm_rows(sk.Q.data(), (int)l, (int)d, cf, 1, (int)d, (int)d, y);
            for (size_t j = 0; j < l; ++j) y[j] -= sk.qmu[j];
            const float yy = sqnorm(y, l), xx = sqdist(cf, sk.mu.data(), d);
            cn[k] = yy;
            cc2[k] = xx;
            rc[k] = std::sqrt(std::max(xx - yy, 0.0f));
            centre(k);
        }
    };

    // ---- 4b. is the bound tight enough? pilot on every 64th cluster; extend the sketch if it is not ----
    const auto t_ext = clk::now();
    std::unique_ptr<float[]> Y2own;
    double pilot_evals = -1.0;
    size_t ext = 0;
    if (reduced && (xsweeps > 0 || fnn) && geti("ext") != 0 && l < d) {
        bool extend = geti("ext") > 0;
        if (!extend) {
            PROF("p.neigh", build_neigh(Cs.data(), l, cn.data(), K, l, Kc, neigh));
            since_build = 0;
            const size_t kstep = std::max<size_t>(1, std::min<size_t>(64, K / 32));
            size_t npts = 0;
            for (size_t k = 0; k < K; k += kstep) npts += (size_t)cnt[k];
            double pe = 0;
            PROF("p.pilot", pe = sweep_proposals(in, cnt, gstart, gmem, neigh, EXACT, weighted, prop.get(), dstay.get(),
                                                 dbest.get(), 0, K, kstep));
            pilot_evals = pe / (double)std::max<size_t>(npts, 1);
            extend = pilot_evals > std::atof(opt["pthr"].c_str());
        }
        if (extend) {
            ext = std::min<size_t>(geti("ext") > 0 ? (size_t)geti("ext") : 128, d - l);
            const size_t l0 = l;
            PROF("p.basis", pca_basis(X, n, d, (int)ext, nsample, power, over, rng, sk, (int)l0));
            const auto tq = clk::now();
            Y2own.reset(new float[n * ext]);
            float* Y2w = Y2own.get();
#pragma omp parallel for schedule(dynamic, 4)
            for (size_t s = 0; s < n; s += 256) {
                const size_t nb = std::min<size_t>(256, n - s);
                gemm_rows(&sk.Q[l0 * d], (int)ext, (int)d, &X[s * d], (int)nb, (int)d, (int)d, &Y2w[s * ext]);
                for (size_t i = 0; i < nb; ++i) {
                    float* y = &Y2w[(s + i) * ext];
                    float yy = 0;
                    for (size_t t = 0; t < ext; ++t) {
                        y[t] -= sk.qmu[l0 + t];
                        yy += y[t] * y[t];
                    }
                    y2[s + i] += yy;
                    rx[s + i] = std::sqrt(std::max(xc2[s + i] - y2[s + i], 0.0f));
                }
            }
            g_prof["p.proj"] += ms_since(tq);
            l = l0 + ext;
            Cs.assign(K * l, 0.0f);
            project_centroids();
            in.Cs = Cs.data(), in.l = l, in.Y2 = Y2own.get();
            since_build = nbr;
        }
    }
    const double ext_ms = ms_since(t_ext);

    // ---- 5b. exact sweeps: sketch bound + full-dimensional verification ----
    const auto t_xs = clk::now();
    size_t xsweep_moves = 0;
    double xevals = 0;
    std::vector<float> C0;
    for (int sw = 0; sw < xsweeps; ++sw) {
        if (omega != 1.0f) C0.assign(Cown.get(), Cown.get() + K * d);
        if (since_build >= nbr) {
            PROF("x.neigh", build_neigh(Cs.data(), l, cn.data(), K, l, Kc, neigh));
            since_build = 0;
        }
        ++since_build;
        size_t mvs = 0;
        for (size_t c = 0; c < nch; ++c) {
            const size_t kb = K * c / nch, ke = K * (c + 1) / nch;
            PROF("x.prop", xevals += sweep_proposals(in, cnt, gstart, gmem, neigh, EXACT, weighted, prop.get(),
                                                     dstay.get(), dbest.get(), kb, ke));
            size_t mv = 0;
            PROF("x.commit", mv = commit(&gmem[gstart[kb]], gstart[ke] - gstart[kb], lab.get(), prop.get(), dstay.get(),
                                         dbest.get(), cnt, weighted, &moved));
            mvs += mv;
            if (!mv) continue;
            const auto tu = clk::now();
            apply_moves(X, d, d, moved, csum.get(), es, ev, touched);
            refresh_full();
            g_prof["x.update"] += ms_since(tu);
        }
        xsweep_moves += mvs;
        moves_log += (moves_log.empty() ? "x" : ",x") + std::to_string(mvs);
        PROF("x.regroup", regroup());
        if (omega != 1.0f) { // all centroids: relaxed between sweeps, the exact means after the last one
            const float om = sw + 1 < xsweeps ? omega : 1.0f;
            float* Cw = Cown.get();
#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < K; ++k) {
                if (cnt[k] <= 0) continue;
                const double inv = 1.0 / (double)cnt[k];
                for (size_t j = 0; j < d; ++j) {
                    const float c0 = C0[k * d + j];
                    Cw[k * d + j] = c0 + om * ((float)(csum[k * d + j] * inv) - c0);
                }
            }
            project_centroids();
        }
    }
    const double xsweep_ms = ms_since(t_xs);

    // ---- 6. final relabelling to the nearest candidate centroid (no update) ----
    const auto t_f = clk::now();
    size_t fnn_moves = 0;
    double fevals = 0;
    if (fnn) {
        SweepIn inf = in;
        inf.Kc = std::min<size_t>(geti("fcandk") > 0 ? (size_t)geti("fcandk") : Kc, K - 1);
        PROF("f.neigh", build_neigh(Cs.data(), l, reduced ? cn.data() : cns.data(), K, l, inf.Kc, neigh));
        PROF("f.prop", fevals = sweep_proposals(inf, cnt, gstart, gmem, neigh, reduced ? EXACT : SKETCH, false, prop.get(),
                                                dstay.get(), dbest.get(), 0, K));
        const bool fupd = geti("fupd") != 0;
        PROF("f.commit", fnn_moves = commit(nullptr, n, lab.get(), prop.get(), dstay.get(), dbest.get(), cnt, false,
                                            fupd ? &moved : nullptr));
        if (fupd && fnn_moves) { // the returned centroids are the exact means of the returned lists
            const auto tu = clk::now();
            if (reduced) {
                apply_moves(X, d, d, moved, csum.get(), es, ev, touched);
                float* Cw = Cown.get();
#pragma omp parallel for schedule(dynamic, 8)
                for (size_t t = 0; t < touched.size(); ++t) {
                    const size_t k = touched[t];
                    if (cnt[k] <= 0) continue;
                    const double inv = 1.0 / (double)cnt[k];
                    for (size_t j = 0; j < d; ++j) Cw[k * d + j] = (float)(csum[k * d + j] * inv);
                }
            } else if (!ssum.empty()) {
                apply_moves(Y, l, ls, moved, ssum.data(), es, ev, touched);
#pragma omp parallel for schedule(dynamic, 8)
                for (size_t t = 0; t < touched.size(); ++t) {
                    const size_t k = touched[t];
                    if (cnt[k] <= 0) continue;
                    const double inv = 1.0 / (double)cnt[k];
                    for (size_t j = 0; j < l; ++j) Cs[k * l + j] = (float)(ssum[k * l + j] * inv);
                }
            }
            g_prof["f.update"] += ms_since(tu);
        }
    }
    const double fnn_ms = ms_since(t_f), total_ms = ms_since(t_total);

    double rx2_mean = 0, rc2_mean = 0, d0_mean = 0, xc2_mean = 0;
    if (reduced) {
#pragma omp parallel for reduction(+ : rx2_mean, d0_mean, xc2_mean)
        for (size_t i = 0; i < n; ++i) {
            rx2_mean += (double)rx[i] * rx[i];
            xc2_mean += xc2[i];
            d0_mean += dstay[i];
        }
        for (size_t k = 0; k < K; ++k) rc2_mean += (double)rc[k] * rc[k];
        rx2_mean /= (double)n, d0_mean /= (double)n, xc2_mean /= (double)n, rc2_mean /= (double)K;
    }
    size_t empty_final = 0;
    {
        std::vector<uint32_t> c2(K, 0);
        for (size_t i = 0; i < n; ++i) ++c2[lab[i]];
        for (size_t k = 0; k < K; ++k) empty_final += c2[k] == 0;
    }
    std::string prof;
    for (auto& kv : g_prof) {
        char b[96];
        snprintf(b, sizeof b, "%s\"%s\":%.1f", prof.empty() ? "" : ",", kv.first.c_str(), kv.second);
        prof += b;
    }
    if (prefix != "-") {
        FILE* fc = fopen((prefix + ".centroids.fvecs").c_str(), "wb");
        const int32_t dd = (int32_t)d;
        for (size_t k = 0; k < K; ++k) {
            fwrite(&dd, 4, 1, fc);
            fwrite(&C[k * d], 4, d, fc);
        }
        fclose(fc);
        FILE* fl = fopen((prefix + ".labels.u32").c_str(), "wb");
        fwrite(lab.get(), 4, n, fl);
        fclose(fl);
    }
    printf("{\"n\":%zu,\"d\":%zu,\"K\":%zu,\"l\":%zu,\"ls\":%zu,\"ext\":%zu,\"ext_ms\":%.1f,\"pilot_evals\":%.2f,\"d1\":%d,\"mode\":\"%s\",\"meso\":%zu,\"weighted\":%d,\"rho\":%g,"
           "\"threads\":%d,\"sweeps\":%d,\"xsweeps\":%d,\"fnn\":%d,\"candk\":%zu,"
           "\"total_ms\":%.1f,\"sketch_ms\":%.1f,\"basis_ms\":%.1f,\"proj_ms\":%.1f,\"pass_ms\":%.1f,\"meso_ms\":%.1f,"
           "\"route_ms\":%.1f,\"cell_ms\":%.1f,\"sweep_ms\":%.1f,\"cent_ms\":%.1f,\"xsweep_ms\":%.1f,\"fnn_ms\":%.1f,"
           "\"fsweep_ms\":%.1f,\"captured_var\":%.4f,\"orth_err\":%.2e,\"cell_cv\":%.3f,\"refreshes\":%zu,"
           "\"empty_after_pass\":%zu,\"empty_final\":%zu,\"sweep_moves\":%zu,\"xsweep_moves\":%zu,\"fnn_moves\":%zu,"
           "\"moves\":\"%s\",\"xevals_per_point\":%.2f,\"fevals_per_point\":%.2f,\"replays\":%d,\"replay_moves\":%zu,"
"\"gs\":%zu,\"rx2_mean\":%.5g,\"rc2_mean\":%.5g,\"d0_mean\":%.5g,\"xc2_mean\":%.5g,"
           "\"prof\":{%s}}\n",
           n, d, K, l, ls, ext, ext_ms, pilot_evals, dq, mode.c_str(), M, (int)weighted, rho, T, sweeps, xsweeps, (int)fnn, Kc, total_ms, sketch_ms,
           basis_ms, proj_ms, pass_ms, meso_ms, route_ms, cell_ms, sweep_ms, cent_ms, xsweep_ms, fnn_ms,
           xsweep_ms + fnn_ms, sk.captured, sk.orth_err, cell_cv, refreshes, empty_after_pass, empty_final, sweep_moves,
           xsweep_moves, fnn_moves, moves_log.c_str(), xsweeps ? xevals / ((double)n * xsweeps) : 0.0,
           fevals / (double)n, replays, replay_moves, nch, rx2_mean, rc2_mean, d0_mean, xc2_mean, prof.c_str());
    return 0;
}
