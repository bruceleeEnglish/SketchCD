// cdkm.cpp -- exact sequential coordinate descent k-means (Hartigan's method / CDKM, Nie et al. TPAMI 2022),
// full dimension, all K candidates. Baseline "CDKM" of the paper.
//
// usage: cdkm <base.fvecs> <K> <out_prefix> [passes=10] [seed=42] [block=256]
//
// Init: K distinct random vectors as centroids, every vector to its nearest centroid, centroids = means.
// Pass: the vectors in a fresh random order. For x in list p with b_p >= 2: stay cost b_p/(b_p-1)|x-c_p|^2, join cost
// b_q/(b_q+1)|x-c_q|^2 for every q != p; x moves to the cheapest list if that is cheaper than staying, and c_p, c_q and
// the sizes are updated at once. These are exactly the decisions of the sequential method. For speed, the distances of a
// block of B vectors to all centroids come from one GEMM against the centroids at the start of the block; distances to
// centroids changed earlier in the block are recomputed exactly from the live centroid.
// Stops after `passes` passes or when a pass moves no vector. Output: <out>.centroids.fvecs, <out>.labels.u32, and one
// JSON line on stdout (times exclude reading the input).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

extern "C" {
int sgemm_(const char*, const char*, int*, int*, int*, const float*, const float*, int*, const float*, int*,
           const float*, float*, int*);
void openblas_set_num_threads(int);
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// G (rows x K, row-major) = Xb (rows x d) * C^T, tiled over (64 rows x 128 centroids), one single-threaded GEMM per tile
static void block_gemm(const float* Xb, size_t rows, const float* C, size_t K, size_t d, float* G) {
    const size_t RT = 64, KT = 128;
    const size_t nr = (rows + RT - 1) / RT, nk = (K + KT - 1) / KT;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t t = 0; t < nr * nk; ++t) {
        const size_t r0 = (t / nk) * RT, k0 = (t % nk) * KT;
        int m = (int)std::min(KT, K - k0), n = (int)std::min(RT, rows - r0), kk = (int)d, lda = (int)d, ldb = (int)d,
            ldc = (int)K;
        const float one = 1.f, zero = 0.f;
        // column-major: G^T[k0.., r0..] (m x n, ldc=K) = C[k0..]^T' (m x d) * Xb[r0..]^T (d x n)
        sgemm_("T", "N", &m, &n, &kk, &one, C + k0 * d, &lda, Xb + r0 * d, &ldb, &zero, G + r0 * K + k0, &ldc);
    }
}

static inline float dot(const float* a, const float* b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t t = 0; t < d; ++t) s += a[t] * b[t];
    return s;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: cdkm <base.fvecs> <K> <out_prefix> [passes=10] [seed=42] [block=256]\n");
        return 1;
    }
    const std::string base = argv[1], out = argv[3];
    const size_t K = std::strtoull(argv[2], nullptr, 10);
    const int passes = argc > 4 ? std::atoi(argv[4]) : 10;
    const unsigned seed = argc > 5 ? (unsigned)std::atoi(argv[5]) : 42u;
    const size_t B = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 256;
    openblas_set_num_threads(1);

    // ---- read fvecs
    FILE* f = fopen(base.c_str(), "rb");
    if (!f) return 2;
    int32_t dim = 0;
    if (fread(&dim, 4, 1, f) != 1) return 2;
    const size_t d = (size_t)dim, rowb = 4 + 4 * d;
    fseek(f, 0, SEEK_END);
    const size_t n = (size_t)ftell(f) / rowb;
    fseek(f, 0, SEEK_SET);
    std::vector<float> X(n * d);
    {
        const size_t nb = 65536;
        std::vector<char> buf(nb * rowb);
        for (size_t s = 0; s < n; s += nb) {
            const size_t m = std::min(nb, n - s);
            if (fread(buf.data(), rowb, m, f) != m) return 2;
            for (size_t i = 0; i < m; ++i) std::memcpy(&X[(s + i) * d], buf.data() + i * rowb + 4, 4 * d);
        }
    }
    fclose(f);

    const auto t_all = clk::now();
    std::vector<float> xn(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) xn[i] = dot(&X[i * d], &X[i * d], d);

    // ---- init: K distinct random vectors, nearest-centroid assignment
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    for (size_t k = 0; k < K; ++k) std::swap(perm[k], perm[k + rng() % (n - k)]);
    std::vector<float> C(K * d), cn(K);
    for (size_t k = 0; k < K; ++k) std::memcpy(&C[k * d], &X[(size_t)perm[k] * d], 4 * d);
    for (size_t k = 0; k < K; ++k) cn[k] = dot(&C[k * d], &C[k * d], d);

    std::vector<uint32_t> lab(n);
    const size_t IB = 4096;
    std::vector<float> G(std::max(IB, B) * K), Xb(std::max(IB, B) * d);
    for (size_t s = 0; s < n; s += IB) {
        const size_t m = std::min(IB, n - s);
        block_gemm(&X[s * d], m, C.data(), K, d, G.data());
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; ++i) {
            const float* g = &G[i * K];
            float best = std::numeric_limits<float>::infinity();
            uint32_t bk = 0;
            for (size_t k = 0; k < K; ++k) {
                const float v = cn[k] - 2.f * g[k];
                if (v < best) best = v, bk = (uint32_t)k;
            }
            lab[s + i] = bk;
        }
    }
    std::vector<double> S(K * d, 0.0);
    std::vector<int64_t> b(K, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t p = lab[i];
        ++b[p];
        double* sp = &S[(size_t)p * d];
        const float* x = &X[i * d];
        for (size_t t = 0; t < d; ++t) sp[t] += x[t];
    }
    std::vector<float> wj(K);  // join weight b/(b+1), live
    auto refresh = [&](size_t k) {
        float* c = &C[k * d];
        if (b[k] > 0) {
            const double inv = 1.0 / (double)b[k];
            const double* s = &S[k * d];
            for (size_t t = 0; t < d; ++t) c[t] = (float)(s[t] * inv);
        }
        cn[k] = dot(c, c, d);
        wj[k] = (float)b[k] / (float)(b[k] + 1);
    };
    for (size_t k = 0; k < K; ++k) refresh(k);
    const double init_ms = ms_since(t_all);

    // ---- sequential passes
    std::vector<double> pass_ms;
    std::vector<size_t> moves_per_pass;
    std::vector<int64_t> stamp(K, -1);
    std::vector<uint32_t> touched;
    touched.reserve(2 * B);
    std::vector<float> key(K);
    int64_t blk_id = 0;
    for (int it = 0; it < passes; ++it) {
        const auto tp = clk::now();
        std::shuffle(perm.begin(), perm.end(), rng);
        size_t moves = 0;
        for (size_t s = 0; s < n; s += B, ++blk_id) {
            const size_t m = std::min(B, n - s);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) std::memcpy(&Xb[i * d], &X[(size_t)perm[s + i] * d], 4 * d);
            block_gemm(Xb.data(), m, C.data(), K, d, G.data());
            touched.clear();
            for (size_t i = 0; i < m; ++i) {
                const size_t id = perm[s + i];
                const uint32_t p = lab[id];
                if (b[p] < 2) continue;
                const float* x = &Xb[i * d];
                const float* g = &G[i * K];
                const float xx = xn[id];
                // join costs from the block GEMM (valid for every centroid not touched in this block)
#pragma omp simd
                for (size_t k = 0; k < K; ++k) key[k] = wj[k] * std::max(0.f, xx + cn[k] - 2.f * g[k]);
                for (uint32_t k : touched) key[k] = wj[k] * std::max(0.f, xx + cn[k] - 2.f * dot(x, &C[(size_t)k * d], d));
                const float dp = key[p] / wj[p];  // |x - c_p|^2 (live)
                const float stay = (float)b[p] / (float)(b[p] - 1) * dp;
                key[p] = std::numeric_limits<float>::infinity();
                float best = std::numeric_limits<float>::infinity();
#pragma omp simd reduction(min : best)
                for (size_t k = 0; k < K; ++k) best = std::min(best, key[k]);
                if (!(best < stay)) continue;
                size_t q = 0;
                while (key[q] != best) ++q;
                // move x from p to q
                double *sp = &S[(size_t)p * d], *sq = &S[q * d];
                for (size_t t = 0; t < d; ++t) sp[t] -= x[t], sq[t] += x[t];
                --b[p];
                ++b[q];
                refresh(p);
                refresh(q);
                lab[id] = (uint32_t)q;
                ++moves;
                if (stamp[p] != blk_id) stamp[p] = blk_id, touched.push_back(p);
                if (stamp[q] != blk_id) stamp[q] = blk_id, touched.push_back((uint32_t)q);
            }
        }
        pass_ms.push_back(ms_since(tp));
        moves_per_pass.push_back(moves);
        fprintf(stderr, "pass %d: %zu moves, %.1f ms\n", it + 1, moves, pass_ms.back());
        if (moves == 0) break;
    }
    // exact means from scratch (removes accumulated rounding of the running sums)
    std::fill(S.begin(), S.end(), 0.0);
    for (size_t i = 0; i < n; ++i) {
        double* sp = &S[(size_t)lab[i] * d];
        const float* x = &X[i * d];
        for (size_t t = 0; t < d; ++t) sp[t] += x[t];
    }
    for (size_t k = 0; k < K; ++k) refresh(k);
    const double total_ms = ms_since(t_all);

    {
        FILE* fc = fopen((out + ".centroids.fvecs").c_str(), "wb");
        for (size_t k = 0; k < K; ++k) {
            fwrite(&dim, 4, 1, fc);
            fwrite(&C[k * d], 4, d, fc);
        }
        fclose(fc);
        FILE* fl = fopen((out + ".labels.u32").c_str(), "wb");
        fwrite(lab.data(), 4, n, fl);
        fclose(fl);
    }
    size_t empty = 0;
    for (size_t k = 0; k < K; ++k) empty += b[k] == 0;
    printf("{\"total_ms\": %.1f, \"init_ms\": %.1f, \"passes\": %zu, \"empty\": %zu, \"pass_ms\": [", total_ms, init_ms,
           pass_ms.size(), empty);
    for (size_t t = 0; t < pass_ms.size(); ++t) printf("%s%.1f", t ? ", " : "", pass_ms[t]);
    printf("], \"moves\": [");
    for (size_t t = 0; t < moves_per_pass.size(); ++t) printf("%s%zu", t ? ", " : "", moves_per_pass[t]);
    printf("]}\n");
    return 0;
}
