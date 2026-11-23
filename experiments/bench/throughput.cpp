// throughput.cpp -- matrix-multiplication and distance throughput of the machine (Section 6.1, "Implementation").
//
// The paper quotes two machine numbers: the throughput of the matrix multiplications of SketchCD, which also prices the
// arithmetic of one exhaustive assignment pass (n K d multiply-adds), and the throughput of full-dimensional distances,
// whose ratio to the former is rho. This program measures both for the shapes of one dataset.
//
// (1) GEMM. Every worker issues single-threaded BLAS calls on tiles of 256 rows, as bin/sketchcd does:
//       projection   256 vectors (d) against the l principal directions            (step 1; skipped when l = d)
//       cells        256 sketches (l) against the lists of one cell, sqrt(K)       (step 2)
//       sweep        256 sketches (l) against 129 lists, own list + 128 candidates (steps 3 and 5)
//       assignment   256 vectors (d) against all K centroids in tiles of 1024      (one exhaustive assignment pass)
//     The vectors of the projection and of the assignment are streamed from a buffer of about 1 GB, larger than the
//     caches; the sketch-space shapes run on buffers of the worker, as in the build.
// (2) Distances. One dot product over d dimensions per distance (the kernel of the verified sweeps): 64 vectors of a
//     block against centroid rows drawn from a pool of 129.
// (3) Read bandwidth. All workers read the 1 GB buffer of vectors once, each its own part: the bytes per second at which
//     the machine streams the input, which bounds every pass over the vectors from below.
//
// Output: one JSON line per measurement, throughput over all workers in GFLOPS (two flops per multiply-add) and, for
// the read bandwidth, in GB/s, best of `reps` runs.
//
// build: bash build.sh throughput       (bin/throughput)
// usage: OMP_NUM_THREADS=20 taskset -c <cores of one socket> bin/throughput <name> <d> <K> <l> [reps=5]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

extern "C" {
int sgemm_(const char*, const char*, int*, int*, int*, const float*, const float*, int*, const float*, int*,
           float*, float*, int*);
void openblas_set_num_threads(int);
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static volatile double g_sink; // receives the sums of the distance loop, so that the compiler keeps it

// C (rb x ra, row-major) = B (rb x k) * A (ra x k)^T, the call of bin/sketchcd
static inline void gemm_rows(const float* A, int ra, int lda, const float* B, int rb, int ldb, int k, float* C) {
    const char ta = 'T', tb = 'N';
    float one = 1.0f, zero = 0.0f;
    int m = ra, n = rb, kk = k, ldc = ra;
    sgemm_(&ta, &tb, &m, &n, &kk, &one, A, &lda, B, &ldb, &zero, C, &ldc);
}

// dot product with four accumulators, the kernel of the verified sweeps in bin/sketchcd
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

static const size_t RB = 256, KT = 1024; // rows of a tile, centroids of a tile

// 256-row tiles of `rows` vectors (dim k, shared buffer B) against A (ra x k, shared): projection and assignment
static double stream_gflops(const std::vector<float>& A, size_t ra, size_t k, const float* B, size_t rows, int reps) {
    const double madd = (double)ra * (double)k * (double)rows;
    const int passes = std::max(1, (int)std::ceil(2.0e11 / madd)); // about 0.4 s per run at 1000 GFLOPS
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        const double t0 = now();
#pragma omp parallel
        {
            std::vector<float> C(RB * std::min(ra, KT));
            for (int p = 0; p < passes; ++p) {
#pragma omp for schedule(dynamic, 4) nowait
                for (size_t s = 0; s < rows; s += RB) {
                    const size_t nb = std::min(RB, rows - s);
                    for (size_t k0 = 0; k0 < ra; k0 += KT) {
                        const size_t kt = std::min(KT, ra - k0);
                        gemm_rows(&A[k0 * k], (int)kt, (int)k, &B[s * k], (int)nb, (int)k, (int)k, C.data());
                    }
                }
            }
        }
        best = std::min(best, now() - t0);
    }
    return 2.0 * madd * passes / best / 1e9;
}

// every worker multiplies its own 256 x k tile with its own ra x k block: cells and sweeps
static double local_gflops(size_t ra, size_t k, int reps) {
    const double madd = (double)ra * (double)k * (double)RB;
    const int T = omp_get_max_threads();
    const long calls = std::max<long>(1, (long)std::ceil(2.0e11 / (madd * T)));
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        double t0 = 0;
#pragma omp parallel
        {
            std::vector<float> A(ra * k, 0.5f), B(RB * k, 0.25f), C(RB * ra);
#pragma omp barrier
#pragma omp master
            t0 = now();
#pragma omp barrier
            for (long c = 0; c < calls; ++c) gemm_rows(A.data(), (int)ra, (int)k, B.data(), (int)RB, (int)k, (int)k, C.data());
        }
        best = std::min(best, now() - t0);
    }
    return 2.0 * madd * calls * T / best / 1e9;
}

// distances of 64 vectors to centroid rows drawn from a pool of 129, one dot product each
static double distance_gflops(size_t d, int reps) {
    const size_t pool = 129, npt = 64, nc = 8;
    const int T = omp_get_max_threads();
    const long rounds = std::max<long>(1, (long)std::ceil(6.0e10 / ((double)d * npt * nc * T)));
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        double t0 = 0, acc = 0;
#pragma omp parallel reduction(+ : acc)
        {
            std::vector<float> C(pool * d, 0.5f), X(npt * d, 0.25f);
            unsigned r = 12345u + (unsigned)omp_get_thread_num();
#pragma omp barrier
#pragma omp master
            t0 = now();
#pragma omp barrier
            for (long it = 0; it < rounds; ++it)
                for (size_t i = 0; i < npt; ++i)
                    for (size_t j = 0; j < nc; ++j) {
                        r = r * 1664525u + 1013904223u;
                        acc += dotp(&X[i * d], &C[(size_t)((r >> 8) % pool) * d], d);
                    }
        }
        best = std::min(best, now() - t0);
        g_sink = acc;
    }
    return 2.0 * (double)d * npt * nc * rounds * T / best / 1e9;
}

// all workers read the vectors once, each its own part (static schedule, the pages were first touched the same way)
static double read_gbps(const float* X, size_t rows, size_t d, int reps) {
    const int passes = 16;
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        double acc = 0;
        const double t0 = now();
        for (int p = 0; p < passes; ++p) {
#pragma omp parallel for schedule(static) reduction(+ : acc)
            for (size_t i = 0; i < rows; ++i) {
                const float* x = &X[i * d];
                float s = 0.0f;
#pragma omp simd reduction(+ : s)
                for (size_t t = 0; t < d; ++t) s += x[t];
                acc += s;
            }
        }
        best = std::min(best, now() - t0);
        g_sink = acc;
    }
    return 4.0 * (double)rows * (double)d * passes / best / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: throughput <name> <d> <K> <l> [reps=5]\n");
        return 2;
    }
    const std::string name = argv[1];
    const size_t d = std::atoll(argv[2]), K = std::atoll(argv[3]), l = std::atoll(argv[4]);
    const int reps = argc > 5 ? std::atoi(argv[5]) : 5;
    const int T = omp_get_max_threads();
    openblas_set_num_threads(1);

    // the vectors: about 1 GB, a multiple of one tile per worker, first touched by the workers
    const size_t rows = std::max<size_t>(RB * (size_t)T, ((size_t)1 << 28) / d / (RB * T) * (RB * T));
    std::unique_ptr<float[]> X(new float[rows * d]);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows * d; ++i) X[i] = (float)((i * 2654435761u) % 1000) * 1e-3f - 0.5f;
    std::vector<float> Q(l * d), Cen(K * d);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = (float)((i * 40503u) % 1000) * 1e-3f - 0.5f;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Cen.size(); ++i) Cen[i] = (float)((i * 69069u) % 1000) * 1e-3f - 0.5f;

    const size_t M = (size_t)std::llround(std::sqrt((double)K));
    auto line = [&](const char* what, size_t a, size_t k, double g) {
        printf("{\"dataset\":\"%s\",\"d\":%zu,\"K\":%zu,\"l\":%zu,\"threads\":%d,\"what\":\"%s\",\"lists\":%zu,\"dim\":%zu,"
               "\"gflops\":%.1f}\n", name.c_str(), d, K, l, T, what, a, k, g);
        fflush(stdout);
    };
    if (l < d) line("projection", l, d, stream_gflops(Q, l, d, X.get(), rows, reps));
    line("cells", M, l, local_gflops(M, l, reps));
    line("sweep", 129, l, local_gflops(129, l, reps));
    line("assignment", K, d, stream_gflops(Cen, K, d, X.get(), rows, reps));
    line("distance", 1, d, distance_gflops(d, reps));
    printf("{\"dataset\":\"%s\",\"d\":%zu,\"K\":%zu,\"l\":%zu,\"threads\":%d,\"what\":\"read\",\"gbps\":%.1f}\n", name.c_str(), d, K,
           l, T, read_gbps(X.get(), rows, d, reps));
    return 0;
}
