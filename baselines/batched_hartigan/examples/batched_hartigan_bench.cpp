// BatchedHartigan benchmark driver (built on the SuperKMeans library).
// usage: batched_hartigan_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> <early_term 0|1>
//                                   <out_prefix|-> [queries.fvecs] [n_queries]
// hartigan = 1: the size-aware rule in every pass after the first (BatchedHartigan); 0: the unmodified library.
// Prints one CSV line per iteration, train_ms, and with SKM_LABELS=1 assign_ms; writes <out_prefix>.centroids.fvecs
// and <out_prefix>.labels.ivecs. Environment: SKM_SEED (42), SKM_LABELS (0), SKM_VERBOSE (0).
#include "superkmeans/superkmeans.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<float> ReadFvecs(const std::string& path, size_t& n, size_t& d) {
    std::ifstream f(path, std::ios::binary);
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    d = dim;
    const size_t row_bytes = 4 + d * 4;
    n = bytes / row_bytes;
    std::vector<float> out(n * d);
    f.seekg(0, std::ios::beg);
    std::vector<char> row(row_bytes);
    for (size_t i = 0; i < n; ++i) {
        f.read(row.data(), row_bytes);
        std::memcpy(out.data() + i * d, row.data() + 4, d * 4);
    }
    return out;
}
static void WriteFvecs(const std::string& path, const float* data, size_t n, size_t d) {
    std::ofstream f(path, std::ios::binary);
    const int32_t dim = static_cast<int32_t>(d);
    for (size_t i = 0; i < n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(data + i * d), d * 4);
    }
}
static void WriteIvecs(const std::string& path, const uint32_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    const int32_t dim = 1; // one id per record
    for (size_t i = 0; i < n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(data + i), 4);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "usage: batched_hartigan_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> "
                     "<early_term 0|1> <out_prefix|-> [queries.fvecs] [n_queries]\n";
        return 1;
    }
    const std::string base_path = argv[1];
    const size_t k = std::atoll(argv[2]);
    const uint32_t iters = std::atoi(argv[3]);
    const bool hartigan = std::atoi(argv[4]) != 0;
    const bool early_term = std::atoi(argv[5]) != 0;
    const std::string out_prefix = argv[6];

    // environment switches
    auto env_u = [](const char* name, long def) {
        const char* v = std::getenv(name);
        return v ? std::atol(v) : def;
    };
    const uint32_t seed = static_cast<uint32_t>(env_u("SKM_SEED", 42));
    const bool write_labels = env_u("SKM_LABELS", 0) != 0; // also time and write the library's fast assignment
    const bool verbose = env_u("SKM_VERBOSE", 0) != 0;

    size_t n = 0, d = 0;
    auto data = ReadFvecs(base_path, n, d);
    std::vector<float> queries;
    size_t n_queries = 0;
    if (argc > 7) {
        size_t nq = 0, dq = 0;
        queries = ReadFvecs(argv[7], nq, dq);
        n_queries = (argc > 8) ? std::min<size_t>(std::atoll(argv[8]), nq) : nq;
    }

    skmeans::SuperKMeansConfig config;
    config.iters = iters;
    config.seed = seed;
    config.sampling_fraction = 1.0f;         // all vectors
    config.max_points_per_cluster = 1 << 30; // no sampling
    config.early_termination = early_term;
    config.verbose = verbose;
    config.hartigan = hartigan;

    skmeans::SuperKMeans kmeans(k, d, config);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> centroids =
        kmeans.Train(data.data(), n, n_queries ? queries.data() : nullptr, n_queries);
    auto t1 = std::chrono::steady_clock::now();
    const double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // one CSV line per iteration
    std::cout << "iter,objective,shift,split,recall,not_pruned_pct,partial_d,gemm_only,moves,iter_ms\n";
    size_t it = 0;
    for (const auto& s : kmeans.iteration_stats) {
        const double ims = it < kmeans.iter_wall_ms.size() ? kmeans.iter_wall_ms[it] : -1.0;
        ++it;
        std::cout << s.iteration << "," << s.objective << "," << s.shift << "," << s.split << ","
                  << s.recall << "," << s.not_pruned_pct << "," << s.partial_d << ","
                  << (s.is_gemm_only ? 1 : 0) << "," << s.hartigan_moves << "," << ims << "\n";
    }
    std::cout << "n," << n << ",d," << d << ",k," << k << "\n";
    std::cout << "train_ms," << train_ms << "\n";

    if (out_prefix != "-") {
        WriteFvecs(out_prefix + ".centroids.fvecs", centroids.data(), k, d);
        if (write_labels) {
            auto ta = std::chrono::steady_clock::now();
            auto assign = kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, k);
            std::cout << "assign_ms,"
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count()
                      << "\n";
            WriteIvecs(out_prefix + ".labels.ivecs", assign.data(), n);
        }
    }
    return 0;
}
