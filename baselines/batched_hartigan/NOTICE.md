# Notice

This folder is a modified copy of the header-only library **SuperKMeans** version 0.1.1 (https://github.com/cwida/SuperKMeans),
Copyright (c) 2025 Leonardo Xavier Kuffo Rivero, distributed under the MIT License in `LICENSE`.

Changes made for the BatchedHartigan baseline of the paper:

- `include/superkmeans/superkmeans.h`: the configuration flags `hartigan` and `hartigan_percand`; the method
  `AssignAndUpdateCentroidsHartigan`, which the pruned pass calls instead of `AssignAndUpdateCentroids` when `hartigan`
  is set; the number of size-aware moves and the wall-clock time of every iteration in the statistics.
- `include/superkmeans/distance_computers/batch_computers.h`: `FindNearestNeighborWithPruningHartigan`, the pruned
  search with the size-aware cost `b/(b+1) |x - c|^2` (stay cost `b/(b-1) |x - c|^2`), per-candidate pruning thresholds,
  and a commit of the moves of a pass with the live list sizes.
- `include/superkmeans/pdx/pdxearch.h`: `Top1PartialSearchHartigan`, the pruned top-1 search with the weighted cost, and
  `PruneHartigan`, its per-candidate pruning step.
- `include/superkmeans/common.h`: the switch `g_hartigan_percand` and the struct `HartiganBest`.
- `examples/batched_hartigan_bench.cpp`: the benchmark driver. The other examples, the tests, the benchmarks, and the
  Python bindings of the library are not part of this copy.

The competitors of the paper use the `superkmeans` wheel 0.2.0, a later version of the library; this copy is version
0.1.1, so BatchedHartigan and FullSuperKMeans also differ in code other than the decision rule.

The size-aware rule runs in the pruned pass of the library. The library uses that pass only for `d >= 128` and more
than 256 clusters, and never in the first iteration; otherwise the iterations are those of the unmodified library.
