# BatchedHartigan benchmark driver (the superkmeans INTERFACE target already propagates BLAS_LIBRARIES)
add_executable(batched_hartigan_bench.out batched_hartigan_bench.cpp)
target_link_libraries(batched_hartigan_bench.out PRIVATE superkmeans)
