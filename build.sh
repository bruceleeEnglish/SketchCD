#!/usr/bin/env bash
# Builds the binaries into bin/.
#
#   bash build.sh                 the three binaries of the comparison (sketchcd, cdkm, hartigan)
#   bash build.sh sketchcd        bin/sketchcd          SketchCD                       (g++, OpenMP, OpenBLAS)
#   bash build.sh cdkm            bin/cdkm              exact coordinate descent       (g++, OpenMP, OpenBLAS)
#   bash build.sh hartigan        bin/batched_hartigan  BatchedHartigan baseline       (cmake >= 3.26, Eigen, OpenBLAS)
#   bash build.sh throughput      bin/throughput        GEMM and distance throughput   (g++, OpenMP, OpenBLAS)
#
#   OPENBLAS=/path/to/libopenblas.so   the OpenBLAS shared library to link (default: the one the linker finds as
#                                      -lopenblas). Use the same library for all three binaries; the numbers of the
#                                      paper were measured with OpenBLAS 0.3.21.
#   EIGEN=/path/to/eigen               Eigen sources for the BatchedHartigan baseline, if cmake cannot find Eigen3
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
CXX=${CXX:-g++}
FLAGS="-O3 -march=native -fopenmp -Wall"
if [ -n "${OPENBLAS:-}" ]; then
  BLASDIR=$(cd "$(dirname "$OPENBLAS")" && pwd)
  LIB="$OPENBLAS -Wl,-rpath,$BLASDIR -Wl,--disable-new-dtags"
else
  LIB="-lopenblas"
fi

build_sketchcd() { $CXX $FLAGS src/sketchcd.cpp -o bin/sketchcd $LIB; }
build_cdkm() { $CXX $FLAGS baselines/cdkm.cpp -o bin/cdkm $LIB; }
build_throughput() { $CXX $FLAGS experiments/bench/throughput.cpp -o bin/throughput $LIB; }
build_hartigan() {
  local src=baselines/batched_hartigan
  if [ -n "${EIGEN:-}" ]; then mkdir -p $src/extern && ln -sfn "$EIGEN" $src/extern/Eigen; fi
  local args=(-DCMAKE_BUILD_TYPE=Release -DSKMEANS_SKIP_FFTW=ON)
  if [ -n "${OPENBLAS:-}" ]; then
    args+=(-DBLAS_LIBRARIES="$OPENBLAS" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$BLASDIR -Wl,--disable-new-dtags")
  fi
  cmake -S $src -B $src/build "${args[@]}" > $src/build.log 2>&1 || { tail -20 $src/build.log; exit 1; }
  cmake --build $src/build --target batched_hartigan_bench.out -j >> $src/build.log 2>&1 || { tail -20 $src/build.log; exit 1; }
  cp $src/build/examples/batched_hartigan_bench.out bin/batched_hartigan
}

for target in "${@:-sketchcd cdkm hartigan}"; do
  for t in $target; do
    case $t in
      sketchcd) build_sketchcd ;;
      cdkm) build_cdkm ;;
      hartigan) build_hartigan ;;
      throughput) build_throughput ;;
      *) echo "unknown target $t (sketchcd, cdkm, hartigan, throughput)"; exit 1 ;;
    esac
    echo "built $t"
  done
done
ls -l bin
