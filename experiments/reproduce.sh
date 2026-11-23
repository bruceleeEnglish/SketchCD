#!/usr/bin/env bash
# Every experiment of the paper as one block of commands.
#
#     bash experiments/reproduce.sh               lists the blocks
#     bash experiments/reproduce.sh exp1 exp2     runs these blocks
#     bash experiments/reproduce.sh all           runs all of them
#
# Records go to runs/ under the names of results/, so that
#     SKETCHCD_RESULTS=runs python experiments/make_figures.py
# rebuilds every table and figure from the new runs.
#
# Protocol of the paper: two sockets with 20 cores each; every run uses the 20 cores of one NUMA node, pinned with
# taskset; the times of the ablation, thread and sensitivity experiments are the best of three runs on an otherwise
# idle machine (--reps 3). NODE selects the NUMA node (default 1), PYTHON the interpreter.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -eq 0 ]; then   # list the blocks
  grep -E '^exp[0-9]+\(\)' "$HERE/reproduce.sh" | sed -E 's/\(\) \{ +#/ /'
  echo "(Exp-3 and Exp-12 use the records of exp1)"
  exit 0
fi
cd "$HERE/bench"
PY=${PYTHON:-python}
NODE=${NODE:-1}
RUNS=${SKETCHCD_RUNS:-$(cd ../.. && pwd)/runs}
mkdir -p "$RUNS"
SWEEPS=${SKETCHCD_SAVE:-$(cd ../.. && pwd)/outputs}_sweeps   # builds of the K sweeps, kept apart from the builds that exp4 and exp9 read
CPUS=$(cat /sys/devices/system/node/node$NODE/cpulist)
E="taskset -c $CPUS env OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20"
HIGH="gist fiqa treccovid dbp3s cifar10 cifar100 svhn tinyimagenet"   # the eight high-dimensional collections
TEN="bigann10m deep10m"                                                # the two ten-million-vector collections
TRIO="gist treccovid svhn"                                             # one dataset of each kind (analysis experiments)
SEVEN="FullSuperKMeans SampleSuperKMeans HierSuperKMeans QuantSuperKMeans EMeans-2 EMeans-5 FaissKMeans"
CURVE="SketchCD@sweeps=0,xsweeps=0 SketchCD@sweeps=2,xsweeps=0 SketchCD@sweeps=4,xsweeps=1 SketchCD SketchCD@sweeps=8,xsweeps=3 SketchCD@sweeps=12,xsweeps=4 SketchCD@sweeps=16,xsweeps=6"

exp0() {   # Section 6.1, Implementation: matrix-multiplication and distance throughput and read bandwidth of one socket (bash build.sh throughput)
  rm -f $RUNS/exp00_throughput.jsonl
  while read -r name d K l; do   # dataset, d, K, sketch dimension (l = d: no sketch)
    $E ${THROUGHPUT_BIN:-../../bin/throughput} $name $d $K $l >> $RUNS/exp00_throughput.jsonl
  done <<'SHAPES'
gist 960 4000 128
fiqa 1024 952 128
treccovid 1024 1651 128
dbp3s 1536 1259 128
cifar10 3072 972 256
cifar100 3072 972 256
svhn 3072 3173 256
tinyimagenet 12288 1321 384
bigann10m 128 12649 128
deep10m 96 12643 96
bigann100m 128 40000 128
SHAPES
}
exp1() {   # Table 3, Figure 5, Table 4, Table 10 (Exp-1, Exp-3, Exp-12): build time and index quality, three seeds
  $E $PY run_methods.py --node $NODE --datasets $HIGH --seeds 42 43 44 --methods SketchCD SketchCD-Fast $SEVEN --out $RUNS/exp01_main.jsonl
  $E $PY run_methods.py --node $NODE --ref-big --datasets $TEN --seeds 42 43 44 --methods SketchCD SketchCD-Fast $SEVEN --out $RUNS/exp01_main.jsonl
  $E $PY run_extra_baselines.py --node $NODE --datasets $HIGH $TEN --seeds 42 43 44 --out $RUNS/exp01_extra_baselines.jsonl
  rm -f $RUNS/exp01_centroid_gap.jsonl   # Metrics of Section 6.1: returned centroids against the means of the returned lists
  $PY centroid_gap.py --out $RUNS/exp01_centroid_gap.jsonl
}
exp2() {   # Figure 6 (Exp-2): trade-off curves, seed 42
  $E $PY run_methods.py --node $NODE --datasets $HIGH --seeds 42 --methods $CURVE --out $RUNS/exp02_tradeoff_sketchcd.jsonl
  $E $PY run_methods.py --node $NODE --no-ref --datasets $TEN --seeds 42 --methods $CURVE --out $RUNS/exp02_tradeoff_sketchcd.jsonl
  $E $PY run_baseline_curves.py --node $NODE --datasets $HIGH --seeds 42 --out $RUNS/exp02_tradeoff_baselines.jsonl
  $E $PY run_baseline_curves.py --node $NODE --big-flat --no-ref --datasets $TEN --seeds 42 --out $RUNS/exp02_tradeoff_baselines.jsonl
}
exp4() {   # Figure 7 (Exp-4): single-threaded query throughput of the indexes saved by exp1 (seed 42)
  taskset -c $(echo $CPUS | cut -d, -f1 | cut -d- -f1) env OMP_NUM_THREADS=20 $PY measure_qps.py --bind-node $NODE --reps 2 --seed 42 \
      --datasets $HIGH $TEN --methods SketchCD FullSuperKMeans HierSuperKMeans QuantSuperKMeans EMeans-5 FaissKMeans SampleSuperKMeans \
      --out $RUNS/exp04_qps.jsonl
}
exp5() {   # Figure 8, Tables 5 and 6 (Exp-5): scalability in K, n, d, threads, and BIGANN-100M
  for km in 1 2 4 8 16 32; do
    $E $PY run_methods.py --node $NODE --no-ref --kmult $km --save "$SWEEPS" --datasets gist --seeds 42 \
        --methods SketchCD FullSuperKMeans HierSuperKMeans QuantSuperKMeans --out $RUNS/exp05_scale_k.jsonl
  done
  M5="SketchCD FullSuperKMeans HierSuperKMeans QuantSuperKMeans EMeans-5"
  $E $PY run_methods.py --node $NODE --no-ref --datasets deep500k deep1m deep2m deep5m --seeds 42 --methods $M5 --out $RUNS/exp05_scale_n.jsonl
  $E $PY run_methods.py --node $NODE --no-ref --datasets tiny8 tiny16 tiny32 --seeds 42 --methods $M5 --out $RUNS/exp05_scale_d.jsonl
  for t in 1 2 5 10 20; do
    $E $PY run_sketchcd_configs.py --node $NODE --threads $t --reps 3 --datasets $TRIO --seeds 42 --cfg "t$t:" --out $RUNS/exp05_threads.jsonl
  done
  $PY run_100m.py --limit-h 6 --out $RUNS/exp05_bigann100m.jsonl --log $RUNS/exp05_bigann100m.log --methods SketchCD SketchCD-Fast \
      HierSuperKMeans QuantSuperKMeans EMeans-2 FaissKMeans SampleSuperKMeans EMeans-5 FullSuperKMeans BatchedHartigan
}
exp6() {   # Exp-6: memory beyond the input vectors
  $E $PY measure_memory.py --node $NODE --datasets $TRIO --out $RUNS/exp06_memory.jsonl
}
exp7() {   # Figure 9 (Exp-7): sketch dimension, without the extension of the sketch
  $E $PY run_sketchcd_configs.py --node $NODE --datasets $TRIO --seeds 42 --cfg "l32:ds=32,dp=32,ext=0" "l64:ds=64,dp=64,ext=0" \
      "l96:ds=96,dp=96,ext=0" "l128:ds=128,dp=128,ext=0" "l192:ds=192,dp=192,ext=0" "l256:ds=256,dp=256,ext=0" \
      "l384:ds=384,dp=384,ext=0" --out $RUNS/exp07_sketch_dim.jsonl
}
exp8() {   # Figure 10 (Exp-8): candidates, over-relaxation, sample size, cells (the cell counts are 1/4 ... 4 times sqrt(K))
  COMMON=("c32:candk=32,fcandk=64" "c64:candk=64,fcandk=128" "c96:candk=96,fcandk=192" "c128:candk=128,fcandk=256"
          "c192:candk=192,fcandk=384" "c256:candk=256,fcandk=512" "w1.0:omega=1.0" "w1.3:omega=1.3" "w1.6:omega=1.6" "w1.9:omega=1.9"
          "w2.2:omega=2.2" "s4096:nsample=4096" "s8192:nsample=8192" "s16384:nsample=16384" "s32768:nsample=32768" "s65536:nsample=65536")
  declare -A MESO=( [gist]="m0.25:meso=16 m0.5:meso=32 m1:meso=63 m2:meso=126 m4:meso=253"
                    [treccovid]="m0.25:meso=10 m0.5:meso=20 m1:meso=41 m2:meso=81 m4:meso=163"
                    [svhn]="m0.25:meso=14 m0.5:meso=28 m1:meso=56 m2:meso=113 m4:meso=225" )
  for ds in $TRIO; do   # quality: three seeds; times: seed 42, best of three on an idle machine
    $E $PY run_sketchcd_configs.py --node $NODE --datasets $ds --seeds 42 43 44 --cfg "${COMMON[@]}" ${MESO[$ds]} --out $RUNS/exp08_sensitivity_quality.jsonl
    $E $PY run_sketchcd_configs.py --node $NODE --datasets $ds --seeds 42 --reps 3 --cfg "${COMMON[@]}" ${MESO[$ds]} --out $RUNS/exp08_sensitivity_times.jsonl
  done
}
exp9() {   # Table 7, Figure 11 (Exp-9): the decision rule; SketchCD@weighted=0 is SketchCD-NC
  NC="SketchCD SketchCD@weighted=0"
  $E $PY run_methods.py --node $NODE --no-ref --datasets $HIGH $TEN --seeds 42 --methods SketchCD@weighted=0 --out $RUNS/exp09_nc_rule.jsonl
  for k in 1 2 4 8 16 32; do $E $PY run_methods.py --node $NODE --no-ref --kmult $k --save "$SWEEPS" --datasets gist --seeds 42 --methods $NC --out $RUNS/exp09_list_size_sweep.jsonl; done
  for k in 4 8 16 32 64; do $E $PY run_methods.py --node $NODE --no-ref --kmult $k --save "$SWEEPS" --datasets $TEN --seeds 42 --methods $NC --out $RUNS/exp09_list_size_sweep.jsonl; done
  $PY list_sizes.py --out $RUNS/exp09_list_sizes.jsonl      # needs the builds of exp1 and of the first line of this block
  $E $PY model_params.py --node $NODE --out $RUNS/exp09_model_params.jsonl   # sigma^2 / Delta^2 of Section 4.2
}
exp10() {  # Table 8 (Exp-10): certified pruning against evaluating every candidate
  $E $PY check_exactness.py --node $NODE --datasets $HIGH --out $RUNS/exp10_exactness.jsonl
}
exp11() {  # Table 9 (Exp-11): ablation, best of three runs on an idle machine
  $E $PY run_sketchcd_configs.py --node $NODE --datasets $TRIO --seeds 42 --reps 3 --cfg "default:" "flat-first-pass:mode=flat,replays=0" \
      "no-replays:replays=0" "omega=1:omega=1" "unweighted:weighted=0" "no-exact-sweeps:xsweeps=0" "no-sketch-sweeps:sweeps=0" \
      "cand64/128:candk=64,fcandk=128" "no-final-relabel:fnn=0" --out $RUNS/exp11_ablation.jsonl
}

for x in "$@"; do
  if [ "$x" = all ]; then for y in exp0 exp1 exp2 exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11; do $y; done; else "$x"; fi
done
