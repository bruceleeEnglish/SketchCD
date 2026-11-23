"""Driver of the scale experiment (Exp-5): runs run_100m_method.py once per method, one after the other, with a watchdog.

A method is stopped (and listed in <out>_stopped.txt) if swap use grows by more than 3 GB -- a method that does not fit
in memory would otherwise thrash for hours -- or if it runs longer than --limit-h hours.

Usage: python run_100m.py --limit-h 6 --methods SketchCD SketchCD-Fast HierSuperKMeans QuantSuperKMeans EMeans-2 \
           FaissKMeans SampleSuperKMeans EMeans-5 FullSuperKMeans BatchedHartigan
"""
import argparse
import os
import signal
import subprocess
import sys
import time

from common import RUNS, SCALE_DATASET

HERE = os.path.dirname(os.path.abspath(__file__))


def swap_used_kb():
    m = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")
        m[k] = int(v.split()[0])
    return m["SwapTotal"] - m["SwapFree"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=SCALE_DATASET)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-h", type=float, default=6.0)
    ap.add_argument("--out", default=f"{RUNS}/exp05_bigann100m.jsonl")
    ap.add_argument("--log", default=f"{RUNS}/exp05_bigann100m.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for m in a.methods:
        with open(a.log, "a") as lg:
            lg.write(f"\n=== {time.strftime('%F %T')} start {m}\n")
            lg.flush()
            sw0 = swap_used_kb()
            p = subprocess.Popen([sys.executable, f"{HERE}/run_100m_method.py", "--dataset", a.dataset, "--method", m,
                                  "--seed", str(a.seed), "--out", a.out], stdout=lg, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            t0, why = time.time(), None
            while p.poll() is None:
                time.sleep(5)
                if swap_used_kb() - sw0 > 3 * 2 ** 20:
                    why = "out of memory (swap grew by more than 3 GB)"
                elif time.time() - t0 > a.limit_h * 3600:
                    why = f"time limit of {a.limit_h} h"
                if why:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    break
            lg.write(f"=== {time.strftime('%F %T')} end {m}: rc={p.returncode} wall={time.time() - t0:.0f}s"
                     f"{' STOPPED: ' + why if why else ''}\n")
            lg.flush()
        if why:
            with open(a.out.replace(".jsonl", "_stopped.txt"), "a") as f:
                f.write(f"{a.dataset}\t{m}\t{why}\t{time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
