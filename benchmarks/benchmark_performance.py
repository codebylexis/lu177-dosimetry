"""
benchmarks/benchmark_performance.py
=====================================
Systematic runtime and accuracy benchmarks for all simulation components.

Measures
--------
1. Population simulation:  wall time vs. patient count, speedup vs. cores
2. Variance reduction:     VRF and runtime vs. sample size
3. MCMC:                   ESS per second vs. chain length
4. NumPy vectorization:    confirms key hot paths are fully vectorized

Usage
-----
    cd monte-carlo-dosimetry
    python benchmarks/benchmark_performance.py

Results are printed as formatted tables and optionally saved to
benchmarks/results.json for CI tracking.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import json
import multiprocessing as mp
import numpy as np
from typing import Dict, List

from simulation import naive_mc_dose, compute_reference_doses
from population_analysis import compare_dosing_strategies
from variance_reduction import benchmark_vrf, ALL_METHODS
from mcmc import simulate_spect_measurements, metropolis_hastings, effective_sample_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _row(label: str, *values) -> None:
    col = f"  {label:<35}"
    for v in values:
        col += f"  {str(v):>10}"
    print(col)


# ---------------------------------------------------------------------------
# 1. Population simulation speedup
# ---------------------------------------------------------------------------

def bench_population_speedup(n: int = 8_000) -> Dict:
    _header(f"Population Simulation — Speedup vs. Workers (n={n:,})")
    _row("Workers", "Time (s)", "Speedup", "Throughput (patients/s)")

    results = {}
    t_serial = None

    max_workers = min(mp.cpu_count(), 8)
    worker_counts = sorted({1, 2, max_workers // 2, max_workers} - {0})

    for n_w in worker_counts:
        _, elapsed = _timed(compare_dosing_strategies, n=n, seed=42, n_workers=n_w)
        throughput = n / elapsed
        if n_w == 1:
            t_serial = elapsed
        speedup = t_serial / elapsed
        _row(f"{n_w} worker{'s' if n_w > 1 else ' '}",
             f"{elapsed:.3f}", f"{speedup:.2f}x", f"{throughput:,.0f}")
        results[n_w] = {"time_s": elapsed, "speedup": speedup}

    return results


# ---------------------------------------------------------------------------
# 2. Variance reduction: VRF vs. sample size
# ---------------------------------------------------------------------------

def bench_vrf_vs_n() -> Dict:
    _header("Variance Reduction Factor vs. Sample Size (n_trials=150)")
    sample_sizes = [100, 250, 500, 1_000, 2_000]
    methods = ["Naive MC", "Antithetic Variates", "Importance Sampling", "Quasi-MC (Halton)"]

    header_vals = [f"n={n}" for n in sample_sizes]
    _row("Method", *header_vals)
    print("  " + "-" * 63)

    results = {}
    for name in methods:
        vrfs_at_n = []
        for n in sample_sizes:
            all_vrfs = benchmark_vrf(n=n, n_trials=150, seed=42)
            vrfs_at_n.append(f"{all_vrfs[name]:.1f}x")
        _row(name, *vrfs_at_n)
        results[name] = vrfs_at_n

    return results


# ---------------------------------------------------------------------------
# 3. MCMC: ESS/second vs. chain length
# ---------------------------------------------------------------------------

def bench_mcmc_ess() -> Dict:
    _header("MCMC Efficiency — ESS per Second vs. Chain Length")
    _row("n_samples", "Time (s)", "ESS (A_slow)", "ESS/s")

    observed, _ = simulate_spect_measurements(seed=17)
    results = {}

    for n in [5_000, 10_000, 20_000]:
        result, elapsed = _timed(
            metropolis_hastings, observed, n_samples=n, n_burnin=int(n * 0.15), seed=17
        )
        ess  = effective_sample_size(result.samples[:, 0])
        ess_per_s = ess / elapsed
        _row(f"{n:,}", f"{elapsed:.2f}", f"{ess:.0f}", f"{ess_per_s:.0f}")
        results[n] = {"time_s": elapsed, "ess": ess, "ess_per_s": ess_per_s}

    return results


# ---------------------------------------------------------------------------
# 4. Vectorization audit
# ---------------------------------------------------------------------------

def bench_vectorization() -> Dict:
    """
    Confirm that key hot paths are fully NumPy-vectorized by measuring
    throughput at large n. A non-vectorized Python loop would scale ~10-100x
    worse than these numbers.
    """
    _header("NumPy Vectorization Throughput")
    _row("Operation", "n", "Time (ms)", "Throughput (ops/s)")

    results = {}

    for n in [10_000, 100_000, 1_000_000]:
        # Time core dose sampling
        _, elapsed = _timed(naive_mc_dose, n, seed=0)
        throughput = n / elapsed
        _row("sample_patient_doses", f"{n:,}", f"{elapsed*1000:.1f}", f"{throughput:,.0f}")
        results[n] = {"time_ms": elapsed * 1000, "throughput": throughput}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(save_json: bool = True) -> Dict:
    print("\nMonte Carlo Dosimetry — Performance Benchmark Suite")
    print(f"Platform: {mp.cpu_count()} CPU cores available\n")

    results = {
        "population_speedup": bench_population_speedup(),
        "vrf_vs_n":           bench_vrf_vs_n(),
        "mcmc_ess":           bench_mcmc_ess(),
        "vectorization":      bench_vectorization(),
    }

    if save_json:
        out_path = os.path.join(os.path.dirname(__file__), "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_all()
