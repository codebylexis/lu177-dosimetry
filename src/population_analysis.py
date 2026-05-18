"""
population_analysis.py
======================
Population-level dosimetry simulation comparing standard vs.
MC-informed individualized Lu-177 DOTATATE dosing strategies.

Central question
----------------
Given 30% inter-patient biokinetic variability, what fraction of patients
receiving the standard 7.4 GBq dose exceed the QUANTEC kidney tolerance
limit of 23 Gy? And how much does MC-informed individualized dosing help?

Dosing strategies
-----------------
Standard     : fixed 7.4 GBq for all patients regardless of biology
Individualized: compute the activity that delivers exactly target_kidney_Gy
               to each patient's kidneys, clipped to the clinical range
               3.7 - 18.5 GBq (approximately 0.5x - 2.5x standard dose)

Patient outcomes
----------------
Each patient is classified into exactly one of three categories:
  Optimal        : kidney dose <= tolerance AND tumor dose >= tumoricidal min
  Kidney toxicity: kidney dose > tolerance
  Under-treatment: kidney dose <= tolerance AND tumor dose < tumoricidal min

Performance
-----------
The simulation is parallelized with multiprocessing.Pool, partitioning the
patient cohort across all available CPU cores. Wall-clock speedup scales
near-linearly with core count for large n (n >= 2,000 per worker).
"""

import numpy as np
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from simulation import (
    ORGAN_PARAMS, STANDARD_ACTIVITY_GBQ,
    KIDNEY_TOLERANCE_GY, TUMOR_MIN_GY, INTER_PATIENT_CV,
)


# ---------------------------------------------------------------------------
# Clinical activity range
# ---------------------------------------------------------------------------

ACTIVITY_MIN_GBQ = 3.7    # minimum prescribed activity (0.5x standard)
ACTIVITY_MAX_GBQ = 18.5   # maximum prescribed activity (2.5x standard)
TARGET_KIDNEY_GY = 20.0   # individualized target (3 Gy below QUANTEC limit)


# ---------------------------------------------------------------------------
# Per-worker simulation functions (must be module-level for pickle)
# ---------------------------------------------------------------------------

def _simulate_standard_chunk(args: Tuple) -> Dict[str, np.ndarray]:
    """
    Simulate one chunk of the patient population under standard dosing.

    Parameters
    ----------
    args : (n_chunk, activity_GBq, cv, seed)
    """
    n_chunk, activity_GBq, cv, seed = args
    rng = np.random.default_rng(seed)

    doses = {}
    for organ, p in ORGAN_PARAMS.items():
        A_slow_s   = p.A_slow   * rng.lognormal(0, cv,        n_chunk)
        lam_slow_s = p.lam_slow * rng.lognormal(0, cv * 0.5,  n_chunk)
        A_fast_s   = p.A_fast   * rng.lognormal(0, cv * 0.8,  n_chunk)
        ca         = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
        doses[organ] = p.S_value * ca * activity_GBq * 1000
    return doses


def _simulate_individualized_chunk(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one chunk under individualized dosing.

    Parameters
    ----------
    args : (n_chunk, target_kidney_Gy, cv, activity_min, activity_max, seed)
    """
    n_chunk, target_kidney_Gy, cv, activity_min, activity_max, seed = args
    rng = np.random.default_rng(seed)

    pk = ORGAN_PARAMS["Kidneys"]
    pt = ORGAN_PARAMS["Tumor"]

    A_slow_k   = pk.A_slow   * rng.lognormal(0, cv,        n_chunk)
    lam_slow_k = pk.lam_slow * rng.lognormal(0, cv * 0.5,  n_chunk)
    A_fast_k   = pk.A_fast   * rng.lognormal(0, cv * 0.8,  n_chunk)
    ca_k       = A_fast_k / pk.lam_fast + A_slow_k / lam_slow_k

    activities = target_kidney_Gy / (pk.S_value * ca_k * 1000)
    activities = np.clip(activities, activity_min, activity_max)

    kidney_doses = pk.S_value * ca_k * activities * 1000

    A_slow_t   = pt.A_slow   * rng.lognormal(0, cv,        n_chunk)
    lam_slow_t = pt.lam_slow * rng.lognormal(0, cv * 0.5,  n_chunk)
    A_fast_t   = pt.A_fast   * rng.lognormal(0, cv * 0.8,  n_chunk)
    ca_t       = A_fast_t / pt.lam_fast + A_slow_t / lam_slow_t
    tumor_doses = pt.S_value * ca_t * activities * 1000

    return kidney_doses, tumor_doses, activities


# ---------------------------------------------------------------------------
# Chunk partitioning utility
# ---------------------------------------------------------------------------

def _make_chunks(n: int, n_workers: int, base_seed: int) -> list:
    """
    Partition n patients into n_workers chunks with independent seeds.

    Each chunk gets a deterministic seed derived from base_seed so that
    results are reproducible regardless of worker scheduling order.
    """
    chunk_size = n // n_workers
    remainder  = n % n_workers
    chunks = []
    for i in range(n_workers):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append((size, base_seed + i * 1000))
    return chunks


# ---------------------------------------------------------------------------
# Parallel population simulation
# ---------------------------------------------------------------------------

def simulate_population(
    n: int,
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate absorbed doses for n patients at a fixed administered activity.

    The patient cohort is partitioned into n_workers chunks and simulated
    in parallel using multiprocessing.Pool. Each worker receives an
    independent RNG seed derived from the base seed, ensuring full
    reproducibility regardless of worker scheduling order.

    Parameters
    ----------
    n            : number of simulated patients
    activity_GBq : administered activity in GBq (same for all patients)
    cv           : inter-patient coefficient of variation
    seed         : base random seed for reproducibility
    n_workers    : number of parallel workers (default: all CPU cores)

    Returns
    -------
    dict mapping organ name -> array of shape (n,) absorbed doses in Gy
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n)

    chunks = _make_chunks(n, n_workers, seed)
    args   = [(size, activity_GBq, cv, s) for size, s in chunks]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_simulate_standard_chunk, args)

    doses = {organ: np.concatenate([r[organ] for r in results])
             for organ in ORGAN_PARAMS}
    return doses


def simulate_individualized(
    n: int,
    target_kidney_Gy: float = TARGET_KIDNEY_GY,
    cv: float = INTER_PATIENT_CV,
    activity_min: float = ACTIVITY_MIN_GBQ,
    activity_max: float = ACTIVITY_MAX_GBQ,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate individualized dosing: for each patient, solve for the
    administered activity that delivers target_kidney_Gy to the kidneys.

    From D = S * CA * A  =>  A = target / (S * CA)

    Activity is clipped to the clinical feasibility range [activity_min, activity_max].
    Parallelized across n_workers processes.

    Parameters
    ----------
    n                : number of simulated patients
    target_kidney_Gy : target kidney absorbed dose (Gy)
    activity_min/max : clinical activity bounds (GBq)
    seed             : base random seed
    n_workers        : number of parallel workers (default: all CPU cores)

    Returns
    -------
    (kidney_doses, tumor_doses, activities) each of shape (n,)
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n)

    chunks = _make_chunks(n, n_workers, seed)
    args   = [(size, target_kidney_Gy, cv, activity_min, activity_max, s)
              for size, s in chunks]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_simulate_individualized_chunk, args)

    kidney_doses = np.concatenate([r[0] for r in results])
    tumor_doses  = np.concatenate([r[1] for r in results])
    activities   = np.concatenate([r[2] for r in results])

    return kidney_doses, tumor_doses, activities


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

@dataclass
class PopulationOutcomes:
    """Outcome masks and summary statistics for a simulated population."""
    kidney_doses:   np.ndarray
    tumor_doses:    np.ndarray
    n_patients:     int

    # Boolean masks
    optimal:         np.ndarray
    kidney_toxicity: np.ndarray
    under_treatment: np.ndarray

    # Summary rates
    p_optimal:          float
    p_kidney_toxicity:  float
    p_under_treatment:  float

    mean_kidney_dose:   float
    mean_tumor_dose:    float


def classify_outcomes(
    kidney_doses: np.ndarray,
    tumor_doses: np.ndarray,
    kidney_limit: float = KIDNEY_TOLERANCE_GY,
    tumor_min: float = TUMOR_MIN_GY,
) -> PopulationOutcomes:
    """
    Classify each patient into one of three mutually exclusive outcome categories.

    Parameters
    ----------
    kidney_doses : array of kidney absorbed doses (Gy)
    tumor_doses  : array of tumor absorbed doses (Gy)
    kidney_limit : QUANTEC mean kidney dose tolerance (Gy)
    tumor_min    : minimum tumoricidal dose threshold (Gy)

    Returns
    -------
    PopulationOutcomes dataclass
    """
    kidney_ok = kidney_doses <= kidney_limit
    tumor_ok  = tumor_doses  >= tumor_min
    n = len(kidney_doses)

    optimal         = kidney_ok & tumor_ok
    kidney_toxicity = ~kidney_ok
    under_treatment = kidney_ok & ~tumor_ok

    return PopulationOutcomes(
        kidney_doses      = kidney_doses,
        tumor_doses       = tumor_doses,
        n_patients        = n,
        optimal           = optimal,
        kidney_toxicity   = kidney_toxicity,
        under_treatment   = under_treatment,
        p_optimal         = float(optimal.mean()),
        p_kidney_toxicity = float(kidney_toxicity.mean()),
        p_under_treatment = float(under_treatment.mean()),
        mean_kidney_dose  = float(kidney_doses.mean()),
        mean_tumor_dose   = float(tumor_doses.mean()),
    )


# ---------------------------------------------------------------------------
# Full population comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    standard:       PopulationOutcomes
    individualized: PopulationOutcomes
    n_patients:     int
    wall_time_s:    float = 0.0


def compare_dosing_strategies(
    n: int = 8_000,
    cv: float = INTER_PATIENT_CV,
    target_kidney_Gy: float = TARGET_KIDNEY_GY,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> ComparisonResult:
    """
    Simulate n patients under both dosing strategies and compare outcomes.

    Parallelized across all available CPU cores by default.

    Parameters
    ----------
    n                : number of simulated patients
    cv               : inter-patient coefficient of variation
    target_kidney_Gy : kidney dose target for individualized strategy
    seed             : random seed
    n_workers        : number of parallel workers (default: all CPU cores)

    Returns
    -------
    ComparisonResult with outcomes for both strategies and wall-clock time
    """
    t0 = time.perf_counter()

    std_doses = simulate_population(n, STANDARD_ACTIVITY_GBQ, cv, seed, n_workers)
    std_out   = classify_outcomes(std_doses["Kidneys"], std_doses["Tumor"])

    k_ind, t_ind, _ = simulate_individualized(
        n, target_kidney_Gy, cv, seed=seed, n_workers=n_workers
    )
    ind_out = classify_outcomes(k_ind, t_ind)

    wall_time = time.perf_counter() - t0

    return ComparisonResult(
        standard       = std_out,
        individualized = ind_out,
        n_patients     = n,
        wall_time_s    = wall_time,
    )


def print_comparison(result: ComparisonResult) -> None:
    """Pretty-print the comparison summary."""
    s, i = result.standard, result.individualized
    n_workers = mp.cpu_count()
    print(f"Population Safety Summary (n = {result.n_patients:,} patients, "
          f"{n_workers} workers, {result.wall_time_s:.2f}s)")
    print("=" * 66)
    print(f"{'Metric':<30} {'Standard':>12} {'Individualized':>14}")
    print("-" * 66)
    print(f"{'Mean kidney dose (Gy)':<30} {s.mean_kidney_dose:>12.1f} {i.mean_kidney_dose:>14.1f}")
    print(f"{'Mean tumor dose (Gy)':<30} {s.mean_tumor_dose:>12.1f} {i.mean_tumor_dose:>14.1f}")
    print(f"{'P(kidney toxicity)':<30} {s.p_kidney_toxicity:>12.1%} {i.p_kidney_toxicity:>14.1%}")
    print(f"{'P(under-treatment)':<30} {s.p_under_treatment:>12.1%} {i.p_under_treatment:>14.1%}")
    print(f"{'P(optimal outcome)':<30} {s.p_optimal:>12.1%} {i.p_optimal:>14.1%}")
    print("=" * 66)
    reduction = s.p_kidney_toxicity / max(i.p_kidney_toxicity, 1e-6)
    print(f"Kidney toxicity risk reduction: {reduction:.1f}x")


if __name__ == "__main__":
    print(f"CPU cores available: {mp.cpu_count()}")

    print("\nRunning serial baseline (n_workers=1)...")
    t_serial = time.perf_counter()
    compare_dosing_strategies(n=8_000, seed=42, n_workers=1)
    t_serial = time.perf_counter() - t_serial
    print(f"  Serial time: {t_serial:.3f}s")

    print(f"\nRunning parallel ({mp.cpu_count()} workers)...")
    result = compare_dosing_strategies(n=8_000, seed=42)
    print(f"  Parallel time: {result.wall_time_s:.3f}s")
    speedup = t_serial / result.wall_time_s
    print(f"  Speedup: {speedup:.2f}x")

    print()
    print_comparison(result)
