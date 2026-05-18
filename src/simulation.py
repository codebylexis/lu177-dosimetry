"""
simulation.py
=============
Core biokinetic model and baseline Monte Carlo dosimetry estimator
for Lu-177 DOTATATE targeted radionuclide therapy.

Physical model
--------------
Organ time-activity curves follow a bi-exponential model:
    A(t) = A_fast * exp(-lambda_fast * t) + A_slow * exp(-lambda_slow * t)

Absorbed dose is computed via the MIRD formula:
    D = S * cumulated_activity * administered_activity

Parameters sourced from:
  - Sandstrom et al. (2013), J Nucl Med 54(1):33-41
  - ICRP Publication 53 (S-values for Lu-177)
  - QUANTEC kidney tolerance: Dawson et al. (2010)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Organ biokinetic parameters and S-values
# ---------------------------------------------------------------------------

@dataclass
class OrganParams:
    """Biokinetic parameters and dosimetric constants for one organ."""
    A_fast:     float   # fast-component amplitude (fraction of injected activity)
    lam_fast:   float   # fast-component clearance rate (h^-1)
    A_slow:     float   # slow-component amplitude (fraction of injected activity)
    lam_slow:   float   # slow-component clearance rate (h^-1)
    mass_g:     float   # organ mass (grams)
    S_value:    float   # MIRD S-value (Gy per MBq·h)


# Population-mean biokinetic parameters (Sandstrom et al. 2013, Table 3)
# S-values for Lu-177 from ICRP Publication 53
ORGAN_PARAMS: Dict[str, OrganParams] = {
    "Tumor":      OrganParams(0.10, 0.080, 0.15, 0.0030,   20.0, 2.1e-4),
    "Kidneys":    OrganParams(0.12, 0.060, 0.18, 0.0045,  290.0, 8.4e-5),
    "Liver":      OrganParams(0.08, 0.050, 0.06, 0.0020, 1800.0, 1.2e-5),
    "Spleen":     OrganParams(0.05, 0.070, 0.04, 0.0025,  150.0, 1.6e-4),
    "Red Marrow": OrganParams(0.02, 0.040, 0.01, 0.0010, 1500.0, 3.1e-5),
}

# Clinical constants
STANDARD_ACTIVITY_GBQ = 7.4   # GBq per treatment cycle (FDA-approved dose)
KIDNEY_TOLERANCE_GY   = 23.0  # Gy  — QUANTEC mean kidney dose limit
TUMOR_MIN_GY          = 40.0  # Gy  — minimum estimated tumoricidal threshold
INTER_PATIENT_CV      = 0.30  # 30% coefficient of variation (Sandstrom 2013)


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def time_activity_curve(
    t: np.ndarray,
    A_fast: float, lam_fast: float,
    A_slow: float, lam_slow: float,
) -> np.ndarray:
    """
    Bi-exponential time-activity curve.

    Parameters
    ----------
    t        : array of time points (hours post-injection)
    A_fast   : fast-component amplitude (fraction of injected activity)
    lam_fast : fast-component clearance rate (h^-1)
    A_slow   : slow-component amplitude (fraction of injected activity)
    lam_slow : slow-component clearance rate (h^-1)

    Returns
    -------
    A(t) : fractional activity remaining in organ at each time point
    """
    return A_fast * np.exp(-lam_fast * t) + A_slow * np.exp(-lam_slow * t)


def cumulated_activity(
    A_fast: float, lam_fast: float,
    A_slow: float, lam_slow: float,
) -> float:
    """
    Analytical residence time (cumulated activity).

    Integral of A(t) from 0 to infinity:
        = A_fast / lam_fast + A_slow / lam_slow

    Units: h * (fraction of injected activity)
    """
    return A_fast / lam_fast + A_slow / lam_slow


def absorbed_dose(
    A_fast: float, lam_fast: float,
    A_slow: float, lam_slow: float,
    S_value: float,
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
) -> float:
    """
    MIRD absorbed dose formula.

    D = S * cumulated_activity * administered_activity

    Parameters
    ----------
    S_value      : organ S-value (Gy per MBq·h)
    activity_GBq : administered activity in GBq (converted to MBq internally)

    Returns
    -------
    Absorbed dose in Gray (Gy)
    """
    ca = cumulated_activity(A_fast, lam_fast, A_slow, lam_slow)
    return S_value * ca * activity_GBq * 1000  # 1000: GBq -> MBq


# ---------------------------------------------------------------------------
# Reference doses (zero inter-patient variability)
# ---------------------------------------------------------------------------

def compute_reference_doses(
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
) -> Dict[str, float]:
    """
    Compute absorbed dose for each organ at population-mean parameters.
    These are the ground-truth values used to measure MC estimator error.

    Returns
    -------
    dict mapping organ name -> absorbed dose in Gy
    """
    return {
        organ: absorbed_dose(
            p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
            p.S_value, activity_GBq,
        )
        for organ, p in ORGAN_PARAMS.items()
    }


# ---------------------------------------------------------------------------
# Naive Monte Carlo dosimetry estimator
# ---------------------------------------------------------------------------

def sample_patient_doses(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate n patients with log-normal inter-patient biokinetic variability.

    Each patient's absorbed dose is computed from individually sampled
    TAC parameters. Log-normal variability is a standard assumption for
    positive biological quantities (Sandstrom et al. 2013).

    Parameters
    ----------
    n           : number of simulated patients
    organ       : organ name (key in ORGAN_PARAMS)
    activity_GBq: administered activity in GBq
    cv          : coefficient of variation for inter-patient variability
    seed        : random seed for reproducibility

    Returns
    -------
    doses : array of shape (n,) — absorbed dose per patient in Gy
    """
    rng = np.random.default_rng(seed)

    p = ORGAN_PARAMS[organ]

    # Log-normal variability: multiply each parameter by exp(sigma * Z)
    # where sigma ~ cv for small cv (exact for log-normal)
    A_slow_s  = p.A_slow  * rng.lognormal(0, cv,        n)
    lam_slow_s = p.lam_slow * rng.lognormal(0, cv * 0.5, n)
    A_fast_s  = p.A_fast  * rng.lognormal(0, cv * 0.8,  n)

    ca    = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
    doses = p.S_value * ca * activity_GBq * 1000

    return doses


def naive_mc_dose(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Naive Monte Carlo mean absorbed dose estimator.

    Returns
    -------
    (estimate, standard_error) in Gy
    """
    doses = sample_patient_doses(n, organ, activity_GBq, cv, seed)
    return float(np.mean(doses)), float(np.std(doses) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def convergence_experiment(
    sample_sizes: np.ndarray = None,
    organ: str = "Kidneys",
    n_trials: int = 80,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Run convergence experiment: measure MC error vs. sample size.

    Returns
    -------
    dict with keys: 'sample_sizes', 'errors', 'stds', 'theory'
    """
    if sample_sizes is None:
        sample_sizes = np.logspace(1, 4.5, 35).astype(int)

    rng = np.random.default_rng(seed)
    true_dose = compute_reference_doses()[organ]
    errors, stds = [], []

    for n in sample_sizes:
        trials = [naive_mc_dose(n, organ, seed=int(rng.integers(1 << 31)))[0] for _ in range(n_trials)]
        errors.append(abs(np.mean(trials) - true_dose))
        stds.append(np.std(trials))

    stds = np.array(stds)
    mid  = len(sample_sizes) // 2
    theory = stds[mid] * np.sqrt(sample_sizes[mid]) / np.sqrt(sample_sizes)

    return {
        "sample_sizes": sample_sizes,
        "errors":  np.array(errors),
        "stds":    stds,
        "theory":  theory,
        "true_dose": true_dose,
    }


if __name__ == "__main__":
    doses = compute_reference_doses()
    print("Reference absorbed doses at 7.4 GBq:")
    print("-" * 40)
    for organ, dose in doses.items():
        flag = " <-- exceeds QUANTEC limit" if organ == "Kidneys" and dose > KIDNEY_TOLERANCE_GY else ""
        print(f"  {organ:<15}: {dose:6.2f} Gy{flag}")

    print("\nNaive MC estimate (n=5000, kidneys):")
    est, se = naive_mc_dose(5000, seed=42)
    print(f"  {est:.4f} +/- {se:.4f} Gy  (true: {doses['Kidneys']:.4f} Gy)")
