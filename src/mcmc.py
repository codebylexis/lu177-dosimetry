"""
mcmc.py
=======
Metropolis-Hastings MCMC for Bayesian biokinetic parameter estimation.

Clinical scenario
-----------------
A patient has undergone SPECT imaging at 4 timepoints post-injection
(t = 4, 24, 96, 168 hours). Each scan gives a noisy measurement of the
kidney activity with ~8% coefficient of variation (typical SPECT uncertainty,
per Ljungberg et al. 2016 MIRD Pamphlet No. 26).

Goal: infer the posterior over slow-component parameters (A_slow, lambda_slow),
which dominate kidney cumulated activity and absorbed dose. Then propagate
that posterior forward through the MIRD formula to get a full probability
distribution over absorbed dose, including a safety exceedance probability
P(dose > QUANTEC limit).

Why this matters
----------------
Standard dosimetry reports a single dose number with no uncertainty.
The posterior predictive distribution shows that with 4 SPECT timepoints,
the 95% credible interval for kidney dose can span > 20 Gy -- enough to
span the difference between safe and toxic.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from simulation import (
    ORGAN_PARAMS, STANDARD_ACTIVITY_GBQ, KIDNEY_TOLERANCE_GY,
    time_activity_curve, absorbed_dose,
)


# ---------------------------------------------------------------------------
# SPECT measurement simulation
# ---------------------------------------------------------------------------

SCAN_TIMES  = np.array([4.0, 24.0, 96.0, 168.0])  # hours post-injection
SPECT_CV    = 0.08   # 8% measurement coefficient of variation


def simulate_spect_measurements(
    organ: str = "Kidneys",
    scan_times: np.ndarray = SCAN_TIMES,
    noise_cv: float = SPECT_CV,
    seed: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SPECT activity measurements for a single patient.

    Measurements are log-normally distributed around the true TAC,
    with the specified coefficient of variation.

    Returns
    -------
    (observed, true_values) : measured and true activity fractions at each timepoint
    """
    rng = np.random.default_rng(seed)
    p = ORGAN_PARAMS[organ]
    true_vals = time_activity_curve(
        scan_times, p.A_fast, p.lam_fast, p.A_slow, p.lam_slow
    )
    observed = true_vals * rng.lognormal(0, noise_cv, len(scan_times))
    return observed, true_vals


# ---------------------------------------------------------------------------
# Bayesian model: prior + likelihood + posterior
# ---------------------------------------------------------------------------

def log_prior(A_slow: float, lam_slow: float) -> float:
    """
    Weakly informative priors anchored to population data.

    A_slow    ~ N(0.18, 0.10^2)      -- from Sandstrom et al. 2013
    lam_slow  ~ HalfNormal(0.012)    -- enforces positivity
    """
    if A_slow <= 0 or lam_slow <= 0:
        return -np.inf
    lp  = stats.norm.logpdf(A_slow, 0.18, 0.10)
    lp += stats.halfnorm.logpdf(lam_slow, scale=0.012)
    return lp


def log_likelihood(
    A_slow: float,
    lam_slow: float,
    observed: np.ndarray,
    scan_times: np.ndarray,
    organ: str = "Kidneys",
    noise_cv: float = SPECT_CV,
) -> float:
    """
    Log-normal likelihood on SPECT measurements.

    Measurement model: log(obs_i) ~ N(log(pred_i), noise_cv^2)
    This is standard for SPECT activity quantification.
    """
    p    = ORGAN_PARAMS[organ]
    pred = time_activity_curve(scan_times, p.A_fast, p.lam_fast, A_slow, lam_slow)
    if np.any(pred <= 0):
        return -np.inf
    return float(np.sum(stats.norm.logpdf(np.log(observed), np.log(pred), noise_cv)))


def log_posterior(
    A_slow: float,
    lam_slow: float,
    observed: np.ndarray,
    scan_times: np.ndarray,
    **kwargs,
) -> float:
    return log_prior(A_slow, lam_slow) + log_likelihood(
        A_slow, lam_slow, observed, scan_times, **kwargs
    )


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler
# ---------------------------------------------------------------------------

@dataclass
class MCMCResult:
    """Container for MCMC output."""
    samples:         np.ndarray   # shape (n_post_burnin, 2): [A_slow, lam_slow]
    acceptance_rate: float
    n_burnin:        int
    posterior_doses: np.ndarray   # posterior predictive kidney dose (Gy)
    true_dose:       float        # MIRD dose at true parameters
    p_exceed_limit:  float        # P(dose > QUANTEC limit)


def metropolis_hastings(
    observed: np.ndarray,
    scan_times: np.ndarray = SCAN_TIMES,
    organ: str = "Kidneys",
    n_samples: int = 20_000,
    n_burnin: int = 3_000,
    prop_A_slow: float = 0.015,
    prop_lam_slow: float = 0.0004,
    seed: int = 17,
) -> MCMCResult:
    """
    Symmetric Gaussian random-walk Metropolis-Hastings for (A_slow, lam_slow).

    Algorithm
    ---------
    1. Start at population-mean parameter values
    2. Propose new state from Gaussian random walk
    3. Accept with probability min(1, pi(x*)/pi(x)) evaluated in log-space
    4. Discard burn-in samples; return post-burnin chain

    Parameters
    ----------
    observed      : array of SPECT activity measurements
    scan_times    : measurement timepoints (hours)
    n_samples     : total MCMC iterations (including burn-in)
    n_burnin      : burn-in samples to discard
    prop_A_slow   : proposal standard deviation for A_slow
    prop_lam_slow : proposal standard deviation for lam_slow
    seed          : random seed

    Returns
    -------
    MCMCResult with posterior samples, acceptance rate, and derived quantities
    """
    rng = np.random.default_rng(seed)
    p = ORGAN_PARAMS[organ]

    samples  = np.zeros((n_samples, 2))
    accepted = 0

    # Initialise at population-mean values
    A_slow_c   = p.A_slow
    lam_slow_c = p.lam_slow
    lp_c = log_posterior(A_slow_c, lam_slow_c, observed, scan_times, organ=organ)

    for i in range(n_samples):
        A_slow_prop   = A_slow_c   + rng.normal(0, prop_A_slow)
        lam_slow_prop = lam_slow_c + rng.normal(0, prop_lam_slow)

        lp_prop = log_posterior(
            A_slow_prop, lam_slow_prop, observed, scan_times, organ=organ
        )

        if np.log(rng.uniform()) < lp_prop - lp_c:
            A_slow_c, lam_slow_c, lp_c = A_slow_prop, lam_slow_prop, lp_prop
            accepted += 1

        samples[i] = [A_slow_c, lam_slow_c]

    post_samples = samples[n_burnin:]
    accept_rate  = accepted / n_samples

    # Posterior predictive dose distribution
    organ_p = ORGAN_PARAMS[organ]
    post_doses = np.array([
        absorbed_dose(organ_p.A_fast, organ_p.lam_fast, s[0], s[1],
                      organ_p.S_value, STANDARD_ACTIVITY_GBQ)
        for s in post_samples[::5]   # thin by 5 to reduce autocorrelation
    ])

    true_dose = absorbed_dose(
        organ_p.A_fast, organ_p.lam_fast,
        organ_p.A_slow, organ_p.lam_slow,
        organ_p.S_value, STANDARD_ACTIVITY_GBQ,
    )

    return MCMCResult(
        samples         = post_samples,
        acceptance_rate = accept_rate,
        n_burnin        = n_burnin,
        posterior_doses = post_doses,
        true_dose       = true_dose,
        p_exceed_limit  = float((post_doses > KIDNEY_TOLERANCE_GY).mean()),
    )


def effective_sample_size(chain: np.ndarray, max_lag: int = 300) -> float:
    """
    Estimate effective sample size from autocorrelation.

    ESS = N / tau,  tau = 1 + 2 * sum_{k=1}^{inf} rho(k)

    Autocorrelation sum is truncated when rho(k) < 0.05.
    """
    acf_vals = [
        float(np.corrcoef(chain[:-k], chain[k:])[0, 1])
        if k > 0 else 1.0
        for k in range(min(max_lag, len(chain) // 2))
    ]
    tau = 1.0 + 2.0 * sum(a for a in acf_vals[1:] if a > 0.05)
    return len(chain) / tau


def posterior_summary(result: MCMCResult, organ: str = "Kidneys") -> Dict:
    """
    Compute summary statistics for the posterior distribution.

    Returns
    -------
    dict with mean, std, 95% CI for each parameter and the dose
    """
    p = ORGAN_PARAMS[organ]
    sp = result.samples

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "A_slow": {
            "true":  p.A_slow,
            "mean":  float(sp[:, 0].mean()),
            "std":   float(sp[:, 0].std()),
            "ci95":  ci(sp[:, 0]),
            "ess":   effective_sample_size(sp[:, 0]),
        },
        "lam_slow": {
            "true":  p.lam_slow,
            "mean":  float(sp[:, 1].mean()),
            "std":   float(sp[:, 1].std()),
            "ci95":  ci(sp[:, 1]),
            "ess":   effective_sample_size(sp[:, 1]),
        },
        "kidney_dose_Gy": {
            "true":           result.true_dose,
            "mean":           float(result.posterior_doses.mean()),
            "std":            float(result.posterior_doses.std()),
            "ci95":           ci(result.posterior_doses),
            "p_exceed_limit": result.p_exceed_limit,
        },
        "acceptance_rate": result.acceptance_rate,
    }


if __name__ == "__main__":
    print("Simulating SPECT measurements...")
    observed, true_vals = simulate_spect_measurements(seed=17)
    print(f"  Scan times: {SCAN_TIMES} h")
    print(f"  True TAC:   {np.round(true_vals, 4)}")
    print(f"  Observed:   {np.round(observed, 4)}")

    print("\nRunning Metropolis-Hastings MCMC (20,000 iterations)...")
    result = metropolis_hastings(observed, seed=17)
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")

    summary = posterior_summary(result)
    print("\nPosterior Summary:")
    print(f"  A_slow:    mean={summary['A_slow']['mean']:.4f}  "
          f"true={summary['A_slow']['true']:.4f}  "
          f"95% CI={summary['A_slow']['ci95']}")
    print(f"  lam_slow:  mean={summary['lam_slow']['mean']:.5f}  "
          f"true={summary['lam_slow']['true']:.5f}  "
          f"95% CI={summary['lam_slow']['ci95']}")
    print(f"  Dose (Gy): mean={summary['kidney_dose_Gy']['mean']:.2f}  "
          f"true={summary['kidney_dose_Gy']['true']:.2f}  "
          f"95% CI={summary['kidney_dose_Gy']['ci95']}")
    print(f"  P(dose > {KIDNEY_TOLERANCE_GY} Gy) = "
          f"{summary['kidney_dose_Gy']['p_exceed_limit']:.1%}")
