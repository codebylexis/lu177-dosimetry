"""
variance_reduction.py
=====================
Four variance reduction techniques for Monte Carlo dosimetry estimation.

Each estimator is unbiased and returns (mean_dose, standard_error).
Variance reduction factor (VRF) = Var(Naive MC) / Var(method).

Methods implemented
-------------------
1. Antithetic Variates   -- pairs each draw with its log-normal mirror
2. Control Variates      -- uses fast-component CA as a correlated control
3. Importance Sampling   -- oversamples the high-dose tail
4. Quasi-Monte Carlo     -- Halton low-discrepancy sequences (bases 2, 3, 5)
"""

import numpy as np
from scipy.stats import norm as sp_norm
from typing import Tuple, Dict

from simulation import ORGAN_PARAMS, INTER_PATIENT_CV, STANDARD_ACTIVITY_GBQ


# ---------------------------------------------------------------------------
# 1. Antithetic Variates
# ---------------------------------------------------------------------------

def antithetic_variates(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Antithetic variates estimator.

    For each log-normal draw z, also evaluate at -z. Because the dose
    function is monotone increasing in each biokinetic parameter, and
    exp(sigma*z) and exp(-sigma*z) are negatively correlated, their
    average has lower variance:

        Var(antithetic) = Var(naive) * (1 + rho) / 2,  rho < 0

    No extra model evaluations beyond the pairing.

    Parameters
    ----------
    n : total number of patient draws (n/2 pairs)
    """
    rng = np.random.default_rng(seed)
    p  = ORGAN_PARAMS[organ]
    m  = n // 2

    z_As  = rng.normal(0, 1, m)
    z_ls  = rng.normal(0, 1, m)
    z_Af  = rng.normal(0, 1, m)

    def dose_from_z(zA, zl, zf):
        A_slow_s  = p.A_slow  * np.exp(cv        * zA)
        lam_slow_s = p.lam_slow * np.exp(cv * 0.5  * zl)
        A_fast_s  = p.A_fast  * np.exp(cv * 0.8   * zf)
        ca = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
        return p.S_value * ca * activity_GBq * 1000

    d1 = dose_from_z( z_As,  z_ls,  z_Af)
    d2 = dose_from_z(-z_As, -z_ls, -z_Af)
    paired = (d1 + d2) / 2

    return float(np.mean(paired)), float(np.std(paired) / np.sqrt(m))


# ---------------------------------------------------------------------------
# 2. Control Variates
# ---------------------------------------------------------------------------

def control_variates(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Control variates estimator.

    Control: fast-component cumulated activity g = A_fast_s / lam_fast
    Known expectation: E[g] = A_fast / lam_fast  (analytical)

    Adjusted estimator:
        D_adj = D + c* * (g - E[g])
        c*    = -Cov(D, g) / Var(g)   (OLS optimal coefficient)

    Variance reduction: 1 - rho^2 where rho = Corr(D, g).
    """
    rng = np.random.default_rng(seed)
    p = ORGAN_PARAMS[organ]

    A_slow_s   = p.A_slow   * rng.lognormal(0, cv,        n)
    lam_slow_s = p.lam_slow * rng.lognormal(0, cv * 0.5,  n)
    A_fast_s   = p.A_fast   * rng.lognormal(0, cv * 0.8,  n)

    ca    = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
    doses = p.S_value * ca * activity_GBq * 1000

    ctrl   = A_fast_s / p.lam_fast          # control variate
    E_ctrl = p.A_fast / p.lam_fast          # analytical expectation

    cov_mat = np.cov(doses, ctrl)
    c_star  = -cov_mat[0, 1] / cov_mat[1, 1]
    adjusted = doses + c_star * (ctrl - E_ctrl)

    return float(np.mean(adjusted)), float(np.std(adjusted) / np.sqrt(n))


# ---------------------------------------------------------------------------
# 3. Importance Sampling
# ---------------------------------------------------------------------------

def importance_sampling(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Importance sampling estimator.

    Proposal: shift the log-normal sampling distribution toward the
    high-dose tail (higher A_slow, slower clearance). The Radon-Nikodym
    derivative (likelihood ratio) corrects for the distributional shift,
    preserving unbiasedness.

    Particularly effective for estimating tail risk (P(dose > threshold))
    and for out-of-the-money scenarios where rare events dominate.
    """
    rng   = np.random.default_rng(seed)
    p     = ORGAN_PARAMS[organ]
    shift = cv * 0.8   # shift magnitude toward high-dose tail

    z_As = rng.normal( shift,          1, n)
    z_ls = rng.normal(-shift * 0.3,    1, n)
    z_Af = rng.normal( shift * 0.5,    1, n)

    A_slow_s   = p.A_slow   * np.exp(cv       * z_As)
    lam_slow_s = p.lam_slow * np.exp(cv * 0.5 * z_ls)
    A_fast_s   = p.A_fast   * np.exp(cv * 0.8 * z_Af)

    ca    = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
    doses = p.S_value * ca * activity_GBq * 1000

    # Likelihood ratio: dP/dQ for each shifted dimension
    lr = np.exp(
        -shift       * z_As + 0.5 * shift**2
        + shift*0.3  * z_ls - 0.5 * (shift * 0.3)**2
        - shift*0.5  * z_Af + 0.5 * (shift * 0.5)**2
    )
    weighted = doses * lr

    return float(np.mean(weighted)), float(np.std(weighted) / np.sqrt(n))


# ---------------------------------------------------------------------------
# 4. Quasi-Monte Carlo (Halton sequences)
# ---------------------------------------------------------------------------

def halton_sequence(n: int, base: int) -> np.ndarray:
    """
    Generate n terms of the Halton low-discrepancy sequence in given base.

    The Halton sequence covers [0,1] more uniformly than pseudo-random
    sampling by reflecting the base-b representation of integers about
    the decimal point. Convergence: O(log(n)^d / n) vs O(1/sqrt(n)).
    """
    seq = np.zeros(n)
    for i in range(1, n + 1):
        f, r, j = 1.0, 0.0, i
        while j > 0:
            f /= base
            r += f * (j % base)
            j  = j // base
        seq[i - 1] = r
    return seq


def quasi_monte_carlo(
    n: int,
    organ: str = "Kidneys",
    activity_GBq: float = STANDARD_ACTIVITY_GBQ,
    cv: float = INTER_PATIENT_CV,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Quasi-Monte Carlo estimator using Halton sequences.

    Three independent Halton sequences (bases 2, 3, 5) drive the three
    biokinetic parameters. Transformed through the standard normal
    inverse CDF (norm.ppf) to produce quasi-normal draws.

    Particularly effective in low dimensions (d <= 10) for smooth
    integrands. Advantage diminishes in high dimensions.
    """
    p = ORGAN_PARAMS[organ]

    u1 = np.clip(halton_sequence(n, 2), 1e-9, 1 - 1e-9)
    u2 = np.clip(halton_sequence(n, 3), 1e-9, 1 - 1e-9)
    u3 = np.clip(halton_sequence(n, 5), 1e-9, 1 - 1e-9)

    A_slow_s   = p.A_slow   * np.exp(cv       * sp_norm.ppf(u1))
    lam_slow_s = p.lam_slow * np.exp(cv * 0.5 * sp_norm.ppf(u2))
    A_fast_s   = p.A_fast   * np.exp(cv * 0.8 * sp_norm.ppf(u3))

    ca    = A_fast_s / p.lam_fast + A_slow_s / lam_slow_s
    doses = p.S_value * ca * activity_GBq * 1000

    return float(np.mean(doses)), float(np.std(doses) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Benchmark all methods
# ---------------------------------------------------------------------------

ALL_METHODS = {
    "Naive MC":            lambda n, **kw: _naive_wrap(n, **kw),
    "Antithetic Variates": antithetic_variates,
    "Control Variates":    control_variates,
    "Importance Sampling": importance_sampling,
    "Quasi-MC (Halton)":   quasi_monte_carlo,
}


def _naive_wrap(n, seed=None, **kw):
    from simulation import naive_mc_dose
    return naive_mc_dose(n, seed=seed, **kw)


def benchmark_vrf(
    n: int = 500,
    organ: str = "Kidneys",
    n_trials: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute variance reduction factors for all methods at sample size n.

    VRF = Var(Naive MC) / Var(method)
    A VRF of k means the method needs only 1/k as many samples as naive MC
    to achieve the same precision.

    Returns
    -------
    dict mapping method name -> VRF
    """
    rng = np.random.default_rng(seed)
    from simulation import naive_mc_dose

    baseline_estimates = [
        naive_mc_dose(n, organ, seed=int(rng.integers(1 << 31)))[0]
        for _ in range(n_trials)
    ]
    baseline_var = np.var(baseline_estimates)

    vrfs = {}
    for name, fn in ALL_METHODS.items():
        estimates = [
            fn(n, organ=organ, seed=int(rng.integers(1 << 31)))[0]
            for _ in range(n_trials)
        ]
        v = np.var(estimates)
        vrfs[name] = float(baseline_var / max(v, 1e-15))

    return vrfs


if __name__ == "__main__":
    print("Variance Reduction Factor Benchmark (n=500, kidneys, 200 trials)")
    print("-" * 55)
    vrfs = benchmark_vrf(n=500, seed=42)
    for name, vrf in vrfs.items():
        bar = "#" * min(int(vrf * 2), 40)
        print(f"  {name:<25}: {vrf:7.1f}x  {bar}")
