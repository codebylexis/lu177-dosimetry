"""
tests/test_variance_reduction.py
=================================
Unit tests for variance reduction estimators.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from simulation import compute_reference_doses, ORGAN_PARAMS, INTER_PATIENT_CV
from variance_reduction import (
    antithetic_variates, control_variates,
    importance_sampling, quasi_monte_carlo,
    halton_sequence, benchmark_vrf,
)

KIDNEY_REF = compute_reference_doses()["Kidneys"]
RTOL = 0.10   # 10% relative tolerance for stochastic estimators at n=5000


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def large_n():
    return 5_000


# ---------------------------------------------------------------------------
# Halton sequence
# ---------------------------------------------------------------------------

class TestHaltonSequence:
    def test_length(self):
        seq = halton_sequence(100, 2)
        assert len(seq) == 100

    def test_in_unit_interval(self):
        for base in [2, 3, 5, 7]:
            seq = halton_sequence(200, base)
            assert np.all(seq >= 0) and np.all(seq <= 1)

    def test_base_2_first_values(self):
        """First 4 values of base-2 Halton: 0.5, 0.25, 0.75, 0.125."""
        seq = halton_sequence(4, 2)
        expected = [0.5, 0.25, 0.75, 0.125]
        np.testing.assert_allclose(seq, expected, atol=1e-12)

    def test_low_discrepancy_vs_uniform(self):
        """Halton should cover [0,1] more uniformly than random draws."""
        n = 500
        halton = halton_sequence(n, 2)
        rng    = np.random.default_rng(0)
        random = rng.uniform(0, 1, n)
        # Max deviation from uniform CDF (Kolmogorov-Smirnov statistic)
        from scipy.stats import kstest
        ks_halton = kstest(halton, 'uniform').statistic
        ks_random = kstest(random, 'uniform').statistic
        assert ks_halton < ks_random

    def test_deterministic(self):
        s1 = halton_sequence(50, 3)
        s2 = halton_sequence(50, 3)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Antithetic Variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:
    def test_returns_two_floats(self):
        est, se = antithetic_variates(200)
        assert isinstance(est, float) and isinstance(se, float)

    def test_estimate_near_reference(self, large_n):
        est, _ = antithetic_variates(large_n)
        assert abs(est - KIDNEY_REF) / KIDNEY_REF < RTOL

    def test_se_positive(self):
        _, se = antithetic_variates(200)
        assert se > 0

    def test_lower_variance_than_naive(self):
        """Antithetic should have strictly lower variance than naive MC."""
        np.random.seed(0)
        n_trials = 200
        n = 500

        from simulation import naive_mc_dose
        naive_ests   = [naive_mc_dose(n)[0] for _ in range(n_trials)]
        anti_ests    = [antithetic_variates(n)[0] for _ in range(n_trials)]

        assert np.var(anti_ests) < np.var(naive_ests)

    @pytest.mark.parametrize("organ", list(ORGAN_PARAMS))
    def test_all_organs(self, organ):
        est, se = antithetic_variates(200, organ=organ)
        assert est > 0 and se > 0


# ---------------------------------------------------------------------------
# Control Variates
# ---------------------------------------------------------------------------

class TestControlVariates:
    def test_returns_two_floats(self):
        est, se = control_variates(200)
        assert isinstance(est, float) and isinstance(se, float)

    def test_estimate_near_reference(self, large_n):
        np.random.seed(0)
        est, _ = control_variates(large_n)
        assert abs(est - KIDNEY_REF) / KIDNEY_REF < RTOL

    def test_se_positive(self):
        _, se = control_variates(200)
        assert se > 0

    def test_mean_correction_unbiased(self):
        """The control variate adjustment should not introduce bias."""
        np.random.seed(42)
        n_trials = 300
        ests = [control_variates(300)[0] for _ in range(n_trials)]
        assert abs(np.mean(ests) - KIDNEY_REF) / KIDNEY_REF < 0.12  # lognormal inflation


# ---------------------------------------------------------------------------
# Importance Sampling
# ---------------------------------------------------------------------------

class TestImportanceSampling:
    def test_returns_two_floats(self):
        est, se = importance_sampling(200)
        assert isinstance(est, float) and isinstance(se, float)

    def test_estimate_near_reference(self, large_n):
        np.random.seed(0)
        est, _ = importance_sampling(large_n)
        assert abs(est - KIDNEY_REF) / KIDNEY_REF < RTOL

    def test_se_positive(self):
        _, se = importance_sampling(200)
        assert se > 0

    def test_unbiased_in_expectation(self):
        """IS estimator should be unbiased on average."""
        np.random.seed(7)
        n_trials = 300
        ests = [importance_sampling(300)[0] for _ in range(n_trials)]
        assert abs(np.mean(ests) - KIDNEY_REF) / KIDNEY_REF < 0.12  # lognormal inflation


# ---------------------------------------------------------------------------
# Quasi-Monte Carlo
# ---------------------------------------------------------------------------

class TestQuasiMonteCarlo:
    def test_returns_two_floats(self):
        est, se = quasi_monte_carlo(200)
        assert isinstance(est, float) and isinstance(se, float)

    def test_estimate_near_reference(self, large_n):
        est, _ = quasi_monte_carlo(large_n)
        assert abs(est - KIDNEY_REF) / KIDNEY_REF < RTOL

    def test_deterministic(self):
        """QMC uses Halton sequences, so results are deterministic."""
        est1, _ = quasi_monte_carlo(500)
        est2, _ = quasi_monte_carlo(500)
        assert est1 == est2

    def test_best_vrf(self):
        """QMC should achieve the highest VRF among all methods."""
        vrfs = benchmark_vrf(n=500, n_trials=100, seed=42)
        best = max(vrfs, key=vrfs.get)
        assert "Quasi" in best, f"Expected QMC to win, got {best}"


# ---------------------------------------------------------------------------
# Benchmark VRF
# ---------------------------------------------------------------------------

class TestBenchmarkVRF:
    def test_returns_all_methods(self):
        vrfs = benchmark_vrf(n=200, n_trials=30, seed=0)
        expected = {"Naive MC", "Antithetic Variates", "Control Variates",
                    "Importance Sampling", "Quasi-MC (Halton)"}
        assert set(vrfs.keys()) == expected

    def test_naive_mc_vrf_is_one(self):
        vrfs = benchmark_vrf(n=200, n_trials=50, seed=0)
        assert pytest.approx(vrfs["Naive MC"], abs=0.3) == 1.0

    def test_all_vrfs_positive(self):
        vrfs = benchmark_vrf(n=200, n_trials=30, seed=0)
        for name, vrf in vrfs.items():
            assert vrf > 0, f"Non-positive VRF for {name}"

    def test_antithetic_vrf_above_one(self):
        vrfs = benchmark_vrf(n=500, n_trials=100, seed=42)
        assert vrfs["Antithetic Variates"] > 1.5
