"""
tests/test_mcmc.py
==================
Unit tests for Metropolis-Hastings MCMC and posterior analysis.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from simulation import ORGAN_PARAMS, KIDNEY_TOLERANCE_GY, STANDARD_ACTIVITY_GBQ
from mcmc import (
    simulate_spect_measurements, log_prior, log_likelihood,
    log_posterior, metropolis_hastings, effective_sample_size,
    posterior_summary, SCAN_TIMES, SPECT_CV,
)


# ---------------------------------------------------------------------------
# simulate_spect_measurements
# ---------------------------------------------------------------------------

class TestSPECTMeasurements:
    def test_output_shape(self):
        observed, true_vals = simulate_spect_measurements()
        assert observed.shape == true_vals.shape == (len(SCAN_TIMES),)

    def test_observed_positive(self):
        observed, _ = simulate_spect_measurements(seed=0)
        assert np.all(observed > 0)

    def test_true_vals_positive(self):
        _, true_vals = simulate_spect_measurements()
        assert np.all(true_vals > 0)

    def test_true_vals_decreasing(self):
        """Activity must decrease monotonically over time."""
        _, true_vals = simulate_spect_measurements()
        assert np.all(np.diff(true_vals) < 0)

    def test_reproducible(self):
        obs1, _ = simulate_spect_measurements(seed=5)
        obs2, _ = simulate_spect_measurements(seed=5)
        np.testing.assert_array_equal(obs1, obs2)

    def test_noise_roughly_correct_cv(self):
        """Observed values should scatter around true with ~SPECT_CV CV."""
        cvs = []
        for seed in range(200):
            obs, true = simulate_spect_measurements(seed=seed)
            cvs.append(np.abs((obs - true) / true).mean())
        assert pytest.approx(np.mean(cvs), abs=0.03) == SPECT_CV


# ---------------------------------------------------------------------------
# log_prior
# ---------------------------------------------------------------------------

class TestLogPrior:
    def test_valid_params_return_finite(self):
        lp = log_prior(0.18, 0.0045)
        assert np.isfinite(lp)

    def test_negative_A_slow_returns_neg_inf(self):
        assert log_prior(-0.01, 0.0045) == -np.inf

    def test_negative_lam_slow_returns_neg_inf(self):
        assert log_prior(0.18, -0.001) == -np.inf

    def test_zero_A_slow_returns_neg_inf(self):
        assert log_prior(0.0, 0.0045) == -np.inf

    def test_peak_near_population_mean(self):
        """Prior should be highest near the population-mean parameters."""
        p = ORGAN_PARAMS["Kidneys"]
        lp_mean = log_prior(p.A_slow, p.lam_slow)
        lp_far  = log_prior(p.A_slow * 3, p.lam_slow * 3)
        assert lp_mean > lp_far


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def setup_method(self):
        self.observed, _ = simulate_spect_measurements(seed=17)

    def test_returns_finite_for_valid_params(self):
        ll = log_likelihood(0.18, 0.0045, self.observed, SCAN_TIMES)
        assert np.isfinite(ll)

    def test_posterior_neg_lam_returns_neg_inf(self):
        """log_posterior (not log_likelihood) guards against negative parameters."""
        lp = log_posterior(0.18, -0.001, self.observed, SCAN_TIMES)
        assert lp == -np.inf

    def test_higher_at_true_params_than_far_away(self):
        p = ORGAN_PARAMS["Kidneys"]
        ll_true = log_likelihood(p.A_slow, p.lam_slow, self.observed, SCAN_TIMES)
        ll_far  = log_likelihood(p.A_slow * 5, p.lam_slow * 5, self.observed, SCAN_TIMES)
        assert ll_true > ll_far


# ---------------------------------------------------------------------------
# metropolis_hastings (fast subset for CI)
# ---------------------------------------------------------------------------

class TestMetropolisHastings:
    @pytest.fixture(scope="class")
    def mcmc_result(self):
        observed, _ = simulate_spect_measurements(seed=17)
        return metropolis_hastings(
            observed, n_samples=5_000, n_burnin=1_000, seed=17
        )

    def test_acceptance_rate_in_valid_range(self, mcmc_result):
        """Acceptance rate for a well-tuned sampler should be 15-70%."""
        assert 0.15 < mcmc_result.acceptance_rate < 0.70

    def test_samples_shape(self, mcmc_result):
        n_post = 5_000 - 1_000
        assert mcmc_result.samples.shape == (n_post, 2)

    def test_samples_positive(self, mcmc_result):
        """Both A_slow and lam_slow must be positive."""
        assert np.all(mcmc_result.samples > 0)

    def test_posterior_doses_positive(self, mcmc_result):
        assert np.all(mcmc_result.posterior_doses > 0)

    def test_p_exceed_in_unit_interval(self, mcmc_result):
        assert 0.0 <= mcmc_result.p_exceed_limit <= 1.0

    def test_true_dose_near_reference(self, mcmc_result):
        """True dose (at population-mean params) should match analytical reference."""
        from simulation import compute_reference_doses
        ref = compute_reference_doses()["Kidneys"]
        assert pytest.approx(mcmc_result.true_dose, rel=1e-6) == ref

    def test_posterior_mean_near_true(self, mcmc_result):
        """Posterior mean should recover the true parameter within 30%."""
        p = ORGAN_PARAMS["Kidneys"]
        post_mean_A   = mcmc_result.samples[:, 0].mean()
        post_mean_lam = mcmc_result.samples[:, 1].mean()
        assert abs(post_mean_A   - p.A_slow)   / p.A_slow   < 0.30
        assert abs(post_mean_lam - p.lam_slow) / p.lam_slow < 0.30


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------

class TestESS:
    def test_independent_chain_ess_near_n(self):
        """IID samples should have ESS close to n."""
        rng = np.random.default_rng(0)
        chain = rng.standard_normal(1000)
        ess = effective_sample_size(chain)
        assert ess > 200   # reasonable ESS for IID (autocorrelation estimator has variance)

    def test_autocorrelated_chain_ess_lower(self):
        """Strongly autocorrelated chain should have much lower ESS."""
        rng   = np.random.default_rng(0)
        noise = rng.standard_normal(1000)
        # AR(1) with rho=0.95
        chain = np.zeros(1000)
        for i in range(1, 1000):
            chain[i] = 0.95 * chain[i-1] + 0.1 * noise[i]
        ess = effective_sample_size(chain)
        assert ess < 200   # strong autocorrelation degrades ESS

    def test_single_element_chain(self):
        ess = effective_sample_size(np.array([1.0]))
        assert ess > 0


# ---------------------------------------------------------------------------
# posterior_summary
# ---------------------------------------------------------------------------

class TestPosteriorSummary:
    @pytest.fixture(scope="class")
    def summary(self):
        observed, _ = simulate_spect_measurements(seed=17)
        result = metropolis_hastings(
            observed, n_samples=5_000, n_burnin=1_000, seed=17
        )
        return posterior_summary(result)

    def test_has_expected_keys(self, summary):
        assert {"A_slow", "lam_slow", "kidney_dose_Gy", "acceptance_rate"} == set(summary.keys())

    def test_ci95_ordered(self, summary):
        """Lower CI bound must be less than upper bound."""
        for key in ["A_slow", "lam_slow", "kidney_dose_Gy"]:
            lo, hi = summary[key]["ci95"]
            assert lo < hi, f"CI95 inverted for {key}"

    def test_mean_inside_ci95(self, summary):
        for key in ["A_slow", "lam_slow", "kidney_dose_Gy"]:
            lo, hi = summary[key]["ci95"]
            mean   = summary[key]["mean"]
            assert lo <= mean <= hi, f"Mean outside CI95 for {key}"

    def test_ess_positive(self, summary):
        assert summary["A_slow"]["ess"] > 0
        assert summary["lam_slow"]["ess"] > 0

    def test_p_exceed_in_unit_interval(self, summary):
        p = summary["kidney_dose_Gy"]["p_exceed_limit"]
        assert 0.0 <= p <= 1.0
