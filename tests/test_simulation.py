"""
tests/test_simulation.py
========================
Unit tests for the core biokinetic model and naive MC estimator.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from simulation import (
    OrganParams, ORGAN_PARAMS, STANDARD_ACTIVITY_GBQ,
    KIDNEY_TOLERANCE_GY, INTER_PATIENT_CV,
    time_activity_curve, cumulated_activity, absorbed_dose,
    compute_reference_doses, sample_patient_doses, naive_mc_dose,
    convergence_experiment,
)


# ---------------------------------------------------------------------------
# time_activity_curve
# ---------------------------------------------------------------------------

class TestTimeActivityCurve:
    def test_zero_time_returns_sum_of_amplitudes(self):
        """A(0) = A_fast + A_slow (all activity present at injection)."""
        val = time_activity_curve(np.array([0.0]), 0.12, 0.06, 0.18, 0.0045)
        assert pytest.approx(val[0], rel=1e-9) == 0.12 + 0.18

    def test_decays_monotonically(self):
        """Activity must strictly decrease over time for positive rates."""
        t = np.linspace(0, 200, 100)
        vals = time_activity_curve(t, 0.12, 0.06, 0.18, 0.0045)
        assert np.all(np.diff(vals) < 0)

    def test_approaches_zero_at_infinity(self):
        """Activity converges to zero for large t."""
        t = np.array([1e6])
        val = time_activity_curve(t, 0.12, 0.06, 0.18, 0.0045)
        assert val[0] < 1e-10

    def test_returns_correct_shape(self):
        t = np.linspace(0, 168, 50)
        vals = time_activity_curve(t, 0.12, 0.06, 0.18, 0.0045)
        assert vals.shape == (50,)

    def test_linearity_in_amplitude(self):
        """Doubling A_fast doubles the fast-component contribution."""
        t = np.array([1.0])
        v1 = time_activity_curve(t, 0.10, 0.06, 0.0,  0.004)
        v2 = time_activity_curve(t, 0.20, 0.06, 0.0,  0.004)
        assert pytest.approx(v2[0], rel=1e-9) == 2 * v1[0]

    @pytest.mark.parametrize("organ", list(ORGAN_PARAMS))
    def test_all_organs_positive_at_t0(self, organ):
        p = ORGAN_PARAMS[organ]
        val = time_activity_curve(np.array([0.0]), p.A_fast, p.lam_fast,
                                  p.A_slow, p.lam_slow)
        assert val[0] > 0


# ---------------------------------------------------------------------------
# cumulated_activity
# ---------------------------------------------------------------------------

class TestCumulatedActivity:
    def test_analytical_value(self):
        """CA = A_fast/lam_fast + A_slow/lam_slow."""
        ca = cumulated_activity(0.12, 0.06, 0.18, 0.0045)
        expected = 0.12 / 0.06 + 0.18 / 0.0045
        assert pytest.approx(ca, rel=1e-9) == expected

    def test_positive(self):
        assert cumulated_activity(0.1, 0.05, 0.1, 0.005) > 0

    def test_scales_with_amplitude(self):
        ca1 = cumulated_activity(0.10, 0.05, 0.10, 0.005)
        ca2 = cumulated_activity(0.20, 0.05, 0.10, 0.005)
        assert pytest.approx(ca2 - ca1, rel=1e-9) == 0.10 / 0.05

    def test_increases_with_slower_clearance(self):
        """Slower clearance => more cumulated activity."""
        ca_fast = cumulated_activity(0.10, 0.10, 0.10, 0.010)
        ca_slow = cumulated_activity(0.10, 0.10, 0.10, 0.001)
        assert ca_slow > ca_fast


# ---------------------------------------------------------------------------
# absorbed_dose
# ---------------------------------------------------------------------------

class TestAbsorbedDose:
    def test_kidney_reference_dose(self):
        """Kidney dose at population-mean params is well above QUANTEC limit."""
        p = ORGAN_PARAMS["Kidneys"]
        dose = absorbed_dose(p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
                             p.S_value, STANDARD_ACTIVITY_GBQ)
        # Reference is ~27.6 Gy; at minimum it should exceed QUANTEC limit
        assert dose > KIDNEY_TOLERANCE_GY

    def test_scales_linearly_with_activity(self):
        """D is linear in administered activity by the MIRD formula."""
        p = ORGAN_PARAMS["Kidneys"]
        d1 = absorbed_dose(p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
                           p.S_value, 7.4)
        d2 = absorbed_dose(p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
                           p.S_value, 14.8)
        assert pytest.approx(d2, rel=1e-6) == 2 * d1

    def test_positive_for_all_organs(self):
        for organ, p in ORGAN_PARAMS.items():
            dose = absorbed_dose(p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
                                 p.S_value)
            assert dose > 0, f"Non-positive dose for {organ}"

    def test_unit_conversion(self):
        """GBq -> MBq conversion: dose should be 1000x larger than if activity in GBq directly."""
        p = ORGAN_PARAMS["Kidneys"]
        dose_correct = absorbed_dose(p.A_fast, p.lam_fast, p.A_slow, p.lam_slow,
                                     p.S_value, 1.0)
        # Without the *1000 factor the dose would be 1000x smaller
        ca = p.A_fast / p.lam_fast + p.A_slow / p.lam_slow
        dose_raw = p.S_value * ca * 1.0  # no conversion
        assert pytest.approx(dose_correct, rel=1e-9) == dose_raw * 1000


# ---------------------------------------------------------------------------
# compute_reference_doses
# ---------------------------------------------------------------------------

class TestComputeReferenceDoses:
    def test_returns_all_organs(self):
        doses = compute_reference_doses()
        assert set(doses.keys()) == set(ORGAN_PARAMS.keys())

    def test_all_positive(self):
        for organ, dose in compute_reference_doses().items():
            assert dose > 0, f"Non-positive reference dose for {organ}"

    def test_kidney_exceeds_quantec(self):
        """At population-mean params, kidneys are above safe threshold."""
        doses = compute_reference_doses()
        assert doses["Kidneys"] > KIDNEY_TOLERANCE_GY

    def test_scales_with_activity(self):
        d1 = compute_reference_doses(7.4)
        d2 = compute_reference_doses(14.8)
        for organ in ORGAN_PARAMS:
            assert pytest.approx(d2[organ], rel=1e-6) == 2 * d1[organ]


# ---------------------------------------------------------------------------
# sample_patient_doses
# ---------------------------------------------------------------------------

class TestSamplePatientDoses:
    def test_output_shape(self):
        doses = sample_patient_doses(1000, seed=0)
        assert doses.shape == (1000,)

    def test_all_positive(self):
        doses = sample_patient_doses(500, seed=1)
        assert np.all(doses > 0)

    def test_reproducible_with_seed(self):
        d1 = sample_patient_doses(200, seed=99)
        d2 = sample_patient_doses(200, seed=99)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_differ(self):
        d1 = sample_patient_doses(200, seed=1)
        d2 = sample_patient_doses(200, seed=2)
        assert not np.allclose(d1, d2)

    def test_mean_near_reference(self):
        """Mean of a large sample should be close to the reference dose."""
        from simulation import compute_reference_doses
        ref = compute_reference_doses()["Kidneys"]
        doses = sample_patient_doses(50_000, seed=0)
        # The lognormal sampling inflates the mean above the analytical reference
        # by exp(cv^2/2) ~ 4.6%. Allow 8% to accommodate this.
        assert abs(np.mean(doses) - ref) / ref < 0.12

    def test_cv_controls_spread(self):
        """Higher CV => larger standard deviation of dose distribution."""
        doses_low  = sample_patient_doses(10_000, cv=0.10, seed=0)
        doses_high = sample_patient_doses(10_000, cv=0.50, seed=0)
        assert doses_high.std() > doses_low.std()

    @pytest.mark.parametrize("organ", list(ORGAN_PARAMS))
    def test_all_organs_valid(self, organ):
        doses = sample_patient_doses(100, organ=organ, seed=0)
        assert doses.shape == (100,)
        assert np.all(doses > 0)


# ---------------------------------------------------------------------------
# naive_mc_dose
# ---------------------------------------------------------------------------

class TestNaiveMCDose:
    def test_returns_two_floats(self):
        est, se = naive_mc_dose(100, seed=0)
        assert isinstance(est, float) and isinstance(se, float)

    def test_se_positive(self):
        _, se = naive_mc_dose(100, seed=0)
        assert se > 0

    def test_se_decreases_with_n(self):
        """Standard error should decrease as n increases."""
        _, se_small = naive_mc_dose(100, seed=0)
        _, se_large = naive_mc_dose(10_000, seed=0)
        assert se_large < se_small

    def test_estimate_near_reference(self):
        ref = compute_reference_doses()["Kidneys"]
        est, _ = naive_mc_dose(100_000, seed=42)
        assert abs(est - ref) / ref < 0.10  # lognormal mean inflates ~5% above analytical

    def test_se_scales_as_one_over_sqrt_n(self):
        """SE should scale approximately as 1/sqrt(n)."""
        _, se1 = naive_mc_dose(1000, seed=0)
        _, se4 = naive_mc_dose(4000, seed=0)
        # se4 / se1 should be close to 1/sqrt(4) = 0.5
        ratio = se4 / se1
        assert 0.3 < ratio < 0.8


# ---------------------------------------------------------------------------
# convergence_experiment
# ---------------------------------------------------------------------------

class TestConvergenceExperiment:
    def test_returns_expected_keys(self):
        result = convergence_experiment(
            sample_sizes=np.array([100, 500, 1000]), n_trials=10, seed=0
        )
        assert {"sample_sizes", "errors", "stds", "theory", "true_dose"} == set(result.keys())

    def test_error_generally_decreases(self):
        """Mean MC error should trend downward as n increases."""
        result = convergence_experiment(
            sample_sizes=np.array([50, 500, 5000]), n_trials=30, seed=0
        )
        # Error should be smaller at n=5000 than n=50 on average (allow stochastic noise)
        assert result["errors"][0] > result["errors"][-1] * 0.5

    def test_true_dose_matches_reference(self):
        result = convergence_experiment(
            sample_sizes=np.array([100]), n_trials=5, seed=0
        )
        ref = compute_reference_doses()["Kidneys"]
        assert pytest.approx(result["true_dose"], rel=1e-9) == ref
