"""
tests/test_population_analysis.py
==================================
Unit tests for population simulation and outcome classification.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from simulation import (
    STANDARD_ACTIVITY_GBQ, KIDNEY_TOLERANCE_GY, TUMOR_MIN_GY, INTER_PATIENT_CV
)
from population_analysis import (
    simulate_population, simulate_individualized, classify_outcomes,
    compare_dosing_strategies, PopulationOutcomes, ComparisonResult,
    ACTIVITY_MIN_GBQ, ACTIVITY_MAX_GBQ, TARGET_KIDNEY_GY,
    _make_chunks, _simulate_standard_chunk, _simulate_individualized_chunk,
)


# ---------------------------------------------------------------------------
# _make_chunks
# ---------------------------------------------------------------------------

class TestMakeChunks:
    def test_total_size_correct(self):
        chunks = _make_chunks(100, 4, 42)
        total = sum(size for size, _ in chunks)
        assert total == 100

    def test_n_chunks(self):
        chunks = _make_chunks(100, 7, 0)
        assert len(chunks) == 7

    def test_seeds_unique(self):
        chunks = _make_chunks(100, 5, 0)
        seeds  = [s for _, s in chunks]
        assert len(set(seeds)) == len(seeds)

    def test_remainder_distributed(self):
        """With n=10, n_workers=3: chunks of 4, 3, 3."""
        chunks = _make_chunks(10, 3, 0)
        sizes  = [s for s, _ in chunks]
        assert sum(sizes) == 10
        assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------------------
# simulate_population
# ---------------------------------------------------------------------------

class TestSimulatePopulation:
    def test_output_shape(self):
        doses = simulate_population(200, seed=0, n_workers=2)
        for organ, arr in doses.items():
            assert arr.shape == (200,), f"Wrong shape for {organ}"

    def test_all_positive(self):
        doses = simulate_population(200, seed=0, n_workers=2)
        for organ, arr in doses.items():
            assert np.all(arr > 0), f"Non-positive dose for {organ}"

    def test_reproducible(self):
        d1 = simulate_population(300, seed=7, n_workers=2)
        d2 = simulate_population(300, seed=7, n_workers=2)
        for organ in d1:
            np.testing.assert_array_equal(d1[organ], d2[organ])

    def test_single_worker_matches_multiworker_mean(self):
        """Single-worker and multi-worker should produce similar population means."""
        n = 5_000
        d1 = simulate_population(n, seed=0, n_workers=1)
        d2 = simulate_population(n, seed=0, n_workers=2)
        # Means should match within 5% (different seeds per chunk, so not identical)
        for organ in d1:
            rel = abs(d1[organ].mean() - d2[organ].mean()) / d1[organ].mean()
            assert rel < 0.05, f"Mean differs too much for {organ}"

    def test_higher_activity_increases_dose(self):
        d_std = simulate_population(2000, STANDARD_ACTIVITY_GBQ, seed=0, n_workers=2)
        d_dbl = simulate_population(2000, STANDARD_ACTIVITY_GBQ * 2, seed=0, n_workers=2)
        for organ in d_std:
            assert d_dbl[organ].mean() > d_std[organ].mean()


# ---------------------------------------------------------------------------
# simulate_individualized
# ---------------------------------------------------------------------------

class TestSimulateIndividualized:
    def test_output_shapes(self):
        k, t, a = simulate_individualized(200, seed=0, n_workers=2)
        assert k.shape == t.shape == a.shape == (200,)

    def test_activities_within_bounds(self):
        _, _, a = simulate_individualized(1000, seed=0, n_workers=2)
        assert np.all(a >= ACTIVITY_MIN_GBQ)
        assert np.all(a <= ACTIVITY_MAX_GBQ)

    def test_kidney_doses_near_target(self):
        """
        Unconstrained patients should have kidney dose close to TARGET_KIDNEY_GY.
        Those at the activity bounds will deviate, but the bulk should be near target.
        """
        k, _, a = simulate_individualized(5000, seed=0, n_workers=2)
        unconstrained = (a > ACTIVITY_MIN_GBQ + 0.1) & (a < ACTIVITY_MAX_GBQ - 0.1)
        if unconstrained.sum() > 100:
            assert pytest.approx(k[unconstrained].mean(), rel=0.05) == TARGET_KIDNEY_GY

    def test_all_doses_positive(self):
        k, t, _ = simulate_individualized(300, seed=0, n_workers=2)
        assert np.all(k > 0) and np.all(t > 0)


# ---------------------------------------------------------------------------
# classify_outcomes
# ---------------------------------------------------------------------------

class TestClassifyOutcomes:
    def test_mutually_exclusive(self):
        """Every patient falls into exactly one category."""
        rng = np.random.default_rng(0)
        k = rng.uniform(10, 40, 1000)
        t = rng.uniform(10, 80, 1000)
        out = classify_outcomes(k, t)
        total = out.optimal + out.kidney_toxicity + out.under_treatment
        np.testing.assert_array_equal(total, np.ones(1000, dtype=bool))

    def test_exhaustive(self):
        """Rates must sum to 1."""
        rng = np.random.default_rng(0)
        k = rng.uniform(10, 40, 5000)
        t = rng.uniform(10, 80, 5000)
        out = classify_outcomes(k, t)
        assert pytest.approx(
            out.p_optimal + out.p_kidney_toxicity + out.p_under_treatment, abs=1e-9
        ) == 1.0

    def test_all_safe_all_effective(self):
        k = np.full(100, KIDNEY_TOLERANCE_GY - 1)
        t = np.full(100, TUMOR_MIN_GY + 1)
        out = classify_outcomes(k, t)
        assert out.p_optimal == 1.0
        assert out.p_kidney_toxicity == 0.0
        assert out.p_under_treatment == 0.0

    def test_all_toxic(self):
        k = np.full(100, KIDNEY_TOLERANCE_GY + 5)
        t = np.full(100, TUMOR_MIN_GY + 1)
        out = classify_outcomes(k, t)
        assert out.p_kidney_toxicity == 1.0
        assert out.p_optimal == 0.0

    def test_all_under_treatment(self):
        k = np.full(100, KIDNEY_TOLERANCE_GY - 1)
        t = np.full(100, TUMOR_MIN_GY - 1)
        out = classify_outcomes(k, t)
        assert out.p_under_treatment == 1.0
        assert out.p_optimal == 0.0

    def test_boundary_kidney(self):
        """Dose exactly at limit should be classified as safe."""
        k = np.array([KIDNEY_TOLERANCE_GY])
        t = np.array([TUMOR_MIN_GY + 1])
        out = classify_outcomes(k, t)
        assert out.optimal[0] == True

    def test_n_patients_set_correctly(self):
        k = np.zeros(77)
        t = np.zeros(77)
        out = classify_outcomes(k, t)
        assert out.n_patients == 77


# ---------------------------------------------------------------------------
# compare_dosing_strategies
# ---------------------------------------------------------------------------

class TestCompareDosingStrategies:
    @pytest.fixture(scope="class")
    def comparison(self):
        return compare_dosing_strategies(n=2_000, seed=42, n_workers=2)

    def test_returns_comparison_result(self, comparison):
        assert isinstance(comparison, ComparisonResult)

    def test_n_patients_correct(self, comparison):
        assert comparison.n_patients == 2_000

    def test_wall_time_positive(self, comparison):
        assert comparison.wall_time_s > 0

    def test_individualized_has_lower_toxicity(self, comparison):
        """Core clinical result: individualized dosing reduces kidney toxicity."""
        assert (comparison.individualized.p_kidney_toxicity
                < comparison.standard.p_kidney_toxicity)

    def test_individualized_has_higher_optimal(self, comparison):
        assert (comparison.individualized.p_optimal
                > comparison.standard.p_optimal)

    def test_standard_kidney_dose_above_quantec(self, comparison):
        """Standard dosing drives mean kidney dose above QUANTEC limit."""
        assert comparison.standard.mean_kidney_dose > KIDNEY_TOLERANCE_GY

    def test_individualized_kidney_dose_below_quantec(self, comparison):
        assert comparison.individualized.mean_kidney_dose < KIDNEY_TOLERANCE_GY

    def test_rates_sum_to_one(self, comparison):
        for outcome in [comparison.standard, comparison.individualized]:
            total = (outcome.p_optimal + outcome.p_kidney_toxicity
                     + outcome.p_under_treatment)
            assert pytest.approx(total, abs=1e-9) == 1.0
