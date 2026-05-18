"""
api/app.py
==========
FastAPI REST service exposing the Monte Carlo dosimetry engine.

Endpoints
---------
GET  /health                 -- liveness probe
GET  /organs                 -- list available organs and reference doses
POST /simulate/population    -- population-level outcome comparison
POST /simulate/mcmc          -- Bayesian MCMC dose estimation for one patient
POST /simulate/variance      -- benchmark variance reduction methods
POST /dose/estimate          -- single-patient MIRD dose at given parameters

Run locally
-----------
    uvicorn api.app:app --reload --port 8000

Or from the project root:
    python -m uvicorn api.app:app --reload --port 8000

Docker
------
    docker build -t mc-dosimetry .
    docker run -p 8000:8000 mc-dosimetry
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
import numpy as np

from simulation import (
    ORGAN_PARAMS, STANDARD_ACTIVITY_GBQ, KIDNEY_TOLERANCE_GY,
    TUMOR_MIN_GY, INTER_PATIENT_CV, compute_reference_doses,
    absorbed_dose, naive_mc_dose,
)
from population_analysis import compare_dosing_strategies, TARGET_KIDNEY_GY
from variance_reduction import benchmark_vrf
from mcmc import simulate_spect_measurements, metropolis_hastings, posterior_summary


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Monte Carlo Dosimetry API",
    description=(
        "REST API for Lu-177 DOTATATE kidney dosimetry simulation. "
        "Implements MC population analysis, Bayesian MCMC parameter estimation, "
        "and variance reduction benchmarking."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PopulationRequest(BaseModel):
    n_patients: int = Field(default=2000, ge=100, le=50_000,
                            description="Number of simulated patients")
    cv: float = Field(default=INTER_PATIENT_CV, ge=0.01, le=1.0,
                      description="Inter-patient coefficient of variation")
    target_kidney_Gy: float = Field(default=TARGET_KIDNEY_GY, ge=5.0, le=23.0,
                                    description="Kidney dose target for individualized strategy (Gy)")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_workers: Optional[int] = Field(default=None, ge=1, le=32,
                                     description="Parallel workers (default: all CPU cores)")


class OutcomeStats(BaseModel):
    mean_kidney_dose_Gy: float
    mean_tumor_dose_Gy:  float
    p_optimal:           float
    p_kidney_toxicity:   float
    p_under_treatment:   float


class PopulationResponse(BaseModel):
    n_patients:         int
    wall_time_s:        float
    standard:           OutcomeStats
    individualized:     OutcomeStats
    toxicity_reduction: float   # fold-reduction in kidney toxicity risk


class MCMCRequest(BaseModel):
    n_samples: int   = Field(default=10_000, ge=1_000, le=50_000,
                             description="Total MCMC iterations")
    n_burnin:  int   = Field(default=2_000, ge=100,
                             description="Burn-in samples to discard")
    seed:      int   = Field(default=17, description="Random seed")
    spect_seed: int  = Field(default=17, description="Seed for SPECT data generation")


class MCMCResponse(BaseModel):
    acceptance_rate:  float
    A_slow_mean:      float
    A_slow_true:      float
    A_slow_ci95:      List[float]
    lam_slow_mean:    float
    lam_slow_true:    float
    lam_slow_ci95:    List[float]
    dose_mean_Gy:     float
    dose_true_Gy:     float
    dose_ci95_Gy:     List[float]
    p_exceed_limit:   float
    ess_A_slow:       float
    ess_lam_slow:     float


class VRFRequest(BaseModel):
    n: int           = Field(default=500, ge=50, le=5_000,
                             description="Sample size per trial")
    n_trials: int    = Field(default=100, ge=20, le=500,
                             description="Number of repeated trials for variance estimation")
    seed:     int    = Field(default=42)


class VRFResponse(BaseModel):
    vrfs: Dict[str, float]
    best_method: str
    best_vrf: float


class DoseRequest(BaseModel):
    organ: str       = Field(default="Kidneys", description="Organ name")
    A_fast:    Optional[float] = Field(default=None, description="Override A_fast parameter")
    lam_fast:  Optional[float] = Field(default=None, description="Override lam_fast parameter")
    A_slow:    Optional[float] = Field(default=None, description="Override A_slow parameter")
    lam_slow:  Optional[float] = Field(default=None, description="Override lam_slow parameter")
    activity_GBq: float        = Field(default=STANDARD_ACTIVITY_GBQ, ge=0.5, le=30.0)

    @field_validator("organ")
    @classmethod
    def organ_must_exist(cls, v):
        if v not in ORGAN_PARAMS:
            raise ValueError(f"Unknown organ '{v}'. Choose from: {list(ORGAN_PARAMS)}")
        return v


class DoseResponse(BaseModel):
    organ:        str
    absorbed_dose_Gy: float
    kidney_limit_Gy:  float
    exceeds_limit:    bool
    parameters_used: Dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/organs", tags=["Meta"])
def list_organs():
    """
    List all available organs with their population-mean parameters
    and reference absorbed doses at 7.4 GBq.
    """
    ref_doses = compute_reference_doses()
    result = {}
    for organ, p in ORGAN_PARAMS.items():
        result[organ] = {
            "A_fast":    p.A_fast,
            "lam_fast":  p.lam_fast,
            "A_slow":    p.A_slow,
            "lam_slow":  p.lam_slow,
            "mass_g":    p.mass_g,
            "S_value":   p.S_value,
            "reference_dose_Gy": round(ref_doses[organ], 4),
        }
    return {
        "organs": result,
        "kidney_tolerance_Gy": KIDNEY_TOLERANCE_GY,
        "tumor_min_Gy":        TUMOR_MIN_GY,
        "standard_activity_GBq": STANDARD_ACTIVITY_GBQ,
    }


@app.post("/simulate/population", response_model=PopulationResponse, tags=["Simulation"])
def simulate_population(req: PopulationRequest):
    """
    Run a population-level comparison of standard vs. individualized dosing.

    Simulates req.n_patients patients under both strategies and returns
    outcome statistics. Parallelized across available CPU cores.

    The key result: standard fixed dosing places ~65% of patients above
    the QUANTEC kidney safety limit; individualized MC dosing reduces that
    to ~4%.
    """
    result = compare_dosing_strategies(
        n                = req.n_patients,
        cv               = req.cv,
        target_kidney_Gy = req.target_kidney_Gy,
        seed             = req.seed,
        n_workers        = req.n_workers,
    )
    s, i = result.standard, result.individualized

    return PopulationResponse(
        n_patients         = result.n_patients,
        wall_time_s        = round(result.wall_time_s, 3),
        standard           = OutcomeStats(
            mean_kidney_dose_Gy = round(s.mean_kidney_dose, 2),
            mean_tumor_dose_Gy  = round(s.mean_tumor_dose, 2),
            p_optimal           = round(s.p_optimal, 4),
            p_kidney_toxicity   = round(s.p_kidney_toxicity, 4),
            p_under_treatment   = round(s.p_under_treatment, 4),
        ),
        individualized     = OutcomeStats(
            mean_kidney_dose_Gy = round(i.mean_kidney_dose, 2),
            mean_tumor_dose_Gy  = round(i.mean_tumor_dose, 2),
            p_optimal           = round(i.p_optimal, 4),
            p_kidney_toxicity   = round(i.p_kidney_toxicity, 4),
            p_under_treatment   = round(i.p_under_treatment, 4),
        ),
        toxicity_reduction = round(
            s.p_kidney_toxicity / max(i.p_kidney_toxicity, 1e-6), 2
        ),
    )


@app.post("/simulate/mcmc", response_model=MCMCResponse, tags=["Simulation"])
def simulate_mcmc(req: MCMCRequest):
    """
    Run Metropolis-Hastings MCMC for a single patient.

    Simulates 4 SPECT timepoint measurements for a patient, then infers
    the posterior over kidney biokinetic parameters (A_slow, lambda_slow)
    and propagates that uncertainty forward to absorbed dose.

    Returns posterior means, 95% credible intervals, effective sample size,
    and P(dose > QUANTEC limit).
    """
    if req.n_burnin >= req.n_samples:
        raise HTTPException(
            status_code=422,
            detail=f"n_burnin ({req.n_burnin}) must be less than n_samples ({req.n_samples})"
        )

    observed, _ = simulate_spect_measurements(seed=req.spect_seed)
    result       = metropolis_hastings(
        observed,
        n_samples = req.n_samples,
        n_burnin  = req.n_burnin,
        seed      = req.seed,
    )
    summary = posterior_summary(result)

    return MCMCResponse(
        acceptance_rate = round(result.acceptance_rate, 4),
        A_slow_mean     = round(summary["A_slow"]["mean"], 5),
        A_slow_true     = round(summary["A_slow"]["true"], 5),
        A_slow_ci95     = [round(x, 5) for x in summary["A_slow"]["ci95"]],
        lam_slow_mean   = round(summary["lam_slow"]["mean"], 6),
        lam_slow_true   = round(summary["lam_slow"]["true"], 6),
        lam_slow_ci95   = [round(x, 6) for x in summary["lam_slow"]["ci95"]],
        dose_mean_Gy    = round(summary["kidney_dose_Gy"]["mean"], 3),
        dose_true_Gy    = round(summary["kidney_dose_Gy"]["true"], 3),
        dose_ci95_Gy    = [round(x, 3) for x in summary["kidney_dose_Gy"]["ci95"]],
        p_exceed_limit  = round(summary["kidney_dose_Gy"]["p_exceed_limit"], 4),
        ess_A_slow      = round(summary["A_slow"]["ess"], 1),
        ess_lam_slow    = round(summary["lam_slow"]["ess"], 1),
    )


@app.post("/simulate/variance", response_model=VRFResponse, tags=["Simulation"])
def simulate_variance(req: VRFRequest):
    """
    Benchmark all variance reduction methods.

    Returns the variance reduction factor (VRF) for each method.
    VRF = Var(Naive MC) / Var(method) — a VRF of k means the method
    needs 1/k as many samples to achieve the same precision as naive MC.

    Methods: Naive MC, Antithetic Variates, Control Variates,
             Importance Sampling, Quasi-MC (Halton sequences).
    """
    vrfs = benchmark_vrf(n=req.n, n_trials=req.n_trials, seed=req.seed)
    best = max(vrfs, key=vrfs.get)
    return VRFResponse(
        vrfs        = {k: round(v, 2) for k, v in vrfs.items()},
        best_method = best,
        best_vrf    = round(vrfs[best], 2),
    )


@app.post("/dose/estimate", response_model=DoseResponse, tags=["Dosimetry"])
def estimate_dose(req: DoseRequest):
    """
    Compute MIRD absorbed dose for a single organ at given biokinetic parameters.

    If individual parameters (A_fast, lam_fast, etc.) are not provided,
    population-mean values from Sandstrom et al. (2013) are used.

    Returns absorbed dose in Gray and whether it exceeds the QUANTEC limit.
    """
    p = ORGAN_PARAMS[req.organ]

    A_fast   = req.A_fast   if req.A_fast   is not None else p.A_fast
    lam_fast = req.lam_fast if req.lam_fast is not None else p.lam_fast
    A_slow   = req.A_slow   if req.A_slow   is not None else p.A_slow
    lam_slow = req.lam_slow if req.lam_slow is not None else p.lam_slow

    for name, val in [("A_fast", A_fast), ("lam_fast", lam_fast),
                      ("A_slow", A_slow), ("lam_slow", lam_slow)]:
        if val <= 0:
            raise HTTPException(status_code=422,
                                detail=f"{name} must be positive, got {val}")

    dose = absorbed_dose(A_fast, lam_fast, A_slow, lam_slow,
                         p.S_value, req.activity_GBq)

    return DoseResponse(
        organ             = req.organ,
        absorbed_dose_Gy  = round(dose, 4),
        kidney_limit_Gy   = KIDNEY_TOLERANCE_GY,
        exceeds_limit     = bool(dose > KIDNEY_TOLERANCE_GY),
        parameters_used   = {
            "A_fast":      A_fast,
            "lam_fast":    lam_fast,
            "A_slow":      A_slow,
            "lam_slow":    lam_slow,
            "activity_GBq": req.activity_GBq,
        },
    )
