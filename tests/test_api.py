"""
tests/test_api.py
=================
Integration tests for the FastAPI REST endpoints.
Uses httpx.AsyncClient with the ASGI transport (no running server needed).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Import after patching sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.app import app


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /organs
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_organs_returns_all(client):
    resp = await client.get("/organs")
    assert resp.status_code == 200
    data = resp.json()
    assert "Kidneys" in data["organs"]
    assert "Tumor" in data["organs"]
    assert data["kidney_tolerance_Gy"] == 23.0


@pytest.mark.anyio
async def test_organs_have_reference_doses(client):
    resp = await client.get("/organs")
    for organ, info in resp.json()["organs"].items():
        assert info["reference_dose_Gy"] > 0


# ---------------------------------------------------------------------------
# /dose/estimate
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_dose_estimate_default(client):
    resp = await client.post("/dose/estimate", json={"organ": "Kidneys"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["absorbed_dose_Gy"] > 0
    assert data["organ"] == "Kidneys"
    assert isinstance(data["exceeds_limit"], bool)


@pytest.mark.anyio
async def test_dose_estimate_custom_params(client):
    resp = await client.post("/dose/estimate", json={
        "organ": "Kidneys",
        "A_slow": 0.10,
        "lam_slow": 0.003,
        "activity_GBq": 7.4,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["parameters_used"]["A_slow"] == 0.10


@pytest.mark.anyio
async def test_dose_estimate_invalid_organ(client):
    resp = await client.post("/dose/estimate", json={"organ": "Brain"})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_dose_estimate_all_organs(client):
    for organ in ["Kidneys", "Tumor", "Liver", "Spleen", "Red Marrow"]:
        resp = await client.post("/dose/estimate", json={"organ": organ})
        assert resp.status_code == 200
        assert resp.json()["absorbed_dose_Gy"] > 0


# ---------------------------------------------------------------------------
# /simulate/population
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_population_small(client):
    resp = await client.post("/simulate/population", json={
        "n_patients": 500, "seed": 42, "n_workers": 2
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_patients"] == 500
    assert data["standard"]["p_kidney_toxicity"] > data["individualized"]["p_kidney_toxicity"]
    assert data["toxicity_reduction"] > 1.0


@pytest.mark.anyio
async def test_population_validates_n_range(client):
    resp = await client.post("/simulate/population", json={"n_patients": 50})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_population_wall_time_present(client):
    resp = await client.post("/simulate/population", json={
        "n_patients": 200, "seed": 1, "n_workers": 1
    })
    assert resp.json()["wall_time_s"] > 0


# ---------------------------------------------------------------------------
# /simulate/mcmc
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_mcmc_basic(client):
    resp = await client.post("/simulate/mcmc", json={
        "n_samples": 3_000, "n_burnin": 500, "seed": 17
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 0.15 < data["acceptance_rate"] < 0.70
    assert data["dose_mean_Gy"] > 0
    assert data["dose_ci95_Gy"][0] < data["dose_ci95_Gy"][1]


@pytest.mark.anyio
async def test_mcmc_burnin_gte_samples_rejected(client):
    resp = await client.post("/simulate/mcmc", json={
        "n_samples": 1_000, "n_burnin": 1_000
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /simulate/variance
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_variance_benchmark(client):
    resp = await client.post("/simulate/variance", json={
        "n": 300, "n_trials": 30, "seed": 42
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "Naive MC" in data["vrfs"]
    assert "Quasi-MC (Halton)" in data["vrfs"]
    assert data["best_vrf"] > 1.0


@pytest.mark.anyio
async def test_variance_naive_mc_near_one(client):
    resp = await client.post("/simulate/variance", json={
        "n": 300, "n_trials": 50, "seed": 0
    })
    naive_vrf = resp.json()["vrfs"]["Naive MC"]
    assert 0.5 < naive_vrf < 2.0
