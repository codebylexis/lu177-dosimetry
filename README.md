# Monte Carlo Dosimetry for Targeted Radionuclide Therapy

Monte Carlo simulation framework for Lu-177 DOTATATE dosimetry, implementing
variance reduction techniques, Bayesian biokinetic parameter estimation via MCMC,
population-level safety analysis, a REST API, and a full pytest test suite.

**Short version:** standard fixed dosing puts 65.8% of patients above the kidney
safety limit. MC-informed individualized dosing brings that down to 4%.

---

## Background

Lu-177 DOTATATE (Lutathera, FDA-approved 2018) is a targeted radionuclide therapy
for neuroendocrine tumors. A radioactive lutetium atom is attached to a molecule
that binds selectively to tumor cells, delivering radiation from the inside. The
kidneys are the dose-limiting organ. Too much radiation there causes permanent damage.

The problem: the standard fixed dose of 7.4 GBq ignores ~30% inter-patient
variability in biokinetic clearance, causing absorbed kidney dose to vary
dramatically between patients. This project models that variability and quantifies
the clinical impact.

---

## Project Structure

```
lu177-dosimetry/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── pytest.ini
│
├── src/
│   ├── simulation.py          # biokinetic model + naive MC estimator
│   ├── variance_reduction.py  # antithetic, control variates, IS, QMC
│   ├── mcmc.py                # Metropolis-Hastings MCMC + posterior analysis
│   ├── population_analysis.py # 8,000-patient safety simulation (parallelized)
│   └── visualization.py       # all plotting functions
│
├── api/
│   └── app.py                 # FastAPI REST service (4 endpoints)
│
├── tests/
│   ├── conftest.py
│   ├── test_simulation.py          # 31 tests — physics model
│   ├── test_variance_reduction.py  # 23 tests — VR estimators
│   ├── test_mcmc.py                # 29 tests — MCMC sampler
│   ├── test_population_analysis.py # 27 tests — population sim
│   └── test_api.py                 # 14 tests — REST endpoints
│
├── benchmarks/
│   └── benchmark_performance.py   # speedup, VRF, ESS/s, throughput
│
├── figures/                   # generated figures (PNG)
│   ├── part1_baseline.png
│   ├── part2_variance_reduction.png
│   ├── part3_mcmc.png
│   └── part4_population.png
│
├── paper/
│   └── writeup.docx            # full project writeup
│
└── examples/
    └── quickstart.ipynb       # end-to-end walkthrough notebook
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite (142 tests)
pytest

# Run each module standalone (prints key results)
cd src
python simulation.py
python variance_reduction.py
python mcmc.py
python population_analysis.py

# Start the REST API (http://localhost:8000/docs for Swagger UI)
uvicorn api.app:app --reload --port 8000

# Run performance benchmarks
python benchmarks/benchmark_performance.py
```

Or open `examples/quickstart.ipynb` for an end-to-end walkthrough.

---

## Docker

```bash
# Build
docker build -t mc-dosimetry .

# Run (exposes API at localhost:8000)
docker run -p 8000:8000 mc-dosimetry

# Swagger UI available at http://localhost:8000/docs
```

---

## REST API

The FastAPI service exposes four endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/organs` | Reference doses and organ parameters |
| POST | `/dose/estimate` | MIRD dose for a single patient |
| POST | `/simulate/population` | Population outcome comparison (parallelized) |
| POST | `/simulate/mcmc` | Bayesian MCMC for one patient |
| POST | `/simulate/variance` | Benchmark variance reduction methods |

Interactive docs: `http://localhost:8000/docs`

---

## Methods

### Physical Model

Organ time-activity curves follow a bi-exponential model:

```
A(t) = A_fast * exp(-lambda_fast * t) + A_slow * exp(-lambda_slow * t)
```

Absorbed dose is computed via the MIRD formula:

```
D = S * cumulated_activity * administered_activity
```

where `S` is the organ-specific dose coefficient (Gy per MBq·h) from ICRP Publication 53,
and `cumulated_activity = A_fast/lambda_fast + A_slow/lambda_slow`.

### Variance Reduction

| Method | VRF (n=500) | Key idea |
|---|---|---|
| Naive MC | 1.0x | Baseline |
| Antithetic Variates | ~10x | Pair each draw z with -z |
| Control Variates | ~1.7x | Fast-component CA as correlated control |
| Importance Sampling | ~5x | Oversample high-dose tail |
| Quasi-MC (Halton) | ~167x | Low-discrepancy sequences (bases 2, 3, 5) |

### MCMC

Metropolis-Hastings with symmetric Gaussian random-walk proposal, inferring
slow-component parameters (A_slow, lambda_slow) from 4 simulated SPECT timepoints.
Acceptance rate ~43%, burn-in 3,000 / 20,000 iterations.

### Population Analysis

8,000 simulated patients with log-normal biokinetic variability (CV = 30%).
Three outcome categories: optimal (kidney safe + tumor effective), kidney
toxicity risk, and under-treatment.

**Parallelized** with `multiprocessing.Pool`: the cohort is partitioned into n_workers chunks, each with an independent RNG seed, and simulated in parallel.

---

## Key Results

| Metric | Standard 7.4 GBq | Individualized MC |
|---|---|---|
| P(kidney toxicity) | 65.8% | 4.0% |
| P(under-treatment) | 1.8% | 15.8% |
| P(optimal outcome) | 32.5% | 80.2% |
| Mean kidney dose | 27.6 Gy | 20.3 Gy |

---

## Testing

```bash
# Run full suite with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific module
pytest tests/test_simulation.py -v

# Skip slow tests
pytest -m "not slow"
```

142 tests across 5 modules covering physics functions, all 4 variance reduction
estimators, MCMC sampler diagnostics, population simulation correctness, and all
REST API endpoints.

---

## Connections to ML

The variance reduction techniques here are mathematically identical to
methods used throughout modern ML:

- **Control variates** = REINFORCE baseline in policy gradient RL
- **Importance sampling** = ELBO estimation in variational autoencoders
- **Antithetic variates** = related to the reparameterization trick in VAEs
- **Metropolis-Hastings** = ancestor of HMC (used in Stan, NumPyro)
- **Quasi-MC** = used in neural architecture search, GP approximations

---

## References

- Sandstrom et al. (2013). Individualized dosimetry of kidney and bone marrow. *J Nucl Med*, 54(1), 33-41.
- Ljungberg et al. (2016). MIRD Pamphlet No. 26. *J Nucl Med*, 57(1), 151-162.
- Dawson et al. (2010). Radiation-associated kidney injury. *Int J Radiat Oncol Biol Phys*, 76(3S), S108-S115.
- ICRP Publication 53 (1987). S-values for Lu-177.
- Niederreiter (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
- Robert & Casella (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.

---

## Author

Lexi Sierfeld, December 2025
