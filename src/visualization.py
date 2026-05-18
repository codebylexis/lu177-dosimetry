"""
visualization.py
================
All plotting functions for the Monte Carlo dosimetry project.
Each function saves its figure and returns the matplotlib Figure object.

Figures
-------
1. plot_baseline_convergence  -- MC error vs. sample size + sampling distribution
2. plot_variance_reduction    -- convergence comparison + VRF bar chart
3. plot_mcmc_results          -- trace plots, posteriors, dose distribution, ACF
4. plot_population_analysis   -- dose distributions + outcome scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from typing import Optional, Dict
import os

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

STYLE = {
    'figure.facecolor': '#0d1117', 'axes.facecolor':  '#161b22',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':     '#8b949e',
    'text.color':       '#e6edf3', 'grid.color':      '#21262d',
    'grid.linestyle':   '--',      'grid.alpha':       0.7,
    'font.family':      'monospace','axes.titlesize':  12,
    'axes.labelsize':   10,        'legend.fontsize':  9,
    'legend.framealpha': 0.3,     'legend.edgecolor': '#30363d',
}

PURPLE = '#a855f7'
GOLD   = '#f59e0b'
TEAL   = '#2dd4bf'
PINK   = '#f472b6'
GRAY   = '#6b7280'
GREEN  = '#4ade80'
RED    = '#ef4444'
BLUE   = '#60a5fa'

METHOD_COLORS = {
    'Naive MC':            GRAY,
    'Antithetic Variates': TEAL,
    'Control Variates':    PINK,
    'Importance Sampling': PURPLE,
    'Quasi-MC (Halton)':   GOLD,
}

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='#0d1117')
    return path


# ---------------------------------------------------------------------------
# Figure 1: Baseline convergence
# ---------------------------------------------------------------------------

def plot_baseline_convergence(
    convergence_data: Dict,
    sample_sizes_dist: list = None,
    n_dist_trials: int = 400,
    save: bool = True,
    filename: str = 'part1_baseline.png',
) -> plt.Figure:
    """
    Two-panel figure showing:
      Left : MC absolute error vs. sample size (log-log) vs. 1/sqrt(n) theory
      Right: Sampling distribution of the estimator at 3 sample sizes
    """
    from simulation import naive_mc_dose

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Part 1 -- Baseline Monte Carlo Dosimetry\n'
        'Kidney Absorbed Dose Estimation, Lu-177 DOTATATE',
        fontsize=12, color='#e6edf3', y=1.03, fontweight='bold',
    )

    ax = axes[0]
    ax.loglog(convergence_data['sample_sizes'], convergence_data['errors'],
              color=PURPLE, lw=2, label='MC absolute error')
    ax.loglog(convergence_data['sample_sizes'], convergence_data['theory'],
              color=GOLD, lw=1.5, ls='--', label='1/sqrt(n) theoretical rate')
    ax.set_xlabel('Simulated patients  n')
    ax.set_ylabel('|Dose error|  (Gy)')
    ax.set_title('Convergence of Dose Estimator')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if sample_sizes_dist is None:
        sample_sizes_dist = [50, 500, 5000]
    for n, c in zip(sample_sizes_dist, [PURPLE, TEAL, PINK]):
        ests = [naive_mc_dose(n)[0] for _ in range(n_dist_trials)]
        ax.hist(ests, bins=35, alpha=0.65, color=c, density=True,
                label=f'n = {n:,}', edgecolor='none')
    ax.axvline(convergence_data['true_dose'], color=GOLD, lw=2, ls='--',
               label=f"True mean = {convergence_data['true_dose']:.2f} Gy")
    ax.set_xlabel('Estimated mean kidney dose  (Gy)')
    ax.set_ylabel('Density')
    ax.set_title('Sampling Distribution of Estimator')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Variance reduction
# ---------------------------------------------------------------------------

def plot_variance_reduction(
    results: Dict,
    vrf_values: Dict,
    n_vrf: int = 500,
    sample_sizes: np.ndarray = None,
    save: bool = True,
    filename: str = 'part2_variance_reduction.png',
) -> plt.Figure:
    """
    Two-panel figure:
      Left : convergence comparison (log-log error vs. n) for all methods
      Right: variance reduction factor bar chart at n_vrf patients
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Part 2 -- Variance Reduction for Patient Dosimetry\n'
        'Kidney Absorbed Dose Estimation across Simulated Patients',
        fontsize=12, color='#e6edf3', y=1.03, fontweight='bold',
    )

    ax = axes[0]
    for name, data in results.items():
        c = METHOD_COLORS.get(name, BLUE)
        ax.loglog(sample_sizes, data['errors'], color=c, lw=2, label=name)
    ax.set_xlabel('Simulated patients  n')
    ax.set_ylabel('|Dose error|  (Gy)')
    ax.set_title('Convergence Comparison')
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1]
    names = list(vrf_values.keys())
    vals  = [vrf_values[n] for n in names]
    colors = [METHOD_COLORS.get(n, BLUE) for n in names]
    disp  = [min(v, 30) for v in vals]
    bars  = ax.bar(names, disp, color=colors, alpha=0.88,
                   edgecolor='#0d1117', linewidth=0.8)
    ax.axhline(1.0, color=GRAY, ls='--', lw=1.5, label='Baseline (1x)')
    ax.set_ylabel('Variance Reduction Factor (x)')
    ax.set_title(f'VRF at n = {n_vrf:,} patients')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=8)
    ax.legend(); ax.grid(True, axis='y', alpha=0.5)
    for bar, val, d in zip(bars, vals, disp):
        lbl = f'{val:.1f}x' if val < 30 else f'{int(val)}x'
        ax.text(bar.get_x() + bar.get_width() / 2, d + 0.2, lbl,
                ha='center', va='bottom', fontsize=8.5, color='#e6edf3')

    plt.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: MCMC results
# ---------------------------------------------------------------------------

def plot_mcmc_results(
    mcmc_result,
    scan_times: np.ndarray = None,
    organ: str = 'Kidneys',
    save: bool = True,
    filename: str = 'part3_mcmc.png',
) -> plt.Figure:
    """
    Six-panel figure:
      [0,0:2] Trace plot -- A_slow
      [1,0:2] Trace plot -- lam_slow
      [0,2]   Posterior -- A_slow
      [1,2]   Posterior -- lam_slow
      [2,0:2] Posterior predictive dose + QUANTEC limit + exceedance
      [2,2]   Autocorrelation function of A_slow chain
    """
    from simulation import ORGAN_PARAMS, KIDNEY_TOLERANCE_GY

    _apply_style()
    sp = mcmc_result.samples
    p  = ORGAN_PARAMS[organ]

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Part 3 -- Bayesian Biokinetic Parameter Estimation via MCMC\n'
        'Kidney Dosimetry Uncertainty from 4-Timepoint SPECT Data',
        fontsize=12, color='#e6edf3', y=1.01, fontweight='bold',
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # Trace -- A_slow
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(sp[:, 0], color=PURPLE, lw=0.5, alpha=0.85)
    ax.axhline(p.A_slow, color=GOLD, lw=1.8, ls='--',
               label=f'True A_slow = {p.A_slow}')
    ax.set_xlabel('Post-burnin iteration')
    ax.set_ylabel('A_slow')
    ax.set_title('Trace -- Slow-Component Amplitude  (A_slow)')
    ax.legend(); ax.grid(True)

    # Trace -- lam_slow
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(sp[:, 1], color=TEAL, lw=0.5, alpha=0.85)
    ax.axhline(p.lam_slow, color=GOLD, lw=1.8, ls='--',
               label=f'True lam_slow = {p.lam_slow:.4f} /h')
    ax.set_xlabel('Post-burnin iteration')
    ax.set_ylabel('lam_slow  (/h)')
    ax.set_title('Trace -- Slow-Component Clearance Rate  (lam_slow)')
    ax.legend(); ax.grid(True)

    # Posterior -- A_slow
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(sp[:, 0], bins=55, color=PURPLE, alpha=0.85,
            density=True, edgecolor='none')
    ax.axvline(p.A_slow, color=GOLD, lw=2, ls='--', label='True')
    ax.axvline(sp[:, 0].mean(), color=PINK, lw=2,
               label=f'Mean = {sp[:,0].mean():.3f}')
    ax.set_xlabel('A_slow')
    ax.set_title('Posterior -- A_slow')
    ax.legend(); ax.grid(True)

    # Posterior -- lam_slow
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(sp[:, 1], bins=55, color=TEAL, alpha=0.85,
            density=True, edgecolor='none')
    ax.axvline(p.lam_slow, color=GOLD, lw=2, ls='--', label='True')
    ax.axvline(sp[:, 1].mean(), color=PINK, lw=2,
               label=f'Mean = {sp[:,1].mean():.5f}')
    ax.set_xlabel('lam_slow (/h)')
    ax.set_title('Posterior -- lam_slow')
    ax.legend(); ax.grid(True)

    # Posterior predictive dose + exceedance
    ax = fig.add_subplot(gs[2, :2])
    pd_ = mcmc_result.posterior_doses
    ax.hist(pd_, bins=60, color=PURPLE, alpha=0.75,
            density=True, edgecolor='none', label='Posterior predictive dose')
    ax.axvline(mcmc_result.true_dose, color=GOLD, lw=2, ls='--',
               label=f'True = {mcmc_result.true_dose:.1f} Gy')
    ax.axvline(pd_.mean(), color=PINK, lw=2,
               label=f'Post. mean = {pd_.mean():.1f} Gy')
    ax.axvline(KIDNEY_TOLERANCE_GY, color=RED, lw=2.5, ls='-.',
               label=f'QUANTEC limit = {KIDNEY_TOLERANCE_GY} Gy')
    x_ex = np.linspace(
        KIDNEY_TOLERANCE_GY,
        min(pd_.max(), pd_.mean() + 4 * pd_.std()), 100
    )
    try:
        kde = gaussian_kde(pd_)
        ax.fill_between(x_ex, kde(x_ex), alpha=0.35, color=RED,
                        label=f'P(exceed) = {mcmc_result.p_exceed_limit:.1%}')
    except Exception:
        pass
    ax.set_xlabel('Predicted kidney dose  (Gy)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Predictive Dose + Safety Exceedance Probability')
    ax.legend(fontsize=8); ax.grid(True)

    # ACF
    ax = fig.add_subplot(gs[2, 2])
    max_lag = 70
    lags = range(max_lag)
    acf  = [1.0 if l == 0 else
            float(np.corrcoef(sp[:-l, 0], sp[l:, 0])[0, 1])
            for l in lags]
    ax.plot(lags, acf, color=PURPLE, lw=2)
    ax.fill_between(lags, acf, alpha=0.15, color=PURPLE)
    ax.axhline(0,    color=GRAY,  lw=1,   ls='--')
    ax.axhline(0.05, color=GREEN, lw=0.8, ls=':', alpha=0.7, label='rho = 0.05')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF -- A_slow chain')
    ax.legend(); ax.grid(True)

    if save:
        _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Population analysis
# ---------------------------------------------------------------------------

def plot_population_analysis(
    comparison,
    save: bool = True,
    filename: str = 'part4_population.png',
) -> plt.Figure:
    """
    Four-panel figure:
      [0,0] Kidney dose distributions (standard vs. individualized)
      [0,1] Tumor dose distributions (standard vs. individualized)
      [1,0] Scatter: kidney vs. tumor dose -- standard dosing
      [1,1] Scatter: kidney vs. tumor dose -- individualized dosing
    """
    from simulation import KIDNEY_TOLERANCE_GY, TUMOR_MIN_GY

    _apply_style()
    s = comparison.standard
    i = comparison.individualized

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Part 4 -- Population Dosimetry and Clinical Safety Analysis\n'
        'Standard vs. MC-Informed Individualized Lu-177 DOTATATE Dosing',
        fontsize=12, color='#e6edf3', y=1.02, fontweight='bold',
    )

    # Kidney dose distributions
    ax = axes[0, 0]
    ax.hist(s.kidney_doses, bins=80, alpha=0.6, color=PURPLE, density=True,
            edgecolor='none',
            label=f'Standard 7.4 GBq  (P(toxicity) = {s.p_kidney_toxicity:.1%})')
    ax.hist(i.kidney_doses, bins=80, alpha=0.6, color=TEAL, density=True,
            edgecolor='none',
            label=f'Individualized  (P(toxicity) = {i.p_kidney_toxicity:.1%})')
    ax.axvline(KIDNEY_TOLERANCE_GY, color=RED, lw=2.5, ls='-.',
               label=f'QUANTEC limit = {KIDNEY_TOLERANCE_GY} Gy')
    ax.set_xlabel('Kidney absorbed dose  (Gy)')
    ax.set_ylabel('Density')
    ax.set_title('Kidney Dose Distribution -- Population Level')
    ax.legend(fontsize=8); ax.grid(True)

    # Tumor dose distributions
    ax = axes[0, 1]
    ax.hist(s.tumor_doses, bins=80, alpha=0.6, color=PURPLE, density=True,
            edgecolor='none',
            label=f'Standard  (P(under-tx) = {s.p_under_treatment:.1%})')
    ax.hist(i.tumor_doses, bins=80, alpha=0.6, color=TEAL, density=True,
            edgecolor='none',
            label=f'Individualized  (P(under-tx) = {i.p_under_treatment:.1%})')
    ax.axvline(TUMOR_MIN_GY, color=GOLD, lw=2.5, ls='-.',
               label=f'Min tumoricidal = {TUMOR_MIN_GY} Gy')
    ax.set_xlabel('Tumor absorbed dose  (Gy)')
    ax.set_ylabel('Density')
    ax.set_title('Tumor Dose Distribution -- Population Level')
    ax.legend(fontsize=8); ax.grid(True)

    # Scatter plots
    for ax, outcomes, title in [
        (axes[1, 0], s, 'Standard Dosing -- Kidney vs. Tumor Dose'),
        (axes[1, 1], i, 'Individualized Dosing -- Kidney vs. Tumor Dose'),
    ]:
        step = max(1, outcomes.n_patients // 2000)
        ax.scatter(
            outcomes.kidney_doses[outcomes.optimal][::step],
            outcomes.tumor_doses[outcomes.optimal][::step],
            alpha=0.12, s=4, color=GREEN,
            label=f'Optimal ({outcomes.p_optimal:.1%})',
        )
        ax.scatter(
            outcomes.kidney_doses[outcomes.kidney_toxicity][::step],
            outcomes.tumor_doses[outcomes.kidney_toxicity][::step],
            alpha=0.18, s=4, color=RED,
            label=f'Kidney toxicity ({outcomes.p_kidney_toxicity:.1%})',
        )
        ax.scatter(
            outcomes.kidney_doses[outcomes.under_treatment][::step],
            outcomes.tumor_doses[outcomes.under_treatment][::step],
            alpha=0.18, s=4, color=GOLD,
            label=f'Under-treatment ({outcomes.p_under_treatment:.1%})',
        )
        ax.axvline(KIDNEY_TOLERANCE_GY, color=RED,  lw=1.5, ls='-.', alpha=0.8)
        ax.axhline(TUMOR_MIN_GY,        color=GOLD, lw=1.5, ls='-.', alpha=0.8)
        ax.set_xlabel('Kidney dose  (Gy)')
        ax.set_ylabel('Tumor dose  (Gy)')
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Convenience: regenerate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(seed: int = 42) -> None:
    """Regenerate all four figures from scratch and save to figures/."""
    from simulation import convergence_experiment
    from variance_reduction import ALL_METHODS, benchmark_vrf
    from mcmc import simulate_spect_measurements, metropolis_hastings
    from population_analysis import compare_dosing_strategies

    np.random.seed(seed)
    sample_sizes = np.logspace(1, 4.2, 30).astype(int)

    print("Generating Figure 1: baseline convergence...")
    conv = convergence_experiment(sample_sizes=np.logspace(1, 4.5, 35).astype(int))
    plot_baseline_convergence(conv)

    print("Generating Figure 2: variance reduction...")
    results = {name: {'errors': [], 'stds': []} for name in ALL_METHODS}
    true_dose = conv['true_dose']
    for sz in sample_sizes:
        for name, fn in ALL_METHODS.items():
            trials = [fn(sz)[0] for _ in range(40)]
            results[name]['errors'].append(abs(np.mean(trials) - true_dose))
            results[name]['stds'].append(np.std(trials))
    for name in results:
        results[name]['errors'] = np.array(results[name]['errors'])
    vrfs = benchmark_vrf(n=500, seed=seed)
    plot_variance_reduction(results, vrfs, n_vrf=500, sample_sizes=sample_sizes)

    print("Generating Figure 3: MCMC...")
    observed, _ = simulate_spect_measurements(seed=17)
    mcmc_result  = metropolis_hastings(observed, seed=17)
    plot_mcmc_results(mcmc_result)

    print("Generating Figure 4: population analysis...")
    comparison = compare_dosing_strategies(n=8_000, seed=seed)
    plot_population_analysis(comparison)

    print("All figures saved to figures/")


if __name__ == "__main__":
    generate_all_figures()
