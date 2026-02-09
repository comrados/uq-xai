"""Run stratified validation for SHAP stability.

Compares SHAP stability between low- and high-epistemic samples for the RF UQ
model at a fixed Gaussian noise level, with optional epistemic bins.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, ttest_ind
import pickle
from pathlib import Path
from tqdm import tqdm

from data.cache import Cache
from data.datasets import (
    WineQualityDataset,
    DryBeanDataset,
    RiceDataset,
)
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from data.perturbations import PerturbationGenerator
from config.settings import GLOBAL_SEED

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

SIGMA = [0.01, 0.05, 0.1]  # Noise levels from RQ4 results
K_NOISE_SEEDS = 10  # Number of noise realizations

PERCENTILE_LOW = 10   # Bottom 10% epistemic
PERCENTILE_HIGH = 90  # Top 10% epistemic
N_SAMPLES_PER_GROUP = 50  # Samples per group

EPI_BINS = 3  # Number of epistemic bins for trend check
N_SAMPLES_PER_BIN = 50  # Samples per bin (cap)
EPI_BINNING = "quantile"  # "quantile" or "uniform"
IMAGE_FILE_EXT = "pdf"

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _stat_corr(stat_result):
    """Extract correlation from SciPy stats result."""
    if hasattr(stat_result, "correlation"):
        return stat_result.correlation
    return stat_result[0]

def _format_ratio(numer, denom):
    if denom == 0:
        return "n/a"
    return f"{numer / denom:.2f}x"

def _print_section(title):
    print(f"\n===== {title} =====")

def _interpret_kendall(delta, p_val, sigma):
    if delta > 0.15 and p_val < 0.05:
        verdict = "PASS VALIDATION SUCCESSFUL"
        interpretation = (
            f"Low-epistemic samples exhibit significantly higher stability than "
            f"high-epistemic samples (delta_Kendall={delta:+.3f}, p={p_val:.4f}). "
            f"Epistemic threshold effectively separates stable from unstable explanations."
        )
    elif delta > 0.0 and p_val < 0.05:
        verdict = "WARN WEAK SEPARATION"
        interpretation = (
            f"Low-epistemic samples show statistically significant but modest stability "
            f"improvement (delta_Kendall={delta:+.3f}, p={p_val:.4f}). "
            f"Epistemic provides some discriminative power but effect size is small."
        )
    else:
        verdict = "FAIL VALIDATION FAILED"
        interpretation = (
            f"No significant difference in stability between low and high epistemic groups "
            f"(delta_Kendall={delta:+.3f}, p={p_val:.4f}). "
            f"Epistemic uncertainty does not predict explanation stability at sigma={sigma}."
        )
    return verdict, interpretation

def _append_cached_results(results, dataset_name, dataset_slug, sigma, table_summaries, plot_payloads, bin_plot_payloads):
    metric = 'kendall'
    low_block = results.get('stability_low', {}).get(metric, {})
    high_block = results.get('stability_high', {}).get(metric, {})
    low_vals = np.array(low_block.get('values', []), dtype=float)
    high_vals = np.array(high_block.get('values', []), dtype=float)

    if low_vals.size and high_vals.size:
        plot_payloads.append({
            'dataset': dataset_name,
            'slug': dataset_slug,
            'low_vals': low_vals,
            'high_vals': high_vals,
        })

    low_mean = low_block.get('mean')
    low_std = low_block.get('std')
    high_mean = high_block.get('mean')
    high_std = high_block.get('std')
    if low_mean is None or low_std is None:
        if low_vals.size:
            low_mean = float(low_vals.mean())
            low_std = float(low_vals.std())
    if high_mean is None or high_std is None:
        if high_vals.size:
            high_mean = float(high_vals.mean())
            high_std = float(high_vals.std())
    if low_mean is None:
        low_mean = float("nan")
    if low_std is None:
        low_std = float("nan")
    if high_mean is None:
        high_mean = float("nan")
    if high_std is None:
        high_std = float("nan")

    delta = results.get('delta', {}).get(metric, {}).get('value')
    p_val = results.get('delta', {}).get(metric, {}).get('p_value')
    if delta is None:
        delta = float(low_mean - high_mean)
    if p_val is None:
        p_val = float("nan")

    k_noise_seeds = results.get('k_noise_seeds', K_NOISE_SEEDS)
    table_lines = [
        f"Stability (mean ± std across {k_noise_seeds} seeds):",
        f"{'Metric':<12} {'Low Epi':<20} {'High Epi':<20} {'Delta':<10} {'p-value'}",
        "-" * 70,
        (
            f"{metric:<12} {low_mean:.3f} ± {low_std:.3f}     "
            f"{high_mean:.3f} ± {high_std:.3f}     "
            f"{delta:+.3f}    {p_val:.4f}"
        ),
    ]

    summary_entry = {
        'dataset': dataset_name,
        'lines': table_lines,
        'bins': None,
    }
    table_summaries.append(summary_entry)

    bin_results = results.get('bins')
    if bin_results:
        epi_bins = results.get('epi_bins', len(bin_results.get('bins', [])))
        epi_binning = results.get('epi_binning', bin_results.get('binning', 'n/a'))
        n_samples_per_bin = results.get('n_samples_per_bin', N_SAMPLES_PER_BIN)

        bin_table_lines = [
            f"Epistemic bins ({epi_bins}, {epi_binning}) with up to {n_samples_per_bin} samples each:",
            f"{'Bin':<18} {'N':>4} {'Kendall':>12}",
            "-" * 50,
        ]
        bins = bin_results.get('bins', [])
        edges = bin_results.get('edges', [])
        n_bins = len(bins) if bins else max(len(edges) - 1, 0)
        bin_labels = []
        bin_values = []
        bin_counts = []
        has_values = True

        for bin_idx in range(n_bins):
            if bins:
                bin_info = bins[bin_idx]
                range_vals = bin_info.get('range', [None, None])
                low_edge = range_vals[0] if len(range_vals) > 0 else None
                high_edge = range_vals[1] if len(range_vals) > 1 else None
                n_bin = int(bin_info.get('n', 0))
                kendall_metrics = bin_info.get('metrics', {}).get('kendall', {})
                mean_val = kendall_metrics.get('mean')
                std_val = kendall_metrics.get('std')
                values = bin_info.get('values')
            else:
                low_edge = edges[bin_idx] if bin_idx < len(edges) else None
                high_edge = edges[bin_idx + 1] if bin_idx + 1 < len(edges) else None
                n_bin = 0
                mean_val = None
                std_val = None
                values = None

            right_bracket = "]" if bin_idx == n_bins - 1 else ")"
            if low_edge is None or high_edge is None:
                bin_label = f"bin {bin_idx + 1}"
            else:
                bin_label = f"[{float(low_edge):.4f}, {float(high_edge):.4f}{right_bracket}"

            if mean_val is None or std_val is None:
                if values is not None and len(values) > 0:
                    vals_arr = np.array(values, dtype=float)
                    mean_val = float(vals_arr.mean())
                    std_val = float(vals_arr.std())

            if n_bin == 0 or mean_val is None or std_val is None:
                bin_table_lines.append(
                    f"{bin_label:<18} {n_bin:>4} {'n/a':>12}"
                )
            else:
                bin_table_lines.append(
                    f"{bin_label:<18} {n_bin:>4} {mean_val:.3f}±{std_val:.3f}"
                )

            bin_labels.append(bin_label)
            bin_counts.append(n_bin)
            if values is None:
                has_values = False
            else:
                bin_values.append(np.array(values, dtype=float))

        summary_entry['bins'] = bin_table_lines
        if has_values:
            bin_plot_payloads.append({
                'dataset': dataset_name,
                'labels': bin_labels,
                'values': bin_values,
                'counts': bin_counts,
            })

    return delta, p_val

def _cache_has_bin_values(results):
    bin_results = results.get('bins')
    if not bin_results:
        return True
    for bin_info in bin_results.get('bins', []):
        n_bin = bin_info.get('n', 0)
        values = bin_info.get('values')
        if n_bin > 0 and (values is None or len(values) == 0):
            return False
    return True

# LOAD DATA MODEL

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    RiceDataset(),
]

table_summaries = {sigma: [] for sigma in SIGMA}
plot_payloads = {sigma: [] for sigma in SIGMA}
bin_plot_payloads = {sigma: [] for sigma in SIGMA}

def run_dataset(ds, cache, registry, splitter, table_summaries, plot_payloads, bin_plot_payloads, sigma):
    info = cache.load_or_create(ds.cache_key, ds.load)
    dataset_name = info.name
    dataset_slug = dataset_name.replace(" ", "_")
    dataset_key = f"{dataset_name}_{ds.uci_id}"
    output_file = RESULTS_DIR / f'stratified_validation_sigma{sigma}_{dataset_slug}.pkl'
    summary_file = RESULTS_DIR / f'stratified_validation_sigma{sigma}_{dataset_slug}.txt'
    
    _print_section(f"STRATIFIED VALIDATION: {dataset_name}")

    if output_file.exists():
        print(f"OK Cached results found, loading: {output_file}")
        with open(output_file, 'rb') as f:
            results = pickle.load(f)
        if EPI_BINS >= 3 and N_SAMPLES_PER_BIN > 0 and not _cache_has_bin_values(results):
            print("WARN Cached results missing per-bin values; recomputing.")
        else:
            delta, p_val = _append_cached_results(
                results,
                dataset_name,
                dataset_slug,
                sigma,
                table_summaries,
                plot_payloads,
                bin_plot_payloads,
            )
            verdict, interpretation = _interpret_kendall(delta, p_val, sigma)
            print(f"\n{verdict}\n")
            print(interpretation)
            return

    splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))
    print(f"Test samples: {len(splits.X_test)}")
    
    # LOAD MODEL
    rf_uq = RandomForestClassifierUQ()
    rf_key = registry.make_key(dataset_key, rf_uq.name)
    
    if registry.exists(rf_key):
        rf_uq = registry.load(rf_key)
        print(f"OK Loaded model: {rf_uq.name}")
    else:
        print(f"WARN Model not found, skipping: {rf_key}")
        return
    
# COMPUTE EPISTEMIC CLEAN
    
    _print_section("EPISTEMIC UNCERTAINTY (CLEAN)")
    
    X_test = splits.X_test
    y_test = splits.y_test
    
    predictions_clean, aleatoric_clean, epistemic_clean = rf_uq.predict_with_uncertainty(X_test)
    
    print(f"\nEpistemic statistics:")
    print(f"  Mean: {epistemic_clean.mean():.4f}")
    print(f"  Std:  {epistemic_clean.std():.4f}")
    print(f"  Min:  {epistemic_clean.min():.4f}")
    print(f"  Max:  {epistemic_clean.max():.4f}")
    print(f"  CV:   {epistemic_clean.std() / epistemic_clean.mean():.4f}")
    
    # STRATIFY BY EPISTEMIC
    
    _print_section("STRATIFICATION")
    
    # COMPUTE PERCENTILES
    epi_low_threshold = np.percentile(epistemic_clean, PERCENTILE_LOW)
    epi_high_threshold = np.percentile(epistemic_clean, PERCENTILE_HIGH)
    
    print(f"\nThresholds:")
    print(f"  Low  (p{PERCENTILE_LOW}):  {epi_low_threshold:.4f}")
    print(f"  High (p{PERCENTILE_HIGH}): {epi_high_threshold:.4f}")
    
    # SELECT SAMPLES
    low_epi_mask = epistemic_clean <= epi_low_threshold
    high_epi_mask = epistemic_clean >= epi_high_threshold
    
    low_epi_indices = np.where(low_epi_mask)[0]
    high_epi_indices = np.where(high_epi_mask)[0]
    
    print(f"\nAvailable samples:")
    print(f"  Low epistemic:  {len(low_epi_indices)}")
    print(f"  High epistemic: {len(high_epi_indices)}")
    
    # SAMPLE N FROM EACH GROUP
    np.random.seed(GLOBAL_SEED)
    
    if len(low_epi_indices) >= N_SAMPLES_PER_GROUP:
        low_epi_sample_idx = np.random.choice(low_epi_indices, N_SAMPLES_PER_GROUP, replace=False)
    else:
        low_epi_sample_idx = low_epi_indices
        print(f"WARN Using all {len(low_epi_indices)} low-epi samples")
    
    if len(high_epi_indices) >= N_SAMPLES_PER_GROUP:
        high_epi_sample_idx = np.random.choice(high_epi_indices, N_SAMPLES_PER_GROUP, replace=False)
    else:
        high_epi_sample_idx = high_epi_indices
        print(f"WARN Using all {len(high_epi_indices)} high-epi samples")
    
    print(f"\nSelected samples:")
    print(f"  Low group:  {len(low_epi_sample_idx)} samples")
    low_epi_mean = float(epistemic_clean[low_epi_sample_idx].mean())
    high_epi_mean = float(epistemic_clean[high_epi_sample_idx].mean())
    epi_ratio = _format_ratio(high_epi_mean, low_epi_mean)
    
    print(f"    Mean epistemic: {low_epi_mean:.4f}")
    print(f"  High group: {len(high_epi_sample_idx)} samples")
    print(f"    Mean epistemic: {high_epi_mean:.4f}")
    print(f"  Ratio: {epi_ratio}")
    
    # EXTRACT SUBSETS
    X_low = X_test[low_epi_sample_idx]
    X_high = X_test[high_epi_sample_idx]
    
    N_LOW = len(low_epi_sample_idx)
    N_HIGH = len(high_epi_sample_idx)
    
# COMPUTE SHAP CLEAN
    
    _print_section("SHAP EXPLANATIONS (CLEAN)")
    
    shap_explainer = SHAPExplainer(rf_uq.base_model)
    
    attr_low_clean = shap_explainer.explain(X_low)
    attr_high_clean = shap_explainer.explain(X_high)
    
    print(f"OK SHAP (low epi, clean):  {attr_low_clean.shape}")
    print(f"OK SHAP (high epi, clean): {attr_high_clean.shape}")
    
    # PRECOMPUTE CLEAN RANKINGS FOR LOW GROUP
    abs_attr_low_clean = np.abs(attr_low_clean)
    rank_low_clean = np.argsort(abs_attr_low_clean, axis=1)[:, ::-1]
    
    # PRECOMPUTE CLEAN RANKINGS FOR HIGH GROUP
    abs_attr_high_clean = np.abs(attr_high_clean)
    rank_high_clean = np.argsort(abs_attr_high_clean, axis=1)[:, ::-1]
    
# APPLY NOISE COMPUTE STABILITY
    
    _print_section(f"NOISE APPLICATION (sigma={sigma}, K={K_NOISE_SEEDS})")
    
    # Storage: [metric][sample_idx] = stability_value
    stability_low = {
        'kendall': np.zeros((N_LOW, K_NOISE_SEEDS)),
    }
    
    stability_high = {
        'kendall': np.zeros((N_HIGH, K_NOISE_SEEDS)),
    }
    
    perturb_generators = [PerturbationGenerator(seed=GLOBAL_SEED + i) for i in range(K_NOISE_SEEDS)]
    
    for seed_idx, perturb in tqdm(
        enumerate(perturb_generators),
        total=K_NOISE_SEEDS,
        desc="Noise seeds",
        leave=False,
    ):
        
        # PERTURB BOTH GROUPS
        X_low_noisy = perturb.gaussian(X_low, sigma)
        X_high_noisy = perturb.gaussian(X_high, sigma)
        
        # SHAP ON NOISY DATA
        attr_low_noisy = shap_explainer.explain(X_low_noisy)
        attr_high_noisy = shap_explainer.explain(X_high_noisy)
        
        # COMPUTE STABILITY FOR LOW GROUP
        for i in range(N_LOW):
            rank_pert = np.argsort(np.abs(attr_low_noisy[i]))[::-1]
            
            # KENDALL
            kendall_val = _stat_corr(kendalltau(rank_low_clean[i], rank_pert))
            if np.isnan(kendall_val):
                kendall_val = 0.0
            stability_low['kendall'][i, seed_idx] = kendall_val
        
        # COMPUTE STABILITY FOR HIGH GROUP
        for i in range(N_HIGH):
            rank_pert = np.argsort(np.abs(attr_high_noisy[i]))[::-1]
            
            # KENDALL
            kendall_val = _stat_corr(kendalltau(rank_high_clean[i], rank_pert))
            if np.isnan(kendall_val):
                kendall_val = 0.0
            stability_high['kendall'][i, seed_idx] = kendall_val
        
    
    # AVERAGE ACROSS NOISE SEEDS
    stability_low_mean = {k: v.mean(axis=1) for k, v in stability_low.items()}
    stability_high_mean = {k: v.mean(axis=1) for k, v in stability_high.items()}
    
    # STATISTICAL ANALYSIS
    
    _print_section("RESULTS")
    
    print(f"\nEpistemic uncertainty:")
    print(f"  Low group:  {low_epi_mean:.4f}")
    print(f"  High group: {high_epi_mean:.4f}")
    print(f"  Ratio:      {epi_ratio}")
    
    table_lines = [
        f"Stability (mean ± std across {K_NOISE_SEEDS} seeds):",
        f"{'Metric':<12} {'Low Epi':<20} {'High Epi':<20} {'Delta':<10} {'p-value'}",
        "-" * 70,
    ]
    
    results = {
        'dataset': dataset_name,
        'sigma': sigma,
        'k_noise_seeds': K_NOISE_SEEDS,
        'n_low': N_LOW,
        'n_high': N_HIGH,
        'epistemic_low_mean': low_epi_mean,
        'epistemic_high_mean': high_epi_mean,
        'epi_bins': EPI_BINS,
        'n_samples_per_bin': N_SAMPLES_PER_BIN,
        'epi_binning': EPI_BINNING,
        'stability_low': {},
        'stability_high': {},
        'delta': {}
    }
    
    metric = 'kendall'
    low_vals = stability_low_mean[metric]
    high_vals = stability_high_mean[metric]
    
    low_mean = low_vals.mean()
    low_std = low_vals.std()
    high_mean = high_vals.mean()
    high_std = high_vals.std()
    delta = low_mean - high_mean
    
    # T TEST
    t_stat, p_val = ttest_ind(low_vals, high_vals)
    
    table_lines.append(
        f"{metric:<12} {low_mean:.3f} ± {low_std:.3f}     "
        f"{high_mean:.3f} ± {high_std:.3f}     "
        f"{delta:+.3f}    {p_val:.4f}"
    )
    
    results['stability_low'][metric] = {
        'mean': float(low_mean),
        'std': float(low_std),
        'values': [float(v) for v in low_vals]
    }
    results['stability_high'][metric] = {
        'mean': float(high_mean),
        'std': float(high_std),
        'values': [float(v) for v in high_vals]
    }
    results['delta'][metric] = {
        'value': float(delta),
        't_stat': float(t_stat),
        'p_value': float(p_val)
    }

# EPISTEMIC BINS TREND OPTIONAL

    bin_table_lines = None
    bin_results = None

    if EPI_BINS >= 3 and N_SAMPLES_PER_BIN > 0:
        _print_section("EPISTEMIC BINS TREND")
        bin_labels = []
        bin_values = []
        bin_counts = []

        use_rank_bins = False
        rank_bins = None

        if EPI_BINNING == "uniform":
            min_epi = float(epistemic_clean.min())
            max_epi = float(epistemic_clean.max())
            bin_edges = np.linspace(min_epi, max_epi, EPI_BINS + 1)
            if min_epi == max_epi:
                print("WARN Epistemic min == max; uniform bins will be empty except last.")
        else:
            quantiles = np.linspace(0.0, 1.0, EPI_BINS + 1)
            bin_edges = np.quantile(epistemic_clean, quantiles)
            if np.any(np.diff(bin_edges) <= 0):
                # Handle tied values by splitting into equal-sized rank bins.
                use_rank_bins = True
                print("WARN Quantile edges are not strictly increasing; using rank-based bins.")
                order = np.argsort(epistemic_clean, kind="mergesort")
                rank_bins = np.array_split(order, EPI_BINS)
        rng = np.random.RandomState(GLOBAL_SEED)

        bin_table_lines = [
            f"Epistemic bins ({EPI_BINS}, {EPI_BINNING}) with up to {N_SAMPLES_PER_BIN} samples each:",
            f"{'Bin':<18} {'N':>4} {'Kendall':>12}",
            "-" * 50,
        ]

        bin_results = {
            'binning': EPI_BINNING,
            'edges': [float(v) for v in bin_edges],
            'bins': []
        }

        for bin_idx in range(EPI_BINS):
            if use_rank_bins:
                indices = rank_bins[bin_idx]
                if len(indices) == 0:
                    low_edge = float(bin_edges[bin_idx])
                    high_edge = float(bin_edges[bin_idx + 1])
                else:
                    low_edge = float(epistemic_clean[indices].min())
                    high_edge = float(epistemic_clean[indices].max())
                right_bracket = "]"
            else:
                low_edge = bin_edges[bin_idx]
                high_edge = bin_edges[bin_idx + 1]
                if bin_idx < EPI_BINS - 1:
                    mask = (epistemic_clean >= low_edge) & (epistemic_clean < high_edge)
                    right_bracket = ")"
                else:
                    mask = (epistemic_clean >= low_edge) & (epistemic_clean <= high_edge)
                    right_bracket = "]"

                indices = np.where(mask)[0]
            if len(indices) == 0:
                bin_label = f"[{low_edge:.4f}, {high_edge:.4f}{right_bracket}"
                bin_table_lines.append(
                    f"{bin_label:<18} {0:>4} {'n/a':>12}"
                )
                bin_results['bins'].append({
                    'range': [float(low_edge), float(high_edge)],
                    'n': 0,
                    'metrics': {},
                    'values': []
                })
                bin_labels.append(bin_label)
                bin_values.append(np.array([]))
                bin_counts.append(0)
                continue

            if len(indices) > N_SAMPLES_PER_BIN:
                sample_idx = rng.choice(indices, N_SAMPLES_PER_BIN, replace=False)
            else:
                sample_idx = indices

            X_bin = X_test[sample_idx]
            attr_bin_clean = shap_explainer.explain(X_bin)
            rank_bin_clean = np.argsort(np.abs(attr_bin_clean), axis=1)[:, ::-1]
            n_bin = len(sample_idx)
            
            stability_bin = {
                'kendall': np.zeros((n_bin, K_NOISE_SEEDS)),
            }

            perturb_generators = [
                PerturbationGenerator(seed=GLOBAL_SEED + i) for i in range(K_NOISE_SEEDS)
            ]

            for seed_idx, perturb in enumerate(perturb_generators):
                X_bin_noisy = perturb.gaussian(X_bin, sigma)
                attr_bin_noisy = shap_explainer.explain(X_bin_noisy)

                for i in range(n_bin):
                    rank_pert = np.argsort(np.abs(attr_bin_noisy[i]))[::-1]

                    kendall_val = _stat_corr(kendalltau(rank_bin_clean[i], rank_pert))
                    if np.isnan(kendall_val):
                        kendall_val = 0.0
                    stability_bin['kendall'][i, seed_idx] = kendall_val

            stability_bin_mean = {k: v.mean(axis=1) for k, v in stability_bin.items()}

            metrics_summary = {}
            mean_val = float(stability_bin_mean['kendall'].mean())
            std_val = float(stability_bin_mean['kendall'].std())
            metrics_summary['kendall'] = {
                'mean': mean_val,
                'std': std_val
            }

            bin_label = f"[{low_edge:.4f}, {high_edge:.4f}{right_bracket}"
            bin_table_lines.append(
                f"{bin_label:<18} {n_bin:>4} "
                f"{metrics_summary['kendall']['mean']:.3f}±{metrics_summary['kendall']['std']:.3f}"
            )

            bin_results['bins'].append({
                'range': [float(low_edge), float(high_edge)],
                'n': int(n_bin),
                'metrics': metrics_summary,
                'values': [float(v) for v in stability_bin_mean['kendall']]
            })
            bin_labels.append(bin_label)
            bin_values.append(stability_bin_mean['kendall'])
            bin_counts.append(int(n_bin))

        for line in bin_table_lines:
            print(line)

    if bin_results is not None:
        results['bins'] = bin_results
        bin_plot_payloads.append({
            'dataset': dataset_name,
            'labels': bin_labels,
            'values': bin_values,
            'counts': bin_counts,
        })

    table_summaries.append({
        'dataset': dataset_name,
        'lines': table_lines,
        'bins': bin_table_lines,
    })

    # INTERPRETATION
    
    _print_section("INTERPRETATION")
    
    kendall_delta = results['delta']['kendall']['value']
    kendall_pval = results['delta']['kendall']['p_value']
    verdict, interpretation = _interpret_kendall(kendall_delta, kendall_pval, sigma)
    
    print(f"\n{verdict}\n")
    print(interpretation)
    
    # VISUALIZATION
    
    plot_payloads.append({
        'dataset': dataset_name,
        'slug': dataset_slug,
        'low_vals': stability_low_mean['kendall'],
        'high_vals': stability_high_mean['kendall'],
    })
    
    # SAVE RESULTS
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"OK Results saved: {output_file}")
    
    # SUMMARY FILE
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_name} STRATIFIED VALIDATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Sigma: {sigma}\n")
        f.write(f"  K_NOISE_SEEDS: {K_NOISE_SEEDS}\n")
        f.write(f"  N_LOW: {N_LOW}\n")
        f.write(f"  N_HIGH: {N_HIGH}\n\n")
        f.write(f"  EPI_BINS: {EPI_BINS}\n")
        f.write(f"  N_SAMPLES_PER_BIN: {N_SAMPLES_PER_BIN}\n\n")
        f.write(f"  EPI_BINNING: {EPI_BINNING}\n\n")
        
        f.write(f"Epistemic uncertainty:\n")
        f.write(f"  Low group:  {results['epistemic_low_mean']:.4f}\n")
        f.write(f"  High group: {results['epistemic_high_mean']:.4f}\n")
        f.write(f"  Ratio:      {epi_ratio}\n\n")
        
        f.write(f"Stability results:\n")
        metric = 'kendall'
        f.write(f"  {metric}:\n")
        f.write(f"    Low:   {results['stability_low'][metric]['mean']:.3f} ± {results['stability_low'][metric]['std']:.3f}\n")
        f.write(f"    High:  {results['stability_high'][metric]['mean']:.3f} ± {results['stability_high'][metric]['std']:.3f}\n")
        f.write(f"    Delta: {results['delta'][metric]['value']:+.3f}\n")
        f.write(f"    p-val: {results['delta'][metric]['p_value']:.4f}\n\n")

        if bin_table_lines:
            f.write("Epistemic bins trend:\n")
            for line in bin_table_lines:
                f.write(f"{line}\n")
            f.write("\n")
        
        f.write(f"Verdict: {verdict}\n\n")
        f.write(f"Interpretation:\n{interpretation}\n")
    
    print(f"OK Summary saved: {summary_file}")
    
    print(f"OK {dataset_name} stratified validation complete")

for sigma in SIGMA:
    _print_section(f"SIGMA {sigma}")
    for ds in datasets:
        run_dataset(
            ds,
            cache,
            registry,
            splitter,
            table_summaries[sigma],
            plot_payloads[sigma],
            bin_plot_payloads[sigma],
            sigma,
        )

if any(bin_plot_payloads[sigma] for sigma in SIGMA):
    _print_section("GENERATING BIN PLOTS")
    dataset_order = [ds.name for ds in datasets]
    title_map = {
        "wine_binary": "Wine",
        "bean": "Bean",
        "rice": "Rice",
    }
    n_rows = len(SIGMA)
    n_cols = len(dataset_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.3 * n_cols, 1.4 * n_rows),
        sharey=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, sigma in enumerate(SIGMA):
        payload_map = {payload['dataset']: payload for payload in bin_plot_payloads[sigma]}
        for col_idx, dataset_name in enumerate(dataset_order):
            ax = axes[row_idx][col_idx]
            payload = payload_map.get(dataset_name)
            if payload is None:
                ax.axis("off")
                continue
            positions = np.arange(1, len(payload['labels']) + 1)
            gap_scale = 0.7
            scaled_positions = positions * gap_scale
            non_empty_positions = []
            non_empty_values = []
            non_empty_indices = []

            for idx, vals in enumerate(payload['values']):
                if len(vals) > 0:
                    non_empty_indices.append(idx)
                    non_empty_positions.append((idx + 1) * gap_scale)
                    non_empty_values.append(vals)

            if non_empty_values:
                if len(payload['labels']) == 3:
                    # LOW MEDIUM HIGH
                    colors = ["#4C78A8", "#59A14F", "#E45756"]
                else:
                    cmap = plt.get_cmap()
                    colors = cmap(np.linspace(0.1, 0.9, len(payload['labels'])))
                vp = ax.violinplot(
                    non_empty_values,
                    positions=non_empty_positions,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body, idx in zip(vp['bodies'], non_empty_indices):
                    color = colors[idx]
                    body.set_facecolor(color)
                    body.set_edgecolor('black')
                    body.set_alpha(0.7)

                if 'cmedians' in vp:
                    vp['cmedians'].set_color('black')
                    vp['cmedians'].set_linewidth(1.5)

            for pos, vals in zip(positions, payload['values']):
                if len(vals) == 0:
                    continue
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                scaled_pos = pos * gap_scale
                ax.errorbar(scaled_pos, mean_val, yerr=std_val, color='black', capsize=3, linewidth=1)
                ax.scatter(scaled_pos, mean_val, color='black', s=22, zorder=3)

            if len(positions) == 5:
                stratum_labels = ["lowest", "low", "medium", "high", "highest"]
            if len(positions) == 3:
                stratum_labels = ["low", "medium", "high"]
            else:
                stratum_labels = [str(i) for i in positions]
            ax.set_xticks(scaled_positions)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(stratum_labels, fontsize=10)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)
            if row_idx == 0:
                ax.set_title(title_map.get(dataset_name, dataset_name), fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f'τ @ σ={sigma}', fontsize=10)

    fig.supxlabel('Epistemic Uncertainty Strata', fontsize=11, y=0.04)
    # No global y-label; each row has its own label.
    plt.suptitle('Stratified XAI Stability (SHAP τ)', fontsize=13, y=0.95)
    fig.tight_layout(rect=[0.00, 0, 1, 1])
    sigma_tag = "_".join(str(s) for s in SIGMA)
    plot_file = RESULTS_DIR / f'stratified_validation_sigma{sigma_tag}_kendall_bins_all_datasets.{IMAGE_FILE_EXT}'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Plot saved: {plot_file}")

if any(table_summaries[sigma] for sigma in SIGMA):
    _print_section("STABILITY TABLES")
    for sigma in SIGMA:
        tables = table_summaries[sigma]
        if not tables:
            continue
        print(f"Sigma: {sigma}")
        for table in tables:
            print(f"Dataset: {table['dataset']}")
            for line in table['lines']:
                print(line)
            if table.get('bins'):
                print("")
                for line in table['bins']:
                    print(line)
            print("")
