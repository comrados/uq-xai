"""Study global XAI approximation via epistemic-guided sampling.

Compares global SHAP rankings from low- and high-epistemic subsets against a
reference ranking to assess distortion from uncertain samples.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
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
from config.settings import GLOBAL_SEED

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

# DATASETS TO PROCESS
DATASETS = ["bean", "rice", "wine_binary"]

# REFERENCE SAMPLE SIZE
N_REFERENCE = 1000

# SAMPLE FRACTIONS TO TEST
SAMPLE_FRACTIONS = np.arange(0.02, 0.31, 0.02)

# PREFERRED SUMMARY FRACTION FALLS BACK TO NEAREST AVAILABLE
SUMMARY_FRACTION = 0.50

# RANDOM SEED
SEED = GLOBAL_SEED

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HELPER FUNCTIONS

def _print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def _stat_corr(stat_result):
    """Extract correlation from SciPy stats result."""
    if hasattr(stat_result, "correlation"):
        return stat_result.correlation
    return stat_result[0]

def _pick_summary_index(fractions, target):
    """Pick the index of the closest available fraction to target."""
    fractions = np.asarray(fractions, dtype=float)
    return int(np.argmin(np.abs(fractions - target)))

def _pretty_dataset_name(dataset_name):
    """Map internal dataset ids to human-friendly labels."""
    return {
        "bean": "Bean",
        "rice": "Rice",
        "wine_binary": "Wine",
    }.get(dataset_name, dataset_name)

def compute_reference_global_shap(explainer, X, N=1000, seed=SEED):
    """
    Compute reference global feature importance.

    Args:
        explainer: Explainer instance (SHAP)
        X: Full test set, shape (n_samples, n_features)
        N: Number of samples to use for reference (default: 1000)
        seed: Random seed for subsampling

    Returns:
        reference_importance: Mean |attributions| per feature, shape (n_features,)
        indices_used: Indices of samples used
    """
    if len(X) > N:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), N, replace=False)
        X_ref = X[indices]
    else:
        indices = np.arange(len(X))
        X_ref = X

    # COMPUTE ATTRIBUTIONS
    attributions = explainer.explain(X_ref)

    # GLOBAL IMPORTANCE MEAN ATTRIBUTIONS PER FEATURE
    reference_importance = np.abs(attributions).mean(axis=0)

    return reference_importance, indices


def compute_approximation(explainer, X, indices):
    """
    Compute global approximation using subset of samples.

    Args:
        explainer: Explainer instance
        X: Full dataset, shape (n_samples, n_features)
        indices: Indices of samples to use

    Returns:
        approx_importance: Mean |attributions| per feature, shape (n_features,)
    """
    X_subset = X[indices]
    attributions = explainer.explain(X_subset)
    approx_importance = np.abs(attributions).mean(axis=0)
    return approx_importance


def select_samples_by_epistemic(epistemic_values, K, strategy):
    """
    Select K samples based on epistemic strategy.

    Args:
        epistemic_values: Epistemic uncertainty per sample, shape (n_samples,)
        K: Number of samples to select
        strategy: "low" or "high"

    Returns:
        indices: Selected sample indices, shape (K,)
    """
    n_samples = len(epistemic_values)
    K = min(K, n_samples)

    if strategy == "low":
        indices = np.argsort(epistemic_values)[:K]
    elif strategy == "high":
        indices = np.argsort(epistemic_values)[-K:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return indices


def compute_spearman_correlation(ref_importance, approx_importance):
    """
    Compute Spearman rank correlation between two importance vectors.

    Args:
        ref_importance: Reference importance, shape (n_features,)
        approx_importance: Approximation importance, shape (n_features,)

    Returns:
        correlation: Spearman rho
    """
    corr = _stat_corr(spearmanr(ref_importance, approx_importance))
    if np.isnan(corr):
        corr = 0.0
    return corr


# EXPERIMENT FUNCTIONS

def run_experiment(dataset_name, explainer, X_test, epistemic):
    """
    Run global approximation experiment on clean data.

    Args:
        dataset_name: Dataset identifier
        explainer: Explainer instance
        X_test: Test data, shape (n_samples, n_features)
        epistemic: Epistemic values, shape (n_samples,)

    Returns:
        results: Dictionary with correlation results per strategy
    """
    _print_section(f"{dataset_name} | random_forest | shap")

    # COMPUTE REFERENCE GLOBAL EXPLANATION
    print(f"Computing reference global explanation (N={N_REFERENCE})...")
    reference_importance, ref_indices = compute_reference_global_shap(
        explainer, X_test, N=N_REFERENCE, seed=SEED
    )
    print(f"Reference computed using {len(ref_indices)} samples")
    print(f"Top-3 features (reference): {np.argsort(reference_importance)[::-1][:3]}")

    summary_idx = _pick_summary_index(SAMPLE_FRACTIONS, SUMMARY_FRACTION)
    summary_fraction = float(SAMPLE_FRACTIONS[summary_idx])

    # STORAGE FOR RESULTS
    results = {
        'n_reference': len(ref_indices),
        'reference_importance': reference_importance,
        'sample_fractions': SAMPLE_FRACTIONS,
        'summary_fraction': summary_fraction,
        'low_epi': [],
        'high_epi': [],
        'low_summary_importance': None,
        'high_summary_importance': None,
    }

    # FOR EACH SAMPLE FRACTION
    for frac_idx, frac in enumerate(tqdm(SAMPLE_FRACTIONS, desc="Sampling", leave=False)):
        K = int(frac * len(X_test))
        K = max(1, K)

        # LOW EPISTEMIC STRATEGY
        indices_low = select_samples_by_epistemic(epistemic, K, "low")
        approx_low = compute_approximation(explainer, X_test, indices_low)
        corr_low = compute_spearman_correlation(reference_importance, approx_low)
        results['low_epi'].append(corr_low)

        # HIGH EPISTEMIC STRATEGY
        indices_high = select_samples_by_epistemic(epistemic, K, "high")
        approx_high = compute_approximation(explainer, X_test, indices_high)
        corr_high = compute_spearman_correlation(reference_importance, approx_high)
        results['high_epi'].append(corr_high)

        if frac_idx == summary_idx:
            results['low_summary_importance'] = approx_low
            results['high_summary_importance'] = approx_high

    # CONVERT TO ARRAYS
    results['low_epi'] = np.array(results['low_epi'])
    results['high_epi'] = np.array(results['high_epi'])

    return results


def plot_efficiency_curves(results, dataset_name, output_path):
    """
    Plot efficiency curves: correlation vs sample fraction.

    Args:
        results: Results dictionary from run_experiment
        dataset_name: Dataset identifier
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    fractions = results['sample_fractions']

    # LOW EPISTEMIC BLUE
    ax.plot(fractions, results['low_epi'],
            color='blue', marker='o', linewidth=2, markersize=6,
            label='Low Epistemic', zorder=3)

    # HIGH EPISTEMIC RED
    ax.plot(fractions, results['high_epi'],
            color='red', marker='^', linewidth=2, markersize=6,
            label='High Epistemic', zorder=3)

    ax.set_xlabel('Sample Fraction (%)', fontsize=12)
    ax.set_ylabel('Spearman rho (vs Reference)', fontsize=12)
    pretty_name = _pretty_dataset_name(dataset_name)
    ax.set_title(f'{pretty_name} | Random Forest + SHAP',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    fractions = np.asarray(fractions, dtype=float)
    x_min = 0.0
    x_max = float(fractions.max())
    x_pad = 0.02 * max(1.0, x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    ax.set_ylim(0.55, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Plot saved: {output_path}")

def plot_summary_scatter(results, dataset_name, output_path):
    """Plot reference vs approximation at the summary fraction."""
    ref = results.get('reference_importance')
    low = results.get('low_summary_importance')
    high = results.get('high_summary_importance')
    summary_fraction = results.get('summary_fraction')

    if ref is None or low is None or high is None:
        print("WARN Missing summary importances; skipping summary scatter plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    title_suffix = f"fraction={summary_fraction:.2f}" if summary_fraction is not None else ""

    axes[0].scatter(ref, low, color='blue', alpha=0.7, s=28)
    axes[0].set_title(f"Low Epistemic ({title_suffix})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Reference |phi|", fontsize=10)
    axes[0].set_ylabel("Approx |phi|", fontsize=10)

    axes[1].scatter(ref, high, color='red', alpha=0.7, s=28)
    axes[1].set_title(f"High Epistemic ({title_suffix})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Reference |phi|", fontsize=10)

    max_val = float(np.max([ref.max(), low.max(), high.max()]))
    pad = 0.05 * max_val if max_val > 0 else 0.1
    for ax in axes:
        ax.plot([0, max_val + pad], [0, max_val + pad], linestyle='--', color='black', linewidth=1, alpha=0.6)
        ax.set_xlim(0, max_val + pad)
        ax.set_ylim(0, max_val + pad)
        ax.grid(True, alpha=0.25)

    pretty_name = _pretty_dataset_name(dataset_name)
    fig.suptitle(f"{pretty_name} | Reference vs Approximation", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Summary scatter saved: {output_path}")

def plot_combined_efficiency(all_results, output_path):
    """Plot all datasets in one figure with subplots."""
    dataset_names = [name for name in DATASETS if name in all_results]
    if not dataset_names:
        print("WARN No results available for combined plot")
        return

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(6 * len(dataset_names), 5), sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, dataset_names):
        results = all_results[dataset_name]
        fractions = np.asarray(results['sample_fractions'], dtype=float)
        pretty_name = _pretty_dataset_name(dataset_name)

        ax.plot(fractions, results['low_epi'],
                color='blue', marker='o', linewidth=2, markersize=5,
                label='Low Epistemic', zorder=3)
        ax.plot(fractions, results['high_epi'],
                color='red', marker='^', linewidth=2, markersize=5,
                label='High Epistemic', zorder=3)

        x_min = 0.0
        x_max = float(fractions.max())
        x_pad = 0.02 * max(1.0, x_max - x_min)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0.55, 1.05)

        ax.set_title(pretty_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Fraction (%)', fontsize=11)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    axes[0].set_ylabel('Spearman rho (vs Reference)', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Combined plot saved: {output_path}")

def plot_combined_gap(all_results, output_path):
    """Plot gap curves for all datasets in one figure."""
    dataset_names = [name for name in DATASETS if name in all_results]
    if not dataset_names:
        print("WARN No results available for combined gap plot")
        return

    fig, ax = plt.subplots(figsize=(6, 3))

    for dataset_name in dataset_names:
        results = all_results[dataset_name]
        fractions = np.asarray(results['sample_fractions'], dtype=float)
        gap = results['low_epi'] - results['high_epi']
        pretty_name = _pretty_dataset_name(dataset_name)
        ax.plot(fractions, gap, linewidth=2, marker= 'o', markersize=4, label=pretty_name)

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Sample Fraction (%)', fontsize=11)
    ax.set_ylabel('Δ Spearman ρ (Low − High)', fontsize=11)
    ax.set_title('Global Explanation Approximation: Low–High Epistemic Gap', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    x_min = 0.0
    x_max = float(np.max(SAMPLE_FRACTIONS))
    x_pad = 0.02 * max(1.0, x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Combined gap plot saved: {output_path}")


# MAIN

def main():
    """Main execution function."""
    _print_section("EXPERIMENT 10: GLOBAL XAI APPROXIMATION")

    print(f"Configuration:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Reference size: {N_REFERENCE}")
    print(f"  Sample fractions: {SAMPLE_FRACTIONS}")

    cache = Cache()
    registry = ModelRegistry()
    splitter = DataSplitter()

    all_results = {}

    dataset_map = {
        "wine_binary": WineQualityDataset(),
        "bean": DryBeanDataset(),
        "rice": RiceDataset(),
    }

    for dataset_name in DATASETS:
        _print_section(f"PROCESSING: {dataset_name}")

        # USE CACHED RESULTS IF AVAILABLE
        save_path = RESULTS_DIR / f"{dataset_name}_results.pkl"
        plot_path = RESULTS_DIR / f"{dataset_name}_efficiency.pdf"
        if save_path.exists():
            print(f"OK Using cached results: {save_path}")
            with open(save_path, "rb") as f:
                results = pickle.load(f)
            all_results[dataset_name] = results
            plot_efficiency_curves(results, dataset_name, plot_path)
            summary_plot_path = RESULTS_DIR / f"{dataset_name}_summary_scatter.pdf"
            plot_summary_scatter(results, dataset_name, summary_plot_path)
            continue

        # LOAD DATASET
        ds = dataset_map[dataset_name]
        info = cache.load_or_create(ds.cache_key, ds.load)
        dataset_key = f"{info.name}_{ds.uci_id}"
        splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

        print(f"Dataset: {info.name}")
        print(f"Test samples: {len(splits.X_test)}")
        print(f"Features: {len(splits.feature_names)}")

        # LOAD MODEL
        model_uq = RandomForestClassifierUQ()
        model_key = registry.make_key(dataset_key, model_uq.name)

        if registry.exists(model_key):
            model_uq = registry.load(model_key)
            print(f"OK Loaded model: {model_uq.name}")
        else:
            print(f"WARN Model not found, skipping: {model_key}")
            continue

        model = model_uq.base_model

        # COMPUTE EPISTEMIC
        print("Computing epistemic uncertainty...")
        _, _, epistemic = model_uq.predict_with_uncertainty(splits.X_test)
        print(f"Epistemic: mean={epistemic.mean():.4f}, std={epistemic.std():.4f}, "
              f"min={epistemic.min():.4f}, max={epistemic.max():.4f}")

        # INITIALIZE EXPLAINER
        explainer = SHAPExplainer(model)
        print(f"OK Explainer initialized: SHAP")

        # RUN EXPERIMENT
        results = run_experiment(
            dataset_name=dataset_name,
            explainer=explainer,
            X_test=splits.X_test,
            epistemic=epistemic
        )

        all_results[dataset_name] = results

        # SAVE RESULTS
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"OK Results saved: {save_path}")

        # PLOT
        plot_efficiency_curves(results, dataset_name, plot_path)
        summary_plot_path = RESULTS_DIR / f"{dataset_name}_summary_scatter.pdf"
        plot_summary_scatter(results, dataset_name, summary_plot_path)

    # SUMMARY
    _print_section("GENERATING SUMMARY")

    if all_results:
        table_path = RESULTS_DIR / "summary_table.tex"

        combined_plot_path = RESULTS_DIR / "combined_efficiency.pdf"
        plot_combined_efficiency(all_results, combined_plot_path)
        combined_gap_path = RESULTS_DIR / "combined_gap_curve.pdf"
        plot_combined_gap(all_results, combined_gap_path)

        print(f"\nOK Experiment 10 completed!")
        print(f"OK Results saved to: {RESULTS_DIR}")
    else:
        print("WARN No results to summarize")


if __name__ == "__main__":
    main()
