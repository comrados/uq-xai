"""Analyze contamination and recovery with high-epistemic samples.

Stress-tests global SHAP stability when batches are contaminated with
high-epistemic samples and evaluates epistemic filtering as a recovery step.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
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
from uncertainty.mlp_uq import MLPClassifierUQ
from config.settings import GLOBAL_SEED, XAI_CONFIG

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

# DATASETS TO PROCESS
DATASETS = ["bean", "rice", "wine_binary"]

# MODELS TO PROCESS
MODEL_SPECS = [
    {
        "key": "rf",
        "label": "RF",
        "uq_cls": RandomForestClassifierUQ,
        "shap": "tree",
    },
]

"""
    {
        "key": "mlp",
        "label": "MLP",
        "uq_cls": MLPClassifierUQ,
        "shap": "kernel",
    },
"""

# BASE SAMPLE SIZE FOR CLEAN SET
N_BASE = 50

# CONTAMINATION RATES TO TEST
CONTAMINATION_RATES = [i * 0.5 for i in range(0, 11)]

# OPTIONAL ADDITIONAL NOISE ON HIGH EPI SAMPLES
# NOISE TYPE GAUSSIAN PERMUTATION OR NONE
NOISE_TYPE = "gaussian"
# GAUSSIAN NOISE SCALE USED WHEN NOISE TYPE GAUSSIAN
HIGH_EPI_NOISE_BOOST = 0.5
# FRACTION OF FEATURES TO PERMUTE USED WHEN NOISE TYPE PERMUTATION
PERMUTATION_FEATURE_FRACTION = 1.0

# FILTERING THRESHOLD KEEP FIXED NUMBER OF LOWEST EPISTEMIC SAMPLES
FILTER_KEEP_N = int(N_BASE * 0.8)

# RANDOM SAMPLING REPEATS FOR NAIVE BASELINE SAMPLED FROM FULL BATCH
NAIVE_SAMPLE_REPEATS = 5

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

def _format_ratio_label(value):
    """Format contamination ratio label as 1:x."""
    value = float(value)
    if np.isclose(value, round(value)):
        return f"1:{int(round(value))}"
    return f"1:{value:.2f}".rstrip("0").rstrip(".")


def compute_global_shap(explainer, X):
    """
    Compute global feature importance for given samples.

    Args:
        explainer: SHAP explainer instance
        X: Sample array, shape (n_samples, n_features)

    Returns:
        importance: Mean |SHAP| per feature
    """
    attributions = explainer.explain(X)
    importance = np.abs(attributions).mean(axis=0)
    return importance


def compute_spearman(ref, approx):
    """Compute Spearman correlation, handling NaN."""
    corr = _stat_corr(spearmanr(ref, approx))
    if np.isnan(corr):
        corr = 0.0
    return corr


def add_noise(X, noise_type, sigma, permute_fraction, seed=SEED):
    """Add noise to samples (gaussian or permutation)."""
    if noise_type == "none":
        return X

    rng = np.random.RandomState(seed)

    if noise_type == "gaussian":
        if sigma <= 0:
            return X
        feature_std = X.std(axis=0, keepdims=True)
        noise = rng.randn(*X.shape) * sigma * feature_std
        return X + noise

    if noise_type == "permutation":
        if permute_fraction <= 0:
            return X
        X_noisy = X.copy()
        n_features = X.shape[1]
        if permute_fraction <= 1.0:
            n_perm = max(1, int(round(n_features * permute_fraction)))
        else:
            n_perm = min(n_features, int(permute_fraction))
        feature_idx = rng.choice(n_features, n_perm, replace=False)
        for j in feature_idx:
            X_noisy[:, j] = rng.permutation(X_noisy[:, j])
        return X_noisy

    raise ValueError(f"Unknown noise_type: {noise_type}")


def build_shap_explainer(model_uq, model_spec, X_train):
    """Build SHAP explainer for a given model spec."""
    if model_spec["shap"] == "tree":
        return SHAPExplainer(model_uq.base_model)
    if model_spec["shap"] == "kernel":
        background_n = min(XAI_CONFIG['shap_background_samples'], len(X_train))
        X_background = X_train[:background_n]
        return SHAPExplainer(model_uq.base_model, X_background=X_background)
    raise ValueError(f"Unknown SHAP mode: {model_spec['shap']}")


# EXPERIMENT

def run_contamination_experiment(dataset_name, model_label, model_uq, explainer, X_test, epistemic):
    """
    Run contamination and recovery experiment with real high-epi samples.

    Args:
        dataset_name: Dataset identifier
        model_uq: Model with uncertainty quantification
        explainer: SHAP explainer
        X_test: Test data
        epistemic: Epistemic uncertainty on clean data

    Returns:
        results: Dictionary with contamination results
    """
    _print_section(f"{dataset_name} | {model_label} | Contamination Experiment (High-Epi Samples)")

    n_samples = len(X_test)

    if N_BASE > n_samples:
        print(f"WARNING: need {N_BASE} samples, have {n_samples}")
        return None

    # SORT INDICES BY EPISTEMIC
    sorted_indices = np.argsort(epistemic)
    low_epi_indices = sorted_indices[:N_BASE]  # Lowest N_BASE
    high_epi_indices = sorted_indices[-N_BASE:][::-1]  # Highest N_BASE (descending)
    overlap = np.intersect1d(low_epi_indices, high_epi_indices).size

    print(f"Total samples: {n_samples}")
    print(f"Low-epi pool: {len(low_epi_indices)}")
    print(f"High-epi pool: {len(high_epi_indices)}")
    if overlap > 0:
        print(f"WARNING: low/high pools overlap by {overlap} samples")
    print(f"Base size: {N_BASE}")
    print(f"Noise type: {NOISE_TYPE}")
    if NOISE_TYPE == "gaussian":
        print(f"High-epi noise boost: {HIGH_EPI_NOISE_BOOST}")
    elif NOISE_TYPE == "permutation":
        print(f"Permutation feature fraction: {PERMUTATION_FEATURE_FRACTION}")

    # CLEAN BASE SAMPLES LOW EPI
    X_clean = X_test[low_epi_indices]
    clean_size = len(X_clean)

    # REFERENCE GLOBAL SHAP ON CLEAN LOW EPI SAMPLES ONLY
    reference_importance = compute_global_shap(explainer, X_clean)
    print(f"Reference computed on {len(X_clean)} low-epi samples")
    print(f"Reference top-3 features: {np.argsort(reference_importance)[::-1][:3]}")

    # RESULTS STORAGE
    results = {
        'contamination_rates': CONTAMINATION_RATES,
        'reference_importance': reference_importance,
        'n_base': N_BASE,
        'model_label': model_label,
        'noise_type': NOISE_TYPE,
        'noise_boost': HIGH_EPI_NOISE_BOOST,
        'permute_fraction': PERMUTATION_FEATURE_FRACTION,
        'rho_naive': [],
        'rho_naive_sampled_mean': [],
        'rho_naive_sampled_std': [],
        'rho_filtered': [],
        'n_batch': [],
        'n_filtered': [],
        'mean_epi_clean': [],
        'mean_epi_contam': [],
    }

    # RUN FOR EACH CONTAMINATION RATE
    for cont_rate in tqdm(CONTAMINATION_RATES, desc="Contamination"):
        n_contam = int(N_BASE * cont_rate)

        if n_contam > 0:
            # BUILD CONTAMINATED BATCH FROM HIGHEST EPI POOL REPEAT WITH NOISE IF NEEDED
            high_pool = X_test[high_epi_indices]
            n_pool = len(high_pool)
            full_repeats = n_contam // n_pool
            remainder = n_contam % n_pool
            noise_seed_base = SEED + int(cont_rate * 1000)

            parts = []
            for i in range(full_repeats):
                X_block = high_pool.copy()
                if NOISE_TYPE != "none":
                    X_block = add_noise(
                        X_block,
                        noise_type=NOISE_TYPE,
                        sigma=HIGH_EPI_NOISE_BOOST,
                        permute_fraction=PERMUTATION_FEATURE_FRACTION,
                        seed=noise_seed_base + i,
                    )
                parts.append(X_block)

            if remainder > 0:
                X_block = high_pool[:remainder].copy()
                if NOISE_TYPE != "none":
                    X_block = add_noise(
                        X_block,
                        noise_type=NOISE_TYPE,
                        sigma=HIGH_EPI_NOISE_BOOST,
                        permute_fraction=PERMUTATION_FEATURE_FRACTION,
                        seed=noise_seed_base + full_repeats,
                    )
                parts.append(X_block)

            X_contam = np.vstack(parts) if parts else np.array([]).reshape(0, X_clean.shape[1])

            # BUILD CONTAMINATED BATCH
            X_batch = np.vstack([X_clean, X_contam])
        else:
            X_batch = X_clean.copy()
            X_contam = np.array([]).reshape(0, X_clean.shape[1])

        results['n_batch'].append(len(X_batch))

        # COMPUTE EPISTEMIC ON THE BATCH
        _, _, batch_epistemic = model_uq.predict_with_uncertainty(X_batch)

        # TRACK EPISTEMIC FOR CLEAN VS CONTAMINATED
        if n_contam > 0:
            mean_epi_clean = batch_epistemic[:clean_size].mean()
            mean_epi_contam = batch_epistemic[clean_size:].mean()
        else:
            mean_epi_clean = batch_epistemic.mean()
            mean_epi_contam = 0.0

        results['mean_epi_clean'].append(mean_epi_clean)
        results['mean_epi_contam'].append(mean_epi_contam)

        # NAIVE USE ALL SAMPLES
        global_naive = compute_global_shap(explainer, X_batch)
        rho_naive = compute_spearman(reference_importance, global_naive)
        results['rho_naive'].append(rho_naive)

        # NAIVE SAMPLED RANDOM SUBSET OF FULL BATCH
        sampled_rhos = []
        sample_n = min(FILTER_KEEP_N, len(X_batch))
        for r in range(NAIVE_SAMPLE_REPEATS):
            rng = np.random.RandomState(SEED + int(cont_rate * 1000) + r)
            sample_idx = rng.choice(len(X_batch), sample_n, replace=False)
            global_sampled = compute_global_shap(explainer, X_batch[sample_idx])
            sampled_rhos.append(compute_spearman(reference_importance, global_sampled))
        results['rho_naive_sampled_mean'].append(float(np.mean(sampled_rhos)))
        results['rho_naive_sampled_std'].append(float(np.std(sampled_rhos)))

        # FILTERED KEEP LOWEST EPISTEMIC N WITHIN BATCH
        keep_n = min(FILTER_KEEP_N, len(X_batch))
        keep_indices = np.argsort(batch_epistemic)[:keep_n]
        X_filtered = X_batch[keep_indices]

        results['n_filtered'].append(len(X_filtered))

        if len(X_filtered) > 0:
            global_filtered = compute_global_shap(explainer, X_filtered)
            rho_filtered = compute_spearman(reference_importance, global_filtered)
        else:
            rho_filtered = 0.0

        results['rho_filtered'].append(rho_filtered)

    # CONVERT TO ARRAYS
    for key in ['rho_naive', 'rho_naive_sampled_mean', 'rho_naive_sampled_std',
                'rho_filtered', 'n_batch', 'n_filtered',
                'mean_epi_clean', 'mean_epi_contam']:
        results[key] = np.array(results[key])

    # COMPUTE RECOVERY METRICS
    results['recovery'] = results['rho_filtered'] - results['rho_naive']
    degradation = 1.0 - results['rho_naive']
    results['recovery_rate'] = np.where(
        degradation > 0.01,
        results['recovery'] / degradation,
        0.0
    )

    # PRINT SUMMARY
    print(f"\n{'Cont%':<8} {'Batch':<8} {'Filt':<8} {'EpiClean':<10} {'EpiContam':<10} "
          f"{'rho_naive':<10} {'rho_filt':<10} {'Recov':<10}")
    print("-" * 85)
    for i, cont_rate in enumerate(CONTAMINATION_RATES):
        print(f"{cont_rate:<8.0%} {results['n_batch'][i]:<8} {results['n_filtered'][i]:<8} "
              f"{results['mean_epi_clean'][i]:<10.4f} {results['mean_epi_contam'][i]:<10.4f} "
              f"{results['rho_naive'][i]:<10.3f} {results['rho_filtered'][i]:<10.3f} "
              f"{results['recovery'][i]:<+10.3f}")

    return results


def plot_degradation_recovery(results, dataset_name, output_path):
    """Plot degradation and recovery curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cont_rates = np.array(results['contamination_rates'], dtype=float)
    x_min = min(0.0, cont_rates.min() - 0.1)
    x_max = cont_rates.max() + 0.1
    mask = np.array(results['contamination_rates']) > 0
    clean_vals = results['mean_epi_clean'][mask] if mask.any() else results['mean_epi_clean']
    contam_vals = results['mean_epi_contam'][mask] if mask.any() else results['mean_epi_contam']
    clean_stats = f"Clean epi {clean_vals.mean():.3f}±{clean_vals.std():.3f}"
    contam_stats = f"Contam epi {contam_vals.mean():.3f}±{contam_vals.std():.3f}"

    ax.plot(cont_rates, results['rho_naive'],
            color='red', marker='o', linewidth=2, markersize=8,
            label='Naive all', zorder=2)
    ax.plot(
        cont_rates,
        results['rho_naive_sampled_mean'],
        color='orange',
        marker='x',
        linewidth=1.5,
        markersize=7,
        label=f'Naive sampled (N={FILTER_KEEP_N}, mean±std)',
        zorder=2,
    )
    ax.fill_between(
        cont_rates,
        results['rho_naive_sampled_mean'] - results['rho_naive_sampled_std'],
        results['rho_naive_sampled_mean'] + results['rho_naive_sampled_std'],
        color='orange',
        alpha=0.2,
        zorder=1,
    )

    ax.plot(cont_rates, results['rho_filtered'],
            color='blue', marker='s', linewidth=2, markersize=8,
            label=f'Filtered (lowest {FILTER_KEEP_N} epistemic)', zorder=3)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
               label='Reference (rho=1.0)', zorder=1)

    ax.set_xlabel('Clean:contaminated ratio', fontsize=12)
    ax.set_ylabel('Spearman rho (vs Clean Reference)', fontsize=12)
    ax.set_xticks(cont_rates)
    ax.set_xticklabels([_format_ratio_label(v) for v in cont_rates])
    
    model_label = results.get('model_label')
    if model_label:
        title = f'{dataset_name} | {model_label} | Contamination & Recovery'
    else:
        title = f'{dataset_name} | Contamination & Recovery'
    noise_type = results.get('noise_type', 'gaussian')
    if noise_type == "gaussian" and results.get('noise_boost', 0) > 0:
        title += f' (gaussian sigma={results["noise_boost"]})'
    elif noise_type == "permutation":
        title += f' (permute frac={results.get("permute_fraction", 1.0)})'
    title += f'\n{clean_stats} | {contam_stats}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Plot saved: {output_path}")


def plot_epistemic_separation(results, dataset_name, output_path):
    """Plot epistemic values for clean vs contaminated samples."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cont_rates = np.array(results['contamination_rates'], dtype=float)

    # ONLY PLOT WHERE WE HAVE CONTAMINATION
    mask = np.array(results['contamination_rates']) > 0

    if mask.sum() > 0:
        ax.plot(cont_rates[mask], results['mean_epi_clean'][mask],
                color='blue', marker='o', linewidth=2, markersize=8,
                label='Clean (low-epi base)')

        ax.plot(cont_rates[mask], results['mean_epi_contam'][mask],
                color='red', marker='^', linewidth=2, markersize=8,
                label='Contaminants (high-epi)')

    ax.set_xlabel('Clean:contaminated ratio', fontsize=12)
    ax.set_ylabel('Mean Epistemic Uncertainty', fontsize=12)
    ax.set_xticks(cont_rates)
    ax.set_xticklabels([_format_ratio_label(v) for v in cont_rates])
    ax.set_title(f'{dataset_name} | Epistemic Separation',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Epistemic plot saved: {output_path}")


def plot_combined_contamination(all_results, output_path, model_label=None):
    """Plot all datasets in one figure."""
    dataset_names = [name for name in DATASETS if name in all_results]
    if not dataset_names:
        return

    max_rate = max(
        np.array(all_results[name]['contamination_rates'], dtype=float).max()
        for name in dataset_names
    )
    x_min = 0.0
    x_max = max_rate + 0.1

    fig, axes = plt.subplots(1, len(dataset_names),
                             figsize=(5 * len(dataset_names), 5), sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, dataset_names):
        results = all_results[dataset_name]
        cont_rates = np.array(results['contamination_rates'], dtype=float)
        mask = np.array(results['contamination_rates']) > 0
        clean_vals = results['mean_epi_clean'][mask] if mask.any() else results['mean_epi_clean']
        contam_vals = results['mean_epi_contam'][mask] if mask.any() else results['mean_epi_contam']
        clean_stats = f"Clean epi {clean_vals.mean():.3f}±{clean_vals.std():.3f}"
        contam_stats = f"Contam epi {contam_vals.mean():.3f}±{contam_vals.std():.3f}"

        ax.plot(cont_rates, results['rho_naive'],
                color='red', marker='o', linewidth=2, markersize=6,
                label='Naive all')
        ax.plot(
            cont_rates,
            results['rho_naive_sampled_mean'],
            color='orange',
            marker='x',
            linewidth=1.5,
            markersize=6,
            label=f'Naive sampled (N={FILTER_KEEP_N}, mean±std)',
        )
        ax.fill_between(
            cont_rates,
            results['rho_naive_sampled_mean'] - results['rho_naive_sampled_std'],
            results['rho_naive_sampled_mean'] + results['rho_naive_sampled_std'],
            color='orange',
            alpha=0.2,
        )
        ax.plot(cont_rates, results['rho_filtered'],
                color='blue', marker='s', linewidth=2, markersize=6,
                label='Filtered')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)

        ax.set_title(f"{dataset_name}\n{clean_stats} | {contam_stats}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Clean:contaminated ratio', fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(cont_rates)
        ax.set_xticklabels([_format_ratio_label(v) for v in cont_rates])
        ax.legend(loc='lower right', fontsize=9)

    axes[0].set_ylabel('Spearman rho', fontsize=11)

    if model_label:
        fig.suptitle(f"Model: {model_label}", fontsize=12, fontweight='bold')
        plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    else:
        plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Combined plot saved: {output_path}")


# MAIN

def main():
    """Main execution function."""
    _print_section("EXPERIMENT 11: CONTAMINATION & RECOVERY (High-Epi Samples)")

    print(f"Configuration:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Models: {[spec['label'] for spec in MODEL_SPECS]}")
    print(f"  Base size: {N_BASE}")
    print(f"  Contamination rates: {CONTAMINATION_RATES}")
    print(f"  Noise type: {NOISE_TYPE}")
    if NOISE_TYPE == "gaussian":
        print(f"  High-epi noise boost: {HIGH_EPI_NOISE_BOOST}")
    elif NOISE_TYPE == "permutation":
        print(f"  Permutation feature fraction: {PERMUTATION_FEATURE_FRACTION}")
    print(f"  Naive sampled repeats: {NAIVE_SAMPLE_REPEATS}")
    print(f"  Filter keep N: {FILTER_KEEP_N}")

    cache = Cache()
    registry = ModelRegistry()
    splitter = DataSplitter()

    all_results_by_model = {spec["key"]: {} for spec in MODEL_SPECS}

    dataset_map = {
        "wine_binary": WineQualityDataset(),
        "bean": DryBeanDataset(),
        "rice": RiceDataset(),
    }

    for dataset_name in DATASETS:
        _print_section(f"PROCESSING: {dataset_name}")

        # LOAD DATASET
        ds = dataset_map[dataset_name]
        info = cache.load_or_create(ds.cache_key, ds.load)
        dataset_key = f"{info.name}_{ds.uci_id}"
        splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

        print(f"Dataset: {info.name}")
        print(f"Test samples: {len(splits.X_test)}")

        for model_spec in MODEL_SPECS:
            model_label = model_spec["label"]
            model_uq = model_spec["uq_cls"]()
            model_key = registry.make_key(dataset_key, model_uq.name)

            if registry.exists(model_key):
                model_uq = registry.load(model_key)
                print(f"OK Loaded model: {model_label} ({model_uq.name})")
            else:
                print(f"WARNING: Model not found, skipping: {model_label} ({model_key})")
                continue

            # COMPUTE EPISTEMIC ON CLEAN DATA
            print(f"Computing epistemic on clean data ({model_label})...")
            _, _, epistemic = model_uq.predict_with_uncertainty(splits.X_test)
            print(f"Epistemic: mean={epistemic.mean():.4f}, std={epistemic.std():.4f}, "
                  f"min={epistemic.min():.4f}, max={epistemic.max():.4f}")

            # INITIALIZE EXPLAINER
            explainer = build_shap_explainer(model_uq, model_spec, splits.X_train)

            # RUN EXPERIMENT
            results = run_contamination_experiment(
                dataset_name=dataset_name,
                model_label=model_label,
                model_uq=model_uq,
                explainer=explainer,
                X_test=splits.X_test,
                epistemic=epistemic
            )

            if results is None:
                continue

            all_results_by_model[model_spec["key"]][dataset_name] = results

            # SAVE RESULTS
            save_path = RESULTS_DIR / f"{dataset_name}_{model_spec['key']}_contamination.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"OK Results saved: {save_path}")

            # PLOTS
            plot_degradation_recovery(
                results, dataset_name,
                RESULTS_DIR / f"{dataset_name}_{model_spec['key']}_degradation_recovery.png"
            )

    # SUMMARY
    _print_section("GENERATING SUMMARY")

    any_results = False
    for model_spec in MODEL_SPECS:
        model_results = all_results_by_model.get(model_spec["key"], {})
        if not model_results:
            continue
        any_results = True
        combined_path = RESULTS_DIR / f"combined_contamination_{model_spec['key']}.png"
        plot_combined_contamination(model_results, combined_path, model_label=model_spec["label"])

    if any_results:
        print(f"\nOK Experiment 11 completed!")
        print(f"OK Results saved to: {RESULTS_DIR}")
    else:
        print("WARNING: No results to summarize")


if __name__ == "__main__":
    main()
