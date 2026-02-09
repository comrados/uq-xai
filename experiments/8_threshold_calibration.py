"""Calibrate epistemic thresholds at a fixed sigma.

Labels SHAP explanations as stable or unstable at a fixed Gaussian sigma and
calibrates epistemic thresholds to hit target coverages.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import pickle
from pathlib import Path
from tqdm import tqdm

from data.cache import Cache
from data.datasets import (
    WineQualityDataset,
    DryBeanDataset,
    RiceDataset,
    EcoliDataset,
)
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from data.perturbations import PerturbationGenerator
from config.settings import GLOBAL_SEED

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

SIGMA = 0.1  # Chosen from RQ4 results (same as 8_stratified.py)
K_NOISE_SEEDS = 5  # Number of noise realizations

MAX_SAMPLES = 1000  # Cap per dataset for speed

STABILITY_METRICS = {
    'kendall': {
        'stable_threshold': 0.7,
        'unstable_threshold': 0.5
    },
    'spearman': {
        'stable_threshold': 0.7,
        'unstable_threshold': 0.5
    },
    'topk_overlap': {
        'stable_threshold': 0.8,
        'unstable_threshold': 0.6
    },
    'jaccard': {
        'stable_threshold': 0.7,
        'unstable_threshold': 0.5
    }
}

TOP_K = 5  # For top-k overlap/Jaccard

TARGET_COVERAGES = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_COVERAGE = 0.2
MAX_COVERAGE = 0.8

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HELPERS

def _stat_corr(stat_result):
    """Extract correlation from SciPy stats result across versions."""
    if hasattr(stat_result, "correlation"):
        return stat_result.correlation
    return stat_result[0]

def _print_section(title):
    print(f"\n===== {title} =====")

# SETUP

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    RiceDataset(),
]

GLOBAL_SUMMARY = []

def run_dataset(ds):
    info = cache.load_or_create(ds.cache_key, ds.load)
    dataset_name = info.name
    dataset_key = f"{dataset_name}_{ds.uci_id}"
    splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

    output_file = RESULTS_DIR / f"{dataset_name}_threshold_calibration.pkl"
    summary_file = RESULTS_DIR / f"{dataset_name}_threshold_calibration.txt"

    with open(summary_file, 'w', encoding='utf-8') as summary:
        def log(text):
            summary.write(text + '\n')
            summary.flush()
            print(text)

        _print_section(f"THRESHOLD CALIBRATION: {dataset_name}")
        log(f"Dataset: {dataset_name}")
        log(f"Test samples: {len(splits.X_test)}")
        log(f"Sigma: {SIGMA}, K_NOISE_SEEDS: {K_NOISE_SEEDS}")

        # SUBSAMPLE TEST SET
        N_SAMPLES = min(MAX_SAMPLES, len(splits.X_test))
        if N_SAMPLES < len(splits.X_test):
            indices = np.random.RandomState(GLOBAL_SEED).choice(
                len(splits.X_test),
                N_SAMPLES,
                replace=False
            )
            X_test = splits.X_test[indices]
        else:
            X_test = splits.X_test

        # LOAD MODEL
        rf_uq = RandomForestClassifierUQ()
        rf_key = registry.make_key(dataset_key, rf_uq.name)
        if registry.exists(rf_key):
            rf_uq = registry.load(rf_key)
            log(f"OK Loaded model: {rf_uq.name}")
        else:
            log(f"WARN Model not found, skipping: {rf_key}")
            return

        # EPISTEMIC CLEAN
        _print_section("EPISTEMIC (CLEAN)")
        _, _, epistemic_clean = rf_uq.predict_with_uncertainty(X_test)
        log(f"Epistemic: mean={epistemic_clean.mean():.4f}, std={epistemic_clean.std():.4f}")

        # SHAP CLEAN
        _print_section("SHAP (CLEAN)")
        shap_explainer = SHAPExplainer(rf_uq.base_model)
        attr_clean = shap_explainer.explain(X_test)
        n_features = attr_clean.shape[1]
        top_k_used = min(TOP_K, n_features)
        log(f"OK SHAP clean: shape {attr_clean.shape}, TOP_K={top_k_used}")

        abs_attr_clean = np.abs(attr_clean)
        rank_clean = np.argsort(abs_attr_clean, axis=1)[:, ::-1]
        pos_clean = np.empty_like(rank_clean, dtype=float)
        pos_clean[np.arange(N_SAMPLES)[:, None], rank_clean] = np.arange(n_features)

        # STABILITY UNDER NOISE
        _print_section("STABILITY UNDER NOISE")
        stability_values = {metric: np.zeros(N_SAMPLES) for metric in STABILITY_METRICS.keys()}

        perturb_generators = [PerturbationGenerator(seed=GLOBAL_SEED + i) for i in range(K_NOISE_SEEDS)]
        for perturb in tqdm(perturb_generators, desc=dataset_name, leave=False):
            X_pert = perturb.gaussian(X_test, SIGMA)
            attr_pert = shap_explainer.explain(X_pert)

            for i in range(N_SAMPLES):
                rank_pert = np.argsort(np.abs(attr_pert[i]))[::-1]
                pos_pert = np.empty(n_features)
                pos_pert[rank_pert] = np.arange(n_features)

                kendall_val = _stat_corr(kendalltau(pos_clean[i], pos_pert))
                if np.isnan(kendall_val):
                    kendall_val = 0.0
                stability_values['kendall'][i] += kendall_val / K_NOISE_SEEDS

                spearman_val = _stat_corr(spearmanr(pos_clean[i], pos_pert))
                if np.isnan(spearman_val):
                    spearman_val = 0.0
                stability_values['spearman'][i] += spearman_val / K_NOISE_SEEDS

                top_clean = rank_clean[i][:top_k_used]
                top_pert = rank_pert[:top_k_used]
                intersection = np.intersect1d(top_clean, top_pert).size
                topk_overlap = intersection / top_k_used
                stability_values['topk_overlap'][i] += topk_overlap / K_NOISE_SEEDS

                union = (2 * top_k_used - intersection)
                jaccard = intersection / union if union > 0 else 0.0
                stability_values['jaccard'][i] += jaccard / K_NOISE_SEEDS

        results_by_metric = {}

        for metric, values in stability_values.items():
            _print_section(f"CALIBRATION: {metric.upper()}")
            thresholds = STABILITY_METRICS[metric]
            stable_thresh = thresholds['stable_threshold']
            unstable_thresh = thresholds['unstable_threshold']

            labels = np.where(values >= stable_thresh, 'stable',
                              np.where(values <= unstable_thresh, 'unstable', 'ambiguous'))
            n_stable = (labels == 'stable').sum()
            n_unstable = (labels == 'unstable').sum()
            n_ambiguous = (labels == 'ambiguous').sum()
            log(f"Stable: {n_stable}, Unstable: {n_unstable}, Ambiguous: {n_ambiguous}")

            mask = labels != 'ambiguous'
            if mask.sum() < 10:
                log("WARN Too few labeled samples, skipping metric.")
                results_by_metric[metric] = {
                    'skipped': True,
                    'reason': 'insufficient_samples'
                }
                continue

            epistemic_filtered = epistemic_clean[mask]
            stability_filtered = values[mask]
            labels_binary = (labels[mask] == 'stable').astype(int)

            if labels_binary.sum() == 0 or labels_binary.sum() == len(labels_binary):
                log("WARN Single-class after filtering, skipping metric.")
                results_by_metric[metric] = {
                    'skipped': True,
                    'reason': 'single_class'
                }
                continue

            fpr, tpr, roc_thresholds = roc_curve(labels_binary, -epistemic_filtered)
            roc_auc = auc(fpr, tpr)
            log(f"ROC AUC: {roc_auc:.3f}")

            threshold_results = []
            log("Coverage sweep:")
            log("  target  threshold   k   coverage  precision  recall   f1   stab_acc  stab_rej  note")

            epi_mean = epistemic_filtered.mean()
            epi_std = epistemic_filtered.std()

            for target_cov in TARGET_COVERAGES:
                threshold = np.quantile(epistemic_filtered, target_cov)
                predictions = (epistemic_filtered <= threshold).astype(int)
                coverage = predictions.mean()

                if predictions.sum() == 0 or predictions.sum() == len(predictions):
                    note = "all rejected" if predictions.sum() == 0 else "all accepted"
                    log(
                        f"  {target_cov:>6.0%} {threshold:>10.4f} "
                        f"{'n/a':>3} {coverage:>9.1%} "
                        f"{'n/a':>9} {'n/a':>7} {'n/a':>5} "
                        f"{'n/a':>9} {'n/a':>8} {note}"
                    )
                    continue

                precision = precision_score(labels_binary, predictions, zero_division=0)
                recall = recall_score(labels_binary, predictions, zero_division=0)
                f1 = f1_score(labels_binary, predictions, zero_division=0)

                accepted_mask = predictions == 1
                rejected_mask = predictions == 0
                mean_stab_acc = stability_filtered[accepted_mask].mean() if accepted_mask.sum() > 0 else 0
                mean_stab_rej = stability_filtered[rejected_mask].mean() if rejected_mask.sum() > 0 else 0

                k_value = (threshold - epi_mean) / epi_std if epi_std > 0 else 0.0

                log(
                    f"  {target_cov:>6.0%} {threshold:>10.4f} "
                    f"{k_value:>3.1f} {coverage:>9.1%} "
                    f"{precision:>9.3f} {recall:>7.3f} {f1:>5.3f} "
                    f"{mean_stab_acc:>9.3f} {mean_stab_rej:>8.3f}"
                )

                threshold_results.append({
                    'target_coverage': target_cov,
                    'threshold': threshold,
                    'k_value': k_value,
                    'coverage': coverage,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'mean_stability_accepted': mean_stab_acc,
                    'mean_stability_rejected': mean_stab_rej
                })

            if not threshold_results:
                log("WARN No valid thresholds found.")
                results_by_metric[metric] = {
                    'skipped': True,
                    'reason': 'no_valid_thresholds'
                }
                continue

            constrained = [
                result for result in threshold_results
                if MIN_COVERAGE <= result['coverage'] <= MAX_COVERAGE
            ]

            if constrained:
                best_result = max(constrained, key=lambda x: x['f1'])
                selection_note = f"{MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%} coverage"
            else:
                best_result = max(threshold_results, key=lambda x: x['f1'])
                selection_note = "no coverage constraint (fallback)"
                log(f"WARN No thresholds meet coverage in [{MIN_COVERAGE:.0%}, {MAX_COVERAGE:.0%}], using best F1 overall.")

            log(
                f"Best threshold ({selection_note}): "
                f"thr={best_result['threshold']:.4f}, k={best_result['k_value']:.2f}, "
                f"F1={best_result['f1']:.3f}, coverage={best_result['coverage']:.1%}"
            )

            results_by_metric[metric] = {
                'skipped': False,
                'n_filtered': int(mask.sum()),
                'n_stable': int(n_stable),
                'n_unstable': int(n_unstable),
                'stable_threshold': stable_thresh,
                'unstable_threshold': unstable_thresh,
                'roc_auc': float(roc_auc),
                'best_threshold': float(best_result['threshold']),
                'best_k_value': float(best_result['k_value']),
                'best_f1': float(best_result['f1']),
                'best_coverage': float(best_result['coverage']),
                'best_selection_note': selection_note,
                'best_mean_stab_accepted': float(best_result['mean_stability_accepted']),
                'best_mean_stab_rejected': float(best_result['mean_stability_rejected']),
                'threshold_results': threshold_results,
                'roc': {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds,
                    'auc': roc_auc
                }
            }

            GLOBAL_SUMMARY.append({
                'dataset': dataset_name,
                'metric': metric,
                'sigma': SIGMA,
                'roc_auc': roc_auc,
                'best_threshold': best_result['threshold'],
                'best_k_value': best_result['k_value'],
                'best_f1': best_result['f1'],
                'best_coverage': best_result['coverage'],
                'selection_note': selection_note
            })

        output = {
            'dataset': dataset_name,
            'config': {
                'sigma': SIGMA,
                'k_noise_seeds': K_NOISE_SEEDS,
                'target_coverages': TARGET_COVERAGES,
                'min_coverage': MIN_COVERAGE,
                'max_coverage': MAX_COVERAGE,
                'stability_metrics': STABILITY_METRICS,
                'top_k': top_k_used,
                'n_samples': N_SAMPLES,
            },
            'epistemic_clean': epistemic_clean,
            'results_by_metric': results_by_metric
        }

        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
        log(f"\nOK Results saved: {output_file}")
        log(f"OK Summary saved: {summary_file}")

for ds in datasets:
    run_dataset(ds)

summary_all_file = RESULTS_DIR / "all_datasets_threshold_calibration_summary.txt"

with open(summary_all_file, 'w', encoding='utf-8') as f:
    header = (f"{'Dataset':<12} {'Metric':>10} {'Sigma':>6} {'AUC':>7} "
              f"{'Thr':>8} {'k':>6} {'F1':>7} {'Coverage':>9} {'Note'}\n")
    f.write(header)
    f.write("-" * (len(header) - 1) + "\n")

    for row in GLOBAL_SUMMARY:
        f.write(
            f"{row['dataset']:<12} {row['metric']:>10} {row['sigma']:>6.2f} "
            f"{row['roc_auc']:>7.3f} {row['best_threshold']:>8.4f} {row['best_k_value']:>6.2f} "
            f"{row['best_f1']:>7.3f} {row['best_coverage']:>8.1%} {row['selection_note']}\n"
        )

print(f"\nOK Summary saved: {summary_all_file}")
