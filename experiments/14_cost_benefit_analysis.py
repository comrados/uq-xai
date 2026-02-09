"""Run a cost-benefit analysis for UQ-gated XAI.

Pools noisy samples across Gaussian sigmas, applies a single epistemic
threshold per coverage, and quantifies stability gains for RF (SHAP) and
MLP (LIME) explainers.
"""

import sys
import os
sys.path.append(os.getcwd())

import csv
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm

from data.cache import Cache
from data.datasets import (
    WineQualityDataset,
    DryBeanDataset,
    RiceDataset,
)
from data.splitter import DataSplitter
from data.perturbations import PerturbationGenerator
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from explainers.lime_explainer import LIMEExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.mlp_uq import MLPClassifierUQ
from config.settings import GLOBAL_SEED


# CONFIGURATION

np.random.seed(GLOBAL_SEED)

SIGMAS = np.round(np.arange(0.02, 0.21, 0.02), 2)
K_NOISE_SEEDS = 5
COVERAGES = [0.30, 0.50, 0.70, 1.00]
MAX_SAMPLES = 500
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "cost_benefit_summary.csv"


# HELPERS

def _stat_corr(stat_result):
    if hasattr(stat_result, "correlation"):
        return stat_result.correlation
    return stat_result[0]


def _kendall_stability(pos_clean, attr_pert):
    n_samples = attr_pert.shape[0]
    values = np.zeros(n_samples)

    for i in range(n_samples):
        rank_pert = np.argsort(np.abs(attr_pert[i]))[::-1]
        pos_pert = np.empty(rank_pert.shape[0])
        pos_pert[rank_pert] = np.arange(rank_pert.shape[0])
        val = _stat_corr(kendalltau(pos_clean[i], pos_pert))
        if np.isnan(val):
            val = 0.0
        values[i] = val

    return values


def _write_csv(rows, output_file):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows):
    if not rows:
        return
    headers = list(rows[0].keys())
    print("\n" + " | ".join(headers))
    print("-" * (len(headers) * 12))
    for row in rows:
        values = []
        for key in headers:
            val = row[key]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        print(" | ".join(values))


# MAIN

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    RiceDataset(),
]

rows = []

for ds in datasets:
    info = cache.load_or_create(ds.cache_key, ds.load)
    dataset_name = info.name
    dataset_key = f"{dataset_name}_{ds.uci_id}"
    splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

    n_samples = min(MAX_SAMPLES, len(splits.X_test))
    if n_samples < len(splits.X_test):
        indices = np.random.RandomState(GLOBAL_SEED).choice(len(splits.X_test), n_samples, replace=False)
        X_test = splits.X_test[indices]
    else:
        X_test = splits.X_test

    model_configs = {
        "RF": RandomForestClassifierUQ,
        "MLP": MLPClassifierUQ,
    }

    for model_name, model_cls in model_configs.items():
        model = model_cls()
        key = registry.make_key(dataset_key, model.name)
        if registry.exists(key):
            model = registry.load(key)
        else:
            if model_name == "MLP":
                model.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
            else:
                model.fit(splits.X_train, splits.y_train)
            registry.save(key, model)

        if model_name == "MLP":
            explainer = LIMEExplainer(
                model.base_model,
                X_train=splits.X_train,
                feature_names=info.feature_names,
            )
        else:
            explainer = SHAPExplainer(model.base_model)

        _, _, epistemic_clean = model.predict_with_uncertainty(X_test)
        attr_clean = explainer.explain(X_test)

        abs_attr_clean = np.abs(attr_clean)
        rank_clean = np.argsort(abs_attr_clean, axis=1)[:, ::-1]
        pos_clean = np.empty_like(rank_clean, dtype=float)
        pos_clean[np.arange(n_samples)[:, None], rank_clean] = np.arange(rank_clean.shape[1])

        baseline_row = {
            "dataset": dataset_name,
            "model": model_name,
            "coverage": "baseline_clean",
            "n_pool": int(n_samples),
            "n_explained": int(n_samples),
            "mean_tau": 1.0,
            "std_tau": 0.0,
        }
        rows.append(baseline_row)

        epistemic_pool = []
        stability_pool = []

        for sigma in SIGMAS:
            epistemic_sum = np.zeros(n_samples)
            stability_sum = np.zeros(n_samples)
            perturb_generators = [
                PerturbationGenerator(seed=GLOBAL_SEED + i) for i in range(K_NOISE_SEEDS)
            ]
            for perturb in tqdm(
                perturb_generators,
                desc=f"{dataset_name} {model_name} sigma={sigma:.2f}",
                leave=False,
            ):
                X_pert = perturb.gaussian(X_test, float(sigma))
                _, _, epistemic_noisy = model.predict_with_uncertainty(X_pert)
                attr_pert = explainer.explain(X_pert)
                stability_values = _kendall_stability(pos_clean, attr_pert)
                epistemic_sum += epistemic_noisy
                stability_sum += stability_values
            epistemic_pool.append(epistemic_sum / K_NOISE_SEEDS)
            stability_pool.append(stability_sum / K_NOISE_SEEDS)

        epistemic_pool = np.concatenate(epistemic_pool)
        stability_pool = np.concatenate(stability_pool)
        n_pool = int(epistemic_pool.shape[0])

        for coverage in COVERAGES:
            threshold = np.quantile(epistemic_pool, coverage)
            accepted_mask = epistemic_pool <= threshold
            accepted_tau = stability_pool[accepted_mask]
            n_explained = int(accepted_mask.sum())

            mean_tau = float(np.mean(accepted_tau)) if n_explained > 0 else np.nan
            std_tau = float(np.std(accepted_tau)) if n_explained > 0 else np.nan
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "coverage": float(coverage),
                "n_pool": n_pool,
                "n_explained": n_explained,
                "mean_tau": mean_tau,
                "std_tau": std_tau,
            })

_write_csv(rows, OUTPUT_FILE)
_print_table(rows)
print(f"\nSaved: {OUTPUT_FILE}")
