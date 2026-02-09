import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset
from data.splitter import DataSplitter
from models.registry import ModelRegistry

from uncertainty.linear_uq import LogisticUQ
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.mlp_uq import MLPClassifierUQ
from uncertainty.lgbm_uq import LightGBMClassifierUQ
from uncertainty.catboost_uq import CatBoostClassifierUQ

from explainers.shap_explainer import SHAPExplainer
from explainers.lime_explainer import LIMEExplainer
from explainers.gradient_explainer import IntegratedGradientsExplainer, SmoothGradExplainer


# SETUP

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

# TEST ON BOTH DATASETS
datasets = [WineQualityDataset(), DryBeanDataset()]

# SETUP OUTPUT FILE
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / (Path(__file__).stem + "_summary.txt")
output_file = open(OUTPUT_FILE, 'w', encoding='utf-8')

def log(text):
    """Print to console and write to file."""
    print(text)
    output_file.write(text + '\n')
    output_file.flush()


def test_explainer(model_name, explainer_name, explainer, X_test, feature_names, K=2):
    """Test a single explainer on clean data."""

    # SUBSAMPLE FOR SPEED
    n_samples = min(100, len(X_test))
    indices = np.random.RandomState(42).choice(len(X_test), n_samples, replace=False)
    X_sub = X_test[indices]

    try:
        # GENERATE EXPLANATIONS WITH STABILITY
        mean_attr, stability = explainer.explain_with_stability(X_sub, K=K)

        # STATS
        mean_abs = np.mean(np.abs(mean_attr))
        std_abs = np.std(np.abs(mean_attr))

        # TOP 3 FEATURES
        feature_imp = np.mean(np.abs(mean_attr), axis=0)
        top_idx = np.argsort(feature_imp)[::-1][:3]
        top_feats = [feature_names[i] for i in top_idx]

        return {
            'model': model_name,
            'explainer': explainer_name,
            'mean_abs_attr': mean_abs,
            'std_abs_attr': std_abs,
            'stability': stability,
            'top_features': top_feats
        }
    except Exception as e:
        log(f"  ERROR ({explainer_name}): {e}")
        return None


# MAIN

log("=" * 80)
log("EXPLAINER TESTS - BASELINE (CLEAN DATA)")
log("=" * 80)

all_results = []

for ds in datasets:
    info = cache.load_or_create(ds.cache_key, ds.load)
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    log("\n" + "=" * 80)
    log(f"DATASET: {info.name}")
    log(f"Train: {splits.X_train.shape}, Test: {splits.X_test.shape}")
    if info.class_names is not None:
        log(f"Classes: {len(info.class_names)} - {list(info.class_names)}")
    log("=" * 80)

    feature_names = info.feature_names

    # LOAD ALL UQ MODELS
    uq_models = {
        'Logistic': LogisticUQ(),
        'RF': RandomForestClassifierUQ(),
        'MLP': MLPClassifierUQ(),
        'LightGBM': LightGBMClassifierUQ(),
        'CatBoost': CatBoostClassifierUQ()
    }

    loaded_models = {}
    log("\nLoading UQ models...")
    for name, model in uq_models.items():
        key = registry.make_key(f"{info.name}_{ds.uci_id}", model.name)
        if registry.exists(key):
            loaded_models[name] = registry.load(key)
            log(f"  OK {name} UQ: loaded from cache")
        else:
            log(f"  FAIL {name} UQ: NOT CACHED (run test_models_uq.py)")

    if not loaded_models:
        log("  ERROR: No models loaded, skipping dataset")
        continue

# TEST EXPLAINERS FOR EACH MODEL

    for model_name, model in loaded_models.items():
        log(f"\n--- {model_name} UQ ---")
        log(f"{'Explainer':<20} | {'Mean |Attr|':>12} | {'Std |Attr|':>12} | {'Stability':>10} | Top-3 Features")
        log("-" * 95)

        # SHAP
        if model_name in ['RF', 'LightGBM', 'CatBoost']:
            # TREEEXPLAINER NO BACKGROUND NEEDED
            explainer = SHAPExplainer(model.base_model)
            result = test_explainer(model_name, 'SHAP (Tree)', explainer, splits.X_test, feature_names)
            if result:
                all_results.append(result)
                top_str = ", ".join(result['top_features'])
                log(f"{result['explainer']:<20} | {result['mean_abs_attr']:>12.4f} | {result['std_abs_attr']:>12.4f} | {result['stability']:>10.3f} | {top_str}")
        else:
            # KERNELEXPLAINER NEEDS BACKGROUND
            explainer = SHAPExplainer(model.base_model, X_background=splits.X_train[:100])
            result = test_explainer(model_name, 'SHAP (Kernel)', explainer, splits.X_test, feature_names)
            if result:
                all_results.append(result)
                top_str = ", ".join(result['top_features'])
                log(f"{result['explainer']:<20} | {result['mean_abs_attr']:>12.4f} | {result['std_abs_attr']:>12.4f} | {result['stability']:>10.3f} | {top_str}")

        # LIME
        explainer = LIMEExplainer(model.base_model, X_train=splits.X_train, feature_names=feature_names)
        result = test_explainer(model_name, 'LIME', explainer, splits.X_test, feature_names)
        if result:
            all_results.append(result)
            top_str = ", ".join(result['top_features'])
            log(f"{result['explainer']:<20} | {result['mean_abs_attr']:>12.4f} | {result['std_abs_attr']:>12.4f} | {result['stability']:>10.3f} | {top_str}")

        # GRADIENT EXPLAINERS MLP ONLY
        if model_name == 'MLP':
            explainer = IntegratedGradientsExplainer(model.base_model)
            result = test_explainer(model_name, 'IntegratedGradients', explainer, splits.X_test, feature_names)
            if result:
                all_results.append(result)
                top_str = ", ".join(result['top_features'])
                log(f"{result['explainer']:<20} | {result['mean_abs_attr']:>12.4f} | {result['std_abs_attr']:>12.4f} | {result['stability']:>10.3f} | {top_str}")

            explainer = SmoothGradExplainer(model.base_model)
            result = test_explainer(model_name, 'SmoothGrad', explainer, splits.X_test, feature_names)
            if result:
                all_results.append(result)
                top_str = ", ".join(result['top_features'])
                log(f"{result['explainer']:<20} | {result['mean_abs_attr']:>12.4f} | {result['std_abs_attr']:>12.4f} | {result['stability']:>10.3f} | {top_str}")

# SUMMARY

log("\n" + "=" * 80)
log("SUMMARY")
log("=" * 80)

log(f"\nTotal tests: {len(all_results)}")
log(f"Datasets: {len(datasets)}")
log(f"Models tested: {len(set(r['model'] for r in all_results))}")
log(f"Explainers tested: {len(set(r['explainer'] for r in all_results))}")

log("\n" + "=" * 80)
log("ALL RESULTS AGGREGATED")
log("=" * 80)
log(f"{'Model':<12} | {'Explainer':<20} | {'Mean |Attr|':>12} | {'Std |Attr|':>12} | {'Stability':>10} | Top Features")
log("-" * 110)

for r in all_results:
    top_str = ", ".join(r['top_features'])
    log(f"{r['model']:<12} | {r['explainer']:<20} | {r['mean_abs_attr']:>12.4f} | {r['std_abs_attr']:>12.4f} | {r['stability']:>10.3f} | {top_str}")

log("\n" + "=" * 80)

output_file.close()
print(f"\nOK Results saved to: {OUTPUT_FILE}")
