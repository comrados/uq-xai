"""Run a perturbation sweep for explainers.

Evaluates explainer stability (rank correlation, attribution variance) for
SHAP/LIME/IG/SmoothGrad under data perturbations and adversarial attacks.
Results are cached per dataset with a text summary.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import pickle

from data.cache import Cache
from data.datasets import (
    WineQualityDataset,
    DryBeanDataset,
    IrisDataset,
    RiceDataset,
    EcoliDataset,
)
from data.splitter import DataSplitter
from data.perturbations import PerturbationGenerator
from data.adversarial import AdversarialPerturbationGenerator
from models.registry import ModelRegistry

from uncertainty.linear_uq import LogisticUQ
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.mlp_uq import MLPClassifierUQ
from uncertainty.lgbm_uq import LightGBMClassifierUQ
from uncertainty.catboost_uq import CatBoostClassifierUQ

from explainers.shap_explainer import SHAPExplainer
from explainers.lime_explainer import LIMEExplainer
from explainers.gradient_explainer import IntegratedGradientsExplainer, SmoothGradExplainer

from evaluation.explanation import ExplanationMetrics

from config.settings import PERTURBATION_CONFIG, ADVERSARIAL_CONFIG


# SETUP

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()
perturb = PerturbationGenerator()
adv_gen = AdversarialPerturbationGenerator()

# TEST ON MULTIPLE DATASETS
datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    IrisDataset(),
    RiceDataset(),
    EcoliDataset(),
]

# PERTURBATION LEVELS FROM CONFIG
gaussian_levels = PERTURBATION_CONFIG['gaussian']
missing_levels = PERTURBATION_CONFIG['missing']
permutation_levels = PERTURBATION_CONFIG['permutation']

# ADVERSARIAL LEVELS
bim_levels = [0.0] + ADVERSARIAL_CONFIG['bim']['epsilons']
pgd_levels = [0.0] + ADVERSARIAL_CONFIG['pgd']['epsilons']
cw_levels = [0.0] + ADVERSARIAL_CONFIG['cw']['c_values']

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


def make_result_key(model_name, explainer_name, perturbation_type, level):
    """Canonical key for checking cached results."""
    return (model_name, explainer_name, perturbation_type, float(level))


def test_perturbation(model_name, explainer_name, explainer, X_test, perturbation_type, level, dataset_name):
    """Test a single explainer on perturbed data."""

    # SUBSAMPLE FOR SPEED
    n_samples = min(100, len(X_test))
    indices = np.random.RandomState(42).choice(len(X_test), n_samples, replace=False)
    X_sub = X_test[indices]

    try:
        # CLEAN EXPLANATIONS
        attr_clean = explainer.explain(X_sub)

        if level == 0.0:
            # CLEAN DATA
            rank_corr = 1.0
            attr_var = 0.0
        else:
            # APPLY PERTURBATION
            if perturbation_type == 'gaussian':
                X_pert = perturb.gaussian(X_sub, level)
            elif perturbation_type == 'missing':
                X_pert = perturb.missing(X_sub, level)
            elif perturbation_type == 'permutation':
                X_pert = perturb.permute(X_sub, level)
            else:
                raise ValueError(f"Unknown perturbation: {perturbation_type}")

            # PERTURBED EXPLANATIONS
            attr_pert = explainer.explain(X_pert)

            # COMPUTE DEGRADATION
            rank_corr = ExplanationMetrics.rank_correlation_under_perturbation(attr_clean, attr_pert)
            attr_var = ExplanationMetrics.attribution_variance([attr_clean, attr_pert])

        return {
            'dataset': dataset_name,
            'model': model_name,
            'explainer': explainer_name,
            'perturbation': perturbation_type,
            'level': level,
            'rank_correlation': rank_corr,
            'attribution_variance': attr_var
        }

    except Exception as e:
        log(f"    ERROR ({model_name}+{explainer_name}, level={level}): {e}")
        return None


def test_adversarial(model_name, explainer_name, explainer, info, ds, splits, attack_type, level, dataset_name):
    """Test an explainer on adversarial data."""

    # SUBSAMPLE
    n_samples = min(100, len(splits.X_test))
    indices = np.random.RandomState(42).choice(len(splits.X_test), n_samples, replace=False)
    X_sub = splits.X_test[indices]

    try:
        # CLEAN EXPLANATIONS
        attr_clean = explainer.explain(X_sub)

        if level == 0.0:
            rank_corr = 1.0
            attr_var = 0.0
        else:
            # LOAD ADVERSARIAL DATA
            if attack_type in ['bim', 'pgd']:
                perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(attack_type, epsilon=level)}"
            elif attack_type == 'cw':
                perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(attack_type, c=level)}"
            else:
                raise ValueError(f"Unknown attack: {attack_type}")

            if cache.exists(perturb_key):
                X_adv_full = cache.load(perturb_key)
                X_adv = X_adv_full[indices]

                # ADVERSARIAL EXPLANATIONS
                attr_adv = explainer.explain(X_adv)

                # COMPUTE DEGRADATION
                rank_corr = ExplanationMetrics.rank_correlation_under_perturbation(attr_clean, attr_adv)
                attr_var = ExplanationMetrics.attribution_variance([attr_clean, attr_adv])
            else:
                log(f"    WARNING: Adversarial data not cached for {attack_type} eps={level} (run test_data_adversarial.py)")
                return None

        return {
            'dataset': dataset_name,
            'model': model_name,
            'explainer': explainer_name,
            'perturbation': f'adversarial_{attack_type}',
            'level': level,
            'rank_correlation': rank_corr,
            'attribution_variance': attr_var
        }

    except Exception as e:
        log(f"    ERROR ({model_name}+{explainer_name}, level={level}): {e}")
        return None


# MAIN

log("=" * 80)
log("EXPLAINER TESTS - PERTURBATIONS")
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

    # DATASET SPECIFIC RESULTS LIST FOR PICKLE SAVING
    results_file = RESULTS_DIR / f"explainers_perturbations_{info.name}_{ds.uci_id}.pkl"
    dataset_results = []
    existing_keys = set()

    if results_file.exists():
        log(f"\nOK Loading cached results from: {results_file}")
        with open(results_file, 'rb') as f:
            dataset_results = pickle.load(f)
        for r in dataset_results:
            existing_keys.add(make_result_key(r['model'], r['explainer'], r['perturbation'], r['level']))
        all_results.extend(dataset_results)
        log(f"  Loaded {len(dataset_results)} cached results, will run only missing experiments")
    else:
        log(f"\nNo cached results found for dataset, running all experiments")

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
            log(f"  OK {name} UQ: loaded")
        else:
            log(f"  FAIL {name} UQ: NOT CACHED")

    if not loaded_models:
        log("  ERROR: No models loaded, skipping dataset")
        continue

    # DEFINE EXPLAINER CONFIGURATIONS FOR EACH MODEL
    explainer_configs = []

    for model_name, model in loaded_models.items():
        if model_name in ['RF', 'LightGBM', 'CatBoost']:
            # TREE MODELS SHAP TREE LIME
            explainer_configs.append((model_name, model, 'SHAP (Tree)', SHAPExplainer(model.base_model)))
            explainer_configs.append((model_name, model, 'LIME', LIMEExplainer(model.base_model, X_train=splits.X_train, feature_names=info.feature_names)))
        elif model_name == 'MLP':
            # MLP SHAP KERNEL LIME INTGRAD SMOOTHGRAD
            explainer_configs.append((model_name, model, 'SHAP (Kernel)', SHAPExplainer(model.base_model, X_background=splits.X_train[:100])))
            explainer_configs.append((model_name, model, 'LIME', LIMEExplainer(model.base_model, X_train=splits.X_train, feature_names=info.feature_names)))
            explainer_configs.append((model_name, model, 'IntGrad', IntegratedGradientsExplainer(model.base_model)))
            explainer_configs.append((model_name, model, 'SmoothGrad', SmoothGradExplainer(model.base_model)))
        else:
            # LOGISTIC SHAP KERNEL LIME
            explainer_configs.append((model_name, model, 'SHAP (Kernel)', SHAPExplainer(model.base_model, X_background=splits.X_train[:100])))
            explainer_configs.append((model_name, model, 'LIME', LIMEExplainer(model.base_model, X_train=splits.X_train, feature_names=info.feature_names)))

# TEST 1 GAUSSIAN NOISE

    log("\n" + "=" * 70)
    log("PERTURBATION: GAUSSIAN NOISE")
    log("=" * 70)

    # COLLECT RESULTS FIRST WITH TQDM THEN PRINT DETAILED TABLES
    gaussian_results = []
    for model_name, model, explainer_name, explainer in tqdm(explainer_configs, desc="Gaussian", leave=False):
        for level in gaussian_levels:
            key = make_result_key(model_name, explainer_name, 'gaussian', level)
            if key in existing_keys:
                continue
            result = test_perturbation(model_name, explainer_name, explainer, splits.X_test, 'gaussian', level, info.name)
            if result:
                all_results.append(result)
                dataset_results.append(result)
                existing_keys.add(key)
                gaussian_results.append(result)

    # PRINT DETAILED TABLE GROUPED BY MODEL EXPLAINER
    for model_name in loaded_models.keys():
        model_results = [r for r in gaussian_results if r['model'] == model_name]
        if model_results:
            log(f"\n--- {model_name} ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for explainer_name in sorted(set(r['explainer'] for r in model_results)):
                exp_results = [r for r in model_results if r['explainer'] == explainer_name]
                for r in exp_results:
                    log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

# TEST 2 MISSING DATA

    log("\n" + "=" * 70)
    log("PERTURBATION: MISSING DATA")
    log("=" * 70)

    missing_results = []
    for model_name, model, explainer_name, explainer in tqdm(explainer_configs, desc="Missing", leave=False):
        for level in missing_levels:
            key = make_result_key(model_name, explainer_name, 'missing', level)
            if key in existing_keys:
                continue
            result = test_perturbation(model_name, explainer_name, explainer, splits.X_test, 'missing', level, info.name)
            if result:
                all_results.append(result)
                dataset_results.append(result)
                existing_keys.add(key)
                missing_results.append(result)

    # PRINT DETAILED TABLE
    for model_name in loaded_models.keys():
        model_results = [r for r in missing_results if r['model'] == model_name]
        if model_results:
            log(f"\n--- {model_name} ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for explainer_name in sorted(set(r['explainer'] for r in model_results)):
                exp_results = [r for r in model_results if r['explainer'] == explainer_name]
                for r in exp_results:
                    log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

# TEST 3 FEATURE PERMUTATION

    log("\n" + "=" * 70)
    log("PERTURBATION: FEATURE PERMUTATION")
    log("=" * 70)

    permutation_results = []
    for model_name, model, explainer_name, explainer in tqdm(explainer_configs, desc="Permutation", leave=False):
        for level in permutation_levels:
            key = make_result_key(model_name, explainer_name, 'permutation', level)
            if key in existing_keys:
                continue
            result = test_perturbation(model_name, explainer_name, explainer, splits.X_test, 'permutation', level, info.name)
            if result:
                all_results.append(result)
                dataset_results.append(result)
                existing_keys.add(key)
                permutation_results.append(result)

    # PRINT DETAILED TABLE
    for model_name in loaded_models.keys():
        model_results = [r for r in permutation_results if r['model'] == model_name]
        if model_results:
            log(f"\n--- {model_name} ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for explainer_name in sorted(set(r['explainer'] for r in model_results)):
                exp_results = [r for r in model_results if r['explainer'] == explainer_name]
                for r in exp_results:
                    log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

# TEST 4 ADVERSARIAL MLP ONLY

    if 'MLP' in loaded_models:
        mlp_model = loaded_models['MLP']
        mlp_background = splits.X_train[:100]

        # TEST BOTH INTGRAD AND SMOOTHGRAD ON ADVERSARIAL
        mlp_explainers = [
            ('SHAP (Kernel)', SHAPExplainer(mlp_model.base_model, X_background=mlp_background)),
            ('LIME', LIMEExplainer(mlp_model.base_model, X_train=splits.X_train, feature_names=info.feature_names)),
            ('IntGrad', IntegratedGradientsExplainer(mlp_model.base_model)),
            ('SmoothGrad', SmoothGradExplainer(mlp_model.base_model))
        ]

        # BIM
        log("\n" + "=" * 70)
        log("ADVERSARIAL ATTACK: BIM (Basic Iterative Method)")
        log("=" * 70)

        bim_results = []
        for explainer_name, explainer in mlp_explainers:
            for level in tqdm(bim_levels, desc=f"BIM-{explainer_name}", leave=False):
                pert_key = 'adversarial_bim'
                key = make_result_key('MLP', explainer_name, pert_key, level)
                if key in existing_keys:
                    continue
                result = test_adversarial('MLP', explainer_name, explainer, info, ds, splits, 'bim', level, info.name)
                if result:
                    all_results.append(result)
                    dataset_results.append(result)
                    existing_keys.add(key)
                    bim_results.append(result)

        if bim_results:
            log(f"\n--- MLP ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for r in bim_results:
                log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

        # PGD
        log("\n" + "=" * 70)
        log("ADVERSARIAL ATTACK: PGD (Projected Gradient Descent)")
        log("=" * 70)

        pgd_results = []
        for explainer_name, explainer in mlp_explainers:
            for level in tqdm(pgd_levels, desc=f"PGD-{explainer_name}", leave=False):
                pert_key = 'adversarial_pgd'
                key = make_result_key('MLP', explainer_name, pert_key, level)
                if key in existing_keys:
                    continue
                result = test_adversarial('MLP', explainer_name, explainer, info, ds, splits, 'pgd', level, info.name)
                if result:
                    all_results.append(result)
                    dataset_results.append(result)
                    existing_keys.add(key)
                    pgd_results.append(result)

        if pgd_results:
            log(f"\n--- MLP ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for r in pgd_results:
                log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

        # C W
        log("\n" + "=" * 70)
        log("ADVERSARIAL ATTACK: C&W (Carlini & Wagner)")
        log("=" * 70)

        cw_results = []
        for explainer_name, explainer in mlp_explainers:
            for level in tqdm(cw_levels, desc=f"C&W-{explainer_name}", leave=False):
                pert_key = 'adversarial_cw'
                key = make_result_key('MLP', explainer_name, pert_key, level)
                if key in existing_keys:
                    continue
                result = test_adversarial('MLP', explainer_name, explainer, info, ds, splits, 'cw', level, info.name)
                if result:
                    all_results.append(result)
                    dataset_results.append(result)
                    existing_keys.add(key)
                    cw_results.append(result)

        if cw_results:
            log(f"\n--- MLP ---")
            log(f"{'Explainer':<20} | {'Level':>7} | {'Rank Corr':>10} | {'Attr Var':>10}")
            log("-" * 55)
            for r in cw_results:
                log(f"{r['explainer']:<20} | {r['level']:>7.2f} | {r['rank_correlation']:>10.3f} | {r['attribution_variance']:>10.4f}")

    # SAVE DATASET SPECIFIC PICKLE
    with open(results_file, 'wb') as f:
        pickle.dump(dataset_results, f)
    log(f"\nOK Dataset pickle saved to: {results_file}")


# SUMMARY

log("\n" + "=" * 80)
log("SUMMARY")
log("=" * 80)

log(f"\nTotal tests: {len(all_results)}")
log(f"Datasets: {len(datasets)}")
log(f"Perturbation types: gaussian, missing, permutation, adversarial (BIM, PGD, C&W)")

# GROUP RESULTS BY DATASET
datasets_in_results = sorted(set(r.get('dataset', 'Unknown') for r in all_results))

# SHOW DEGRADATION BY PERTURBATION TYPE
for pert_type in ['gaussian', 'missing', 'permutation', 'adversarial_bim', 'adversarial_pgd', 'adversarial_cw']:
    pert_results = [r for r in all_results if r['perturbation'] == pert_type]
    if pert_results:
        log(f"\n{pert_type.upper()}:")
        max_level = max(r['level'] for r in pert_results if r['level'] > 0) if any(r['level'] > 0 for r in pert_results) else 0
        if max_level > 0:
            max_results = [r for r in pert_results if r['level'] == max_level]
            # GROUP BY DATASET FOR BETTER ORGANIZATION
            for dataset_name in sorted(set(r.get('dataset', 'Unknown') for r in max_results)):
                dataset_results = [r for r in max_results if r.get('dataset', 'Unknown') == dataset_name]
                for r in dataset_results:
                    log(f"  [{dataset_name}] {r['model']}+{r['explainer']} (level={r['level']:.2f}): Rank Corr={r['rank_correlation']:.3f}, Var={r['attribution_variance']:.4f}")

log("\n" + "=" * 80)

output_file.close()
print(f"\nOK Text summary saved to: {OUTPUT_FILE}")
print(f"OK Dataset-specific pickles saved to: {RESULTS_DIR}")
