"""Run a perturbation sweep for uncertainty metrics.

Measures how UQ metrics and error change under gaussian/missing/permutation
noise and adversarial attacks (MLP only), and saves per-dataset results.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

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

from config.settings import PERTURBATION_CONFIG, ADVERSARIAL_CONFIG


cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()
perturb = PerturbationGenerator()
adv_gen = AdversarialPerturbationGenerator()


def load_uq_models(info, ds, splits):
    """Load or train all UQ models for classification."""
    models = {}

    model_classes = {
        'Logistic': LogisticUQ,
        'RF': RandomForestClassifierUQ,
        'MLP': MLPClassifierUQ,
        'LightGBM': LightGBMClassifierUQ,
        'CatBoost': CatBoostClassifierUQ
    }
    
    for name, cls in model_classes.items():
        model = cls()
        key = registry.make_key(f"{info.name}_{ds.uci_id}", model.name)
        
        if registry.exists(key):
            models[name] = registry.load(key)
        else:
            if name in ['MLP', 'LightGBM', 'CatBoost']:
                model.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
            else:
                model.fit(splits.X_train, splits.y_train)
            registry.save(key, model)
            models[name] = model
    
    return models


def compute_metrics(model, X_test, y_test):
    """Compute predictions and uncertainty for classification."""
    preds, aleatoric, epistemic = model.predict_with_uncertainty(X_test)

    # Classification: total = aleatoric + epistemic
    total = aleatoric + epistemic
    error = 1 - np.mean(preds == y_test)
    error_name = "Error"

    # GET PREDICTION PROBABILITIES FOR CONFIDENCE
    y_proba = model.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)

    ale_mean, ale_std = np.mean(aleatoric), np.std(aleatoric)
    epi_mean, epi_std = np.mean(epistemic), np.std(epistemic)
    tot_mean, tot_std = np.mean(total), np.std(total)
    conf_mean, conf_std = np.mean(confidence), np.std(confidence)

    return {
        'error': error,
        'error_name': error_name,
        'aleatoric': ale_mean,
        'aleatoric_std': ale_std,
        'epistemic': epi_mean,
        'epistemic_std': epi_std,
        'total': tot_mean,
        'total_std': tot_std,
        'confidence': conf_mean,
        'confidence_std': conf_std
    }


def run_perturbation_test(models, splits, perturbation_type, levels):
    """Run perturbation tests for all models."""
    results = []

    # Subsample for consistency with explainer tests (same 100 samples, seed=42)
    n_samples = min(100, len(splits.X_test))
    indices = np.random.RandomState(42).choice(len(splits.X_test), n_samples, replace=False)
    X_test_sub = splits.X_test[indices]
    y_test_sub = splits.y_test[indices]

    print(f"\n{'Level':>7} | {'Error':>6} | {'Aleatoric mu±sigma':>15} | {'Epistemic mu±sigma':>15} | {'Total mu±sigma':>15} | {'Confidence mu±sigma':>15}")
    print("-" * 90)

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        for level in levels:
            if level == 0.0:
                X_test_pert = X_test_sub
            elif perturbation_type == 'gaussian':
                X_test_pert = perturb.gaussian(X_test_sub, level)
            elif perturbation_type == 'missing':
                X_test_pert = perturb.missing(X_test_sub, level)
            elif perturbation_type == 'permutation':
                X_test_pert = perturb.permute(X_test_sub, level)

            metrics = compute_metrics(model, X_test_pert, y_test_sub)

            ale_str = f"{metrics['aleatoric']:.5f}±{metrics['aleatoric_std']:.5f}"
            epi_str = f"{metrics['epistemic']:.5f}±{metrics['epistemic_std']:.5f}"
            tot_str = f"{metrics['total']:.5f}±{metrics['total_std']:.5f}"
            conf_str = f"{metrics['confidence']:.5f}±{metrics['confidence_std']:.5f}"
            print(f"{level:>7.2f} | {metrics['error']:>6.3f} | {ale_str:>15} | {epi_str:>15} | {tot_str:>15} | {conf_str:>15}")
            
            results.append({
                'model': model_name,
                'perturbation': perturbation_type,
                'level': level,
                'error': metrics['error'],
                'aleatoric_std': metrics['aleatoric_std'],
                'epistemic_std': metrics['epistemic_std'],
                'total_std': metrics['total_std'],
                'aleatoric': metrics['aleatoric'],
                'epistemic': metrics['epistemic'],
                'total': metrics['total'],
                'confidence': metrics['confidence'],
                'confidence_std': metrics['confidence_std']
            })
    
    return results


def run_adversarial_test(mlp_model, info, ds, splits, attack_type, levels):
    """Run adversarial perturbation tests for the MLP model only."""
    results = []

    # Subsample for consistency with explainer tests (same 100 samples, seed=42)
    n_samples = min(100, len(splits.X_test))
    indices = np.random.RandomState(42).choice(len(splits.X_test), n_samples, replace=False)
    y_test_sub = splits.y_test[indices]

    print(f"\n{'Level':>7} | {'Error':>6} | {'Aleatoric mu±sigma':>15} | {'Epistemic mu±sigma':>15} | {'Total mu±sigma':>15} | {'Confidence mu±sigma':>15}")
    print("-" * 90)

    print(f"\n--- MLP (adversarial: {attack_type.upper()}) ---")

    for level in levels:
        if level == 0.0:
            X_test_pert = splits.X_test[indices]
        else:
            # LOAD FROM CACHE
            if attack_type in ['bim', 'pgd']:
                perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(attack_type, epsilon=level)}"
            else:  # cw
                perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(attack_type, c=level)}"

            if cache.exists(perturb_key):
                X_test_pert_full = cache.load(perturb_key)
                X_test_pert = X_test_pert_full[indices]  # Subsample adversarial data
            else:
                print(f"  WARNING: Adversarial data not cached for {attack_type} level={level}")
                print(f"  Run: python tests/test_data_adversarial.py")
                continue

        metrics = compute_metrics(mlp_model, X_test_pert, y_test_sub)

        ale_str = f"{metrics['aleatoric']:.5f}±{metrics['aleatoric_std']:.5f}"
        epi_str = f"{metrics['epistemic']:.5f}±{metrics['epistemic_std']:.5f}"
        tot_str = f"{metrics['total']:.5f}±{metrics['total_std']:.5f}"
        conf_str = f"{metrics['confidence']:.5f}±{metrics['confidence_std']:.5f}"
        print(f"{level:>7.2f} | {metrics['error']:>6.3f} | {ale_str:>15} | {epi_str:>15} | {tot_str:>15} | {conf_str:>15}")

        results.append({
            'model': 'MLP',
            'perturbation': attack_type,
            'level': level,
            'error': metrics['error'],
            'aleatoric_std': metrics['aleatoric_std'],
            'epistemic_std': metrics['epistemic_std'],
            'total_std': metrics['total_std'],
            'aleatoric': metrics['aleatoric'],
            'epistemic': metrics['epistemic'],
            'total': metrics['total'],
            'confidence': metrics['confidence'],
            'confidence_std': metrics['confidence_std']
        })

    return results


def check_monotonicity(df, models, include_adversarial=False):
    """Check whether uncertainty increases with perturbation."""
    print("\n" + "=" * 70)
    print("SUMMARY: Does uncertainty increase with perturbation?")
    print("=" * 70)

    for model_name in models.keys():
        print(f"\n{model_name}:")

        pert_types = ['gaussian', 'missing', 'permutation']
        if include_adversarial and model_name == 'MLP':
            pert_types.extend(['bim', 'pgd', 'cw'])

        for pert_type in pert_types:
            subset = df[(df['model'] == model_name) & (df['perturbation'] == pert_type)]
            subset = subset.sort_values('level')

            if len(subset) == 0:
                continue

            total_values = subset['total'].values
            is_monotonic = all(total_values[i] <= total_values[i+1] * 1.01 for i in range(len(total_values)-1))

            trend = "INCREASING" if is_monotonic else "NOT MONOTONIC"
            print(f"  {pert_type:12}: {trend:13} ({total_values[0]:.5f} -> {total_values[-1]:.5f})")


# MAIN

datasets = [
    (WineQualityDataset(), "wine_binary"),  # Binary classification
    (DryBeanDataset(), "bean"),             # Multi-class classification
    (IrisDataset(), "iris"),
    (RiceDataset(), "rice"),
    (EcoliDataset(), "ecoli"),
]

gaussian_levels = PERTURBATION_CONFIG['gaussian']
missing_levels = PERTURBATION_CONFIG['missing']
permute_levels = PERTURBATION_CONFIG['permutation']

# ADVERSARIAL LEVELS
bim_levels = [0.0] + ADVERSARIAL_CONFIG['bim']['epsilons']
pgd_levels = [0.0] + ADVERSARIAL_CONFIG['pgd']['epsilons']
cw_levels = [0.0] + ADVERSARIAL_CONFIG['cw']['c_values']


for ds, dataset_id in datasets:
    info = cache.load_or_create(ds.cache_key, ds.load)
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print("\n" + "=" * 70)
    print(f"DATASET: {info.name} (CLASSIFICATION)")
    print(f"Train: {splits.X_train.shape}, Test: {splits.X_test.shape}")
    if info.class_names is not None:
        print(f"Classes: {len(info.class_names)} - {list(info.class_names)}")
    print("=" * 70)

    models = load_uq_models(info, ds, splits)
    
    all_results = []
    
    # GAUSSIAN NOISE
    print("\n" + "=" * 70)
    print("PERTURBATION: GAUSSIAN NOISE")
    print("=" * 70)
    results = run_perturbation_test(models, splits, 'gaussian', gaussian_levels)
    all_results.extend(results)
    
    # MISSING DATA
    print("\n" + "=" * 70)
    print("PERTURBATION: MISSING DATA")
    print("=" * 70)
    results = run_perturbation_test(models, splits, 'missing', missing_levels)
    all_results.extend(results)
    
    # FEATURE PERMUTATION
    print("\n" + "=" * 70)
    print("PERTURBATION: FEATURE PERMUTATION")
    print("=" * 70)
    results = run_perturbation_test(models, splits, 'permutation', permute_levels)
    all_results.extend(results)

    # ADVERSARIAL ATTACKS MLP ONLY
    if 'MLP' in models:
        mlp_model = models['MLP']

        # BIM ATTACK
        print("\n" + "=" * 70)
        print("ADVERSARIAL ATTACK: BIM (Basic Iterative Method)")
        print("=" * 70)
        results = run_adversarial_test(mlp_model, info, ds, splits, 'bim', bim_levels)
        all_results.extend(results)

        # PGD ATTACK
        print("\n" + "=" * 70)
        print("ADVERSARIAL ATTACK: PGD (Projected Gradient Descent)")
        print("=" * 70)
        results = run_adversarial_test(mlp_model, info, ds, splits, 'pgd', pgd_levels)
        all_results.extend(results)

        # C W ATTACK
        print("\n" + "=" * 70)
        print("ADVERSARIAL ATTACK: C&W (Carlini & Wagner)")
        print("=" * 70)
        results = run_adversarial_test(mlp_model, info, ds, splits, 'cw', cw_levels)
        all_results.extend(results)

    # SAVE RESULTS TO PICKLE
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_file = RESULTS_DIR / f"perturbation_uncertainty_{info.name}_{ds.uci_id}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"OK Results saved to: {results_file}")

    # SUMMARY
    df = pd.DataFrame(all_results)
    check_monotonicity(df, models, include_adversarial=True)

    print("\n")
