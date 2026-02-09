import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset
from data.splitter import DataSplitter
from data.adversarial import AdversarialPerturbationGenerator
from models.registry import ModelRegistry
from models.mlp import MLPClassifier
from evaluation.performance import ClassificationMetrics


# SETUP

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()
adv_gen = AdversarialPerturbationGenerator()

datasets = [WineQualityDataset(), DryBeanDataset()]


def evaluate_and_print(model, X_test, y_test, label=""):
    """Evaluate the model and print metrics."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    metrics = ClassificationMetrics.compute(y_test, preds, proba)
    print(f"  {label}Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    return metrics


# TEST ADVERSARIAL PERTURBATIONS

for ds in datasets:
    info = cache.load_or_create(ds.cache_key, ds.load)
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print("=" * 70)
    print(f"Dataset: {info.name} ({info.task_type})")
    print(f"Dataset UCI ID: {ds.uci_id}")
    print(f"Train: {splits.X_train.shape}, Test: {splits.X_test.shape}")

    # LOAD CACHED MLP MODEL

    mlp_model = MLPClassifier()

    dataset_key = f"{info.name}_{ds.uci_id}"
    mlp_key = registry.make_key(dataset_key, mlp_model.name)

    print(f"Dataset key: {dataset_key}")
    print(f"Model name: {mlp_model.name}")
    print(f"Full cache key: {mlp_key}")

    if not registry.exists(mlp_key):
        print(f"\nERROR: MLP model not found in cache: {mlp_key}")
        print(f"Expected path: {registry._path(mlp_key)}")
        print(f"Please train it first by running: python tests/test_models.py")
        continue

    mlp_model = registry.load(mlp_key)
    print(f"\nLoaded MLP model: {mlp_model.name} (device: {mlp_model.device})")

    # USE A SUBSET OF TEST DATA FOR FASTER TESTING
    test_size = min(500, len(splits.X_test))
    X_test_subset = splits.X_test[:test_size]
    y_test_subset = splits.y_test[:test_size]

    print(f"Testing on {test_size} samples")

    # BASELINE PERFORMANCE ON CLEAN DATA
    print("\n--- Clean Data ---")
    clean_metrics = evaluate_and_print(mlp_model, X_test_subset, y_test_subset)

    # BIM ATTACK

    print("\n=== BIM Attack ===")

    bim_params = [
        {"epsilon": 0.01, "num_iter": 10},
        {"epsilon": 0.05, "num_iter": 10},
        {"epsilon": 0.1, "num_iter": 10},
    ]

    for params in bim_params:
        epsilon = params["epsilon"]
        num_iter = params["num_iter"]

        print(f"\nBIM (eps={epsilon}, iter={num_iter}):")

        X_adv = adv_gen.bim(
            model=mlp_model.model,
            X=X_test_subset,
            y=y_test_subset,
            epsilon=epsilon,
            alpha=epsilon / num_iter * 2.5,
            num_iter=num_iter,
            task_type=info.task_type,
            device=mlp_model.device
        )

        # COMPUTE PERTURBATION STATISTICS
        perturbation = X_adv - X_test_subset
        print(f"  Perturbation: L2={np.linalg.norm(perturbation, axis=1).mean():.4f}, "
              f"L={np.abs(perturbation).max():.4f}")

        # EVALUATE ON ADVERSARIAL EXAMPLES
        evaluate_and_print(mlp_model, X_adv, y_test_subset, label="Adv ")

# PGD ATTACK

    print("\n=== PGD Attack ===")

    pgd_params = [
        {"epsilon": 0.01, "num_iter": 10},
        {"epsilon": 0.05, "num_iter": 10},
        {"epsilon": 0.1, "num_iter": 20},
    ]

    for params in pgd_params:
        epsilon = params["epsilon"]
        num_iter = params["num_iter"]

        print(f"\nPGD (eps={epsilon}, iter={num_iter}, random_start=True):")

        X_adv = adv_gen.pgd(
            model=mlp_model.model,
            X=X_test_subset,
            y=y_test_subset,
            epsilon=epsilon,
            alpha=epsilon / num_iter * 2.5,
            num_iter=num_iter,
            task_type=info.task_type,
            device=mlp_model.device,
            random_start=True
        )

        # COMPUTE PERTURBATION STATISTICS
        perturbation = X_adv - X_test_subset
        print(f"  Perturbation: L2={np.linalg.norm(perturbation, axis=1).mean():.4f}, "
              f"L={np.abs(perturbation).max():.4f}")

        # EVALUATE ON ADVERSARIAL EXAMPLES
        evaluate_and_print(mlp_model, X_adv, y_test_subset, label="Adv ")

# C W ATTACK

    print("\n=== C&W Attack ===")

    # USE EVEN SMALLER SUBSET FOR C W IT S SLOWER
    cw_size = min(100, test_size)
    X_test_cw = X_test_subset[:cw_size]
    y_test_cw = y_test_subset[:cw_size]

    print(f"Testing C&W on {cw_size} samples (slower attack)")

    cw_params = [
        {"c": 0.1, "num_iter": 50, "learning_rate": 0.01},
        {"c": 1.0, "num_iter": 50, "learning_rate": 0.01},
        {"c": 10.0, "num_iter": 100, "learning_rate": 0.01},
    ]

    for params in cw_params:
        c = params["c"]
        num_iter = params["num_iter"]
        lr = params["learning_rate"]

        print(f"\nC&W (c={c}, iter={num_iter}, lr={lr}):")

        X_adv = adv_gen.cw(
            model=mlp_model.model,
            X=X_test_cw,
            y=y_test_cw,
            c=c,
            kappa=0.0,
            num_iter=num_iter,
            learning_rate=lr,
            task_type=info.task_type,
            device=mlp_model.device,
            targeted=False
        )

        # COMPUTE PERTURBATION STATISTICS
        perturbation = X_adv - X_test_cw
        print(f"  Perturbation: L2={np.linalg.norm(perturbation, axis=1).mean():.4f}, "
              f"L={np.abs(perturbation).max():.4f}")

        # EVALUATE ON ADVERSARIAL EXAMPLES
        evaluate_and_print(mlp_model, X_adv, y_test_cw, label="Adv ")

# TEST GENERIC PERTURB METHOD

    print("\n=== Generic perturb() method ===")

    methods_to_test = [
        {"method": "bim", "epsilon": 0.05},
        {"method": "pgd", "epsilon": 0.05},
        {"method": "cw", "c": 1.0},
    ]

    for test_case in methods_to_test:
        method = test_case["method"]

        if method in ["bim", "pgd"]:
            epsilon = test_case["epsilon"]
            print(f"\n{method.upper()} via perturb() (eps={epsilon}):")

            X_adv = adv_gen.perturb(
                model=mlp_model.model,
                X=X_test_subset[:100],
                y=y_test_subset[:100],
                method=method,
                task_type=info.task_type,
                device=mlp_model.device,
                epsilon=epsilon,
                num_iter=10
            )
        else:
            c = test_case["c"]
            print(f"\n{method.upper()} via perturb() (c={c}):")

            X_adv = adv_gen.perturb(
                model=mlp_model.model,
                X=X_test_subset[:100],
                y=y_test_subset[:100],
                method=method,
                task_type=info.task_type,
                device=mlp_model.device,
                c=c,
                num_iter=50
            )

        perturbation = X_adv - X_test_subset[:100]
        print(f"  Perturbation: L2={np.linalg.norm(perturbation, axis=1).mean():.4f}, "
              f"L={np.abs(perturbation).max():.4f}")

        evaluate_and_print(mlp_model, X_adv, y_test_subset[:100], label="Adv ")

# TEST CACHE KEY GENERATION

    print("\n=== Cache Key Generation ===")

    key1 = adv_gen.make_key("bim", epsilon=0.1)
    key2 = adv_gen.make_key("pgd", epsilon=0.05)
    key3 = adv_gen.make_key("cw", c=1.0)

    print(f"  BIM key: {key1}")
    print(f"  PGD key: {key2}")
    print(f"  C&W key: {key3}")

    print()

print("\n" + "=" * 70)
print("All adversarial perturbation tests completed!")
print("=" * 70)
