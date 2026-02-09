"""Generate adversarial test sets.

Uses a cached MLP classifier to create BIM/PGD/CW adversarial perturbations
for each dataset and stores them in cache with basic magnitude statistics.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, IrisDataset, RiceDataset, EcoliDataset
from data.splitter import DataSplitter
from data.adversarial import AdversarialPerturbationGenerator
from models.registry import ModelRegistry
from models.mlp import MLPClassifier
from config.settings import ADVERSARIAL_CONFIG
from ucimlrepo.fetch import DatasetNotFoundError

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()
adv_gen = AdversarialPerturbationGenerator()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    IrisDataset(),
    RiceDataset(),
    EcoliDataset(),
]

unavailable_cache_keys = set()

def safe_load_dataset(ds):
    """Load dataset, skipping those that are not available from ucimlrepo."""
    if ds.cache_key in unavailable_cache_keys:
        return None
    try:
        return cache.load_or_create(ds.cache_key, ds.load)
    except DatasetNotFoundError as e:
        print(f"Skipping dataset {ds.name} (id={ds.uci_id}): {e}")
        unavailable_cache_keys.add(ds.cache_key)
        return None

# ADVERSARIAL PERTURBATIONS WITH CACHE

print("=== Adversarial Perturbations ===\n")

for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print(f"Dataset: {info.name}")

    # LOAD CACHED MLP MODEL
    mlp_model = MLPClassifier()

    mlp_key = registry.make_key(f"{info.name}_{ds.uci_id}", mlp_model.name)

    if not registry.exists(mlp_key):
        print(f"  WARNING: MLP model not found in cache. Skipping adversarial perturbations.")
        print(f"  Please train it first by running: python tests/test_models.py")
        print()
        continue

    mlp_model = registry.load(mlp_key)

    # BIM PERTURBATIONS
    bim_config = ADVERSARIAL_CONFIG['bim']
    for epsilon in bim_config['epsilons']:
        if epsilon == 0.0:
            continue  # skip clean

        method = "bim"
        perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(method, epsilon=epsilon)}"

        def generate_bim(eps=epsilon):
            return adv_gen.bim(
                model=mlp_model.model,
                X=splits.X_test,
                y=splits.y_test,
                epsilon=eps,
                alpha=eps / bim_config['n_iter'] * 2.5,
                num_iter=bim_config['n_iter'],
                task_type=info.task_type,
                device=mlp_model.device
            )

        X_perturbed = cache.load_or_create(perturb_key, generate_bim)

        delta = X_perturbed - splits.X_test
        abs_diff = np.abs(delta).mean()
        rmse = np.sqrt(np.mean(delta ** 2))
        scale = np.sqrt(np.mean(splits.X_test ** 2))
        rel_rmse = rmse / (scale + 1e-12)
        l_inf = np.abs(delta).max()

        str_key = f"bim_eps{epsilon}"
        print(
            f"  {str_key:<15}: shape={X_perturbed.shape}, "
            f"abs_diff={abs_diff:8.4f}, rel_rmse={rel_rmse:8.4f}, "
            f"L_inf={l_inf:8.4f}, cached={cache.exists(perturb_key)}")

    # PGD PERTURBATIONS
    pgd_config = ADVERSARIAL_CONFIG['pgd']
    for epsilon in pgd_config['epsilons']:
        if epsilon == 0.0:
            continue  # skip clean

        method = "pgd"
        perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(method, epsilon=epsilon)}"

        def generate_pgd(eps=epsilon):
            return adv_gen.pgd(
                model=mlp_model.model,
                X=splits.X_test,
                y=splits.y_test,
                epsilon=eps,
                alpha=eps / pgd_config['n_iter'] * 2.5,
                num_iter=pgd_config['n_iter'],
                task_type=info.task_type,
                device=mlp_model.device,
                random_start=True
            )

        X_perturbed = cache.load_or_create(perturb_key, generate_pgd)

        delta = X_perturbed - splits.X_test
        abs_diff = np.abs(delta).mean()
        rmse = np.sqrt(np.mean(delta ** 2))
        scale = np.sqrt(np.mean(splits.X_test ** 2))
        rel_rmse = rmse / (scale + 1e-12)
        l_inf = np.abs(delta).max()

        str_key = f"pgd_eps{epsilon}"
        print(
            f"  {str_key:<15}: shape={X_perturbed.shape}, "
            f"abs_diff={abs_diff:8.4f}, rel_rmse={rel_rmse:8.4f}, "
            f"L_inf={l_inf:8.4f}, cached={cache.exists(perturb_key)}")

    # C W PERTURBATIONS
    cw_config = ADVERSARIAL_CONFIG['cw']
    for c_value in cw_config['c_values']:
        method = "cw"
        perturb_key = f"{info.name}_{ds.uci_id}/{AdversarialPerturbationGenerator.make_key(method, c=c_value)}"

        def generate_cw(c=c_value):
            return adv_gen.cw(
                model=mlp_model.model,
                X=splits.X_test,
                y=splits.y_test,
                c=c,
                kappa=cw_config['kappa'],
                num_iter=cw_config['n_iter'],
                learning_rate=cw_config['lr'],
                task_type=info.task_type,
                device=mlp_model.device,
                targeted=False
            )

        X_perturbed = cache.load_or_create(perturb_key, generate_cw)

        delta = X_perturbed - splits.X_test
        abs_diff = np.abs(delta).mean()
        rmse = np.sqrt(np.mean(delta ** 2))
        scale = np.sqrt(np.mean(splits.X_test ** 2))
        rel_rmse = rmse / (scale + 1e-12)
        l2_mean = np.linalg.norm(delta, axis=1).mean()

        str_key = f"cw_c{c_value}"
        print(
            f"  {str_key:<15}: shape={X_perturbed.shape}, "
            f"abs_diff={abs_diff:8.4f}, rel_rmse={rel_rmse:8.4f}, "
            f"L2={l2_mean:8.4f}, cached={cache.exists(perturb_key)}")

    print()


"""
Adversarial perturbations are fundamentally different from random noise:

BIM (Basic Iterative Method):
- Gradient-based attack with L_inf constraint
- Iteratively steps in the direction of the gradient sign
- Creates small, targeted perturbations that maximize model error
- L_inf is bounded by epsilon, but actual perturbations are often smaller

PGD (Projected Gradient Descent):
- Stronger variant of BIM with random initialization
- Starts from random point in epsilon ball
- Multiple restarts can find stronger adversarial examples
- Generally achieves higher error than BIM for same epsilon

C&W (Carlini & Wagner):
- L2-based optimization attack
- Balances between small perturbations and high attack success
- Parameter c controls trade-off: higher c = stronger attack, larger perturbations
- More sophisticated than BIM/PGD but also slower

All adversarial attacks exploit model gradients and are MLP-specific.
Unlike random perturbations, they are carefully crafted to maximize prediction error.

Expected behavior:
- Adversarial perturbations should have MUCH higher impact on error than random noise of same magnitude
- rel_RMSE doesn't capture adversarial strength - error increase is the key metric
- Epistemic uncertainty should increase when model is attack
"""
