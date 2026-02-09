"""Prepare datasets and cache perturbations.

Loads UCI tabular datasets, caches splits and perturbations, and reports
dataset and perturbation statistics for sanity checks.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, IrisDataset, RiceDataset, EcoliDataset
from data.splitter import DataSplitter
from data.perturbations import PerturbationGenerator
from config.settings import PERTURBATION_CONFIG
from ucimlrepo.fetch import DatasetNotFoundError

cache = Cache()
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

def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    header_line = "| " + " | ".join(
        f"{headers[i]:<{widths[i]}}" for i in range(len(headers))
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"

    lines = [header_line, sep_line]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{str(row[i]):<{widths[i]}}" for i in range(len(headers))
        ) + " |")
    return "\n".join(lines)

def num_classes_from_info(info):
    if info.task_type != "classification":
        return "-"
    if info.class_names is not None:
        return len(info.class_names)
    return len(np.unique(info.y))

def num_classes_from_labels(y, task_type):
    if task_type != "classification":
        return "-"
    return len(np.unique(y))

# DATASET STATS

print("=== Dataset Stats ===\n")

stats_rows = []
split_rows = []
splitter = DataSplitter()

for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue

    stats_rows.append([
        info.name,
        info.task_type,
        info.X.shape[0],
        info.X.shape[1],
        num_classes_from_info(info),
    ])

    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    split_rows.extend([
        [
            info.name,
            "train",
            info.task_type,
            splits.X_train.shape[0],
            splits.X_train.shape[1],
            num_classes_from_labels(splits.y_train, info.task_type),
        ],
        [
            info.name,
            "val",
            info.task_type,
            splits.X_val.shape[0],
            splits.X_val.shape[1],
            num_classes_from_labels(splits.y_val, info.task_type),
        ],
        [
            info.name,
            "test",
            info.task_type,
            splits.X_test.shape[0],
            splits.X_test.shape[1],
            num_classes_from_labels(splits.y_test, info.task_type),
        ],
    ])

if stats_rows:
    print(format_table(
        ["Dataset", "Task", "Size", "Features", "Classes"],
        stats_rows
    ))
    print()

if split_rows:
    print(format_table(
        ["Dataset", "Split", "Task", "Size", "Features", "Classes"],
        split_rows
    ))
    print()

# DATASETS WITH CACHE

print("=== Datasets ===\n")

for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue
    print(f"Dataset: {info.name}")
    print(f"Shape: {info.X.shape}")
    print(f"Target: {info.y.shape}")
    print(f"Features: {info.feature_names}")
    print(f"Task: {info.task_type}")
    print(f"Cached: {cache.exists(ds.cache_key)}")
    print()

# DATA SPLIT WITH CACHE

print("=== Splits ===\n")

splitter = DataSplitter()

for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue
    
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))
    
    print(f"Dataset: {info.name}")
    print(f"Train: {splits.X_train.shape}")
    print(f"Val: {splits.X_val.shape}")
    print(f"Test: {splits.X_test.shape}")
    print(f"Total: {len(splits.X_train) + len(splits.X_val) + len(splits.X_test)}")
    print(f"Cached: {cache.exists(split_key)}")
    print()

# PERTURBATIONS WITH CACHE

print("=== Perturbations ===\n")

perturber = PerturbationGenerator()

for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print(f"Dataset: {info.name}")

    for method in ["gaussian", "missing", "permute"]:
        config_key = "permutation" if method == "permute" else method
        levels = PERTURBATION_CONFIG.get(config_key, [])
        
        for level in levels:
            if level == 0.0:
                continue  # skip clean
            
            perturb_key = f"{info.name}_{ds.uci_id}/{PerturbationGenerator.make_key(method, level)}"
            
            X_perturbed = cache.load_or_create(
                perturb_key,
                lambda m=method, l=level: perturber.perturb(splits.X_test, m, l)
            )

            delta = X_perturbed - splits.X_test

            abs_diff = np.abs(delta).mean()
            rmse = np.sqrt(np.mean(delta ** 2))

            scale = np.sqrt(np.mean(splits.X_test ** 2))  # "energy" of initial features
            rel_rmse = rmse / (scale + 1e-12)            # relative value

            str_key = f"{method}_{level}"
            print(
                f"{str_key:<15}: shape={X_perturbed.shape}, "
                f"abs_diff={abs_diff:8.4f}, rel_rmse={rel_rmse:8.4f}, "
                f"cached={cache.exists(perturb_key)}")

    print()


"""

Gaussian noise is purely additive: we inject random deviations whose magnitude is directly controlled by sigma, and in standardized space the relative distortion (rel_RMSE) aligns closely with sigma. This makes Gaussian a natural reference scale for perturbation strength.

Missing-value noise (random masking with median imputation) is also effectively additive: each masked entry is replaced by a constant, which behaves like adding a fixed offset. As a result, its rel_RMSE grows smoothly and monotonically, allowing direct comparison with equivalent Gaussian sigma-levels.

Permutation noise is non-additive and structurally disruptive: values remain the same but are reassigned to different samples. Its impact depends on the feature distribution and therefore produces irregular, dataset-dependent rel_RMSE growth. This explains its higher variability and the lack of a clean monotonic relationship with the perturbation level.

All perturbations are mapped to a common scale using
rel_RMSE = RMSE(X' - X) / RMS(X),
which quantifies the actual distortion applied to the data and makes different noise types directly comparable.
    
"""
