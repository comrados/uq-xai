"""Train baseline models.

Trains and caches baseline classifiers per dataset, evaluates on test splits,
and writes summary metrics to text/CSV outputs. Models: Logistic, Random Forest,
MLP, LightGBM, CatBoost.
"""

import sys
import os
import csv
from pathlib import Path
sys.path.append(os.getcwd())

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, IrisDataset, RiceDataset, EcoliDataset
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from models.linear import LogisticModel
from models.forest import RandomForestClassifierModel
from models.mlp import MLPClassifier
from models.lgbm import LightGBMClassifierModel
from models.catboost import CatBoostClassifierModel
from evaluation.performance import ClassificationMetrics
from ucimlrepo.fetch import DatasetNotFoundError

# SETUP

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    IrisDataset(),
    RiceDataset(),
    EcoliDataset(),
]

# COLLECT ALL RESULTS FOR SUMMARY
all_results = []
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


def evaluate_and_print(model, splits, dataset_name):
    """Evaluate the model and print metrics."""
    preds = model.predict(splits.X_test)
    proba = model.predict_proba(splits.X_test) if hasattr(model, 'predict_proba') else None
    metrics = ClassificationMetrics.compute(splits.y_test, preds, proba)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    if metrics.get('auc'):
        print(f"  AUC: {metrics['auc']:.4f}")

    # SAVE TO GLOBAL RESULTS
    result = {
        'dataset': dataset_name,
        'model': model.name,
        'metrics': metrics
    }
    all_results.append(result)

    return metrics


# MODELS

for ds in datasets:

    info = safe_load_dataset(ds)
    if info is None:
        continue
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print("=" * 50)
    print(f"Dataset: {info.name} ({info.task_type})")
    print(f"Train: {splits.X_train.shape}, Test: {splits.X_test.shape}")

    # LOGISTIC

    print(f"\n=== Logistic ===")

    model = LogisticModel()

    model_key = registry.make_key(f"{info.name}_{ds.uci_id}", model.name)
    model = registry.get_or_train(
        model_key, model,
        splits.X_train, splits.y_train,
        splits.X_val, splits.y_val
    )

    print(f"Model: {model.name} (cached: {registry.exists(model_key)})")
    evaluate_and_print(model, splits, info.name)

    # FOREST

    print(f"\n=== Random Forest ===")

    rf_model = RandomForestClassifierModel()

    rf_key = registry.make_key(f"{info.name}_{ds.uci_id}", rf_model.name)
    rf_model = registry.get_or_train(
        rf_key, rf_model,
        splits.X_train, splits.y_train
    )

    print(f"Model: {rf_model.name} (cached: {registry.exists(rf_key)})")
    evaluate_and_print(rf_model, splits, info.name)

    # MLP

    print(f"\n=== MLP ===")

    mlp_model = MLPClassifier()

    mlp_key = registry.make_key(f"{info.name}_{ds.uci_id}", mlp_model.name)
    mlp_model = registry.get_or_train(
        mlp_key, mlp_model,
        splits.X_train, splits.y_train,
        splits.X_val, splits.y_val
    )

    print(f"Model: {mlp_model.name} (device: {mlp_model.device}, cached: {registry.exists(mlp_key)})")
    evaluate_and_print(mlp_model, splits, info.name)

    # LIGHTGBM

    print(f"\n=== LightGBM ===")

    lgbm_model = LightGBMClassifierModel()

    lgbm_key = registry.make_key(f"{info.name}_{ds.uci_id}", lgbm_model.name)
    lgbm_model = registry.get_or_train(
        lgbm_key, lgbm_model,
        splits.X_train, splits.y_train,
        splits.X_val, splits.y_val
    )

    print(f"Model: {lgbm_model.name} (cached: {registry.exists(lgbm_key)})")
    evaluate_and_print(lgbm_model, splits, info.name)

    # CATBOOST

    print(f"\n=== CatBoost ===")

    catboost_model = CatBoostClassifierModel()

    catboost_key = registry.make_key(f"{info.name}_{ds.uci_id}", catboost_model.name)
    catboost_model = registry.get_or_train(
        catboost_key, catboost_model,
        splits.X_train, splits.y_train,
        splits.X_val, splits.y_val
    )

    print(f"Model: {catboost_model.name} (cached: {registry.exists(catboost_key)})")
    evaluate_and_print(catboost_model, splits, info.name)

    print()

# SUMMARY

print("\n" + "=" * 80)
print("SUMMARY - BASE MODELS")
print("=" * 80)

print(f"\nTotal tests: {len(all_results)}")
print(f"Datasets: {len({r['dataset'] for r in all_results})}")
print(f"Models: {len(set(r['model'] for r in all_results))}")

print("\n" + "=" * 80)
print("ALL RESULTS")
print("=" * 80)
print(f"{'Dataset':<15} | {'Model':<30} | {'Accuracy':>8} | {'F1':>6} | {'Precision':>9} | {'Recall':>6}")
print("-" * 100)

for r in all_results:
    dataset = r['dataset']
    model = r['model']
    acc = r['metrics']['accuracy']
    f1 = r['metrics']['f1']
    prec = r['metrics']['precision']
    rec = r['metrics']['recall']
    print(f"{dataset:<15} | {model:<30} | {acc:>8.4f} | {f1:>6.4f} | {prec:>9.4f} | {rec:>6.4f}")

print("\n" + "=" * 80)

# SAVE TO FILE

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / (Path(__file__).stem + "_summary.txt")
CSV_FILE = RESULTS_DIR / (Path(__file__).stem + "_results.csv")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("BASE MODELS TESTS - SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total tests: {len(all_results)}\n")
    f.write(f"Datasets: {len(datasets)}\n")
    f.write(f"Models: {len(set(r['model'] for r in all_results))}\n\n")

    f.write("=" * 80 + "\n")
    f.write("ALL RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Dataset':<15} | {'Model':<30} | {'Accuracy':>8} | {'F1':>6} | {'Precision':>9} | {'Recall':>6}\n")
    f.write("-" * 100 + "\n")

    for r in all_results:
        dataset = r['dataset']
        model = r['model']
        acc = r['metrics']['accuracy']
        f1 = r['metrics']['f1']
        prec = r['metrics']['precision']
        rec = r['metrics']['recall']
        f.write(f"{dataset:<15} | {model:<30} | {acc:>8.4f} | {f1:>6.4f} | {prec:>9.4f} | {rec:>6.4f}\n")

    f.write("\n" + "=" * 80 + "\n")

# SAVE FLAT METRICS TO CSV FOR DOWNSTREAM ANALYSIS
csv_cols = ["dataset", "model", "accuracy", "f1", "precision", "recall", "auc"]
with open(CSV_FILE, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
    writer.writeheader()
    for r in all_results:
        metrics = r["metrics"]
        writer.writerow({
            "dataset": r["dataset"],
            "model": r["model"],
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "auc": metrics.get("auc"),
        })

print(f"\nOK Results saved to: {OUTPUT_FILE}")
print(f"OK CSV saved to: {CSV_FILE}")
