"""Train UQ models and summarize uncertainty.

Trains and caches uncertainty-aware classifiers, computes performance and
uncertainty metrics (aleatoric, epistemic, confidence, ECE), and saves summary
results to text/CSV.
"""

import sys
import os
import csv
from pathlib import Path

sys.path.append(os.getcwd())
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, IrisDataset, RiceDataset, EcoliDataset
from data.splitter import DataSplitter
from models.registry import ModelRegistry

from uncertainty.linear_uq import LogisticUQ
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.mlp_uq import MLPClassifierUQ
from uncertainty.lgbm_uq import LightGBMClassifierUQ
from uncertainty.catboost_uq import CatBoostClassifierUQ

from evaluation.performance import ClassificationMetrics
from evaluation.uncertainty import UncertaintyMetrics
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
PLUS_MINUS = "\u00b1"
BOOTSTRAP_SAMPLES = 200


def format_optional(value, decimals=4):
    if value is None or not np.isfinite(value):
        return "----"
    return f"{value:.{decimals}f}"


def format_with_std(mean, std, decimals=4):
    if mean is None or std is None:
        return "----"
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "----"
    return f"{mean:.{decimals}f}{PLUS_MINUS}{std:.{decimals}f}"


def bootstrap_metric_stds(y_true, preds, y_proba, n_bootstrap=BOOTSTRAP_SAMPLES):
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    y_proba = None if y_proba is None else np.asarray(y_proba)
    if n_bootstrap <= 1:
        return {
            "accuracy_std": 0.0,
            "f1_std": 0.0,
            "ece_std": 0.0,
        }
    rng = np.random.default_rng(0)
    n = len(y_true)
    labels = np.unique(y_true)
    acc_vals = np.empty(n_bootstrap)
    f1_vals = np.empty(n_bootstrap)
    ece_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_i = y_true[idx]
        preds_i = preds[idx]
        y_proba_i = y_proba[idx] if y_proba is not None else None
        acc_vals[i] = accuracy_score(y_true_i, preds_i)
        f1_vals[i] = f1_score(y_true_i, preds_i, labels=labels, average="weighted", zero_division=0)
        if y_proba_i is None:
            ece_vals[i] = np.nan
        else:
            ece_vals[i] = UncertaintyMetrics.ece_classification(y_true_i, y_proba_i)
    ddof = 1 if n_bootstrap > 1 else 0
    return {
        "accuracy_std": float(np.nanstd(acc_vals, ddof=ddof)),
        "f1_std": float(np.nanstd(f1_vals, ddof=ddof)),
        "ece_std": float(np.nanstd(ece_vals, ddof=ddof)),
    }


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


def print_uncertainty_stats(aleatoric, epistemic, total):
    print(f"  Aleatoric: {format_with_std(np.mean(aleatoric), np.std(aleatoric))}")
    print(f"  Epistemic: {format_with_std(np.mean(epistemic), np.std(epistemic))}")
    print(f"  Total    : {format_with_std(np.mean(total), np.std(total))}")


def evaluate_and_print(uq_model, splits, dataset_name):
    preds, aleatoric, epistemic = uq_model.predict_with_uncertainty(splits.X_test)
    total = uq_model.total_uncertainty(splits.X_test)

    y_proba = uq_model.predict_proba(splits.X_test) if hasattr(uq_model, "predict_proba") else None
    perf_metrics = ClassificationMetrics.compute(splits.y_test, preds, y_proba)
    metric_stds = bootstrap_metric_stds(splits.y_test, preds, y_proba)
    print(f"  Accuracy: {format_with_std(perf_metrics['accuracy'], metric_stds['accuracy_std'])}")
    print(f"  Precision: {perf_metrics['precision']:.4f}")
    print(f"  Recall: {perf_metrics['recall']:.4f}")
    print(f"  F1: {format_with_std(perf_metrics['f1'], metric_stds['f1_std'])}")
    if perf_metrics.get("auc") is not None:
        print(f"  AUC: {perf_metrics['auc']:.4f}")

    print_uncertainty_stats(aleatoric, epistemic, total)
    alea_mean = float(np.mean(aleatoric))
    alea_std = float(np.std(aleatoric))
    epis_mean = float(np.mean(epistemic))
    epis_std = float(np.std(epistemic))
    epis_cv = float(epis_std / epis_mean) if not np.isclose(epis_mean, 0.0) else np.nan
    alea_var = float(np.var(aleatoric))
    epis_var = float(np.var(epistemic))
    denom = alea_var + epis_var
    epis_snr = float(epis_var / denom) if denom > 0 else np.nan

    if not hasattr(uq_model, "predict_proba"):
        raise RuntimeError("UQ classification model must implement predict_proba for UQ metrics.")
    y_proba = uq_model.predict_proba(splits.X_test)
    confidence = np.max(y_proba, axis=1)
    conf_mean = float(np.mean(confidence))
    conf_std = float(np.std(confidence))
    uq_metrics = UncertaintyMetrics.compute_all_classification(
        splits.y_test,
        y_proba,
        aleatoric,
        epistemic,
    )
    print(f"  ECE: {format_with_std(uq_metrics['ece'], metric_stds['ece_std'])}")
    print(f"  Confidence mean: {format_with_std(conf_mean, conf_std)}")
    print(f"  Mean entropy: {uq_metrics['mean_entropy']:.4f}")
    print(f"  CV (epistemic): {format_optional(epis_cv)}")
    print(f"  SNR (epistemic): {format_optional(epis_snr)}")

    result = {
        "dataset": dataset_name,
        "model": uq_model.name,
        "preds": preds,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total_uncertainty": total,
        "performance": perf_metrics,
        "uq_metrics": uq_metrics,
        "confidence_mean": conf_mean,
        "confidence_std": conf_std,
        "accuracy_std": metric_stds["accuracy_std"],
        "f1_std": metric_stds["f1_std"],
        "ece_std": metric_stds["ece_std"],
        "aleatoric_mean": alea_mean,
        "aleatoric_std": alea_std,
        "epistemic_mean": epis_mean,
        "epistemic_std": epis_std,
        "epistemic_cv": epis_cv,
        "epistemic_snr": epis_snr,
    }

    # SAVE TO GLOBAL RESULTS
    all_results.append(result)

    return result


# UQ MODELS


for ds in datasets:
    info = safe_load_dataset(ds)
    if info is None:
        continue
    split_key = f"{info.name}_{ds.uci_id}/splits"
    splits = cache.load_or_create(split_key, lambda: splitter.split(info))

    print("=" * 50)
    print(f"Dataset: {info.name} ({info.task_type})")
    print(f"Train: {splits.X_train.shape}, Test: {splits.X_test.shape}")

    # LOGISTIC UQ

    print(f"\n=== Logistic UQ ===")

    uq_model = LogisticUQ()

    uq_key = registry.make_key(f"{info.name}_{ds.uci_id}", uq_model.name)

    if registry.exists(uq_key):
        uq_model = registry.load(uq_key)
    else:
        uq_model.fit(splits.X_train, splits.y_train)
        registry.save(uq_key, uq_model)

    print(f"Model: {uq_model.name} (cached: {registry.exists(uq_key)})")
    evaluate_and_print(uq_model, splits, info.name)
    print()

    # RANDOM FOREST UQ

    print(f"\n=== Random Forest UQ ===")

    rf_uq = RandomForestClassifierUQ()

    rf_uq_key = registry.make_key(f"{info.name}_{ds.uci_id}", rf_uq.name)

    if registry.exists(rf_uq_key):
        rf_uq = registry.load(rf_uq_key)
    else:
        rf_uq.fit(splits.X_train, splits.y_train)
        registry.save(rf_uq_key, rf_uq)

    print(f"Model: {rf_uq.name} (cached: {registry.exists(rf_uq_key)})")
    evaluate_and_print(rf_uq, splits, info.name)

    # MLP UQ

    print(f"\n=== MLP UQ ===")

    mlp_uq = MLPClassifierUQ()

    mlp_uq_key = registry.make_key(f"{info.name}_{ds.uci_id}", mlp_uq.name)

    if registry.exists(mlp_uq_key):
        mlp_uq = registry.load(mlp_uq_key)
    else:
        mlp_uq.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        registry.save(mlp_uq_key, mlp_uq)

    print(f"Model: {mlp_uq.name} (cached: {registry.exists(mlp_uq_key)})")
    evaluate_and_print(mlp_uq, splits, info.name)

    # LIGHTGBM UQ

    print(f"\n=== LightGBM UQ ===")

    lgbm_uq = LightGBMClassifierUQ()

    lgbm_uq_key = registry.make_key(f"{info.name}_{ds.uci_id}", lgbm_uq.name)

    if registry.exists(lgbm_uq_key):
        lgbm_uq = registry.load(lgbm_uq_key)
    else:
        lgbm_uq.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        registry.save(lgbm_uq_key, lgbm_uq)

    print(f"Model: {lgbm_uq.name} (cached: {registry.exists(lgbm_uq_key)})")
    evaluate_and_print(lgbm_uq, splits, info.name)

    # CATBOOST UQ

    print(f"\n=== CatBoost UQ ===")

    catboost_uq = CatBoostClassifierUQ()

    catboost_uq_key = registry.make_key(f"{info.name}_{ds.uci_id}", catboost_uq.name)

    if registry.exists(catboost_uq_key):
        catboost_uq = registry.load(catboost_uq_key)
    else:
        catboost_uq.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        registry.save(catboost_uq_key, catboost_uq)

    print(f"Model: {catboost_uq.name} (cached: {registry.exists(catboost_uq_key)})")
    evaluate_and_print(catboost_uq, splits, info.name)
    print()

# SUMMARY

print("\n" + "=" * 80)
print("SUMMARY - UQ MODELS")
print("=" * 80)

print(f"\nTotal tests: {len(all_results)}")
print(f"Datasets: {len({r['dataset'] for r in all_results})}")
print(f"Models: {len(set(r['model'] for r in all_results))}")

print("\n" + "=" * 80)
print("ALL RESULTS")
print("=" * 80)
header = (
    f"{'Dataset':<15} | {'Model':<30} | {'Accuracy':>14} | {'F1':>14} | {'ECE':>14} | "
    f"{'Aleatoric':>14} | {'Epistemic':>14} | {'Conf':>14} | {'CV':>8} | {'SNR':>8}"
)
print(header)
print("-" * len(header))

for r in all_results:
    dataset = r['dataset']
    model = r['model']
    acc = r['performance']['accuracy']
    f1 = r['performance']['f1']
    ece = r['uq_metrics']['ece']
    acc_std = r.get('accuracy_std')
    f1_std = r.get('f1_std')
    ece_std = r.get('ece_std')
    alea_mean = r.get('aleatoric_mean', float(np.mean(r['aleatoric'])))
    alea_std = r.get('aleatoric_std', float(np.std(r['aleatoric'])))
    epis_mean = r.get('epistemic_mean', float(np.mean(r['epistemic'])))
    epis_std = r.get('epistemic_std', float(np.std(r['epistemic'])))
    conf = r.get('confidence_mean')
    conf_std = r.get('confidence_std')
    cv = r.get('epistemic_cv')
    snr = r.get('epistemic_snr')
    acc_val = format_with_std(acc, acc_std)
    f1_val = format_with_std(f1, f1_std)
    ece_val = format_with_std(ece, ece_std)
    alea_val = format_with_std(alea_mean, alea_std)
    epis_val = format_with_std(epis_mean, epis_std)
    conf_val = format_with_std(conf, conf_std)
    cv_val = format_optional(cv)
    snr_val = format_optional(snr)
    print(
        f"{dataset:<15} | {model:<30} | {acc_val:>14} | {f1_val:>14} | {ece_val:>14} | "
        f"{alea_val:>14} | {epis_val:>14} | {conf_val:>14} | {cv_val:>8} | {snr_val:>8}"
    )

print("\n" + "=" * 80)

# SAVE TO FILE

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / (Path(__file__).stem + "_summary.txt")
CSV_FILE = RESULTS_DIR / (Path(__file__).stem + "_results.csv")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("UQ MODELS TESTS - SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total tests: {len(all_results)}\n")
    f.write(f"Datasets: {len(datasets)}\n")
    f.write(f"Models: {len(set(r['model'] for r in all_results))}\n\n")

    f.write("=" * 80 + "\n")
    f.write("ALL RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    for r in all_results:
        dataset = r['dataset']
        model = r['model']
        acc = r['performance']['accuracy']
        f1 = r['performance']['f1']
        ece = r['uq_metrics']['ece']
        acc_std = r.get('accuracy_std')
        f1_std = r.get('f1_std')
        ece_std = r.get('ece_std')
        alea_mean = r.get('aleatoric_mean', float(np.mean(r['aleatoric'])))
        alea_std = r.get('aleatoric_std', float(np.std(r['aleatoric'])))
        epis_mean = r.get('epistemic_mean', float(np.mean(r['epistemic'])))
        epis_std = r.get('epistemic_std', float(np.std(r['epistemic'])))
        conf = r.get('confidence_mean')
        conf_std = r.get('confidence_std')
        cv = r.get('epistemic_cv')
        snr = r.get('epistemic_snr')
        acc_val = format_with_std(acc, acc_std)
        f1_val = format_with_std(f1, f1_std)
        ece_val = format_with_std(ece, ece_std)
        alea_val = format_with_std(alea_mean, alea_std)
        epis_val = format_with_std(epis_mean, epis_std)
        conf_val = format_with_std(conf, conf_std)
        cv_val = format_optional(cv)
        snr_val = format_optional(snr)
        f.write(
            f"{dataset:<15} | {model:<30} | {acc_val:>14} | {f1_val:>14} | {ece_val:>14} | "
            f"{alea_val:>14} | {epis_val:>14} | {conf_val:>14} | {cv_val:>8} | {snr_val:>8}\n"
        )

    f.write("\n" + "=" * 80 + "\n")

# SAVE FLAT METRICS TO CSV FOR DOWNSTREAM ANALYSIS
csv_cols = [
    "dataset",
    "model",
    "accuracy",
    "accuracy_std",
    "f1",
    "f1_std",
    "ece",
    "ece_std",
    "aleatoric_mean",
    "aleatoric_std",
    "epistemic_mean",
    "epistemic_std",
    "confidence_mean",
    "confidence_std",
    "epistemic_cv",
    "epistemic_snr",
]
with open(CSV_FILE, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
    writer.writeheader()
    for r in all_results:
        perf = r["performance"]
        uq_m = r["uq_metrics"]
        writer.writerow({
            "dataset": r["dataset"],
            "model": r["model"],
            "accuracy": perf.get("accuracy"),
            "accuracy_std": r.get("accuracy_std"),
            "f1": perf.get("f1"),
            "f1_std": r.get("f1_std"),
            "ece": uq_m.get("ece"),
            "ece_std": r.get("ece_std"),
            "aleatoric_mean": r.get("aleatoric_mean", float(np.mean(r["aleatoric"]))),
            "aleatoric_std": r.get("aleatoric_std", float(np.std(r["aleatoric"]))),
            "epistemic_mean": r.get("epistemic_mean", float(np.mean(r["epistemic"]))),
            "epistemic_std": r.get("epistemic_std", float(np.std(r["epistemic"]))),
            "confidence_mean": r.get("confidence_mean"),
            "confidence_std": r.get("confidence_std"),
            "epistemic_cv": r.get("epistemic_cv"),
            "epistemic_snr": r.get("epistemic_snr"),
        })

print(f"\nOK Results saved to: {OUTPUT_FILE}")
print(f"OK CSV saved to: {CSV_FILE}")
