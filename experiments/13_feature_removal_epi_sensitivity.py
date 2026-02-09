"""Measure feature-removal sensitivity via SHAP.

Measures how predictions shift when top-k SHAP features are imputed, comparing
low- and high-epistemic groups against random samples.
"""

import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, RiceDataset
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.mlp_uq import MLPClassifierUQ
from uncertainty.linear_uq import LogisticUQ
from config.settings import GLOBAL_SEED, XAI_CONFIG

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

DATASETS = [
    WineQualityDataset(),
    DryBeanDataset(),
    RiceDataset(),
]

MODEL_SPECS = [
    {
        "key": "rf",
        "label": "Random Forest",
        "uq_cls": RandomForestClassifierUQ,
        "explainer": "shap",
        "shap_mode": "tree",
    },
    {
        "key": "lr",
        "label": "Logistic Regression",
        "uq_cls": LogisticUQ,
        "explainer": "shap",
        "shap_mode": "kernel",
    },
]

DATASET_LABELS = {
    "wine_binary": "Wine",
    "bean": "Bean",
    "rice": "Rice",
}
MODEL_LABEL_ALIASES = {
    "RF": "Random Forest",
    "LR": "Logistic Regression",
}

REMOVE_COUNTS = [1, 2, 3, 4, 5]
STRAT_LOW_SAMPLES = 50
STRAT_MID_SAMPLES = 50
STRAT_HIGH_SAMPLES = 50
STRAT_MID_STRATEGY = "none"  # "none", "median" or "mean"
DIAG_VERSION = 3
DIAG_TOPK = max(REMOVE_COUNTS)
IMPUTE_STRATEGY = "median"  # "mean" or "median"
ALLOW_RANDOM_OVERLAP = False
RANDOM_SEED = GLOBAL_SEED + 101
RANDOM_REPEATS = 5
LOGIT_EPS = 1e-12
SHIFT_SPACES = ["probs", "logits"]
LOGITS_REF_INDEX = 0  # Log-odds reference class: 0 == first class in model.classes_

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HELPERS

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")


def _compute_impute_values(X_train: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        return np.mean(X_train, axis=0)
    if strategy == "median":
        return np.median(X_train, axis=0)
    raise ValueError(f"Unknown impute strategy: {strategy}")


def _select_stratified_groups(epistemic: np.ndarray,
                              low_n: int,
                              mid_n: int,
                              high_n: int,
                              mid_strategy: str,
                              allow_overlap: bool,
                              random_seed: int,
                              random_repeats: int) -> dict:
    n_samples = len(epistemic)
    if n_samples == 0:
        return {"random": []}

    sorted_idx = np.argsort(epistemic)
    low_n = min(low_n, n_samples)
    high_n = min(high_n, n_samples)
    mid_n = min(mid_n, n_samples)

    low_idx = sorted_idx[:low_n]
    high_idx = sorted_idx[-high_n:][::-1]

    mid_idx = np.array([], dtype=int)
    if mid_strategy != "none" and mid_n > 0:
        if allow_overlap:
            mid_pool = np.arange(n_samples)
        else:
            mid_pool = np.setdiff1d(np.arange(n_samples), np.concatenate([low_idx, high_idx]))
            if len(mid_pool) < mid_n:
                mid_pool = np.arange(n_samples)

        if mid_strategy == "mean":
            target = float(np.mean(epistemic))
        elif mid_strategy == "median":
            target = float(np.median(epistemic))
        else:
            raise ValueError(f"Unknown mid_strategy: {mid_strategy}")

        if len(mid_pool) > 0:
            distances = np.abs(epistemic[mid_pool] - target)
            mid_order = np.argsort(distances)
            mid_idx = mid_pool[mid_order[:mid_n]]

    group_indices = {
        "lowest": low_idx,
        "highest": high_idx,
    }
    if len(mid_idx):
        group_indices["mid"] = mid_idx

    selected_idx = np.concatenate([low_idx, mid_idx, high_idx]) if len(mid_idx) else np.concatenate([low_idx, high_idx])

    if allow_overlap:
        rand_pool = np.arange(n_samples)
    else:
        rand_pool = np.setdiff1d(np.arange(n_samples), selected_idx)
        if len(rand_pool) < mid_n:
            rand_pool = np.arange(n_samples)

    rand_indices = []
    for i in range(random_repeats):
        seed_rng = np.random.RandomState(random_seed + i)
        rand_idx = seed_rng.choice(rand_pool, min(mid_n, n_samples), replace=False)
        rand_indices.append(rand_idx)

    group_indices["random"] = rand_indices
    return group_indices


def _compute_shift_metrics(X_group: np.ndarray,
                           rank_group: np.ndarray,
                           base_values: np.ndarray,
                           impute_values: np.ndarray,
                           remove_counts: list,
                           predict_values) -> dict:
    n_samples, n_features = X_group.shape
    metrics = {
        "mae": [],
        "mse": [],
        "values": {
            "mae": [],
            "mse": [],
        },
    }

    for k in remove_counts:
        k = min(k, n_features)
        X_mod = X_group.copy()
        for i in range(n_samples):
            top_idx = rank_group[i, :k]
            X_mod[i, top_idx] = impute_values[top_idx]

        mod_values = predict_values(X_mod)
        diff = base_values - mod_values
        mae = np.mean(np.abs(diff), axis=1)
        mse = np.mean(diff ** 2, axis=1)

        metrics["mae"].append((float(mae.mean()), float(mae.std())))
        metrics["mse"].append((float(mse.mean()), float(mse.std())))
        metrics["values"]["mae"].append(mae)
        metrics["values"]["mse"].append(mse)

    return metrics


def _compute_topk_ratio(attributions: np.ndarray, k: int) -> np.ndarray:
    n_features = attributions.shape[1]
    k = min(k, n_features)
    abs_vals = np.abs(attributions)
    total = abs_vals.sum(axis=1)
    if k == n_features:
        topk_sum = total
    else:
        idx = np.argsort(abs_vals, axis=1)[:, ::-1][:, :k]
        topk_sum = np.take_along_axis(abs_vals, idx, axis=1).sum(axis=1)
    return np.divide(topk_sum, total, out=np.zeros_like(total), where=total > 0)


def _proba_to_log_odds(probs: np.ndarray, ref_index: int) -> np.ndarray:
    probs = np.clip(probs, LOGIT_EPS, 1.0 - LOGIT_EPS)
    log_probs = np.log(probs)
    return log_probs - log_probs[:, [ref_index]]


def _predict_logits(model_uq, X: np.ndarray, base_probs: np.ndarray = None) -> np.ndarray:
    probs = base_probs if base_probs is not None else model_uq.predict_proba(X)
    base_classes = model_uq.base_model.model.classes_
    ref_index = min(LOGITS_REF_INDEX, len(base_classes) - 1)
    return _proba_to_log_odds(probs, ref_index)


def _predict_shift_values(model_uq,
                          X: np.ndarray,
                          shift_space: str,
                          base_probs: np.ndarray = None) -> np.ndarray:
    if shift_space == "probs":
        return base_probs if base_probs is not None else model_uq.predict_proba(X)
    if shift_space == "logits":
        return _predict_logits(model_uq, X, base_probs=base_probs)
    raise ValueError(f"Unknown shift_space: {shift_space}")


def _diagnostics_compatible(results: dict, shift_space: str) -> bool:
    diagnostics = results.get("diagnostics")
    if diagnostics is None:
        return False
    config = diagnostics.get("config", {})
    if config.get("version") != DIAG_VERSION:
        return False
    if config.get("strat_low_n") != STRAT_LOW_SAMPLES:
        return False
    if config.get("strat_mid_n") != STRAT_MID_SAMPLES:
        return False
    if config.get("strat_high_n") != STRAT_HIGH_SAMPLES:
        return False
    if config.get("strat_mid_strategy") != STRAT_MID_STRATEGY:
        return False
    if config.get("topk") != DIAG_TOPK:
        return False
    config_shift = config.get("shift_space")
    if config_shift is None:
        return shift_space == "probs"
    if config_shift != shift_space:
        return False
    if shift_space == "logits" and config.get("logits_ref_index") != LOGITS_REF_INDEX:
        return False
    return True


def _aggregate_seed_metrics(seed_metrics_list: list, remove_counts: list) -> dict:
    metrics = {
        "mae": [],
        "mse": [],
        "values": {
            "mae": [],
            "mse": [],
        },
    }
    for k_idx in range(len(remove_counts)):
        mae_parts = [m["values"]["mae"][k_idx] for m in seed_metrics_list]
        mse_parts = [m["values"]["mse"][k_idx] for m in seed_metrics_list]
        mae_vals = np.concatenate(mae_parts) if mae_parts else np.array([])
        mse_vals = np.concatenate(mse_parts) if mse_parts else np.array([])

        mae_mean = float(mae_vals.mean()) if mae_vals.size else 0.0
        mae_std = float(mae_vals.std()) if mae_vals.size else 0.0
        mse_mean = float(mse_vals.mean()) if mse_vals.size else 0.0
        mse_std = float(mse_vals.std()) if mse_vals.size else 0.0

        metrics["mae"].append((mae_mean, mae_std))
        metrics["mse"].append((mse_mean, mse_std))
        metrics["values"]["mae"].append(mae_vals)
        metrics["values"]["mse"].append(mse_vals)

    return metrics


def _append_summary_rows(summary_rows: list, results: dict) -> None:
    dataset_name = results.get("dataset", "unknown")
    model_label = results.get("model", "n/a")
    shift_space = results.get("shift_space", "probs")
    remove_counts = results.get("remove_counts", REMOVE_COUNTS)
    groups = results.get("groups", {})
    for group_name, payload in groups.items():
        metrics = payload.get("metrics")
        if not metrics:
            continue
        for k, mae_pair, mse_pair in zip(remove_counts, metrics["mae"], metrics["mse"]):
            summary_rows.append({
                "dataset": dataset_name,
                "model": model_label,
                "shift_space": shift_space,
                "group": group_name,
                "k": int(k),
                "mae_mean": float(mae_pair[0]),
                "mae_std": float(mae_pair[1]),
                "mse_mean": float(mse_pair[0]),
                "mse_std": float(mse_pair[1]),
            })


def _append_diagnostic_rows(diagnostic_rows: list, results: dict) -> None:
    diagnostics = results.get("diagnostics")
    if diagnostics is None:
        return
    diagnostic_rows.append({
        "dataset": results.get("dataset"),
        "model": results.get("model"),
        "shift_space": results.get("shift_space", "probs"),
        "remove_counts": results.get("remove_counts", REMOVE_COUNTS),
        "diagnostics": diagnostics,
    })


def _print_group_table_console(group_name: str, remove_counts: list, metrics: dict) -> None:
    print(f"\nGroup: {group_name}")
    print(f"{'k':>3} | {'MAE (mean±std)':>18} | {'MSE (mean±std)':>18}")
    print("-" * 48)
    for k, mae_pair, mse_pair in zip(remove_counts, metrics["mae"], metrics["mse"]):
        mae_mean, mae_std = mae_pair
        mse_mean, mse_std = mse_pair
        print(f"{k:>3} | {mae_mean:.4f}±{mae_std:.4f} | {mse_mean:.4f}±{mse_std:.4f}")


def _print_cached_results(results: dict) -> None:
    dataset_name = results.get("dataset", "unknown")
    model_label = results.get("model", "n/a")
    shift_space = results.get("shift_space", "probs")
    group_stats = results.get("group_stats", {})
    remove_counts = results.get("remove_counts", REMOVE_COUNTS)

    _print_section(f"CACHED RESULTS - {dataset_name} - {model_label} - {shift_space}")
    if group_stats:
        print("Epistemic summary (mean ± std):")
        for group_name, stats in group_stats.items():
            if group_name == "random":
                n_repeats = stats.get("n_repeats", 1)
                n = stats.get("n", stats.get("n_total", 0))
                print(
                    f"  {group_name:>6}: {stats.get('epistemic_mean', 0.0):.4f} ± "
                    f"{stats.get('epistemic_std', 0.0):.4f} (n={n} x{n_repeats})"
                )
            else:
                print(
                    f"  {group_name:>6}: {stats.get('epistemic_mean', 0.0):.4f} ± "
                    f"{stats.get('epistemic_std', 0.0):.4f} (n={stats.get('n', 0)})"
                )

    groups = results.get("groups", {})
    for group_name, payload in groups.items():
        metrics = payload.get("metrics")
        if metrics:
            _print_group_table_console(group_name, remove_counts, metrics)


def _coerce_axes(axes, n_rows: int, n_cols: int):
    if n_rows == 1 and n_cols == 1:
        return [[axes]]
    if n_rows == 1:
        return [list(axes)]
    if n_cols == 1:
        return [[ax] for ax in axes]
    return axes


def _normalize_model_label(label: str) -> str:
    return MODEL_LABEL_ALIASES.get(label, label)


def _find_diag_entry(diagnostic_rows: list, dataset: str, model: str, shift_space: str):
    for row in diagnostic_rows:
        if (
            row.get("dataset") == dataset
            and row.get("model") == model
            and row.get("shift_space", "probs") == shift_space
        ):
            return row
    return None


def _plot_group_violin(diagnostic_rows: list,
                       dataset_order: list,
                       model_order: list,
                       shift_space: str,
                       output_path: Path,
                       value_key: str,
                       ylabel: str) -> None:
    n_rows = len(dataset_order)
    n_cols = len(model_order)
    fig_width = 3.2 * n_cols
    fig_height = 2.2 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
    )
    axes = _coerce_axes(axes, n_rows, n_cols)

    group_specs = [
        ("lowest", "Low", "#66c2a5"),
        ("highest", "High", "#fc8d62"),
        ("random", "Random", "lightgray"),
    ]
    if STRAT_MID_STRATEGY != "none":
        if STRAT_MID_STRATEGY == "mean":
            mid_label = "Mean"
        else:
            mid_label = "Median"
        group_specs.insert(1, ("mid", mid_label, "#ffd92f"))

    for i, dataset_name in enumerate(dataset_order):
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        for j, model_label in enumerate(model_order):
            ax = axes[i][j]
            entry = _find_diag_entry(diagnostic_rows, dataset_name, model_label, shift_space)
            if entry is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            diagnostics = entry.get("diagnostics", {})
            group_values = diagnostics.get(value_key, {})
            data = []
            labels = []
            colors = []
            for group_key, group_label, color in group_specs:
                values = group_values.get(group_key)
                if values is None or len(values) == 0:
                    data.append(np.array([0.0]))
                else:
                    data.append(np.asarray(values))
                labels.append(group_label)
                colors.append(color)

            positions = np.arange(1, len(data) + 1)
            vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, showextrema=False)
            for body, color in zip(vp["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.7)
            if "cmedians" in vp:
                vp["cmedians"].set_color("black")
                vp["cmedians"].set_linewidth(1.2)

            means = [float(np.mean(d)) for d in data]
            ax.scatter(positions, means, color="black", s=18, zorder=3)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(f"{dataset_label} | {model_label}", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=spec[2], ec="black", lw=0.3, label=spec[1])
        for spec in group_specs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mse_bars(summary_rows: list,
                   dataset_order: list,
                   model_order: list,
                   output_path: Path) -> None:
    if not summary_rows:
        return

    data = {}
    for row in summary_rows:
        key = (row["dataset"], row["model"], row["group"], row["k"])
        data[key] = (row["mse_mean"], row["mse_std"])

    k_values = list(REMOVE_COUNTS)
    n_rows = len(dataset_order)
    n_cols = len(model_order)

    width_ratios = [len(k_values) for _ in range(n_cols)]
    base_width_per_k = 0.85
    fig_width = base_width_per_k * sum(width_ratios)
    fig_height = 1.3 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = _coerce_axes(axes, n_rows, n_cols)

    group_specs = [
        ("lowest", "Low", "#4C78A8"),
        ("highest", "High", "#E45756"),
        ("random", "Random", "#9E9E9E"),
    ]
    if STRAT_MID_STRATEGY != "none":
        if STRAT_MID_STRATEGY == "mean":
            mid_label = "Mean"
        else:
            mid_label = "Median"
        group_specs.insert(1, ("mid", mid_label, "#F2CF5B"))
    bar_w = 0.15
    k_spacing = 0.6
    x = np.arange(len(k_values)) * k_spacing

    for i, dataset_name in enumerate(dataset_order):
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        for j, model_label in enumerate(model_order):
            ax = axes[i][j]
            has_data = False
            local_max = 0.0

            n_groups = len(group_specs)
            for idx, (group_key, group_label, color) in enumerate(group_specs):
                values = []
                errors = []
                for k in k_values:
                    entry = data.get((dataset_name, model_label, group_key, k))
                    if entry is None:
                        values.append(0.0)
                        errors.append(0.0)
                    else:
                        values.append(entry[0])
                        errors.append(entry[1])
                        has_data = True
                        local_max = max(local_max, entry[0] + entry[1])
                positions = x + (idx - (n_groups - 1) / 2) * bar_w
                ax.bar(
                    positions,
                    values,
                    width=bar_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    yerr=errors,
                    capsize=2,
                    label=group_label,
                )

            if i == 0:
                ax.set_title(model_label, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in k_values])
            if has_data:
                ax.set_ylim(0.0, max(0.05, local_max * 1.15))
            ax.grid(True, axis="y", alpha=0.25)
            if i == n_rows - 1:
                ax.set_xlabel("Top-k removed", fontsize=11)
            if j == 0:
                ax.set_ylabel(dataset_label)
            if not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="dimgray",
                )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=spec[2], ec="black", lw=0.3, label=spec[1])
        for spec in group_specs
    ]
    fig.suptitle("Feature Removal Sensitivity", y=0.95)
    fig.supylabel("Mean Square Error", fontsize=11, x=0.025)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0.00, 0.02, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_explainer(model_spec: dict,
                     model_uq,
                     X_train: np.ndarray,
                     feature_names) -> object:
    if model_spec["explainer"] == "shap":
        if model_spec.get("shap_mode") == "tree":
            return SHAPExplainer(model_uq.base_model)
        if model_spec.get("shap_mode") == "kernel":
            background_n = min(XAI_CONFIG["shap_background_samples"], len(X_train))
            X_background = X_train[:background_n]
            return SHAPExplainer(model_uq.base_model, X_background=X_background)
        raise ValueError(f"Unknown SHAP mode: {model_spec.get('shap_mode')}")
    raise ValueError(f"Unknown explainer: {model_spec['explainer']}")


def _write_group_table(f, group_name: str, remove_counts: list, metrics: dict) -> None:
    f.write(f"\nGroup: {group_name}\n")
    f.write(f"{'k':>3} | {'MAE (mean±std)':>18} | {'MSE (mean±std)':>18}\n")
    f.write("-" * 48 + "\n")
    for k, mae_pair, mse_pair in zip(remove_counts, metrics["mae"], metrics["mse"]):
        mae_mean, mae_std = mae_pair
        mse_mean, mse_std = mse_pair
        f.write(f"{k:>3} | {mae_mean:.4f}±{mae_std:.4f} | {mse_mean:.4f}±{mse_std:.4f}\n")


# MAIN

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()
summary_rows = []
diagnostic_rows = []

for ds in DATASETS:
    info = cache.load_or_create(ds.cache_key, ds.load)
    dataset_name = info.name
    dataset_slug = dataset_name.replace(" ", "_")
    dataset_key = f"{dataset_name}_{ds.uci_id}"
    splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

    _print_section(f"FEATURE REMOVAL SENSITIVITY: {dataset_name}")
    print(f"Test samples: {len(splits.X_test)}")

    for model_spec in MODEL_SPECS:
        cached_by_space = {}
        cache_paths = {}
        needs_compute_spaces = []

        for shift_space in SHIFT_SPACES:
            output_file = RESULTS_DIR / (
                f"feature_removal_{dataset_slug}_{model_spec['key']}_{model_spec['explainer']}_{shift_space}.pkl"
            )
            legacy_output_file = RESULTS_DIR / (
                f"feature_removal_{dataset_slug}_{model_spec['key']}_{model_spec['explainer']}.pkl"
            )
            legacy_output_file_alt = RESULTS_DIR / f"feature_removal_{dataset_slug}_{model_spec['key']}.pkl"
            cached_results = None
            cache_path = None

            if output_file.exists():
                with open(output_file, "rb") as f:
                    cached_results = pickle.load(f)
                cache_path = output_file
            elif shift_space == "probs" and legacy_output_file.exists():
                with open(legacy_output_file, "rb") as f:
                    cached_results = pickle.load(f)
                cache_path = legacy_output_file
            elif shift_space == "probs" and legacy_output_file_alt.exists():
                with open(legacy_output_file_alt, "rb") as f:
                    candidate = pickle.load(f)
                if candidate.get("explainer") == model_spec["explainer"]:
                    cached_results = candidate
                    cache_path = legacy_output_file_alt

            if cached_results is not None and _diagnostics_compatible(cached_results, shift_space):
                cached_results.setdefault("dataset", dataset_name)
                cached_results["model"] = _normalize_model_label(
                    cached_results.get("model", model_spec["label"])
                )
                cached_results.setdefault("remove_counts", list(REMOVE_COUNTS))
                cached_results.setdefault("groups", {})
                cached_results.setdefault("group_stats", {})
                cached_results.setdefault("shift_space", shift_space)
                _print_cached_results(cached_results)
                _append_summary_rows(summary_rows, cached_results)
                _append_diagnostic_rows(diagnostic_rows, cached_results)
                if cache_path is None:
                    cache_path = output_file
                cache_paths[shift_space] = cache_path
                cached_by_space[shift_space] = cached_results
                print(f"OK: loaded cached results: {cache_path}")
                continue

            if cached_results is not None:
                print(
                    f"WARNING: Cached results missing diagnostics or config mismatch for {shift_space}; "
                    "recomputing."
                )
            needs_compute_spaces.append(shift_space)

        if not needs_compute_spaces:
            continue

        model_uq = model_spec["uq_cls"]()
        model_key = registry.make_key(dataset_key, model_uq.name)

        if registry.exists(model_key):
            model_uq = registry.load(model_key)
            print(f"OK: loaded model: {model_uq.name}")
        else:
            print(f"WARNING: model not found, skipping: {model_key}")
            continue

        X_test = splits.X_test
        X_train = splits.X_train

        _print_section(f"EPISTEMIC UNCERTAINTY (CLEAN) - {model_spec['label']}")
        _, _, epistemic = model_uq.predict_with_uncertainty(X_test)
        print(f"Epistemic: mean={epistemic.mean():.4f}, std={epistemic.std():.4f}")

        group_indices = _select_stratified_groups(
            epistemic,
            STRAT_LOW_SAMPLES,
            STRAT_MID_SAMPLES,
            STRAT_HIGH_SAMPLES,
            STRAT_MID_STRATEGY,
            ALLOW_RANDOM_OVERLAP,
            RANDOM_SEED,
            RANDOM_REPEATS,
        )

        group_stats = {}
        for group_name, indices in group_indices.items():
            if group_name == "random":
                if indices:
                    rand_concat = np.concatenate(indices)
                else:
                    rand_concat = np.array([], dtype=int)
                group_stats[group_name] = {
                    "n": int(STRAT_MID_SAMPLES),
                    "n_repeats": int(RANDOM_REPEATS),
                    "n_total": int(len(rand_concat)),
                    "epistemic_mean": float(epistemic[rand_concat].mean()) if len(rand_concat) else 0.0,
                    "epistemic_std": float(epistemic[rand_concat].std()) if len(rand_concat) else 0.0,
                }
                continue
            group_stats[group_name] = {
                "n": int(len(indices)),
                "epistemic_mean": float(epistemic[indices].mean()) if len(indices) else 0.0,
                "epistemic_std": float(epistemic[indices].std()) if len(indices) else 0.0,
            }

        _print_section(f"GROUPS - {model_spec['label']}")
        for group_name, stats in group_stats.items():
            if group_name == "random":
                print(
                    f"{group_name:>6}: n={stats['n']} x{stats['n_repeats']}, "
                    f"epi={stats['epistemic_mean']:.4f}±{stats['epistemic_std']:.4f}"
                )
            else:
                print(
                    f"{group_name:>6}: n={stats['n']}, "
                    f"epi={stats['epistemic_mean']:.4f}±{stats['epistemic_std']:.4f}"
                )

        _print_section(f"EXPLANATIONS (CLEAN) - {model_spec['label']}")
        explainer = _build_explainer(
            model_spec=model_spec,
            model_uq=model_uq,
            X_train=X_train,
            feature_names=splits.feature_names,
        )

        impute_values = _compute_impute_values(X_train, IMPUTE_STRATEGY)

        results_by_space = {}
        for shift_space in needs_compute_spaces:
            results_by_space[shift_space] = {
                "dataset": dataset_name,
                "model": model_spec["label"],
                "model_key": model_spec["key"],
                "explainer": model_spec["explainer"],
                "shap_mode": model_spec.get("shap_mode"),
                "impute_strategy": IMPUTE_STRATEGY,
                "remove_counts": list(REMOVE_COUNTS),
                "random_repeats": int(RANDOM_REPEATS),
                "random_seed": int(RANDOM_SEED),
                "shift_space": shift_space,
                "group_stats": group_stats,
                "groups": {},
                "diagnostics": {
                    "config": {
                        "version": DIAG_VERSION,
                        "strat_low_n": STRAT_LOW_SAMPLES,
                        "strat_mid_n": STRAT_MID_SAMPLES,
                        "strat_high_n": STRAT_HIGH_SAMPLES,
                        "strat_mid_strategy": STRAT_MID_STRATEGY,
                    "topk": DIAG_TOPK,
                    "shift_space": shift_space,
                    "logits_ref_index": LOGITS_REF_INDEX,
                },
                    "group_confidence": {},
                    "group_topk_ratio": {},
                },
            }

        for group_name, indices in group_indices.items():
            if group_name == "random":
                if not indices:
                    continue
                seed_metrics_by_space = {space: [] for space in needs_compute_spaces}
                rand_conf = []
                rand_topk = []
                for seed_idx, rand_idx in enumerate(indices):
                    X_group = X_test[rand_idx]
                    attr_group = explainer.explain(X_group)
                    rank_group = np.argsort(np.abs(attr_group), axis=1)[:, ::-1]
                    base_probs = model_uq.predict_proba(X_group)
                    rand_conf.append(base_probs.max(axis=1))
                    rand_topk.append(_compute_topk_ratio(attr_group, DIAG_TOPK))

                    for shift_space in needs_compute_spaces:
                        base_values = _predict_shift_values(
                            model_uq,
                            X_group,
                            shift_space,
                            base_probs=base_probs,
                        )
                        metrics = _compute_shift_metrics(
                            X_group=X_group,
                            rank_group=rank_group,
                            base_values=base_values,
                            impute_values=impute_values,
                            remove_counts=REMOVE_COUNTS,
                            predict_values=lambda X, ss=shift_space: _predict_shift_values(model_uq, X, ss),
                        )
                        seed_metrics_by_space[shift_space].append(metrics)
                    print(f"OK: random seed {seed_idx + 1}/{RANDOM_REPEATS} computed ({len(rand_idx)} samples)")

                for shift_space, seed_metrics in seed_metrics_by_space.items():
                    agg_metrics = _aggregate_seed_metrics(seed_metrics, REMOVE_COUNTS)
                    results_by_space[shift_space]["groups"][group_name] = {
                        "indices": indices,
                        "seed_metrics": seed_metrics,
                        "metrics": agg_metrics,
                    }
                    if rand_conf:
                        results_by_space[shift_space]["diagnostics"]["group_confidence"][group_name] = np.concatenate(
                            rand_conf
                        )
                    if rand_topk:
                        results_by_space[shift_space]["diagnostics"]["group_topk_ratio"][group_name] = np.concatenate(
                            rand_topk
                        )

                    for k, mae_pair, mse_pair in zip(REMOVE_COUNTS, agg_metrics["mae"], agg_metrics["mse"]):
                        summary_rows.append({
                            "dataset": dataset_name,
                            "model": model_spec["label"],
                            "shift_space": shift_space,
                            "group": group_name,
                            "k": int(k),
                            "mae_mean": float(mae_pair[0]),
                            "mae_std": float(mae_pair[1]),
                            "mse_mean": float(mse_pair[0]),
                            "mse_std": float(mse_pair[1]),
                        })

                print(f"OK: random group metrics aggregated ({len(indices)} seeds)")
                continue

            if len(indices) == 0:
                continue
            X_group = X_test[indices]
            attr_group = explainer.explain(X_group)
            rank_group = np.argsort(np.abs(attr_group), axis=1)[:, ::-1]
            base_probs = model_uq.predict_proba(X_group)
            topk_ratio = _compute_topk_ratio(attr_group, DIAG_TOPK)

            for shift_space in needs_compute_spaces:
                base_values = _predict_shift_values(model_uq, X_group, shift_space, base_probs=base_probs)
                metrics = _compute_shift_metrics(
                    X_group=X_group,
                    rank_group=rank_group,
                    base_values=base_values,
                    impute_values=impute_values,
                    remove_counts=REMOVE_COUNTS,
                    predict_values=lambda X, ss=shift_space: _predict_shift_values(model_uq, X, ss),
                )

                results_by_space[shift_space]["groups"][group_name] = {
                    "indices": indices,
                    "metrics": metrics,
                }
                results_by_space[shift_space]["diagnostics"]["group_confidence"][group_name] = base_probs.max(axis=1)
                results_by_space[shift_space]["diagnostics"]["group_topk_ratio"][group_name] = topk_ratio

                for k, mae_pair, mse_pair in zip(REMOVE_COUNTS, metrics["mae"], metrics["mse"]):
                    summary_rows.append({
                        "dataset": dataset_name,
                        "model": model_spec["label"],
                        "shift_space": shift_space,
                        "group": group_name,
                        "k": int(k),
                        "mae_mean": float(mae_pair[0]),
                        "mae_std": float(mae_pair[1]),
                        "mse_mean": float(mse_pair[0]),
                        "mse_std": float(mse_pair[1]),
                    })

            print(f"OK: {group_name} group metrics computed ({len(indices)} samples)")

        for shift_space, results in results_by_space.items():
            output_file = RESULTS_DIR / (
                f"feature_removal_{dataset_slug}_{model_spec['key']}_{model_spec['explainer']}_{shift_space}.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
            print(f"OK: results saved: {output_file}")

            summary_file = RESULTS_DIR / (
                f"feature_removal_{dataset_slug}_{model_spec['key']}_{model_spec['explainer']}_{shift_space}.txt"
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"FEATURE REMOVAL SENSITIVITY - {dataset_name}\n")
                f.write("=" * 80 + "\n\n")

                f.write("Configuration:\n")
                f.write(f"  Model: {model_spec['label']}\n")
                if model_spec["explainer"] == "shap":
                    f.write(f"  Explainer: SHAP ({model_spec.get('shap_mode')})\n")
                else:
                    f.write(f"  Explainer: {model_spec['explainer']}\n")
                f.write(f"  STRAT_LOW_SAMPLES: {STRAT_LOW_SAMPLES}\n")
                f.write(f"  STRAT_MID_SAMPLES: {STRAT_MID_SAMPLES}\n")
                f.write(f"  STRAT_HIGH_SAMPLES: {STRAT_HIGH_SAMPLES}\n")
                f.write(f"  STRAT_MID_STRATEGY: {STRAT_MID_STRATEGY}\n")
                f.write(f"  REMOVE_COUNTS: {REMOVE_COUNTS}\n")
                f.write(f"  IMPUTE_STRATEGY: {IMPUTE_STRATEGY}\n")
                f.write(f"  ALLOW_RANDOM_OVERLAP: {ALLOW_RANDOM_OVERLAP}\n")
                f.write(f"  RANDOM_REPEATS: {RANDOM_REPEATS}\n")
                f.write(f"  SHIFT_SPACE: {shift_space}\n")
                if shift_space == "logits":
                    f.write(f"  LOGITS_REF_INDEX: {LOGITS_REF_INDEX}\n")
                f.write("\n")

                f.write("Epistemic summary (mean ± std):\n")
                for group_name, stats in group_stats.items():
                    if group_name == "random":
                        f.write(
                            f"  {group_name:>6}: {stats['epistemic_mean']:.4f} ± {stats['epistemic_std']:.4f} "
                            f"(n={stats['n']} x{stats['n_repeats']})\n"
                        )
                    else:
                        f.write(
                            f"  {group_name:>6}: {stats['epistemic_mean']:.4f} ± {stats['epistemic_std']:.4f} "
                            f"(n={stats['n']})\n"
                        )

                for group_name, payload in results["groups"].items():
                    _write_group_table(f, group_name, REMOVE_COUNTS, payload["metrics"])

            print(f"OK: summary saved: {summary_file}")
            _append_diagnostic_rows(diagnostic_rows, results)

if summary_rows:
    dataset_order = [ds.name for ds in DATASETS]
    model_order = [spec["label"] for spec in MODEL_SPECS]
    for shift_space in SHIFT_SPACES:
        filtered_rows = [row for row in summary_rows if row.get("shift_space", "probs") == shift_space]
        if not filtered_rows:
            continue
        _print_section(f"SUMMARY TABLE - {shift_space}")
        header = (
            f"{'Dataset':<12} {'Model':<20} {'Group':<6} {'k':>2} | "
            f"{'MAE (mean±std)':>19} | {'MSE (mean±std)':>19}"
        )
        print(header)
        print("-" * len(header))
        for row in filtered_rows:
            print(
                f"{row['dataset']:<12} {row['model']:<20} {row['group']:<6} {row['k']:>2} | "
                f"{row['mae_mean']:>9.4f} ±{row['mae_std']:>8.4f} | "
                f"{row['mse_mean']:>9.4f} ±{row['mse_std']:>8.4f}"
            )

        summary_path = RESULTS_DIR / f"feature_removal_summary_{shift_space}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in filtered_rows:
                f.write(
                    f"{row['dataset']:<12} {row['model']:<20} {row['group']:<6} {row['k']:>2} | "
                    f"{row['mae_mean']:>9.4f} ±{row['mae_std']:>8.4f} | "
                    f"{row['mse_mean']:>9.4f} ±{row['mse_std']:>8.4f}\n"
                )
        print(f"OK: summary table saved: {summary_path}")

        plot_path = RESULTS_DIR / f"feature_removal_mse_bars_{shift_space}.pdf"
        _plot_mse_bars(filtered_rows, dataset_order, model_order, plot_path)
        print(f"OK: MSE plot saved: {plot_path}")

if diagnostic_rows:
    dataset_order = [ds.name for ds in DATASETS]
    model_order = [spec["label"] for spec in MODEL_SPECS]
    for shift_space in SHIFT_SPACES:
        conf_plot_path = RESULTS_DIR / f"feature_removal_confidence_violin_{shift_space}.pdf"
        _plot_group_violin(
            diagnostic_rows,
            dataset_order,
            model_order,
            shift_space,
            conf_plot_path,
            value_key="group_confidence",
            ylabel="Max probability",
        )
        print(f"OK: confidence plot saved: {conf_plot_path}")

        topk_plot_path = RESULTS_DIR / f"feature_removal_topk_ratio_violin_{shift_space}.pdf"
        _plot_group_violin(
            diagnostic_rows,
            dataset_order,
            model_order,
            shift_space,
            topk_plot_path,
            value_key="group_topk_ratio",
            ylabel=f"Top-{DIAG_TOPK} |SHAP| ratio",
        )
        print(f"OK: top-k ratio plot saved: {topk_plot_path}")
