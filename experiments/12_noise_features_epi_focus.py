"""Compare noise features vs epistemic focus.

Adds synthetic Gaussian noise features at multiple ratios and tests whether
low-epistemic explanations focus on signal features more than high-epistemic
or random samples.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from data.cache import Cache
from data.datasets import WineQualityDataset, DryBeanDataset, RiceDataset, DatasetInfo
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from uncertainty.linear_uq import LogisticUQ
from config.settings import GLOBAL_SEED, XAI_CONFIG

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

DATASET_SPECS = [
    ("wine_binary", WineQualityDataset()),
    ("bean", DryBeanDataset()),
    ("rice", RiceDataset()),
]
MODEL_SPECS = [
    {
        "key": "rf",
        "label": "RF",
        "uq_cls": RandomForestClassifierUQ,
        "shap": "tree",
    },
    {
        "key": "logreg",
        "label": "LogReg",
        "uq_cls": LogisticUQ,
        "shap": "kernel",
    },
]
DATASET_LABELS = {
    "wine_binary": "Wine",
    "bean": "Bean",
    "rice": "Rice",
}
MODEL_LABELS = {
    "RF": "Random Forest",
    "LogReg": "Logistic Regression",
}
NOISE_FEATURE_RATIOS = [1.0, 2.0, 3.0]
NOISE_FEATURE_RATIOS_BY_MODEL = {
    "rf": [1.0, 2.0, 3.0, 5.0, 10],
    "logreg": [1.0, 2.0, 3.0],
}
NOISE_SEED = GLOBAL_SEED + 123

EPI_SELECT_FRACTION = 0.1
MIN_SELECT_N = 50
MAX_SELECT_N = 50
TOP_K_EXTRA = 0
RANDOM_SAMPLE_SEED = GLOBAL_SEED + 456
RANDOM_REPEATS = 5

SEED = GLOBAL_SEED

REQUIRED_RESULT_KEYS = {
    "dataset_name",
    "model_label",
    "n_noise_features",
    "noise_feature_ratio",
    "random_repeats",
    "signal_low",
    "noise_low",
    "signal_high",
    "noise_high",
    "signal_rand",
    "noise_rand",
    "noise_ratio_low",
    "noise_ratio_high",
    "noise_ratio_rand",
    "importance_low",
    "importance_high",
    "importance_rand",
    "feature_names",
    "top_k",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HELPERS

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")


def add_gaussian_noise_features(info: DatasetInfo,
                                n_noise: int,
                                seed: int) -> DatasetInfo:
    """Append Gaussian noise features to a dataset."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 1.0, size=(info.X.shape[0], n_noise))
    X_aug = np.hstack([info.X, noise])
    noise_names = [f"noise_{i+1:02d}" for i in range(n_noise)]
    return DatasetInfo(
        X=X_aug,
        y=info.y,
        feature_names=info.feature_names + noise_names,
        task_type=info.task_type,
        name=info.name,
        perturbation=f"gaussian_noise_{n_noise}",
        class_names=info.class_names,
    )


def pick_select_n(n_samples: int,
                  fraction: float,
                  min_n: int,
                  max_n: int) -> int:
    """Pick a robust per-group sample size for low/high-epi sets."""
    if n_samples < 2:
        return 0
    max_n = min(max_n, n_samples // 2)
    n = int(round(n_samples * fraction))
    n = max(min_n, n)
    n = min(max_n, n)
    return max(1, n)


def compute_global_importance(shap_values: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature."""
    return np.abs(shap_values).mean(axis=0)


def signal_noise_mass(importance: np.ndarray,
                      signal_idx: np.ndarray,
                      noise_idx: np.ndarray) -> tuple:
    """Return signal and noise attribution mass fractions."""
    signal_mass = float(importance[signal_idx].sum())
    noise_mass = float(importance[noise_idx].sum())
    total = signal_mass + noise_mass
    if total <= 0:
        return 0.0, 0.0
    return signal_mass / total, noise_mass / total


def per_sample_noise_ratio(shap_values: np.ndarray,
                           noise_idx: np.ndarray) -> np.ndarray:
    """Noise attribution share per sample."""
    abs_vals = np.abs(shap_values)
    total = abs_vals.sum(axis=1)
    noise = abs_vals[:, noise_idx].sum(axis=1)
    return np.divide(noise, total, out=np.zeros_like(noise), where=total > 0)


def top_features(importance: np.ndarray, feature_names, k: int):
    """Return top-k features by importance."""
    idx = np.argsort(importance)[::-1][:k]
    return [(feature_names[i], float(importance[i])) for i in idx]


def count_noise_in_top_k(importance: np.ndarray,
                         feature_names,
                         k: int) -> int:
    """Count noise features in top-k by importance."""
    top = top_features(importance, feature_names, k)
    return sum(1 for name, _ in top if str(name).startswith("noise_"))


def build_summary_row(dataset: str,
                      model: str,
                      noise_ratio: float,
                      signal_low: float,
                      noise_low: float,
                      signal_high: float,
                      noise_high: float,
                      signal_rand: float,
                      noise_rand: float,
                      top_k: int,
                      top_low_noise: int,
                      top_high_noise: int,
                      top_rand_noise: int) -> dict:
    """Build a summary row dictionary."""
    model_display = MODEL_LABELS.get(model, model)
    return {
        "dataset": dataset,
        "model": model_display,
        "noise_ratio": f"{noise_ratio:.1f}",
        "mass_low": f"{signal_low:.3f}/{noise_low:.3f}",
        "mass_high": f"{signal_high:.3f}/{noise_high:.3f}",
        "mass_rand": f"{signal_rand:.3f}/{noise_rand:.3f}",
        "topk_low": f"{top_low_noise}/{top_k}",
        "topk_high": f"{top_high_noise}/{top_k}",
        "topk_rand": f"{top_rand_noise}/{top_k}",
    }


def build_plot_row(dataset: str,
                   model: str,
                   noise_ratio: float,
                   signal_low: float,
                   noise_low: float,
                   signal_high: float,
                   noise_high: float,
                   signal_rand: float,
                   noise_rand: float) -> dict:
    """Build a row for aggregated plotting."""
    return {
        "dataset": dataset,
        "model": model,
        "noise_ratio": float(noise_ratio),
        "signal_low": float(signal_low),
        "noise_low": float(noise_low),
        "signal_high": float(signal_high),
        "noise_high": float(noise_high),
        "signal_rand": float(signal_rand),
        "noise_rand": float(noise_rand),
    }


def results_are_compatible(results: dict,
                           dataset_name: str,
                           model_label: str,
                           n_noise: int,
                           noise_ratio: float) -> bool:
    """Validate cached results for current run settings."""
    if not all(key in results for key in REQUIRED_RESULT_KEYS):
        return False
    if results.get("dataset_name") != dataset_name:
        return False
    if results.get("model_label") != model_label:
        return False
    if results.get("n_noise_features") != n_noise:
        return False
    if results.get("random_repeats") != RANDOM_REPEATS:
        return False
    ratio_val = results.get("noise_feature_ratio")
    if ratio_val is not None and abs(float(ratio_val) - float(noise_ratio)) > 1e-6:
        return False
    return True


def normalize_ratio(value: float) -> float:
    return float(f"{float(value):.1f}")


def get_model_noise_ratios(model_spec) -> list:
    ratios = NOISE_FEATURE_RATIOS_BY_MODEL.get(model_spec["key"], NOISE_FEATURE_RATIOS)
    return [normalize_ratio(r) for r in ratios]


def _coerce_axes(axes, n_rows: int, n_cols: int):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.array([axes])
    if n_cols == 1:
        return np.array([[ax] for ax in axes])
    return axes


def build_shap_explainer(model_uq, model_spec, X_train: np.ndarray) -> SHAPExplainer:
    """Create SHAP explainer for a model spec."""
    if model_spec["shap"] == "tree":
        return SHAPExplainer(model_uq.base_model)
    if model_spec["shap"] == "kernel":
        background_n = min(XAI_CONFIG["shap_background_samples"], len(X_train))
        X_background = X_train[:background_n]
        return SHAPExplainer(model_uq.base_model, X_background=X_background)
    raise ValueError(f"Unknown SHAP mode: {model_spec['shap']}")

def build_summary_table(rows) -> str:
    """Build an ASCII summary table."""
    if not rows:
        return "No summary rows."

    headers = [
        "Dataset",
        "Model",
        "NoiseRatio",
        "Mass Low (S/N)",
        "Mass High (S/N)",
        "Mass Rand (S/N)",
        "TopK Noise Low",
        "TopK Noise High",
        "TopK Noise Rand",
    ]

    values = [headers]
    for row in rows:
        values.append([
            row["dataset"],
            row["model"],
            row["noise_ratio"],
            row["mass_low"],
            row["mass_high"],
            row["mass_rand"],
            row["topk_low"],
            row["topk_high"],
            row["topk_rand"],
        ])

    widths = []
    for col in range(len(headers)):
        widths.append(max(len(str(values[row_idx][col])) for row_idx in range(len(values))))

    def fmt_line(items):
        return " | ".join(str(items[i]).ljust(widths[i]) for i in range(len(widths)))

    header_line = fmt_line(headers)
    sep_line = "-+-".join("-" * width for width in widths)
    data_lines = [fmt_line(row) for row in values[1:]]

    return "\n".join([header_line, sep_line] + data_lines)


def plot_aggregated_mass_bars(rows,
                              dataset_specs,
                              model_specs,
                              model_ratio_map,
                              output_path: Path) -> None:
    if not rows:
        return

    data = {}
    for row in rows:
        key = (row["dataset"], row["model"])
        data.setdefault(key, {})[float(row["noise_ratio"])] = row

    n_rows = len(dataset_specs)
    n_cols = len(model_specs)
    width_ratios = [
        max(1, len(model_ratio_map.get(spec["label"], NOISE_FEATURE_RATIOS)))
        for spec in model_specs
    ]
    base_width_per_ratio = 0.9
    fig_width = base_width_per_ratio * sum(width_ratios)
    fig_height = 1.2 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = _coerce_axes(axes, n_rows, n_cols)

    group_specs = [
        ("low", "Low", ""),
        ("rand", "Random", "///"),
        ("high", "High", "xxx"),
    ]
    signal_color = "#4C78A8"
    noise_color = "#E45756"
    bar_w = 0.24
    label_fontsize = 11

    for i, (dataset_name, _) in enumerate(dataset_specs):
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        for j, model_spec in enumerate(model_specs):
            model_label = model_spec["label"]
            model_display = MODEL_LABELS.get(model_label, model_label)
            ax = axes[i][j]
            key = (dataset_name, model_label)
            by_ratio = data.get(key, {})
            ratios = model_ratio_map.get(model_label, NOISE_FEATURE_RATIOS)
            ratios = [normalize_ratio(r) for r in ratios]
            x = np.arange(len(ratios))

            for idx, (suffix, _, hatch) in enumerate(group_specs):
                signal_vals = []
                noise_vals = []
                for ratio in ratios:
                    row = by_ratio.get(float(ratio))
                    if row is None:
                        signal_vals.append(0.0)
                        noise_vals.append(0.0)
                    else:
                        signal_vals.append(row[f"signal_{suffix}"])
                        noise_vals.append(row[f"noise_{suffix}"])
                positions = x + (idx - 1) * bar_w
                signal_vals = np.asarray(signal_vals)
                noise_vals = np.asarray(noise_vals)
                ax.bar(positions, signal_vals, width=bar_w, color=signal_color,
                       hatch=hatch, edgecolor="black", linewidth=0.3)
                ax.bar(positions, noise_vals, width=bar_w, bottom=signal_vals,
                       color=noise_color, hatch=hatch, edgecolor="black", linewidth=0.3)

            if i == 0:
                ax.set_title(model_display, fontsize=label_fontsize)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(round(r))}" for r in ratios])
            ax.grid(True, axis="y", alpha=0.25)
            if i == n_rows - 1:
                ax.set_xlabel("Noise Ratio", fontsize=label_fontsize)
            if j == 0:
                ax.set_ylabel(dataset_label)
            if j == 1:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    from matplotlib.patches import Patch
    signal_patch = Patch(facecolor=signal_color, label="Signal")
    noise_patch = Patch(facecolor=noise_color, label="Noise")
    group_patches = [
        Patch(facecolor="white", edgecolor="black", hatch=spec[2], label=spec[1])
        for spec in group_specs
    ]
    fig.suptitle("Noise Feature Attribution", y=0.95)
    fig.legend(handles=[signal_patch, noise_patch] + group_patches,
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.supylabel("Attribution Mass", fontsize=label_fontsize, x=0.05)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# MAIN

def main():
    _print_section("EXPERIMENT 12: NOISE FEATURES VS EPISTEMIC FOCUS")

    print("Configuration:")
    print(f"  Datasets: {[name for name, _ in DATASET_SPECS]}")
    print(f"  Models: {[MODEL_LABELS.get(spec['label'], spec['label']) for spec in MODEL_SPECS]}")
    print(f"  Noise feature ratios (default): {NOISE_FEATURE_RATIOS} (noise = ratio * signal)")
    print(f"  Noise feature ratios by model: {NOISE_FEATURE_RATIOS_BY_MODEL}")
    print(f"  Noise seed: {NOISE_SEED}")
    print(f"  Random repeats: {RANDOM_REPEATS}")
    print(f"  Epi select fraction: {EPI_SELECT_FRACTION}")
    print(f"  Select bounds: {MIN_SELECT_N}..{MAX_SELECT_N} per group")
    print(f"  Top-k extra features: {TOP_K_EXTRA} (top_k = n_signal + extra)")

    cache = Cache()
    splitter = DataSplitter()
    registry = ModelRegistry()
    summary_rows = []
    plot_rows = []
    model_ratio_map = {spec["label"]: get_model_noise_ratios(spec) for spec in MODEL_SPECS}

    for dataset_name, ds in DATASET_SPECS:
        all_ratios = sorted({
            normalize_ratio(ratio)
            for spec in MODEL_SPECS
            for ratio in get_model_noise_ratios(spec)
        })
        for noise_ratio in all_ratios:
            ratio_tag = f"{noise_ratio:.1f}".replace(".", "p")
            _print_section(f"DATASET: {dataset_name} | noise ratio {noise_ratio:.1f}")

            # LOAD DATASET AND APPEND NOISE FEATURES
            info = cache.load_or_create(ds.cache_key, ds.load)
            n_signal = len(info.feature_names)
            n_noise = int(round(n_signal * noise_ratio))
            n_noise = max(1, n_noise)
            info_noise = add_gaussian_noise_features(info, n_noise, NOISE_SEED)

            splits = splitter.split(info_noise)

            n_features = splits.X_train.shape[1]
            if n_signal + n_noise != n_features:
                print("WARNING: feature count mismatch after noise augmentation")

            feature_names = splits.feature_names
            signal_idx = np.arange(n_signal)
            noise_idx = np.arange(n_signal, n_features)

            print(f"Noise feature ratio: {noise_ratio:.1f} (noise = ratio * signal)")
            print(f"Samples: train={len(splits.X_train)}, val={len(splits.X_val)}, test={len(splits.X_test)}")
            print(f"Features: total={n_features}, signal={len(signal_idx)}, noise={len(noise_idx)}")

            for model_spec in MODEL_SPECS:
                model_label = model_spec["label"]
                model_ratios = model_ratio_map.get(model_label, NOISE_FEATURE_RATIOS)
                if normalize_ratio(noise_ratio) not in model_ratios:
                    continue
                model_display = MODEL_LABELS.get(model_label, model_label)
                print(f"\nModel: {model_display}")

                save_path = RESULTS_DIR / (f"{dataset_name}_{model_spec['key']}_noise{n_noise}"
                                           f"_ratio{ratio_tag}_noise_epi_focus.pkl")

                if save_path.exists():
                    with open(save_path, "rb") as f:
                        cached = pickle.load(f)
                    if results_are_compatible(cached, dataset_name, model_label, n_noise, noise_ratio):
                        print(f"OK Cached results found: {save_path}")
                        feature_names = cached["feature_names"]
                        top_k = int(cached["top_k"])
                        top_low_noise = count_noise_in_top_k(cached["importance_low"], feature_names, top_k)
                        top_high_noise = count_noise_in_top_k(cached["importance_high"], feature_names, top_k)
                        top_rand_noise = count_noise_in_top_k(cached["importance_rand"], feature_names, top_k)

                        summary_rows.append(build_summary_row(
                            dataset_name,
                            model_label,
                            noise_ratio,
                            cached["signal_low"],
                            cached["noise_low"],
                            cached["signal_high"],
                            cached["noise_high"],
                            cached["signal_rand"],
                            cached["noise_rand"],
                            top_k,
                            top_low_noise,
                            top_high_noise,
                            top_rand_noise,
                        ))
                        plot_rows.append(build_plot_row(
                            dataset_name,
                            model_label,
                            noise_ratio,
                            cached["signal_low"],
                            cached["noise_low"],
                            cached["signal_high"],
                            cached["noise_high"],
                            cached["signal_rand"],
                            cached["noise_rand"],
                        ))
                        continue
                    print(f"WARNING: Cached results missing keys or mismatch: {save_path}")

                # TRAIN OR LOAD MODEL ON AUGMENTED DATA
                model_uq = model_spec["uq_cls"]()
                dataset_key = (f"{info.name}_{ds.uci_id}_noise{n_noise}_ratio{ratio_tag}"
                               f"_seed{NOISE_SEED}")
                model_key = registry.make_key(dataset_key, model_uq.name)

                if registry.exists(model_key):
                    model_uq = registry.load(model_key)
                    print(f"OK Loaded model: {model_key}")
                else:
                    print(f"Training model: {model_key}")
                    model_uq.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
                    registry.save(model_key, model_uq)
                    print("OK Model trained and cached")

                # EPISTEMIC ON TEST SET
                _, _, epistemic = model_uq.predict_with_uncertainty(splits.X_test)
                print(f"Epistemic: mean={epistemic.mean():.4f}, std={epistemic.std():.4f}, "
                      f"min={epistemic.min():.4f}, max={epistemic.max():.4f}")

                n_select = pick_select_n(len(splits.X_test), EPI_SELECT_FRACTION, MIN_SELECT_N, MAX_SELECT_N)
                if n_select <= 0:
                    print("WARNING: Not enough samples to select low/high-epi groups")
                    continue

                sorted_idx = np.argsort(epistemic)
                low_idx = sorted_idx[:n_select]
                high_idx = sorted_idx[-n_select:]

                random_seed_base = RANDOM_SAMPLE_SEED + n_noise
                rand_seed_list = []
                rand_indices = []
                all_idx = np.arange(len(splits.X_test))
                remaining_idx = np.setdiff1d(all_idx, np.concatenate([low_idx, high_idx]))
                if len(remaining_idx) < n_select:
                    print("WARNING: Not enough remaining samples for random group; sampling from full test set")
                for repeat_idx in range(RANDOM_REPEATS):
                    seed = random_seed_base + repeat_idx
                    rng = np.random.RandomState(seed)
                    if len(remaining_idx) >= n_select:
                        rand_idx = rng.choice(remaining_idx, size=n_select, replace=False)
                    else:
                        rand_idx = rng.choice(all_idx, size=n_select, replace=False)
                    rand_indices.append(rand_idx)
                    rand_seed_list.append(seed)

                rand_concat = np.concatenate(rand_indices) if rand_indices else np.array([], dtype=int)

                print(f"Selected per group: {n_select} samples (random x{RANDOM_REPEATS})")
                print(f"Low-epi mean: {epistemic[low_idx].mean():.4f} | High-epi mean: {epistemic[high_idx].mean():.4f}")
                if rand_concat.size:
                    rand_mean = float(epistemic[rand_concat].mean())
                    rand_std = float(epistemic[rand_concat].std())
                    print(f"Random-epi mean: {rand_mean:.4f} ± {rand_std:.4f}")
                else:
                    print("Random-epi mean: n/a")

                # SHAP EXPLANATIONS
                explainer = build_shap_explainer(model_uq, model_spec, splits.X_train)
                shap_low = explainer.explain(splits.X_test[low_idx])
                shap_high = explainer.explain(splits.X_test[high_idx])
                rand_importances = []
                rand_noise_ratios = []
                for rand_idx in rand_indices:
                    shap_rand = explainer.explain(splits.X_test[rand_idx])
                    rand_importances.append(compute_global_importance(shap_rand))
                    rand_noise_ratios.append(per_sample_noise_ratio(shap_rand, noise_idx))

                # GLOBAL IMPORTANCE PER GROUP
                importance_low = compute_global_importance(shap_low)
                importance_high = compute_global_importance(shap_high)
                if rand_importances:
                    importance_rand = np.mean(np.stack(rand_importances, axis=0), axis=0)
                else:
                    importance_rand = np.zeros(n_features, dtype=float)

                signal_low, noise_low = signal_noise_mass(importance_low, signal_idx, noise_idx)
                signal_high, noise_high = signal_noise_mass(importance_high, signal_idx, noise_idx)
                signal_rand, noise_rand = signal_noise_mass(importance_rand, signal_idx, noise_idx)

                noise_ratio_low = per_sample_noise_ratio(shap_low, noise_idx)
                noise_ratio_high = per_sample_noise_ratio(shap_high, noise_idx)
                if rand_noise_ratios:
                    noise_ratio_rand = np.concatenate(rand_noise_ratios)
                else:
                    noise_ratio_rand = np.array([])

                # REPORTING
                print("\nSignal/Noise attribution mass:")
                print(f"  Low-epi:  signal={signal_low:.3f}, noise={noise_low:.3f}")
                print(f"  High-epi: signal={signal_high:.3f}, noise={noise_high:.3f}")
                print(f"  Random:   signal={signal_rand:.3f}, noise={noise_rand:.3f}")

                print("\nPer-sample noise ratio (mean ± std):")
                print(f"  Low-epi:  {noise_ratio_low.mean():.3f} ± {noise_ratio_low.std():.3f}")
                print(f"  High-epi: {noise_ratio_high.mean():.3f} ± {noise_ratio_high.std():.3f}")
                if noise_ratio_rand.size:
                    print(f"  Random:   {noise_ratio_rand.mean():.3f} ± {noise_ratio_rand.std():.3f}")
                else:
                    print("  Random:   n/a")

                top_k = max(1, min(n_features, n_signal + TOP_K_EXTRA))
                top_low = top_features(importance_low, feature_names, top_k)
                top_high = top_features(importance_high, feature_names, top_k)
                top_rand = top_features(importance_rand, feature_names, top_k)
                top_low_noise = sum(1 for name, _ in top_low if name.startswith("noise_"))
                top_high_noise = sum(1 for name, _ in top_high if name.startswith("noise_"))
                top_rand_noise = sum(1 for name, _ in top_rand if name.startswith("noise_"))

                print(f"\nTop-{top_k} features (low-epi):")
                for name, val in top_low:
                    print(f"  {name:<14} {val:.4f}")
                print(f"Noise features in top-{top_k} (low-epi): {top_low_noise}")

                print(f"\nTop-{top_k} features (high-epi):")
                for name, val in top_high:
                    print(f"  {name:<14} {val:.4f}")
                print(f"Noise features in top-{top_k} (high-epi): {top_high_noise}")

                print(f"\nTop-{top_k} features (random):")
                for name, val in top_rand:
                    print(f"  {name:<14} {val:.4f}")
                print(f"Noise features in top-{top_k} (random): {top_rand_noise}")

                summary_rows.append({
                    **build_summary_row(
                        dataset_name,
                        model_label,
                        noise_ratio,
                        signal_low,
                        noise_low,
                        signal_high,
                        noise_high,
                        signal_rand,
                        noise_rand,
                        top_k,
                        top_low_noise,
                        top_high_noise,
                        top_rand_noise,
                    )
                })
                plot_rows.append(build_plot_row(
                    dataset_name,
                    model_label,
                    noise_ratio,
                    signal_low,
                    noise_low,
                    signal_high,
                    noise_high,
                    signal_rand,
                    noise_rand,
                ))

                # SAVE RESULTS
                results = {
                    "dataset_name": dataset_name,
                    "model_label": model_label,
                    "model_key": model_key,
                    "n_signal_features": n_signal,
                    "n_noise_features": n_noise,
                    "noise_feature_ratio": noise_ratio,
                    "noise_seed": NOISE_SEED,
                    "random_sample_seed": RANDOM_SAMPLE_SEED + n_noise,
                    "random_repeats": RANDOM_REPEATS,
                    "random_seed_list": rand_seed_list,
                    "n_select": n_select,
                    "top_k": top_k,
                    "signal_low": signal_low,
                    "noise_low": noise_low,
                    "signal_high": signal_high,
                    "noise_high": noise_high,
                    "signal_rand": signal_rand,
                    "noise_rand": noise_rand,
                    "noise_ratio_low": noise_ratio_low,
                    "noise_ratio_high": noise_ratio_high,
                    "noise_ratio_rand": noise_ratio_rand,
                    "importance_low": importance_low,
                    "importance_high": importance_high,
                    "importance_rand": importance_rand,
                    "feature_names": feature_names,
                }

                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"\nOK Results saved: {save_path}")

    if plot_rows:
        plot_path = RESULTS_DIR / "summary_mass_bars.pdf"
        plot_aggregated_mass_bars(
            plot_rows,
            DATASET_SPECS,
            MODEL_SPECS,
            model_ratio_map,
            plot_path,
        )
        print(f"OK Summary plot saved: {plot_path}")

    _print_section("SUMMARY TABLE")
    print(build_summary_table(summary_rows))


if __name__ == "__main__":
    main()
