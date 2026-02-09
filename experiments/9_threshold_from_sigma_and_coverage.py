"""Calibrate thresholds under unknown noise.

Evaluates epistemic thresholds across sigma values and coverages using SHAP
stability, and saves sigma-by-coverage results and plots.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import pickle
import matplotlib.patheffects as patheffects
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path
from tqdm import tqdm

from data.cache import Cache
from data.datasets import (
    WineQualityDataset,
    DryBeanDataset,
    RiceDataset,
)
from data.splitter import DataSplitter
from models.registry import ModelRegistry
from explainers.shap_explainer import SHAPExplainer
from uncertainty.forest_uq import RandomForestClassifierUQ
from data.perturbations import PerturbationGenerator
from config.settings import GLOBAL_SEED

# CONFIGURATION

np.random.seed(GLOBAL_SEED)

SIGMAS = np.round(np.arange(0.02, 0.21, 0.02), 2)
TARGET_COVERAGES = np.round(np.arange(0.3, 0.71, 0.05), 2)

METRIC_NAME = "kendall"  # kendall, spearman, topk_overlap, jaccard
HEATMAP_VALUE = "mean_stability_accepted"  # f1, precision, recall, coverage, threshold, k_value, mean_stability_accepted, mean_stability_rejected, roc_auc

CONTOUR_VALUE = "k_value"  # threshold, k_value, or None
CONTOUR_LEVELS = None  # Optional list of levels, e.g. [0.02, 0.05, 0.1]

EPIS_COVERAGE_LINES = [0.3, 0.5, 0.7]  # coverage markers on epistemic plots
EPIS_COVERAGE_STATS = [i * 0.1 for i in range(11)]
EPIS_SCATTER_PLOT = True
EPIS_SCATTER_ALPHA = 0.7
EPIS_SCATTER_SIZE = 16
EPIS_SCATTER_MARGINALS = True
EPIS_SCATTER_MARGINAL_KIND = "kde"  # "kde" or "hist"
EPIS_SCATTER_MARGINAL_BINS_X = 50
EPIS_SCATTER_MARGINAL_BINS_Y = 15
EPIS_SCATTER_MARGINAL_Y_KIND = "kde"  # "kde", "hist", or "none" when X uses KDE
PLOT_AMBIGUOUS = True

SCATTER_SIGMA = 0.15  # None -> middle of SIGMAS
SCATTER_ALL_SIGMAS = True

K_NOISE_SEEDS = 5
MAX_SAMPLES = 500
TOP_K = 5

STABILITY_METRICS = {
    'kendall': {
        'stable_threshold': 0.7,
        'unstable_threshold': 0.7
    }
}

ALLOWED_VALUES = {
    "f1",
    "precision",
    "recall",
    "coverage",
    "threshold",
    "k_value",
    "mean_stability_accepted",
    "mean_stability_rejected",
    "roc_auc",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / Path(__file__).stem
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HELPERS

def _stat_corr(stat_result):
    """Extract correlation from SciPy stats result across versions."""
    if hasattr(stat_result, "correlation"):
        return stat_result.correlation
    return stat_result[0]

def _print_section(title):
    print(f"\n===== {title} =====")

def _metric_from_attr(metric_name, rank_clean, pos_clean, attr_pert, top_k_used):
    n_samples = attr_pert.shape[0]
    values = np.zeros(n_samples)

    for i in range(n_samples):
        rank_pert = np.argsort(np.abs(attr_pert[i]))[::-1]
        if metric_name in ("kendall", "spearman"):
            pos_pert = np.empty(rank_pert.shape[0])
            pos_pert[rank_pert] = np.arange(rank_pert.shape[0])

            if metric_name == "kendall":
                val = _stat_corr(kendalltau(pos_clean[i], pos_pert))
            else:
                val = _stat_corr(spearmanr(pos_clean[i], pos_pert))
        elif metric_name == "topk_overlap":
            top_clean = rank_clean[i][:top_k_used]
            top_pert = rank_pert[:top_k_used]
            intersection = np.intersect1d(top_clean, top_pert).size
            val = intersection / top_k_used
        elif metric_name == "jaccard":
            top_clean = rank_clean[i][:top_k_used]
            top_pert = rank_pert[:top_k_used]
            intersection = np.intersect1d(top_clean, top_pert).size
            union = (2 * top_k_used - intersection)
            val = intersection / union if union > 0 else 0.0
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

        if np.isnan(val):
            val = 0.0
        values[i] = val

    return values

def _plot_heatmap(matrix, sigmas, coverages, value_name, metric_name, dataset_name, out_file, contour_matrix):
    fig, ax = plt.subplots(figsize=(15, 5))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis
    cmap.set_bad(color="lightgray")

    im = ax.imshow(
        masked,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{s:.2f}" for s in sigmas], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(coverages)))
    ax.set_yticklabels([f"{c:.0%}" for c in coverages])

    ax.set_xlabel("Sigma")
    ax.set_ylabel("Target coverage")
    display_name = _display_dataset_name(dataset_name)
    ax.set_title(f"{display_name}: {metric_name} / {value_name}")

    if CONTOUR_VALUE is not None and contour_matrix is not None:
        contour_data = np.ma.masked_invalid(contour_matrix)
        if contour_data.count() > 0:
            x_grid, y_grid = np.meshgrid(np.arange(len(sigmas)), np.arange(len(coverages)))
            contours = ax.contour(
                x_grid,
                y_grid,
                contour_data,
                levels=CONTOUR_LEVELS,
                colors="white",
                linewidths=0.8,
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.3f")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_name)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _display_dataset_name(dataset_name):
    name = dataset_name.lower()
    if "wine" in name:
        return "Wine"
    if "bean" in name:
        return "Bean"
    if "rice" in name:
        return "Rice"
    return dataset_name

def _add_coverage_lines(ax, epistemic_values, labels, coverage_values):
    mask = labels != "ambiguous"
    if not np.any(mask):
        return
    coverage_lines = []
    filtered = epistemic_values[mask]
    for cov in coverage_values:
        if cov <= 0 or cov >= 1:
            continue
        x = float(np.quantile(filtered, cov))
        if np.isclose(cov, 0.5):
            style = ":"
        elif np.isclose(cov, 0.4) or np.isclose(cov, 0.6):
            style = ":"
        else:
            style = ":"
        ax.axvline(x, color="black", linestyle=style, linewidth=2)
        coverage_lines.append((x, cov))

    if coverage_lines:
        for (x, cov) in coverage_lines:
            y_frac = 0.4
            nu = 1.0 - cov
            text = ax.text(
                x,
                y_frac,
                rf"$\nu$={nu:.1f}",
                rotation=90,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                bbox=dict(facecolor="none", edgecolor="none", boxstyle="square,pad=0.1"),
                transform=ax.get_xaxis_transform(),
            )
            text.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="white")]
            )

def _add_stability_threshold_line(ax, metric_name):
    thresholds = STABILITY_METRICS.get(metric_name)
    if not thresholds:
        return
    stable_threshold = thresholds.get("stable_threshold")
    if stable_threshold is None:
        return
    ax.axhline(stable_threshold, color="black", linestyle="--", linewidth=2)
    text = ax.text(
        0.98,
        stable_threshold,
        f"stability threshold {stable_threshold:.1f}",
        ha="right",
        va="center",
        color="black",
        fontsize=9,
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="square,pad=0.1"),
        transform=ax.get_yaxis_transform(),
    )
    text.set_path_effects(
        [patheffects.withStroke(linewidth=3, foreground="white")]
    )

def _scatter_marginal_y_kind():
    if not EPIS_SCATTER_MARGINALS:
        return None
    if EPIS_SCATTER_MARGINAL_KIND == "kde":
        return EPIS_SCATTER_MARGINAL_Y_KIND
    return EPIS_SCATTER_MARGINAL_KIND

def _create_scatter_axes(fig, subplot_spec=None):
    y_kind = _scatter_marginal_y_kind()
    if EPIS_SCATTER_MARGINALS:
        if y_kind == "none":
            if subplot_spec is None:
                gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 4], hspace=0.0)
            else:
                gs = GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=subplot_spec, height_ratios=[1.2, 4], hspace=0.0
                )
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = None
        else:
            if subplot_spec is None:
                gs = fig.add_gridspec(
                    2,
                    2,
                    width_ratios=[4, 0.25],
                    height_ratios=[0.5, 4],
                    wspace=0.0,
                    hspace=0.0,
                )
            else:
                gs = GridSpecFromSubplotSpec(
                    2,
                    2,
                    subplot_spec=subplot_spec,
                    width_ratios=[4, 0.25],
                    height_ratios=[0.5, 4],
                    wspace=0.0,
                    hspace=0.0,
                )
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        ax_histx.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_histx.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_histx.grid(False)
        ax_histx.set_frame_on(False)
        if ax_histy is not None:
            ax_histy.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax_histy.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax_histy.grid(False)
            ax_histy.set_frame_on(False)
    else:
        if subplot_spec is None:
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(subplot_spec)
        ax_histx = None
        ax_histy = None

    return ax, ax_histx, ax_histy, y_kind

def _scatter_panel(
    epistemic_values,
    stability_values,
    labels,
    coverage_values,
    dataset_name,
    metric_name,
    sigma,
    ax,
    ax_histx,
    ax_histy,
    y_kind,
):
    stable_mask = labels == "stable"
    unstable_mask = labels == "unstable"
    ambiguous_mask = labels == "ambiguous"

    n_stable = int(stable_mask.sum())
    n_unstable = int(unstable_mask.sum())
    n_ambiguous = int(ambiguous_mask.sum())

    if stable_mask.any():
        ax.scatter(
            epistemic_values[stable_mask],
            stability_values[stable_mask],
            s=EPIS_SCATTER_SIZE,
            color="forestgreen",
            alpha=EPIS_SCATTER_ALPHA,
            label=f"stable (n={n_stable})",
        )
    if unstable_mask.any():
        ax.scatter(
            epistemic_values[unstable_mask],
            stability_values[unstable_mask],
            s=EPIS_SCATTER_SIZE,
            color="crimson",
            alpha=EPIS_SCATTER_ALPHA,
            label=f"unstable (n={n_unstable})",
        )
    if PLOT_AMBIGUOUS and ambiguous_mask.any():
        ax.scatter(
            epistemic_values[ambiguous_mask],
            stability_values[ambiguous_mask],
            s=EPIS_SCATTER_SIZE,
            color="gold",
            alpha=EPIS_SCATTER_ALPHA,
            label=f"ambiguous (n={n_ambiguous})",
        )

    if epistemic_values.size > 0:
        x_min = float(np.min(epistemic_values))
        x_max = float(np.max(epistemic_values))
        x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 1.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.0, 1.05)

    if EPIS_SCATTER_MARGINALS and ax_histx is not None:
        def add_marginal(ax_marg, values, color, orientation, kind):
            if values.size == 0:
                return
            if kind == "none":
                return
            if kind == "kde":
                if values.size < 2 or np.std(values) == 0:
                    return
                v_min = float(np.min(values))
                v_max = float(np.max(values))
                if v_max <= v_min:
                    return
                grid = np.linspace(v_min, v_max, 200)
                kde = gaussian_kde(values)
                dens = kde(grid)
                if orientation == "x":
                    ax_marg.fill_between(grid, 0, dens, color=color, alpha=0.25)
                    ax_marg.plot(grid, dens, color=color, linewidth=1.2)
                else:
                    ax_marg.fill_betweenx(grid, 0, dens, color=color, alpha=0.25)
                    ax_marg.plot(dens, grid, color=color, linewidth=1.2)
            else:
                if orientation == "x":
                    bins = EPIS_SCATTER_MARGINAL_BINS_X
                    ax_marg.hist(values, bins=bins, density=True, color=color, alpha=0.4)
                else:
                    bins = EPIS_SCATTER_MARGINAL_BINS_Y
                    ax_marg.hist(
                        values,
                        bins=bins,
                        density=True,
                        color=color,
                        alpha=0.4,
                        orientation="horizontal",
                    )

        if stable_mask.any():
            add_marginal(ax_histx, epistemic_values[stable_mask], "forestgreen", "x", EPIS_SCATTER_MARGINAL_KIND)
            if ax_histy is not None:
                add_marginal(ax_histy, stability_values[stable_mask], "forestgreen", "y", y_kind)
        if unstable_mask.any():
            add_marginal(ax_histx, epistemic_values[unstable_mask], "crimson", "x", EPIS_SCATTER_MARGINAL_KIND)
            if ax_histy is not None:
                add_marginal(ax_histy, stability_values[unstable_mask], "crimson", "y", y_kind)
        if PLOT_AMBIGUOUS and ambiguous_mask.any():
            add_marginal(ax_histx, epistemic_values[ambiguous_mask], "gold", "x", EPIS_SCATTER_MARGINAL_KIND)
            if ax_histy is not None:
                add_marginal(ax_histy, stability_values[ambiguous_mask], "gold", "y", y_kind)

    _add_coverage_lines(ax, epistemic_values, labels, coverage_values)
    _add_stability_threshold_line(ax, metric_name)


    display_name = _display_dataset_name(dataset_name)
    #title = f"{display_name}: Epistemic vs {y_label}"
    title = display_name
    ax.set_xlabel("Epistemic uncertainty", fontsize=12)
    ax.set_ylabel("SHAP τ", fontsize=12)
    ax.grid(alpha=0.2)
    if stable_mask.any() or unstable_mask.any() or (PLOT_AMBIGUOUS and ambiguous_mask.any()):
        ax.legend(loc="best", fontsize=8)

    if ax_histx is not None:
        ax_histx.set_title(title, fontsize=14, pad=2)
    else:
        ax.set_title(title)

def _plot_epistemic_stability_scatter(
    epistemic_values,
    stability_values,
    labels,
    coverage_values,
    dataset_name,
    metric_name,
    sigma,
    out_file,
):
    fig = plt.figure(figsize=(8, 5) if EPIS_SCATTER_MARGINALS else (7, 4))
    ax, ax_histx, ax_histy, y_kind = _create_scatter_axes(fig)
    _scatter_panel(
        epistemic_values,
        stability_values,
        labels,
        coverage_values,
        dataset_name,
        metric_name,
        sigma,
        ax,
        ax_histx,
        ax_histy,
        y_kind,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fmt_metric(value):
    if value is None or not np.isfinite(value):
        return "   nan"
    return f"{value:7.3f}"

def _print_scatter_coverage_metrics(
    dataset_name,
    epistemic_values,
    labels,
    coverage_values,
    sigma_label,
):
    mask = labels != "ambiguous"
    if not np.any(mask):
        print(f"No labeled samples for scatter metrics: {dataset_name}")
        return
    epistemic_filtered = epistemic_values[mask]
    labels_binary = (labels[mask] == "stable").astype(int)
    if epistemic_filtered.size == 0:
        print(f"No epistemic values for scatter metrics: {dataset_name}")
        return

    print(f"\nScatter metrics by coverage ({dataset_name}, sigma={sigma_label})")
    print("coverage  precision  recall  f1")
    for cov in coverage_values:
        if cov <= 0 or cov >= 1:
            continue
        threshold = np.quantile(epistemic_filtered, cov)
        predictions = (epistemic_filtered <= threshold).astype(int)
        precision = precision_score(labels_binary, predictions, zero_division=0)
        recall = recall_score(labels_binary, predictions, zero_division=0)
        f1 = f1_score(labels_binary, predictions, zero_division=0)
        print(f"{cov:8.2f}  {_fmt_metric(precision)}  {_fmt_metric(recall)}  {_fmt_metric(f1)}")

def _plot_combined_scatter(scatter_items, out_file):
    if not scatter_items:
        return

    panel_width = 5 if EPIS_SCATTER_MARGINALS else 5
    panel_height = 3 if EPIS_SCATTER_MARGINALS else 3
    fig = plt.figure(
        figsize=(max(8, panel_width * len(scatter_items)), panel_height)
    )
    outer_gs = fig.add_gridspec(1, len(scatter_items), wspace=0.05)

    for idx, item in enumerate(scatter_items):
        ax, ax_histx, ax_histy, y_kind = _create_scatter_axes(fig, outer_gs[0, idx])
        _scatter_panel(
            item["epistemic_values"],
            item["stability_values"],
            item["labels"],
            EPIS_COVERAGE_LINES,
            item["dataset_name"],
            item["metric_name"],
            item["sigma_label"],
            ax,
            ax_histx,
            ax_histy,
            y_kind,
        )
        if idx > 0:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.set_ylabel("")
            if ax_histy is not None:
                ax_histy.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

# SETUP

if METRIC_NAME not in STABILITY_METRICS:
    raise ValueError(f"Unknown metric: {METRIC_NAME}")

if EPIS_SCATTER_MARGINAL_KIND not in {"kde", "hist"}:
    raise ValueError(f"Unknown EPIS_SCATTER_MARGINAL_KIND: {EPIS_SCATTER_MARGINAL_KIND}")

if EPIS_SCATTER_MARGINAL_Y_KIND not in {"kde", "hist", "none"}:
    raise ValueError(f"Unknown EPIS_SCATTER_MARGINAL_Y_KIND: {EPIS_SCATTER_MARGINAL_Y_KIND}")

if HEATMAP_VALUE not in ALLOWED_VALUES:
    raise ValueError(f"Unknown heatmap value: {HEATMAP_VALUE}")

if CONTOUR_VALUE is not None and CONTOUR_VALUE not in ALLOWED_VALUES:
    raise ValueError(f"Unknown contour value: {CONTOUR_VALUE}")

cache = Cache()
registry = ModelRegistry()
splitter = DataSplitter()

datasets = [
    WineQualityDataset(),
    DryBeanDataset(),
    RiceDataset(),
]
COMBINED_SCATTER_DATASETS = ["wine", "bean"]  # ["bean", "wine", "rice"]
COMBINED_SCATTER_ITEMS = []

def _include_in_combined(dataset_name, dataset_slug):
    if COMBINED_SCATTER_DATASETS is None:
        return True
    allowed = [str(item).lower() for item in COMBINED_SCATTER_DATASETS]
    name = dataset_name.lower()
    slug = dataset_slug.lower()
    return any(token in name or token in slug for token in allowed)

def run_dataset(ds):
    info = cache.load_or_create(ds.cache_key, ds.load)
    dataset_name = info.name
    dataset_slug = dataset_name.replace(" ", "_")
    dataset_key = f"{dataset_name}_{ds.uci_id}"
    splits = cache.load_or_create(f"{dataset_key}/splits", lambda: splitter.split(info))

    _print_section(f"SIGMA x COVERAGE: {dataset_name}")
    print(f"Metric: {METRIC_NAME}, Heatmap value: {HEATMAP_VALUE}")

    # SUBSAMPLE TEST SET
    N_SAMPLES = min(MAX_SAMPLES, len(splits.X_test))
    if N_SAMPLES < len(splits.X_test):
        indices = np.random.RandomState(GLOBAL_SEED).choice(
            len(splits.X_test),
            N_SAMPLES,
            replace=False
        )
        X_test = splits.X_test[indices]
    else:
        X_test = splits.X_test

    n_features = X_test.shape[1] if X_test.ndim > 1 else 1
    top_k_used = min(TOP_K, n_features)

    config = {
        "dataset_key": dataset_key,
        "sigmas": [float(s) for s in SIGMAS],
        "target_coverages": [float(c) for c in TARGET_COVERAGES],
        "metric": METRIC_NAME,
        "heatmap_value": "f1",
        "contour_value": CONTOUR_VALUE,
        "contour_levels": CONTOUR_LEVELS,
        "matrix_orientation": "coverage_x_sigma",
        "k_noise_seeds": K_NOISE_SEEDS,
        "max_samples": MAX_SAMPLES,
        "n_samples": N_SAMPLES,
        "top_k": top_k_used,
        "global_seed": GLOBAL_SEED,
        "scatter_sigma": (
            float(SCATTER_SIGMA)
            if SCATTER_SIGMA is not None
            else float(SIGMAS[len(SIGMAS) // 2])
        ),
        "scatter_all_sigmas": SCATTER_ALL_SIGMAS,
    }

    output_file = RESULTS_DIR / f"{dataset_slug}_sigma_coverage_grid_{METRIC_NAME}_{HEATMAP_VALUE}.pkl"
    plot_file = RESULTS_DIR / f"{dataset_slug}_sigma_coverage_{METRIC_NAME}_{HEATMAP_VALUE}.pdf"

    dist_sigma = config["scatter_sigma"]
    dist_sigma_label = "all" if SCATTER_ALL_SIGMAS else f"{dist_sigma:.2f}"
    scatter_file = RESULTS_DIR / (
        f"{dataset_slug}_epistemic_scatter_{METRIC_NAME}_sigma{dist_sigma_label}.pdf"
    )

    if output_file.exists():
        with open(output_file, "rb") as f:
            cached = pickle.load(f)
        values_matrix = cached.get("values_matrix")
        contour_matrix = cached.get("contour_matrix")
        dist_labels = cached.get("dist_labels")
        dist_epistemic_values = cached.get("dist_epistemic_values")
        dist_stability_values = cached.get("dist_stability_values")

        if values_matrix is None:
            print(f"Cached file missing values_matrix: {output_file}")
            return

        _plot_heatmap(
            values_matrix,
            SIGMAS,
            TARGET_COVERAGES,
            HEATMAP_VALUE,
            METRIC_NAME,
            dataset_name,
            plot_file,
            contour_matrix if CONTOUR_VALUE is not None else None,
        )
        if (
            EPIS_SCATTER_PLOT
            and dist_epistemic_values is not None
            and dist_labels is not None
            and dist_stability_values is not None
        ):
            _plot_epistemic_stability_scatter(
                dist_epistemic_values,
                dist_stability_values,
                dist_labels,
                EPIS_COVERAGE_LINES,
                dataset_name,
                METRIC_NAME,
                dist_sigma_label,
                scatter_file,
            )
            if dist_epistemic_values is not None and dist_labels is not None:
                _print_scatter_coverage_metrics(
                    dataset_name,
                    dist_epistemic_values,
                    dist_labels,
                    EPIS_COVERAGE_STATS,
                    dist_sigma_label,
                )
            if (
                dist_epistemic_values is not None
                and dist_stability_values is not None
                and dist_labels is not None
                and _include_in_combined(dataset_name, dataset_slug)
            ):
                COMBINED_SCATTER_ITEMS.append(
                    {
                        "dataset_name": dataset_name,
                        "metric_name": METRIC_NAME,
                        "sigma_label": dist_sigma_label,
                        "epistemic_values": dist_epistemic_values,
                        "stability_values": dist_stability_values,
                        "labels": dist_labels,
                    }
                )
            print(f"Loaded cache: {output_file}")
            print(f"Saved: {plot_file}")
            return

    # LOAD MODEL
    rf_uq = RandomForestClassifierUQ()
    rf_key = registry.make_key(dataset_key, rf_uq.name)
    if registry.exists(rf_key):
        rf_uq = registry.load(rf_key)
    else:
        print(f"Model not found, skipping: {rf_key}")
        return

    # EPISTEMIC CLEAN
    _, _, epistemic_clean = rf_uq.predict_with_uncertainty(X_test)

    # SHAP CLEAN
    shap_explainer = SHAPExplainer(rf_uq.base_model)
    attr_clean = shap_explainer.explain(X_test)
    n_features = attr_clean.shape[1]
    top_k_used = min(TOP_K, n_features)

    abs_attr_clean = np.abs(attr_clean)
    rank_clean = np.argsort(abs_attr_clean, axis=1)[:, ::-1]
    pos_clean = np.empty_like(rank_clean, dtype=float)
    pos_clean[np.arange(N_SAMPLES)[:, None], rank_clean] = np.arange(n_features)

    values_matrix = np.full((len(TARGET_COVERAGES), len(SIGMAS)), np.nan, dtype=float)
    contour_matrix = np.full((len(TARGET_COVERAGES), len(SIGMAS)), np.nan, dtype=float)
    roc_auc_by_sigma = np.full(len(SIGMAS), np.nan, dtype=float)
    coverage_matrix = np.full((len(TARGET_COVERAGES), len(SIGMAS)), np.nan, dtype=float)

    results_by_sigma = {}
    dist_labels = None
    dist_stability_values = None
    dist_epistemic_values = None
    dist_sigma_used = "all" if SCATTER_ALL_SIGMAS else dist_sigma
    if SCATTER_ALL_SIGMAS:
        dist_labels_list = []
        dist_stability_list = []
        dist_epistemic_list = []

    for s_idx, sigma in enumerate(SIGMAS):
        stability_values = np.zeros(N_SAMPLES)
        epistemic_values = np.zeros(N_SAMPLES)
        perturb_generators = [
            PerturbationGenerator(seed=GLOBAL_SEED + i) for i in range(K_NOISE_SEEDS)
        ]

        for perturb in tqdm(perturb_generators, desc=f"{dataset_name} sigma={sigma:.2f}", leave=False):
            X_pert = perturb.gaussian(X_test, float(sigma))
            _, _, epistemic_noisy = rf_uq.predict_with_uncertainty(X_pert)
            attr_pert = shap_explainer.explain(X_pert)
            stability_values += _metric_from_attr(
                METRIC_NAME, rank_clean, pos_clean, attr_pert, top_k_used
            ) / K_NOISE_SEEDS
            epistemic_values += epistemic_noisy / K_NOISE_SEEDS

        thresholds = STABILITY_METRICS[METRIC_NAME]
        stable_thresh = thresholds["stable_threshold"]
        unstable_thresh = thresholds["unstable_threshold"]

        labels = np.where(
            stability_values >= stable_thresh, "stable",
            np.where(stability_values <= unstable_thresh, "unstable", "ambiguous")
        )

        mask = labels != "ambiguous"
        results_by_sigma[sigma] = {
            "n_stable": int((labels == "stable").sum()),
            "n_unstable": int((labels == "unstable").sum()),
            "n_ambiguous": int((labels == "ambiguous").sum()),
            "coverage_results": [],
        }

        if SCATTER_ALL_SIGMAS:
            dist_labels_list.append(labels)
            dist_stability_list.append(stability_values)
            dist_epistemic_list.append(epistemic_values)
        elif np.isclose(float(sigma), float(dist_sigma)):
            dist_labels = labels
            dist_stability_values = stability_values
            dist_epistemic_values = epistemic_values

        if mask.sum() < 10:
            continue

        epistemic_filtered = epistemic_values[mask]
        stability_filtered = stability_values[mask]
        labels_binary = (labels[mask] == "stable").astype(int)

        if labels_binary.sum() == 0 or labels_binary.sum() == len(labels_binary):
            continue

        fpr, tpr, roc_thresholds = roc_curve(labels_binary, -epistemic_filtered)
        roc_auc = auc(fpr, tpr)
        roc_auc_by_sigma[s_idx] = roc_auc

        epi_mean = epistemic_filtered.mean()
        epi_std = epistemic_filtered.std()

        for c_idx, target_cov in enumerate(TARGET_COVERAGES):
            threshold = np.quantile(epistemic_filtered, target_cov)
            predictions = (epistemic_filtered <= threshold).astype(int)
            coverage = predictions.mean()

            if predictions.sum() == 0 or predictions.sum() == len(predictions):
                results_by_sigma[sigma]["coverage_results"].append({
                    "target_coverage": float(target_cov),
                    "coverage": float(coverage),
                    "threshold": float(threshold),
                    "k_value": 0.0,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "mean_stability_accepted": np.nan,
                    "mean_stability_rejected": np.nan,
                })
                continue

            precision = precision_score(labels_binary, predictions, zero_division=0)
            recall = recall_score(labels_binary, predictions, zero_division=0)
            f1 = f1_score(labels_binary, predictions, zero_division=0)

            accepted_mask = predictions == 1
            rejected_mask = predictions == 0
            mean_stab_acc = stability_filtered[accepted_mask].mean() if accepted_mask.sum() > 0 else 0
            mean_stab_rej = stability_filtered[rejected_mask].mean() if rejected_mask.sum() > 0 else 0

            k_value = (threshold - epi_mean) / epi_std if epi_std > 0 else 0.0

            results_by_sigma[sigma]["coverage_results"].append({
                "target_coverage": float(target_cov),
                "coverage": float(coverage),
                "threshold": float(threshold),
                "k_value": float(k_value),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_stability_accepted": float(mean_stab_acc),
                "mean_stability_rejected": float(mean_stab_rej),
            })

            coverage_matrix[c_idx, s_idx] = coverage
            contour_matrix[c_idx, s_idx] = results_by_sigma[sigma]["coverage_results"][-1].get(
                CONTOUR_VALUE, np.nan
            )

            if HEATMAP_VALUE == "roc_auc":
                values_matrix[c_idx, s_idx] = roc_auc
            else:
                values_matrix[c_idx, s_idx] = results_by_sigma[sigma]["coverage_results"][-1][HEATMAP_VALUE]

    if SCATTER_ALL_SIGMAS and dist_labels_list:
        dist_labels = np.concatenate(dist_labels_list)
        dist_stability_values = np.concatenate(dist_stability_list)
        dist_epistemic_values = np.concatenate(dist_epistemic_list)

    output = {
        "dataset": dataset_name,
        "config": config,
        "epistemic_clean": epistemic_clean,
        "scatter_sigma": dist_sigma_used,
        "dist_labels": dist_labels,
        "dist_epistemic_values": dist_epistemic_values,
        "dist_stability_values": dist_stability_values,
        "roc_auc_by_sigma": roc_auc_by_sigma,
        "coverage_matrix": coverage_matrix,
        "values_matrix": values_matrix,
        "contour_value": CONTOUR_VALUE,
        "contour_matrix": contour_matrix,
        "results_by_sigma": results_by_sigma,
    }

    with open(output_file, "wb") as f:
        pickle.dump(output, f)

    _plot_heatmap(
        values_matrix,
        SIGMAS,
        TARGET_COVERAGES,
        HEATMAP_VALUE,
        METRIC_NAME,
        dataset_name,
        plot_file,
        contour_matrix if CONTOUR_VALUE is not None else None,
    )
    if (
        EPIS_SCATTER_PLOT
        and dist_epistemic_values is not None
        and dist_labels is not None
        and dist_stability_values is not None
    ):
        _plot_epistemic_stability_scatter(
            dist_epistemic_values,
            dist_stability_values,
            dist_labels,
            EPIS_COVERAGE_LINES,
            dataset_name,
            METRIC_NAME,
            dist_sigma_label,
            scatter_file,
        )
    if dist_epistemic_values is not None and dist_labels is not None:
        _print_scatter_coverage_metrics(
            dataset_name,
            dist_epistemic_values,
            dist_labels,
            EPIS_COVERAGE_STATS,
            dist_sigma_label,
        )
    if (
        dist_epistemic_values is not None
        and dist_stability_values is not None
        and dist_labels is not None
        and _include_in_combined(dataset_name, dataset_slug)
    ):
        COMBINED_SCATTER_ITEMS.append(
            {
                "dataset_name": dataset_name,
                "metric_name": METRIC_NAME,
                "sigma_label": dist_sigma_label,
                "epistemic_values": dist_epistemic_values,
                "stability_values": dist_stability_values,
                "labels": dist_labels,
            }
        )
    print(f"Saved: {plot_file}")

for ds in datasets:
    run_dataset(ds)

if COMBINED_SCATTER_ITEMS:
    sigma_labels = {item["sigma_label"] for item in COMBINED_SCATTER_ITEMS}
    combined_sigma = sigma_labels.pop() if len(sigma_labels) == 1 else "mixed"
    combined_scatter_file = RESULTS_DIR / (
        f"combined_epistemic_scatter_{METRIC_NAME}_sigma{combined_sigma}.pdf"
    )
    _plot_combined_scatter(COMBINED_SCATTER_ITEMS, combined_scatter_file)
    print(f"Saved: {combined_scatter_file}")
