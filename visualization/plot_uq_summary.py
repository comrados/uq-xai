from pathlib import Path
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT_DIR / "results"
INPUT_DIR = RESULTS_ROOT / "2_train_models_uq"
OUTPUT_DIR = RESULTS_ROOT / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_NAME = "2_train_models_uq_results.csv"
HEATMAP_PATH = OUTPUT_DIR / "fig_uq_heatmap.png"
SMALL_MULTIPLES_PATH = OUTPUT_DIR / "fig_uq_small_multiples.png"

MODEL_ORDER = ["log", "rf", "mlp", "lgbm", "cb"]
DATASET_ORDER = ["iris", "rice", "bean", "ecoli", "wine_binary"]
METRIC_CANDIDATES = {
    "ECE": ["ECE", "ece"],
    "Aleatoric": ["Aleatoric", "aleatoric", "aleatoric_mean"],
    "Epistemic": ["Epistemic", "epistemic", "epistemic_mean"],
    #"Confidence": ["Confidence", "confidence", "confidence_mean"],
}

MODEL_COLORS = {
    "log": "tab:blue",
    "rf": "tab:orange",
    "mlp": "tab:green",
    "lgbm": "tab:red",
    "cb": "tab:purple",
}

# PER MODEL METRIC SELECTION FOR BAR PLOTS
DEFAULT_BAR_METRICS = ["ECE", "Aleatoric", "Epistemic", "Confidence"]
REDUCED_BAR_METRICS = ["ECE", "Aleatoric", "Confidence"]
BAR_MODEL_METRICS = {
    "log": DEFAULT_BAR_METRICS,
    "rf": DEFAULT_BAR_METRICS,
    "mlp": DEFAULT_BAR_METRICS,
    "lgbm": REDUCED_BAR_METRICS,
    "cb": REDUCED_BAR_METRICS,
}


def find_csv() -> Path:
    candidates = [
        INPUT_DIR / CSV_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {CSV_NAME} in {candidates}")


def map_model(name: str) -> str:
    n = name.lower()
    if "logistic" in n:
        return "log"
    if "rf" in n:
        return "rf"
    if "mlp" in n:
        return "mlp"
    if "lgbm" in n:
        return "lgbm"
    if "catboost" in n:
        return "cb"
    return name


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # COLUMN VARIANTS
    col_map = {}
    for canonical, alt in [("dataset", "Dataset"), ("model", "Model")]:
        if canonical in df.columns:
            col_map[canonical] = canonical
        elif alt in df.columns:
            col_map[canonical] = alt
        else:
            raise KeyError(f"Missing required column: {canonical}/{alt}")
    metric_cols: Dict[str, str] = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            metric_cols[canonical] = found
        else:
            warnings.warn(f"Metric column missing for {canonical}: tried {candidates} (will skip)")
    df = df.rename(columns=col_map)
    df["model_short"] = df["model"].apply(map_model)
    df["dataset"] = df["dataset"].astype(str)
    keep_cols = ["dataset", "model_short"] + list(metric_cols.values())
    df = df[keep_cols]
    # STANDARDIZE METRIC NAMES
    df = df.rename(columns={v: k for k, v in metric_cols.items()})
    return df


def ensure_metrics(df: pd.DataFrame) -> List[str]:
    available = [m for m in METRIC_CANDIDATES.keys() if m in df.columns]
    missing = [m for m in METRIC_CANDIDATES.keys() if m not in df.columns]
    if missing:
        warnings.warn(f"Missing metrics (will skip in plots): {missing}")
    return available

def plot_small_multiples(df: pd.DataFrame, metrics: List[str]) -> None:
    present = set(df["dataset"].unique())
    datasets = [d for d in DATASET_ORDER if d in present] + sorted(present - set(DATASET_ORDER))
    n_metrics = len(metrics)
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_metrics, n_datasets, figsize=(2 * n_datasets, 1.8 * n_metrics), sharey="row", squeeze=False)

    for mi, metric in enumerate(metrics):
        metric_max = df[metric].max()
        metric_min = df[metric].min()
        ymin = 0 if metric_min >= 0 else metric_min * 1.05
        ymax = metric_max * 1.1 if metric_max > 0 else 1.0
        for di, ds in enumerate(datasets):
            ax = axes[mi][di]
            allowed_models = [m for m in MODEL_ORDER if metric in BAR_MODEL_METRICS.get(m, DEFAULT_BAR_METRICS)]
            sub = df[(df["dataset"] == ds) & (df["model_short"].isin(allowed_models))]
            sub = sub.set_index("model_short").reindex(allowed_models)
            if sub.empty or sub[metric].isna().all():
                ax.axis("off")
                continue
            colors = [MODEL_COLORS.get(m, "tab:blue") for m in allowed_models]
            pretty_labels = [m.upper() if m != "log" else "LR" for m in allowed_models]
            ax.bar(range(len(allowed_models)), sub[metric].values, color=colors)
            ax.set_xticks(range(len(allowed_models)))
            ax.set_xticklabels(pretty_labels, rotation=45, ha="center", va="top", fontsize=8)
            ax.set_ylim(ymin, ymax)
            if mi == 0:
                ax.set_title(ds)
            if di == 0:
                ax.set_ylabel(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("UQ metrics by dataset/model", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SMALL_MULTIPLES_PATH, dpi=300)
    print(f"OK Small-multiples saved to {SMALL_MULTIPLES_PATH}")


def main():
    csv_path = find_csv()
    df = load_data(csv_path)
    metrics = ensure_metrics(df)
    if not metrics:
        raise ValueError("No metrics available to plot.")
    plot_small_multiples(df, metrics)


if __name__ == "__main__":
    main()
