from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT_DIR / "results"
BASE_CSV = RESULTS_ROOT / "1_train_models" / "1_train_models_results.csv"
UQ_CSV = RESULTS_ROOT / "2_train_models_uq" / "2_train_models_uq_results.csv"
OUTPUT_DIR = RESULTS_ROOT / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "f1_comparison_base_vs_uq.png"

MODEL_SHORT = [
    (r"logistic", "LR"),
    (r"_rf", "RF"),
    (r"mlp", "MLP"),
    (r"lgbm", "LGBM"),
    (r"catboost", "CB"),
]


def short_model_name(name: str) -> str:
    lname = str(name).lower()
    for pattern, short in MODEL_SHORT:
        if pattern in lname:
            return short
    return name


def load_base() -> pd.DataFrame:
    df = pd.read_csv(BASE_CSV)
    df["base_model"] = df["model"]
    df = df[["dataset", "base_model", "f1"]].rename(columns={"f1": "f1_base"})
    return df


def load_uq() -> pd.DataFrame:
    df = pd.read_csv(UQ_CSV)
    # STRIP UQ SUFFIX LIKE UQ B20 TO ALIGN WITH BASE MODEL NAME
    df["base_model"] = df["model"].apply(lambda m: re.sub(r"_uq.*$", "", str(m)))
    df = df[["dataset", "base_model", "f1"]].rename(columns={"f1": "f1_uq"})
    return df


def build_plot(df: pd.DataFrame) -> None:
    df["model_short"] = df["base_model"].apply(short_model_name)
    order = ["Logistic", "RF", "MLP", "LGBM", "CatBoost"]
    df["model_order"] = df["model_short"].apply(lambda x: order.index(x) if x in order else len(order))
    # DO NOT RESORT DATASETS GLOBALLY KEEP MODEL ORDERING PER DATASET ONLY
    desired_order = ["iris", "rice", "bean", "ecoli", "wine_binary"]
    # NORMALIZE DATASET NAMES TO COMPARE AGAINST DESIRED ORDER
    def norm_name(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")

    # KEEP FIRST OCCURRENCE MAPPING FOR EACH NORMALIZED NAME
    norm_map = {}
    for d in df["dataset"]:
        nd = norm_name(d)
        if nd not in norm_map:
            norm_map[nd] = d
    ordered = [norm_map[n] for n in desired_order if n in norm_map]
    extras = [v for k, v in norm_map.items() if k not in desired_order]
    datasets = ordered + extras

    n = len(datasets)
    ncols = n if n > 0 else 1
    nrows = 1
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(2 * ncols, 4), squeeze=False, sharey=True)

    width = 0.35
    for idx, dataset in enumerate(datasets):
        ax = axes[0][idx]
        sub = df[df["dataset"] == dataset].sort_values("model_order")
        x = range(len(sub))
        ax.bar([i - width / 2 for i in x], sub["f1_base"], width=width, label="Base F1", color="#4C7AD1")
        ax.bar([i + width / 2 for i in x], sub["f1_uq"], width=width, label="UQ F1", color="#D17A4C")

        ax.set_title(dataset)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["model_short"], rotation=45, ha="center", va="top")
        ax.set_ylim(0.7, 1.01)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if idx == len(datasets) - 1:
            ax.legend(loc="upper right")

    # HIDE ANY UNUSED AXES
    for j in range(idx + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.suptitle("Base vs UQ F1 scores", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"OK Plot saved to {OUTPUT_PNG}")


def main():
    if not BASE_CSV.exists() or not UQ_CSV.exists():
        raise FileNotFoundError("Expected CSV files not found. Run the training scripts to generate them.")

    base_df = load_base()
    uq_df = load_uq()

    merged = pd.merge(base_df, uq_df, on=["dataset", "base_model"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping dataset/model pairs between base and UQ results.")

    build_plot(merged)


if __name__ == "__main__":
    main()
