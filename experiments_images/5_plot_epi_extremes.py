"""Visualize epistemic extremes.

Selects low/high epistemic samples, computes IG/SmoothGrad attributions across
noise levels, and saves grid visualizations with SSIM.
"""

import sys
import os
sys.path.append(os.getcwd())

import csv
import pickle
from itertools import islice
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler
from torchvision import datasets

from utils import (
    EPIS_PATH,
    RESULTS_DIR,
    TEST_DIR,
    add_gaussian_noise,
    compute_ig,
    compute_smoothgrad,
    compute_ssim,
    get_device,
    get_transform,
    load_model,
)

# SETTINGS
SIGMA: float = 0.0
SIGMA_LEVELS: list[float] = [0.0, 0.1, 0.2]
TOP_K: int = 5
MANUAL_INDICES: list[int] = []
EXTRA_INDICES: list[int] = []
SEED: int = 42
NOISE_SEED: int | None = None
IGNORE_LABEL_NAMES: list[str] = []  # ["Tomato_healthy"]
IGNORE_LABEL_SUBSTRINGS: list[str] = []  # ["healthy"]
LOW_RANKS: list[int] = []
HIGH_RANKS: list[int] = []
# PER CLASS RANKS WITHIN CLASS EPI ORDER KEYS MATCH FULL NAME OR SUBSTRING
LOW_CLASS_RANKS: dict[str, list[int]] = {
    "Tomato_healthy": [0],
    "Tomato_Bacterial_spot": [], #[14],
    "Tomato_Late_blight": [3],
}
HIGH_CLASS_RANKS: dict[str, list[int]] = {
    "Tomato_healthy": [2],
    "Tomato_Bacterial_spot": [], #[0],
    "Tomato_Late_blight": [0],
}
ROW_CLASS_ORDER: list[str] = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
]
ROW_GROUP_ORDER: list[str] = ["low", "high"]

SMOOTHGRAD_SAMPLES: int = 20
SMOOTHGRAD_NOISE: float = 0.1

OUTPUT_NAME: str = "epi_extremes_ig_sg.pdf"
SAVE_SELECTIONS_CSV: bool = True
HEATMAP_CMAP: str = "cividis"
HEATMAP_VMIN: float = -0.05
HEATMAP_VMAX: float = 0.05
AUTO_HEATMAP_SCALE: bool = True
HEATMAP_ABS_PERCENTILE: float = 99.0
HEATMAP_INTERPOLATION: str = "bilinear"  # e.g. "nearest", "bilinear", "bicubic", "lanczos"
HEATMAP_BLUR_KERNEL: int = 5  # 0 disables; must be odd (e.g. 5)
HEATMAP_BLUR_SIGMA: float = 0.0  # 0 uses an automatic sigma based on kernel size
HEATMAP_BLEND_ON_IMAGE: bool = True
HEATMAP_ALPHA: float = 0.95  # 0 = only image, 1 = only heatmap
SUBPLOT_WSPACE: float = 0.04
SINGLE_SUBPLOT_WSPACE: float = 0.04
SUBPLOT_TOP: float = 0.92
SINGLE_SUBPLOT_TOP: float = 0.88
SUBPLOT_LEFT: float = 0.08
SINGLE_TILE_WIDTH: float = 2.2
SINGLE_GROUP_SPACER_RATIO: float = 0.10
CLASS_LABEL_DISPLAY: dict[str, str] = {
    "Tomato_healthy": "Healthy",
    "Tomato_Healthy": "Healthy",
    "Tomato_Late_blight": "Blight",
}
GROUP_LABEL_DISPLAY: dict[str, str] = {
    "low": "Low-epistemic",
    "high": "High-epistemic",
}
OUTPUT_DPI: int = 150
PDF_DPI: int = 50
PDF_COMPRESSION: int = 9
GROUP_SPACER_RATIO: float = 0.1
SEPARATOR_COLOR: str = "#d0d0d0"
SEPARATOR_LW: float = 2
FONT_SIZE: int = 16
FONT_SIZE_SMALL: int = 13


def gaussian_blur_2d(array: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 1:
        return array
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"HEATMAP_BLUR_KERNEL must be odd; using {kernel_size} instead.")
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d[None, None, :, :]

    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None, :, :]
    padding = kernel_size // 2
    if padding > 0:
        tensor = F.pad(tensor, (padding, padding, padding, padding), mode="reflect")
    blurred = F.conv2d(tensor, kernel_2d, padding=0)
    return blurred.squeeze(0).squeeze(0).numpy()


def prepare_heatmap_for_display(array: np.ndarray) -> np.ndarray:
    if HEATMAP_BLUR_KERNEL > 1:
        return gaussian_blur_2d(array, HEATMAP_BLUR_KERNEL, HEATMAP_BLUR_SIGMA)
    return array


def normalize_indices(indices: list[int], n_samples: int) -> list[int]:
    normalized: list[int] = []
    for raw in indices:
        idx = raw
        if idx < 0:
            idx = n_samples + idx
        if idx < 0 or idx >= n_samples:
            print(f"Skipping out-of-range index: {raw}")
            continue
        if idx not in normalized:
            normalized.append(idx)
    return normalized


def resolve_sample_paths(n_samples: int) -> tuple[list[Path], list[int], list[str] | None]:
    if not TEST_DIR.exists():
        return [], [], None

    dataset = datasets.ImageFolder(TEST_DIR, transform=get_transform())
    gen = torch.Generator()
    gen.manual_seed(SEED)
    sampler = RandomSampler(dataset, generator=gen)
    sample_indices = list(islice(sampler, n_samples))

    sample_paths = [Path(dataset.samples[i][0]) for i in sample_indices]
    sample_labels = [dataset.samples[i][1] for i in sample_indices]
    return sample_paths, sample_labels, list(dataset.classes)


def select_indices(
    epi_values: np.ndarray,
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
) -> list[tuple[str, int]]:
    n_samples = len(epi_values)
    available_indices = [
        idx for idx in range(n_samples)
        if not is_ignored(idx, labels[idx], label_names, class_names)
    ]
    if not available_indices:
        print("No samples available after IGNORE_LABEL_NAMES filter.")
        return []

    if MANUAL_INDICES:
        manual = normalize_indices(MANUAL_INDICES, n_samples)
        filtered: list[int] = []
        for idx in manual:
            if is_ignored(idx, labels[idx], label_names, class_names):
                label_name = build_label_name(idx, labels[idx], label_names, class_names)
                print(f"Skipping ignored manual index {idx} ({label_name})")
                continue
            filtered.append(idx)
        if not filtered:
            print("All manual indices are ignored; nothing to plot.")
            return []
        return [("manual", idx) for idx in filtered]

    epi_subset = epi_values[available_indices]
    order = np.argsort(epi_subset)
    sorted_indices = [available_indices[i] for i in order]

    selected: list[tuple[str, int]] = []
    has_rank_overrides = bool(
        LOW_RANKS or HIGH_RANKS or LOW_CLASS_RANKS or HIGH_CLASS_RANKS
    )
    indices_by_class: dict[str, list[int]] = {}
    if LOW_CLASS_RANKS or HIGH_CLASS_RANKS:
        indices_by_class = build_indices_by_class(
            available_indices, labels, label_names, class_names
        )
    if LOW_CLASS_RANKS:
        selected += select_from_class_ranks(
            "low", LOW_CLASS_RANKS, indices_by_class, epi_values
        )
    if HIGH_CLASS_RANKS:
        selected += select_from_class_ranks(
            "high", HIGH_CLASS_RANKS, indices_by_class, epi_values
        )
    if LOW_RANKS:
        selected += select_from_ranks("low", sorted_indices, LOW_RANKS)
    if HIGH_RANKS:
        selected += select_from_ranks("high", list(reversed(sorted_indices)), HIGH_RANKS)

    if not has_rank_overrides:
        top_k = min(TOP_K, len(available_indices))
        if 2 * top_k > len(available_indices):
            print("Warning: TOP_K too large; low/high sets may overlap.")
        low = [available_indices[i] for i in order[:top_k]]
        high = [available_indices[i] for i in order[-top_k:][::-1]]
        selected = [("low", idx) for idx in low] + [("high", idx) for idx in high]

    for idx in normalize_indices(EXTRA_INDICES, n_samples):
        if is_ignored(idx, labels[idx], label_names, class_names):
            label_name = build_label_name(idx, labels[idx], label_names, class_names)
            print(f"Skipping ignored extra index {idx} ({label_name})")
            continue
        selected.append(("extra", idx))

    seen: set[int] = set()
    deduped: list[tuple[str, int]] = []
    for group, idx in selected:
        if idx in seen:
            continue
        deduped.append((group, idx))
        seen.add(idx)
    return deduped


def build_label_name(
    idx: int,
    label: int,
    label_names: list[str] | None,
    class_names: list[str] | None,
) -> str:
    if label_names and idx < len(label_names):
        return label_names[idx]
    if class_names and 0 <= label < len(class_names):
        return class_names[label]
    return f"class_{label}"


def is_ignored(
    idx: int,
    label: int,
    label_names: list[str] | None,
    class_names: list[str] | None,
) -> bool:
    if not IGNORE_LABEL_NAMES and not IGNORE_LABEL_SUBSTRINGS:
        return False
    name = build_label_name(idx, label, label_names, class_names)
    if name in IGNORE_LABEL_NAMES:
        return True
    lowered = name.lower()
    return any(token.lower() in lowered for token in IGNORE_LABEL_SUBSTRINGS)


def pick_by_rank(order: list[int], rank: int, label: str) -> int | None:
    n_samples = len(order)
    if n_samples == 0:
        return None
    idx = rank
    if idx < 0:
        idx = n_samples + idx
    if idx < 0 or idx >= n_samples:
        print(f"Skipping out-of-range {label} rank: {rank}")
        return None
    return order[idx]


def class_key_matches(key: str, name: str) -> bool:
    if key == name:
        return True
    return key.lower() in name.lower()


def build_indices_by_class(
    indices: list[int],
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx in indices:
        label_name = build_label_name(idx, labels[idx], label_names, class_names)
        grouped.setdefault(label_name, []).append(idx)
    return grouped


def select_from_ranks(kind: str, order: list[int], ranks: list[int]) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    for rank in ranks:
        idx = pick_by_rank(order, rank, kind)
        if idx is not None:
            selected.append((kind, idx))
    return selected


def select_from_class_ranks(
    kind: str,
    class_ranks: dict[str, list[int]],
    indices_by_class: dict[str, list[int]],
    epi_values: np.ndarray,
) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    for key, ranks in class_ranks.items():
        if isinstance(ranks, int):
            ranks = [ranks]
        matches = [name for name in indices_by_class if class_key_matches(key, name)]
        if not matches:
            print(f"Warning: class key not found: {key}")
            continue
        for name in matches:
            class_indices = indices_by_class[name]
            class_sorted = sorted(class_indices, key=lambda i: epi_values[i])
            if kind == "high":
                class_sorted = list(reversed(class_sorted))
            for rank in ranks:
                idx = pick_by_rank(class_sorted, rank, f"{kind}/{name}")
                if idx is not None:
                    selected.append((kind, idx))
    return selected


def sort_selected_rows(
    selected: list[tuple[str, int]],
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
) -> list[tuple[str, int]]:
    def class_order(label_name: str) -> int:
        for idx, key in enumerate(ROW_CLASS_ORDER):
            if class_key_matches(key, label_name):
                return idx
        return len(ROW_CLASS_ORDER)

    def group_order(group: str) -> int:
        if group in ROW_GROUP_ORDER:
            return ROW_GROUP_ORDER.index(group)
        return len(ROW_GROUP_ORDER)

    return sorted(
        selected,
        key=lambda item: (
            class_order(build_label_name(item[1], labels[item[1]], label_names, class_names)),
            group_order(item[0]),
            item[1],
        ),
    )


def print_selection(
    selected: list[tuple[str, int]],
    epi_values: np.ndarray,
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
    sample_paths: list[Path],
) -> None:
    print("\nSelected samples:")
    for group, idx in selected:
        label = labels[idx]
        label_name = build_label_name(idx, label, label_names, class_names)
        epi = float(epi_values[idx])
        name = sample_paths[idx].name if idx < len(sample_paths) else "unknown"
        print(f"  {group:<5} idx={idx:>3} epi={epi:.6f} label={label_name} file={name}")


def save_selection_csv(
    selected: list[tuple[str, int]],
    epi_values: np.ndarray,
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
    sample_paths: list[Path],
) -> None:
    if not SAVE_SELECTIONS_CSV:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "epi_extremes_selection.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "index", "epi", "label", "label_name", "filename", "path"])
        for group, idx in selected:
            label = labels[idx]
            label_name = build_label_name(idx, label, label_names, class_names)
            epi = float(epi_values[idx])
            if idx < len(sample_paths):
                path = sample_paths[idx]
                writer.writerow([group, idx, epi, label, label_name, path.name, str(path)])
            else:
                writer.writerow([group, idx, epi, label, label_name, "unknown", ""])
    print(f"Saved selections to {out_path}")


def plot_extremes(
    selected: list[tuple[str, int]],
    epi_values: np.ndarray,
    images: list[torch.Tensor],
    labels: list[int],
    label_names: list[str] | None,
    class_names: list[str] | None,
    sample_paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    n_rows = len(selected)
    if 0.0 in SIGMA_LEVELS:
        sigma_order = [0.0] + [s for s in SIGMA_LEVELS if s != 0.0]
    else:
        sigma_order = list(SIGMA_LEVELS)
        print("Warning: SIGMA_LEVELS has no 0.0; using provided order for columns.")
    clean_sigma_idx = sigma_order.index(0.0) if 0.0 in sigma_order else None
    n_sigma = len(sigma_order)
    image_col = 0
    spacer1_col = 1
    ig_start = spacer1_col + 1
    ig_cols = list(range(ig_start, ig_start + n_sigma))
    spacer2_col = ig_start + n_sigma
    sg_start = spacer2_col + 1
    sg_cols = list(range(sg_start, sg_start + n_sigma))
    n_cols = sg_start + n_sigma

    width_ratios = (
        [1.0]
        + [GROUP_SPACER_RATIO]
        + [1.0] * n_sigma
        + [GROUP_SPACER_RATIO]
        + [1.0] * n_sigma
    )
    content_cols = 1 + 2 * n_sigma
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * content_cols, 2.7 * n_rows),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row in range(n_rows):
        axes[row, spacer1_col].axis("off")
        axes[row, spacer2_col].axis("off")

    if NOISE_SEED is not None:
        torch.manual_seed(NOISE_SEED)

    attr_rows: list[list[tuple[np.ndarray, np.ndarray]]] = []
    ig_attrs: list[np.ndarray] = []
    sg_attrs: list[np.ndarray] = []
    for row, (_, idx) in enumerate(selected):
        image = images[idx]
        label = labels[idx]
        row_attrs: list[tuple[np.ndarray, np.ndarray]] = []
        for sigma in sigma_order:
            if sigma == 0.0:
                image_noisy = image
            else:
                image_noisy = add_gaussian_noise(image, sigma=sigma)

            ig_attr = compute_ig(model, image_noisy, target_class=label, device=device)
            sg_attr = compute_smoothgrad(
                model,
                image_noisy,
                target_class=label,
                device=device,
                n_samples=SMOOTHGRAD_SAMPLES,
                noise_sigma=SMOOTHGRAD_NOISE,
            )
            row_attrs.append((ig_attr, sg_attr))
            ig_attrs.append(ig_attr)
            sg_attrs.append(sg_attr)
        attr_rows.append(row_attrs)

    ig_vmin = HEATMAP_VMIN
    ig_vmax = HEATMAP_VMAX
    sg_vmin = HEATMAP_VMIN
    sg_vmax = HEATMAP_VMAX
    if AUTO_HEATMAP_SCALE and (ig_attrs or sg_attrs):
        if ig_attrs:
            ig_flat = np.concatenate([arr.ravel() for arr in ig_attrs])
            ig_scale = np.percentile(np.abs(ig_flat), HEATMAP_ABS_PERCENTILE)
            if ig_scale <= 0:
                ig_scale = np.max(np.abs(ig_flat))
            if ig_scale <= 0:
                ig_scale = 1.0
            ig_vmin = -ig_scale
            ig_vmax = ig_scale
        if sg_attrs:
            sg_flat = np.concatenate([arr.ravel() for arr in sg_attrs])
            sg_scale = np.percentile(np.abs(sg_flat), HEATMAP_ABS_PERCENTILE)
            if sg_scale <= 0:
                sg_scale = np.max(np.abs(sg_flat))
            if sg_scale <= 0:
                sg_scale = 1.0
            sg_vmin = -sg_scale
            sg_vmax = sg_scale
        print(
            f"Auto heatmap scale: IG +/-{abs(ig_vmax):.6f}, "
            f"SG +/-{abs(sg_vmax):.6f} (p{HEATMAP_ABS_PERCENTILE})"
        )

    def format_sigma_label(sigma: float) -> str:
        return "clean" if sigma == 0.0 else f"σ={sigma}"

    def build_output_path(suffix: str) -> Path:
        output_path = RESULTS_DIR / OUTPUT_NAME
        if not suffix:
            return output_path
        stem = output_path.stem
        if stem.endswith("_ig_sg"):
            stem = stem[: -len("_ig_sg")]
        return output_path.with_name(f"{stem}{suffix}{output_path.suffix}")

    def save_figure(fig_to_save: plt.Figure, out_path: Path) -> None:
        is_pdf = out_path.suffix.lower() == ".pdf"
        save_dpi = PDF_DPI if is_pdf else OUTPUT_DPI
        save_kwargs = {"dpi": save_dpi, "bbox_inches": "tight"}
        if is_pdf:
            with plt.rc_context({"pdf.compression": PDF_COMPRESSION}):
                fig_to_save.savefig(out_path, **save_kwargs)
        else:
            fig_to_save.savefig(out_path, **save_kwargs)
        print(f"Saved plot to {out_path}")

    def add_column_labels(fig: plt.Figure) -> None:
        y_top = axes[0, image_col].get_position().y1
        label_y = min(y_top + 0.03, 0.98)
        bbox = axes[0, image_col].get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            label_y,
            "Original",
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
        )

        for sigma_idx, sigma in enumerate(sigma_order):
            ig_col = ig_cols[sigma_idx]
            sg_col = sg_cols[sigma_idx]
            ig_label = f"IG {format_sigma_label(sigma)}"
            sg_label = f"SG {format_sigma_label(sigma)}"
            for col, label in ((ig_col, ig_label), (sg_col, sg_label)):
                bbox = axes[0, col].get_position()
                fig.text(
                    (bbox.x0 + bbox.x1) / 2,
                    label_y,
                    label,
                    ha="center",
                    va="top",
                    fontsize=FONT_SIZE,
                )

    def add_row_labels(fig: plt.Figure, labels_for_rows: list[tuple[str, str]]) -> None:
        for row_idx, (label, group) in enumerate(labels_for_rows):
            display_label = CLASS_LABEL_DISPLAY.get(label, label)
            display_label = f"{display_label} ({group})"
            bbox = axes[row_idx, image_col].get_position()
            fig.text(
                bbox.x0 - 0.01,
                (bbox.y0 + bbox.y1) / 2,
                display_label,
                ha="right",
                va="center",
                rotation=90,
                fontsize=FONT_SIZE,
            )

    def add_group_separators(fig: plt.Figure) -> None:
        top = axes[0, image_col].get_position().y1
        bottom = axes[-1, image_col].get_position().y0
        for spacer_col in (spacer1_col, spacer2_col):
            bbox = axes[0, spacer_col].get_position()
            x = (bbox.x0 + bbox.x1) / 2
            fig.add_artist(
                plt.Line2D(
                    [x, x],
                    [bottom, top],
                    transform=fig.transFigure,
                    color=SEPARATOR_COLOR,
                    linewidth=SEPARATOR_LW,
                    zorder=1000,
                    clip_on=False,
                )
            )

    def add_text_box(ax: plt.Axes, label: str, bold: bool = False) -> None:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_SMALL,
            fontweight="bold" if bold else "normal",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    def add_ssim_box(ax: plt.Axes, ssim_value: float | None) -> None:
        if ssim_value is None:
            return
        label = f"SSIM {ssim_value:.3f}"
        add_text_box(ax, label, bold=True)

    for row, (group, idx) in enumerate(selected):
        image = images[idx]
        label = labels[idx]
        epi = float(epi_values[idx])
        label_name = build_label_name(idx, label, label_names, class_names)

        img = image.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5)
        img = np.clip(img, 0, 1)

        axes[row, image_col].imshow(img)
        axes[row, image_col].axis("off")

        axes[row, image_col].set_title("", fontsize=FONT_SIZE)
        add_text_box(axes[row, image_col], f"epi={epi:.4f}")

        for sigma_idx, sigma in enumerate(sigma_order):
            ig_attr, sg_attr = attr_rows[row][sigma_idx]
            ig_col = ig_cols[sigma_idx]
            sg_col = sg_cols[sigma_idx]
            ig_attr_vis = prepare_heatmap_for_display(ig_attr)
            sg_attr_vis = prepare_heatmap_for_display(sg_attr)
            if HEATMAP_BLEND_ON_IMAGE:
                axes[row, ig_col].imshow(img)
            axes[row, ig_col].imshow(
                ig_attr_vis,
                cmap=HEATMAP_CMAP,
                vmin=ig_vmin,
                vmax=ig_vmax,
                interpolation=HEATMAP_INTERPOLATION,
                alpha=HEATMAP_ALPHA if HEATMAP_BLEND_ON_IMAGE else 1.0,
            )
            axes[row, ig_col].axis("off")
            if HEATMAP_BLEND_ON_IMAGE:
                axes[row, sg_col].imshow(img)
            axes[row, sg_col].imshow(
                sg_attr_vis,
                cmap=HEATMAP_CMAP,
                vmin=sg_vmin,
                vmax=sg_vmax,
                interpolation=HEATMAP_INTERPOLATION,
                alpha=HEATMAP_ALPHA if HEATMAP_BLEND_ON_IMAGE else 1.0,
            )
            axes[row, sg_col].axis("off")

            if clean_sigma_idx is None or sigma_idx == clean_sigma_idx:
                ig_ssim = None
                sg_ssim = None
            else:
                ig_clean, sg_clean = attr_rows[row][clean_sigma_idx]
                ig_ssim = compute_ssim(ig_clean, ig_attr)
                sg_ssim = compute_ssim(sg_clean, sg_attr)
            add_ssim_box(axes[row, ig_col], ig_ssim)
            add_ssim_box(axes[row, sg_col], sg_ssim)

    plt.tight_layout()
    fig.subplots_adjust(wspace=SUBPLOT_WSPACE, top=SUBPLOT_TOP, left=SUBPLOT_LEFT)
    add_column_labels(fig)
    add_row_labels(
        fig,
        [
            (build_label_name(idx, labels[idx], label_names, class_names), group)
            for group, idx in selected
        ],
    )
    add_group_separators(fig)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = build_output_path("")
    save_figure(fig, out_path)
    plt.close(fig)

    def render_single_method(
        method_name: str,
        attr_index: int,
        vmin: float,
        vmax: float,
        out_suffix: str,
    ) -> None:
        selected_label_names = [
            build_label_name(idx, labels[idx], label_names, class_names)
            for _, idx in selected
        ]
        ordered_classes: list[str] = []
        for class_key in ROW_CLASS_ORDER:
            for label_name in selected_label_names:
                if class_key_matches(class_key, label_name) and label_name not in ordered_classes:
                    ordered_classes.append(label_name)
        for label_name in selected_label_names:
            if label_name not in ordered_classes:
                ordered_classes.append(label_name)

        ordered_groups: list[str] = []
        for group_name in ROW_GROUP_ORDER:
            if any(group == group_name for group, _ in selected):
                ordered_groups.append(group_name)
        for group_name, _ in selected:
            if group_name not in ordered_groups:
                ordered_groups.append(group_name)

        cols_per_class = 1 + n_sigma
        single_n_rows = len(ordered_groups)
        class_block_starts: list[int] = []
        width_ratios: list[float] = []
        col_cursor = 0
        for class_idx in range(len(ordered_classes)):
            class_block_starts.append(col_cursor)
            width_ratios.extend([1.0] * cols_per_class)
            col_cursor += cols_per_class
            if class_idx < len(ordered_classes) - 1:
                width_ratios.append(SINGLE_GROUP_SPACER_RATIO)
                col_cursor += 1
        single_n_cols = col_cursor
        fig_single, axes_single = plt.subplots(
            single_n_rows,
            single_n_cols,
            figsize=(SINGLE_TILE_WIDTH * single_n_cols, 2.7 * single_n_rows),
            gridspec_kw={"width_ratios": width_ratios},
        )
        if single_n_rows == 1:
            axes_single = np.array([axes_single])
        if single_n_cols == 1:
            axes_single = axes_single[:, np.newaxis]

        for class_idx in range(len(ordered_classes) - 1):
            spacer_col = class_block_starts[class_idx] + cols_per_class
            for row_idx in range(single_n_rows):
                axes_single[row_idx, spacer_col].axis("off")

        selected_row_lookup: dict[tuple[str, str], int] = {}
        for row_idx, (group_name, idx) in enumerate(selected):
            label_name = build_label_name(idx, labels[idx], label_names, class_names)
            selected_row_lookup[(group_name, label_name)] = row_idx

        def add_single_column_labels(fig_single_local: plt.Figure) -> None:
            y_top = axes_single[0, 0].get_position().y1
            label_y = min(y_top + 0.05, 0.99)
            for class_idx, class_name in enumerate(ordered_classes):
                display_class = CLASS_LABEL_DISPLAY.get(class_name, class_name)
                block_start = class_block_starts[class_idx]
                original_bbox = axes_single[0, block_start].get_position()
                fig_single_local.text(
                    (original_bbox.x0 + original_bbox.x1) / 2,
                    label_y,
                    display_class,
                    ha="center",
                    va="top",
                    fontsize=FONT_SIZE,
                )
                for sigma_idx, sigma in enumerate(sigma_order):
                    bbox = axes_single[0, block_start + 1 + sigma_idx].get_position()
                    fig_single_local.text(
                        (bbox.x0 + bbox.x1) / 2,
                        label_y,
                        f"{method_name} {format_sigma_label(sigma)}",
                        ha="center",
                        va="top",
                        fontsize=FONT_SIZE,
                    )

        def add_single_row_labels(fig_single_local: plt.Figure) -> None:
            for row_idx, group_name in enumerate(ordered_groups):
                display_group = GROUP_LABEL_DISPLAY.get(group_name, group_name)
                bbox = axes_single[row_idx, 0].get_position()
                fig_single_local.text(
                    bbox.x0 - 0.01,
                    (bbox.y0 + bbox.y1) / 2,
                    display_group,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=FONT_SIZE,
                )

        def add_single_separator(fig_single_local: plt.Figure) -> None:
            top = axes_single[0, 0].get_position().y1
            bottom = axes_single[-1, 0].get_position().y0
            for class_idx in range(1, len(ordered_classes)):
                spacer_col = class_block_starts[class_idx] - 1
                spacer_bbox = axes_single[0, spacer_col].get_position()
                x = (spacer_bbox.x0 + spacer_bbox.x1) / 2
                fig_single_local.add_artist(
                    plt.Line2D(
                        [x, x],
                        [bottom, top],
                        transform=fig_single_local.transFigure,
                        color=SEPARATOR_COLOR,
                        linewidth=SEPARATOR_LW,
                        zorder=1000,
                        clip_on=False,
                    )
                )

        for row_idx, group_name in enumerate(ordered_groups):
            for class_idx, class_name in enumerate(ordered_classes):
                lookup_row = selected_row_lookup.get((group_name, class_name))
                original_col = class_block_starts[class_idx]
                method_cols = [original_col + 1 + sigma_idx for sigma_idx in range(n_sigma)]

                if lookup_row is None:
                    axes_single[row_idx, original_col].axis("off")
                    for col in method_cols:
                        axes_single[row_idx, col].axis("off")
                    continue

                _, idx = selected[lookup_row]
                image = images[idx]
                epi = float(epi_values[idx])
                img = image.squeeze(0).permute(1, 2, 0).numpy()
                img = (img * 0.5 + 0.5)
                img = np.clip(img, 0, 1)

                axes_single[row_idx, original_col].imshow(img)
                axes_single[row_idx, original_col].axis("off")
                axes_single[row_idx, original_col].set_title("", fontsize=FONT_SIZE)
                add_text_box(axes_single[row_idx, original_col], f"epi={epi:.4f}")

                for sigma_idx, col in enumerate(method_cols):
                    attr = attr_rows[lookup_row][sigma_idx][attr_index]
                    attr_vis = prepare_heatmap_for_display(attr)
                    ax = axes_single[row_idx, col]
                    if HEATMAP_BLEND_ON_IMAGE:
                        ax.imshow(img)
                    ax.imshow(
                        attr_vis,
                        cmap=HEATMAP_CMAP,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation=HEATMAP_INTERPOLATION,
                        alpha=HEATMAP_ALPHA if HEATMAP_BLEND_ON_IMAGE else 1.0,
                    )
                    ax.axis("off")

                    if clean_sigma_idx is None or sigma_idx == clean_sigma_idx:
                        ssim_value = None
                    else:
                        clean_attr = attr_rows[lookup_row][clean_sigma_idx][attr_index]
                        ssim_value = compute_ssim(clean_attr, attr)
                    add_ssim_box(ax, ssim_value)

        plt.tight_layout()
        fig_single.subplots_adjust(
            wspace=SINGLE_SUBPLOT_WSPACE,
            top=SINGLE_SUBPLOT_TOP,
            left=SUBPLOT_LEFT,
        )
        add_single_column_labels(fig_single)
        add_single_row_labels(fig_single)
        add_single_separator(fig_single)
        save_figure(fig_single, build_output_path(out_suffix))
        plt.close(fig_single)

    render_single_method("IG", 0, ig_vmin, ig_vmax, "_ig")
    render_single_method("SG", 1, sg_vmin, sg_vmax, "_sg")


def main() -> None:
    if not EPIS_PATH.exists():
        raise FileNotFoundError(f"Missing {EPIS_PATH}. Run 2_compute_epistemic.py first.")

    with open(EPIS_PATH, "rb") as f:
        data = pickle.load(f)

    images = data["images"]
    labels = data["labels"]
    label_names = data.get("label_names")
    class_names = data.get("class_names")

    if SIGMA not in data["epistemic_per_sigma"]:
        available = sorted(data["epistemic_per_sigma"].keys())
        raise ValueError(f"Sigma {SIGMA} missing in data. Available: {available}")

    epi_values = np.array(data["epistemic_per_sigma"][SIGMA])
    sample_paths, sample_labels, class_names_from_ds = resolve_sample_paths(len(images))
    if sample_labels and sample_labels != labels:
        print("Warning: label order mismatch vs EPIS data; file names may be offset.")

    if class_names is None and class_names_from_ds is not None:
        class_names = class_names_from_ds

    selected = select_indices(epi_values, labels, label_names, class_names)
    if not selected:
        print("No samples selected. Check indices and TOP_K settings.")
        return
    selected = sort_selected_rows(selected, labels, label_names, class_names)

    print_selection(selected, epi_values, labels, label_names, class_names, sample_paths)
    save_selection_csv(selected, epi_values, labels, label_names, class_names, sample_paths)

    device = get_device()
    model = load_model(device)
    plot_extremes(
        selected,
        epi_values,
        images,
        labels,
        label_names,
        class_names,
        sample_paths,
        model,
        device,
    )


if __name__ == "__main__":
    main()
