"""Plot epistemic vs stability correlations.

Loads epistemic and SSIM sweep results, computes Spearman correlations, and
saves correlation scatter plots.
"""

import sys
import os
sys.path.append(os.getcwd())

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr

from utils import RESULTS_DIR, EPIS_PATH, SSIM_PATH
from utils import get_device, load_model, compute_ig, compute_smoothgrad, add_gaussian_noise


def plot_correlation() -> None:
    """Create correlation and sweep plots for epistemic vs SSIM."""
    print("========== Correlation Analysis ==========")

    with open(EPIS_PATH, "rb") as f:
        epis_data = pickle.load(f)
    with open(SSIM_PATH, "rb") as f:
        ssim_data = pickle.load(f)

    sigma_levels = epis_data['sigma_levels']
    has_smoothgrad = 'ssim_sg_per_sigma' in ssim_data
    
    # AGGREGATE MEAN EPISTEMIC AND SSIM PER SIGMA
    epistemic_means = []
    epistemic_stds = []
    ssim_means = []
    ssim_stds = []
    ssim_sg_means = []
    ssim_sg_stds = []
    
    for sigma in sigma_levels:
        epi = epis_data['epistemic_per_sigma'][sigma]
        ssim = ssim_data['ssim_per_sigma'][sigma]
        epistemic_means.append(np.mean(epi))
        epistemic_stds.append(np.std(epi))
        ssim_means.append(np.mean(ssim))
        ssim_stds.append(np.std(ssim))
        if has_smoothgrad:
            ssim_sg = ssim_data['ssim_sg_per_sigma'][sigma]
            ssim_sg_means.append(np.mean(ssim_sg))
            ssim_sg_stds.append(np.std(ssim_sg))
    
    # CORRELATION
    rho, pval = spearmanr(epistemic_means, ssim_means)
    print(f"\nSpearman rho(epistemic, SSIM) = {rho:.3f} (p={pval:.4f})")
    if has_smoothgrad:
        rho_sg, pval_sg = spearmanr(epistemic_means, ssim_sg_means)
        print(f"Spearman rho(epistemic, SSIM_SG) = {rho_sg:.3f} (p={pval_sg:.4f})")
    
    # PLOT
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sigma_array = np.array(sigma_levels, dtype=float)
    sigma_sorted = np.sort(sigma_array)
    base_colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_sorted)))
    cmap = ListedColormap(base_colors)
    if len(sigma_sorted) > 1:
        deltas = np.diff(sigma_sorted)
        boundaries = np.concatenate((
            [sigma_sorted[0] - deltas[0] / 2],
            sigma_sorted[:-1] + deltas / 2,
            [sigma_sorted[-1] + deltas[-1] / 2],
        ))
    else:
        boundaries = np.array([sigma_sorted[0] - 0.5, sigma_sorted[0] + 0.5])
    norm = BoundaryNorm(boundaries, ncolors=len(sigma_sorted))
    
    for i, sigma in enumerate(sigma_levels):
        color = cmap(norm(sigma))
        ax.scatter(
            epistemic_means[i],
            ssim_means[i],
            s=150,
            c=[color],
            edgecolors='black',
            linewidths=1.5,
            zorder=10 - i,
        )
        if has_smoothgrad:
            ax.scatter(
                epistemic_means[i],
                ssim_sg_means[i],
                s=110,
                c=[color],
                marker='s',
                edgecolors='black',
                linewidths=1.2,
                zorder=10 - i,
            )
    
    ax.set_xlabel("Mean Epistemic Uncertainty", fontsize=12)
    ax.set_ylabel("Mean SSIM (IG Stability)", fontsize=12)
    if has_smoothgrad:
        ax.set_title(
            "PlantVillage: Epistemic vs SSIM across noise levels\n"
            f"Spearman rho = {rho:.3f} (p={pval:.4f}) | circles=IG, squares=SG",
            fontsize=13,
        )
        ig_handle = plt.Line2D([0], [0], marker='o', linestyle='None',
                               markerfacecolor='white', markeredgecolor='black',
                               markersize=8, label='IG')
        sg_handle = plt.Line2D([0], [0], marker='s', linestyle='None',
                               markerfacecolor='white', markeredgecolor='black',
                               markersize=8, label='SG')
    else:
        ax.set_title(
            "PlantVillage: Epistemic vs SSIM across noise levels\n"
            f"Spearman rho = {rho:.3f} (p={pval:.4f})",
            fontsize=13,
        )
    if has_smoothgrad:
        ax.legend(handles=[ig_handle, sg_handle], title="Marker", fontsize=9, title_fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, ticks=sigma_sorted)
    cbar.set_label("noise level (sigma)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_sweep.png", dpi=300)
    print("Saved correlation plot")

    # SmoothGrad trend is included in the same sweep plot above.


def plot_dual_axis_trends() -> None:
    """Create a dual-axis trend plot for epistemic and SSIM vs sigma."""
    print("\n========== Dual Axis Trends ==========")

    with open(EPIS_PATH, "rb") as f:
        epis_data = pickle.load(f)
    with open(SSIM_PATH, "rb") as f:
        ssim_data = pickle.load(f)

    sigma_levels = epis_data['sigma_levels']
    has_smoothgrad = 'ssim_sg_per_sigma' in ssim_data

    epistemic_means = []
    epistemic_stds = []
    ssim_means = []
    ssim_stds = []
    ssim_sg_means = []
    ssim_sg_stds = []

    for sigma in sigma_levels:
        epi = epis_data['epistemic_per_sigma'][sigma]
        ssim = ssim_data['ssim_per_sigma'][sigma]
        epistemic_means.append(np.mean(epi))
        epistemic_stds.append(np.std(epi))
        ssim_means.append(np.mean(ssim))
        ssim_stds.append(np.std(ssim))
        if has_smoothgrad:
            ssim_sg = ssim_data['ssim_sg_per_sigma'][sigma]
            ssim_sg_means.append(np.mean(ssim_sg))
            ssim_sg_stds.append(np.std(ssim_sg))

    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax2 = ax1.twinx()

    c_left = "steelblue"
    c_right = "darkorange"

    line1 = ax1.plot(
        sigma_levels,
        epistemic_means,
        'X-',
        linewidth=2,
        markersize=7,
        color=c_left,
        label='Epistemic',
    )[0]
    ax1.fill_between(
        sigma_levels,
        np.array(epistemic_means) - np.array(epistemic_stds),
        np.array(epistemic_means) + np.array(epistemic_stds),
        color=c_left,
        alpha=0.15,
        label='_nolegend_',
    )
    line2 = ax2.plot(
        sigma_levels,
        ssim_means,
        'o-',
        linewidth=2,
        markersize=6,
        color=c_right,
        label='IG SSIM',
        markeredgewidth=1.5,
        markeredgecolor=c_right,
        markerfacecolor=c_right,
    )[0]
    ax2.fill_between(
        sigma_levels,
        np.array(ssim_means) - np.array(ssim_stds),
        np.array(ssim_means) + np.array(ssim_stds),
        color=c_right,
        alpha=0.15,
        label='_nolegend_',
    )
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]

    if has_smoothgrad:
        line3 = ax2.plot(
            sigma_levels,
            ssim_sg_means,
            's-',
            linewidth=2,
            markersize=6,
            color=c_right,
            linestyle='--',
            dashes=(2, 2),
            label='SG SSIM',
            markeredgewidth=1.5,
            markeredgecolor=c_right,
            markerfacecolor=c_right,
        )[0]
        ax2.fill_between(
            sigma_levels,
            np.array(ssim_sg_means) - np.array(ssim_sg_stds),
            np.array(ssim_sg_means) + np.array(ssim_sg_stds),
            color=c_right,
            alpha=0.15,
            label='_nolegend_',
        )
        lines.append(line3)
        labels = [l.get_label() for l in lines]

    ax1.set_xlabel("Noise Level (sigma)", fontsize=12)
    ax1.set_xlim(0.0, 0.2)
    ax1.set_ylabel("Epistemic Uncertainty", fontsize=12, color=c_left)
    ax1.tick_params(axis="y", colors=c_left)
    ax1.spines["left"].set_color(c_left)
    ax1.spines["right"].set_visible(False)
    ax2.set_ylabel("SSIM", fontsize=12, color=c_right)
    ax2.tick_params(axis="y", colors=c_right)
    ax2.spines["right"].set_color(c_right)
    ax2.spines["left"].set_visible(False)
    ax1.set_title("PlantVillage: Epistemic and SSIM vs Noise", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(lines, labels, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dual_axis_trends.pdf", dpi=300)
    print("Saved dual-axis trends plot")


def plot_visual_examples(requested_sigmas = [0.0, 0.1, 0.2], sample_indices = [1, 2, 3]) -> None:
    """Show images and attribution heatmaps across noise levels."""
    print("\n========== Visual Examples ==========")
    
    with open(EPIS_PATH, "rb") as f:
        epis_data = pickle.load(f)
    with open(SSIM_PATH, "rb") as f:
        ssim_data = pickle.load(f)
    
    device = get_device()
    model = load_model(device)
    
    sigma_levels = epis_data['sigma_levels']
    images = epis_data['images']
    labels = epis_data['labels']
    has_smoothgrad = 'ssim_sg_per_sigma' in ssim_data
    class_names = epis_data.get('class_names')
    label_names = epis_data.get('label_names')

    missing_sigmas = [s for s in requested_sigmas if s not in sigma_levels]
    if missing_sigmas:
        print(f"Warning: requested sigma levels missing in data: {missing_sigmas}")
    sigma_levels = [s for s in requested_sigmas if s in sigma_levels]
    if not sigma_levels:
        raise ValueError("No requested sigma levels available in data for visual examples.")

    def plot_examples(kind: str) -> None:
        rows_per_sample = 2
        n_rows = len(sample_indices) * rows_per_sample
        n_cols = len(sigma_levels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, n_cols)
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for row, idx in enumerate(sample_indices):
            image = images[idx]
            label = labels[idx]
            base_row = row * rows_per_sample
            img_row = base_row
            heat_row = base_row + 1
            label_index = int(label)
            if label_names and idx < len(label_names):
                label_name = label_names[idx]
            elif class_names and 0 <= label_index < len(class_names):
                label_name = class_names[label_index]
            else:
                label_name = f"class_{label_index}"

            for col, sigma in enumerate(sigma_levels):
                if sigma == 0.0:
                    img_noisy = image
                else:
                    img_noisy = add_gaussian_noise(image, sigma=sigma)

                if kind == "ig":
                    attr = compute_ig(model, img_noisy, target_class=label, device=device)
                    ssim = ssim_data['ssim_per_sigma'][sigma][idx]
                    title_label = "IG"
                else:
                    attr = compute_smoothgrad(model, img_noisy, target_class=label, device=device)
                    ssim = ssim_data['ssim_sg_per_sigma'][sigma][idx]
                    title_label = "SG"

                epi = epis_data['epistemic_per_sigma'][sigma][idx]

                img = img_noisy.squeeze().permute(1, 2, 0).numpy()
                img = (img * 0.5 + 0.5)
                img = np.clip(img, 0, 1)
                axes[img_row, col].imshow(img)
                axes[img_row, col].axis('off')

                if row == 0:
                    if sigma == 0.0:
                        axes[img_row, col].set_title("Clean", fontsize=10)
                    else:
                        axes[img_row, col].set_title(f"sigma={sigma}", fontsize=10)
                if col == 0:
                    axes[img_row, col].text(
                        -0.08,
                        0.5,
                        f"{label_name}\nidx={idx}",
                        transform=axes[img_row, col].transAxes,
                        ha='right',
                        va='center',
                        fontsize=9,
                        clip_on=False,
                    )

                axes[heat_row, col].imshow(attr, cmap='viridis', interpolation='bilinear')
                axes[heat_row, col].axis('off')

                if sigma == 0.0:
                    axes[heat_row, col].set_title(f"{title_label}\nEpi={epi:.4f}", fontsize=10)
                else:
                    axes[heat_row, col].set_title(
                        f"{title_label}\nEpi={epi:.4f}\nSSIM={ssim:.3f}",
                        fontsize=10,
                    )

        plt.tight_layout()
        out_name = "visual_examples_ig.png" if kind == "ig" else "visual_examples_sg.png"
        plt.savefig(RESULTS_DIR / out_name, dpi=150, bbox_inches='tight')
        print(f"Saved visual examples: {out_name}")
        plt.close(fig)

    plot_examples("ig")
    if has_smoothgrad:
        plot_examples("sg")


def main() -> None:
    """Entry point."""
    plot_correlation()
    plot_visual_examples(requested_sigmas = [0.0, 0.1, 0.2], sample_indices = [1, 29, 34])
    plot_dual_axis_trends()
    print("\nAnalysis complete")


if __name__ == "__main__":
    main()
