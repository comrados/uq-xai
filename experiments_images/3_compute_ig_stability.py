"""Run an IG/SmoothGrad stability sweep.

Computes IG (and optionally SmoothGrad) attributions on noisy images across
sigma levels and stores SSIM stability metrics.
"""

import sys
import os
sys.path.append(os.getcwd())

import pickle
from typing import Any
import numpy as np
from tqdm import tqdm

from utils import (
    EPIS_PATH,
    SSIM_PATH,
    add_gaussian_noise,
    compute_ig,
    compute_smoothgrad,
    compute_ssim,
    get_device,
    load_model,
)

# SETTINGS
USE_SMOOTHGRAD: bool = True
SMOOTHGRAD_SAMPLES: int = 20
SMOOTHGRAD_NOISE: float = 0.1


def compute() -> None:
    """Compute IG/SG stability across noise levels and save results."""
    print("========== Compute IG Stability Sweep ==========")

    if not EPIS_PATH.exists():
        raise FileNotFoundError(f"Run 02_compute_epistemic.py first")

    with open(EPIS_PATH, "rb") as f:
        data = pickle.load(f)

    sigma_levels = data['sigma_levels']
    images = data['images']
    labels = data['labels']

    device = get_device()
    model = load_model(device)

    results: dict[str, Any] = {
        'sigma_levels': sigma_levels,
        'ssim_per_sigma': {}
    }
    if USE_SMOOTHGRAD:
        results['ssim_sg_per_sigma'] = {}
        results['smoothgrad'] = {
            'n_samples': SMOOTHGRAD_SAMPLES,
            'noise_sigma': SMOOTHGRAD_NOISE,
        }

    # COMPUTE IG ON CLEAN IMAGES ONCE
    print("\nComputing IG on clean images...")
    ig_clean = []
    sg_clean = []
    for image, label in tqdm(zip(images, labels), total=len(images), desc="IG Clean"):
        attr = compute_ig(model, image, target_class=label, device=device)
        ig_clean.append(attr)
        if USE_SMOOTHGRAD:
            sg_attr = compute_smoothgrad(
                model,
                image,
                target_class=label,
                device=device,
                n_samples=SMOOTHGRAD_SAMPLES,
                noise_sigma=SMOOTHGRAD_NOISE,
            )
            sg_clean.append(sg_attr)

    # FOR EACH SIGMA COMPUTE IG ON NOISY AND MEASURE SSIM
    for sigma in sigma_levels:
        if sigma == 0.0:
            # Clean vs clean = SSIM = 1.0
            results['ssim_per_sigma'][sigma] = np.ones(len(images))
            if USE_SMOOTHGRAD:
                results['ssim_sg_per_sigma'][sigma] = np.ones(len(images))
            print(f"\nsigma={sigma}: SSIM=1.0 (clean)")
            continue

        print(f"\n--- Sigma = {sigma} ---")
        ssim_values = []
        ssim_sg_values = []
        running_mean = 0.0
        running_mean_sg = 0.0

        progress = tqdm(zip(images, labels, ig_clean), total=len(images), desc=f"sigma={sigma}")
        for i, (image, label, attr_clean) in enumerate(progress, start=1):
            image_noisy = add_gaussian_noise(image, sigma=sigma)
            attr_noisy = compute_ig(model, image_noisy, target_class=label, device=device)
            
            ssim = compute_ssim(attr_clean, attr_noisy)
            ssim_values.append(ssim)
            running_mean += (ssim - running_mean) / i
            if USE_SMOOTHGRAD:
                sg_clean_attr = sg_clean[i - 1]
                sg_noisy = compute_smoothgrad(
                    model,
                    image_noisy,
                    target_class=label,
                    device=device,
                    n_samples=SMOOTHGRAD_SAMPLES,
                    noise_sigma=SMOOTHGRAD_NOISE,
                )
                ssim_sg = compute_ssim(sg_clean_attr, sg_noisy)
                ssim_sg_values.append(ssim_sg)
                running_mean_sg += (ssim_sg - running_mean_sg) / i
                progress.set_postfix(ig=f"{running_mean:.3f}", sg=f"{running_mean_sg:.3f}")
            else:
                progress.set_postfix(mean=f"{running_mean:.3f}")

        results['ssim_per_sigma'][sigma] = np.array(ssim_values)
        mean_val = np.mean(ssim_values)
        std_val = np.std(ssim_values)
        print(f"  IG SSIM: {mean_val:.3f} ± {std_val:.3f}")
        if USE_SMOOTHGRAD:
            results['ssim_sg_per_sigma'][sigma] = np.array(ssim_sg_values)
            mean_sg = np.mean(ssim_sg_values)
            std_sg = np.std(ssim_sg_values)
            print(f"  SG SSIM: {mean_sg:.3f} ± {std_sg:.3f}")

    # SAVE
    with open(SSIM_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"\n  Saved SSIM sweep to {SSIM_PATH}")


def main() -> None:
    """Entry point."""
    compute()


if __name__ == "__main__":
    main()
