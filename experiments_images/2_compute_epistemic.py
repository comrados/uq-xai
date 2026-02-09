"""Run an MC Dropout epistemic sweep.

Runs MC Dropout on a fixed sample of test images across noise levels and saves
epistemic/entropy/confidence/logit variance metrics.
"""

import sys
import os
sys.path.append(os.getcwd())

from itertools import islice
import pickle
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from utils import (
    EPIS_PATH,
    RESULTS_DIR,
    TEST_DIR,
    add_gaussian_noise,
    get_device,
    get_transform,
    load_model,
)

# SETTINGS
BATCH_SIZE: int = 1
N_SAMPLES: int = 100
MC_SAMPLES: int = 50
SIGMA_LEVELS: list[float] = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
SEED: int = 42


def build_loader() -> DataLoader:
    """Create a deterministic, shuffled test loader.

    Returns:
        Test DataLoader with deterministic shuffling.
    """
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Missing test folder at {TEST_DIR}")

    transform = get_transform()
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, generator=gen)


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout.

    Args:
        model: Model whose dropout layers should be enabled.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def compute_saturation_metrics(
    model: nn.Module,
    image: torch.Tensor,
    n_samples: int = 50,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """
    Compute uncertainty metrics for saturation analysis.

    Args:
        model: Model to evaluate.
        image: Input image tensor with shape (1, 3, 128, 128).
        n_samples: Number of MC Dropout samples.
        device: Torch device or device string.
    Returns:
        dict with keys:
            - epistemic: MC Dropout variance (original metric)
            - entropy: Predictive entropy H(y|x)
            - max_prob: Maximum softmax probability (confidence)
            - logit_variance: Variance of logits before softmax
            - prediction_diversity: Fraction of unique predicted classes across MC runs
    """
    model.eval()
    enable_dropout(model)
    
    mc_probs = []
    mc_logits = []
    mc_predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(image.to(device))
            probs = F.softmax(logits, dim=1)
            
            mc_probs.append(probs.cpu().numpy())
            mc_logits.append(logits.cpu().numpy())
            mc_predictions.append(probs.argmax(dim=1).cpu().numpy())
    
    mc_probs = np.array(mc_probs)  # (n_samples, batch_size, n_classes)
    mc_logits = np.array(mc_logits)
    mc_predictions = np.array(mc_predictions)  # (n_samples, batch_size)
    
    # 1 EPISTEMIC UNCERTAINTY VARIANCE OF PROBABILITIES
    epistemic = np.var(mc_probs, axis=0).mean()
    
    # 2 PREDICTIVE ENTROPY MONOTONIC WITH NOISE
    mean_probs = mc_probs.mean(axis=0)[0]  # (n_classes,)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    
    # 3 MAX PROBABILITY CONFIDENCE DROPS WITH NOISE
    max_prob = mean_probs.max()
    
    # 4 LOGIT VARIANCE VARIANCE BEFORE SOFTMAX
    logit_variance = np.var(mc_logits, axis=0).mean()
    
    # 5 PREDICTION DIVERSITY UNIQUE PREDICTED CLASSES
    unique_predictions = len(np.unique(mc_predictions))
    n_classes = mc_probs.shape[2]
    prediction_diversity = unique_predictions / n_classes  # Normalize by n_classes
    
    return {
        'epistemic': epistemic,
        'entropy': entropy,
        'max_prob': max_prob,
        'logit_variance': logit_variance,
        'prediction_diversity': prediction_diversity
    }


def compute() -> None:
    """Compute uncertainty metrics across noise levels and save results."""
    print("========== Compute Epistemic Sweep with Saturation Metrics ==========")
    print(f"Sigma levels: {SIGMA_LEVELS}")
    print(f"MC samples: {MC_SAMPLES}")

    device = get_device()
    model = load_model(device)
    test_loader = build_loader()
    class_names = getattr(test_loader.dataset, "classes", None)

    # COLLECT CLEAN IMAGES FIRST
    print(f"\nLoading {N_SAMPLES} clean samples...")
    test_images = []
    test_labels = []
    
    for image, label in tqdm(islice(test_loader, N_SAMPLES), total=N_SAMPLES, desc="Loading"):
        test_images.append(image)
        test_labels.append(label.item())

    # COMPUTE ALL METRICS FOR EACH SIGMA LEVEL
    results: dict[str, Any] = {
        'sigma_levels': SIGMA_LEVELS,
        'images': test_images,
        'labels': test_labels,
        'epistemic_per_sigma': {},
        'entropy_per_sigma': {},
        'max_prob_per_sigma': {},
        'logit_variance_per_sigma': {},
        'prediction_diversity_per_sigma': {}
    }
    if class_names:
        results['class_names'] = list(class_names)
        results['label_names'] = [
            class_names[label] if 0 <= label < len(class_names) else f"class_{label}"
            for label in test_labels
        ]

    for sigma in SIGMA_LEVELS:
        print(f"\n--- Sigma = {sigma} ---")
        
        epistemic_values = []
        entropy_values = []
        max_prob_values = []
        logit_var_values = []
        diversity_values = []
        
        running_means = {
            'epistemic': 0.0,
            'entropy': 0.0,
            'max_prob': 0.0,
            'logit_var': 0.0,
            'diversity': 0.0
        }

        progress = tqdm(test_images, desc=f"sigma={sigma}")
        for i, image in enumerate(progress, start=1):
            if sigma == 0.0:
                image_noisy = image
            else:
                image_noisy = add_gaussian_noise(image, sigma=sigma)
            
            # COMPUTE ALL METRICS
            metrics = compute_saturation_metrics(
                model, 
                image_noisy, 
                n_samples=MC_SAMPLES, 
                device=device
            )
            
            epistemic_values.append(metrics['epistemic'])
            entropy_values.append(metrics['entropy'])
            max_prob_values.append(metrics['max_prob'])
            logit_var_values.append(metrics['logit_variance'])
            diversity_values.append(metrics['prediction_diversity'])
            
            # UPDATE RUNNING MEANS
            for key, val in metrics.items():
                metric_name = 'logit_var' if key == 'logit_variance' else key.replace('prediction_', '')
                running_means[metric_name] += (val - running_means[metric_name]) / i
            
            progress.set_postfix(
                epi=f"{running_means['epistemic']:.4f}",
                ent=f"{running_means['entropy']:.3f}",
                conf=f"{running_means['max_prob']:.3f}"
            )

        # STORE RESULTS
        results['epistemic_per_sigma'][sigma] = np.array(epistemic_values)
        results['entropy_per_sigma'][sigma] = np.array(entropy_values)
        results['max_prob_per_sigma'][sigma] = np.array(max_prob_values)
        results['logit_variance_per_sigma'][sigma] = np.array(logit_var_values)
        results['prediction_diversity_per_sigma'][sigma] = np.array(diversity_values)
        
        # PRINT SUMMARY
        print(f"  Epistemic:    {np.mean(epistemic_values):.4f} ± {np.std(epistemic_values):.4f}")
        print(f"  Entropy:      {np.mean(entropy_values):.4f} ± {np.std(entropy_values):.4f}")
        print(f"  Max Prob:     {np.mean(max_prob_values):.4f} ± {np.std(max_prob_values):.4f}")
        print(f"  Logit Var:    {np.mean(logit_var_values):.4f} ± {np.std(logit_var_values):.4f}")
        print(f"  Diversity:    {np.mean(diversity_values):.4f} ± {np.std(diversity_values):.4f}")

    # SAVE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EPIS_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved epistemic sweep with saturation metrics to {EPIS_PATH}")
    
    # PRINT SUMMARY TABLE
    print("\n" + "="*80)
    print("SUMMARY TABLE - Saturation Analysis")
    print("="*80)
    print(f"{'sigma':<6} {'Epistemic':<12} {'Entropy':<12} {'MaxProb':<12} {'LogitVar':<12} {'Diversity':<12}")
    print("-"*80)
    
    for sigma in SIGMA_LEVELS:
        epi_mean = np.mean(results['epistemic_per_sigma'][sigma])
        ent_mean = np.mean(results['entropy_per_sigma'][sigma])
        prob_mean = np.mean(results['max_prob_per_sigma'][sigma])
        logit_mean = np.mean(results['logit_variance_per_sigma'][sigma])
        div_mean = np.mean(results['prediction_diversity_per_sigma'][sigma])
        
        print(f"{sigma:<6.3f} {epi_mean:<12.4f} {ent_mean:<12.4f} {prob_mean:<12.4f} {logit_mean:<12.4f} {div_mean:<12.4f}")
    
    print("="*80)
    print("\nExpected behavior:")
    print("  Entropy increases monotonically")
    print("  MaxProb decreases monotonically")
    print("  Epistemic peaks then saturates (variance collapse)")
    print("  LogitVar peaks then saturates")
    print("  Diversity peaks then saturates (all runs confused similarly)")


def main() -> None:
    """Entry point."""
    compute()


if __name__ == "__main__":
    main()
