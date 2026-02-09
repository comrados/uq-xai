from typing import List
import numpy as np
from scipy.stats import kendalltau, spearmanr

from models.base import BaseModel


class ExplanationMetrics:
    """Metrics for evaluating explanation quality and stability.

    Provides methods to assess:
    - Stability: consistency of explanations across multiple runs
    - Fidelity: correlation between feature importance and actual impact
    - Rank correlation: consistency under perturbations
    - Attribution variance: variability of feature attributions
    """

    @staticmethod
    def stability_kendall_tau(attributions_list: List[np.ndarray]) -> float:
        """Measure explanation stability using Kendall's tau rank correlation.

        Computes pairwise Kendall's tau between feature rankings across K runs.
        Higher values indicate more stable explanations (consistent feature rankings).

        Args:
            attributions_list: List of attribution arrays from K runs,
                              each of shape (n_samples, n_features)

        Returns:
            Average Kendall's tau across all pairwise comparisons (0-1 scale)
        """
        if len(attributions_list) < 2:
            return 1.0  # Perfect stability if only one run

        tau_values = []

        # COMPUTE PAIRWISE KENDALL S TAU FOR ALL RUN PAIRS
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                # FLATTEN ATTRIBUTIONS FOR BOTH RUNS
                attr_i = attributions_list[i].flatten()
                attr_j = attributions_list[j].flatten()

                # RANK BY ABSOLUTE ATTRIBUTION VALUES
                ranks_i = np.argsort(np.abs(attr_i))
                ranks_j = np.argsort(np.abs(attr_j))

                # KENDALL S TAU RANK CORRELATION
                tau, _ = kendalltau(ranks_i, ranks_j)
                tau_values.append(tau)

        return float(np.mean(tau_values))

    @staticmethod
    def fidelity(
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        attributions: np.ndarray,
        top_k: int = 5
    ) -> float:
        """Measure explanation fidelity via feature ablation.

        Measures correlation between feature importance (from attributions)
        and actual performance drop when features are ablated (set to 0).

        High fidelity means important features (high attribution) cause
        large performance drops when removed.

        Args:
            model: Fitted BaseModel instance
            X: Input features, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)
            attributions: Feature attributions, shape (n_samples, n_features)
            top_k: Number of top features to ablate. Default: 5.

        Returns:
            Spearman correlation between attribution rank and performance drop
        """
        n_features = X.shape[1]

        # BASELINE PERFORMANCE NO ABLATION
        if model.task_type == "classification":
            y_pred_base = model.predict(X)
            baseline_acc = np.mean(y_pred_base == y)
        else:
            y_pred_base = model.predict(X)
            baseline_mse = np.mean((y_pred_base - y) ** 2)

        # COMPUTE AVERAGE ATTRIBUTION MAGNITUDE PER FEATURE
        feature_importance = np.mean(np.abs(attributions), axis=0)

        # RANK FEATURES BY IMPORTANCE
        feature_ranks = np.argsort(feature_importance)[::-1]  # Descending

        # ABLATE TOP K FEATURES ONE AT A TIME AND MEASURE PERFORMANCE DROP
        performance_drops = []

        for k in range(min(top_k, n_features)):
            feature_idx = feature_ranks[k]

            # ABLATE FEATURE SET TO 0
            X_ablated = X.copy()
            X_ablated[:, feature_idx] = 0

            # MEASURE PERFORMANCE
            if model.task_type == "classification":
                y_pred_ablated = model.predict(X_ablated)
                acc_ablated = np.mean(y_pred_ablated == y)
                drop = baseline_acc - acc_ablated
            else:
                y_pred_ablated = model.predict(X_ablated)
                mse_ablated = np.mean((y_pred_ablated - y) ** 2)
                drop = mse_ablated - baseline_mse

            performance_drops.append(drop)

        # CORRELATION BETWEEN RANK 1 TO TOP K AND PERFORMANCE DROP
        # HIGHER RANKED FEATURES SHOULD CAUSE LARGER DROPS
        ranks = np.arange(1, len(performance_drops) + 1)
        correlation, _ = spearmanr(ranks, performance_drops)

        return float(correlation)

    @staticmethod
    def rank_correlation_under_perturbation(
        attr_clean: np.ndarray,
        attr_perturbed: np.ndarray
    ) -> float:
        """Measure explanation consistency under data perturbation.

        Computes Spearman's rank correlation between feature rankings
        on clean vs perturbed data. High correlation indicates robust
        explanations that are consistent under noise.

        Args:
            attr_clean: Attributions on clean data, shape (n_samples, n_features)
            attr_perturbed: Attributions on perturbed data, shape (n_samples, n_features)

        Returns:
            Average Spearman's rho across all samples (0-1 scale)
        """
        n_samples = attr_clean.shape[0]
        correlations = []

        for i in range(n_samples):
            # RANK FEATURES BY ABSOLUTE ATTRIBUTION
            ranks_clean = np.argsort(np.abs(attr_clean[i]))
            ranks_perturbed = np.argsort(np.abs(attr_perturbed[i]))

            # SPEARMAN S RHO RANK CORRELATION
            rho, _ = spearmanr(ranks_clean, ranks_perturbed)
            correlations.append(rho)

        return float(np.mean(correlations))

    @staticmethod
    def attribution_variance(attributions_list: List[np.ndarray]) -> float:
        """Measure variance of attribution values across multiple runs.

        Lower variance indicates more stable explanations (less variability
        in attribution magnitudes across runs).

        Args:
            attributions_list: List of attribution arrays from K runs,
                              each of shape (n_samples, n_features)

        Returns:
            Average variance of attribution values across features and samples
        """
        # STACK ATTRIBUTIONS SHAPE K N SAMPLES N FEATURES
        attributions_stack = np.array(attributions_list)

        # VARIANCE ACROSS K RUNS SHAPE N SAMPLES N FEATURES
        variance_per_feature = np.var(attributions_stack, axis=0)

        # AVERAGE VARIANCE ACROSS ALL SAMPLES AND FEATURES
        return float(np.mean(variance_per_feature))
