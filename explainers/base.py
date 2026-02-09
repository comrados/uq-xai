from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from models.base import BaseModel


class BaseExplainer(ABC):
    """Abstract base class for all explainers.

    Defines the interface for generating feature attributions (explanations)
    for model predictions. All explainer implementations must inherit from this.

    Attributes:
        model: Fitted BaseModel instance to explain
    """

    def __init__(self, model: BaseModel):
        """Initialize explainer.

        Args:
            model: Fitted BaseModel instance to explain
        """
        self.model = model

    @abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Generate feature attributions for input samples.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Feature attributions, shape (n_samples, n_features)
        """
        pass

    def explain_with_stability(self, X: np.ndarray, K: int = 10) -> Tuple[np.ndarray, float]:
        """Generate explanations with stability measurement.

        Runs explanation K times and computes stability score using Kendall's tau.
        Higher stability indicates more consistent feature rankings across runs.

        Args:
            X: Input features, shape (n_samples, n_features)
            K: Number of explanation runs for stability estimation. Default: 10.

        Returns:
            mean_attributions: Mean attributions across K runs, shape (n_samples, n_features)
            stability: Kendall's tau stability score (0-1, higher = more stable)
        """
        from evaluation.explanation import ExplanationMetrics

        attributions_list = [self.explain(X) for _ in range(K)]
        stability = ExplanationMetrics.stability_kendall_tau(attributions_list)
        mean_attr = np.mean(attributions_list, axis=0)

        return mean_attr, stability


class ExplanationSubsampler:
    """Utility for subsampling large datasets before explanation.

    For large test sets, generating explanations on all samples is slow.
    This class provides random subsampling to speed up explanation generation.

    Attributes:
        n_samples: Maximum number of samples to explain
        random_state: Random seed for reproducibility
    """

    def __init__(self, n_samples: int = 1000, random_state: int = 42):
        """Initialize subsampler.

        Args:
            n_samples: Maximum number of samples to keep. Default: 1000.
            random_state: Random seed for reproducibility. Default: 42.
        """
        self.n_samples = n_samples
        self.random_state = random_state

    def subsample(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Subsample data for explanation.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Optional labels, shape (n_samples,)

        Returns:
            X_sub: Subsampled features, shape (min(n_samples, len(X)), n_features)
            y_sub: Subsampled labels (None if y is None)
            indices: Selected indices
        """
        if len(X) <= self.n_samples:
            return X, y, np.arange(len(X))

        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_samples, replace=False)

        X_sub = X[indices]
        y_sub = y[indices] if y is not None else None

        return X_sub, y_sub, indices
