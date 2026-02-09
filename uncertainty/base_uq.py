from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from models.base import BaseModel


class UQWrapper(ABC):
    """Base wrapper that adds uncertainty estimation to a model.

    Abstract base class for all uncertainty quantification wrappers.
    Wraps a base model and adds methods for uncertainty decomposition.
    """

    def __init__(self, base_model: BaseModel):
        """Initialize UQ wrapper.

        Args:
            base_model: Base model to wrap with uncertainty estimation.
        """
        self.base_model = base_model
        self.is_fitted = False
    
    @property
    def name(self) -> str:
        """Model name with UQ suffix."""
        return f"{self.base_model.name}_uq"

    @property
    def task_type(self) -> str:
        """Task type ('classification' or 'regression')."""
        return self.base_model.task_type
    
    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "UQWrapper":
        """Fit the UQ wrapper (train base model and uncertainty estimation).

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            self: Fitted UQWrapper instance
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty decomposition (aleatoric + epistemic).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            predictions: Point predictions, shape (n_samples,)
            aleatoric: Aleatoric uncertainty per sample, shape (n_samples,)
            epistemic: Epistemic uncertainty per sample, shape (n_samples,)
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions only (no uncertainty).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        preds, _, _ = self.predict_with_uncertainty(X)
        return preds

    def total_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Compute total uncertainty (aleatoric + epistemic).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Total uncertainty per sample, shape (n_samples,)
        """
        _, aleatoric, epistemic = self.predict_with_uncertainty(X)

        if self.task_type == "regression":
            # Regression: total = sqrt(aleatoric^2 + epistemic^2)
            return np.sqrt(aleatoric**2 + epistemic**2)
        else:
            # Classification: total = aleatoric + epistemic
            return aleatoric + epistemic
