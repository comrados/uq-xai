from abc import ABC, abstractmethod
from typing import Optional, Literal
import numpy as np

from config.settings import GLOBAL_SEED


class BaseModel(ABC):
    """Abstract base class for all models.

    Defines the interface that all model implementations must follow.
    Supports both classification tasks (regression removed Dec 6, 2025).
    """

    def __init__(self, seed: int = GLOBAL_SEED):
        """Initialize base model.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.is_fitted = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name (includes hyperparameters)."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> Literal["regression", "classification"]:
        """Task type ('classification' only in current version)."""
        pass

    @property
    def supports_gradients(self) -> bool:
        """Whether model supports gradient-based methods (True for MLP only).

        Returns:
            True if model supports gradients (for gradient-based XAI), False otherwise.
        """
        return False
    
    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseModel":
        """Train the model.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)

        Returns:
            self: Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)

        Raises:
            NotImplementedError: If called on non-classification model
        """
        raise NotImplementedError("Only for classification models")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed}, fitted={self.is_fitted})"