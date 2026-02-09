from typing import Optional, Literal
import numpy as np
from sklearn.linear_model import LogisticRegression

from config.settings import GLOBAL_SEED, LOGISTIC_DEFAULTS
from models.base import BaseModel


class LogisticModel(BaseModel):
    """Logistic regression for classification.

    Wrapper around scikit-learn LogisticRegression with L2 regularization.

    Attributes:
        C: Inverse regularization strength (higher = less regularization)
        model: Underlying sklearn LogisticRegression instance
    """

    task_type: Literal["classification"] = "classification"

    def __init__(self, C: float = LOGISTIC_DEFAULTS['C'], seed: int = GLOBAL_SEED):
        """Initialize logistic regression model.

        Args:
            C: Inverse regularization strength. Default: 1.0.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.C = C
        self.model = LogisticRegression(
            C=C,
            random_state=seed,
            max_iter=1000,
            n_jobs=-1
        )
    
    @property
    def name(self) -> str:
        return f"{self.task_type}_logistic_C{self.C}"
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LogisticModel":
        """Train logistic regression model.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (unused for logistic)
            y_val: Validation labels (unused for logistic)

        Returns:
            self: Fitted LogisticModel instance
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        return self.model.predict_proba(X)