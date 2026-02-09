from typing import Optional, Literal
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.settings import GLOBAL_SEED, RANDOM_FOREST_DEFAULTS
from models.base import BaseModel


class RandomForestClassifierModel(BaseModel):
    """Random Forest classifier implementation using scikit-learn.

    This class wraps scikit-learn's RandomForestClassifier and provides
    a consistent interface with other models in the framework.

    Attributes:
        task_type: Classification task identifier.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        model: Underlying RandomForestClassifier instance.
        is_fitted: Whether the model has been trained.
    """

    task_type: Literal["classification"] = "classification"

    def __init__(self,
                 n_estimators: int = RANDOM_FOREST_DEFAULTS['n_estimators'],
                 max_depth: int = RANDOM_FOREST_DEFAULTS['max_depth'],
                 min_samples_split: int = RANDOM_FOREST_DEFAULTS['min_samples_split'],
                 min_samples_leaf: int = RANDOM_FOREST_DEFAULTS['min_samples_leaf'],
                 seed: int = GLOBAL_SEED):
        """Initialize Random Forest classifier.

        Args:
            n_estimators: Number of decision trees in the forest. Default from config.
            max_depth: Maximum depth of each tree. Default from config.
            min_samples_split: Minimum number of samples required to split an
                internal node. Default from config.
            min_samples_leaf: Minimum number of samples required to be at a
                leaf node. Default from config.
            seed: Random seed for reproducibility. Default from config.
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1
        )
    
    @property
    def name(self) -> str:
        """Generate a unique identifier for this model configuration.

        Returns:
            String in format: classification_rf_n{n_estimators}_d{max_depth}.
        """
        return f"{self.task_type}_rf_n{self.n_estimators}_d{self.max_depth}"

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "RandomForestClassifierModel":
        """Train the Random Forest model on training data.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features, shape (n_val_samples, n_features).
                Not used by Random Forest but kept for interface consistency.
            y_val: Validation labels, shape (n_val_samples,).
                Not used by Random Forest but kept for interface consistency.

        Returns:
            Self for method chaining.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted class labels, shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X)