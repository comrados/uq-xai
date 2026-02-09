from typing import Optional, Literal, List
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np

from config.settings import GLOBAL_SEED, CATBOOST_DEFAULTS
from models.base import BaseModel


class BaseCatBoostModel(BaseModel):
    """Abstract base class for CatBoost models with shared functionality.

    Provides common training logic, progress tracking, early stopping, and
    DataFrame handling for CatBoost models. Subclasses must implement
    the _get_model() method to return the appropriate CatBoost model.

    Attributes:
        iterations: Number of boosting iterations.
        learning_rate: Learning rate for boosting.
        depth: Maximum tree depth.
        l2_leaf_reg: L2 regularization coefficient.
        early_stopping_rounds: Number of rounds without improvement before stopping.
        model: The underlying CatBoost model instance.
        show_progress: Whether to display training progress.
        _feature_names: Cached feature names for consistent DataFrame conversion.
    """

    def __init__(self,
                 iterations: int = CATBOOST_DEFAULTS['iterations'],
                 learning_rate: float = CATBOOST_DEFAULTS['learning_rate'],
                 depth: int = CATBOOST_DEFAULTS['depth'],
                 l2_leaf_reg: float = CATBOOST_DEFAULTS['l2_leaf_reg'],
                 early_stopping_rounds: int = CATBOOST_DEFAULTS['early_stopping_rounds'],
                 seed: int = GLOBAL_SEED,
                 show_progress: bool = True):
        """Initialize base CatBoost model.

        Args:
            iterations: Number of boosting rounds. Default from config.
            learning_rate: Boosting learning rate. Default from config.
            depth: Maximum tree depth. Default from config.
            l2_leaf_reg: L2 regularization coefficient. Default from config.
            early_stopping_rounds: Rounds without improvement before stopping.
                Default from config.
            seed: Random seed for reproducibility. Default from config.
            show_progress: Whether to show training progress bar.
        """
        super().__init__(seed=seed)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._feature_names: Optional[List[str]] = None
        self.show_progress = show_progress
    
    @property
    def name(self) -> str:
        """Generate a unique identifier for this model configuration.

        Returns:
            String in format: {task_type}_catboost_n{iterations}_lr{learning_rate}.
        """
        return f"{self.task_type}_catboost_n{self.iterations}_lr{self.learning_rate}"

    def _get_model(self):
        """Create and return the CatBoost model instance.

        Must be implemented by subclasses to return task-specific model.

        Returns:
            CatBoost model instance (e.g., CatBoostClassifier, CatBoostRegressor).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseCatBoostModel":
        """Train the CatBoost model with optional early stopping.

        Trains the model using gradient boosting. If validation data is
        provided, performs early stopping based on validation performance.
        Handles DataFrame conversion automatically for CatBoost compatibility.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features, shape (n_val_samples, n_features).
                Optional, enables early stopping if provided.
            y_val: Validation labels, shape (n_val_samples,).
                Optional, enables early stopping if provided.

        Returns:
            Self for method chaining.
        """
        self.model = self._get_model()
        self._set_feature_names(X_train)

        X_train_df = self._to_dataframe(X_train)
        y_train = np.asarray(y_train)

        if X_val is not None and y_val is not None:
            X_val_df = self._to_dataframe(X_val)
            y_val = np.asarray(y_val)
            
            self.model.fit(
                X_train_df, y_train,
                eval_set=(X_val_df, y_val),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=100 if self.show_progress else False
            )
        else:
            self.model.fit(
                X_train_df, y_train,
                verbose=100 if self.show_progress else False
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted labels, shape (n_samples,).
        """
        return self.model.predict(self._to_dataframe(X))

    def _set_feature_names(self, X: np.ndarray) -> None:
        """Capture feature names once for consistent DataFrame conversion.

        Extracts feature names from DataFrame or generates default names
        for numpy arrays. Stores names for reuse across all predictions.

        Args:
            X: Input data (DataFrame or ndarray).
        """
        if not hasattr(self, "_feature_names"):
            self._feature_names = None
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        elif self._feature_names is None:
            n_features = np.asarray(X).shape[1]
            self._feature_names = [f"f{i}" for i in range(n_features)]

    def _to_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """Convert input to DataFrame with stored feature names.

        Ensures consistent feature names across training and prediction
        by using cached names. Required for CatBoost compatibility.

        Args:
            X: Input data (DataFrame or ndarray), shape (n_samples, n_features).

        Returns:
            DataFrame with consistent column names.
        """
        if not hasattr(self, "_feature_names"):
            self._feature_names = None
        if isinstance(X, pd.DataFrame):
            return X
        if self._feature_names is None:
            if getattr(self, "model", None) is not None and hasattr(self.model, "feature_names_"):
                self._feature_names = list(self.model.feature_names_)
            else:
                self._set_feature_names(X)
        return pd.DataFrame(np.asarray(X), columns=self._feature_names)


class CatBoostClassifierModel(BaseCatBoostModel):
    """CatBoost gradient boosting classifier.

    Uses CatBoost's gradient boosting framework for efficient classification
    with support for large datasets and categorical features.

    Attributes:
        task_type: Classification task identifier.
    """

    task_type: Literal["classification"] = "classification"

    def _get_model(self):
        """Create CatBoost classifier with configured hyperparameters.

        Returns:
            Configured CatBoostClassifier instance.
        """
        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False  # Don't create info files
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, n_classes).
        """
        return self.model.predict_proba(self._to_dataframe(X))