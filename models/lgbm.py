from typing import Optional, Literal, List, Tuple, Callable
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from tqdm import tqdm

from config.settings import GLOBAL_SEED, LIGHTGBM_DEFAULTS
from models.base import BaseModel


class BaseLightGBMModel(BaseModel):
    """Abstract base class for LightGBM models with shared functionality.

    Provides common training logic, progress tracking, early stopping, and
    DataFrame handling for LightGBM models. Subclasses must implement
    the _get_model() method to return the appropriate LightGBM model.

    Attributes:
        n_estimators: Number of boosting iterations.
        learning_rate: Learning rate for boosting.
        max_depth: Maximum tree depth.
        num_leaves: Maximum number of leaves in one tree.
        early_stopping_rounds: Number of rounds without improvement before stopping.
        subsample: Fraction of samples used for each iteration.
        subsample_freq: Frequency of subsampling.
        colsample_bytree: Fraction of features used for each tree.
        model: The underlying LightGBM model instance.
        show_progress: Whether to display training progress.
        _feature_names: Cached feature names for consistent DataFrame conversion.
    """

    def __init__(self,
                 n_estimators: int = LIGHTGBM_DEFAULTS['n_estimators'],
                 learning_rate: float = LIGHTGBM_DEFAULTS['learning_rate'],
                 max_depth: int = LIGHTGBM_DEFAULTS['max_depth'],
                 num_leaves: int = LIGHTGBM_DEFAULTS['num_leaves'],
                 early_stopping_rounds: int = LIGHTGBM_DEFAULTS['early_stopping_rounds'],
                 subsample=LIGHTGBM_DEFAULTS['subsample'],
                 subsample_freq=LIGHTGBM_DEFAULTS['subsample_freq'],
                 colsample_bytree=LIGHTGBM_DEFAULTS['colsample_bytree'],
                 seed: int = GLOBAL_SEED,
                 show_progress: bool = True):
        """Initialize base LightGBM model.

        Args:
            n_estimators: Number of boosting rounds. Default from config.
            learning_rate: Boosting learning rate. Default from config.
            max_depth: Maximum tree depth. -1 means no limit. Default from config.
            num_leaves: Maximum tree leaves. Default from config.
            early_stopping_rounds: Rounds without improvement before stopping.
                Default from config.
            subsample: Row sampling ratio. Default from config.
            subsample_freq: Frequency of row sampling. Default from config.
            colsample_bytree: Feature sampling ratio per tree. Default from config.
            seed: Random seed for reproducibility. Default from config.
            show_progress: Whether to show training progress bar.
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.model = None
        self._feature_names: Optional[List[str]] = None
        self.show_progress = show_progress
    
    @property
    def name(self) -> str:
        """Generate a unique identifier for this model configuration.

        Returns:
            String in format: {task_type}_lgbm_n{n_estimators}_lr{learning_rate}.
        """
        return f"{self.task_type}_lgbm_n{self.n_estimators}_lr{self.learning_rate}"

    def _get_model(self):
        """Create and return the LightGBM model instance.

        Must be implemented by subclasses to return task-specific model.

        Returns:
            LightGBM model instance (e.g., LGBMClassifier, LGBMRegressor).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseLightGBMModel":
        """Train the LightGBM model with optional early stopping.

        Trains the model using gradient boosting. If validation data is
        provided, performs early stopping based on validation performance.
        Handles DataFrame conversion automatically for LightGBM compatibility.

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

        progress_cb, pbar = self._progress_callback(self.name)
        callbacks = []
        if progress_cb:
            callbacks.append(progress_cb)

        if X_val is not None and y_val is not None:
            X_val = self._to_dataframe(X_val)
            y_val = np.asarray(y_val)
            callbacks.append(early_stopping(self.early_stopping_rounds, verbose=False))
            try:
                self.model.fit(
                    X_train_df, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks if callbacks else None
                )
            finally:
                if pbar:
                    pbar.close()
        else:
            try:
                self.model.fit(X_train_df, y_train, callbacks=callbacks if callbacks else None)
            finally:
                if pbar:
                    pbar.close()
        
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
        by using cached names. Required for LightGBM compatibility.

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
            if getattr(self, "model", None) is not None and hasattr(self.model, "feature_name_"):
                self._feature_names = list(self.model.feature_name_)
            else:
                self._set_feature_names(X)
        return pd.DataFrame(np.asarray(X), columns=self._feature_names)

    def _progress_callback(self, desc: Optional[str] = None) -> Tuple[Optional[Callable], Optional[tqdm]]:
        """Create a LightGBM-compatible callback for progress tracking.

        Generates a callback function and associated tqdm progress bar
        for monitoring training progress. The callback updates the progress
        bar at each boosting iteration.

        Args:
            desc: Description to display on the progress bar.

        Returns:
            Tuple of (callback_function, progress_bar). Both are None if
            show_progress is False.
        """
        if not self.show_progress:
            return None, None

        pbar = tqdm(total=self.n_estimators, desc=desc or self.name, leave=False)

        def _callback(env):
            # env.iteration is 0-based; update delta to current iteration count.
            current = env.iteration + 1
            delta = current - pbar.n
            if delta > 0:
                pbar.update(delta)

        _callback.order = 0  # run early
        return _callback, pbar


class LightGBMClassifierModel(BaseLightGBMModel):
    """LightGBM gradient boosting classifier.

    Uses LightGBM's gradient boosting framework for efficient classification
    with support for large datasets and categorical features.

    Attributes:
        task_type: Classification task identifier.
    """

    task_type: Literal["classification"] = "classification"

    def _get_model(self):
        """Create LightGBM classifier with configured hyperparameters.

        Returns:
            Configured LGBMClassifier instance.
        """
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            verbose=-1
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, n_classes).
        """
        return self.model.predict_proba(self._to_dataframe(X))
