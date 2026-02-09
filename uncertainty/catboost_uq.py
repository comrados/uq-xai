from typing import Tuple, Optional
import numpy as np
from scipy.special import softmax

from config.settings import GLOBAL_SEED
from uncertainty.base_uq import UQWrapper
from models.catboost import CatBoostClassifierModel


class CatBoostClassifierUQ(UQWrapper):
    """CatBoost classification with tree-based uncertainty estimation.

    Leverages the internal boosting tree ensemble of CatBoost for UQ.
    Each tree contributes additively to the final prediction.

    Uncertainty decomposition:
    - Aleatoric: Predictive entropy of final model probabilities
    - Epistemic: Variance of predictions across individual boosting trees
    """

    def __init__(self,
                 base_model: Optional[CatBoostClassifierModel] = None,
                 seed: int = GLOBAL_SEED):
        """Initialize CatBoostClassifierUQ wrapper.

        Args:
            base_model: Pre-trained CatBoostClassifierModel. If None, creates new model.
            seed: Random seed for reproducibility.
        """
        if base_model is None:
            base_model = CatBoostClassifierModel(seed=seed)
        super().__init__(base_model)
        self.seed = seed

    @property
    def name(self) -> str:
        return f"{self.base_model.name}_uq"

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "CatBoostClassifierUQ":
        """Fit CatBoost model (uses internal boosting trees for epistemic UQ).

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features for early stopping
            y_val: Validation labels for early stopping

        Returns:
            self: Fitted CatBoostClassifierUQ instance
        """
        # CATBOOST BUILDS AN INTERNAL SEQUENCE OF BOOSTING TREES
        # NO EXTERNAL ENSEMBLES REQUIRED FOR EPISTEMIC UNCERTAINTY
        self.base_model.fit(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty decomposition.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            predictions: Class predictions, shape (n_samples,)
            aleatoric: Aleatoric uncertainty per sample, shape (n_samples,)
            epistemic: Epistemic uncertainty per sample, shape (n_samples,)
        """
        X_df = self.base_model._to_dataframe(X)

        # ALEATORIC FINAL MODEL PROBABILITIES
        mean_probs = self.base_model.model.predict_proba(X_df)
        predictions = np.argmax(mean_probs, axis=1)

        # ALEATORIC UNCERTAINTY PREDICTIVE ENTROPY OF FINAL PROBABILITIES
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

        # EPISTEMIC EXTRACT PER TREE PREDICTIONS
        model = self.base_model.model
        n_trees = model.tree_count_
        tree_probs = []

        for t in range(1, n_trees + 1):
            # RAW SCORE USING ONLY THE FIRST T TREES
            raw = model.predict(
                X_df,
                prediction_type="RawFormulaVal",
                ntree_end=t
            )

            # BINARY CLASSIFICATION
            if raw.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-raw))
                p0 = 1.0 - p1
                probs = np.stack([p0, p1], axis=1)
            else:
                # MULTI CLASS
                probs = softmax(raw, axis=1)

            tree_probs.append(probs)

        # SHAPE N TREES N SAMPLES N CLASSES
        tree_probs = np.array(tree_probs)

        # EPISTEMIC UNCERTAINTY VARIANCE ACROSS BOOSTING TREES
        epistemic = np.mean(np.var(tree_probs, axis=0), axis=1)

        return predictions, aleatoric, epistemic

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (mean across boosting trees).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        X_df = self.base_model._to_dataframe(X)
        model = self.base_model.model

        n_trees = model.tree_count_
        raw_first = model.predict(X_df, prediction_type="RawFormulaVal")
        n_classes = raw_first.shape[1] if raw_first.ndim == 2 else 2

        tree_probs = []

        for t in range(1, n_trees + 1):
            raw = model.predict(
                X_df,
                prediction_type="RawFormulaVal",
                ntree_end=t
            )

            if raw.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-raw))
                p0 = 1.0 - p1
                probs = np.stack([p0, p1], axis=1)
            else:
                probs = softmax(raw, axis=1)

            tree_probs.append(probs)

        return np.mean(tree_probs, axis=0)
