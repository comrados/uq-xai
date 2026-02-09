from typing import Tuple, Optional
import numpy as np
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV

from config.settings import GLOBAL_SEED
from uncertainty.base_uq import UQWrapper
from models.lgbm import LightGBMClassifierModel


class LightGBMClassifierUQ(UQWrapper):
    """LightGBM classification with tree-based uncertainty estimation and Platt Scaling calibration.

    Leverages the internal boosting tree ensemble of LightGBM for UQ.
    Applies Platt Scaling (sigmoid calibration) on validation set to fix miscalibration.

    Uncertainty decomposition:
    - Aleatoric: Predictive entropy of CALIBRATED probabilities
    - Epistemic: Variance of predictions across boosting trees
    
    Calibration:
    - Uses sklearn.calibration.CalibratedClassifierCV with method='sigmoid'
    - Requires validation set (X_val, y_val) during fit()
    - ECE should drop from 0.3-0.7 to 0.05-0.15 after calibration
    """

    def __init__(self,
                 base_model: Optional[LightGBMClassifierModel] = None,
                 seed: int = GLOBAL_SEED):
        """Initialize LightGBMClassifierUQ wrapper.

        Args:
            base_model: Pre-trained LightGBMClassifierModel. If None, creates new model.
            seed: Random seed for reproducibility.
        """
        if base_model is None:
            base_model = LightGBMClassifierModel(seed=seed)
        super().__init__(base_model)
        self.seed = seed
        self.calibrated_model = None  # Platt Scaling calibrated model

    @property
    def name(self) -> str:
        return f"{self.base_model.name}_uq"

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "LightGBMClassifierUQ":
        """Fit LightGBM model with Platt Scaling calibration.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features for early stopping AND calibration (required!)
            y_val: Validation labels for calibration (required!)

        Returns:
            self: Fitted LightGBMClassifierUQ instance
        """
        # FIT BASE LIGHTGBM MODEL
        self.base_model.fit(X_train, y_train, X_val, y_val)
        
        # APPLY PLATT SCALING CALIBRATION ON VALIDATION SET
        if X_val is not None and y_val is not None:
            print("  [LightGBM UQ] Applying Platt Scaling calibration on validation set...")
            
            # CalibratedClassifierCV with cv='prefit' uses the already-trained model
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model.model,
                method='sigmoid',  # Platt Scaling (logistic regression on predictions)
                cv='prefit'        # Model already trained, just calibrate
            )
            
            # FIT CALIBRATION ON VALIDATION SET LEARNS SIGMOID PARAMETERS
            X_val_df = self.base_model._to_dataframe(X_val)
            self.calibrated_model.fit(X_val_df, y_val)
            
            print("  [LightGBM UQ] Calibration complete (ECE should improve)")
        else:
            print("  [LightGBM UQ] WARNING: No validation set provided, skipping calibration!")
            print("               ECE may be high (0.3-0.7). Provide X_val, y_val to fix.")
            self.calibrated_model = None
        
        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty decomposition using CALIBRATED probabilities.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            predictions: Class predictions, shape (n_samples,)
            aleatoric: Aleatoric uncertainty per sample, shape (n_samples,)
            epistemic: Epistemic uncertainty per sample, shape (n_samples,)
        """
        X_df = self.base_model._to_dataframe(X)

        # ALEATORIC USE CALIBRATED PROBABILITIES IF AVAILABLE
        if self.calibrated_model is not None:
            mean_probs = self.calibrated_model.predict_proba(X_df)
        else:
            # FALLBACK TO UNCALIBRATED PROBABILITIES IF NO VALIDATION SET WAS PROVIDED
            mean_probs = self.base_model.model.predict_proba(X_df)
        
        predictions = np.argmax(mean_probs, axis=1)

        # ALEATORIC UNCERTAINTY PREDICTIVE ENTROPY OF CALIBRATED PROBABILITIES
        # Formula: H[E[p]] = -sum(mean_p * log(mean_p))
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

        # EPISTEMIC VARIANCE ACROSS BOOSTING TREES UNCHANGED BY CALIBRATION
        epistemic = self._compute_tree_variance(X_df)

        return predictions, aleatoric, epistemic

    def _compute_tree_variance(self, X_df) -> np.ndarray:
        """Compute epistemic uncertainty from tree-level variance.
        
        NOTE: This is INDEPENDENT of calibration (operates on tree structure).
        
        Args:
            X_df: Input features as DataFrame
            
        Returns:
            Epistemic uncertainty per sample, shape (n_samples,)
        """
        n_classes = self.base_model.model.n_classes_
        booster = self.base_model.model.booster_
        n_trees = booster.num_trees()

        if n_classes == 2:
            # BINARY CLASSIFICATION 1 TREE PER ITERATION
            tree_raw_scores = []
            for i in range(n_trees):
                score = self.base_model.model.predict(
                    X_df,
                    raw_score=True,
                    start_iteration=i,
                    num_iteration=1
                )
                tree_raw_scores.append(score)

            # STACK TO GET SHAPE N TREES N SAMPLES
            tree_raw_scores = np.array(tree_raw_scores)

            # CONVERT EACH TREE S RAW SCORE TO PROBABILITY
            tree_probs_class1 = 1.0 / (1.0 + np.exp(-tree_raw_scores))
            tree_probs_class0 = 1.0 - tree_probs_class1

            # SHAPE N TREES N SAMPLES 2
            tree_probs = np.stack([tree_probs_class0, tree_probs_class1], axis=2)

        else:
            # MULTI CLASS N CLASSES TREES PER ITERATION
            n_iterations = n_trees // n_classes

            tree_raw_scores = []
            for iter_idx in range(n_iterations):
                iter_scores = []
                for class_idx in range(n_classes):
                    tree_idx = iter_idx * n_classes + class_idx
                    score = self.base_model.model.predict(
                        X_df,
                        raw_score=True,
                        start_iteration=tree_idx,
                        num_iteration=1
                    )
                    if score.ndim == 2:
                        score = score[:, class_idx]
                    iter_scores.append(score)

                iter_scores = np.stack(iter_scores, axis=1)
                tree_raw_scores.append(iter_scores)

            # SHAPE N ITERATIONS N SAMPLES N CLASSES
            tree_raw_scores = np.array(tree_raw_scores)

            # CONVERT TO PROBABILITIES USING SOFTMAX PER TREE ITERATION
            tree_probs = softmax(tree_raw_scores, axis=2)

        # EPISTEMIC UNCERTAINTY VARIANCE OF PREDICTIONS ACROSS TREES
        # Formula: mean(Var[p_tree(y|x)]) - variance across trees, averaged over classes
        epistemic = np.mean(np.var(tree_probs, axis=0), axis=1)

        return epistemic

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict CALIBRATED class probabilities.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities (calibrated if available), shape (n_samples, n_classes)
        """
        X_df = self.base_model._to_dataframe(X)
        
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_df)
        else:
            return self.base_model.model.predict_proba(X_df)