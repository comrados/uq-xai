from typing import Optional
import numpy as np
import shap

from explainers.base import BaseExplainer
from models.base import BaseModel
from models.forest import RandomForestClassifierModel
from models.lgbm import LightGBMClassifierModel
from models.catboost import CatBoostClassifierModel


class SHAPExplainer(BaseExplainer):
    """SHAP explainer with automatic model type detection.

    Uses TreeExplainer for tree-based models (fast, exact) and
    KernelExplainer for other models (slower, model-agnostic).

    For classification, returns SHAP values for the predicted class only.

    Attributes:
        model: Fitted BaseModel instance to explain
        X_background: Background data for Kernel SHAP, shape (n_background, n_features)
        explainer: Underlying SHAP explainer instance (TreeExplainer or KernelExplainer)
    """

    def __init__(self, model: BaseModel, X_background: Optional[np.ndarray] = None):
        """Initialize SHAP explainer.

        Args:
            model: Fitted BaseModel instance to explain
            X_background: Background data for Kernel SHAP, shape (n_background, n_features).
                         Required for non-tree models. Default: None.
                         Recommended size: 100 samples for speed.
        """
        super().__init__(model)
        self.X_background = X_background
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        # TREEEXPLAINER FOR TREE BASED MODELS FAST EXACT
        if isinstance(self.model, (RandomForestClassifierModel, LightGBMClassifierModel, CatBoostClassifierModel)):
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            # KERNELEXPLAINER FOR OTHER MODELS SLOWER MODEL AGNOSTIC
            # REQUIRES BACKGROUND DATA FOR SAMPLING
            if self.X_background is None:
                raise ValueError(
                    "X_background required for non-tree models (Logistic, MLP). "
                    "Provide ~100 samples from training data."
                )

            # USE PREDICT PROBA FOR CLASSIFICATION RETURNS PROBABILITIES
            if self.model.task_type == "classification":
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            # SUBSAMPLE BACKGROUND DATA IF TOO LARGE KERNELSHAP IS SLOW
            background_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
            self.explainer = shap.KernelExplainer(predict_fn, background_sample, silent=True)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Generate SHAP feature attributions.

        For classification, returns SHAP values for the predicted class only.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            SHAP values, shape (n_samples, n_features)
        """
        # FOR KERNELEXPLAINER SUPPRESS PROGRESS BAR
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(X, silent=True)
        else:
            shap_values = self.explainer.shap_values(X)

        # FOR CLASSIFICATION SHAP RETURNS VALUES PER CLASS
        # EXTRACT VALUES FOR PREDICTED CLASS ONLY
        if self.model.task_type == "classification":
            # TREEEXPLAINER RETURNS LIST OF ARRAYS ONE PER CLASS
            # KERNELEXPLAINER RETURNS ARRAY OF SHAPE N SAMPLES N FEATURES N CLASSES
            if isinstance(shap_values, list):
                # TreeExplainer: list of arrays, shape [(n_samples, n_features)] * n_classes
                predictions = self.model.predict(X)
                # FOR EACH SAMPLE GET SHAP VALUES FOR ITS PREDICTED CLASS
                shap_for_pred_class = np.array([
                    shap_values[int(pred_class)][i]
                    for i, pred_class in enumerate(predictions)
                ])
                return shap_for_pred_class
            elif shap_values.ndim == 2:
                # CATBOOST TREEEXPLAINER FOR BINARY CLASSIFICATION SHAPE N SAMPLES N FEATURES
                # RETURNS SHAP VALUES FOR POSITIVE CLASS ONLY
                return shap_values
            else:
                # KERNELEXPLAINER SHAPE N SAMPLES N FEATURES N CLASSES
                predictions = self.model.predict(X)
                # FOR EACH SAMPLE GET SHAP VALUES FOR ITS PREDICTED CLASS
                shap_for_pred_class = np.array([
                    shap_values[i, :, int(pred_class)]
                    for i, pred_class in enumerate(predictions)
                ])
                return shap_for_pred_class
        else:
            # REGRESSION SHAPE N SAMPLES N FEATURES
            return shap_values
