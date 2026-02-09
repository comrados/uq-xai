from typing import Optional, List
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from explainers.base import BaseExplainer
from models.base import BaseModel


class LIMEExplainer(BaseExplainer):
    """LIME explainer for tabular data (model-agnostic).

    Local Interpretable Model-agnostic Explanations (LIME) generates
    local explanations by fitting linear models around individual predictions.

    Works for all model types (Logistic, RF, MLP, LightGBM).

    Attributes:
        model: Fitted BaseModel instance to explain
        explainer: Underlying LimeTabularExplainer instance
        num_features: Number of top features to include in explanations
    """

    def __init__(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_features: int = 10,
    ):
        """Initialize LIME explainer.

        Args:
            model: Fitted BaseModel instance to explain
            X_train: Training data for LIME to sample from, shape (n_samples, n_features)
            feature_names: List of feature names. If None, uses "f0", "f1", etc.
            num_features: Number of top features to include in explanations. Default: 10.
        """
        super().__init__(model)
        self.num_features = num_features

        # GENERATE FEATURE NAMES IF NOT PROVIDED
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        # DETERMINE MODE BASED ON TASK TYPE
        mode = "classification" if model.task_type == "classification" else "regression"

        # INITIALIZE LIME EXPLAINER
        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Generate LIME feature attributions.

        For each sample, generates local linear explanation and returns
        feature weights. Features not in top-k have weight 0.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Feature attributions, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        attributions = np.zeros((n_samples, n_features))

        # GET PREDICTION FUNCTION BASED ON TASK TYPE
        if self.model.task_type == "classification":
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        # GENERATE EXPLANATION FOR EACH SAMPLE
        for i in range(n_samples):
            # LIME EXPLAINS INDIVIDUAL INSTANCE
            exp = self.explainer.explain_instance(
                data_row=X[i],
                predict_fn=predict_fn,
                num_features=self.num_features,
            )

            # EXTRACT FEATURE WEIGHTS FROM EXPLANATION
            # exp.as_list() returns [(feature_name, weight), ...]
            # exp.local_exp[label] returns [(feature_idx, weight), ...] for classification
            # FOR REGRESSION LABEL IS 0
            if self.model.task_type == "classification":
                # FOR CLASSIFICATION LIME EXPLAINS ONE CLASS USUALLY PREDICTED CLASS
                # GET THE FIRST AND TYPICALLY ONLY KEY IN LOCAL EXP
                explained_class = list(exp.local_exp.keys())[0]
                feature_weights = exp.local_exp[explained_class]
            else:
                # FOR REGRESSION LABEL IS ALWAYS 0
                feature_weights = exp.local_exp[0]

            # FILL IN ATTRIBUTIONS FOR TOP K FEATURES
            for feature_idx, weight in feature_weights:
                attributions[i, feature_idx] = weight

        return attributions
