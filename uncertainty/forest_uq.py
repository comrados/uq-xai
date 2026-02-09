from typing import Tuple, Optional
import numpy as np

from config.settings import GLOBAL_SEED
from uncertainty.base_uq import UQWrapper
from models.forest import RandomForestClassifierModel


class RandomForestClassifierUQ(UQWrapper):
    """Random Forest classification with tree-based uncertainty estimation.

    Leverages the internal ensemble structure of Random Forest for UQ.
    Each tree is trained on a bootstrap sample (Random Forest default).

    Uncertainty decomposition:
    - Aleatoric: Predictive entropy of mean tree probabilities
    - Epistemic: Variance of predictions across trees
    """
    
    def __init__(self,
                 base_model: Optional[RandomForestClassifierModel] = None,
                 seed: int = GLOBAL_SEED):
        """Initialize RandomForestClassifierUQ wrapper.

        Args:
            base_model: Pre-trained RandomForestClassifierModel. If None, creates new model.
            seed: Random seed for reproducibility.
        """
        if base_model is None:
            base_model = RandomForestClassifierModel(seed=seed)
        super().__init__(base_model)
        self.seed = seed
    
    @property
    def name(self) -> str:
        return f"{self.base_model.name}_uq"
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "RandomForestClassifierUQ":
        """Fit Random Forest model (uses internal bootstrap for epistemic UQ).

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (unused for RF)
            y_val: Validation labels (unused for RF)

        Returns:
            self: Fitted RandomForestClassifierUQ instance
        """
        # Random Forest already uses bootstrap internally (default: bootstrap=True)
        # EACH TREE IS TRAINED ON A BOOTSTRAP SAMPLE OF THE DATA
        # NO ADDITIONAL BOOTSTRAPPING NEEDED FOR EPISTEMIC UNCERTAINTY
        self.base_model.fit(X_train, y_train)
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
        # GET PROBABILITIES FROM EACH TREE SHAPE N TREES N SAMPLES N CLASSES
        # Each tree trained on bootstrap sample (Random Forest default: bootstrap=True)
        tree_probs = np.array([tree.predict_proba(X) for tree in self.base_model.model.estimators_])

        # MEAN PROBABILITIES ACROSS ALL TREES SHAPE N SAMPLES N CLASSES
        mean_probs = np.mean(tree_probs, axis=0)
        predictions = np.argmax(mean_probs, axis=1)

        # ALEATORIC UNCERTAINTY PREDICTIVE ENTROPY OF MEAN PROBABILITIES
        # Formula: H[E[p]] = -sum(mean_p * log(mean_p))
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

        # EPISTEMIC UNCERTAINTY VARIANCE OF PREDICTIONS ACROSS TREES
        # Formula: mean(Var[p_tree(y|x)]) - captures tree diversity due to bootstrap
        epistemic = np.mean(np.var(tree_probs, axis=0), axis=1)

        return predictions, aleatoric, epistemic
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (mean across trees).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        tree_probs = np.array([tree.predict_proba(X) for tree in self.base_model.model.estimators_])
        return np.mean(tree_probs, axis=0)