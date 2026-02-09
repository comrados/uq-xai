from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from config.settings import GLOBAL_SEED, UQ_CONFIG
from uncertainty.base_uq import UQWrapper
from models.linear import LogisticModel


class LogisticUQ(UQWrapper):
    """Logistic regression with bootstrap uncertainty estimation.

    Uses bootstrap ensemble (20 models) to estimate epistemic uncertainty.
    Each bootstrap model is trained on a resampled version of the training data.

    Uncertainty decomposition:
    - Aleatoric: Predictive entropy of mean probabilities
    - Epistemic: Variance across bootstrap model predictions
    """
    
    def __init__(self,
                 base_model: Optional[LogisticModel] = None,
                 n_bootstrap: int = UQ_CONFIG['bootstrap_n_models'],
                 seed: int = GLOBAL_SEED):
        """Initialize LogisticUQ wrapper.

        Args:
            base_model: Pre-trained LogisticModel. If None, creates new model.
            n_bootstrap: Number of bootstrap models to train. Default: 20.
            seed: Random seed for reproducibility.
        """
        if base_model is None:
            base_model = LogisticModel(seed=seed)
        super().__init__(base_model)
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.bootstrap_models = []
    
    @property
    def name(self) -> str:
        return f"{self.base_model.name}_uq_b{self.n_bootstrap}"
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "LogisticUQ":
        """Fit base model and bootstrap ensemble.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (unused for logistic)
            y_val: Validation labels (unused for logistic)

        Returns:
            self: Fitted LogisticUQ instance
        """
        # FIT BASE MODEL
        self.base_model.fit(X_train, y_train)

        # FIT BOOTSTRAP ENSEMBLE FOR EPISTEMIC UNCERTAINTY ESTIMATION
        # Bootstrap = sampling with replacement, same size as original data
        # PROVIDES PROPER ESTIMATION OF MODEL VARIABILITY
        self.bootstrap_models = []
        rng = np.random.RandomState(self.seed)

        for i in range(self.n_bootstrap):
            # sklearn.utils.resample: bootstrap (n_samples=len(X), replace=True)
            X_boot, y_boot = resample(X_train, y_train, random_state=rng.randint(0, 10000))
            model = LogisticRegression(C=self.base_model.C, random_state=self.seed, max_iter=1000, n_jobs=-1)
            model.fit(X_boot, y_boot)
            self.bootstrap_models.append(model)

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
        base_classes = self.base_model.model.classes_

        def aligned_proba(model):
            """Align model probabilities to base class ordering."""
            probs = model.predict_proba(X)
            model_classes = model.classes_
            if np.array_equal(model_classes, base_classes):
                return probs

            aligned = np.zeros((probs.shape[0], len(base_classes)))
            class_to_idx = {c: i for i, c in enumerate(base_classes)}
            for col_idx, cls in enumerate(model_classes):
                aligned[:, class_to_idx[cls]] = probs[:, col_idx]
            return aligned

        # GET PROBABILITIES FROM ALL BOOTSTRAP MODELS SHAPE N BOOTSTRAP N SAMPLES N CLASSES
        bootstrap_probs = np.array([aligned_proba(m) for m in self.bootstrap_models])

        # MEAN PROBABILITIES ACROSS BOOTSTRAP MODELS SHAPE N SAMPLES N CLASSES
        mean_probs = np.mean(bootstrap_probs, axis=0)
        predictions = np.argmax(mean_probs, axis=1)

        # ALEATORIC UNCERTAINTY PREDICTIVE ENTROPY OF MEAN PROBABILITIES
        # Formula: H[E[p]] = -sum(mean_p * log(mean_p))
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

        # EPISTEMIC UNCERTAINTY VARIANCE OF PREDICTIONS ACROSS BOOTSTRAP MODELS
        # Formula: mean(Var[p(y|x, theta_i)]) where theta_i are bootstrap models
        epistemic = np.mean(np.var(bootstrap_probs, axis=0), axis=1)

        return predictions, aleatoric, epistemic
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (mean across bootstrap models).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        base_classes = self.base_model.model.classes_

        def aligned_proba(model):
            probs = model.predict_proba(X)
            model_classes = model.classes_
            if np.array_equal(model_classes, base_classes):
                return probs

            aligned = np.zeros((probs.shape[0], len(base_classes)))
            class_to_idx = {c: i for i, c in enumerate(base_classes)}
            for col_idx, cls in enumerate(model_classes):
                aligned[:, class_to_idx[cls]] = probs[:, col_idx]
            return aligned

        bootstrap_probs = np.array([aligned_proba(m) for m in self.bootstrap_models])
        return np.mean(bootstrap_probs, axis=0)
