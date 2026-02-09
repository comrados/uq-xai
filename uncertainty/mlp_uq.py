from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from config.settings import GLOBAL_SEED, UQ_CONFIG
from uncertainty.base_uq import UQWrapper
from models.mlp import MLPClassifier, get_device


class BaseMLPUQ(UQWrapper):
    """Base MLP UQ with shared MC Dropout logic."""

    def __init__(self,
                 base_model,
                 n_mc_samples: int = UQ_CONFIG['mc_dropout_n_samples'],
                 seed: int = GLOBAL_SEED):
        """Initialize BaseMLPUQ wrapper.

        Args:
            base_model: Pre-trained MLP model.
            n_mc_samples: Number of MC Dropout samples. Default: 50.
            seed: Random seed for reproducibility.
        """
        super().__init__(base_model)
        self.n_mc_samples = n_mc_samples
        self.seed = seed
        self.device = get_device()

        torch.manual_seed(seed)

    @property
    def name(self) -> str:
        return f"{self.base_model.name}_uq_mc{self.n_mc_samples}"

    def _enable_dropout(self, model: nn.Module):
        """Enable dropout and batchnorm at inference time for MC Dropout.

        MC Dropout approximates Bayesian inference by treating dropout as
        variational inference. Enabling BatchNorm in train mode adds additional
        stochasticity for better epistemic uncertainty estimation.
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
            # ENABLE BATCHNORM IN TRAINING MODE TO USE BATCH STATISTICS INSTEAD OF RUNNING STATS
            # THIS INCREASES STOCHASTICITY FOR BETTER EPISTEMIC UNCERTAINTY ESTIMATION
            if isinstance(module, nn.BatchNorm1d):
                module.train()

    def _mc_forward(self, model: nn.Module, X_tensor: torch.Tensor, get_output) -> np.ndarray:
        """Run MC Dropout forward passes.

        Args:
            model: PyTorch model with dropout layers.
            X_tensor: Input tensor, shape (n_samples, n_features).
            get_output: Function to extract output from model.

        Returns:
            MC samples, shape (n_mc_samples, n_samples, ...)
        """
        model.eval()
        self._enable_dropout(model)

        mc_outputs = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                output = get_output(model, X_tensor)
                mc_outputs.append(output)

        return np.array(mc_outputs)


class MLPClassifierUQ(BaseMLPUQ):
    """MLP classification with MC Dropout uncertainty estimation.

    Uses Monte Carlo Dropout (50 stochastic forward passes) for UQ.
    Dropout and BatchNorm are enabled at inference time for stochasticity.

    Uncertainty decomposition:
    - Aleatoric: Predictive entropy of mean probabilities
    - Epistemic: Mutual Information (MI) = H[E[p]] - E[H[p]]
    """

    def __init__(self,
                 base_model: Optional[MLPClassifier] = None,
                 n_mc_samples: int = UQ_CONFIG['mc_dropout_n_samples'],
                 seed: int = GLOBAL_SEED):
        """Initialize MLPClassifierUQ wrapper.

        Args:
            base_model: Pre-trained MLPClassifier. If None, creates new model.
            n_mc_samples: Number of MC Dropout samples. Default: 50.
            seed: Random seed for reproducibility.
        """
        if base_model is None:
            base_model = MLPClassifier(seed=seed)
        super().__init__(base_model, n_mc_samples, seed)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> "MLPClassifierUQ":
        """Fit MLP model with early stopping.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features for early stopping
            y_val: Validation labels for early stopping

        Returns:
            self: Fitted MLPClassifierUQ instance
        """
        self.base_model.fit(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty decomposition using MC Dropout.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            predictions: Class predictions, shape (n_samples,)
            aleatoric: Aleatoric uncertainty per sample, shape (n_samples,)
            epistemic: Epistemic uncertainty per sample, shape (n_samples,)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)

        # MC DROPOUT RUN MULTIPLE STOCHASTIC FORWARD PASSES
        # EACH PASS USES DIFFERENT DROPOUT MASK BATCH STATISTICS BATCHNORM IN TRAIN MODE
        # SHAPE N MC SAMPLES N SAMPLES N CLASSES
        mc_probs = self._mc_forward(
            self.base_model.model, X_tensor,
            lambda m, x: torch.softmax(m(x), dim=1).cpu().numpy()
        )

        # MEAN PROBABILITIES ACROSS MC SAMPLES SHAPE N SAMPLES N CLASSES
        mean_probs = np.mean(mc_probs, axis=0)
        predictions = np.argmax(mean_probs, axis=1)

        # ALEATORIC UNCERTAINTY PREDICTIVE ENTROPY OF MEAN PROBABILITIES
        # Formula: H[E[p]] = -sum(mean_p * log(mean_p))
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

        # EPISTEMIC UNCERTAINTY MUTUAL INFORMATION MI
        # Formula: MI = H[E[p]] - E[H[p]]
        # = (entropy of mean) - (mean of entropies)
        # CAPTURES MODEL UNCERTAINTY DUE TO DROPOUT STOCHASTICITY
        expected_entropy = np.mean([-np.sum(p * np.log(p + 1e-8), axis=1) for p in mc_probs], axis=0)
        epistemic = np.maximum(aleatoric - expected_entropy, 0)

        return predictions, aleatoric, epistemic

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (mean across MC samples).

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        mc_probs = self._mc_forward(
            self.base_model.model, X_tensor,
            lambda m, x: torch.softmax(m(x), dim=1).cpu().numpy()
        )
        return np.mean(mc_probs, axis=0)
