import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel

from explainers.base import BaseExplainer
from models.base import BaseModel
from models.mlp import MLPClassifier


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer for gradient-based models (MLP only).

    Computes feature attributions by integrating gradients along a path
    from a baseline (zero vector) to the input.

    For classification, explains the predicted class logit.

    Attributes:
        model: Fitted MLP model instance
        ig: Captum IntegratedGradients instance
        n_steps: Number of integration steps
    """

    def __init__(self, model: BaseModel, n_steps: int = 50):
        """Initialize Integrated Gradients explainer.

        Args:
            model: Fitted BaseModel instance (must support gradients)
            n_steps: Number of steps for Riemann approximation. Default: 50.

        Raises:
            AssertionError: If model does not support gradients
        """
        assert model.supports_gradients, (
            "IntegratedGradients requires gradient-based model (MLP only). "
            f"Got {model.__class__.__name__} with supports_gradients={model.supports_gradients}"
        )
        super().__init__(model)
        self.n_steps = n_steps

        # CAPTUM INTEGRATEDGRADIENTS REQUIRES NN MODULE
        # FOR MLPCLASSIFIER USE MODEL MODEL THE MLPNETWORK
        if isinstance(model, MLPClassifier):
            self.ig = IntegratedGradients(model.model)
        else:
            # GENERIC CASE FOR FUTURE GRADIENT BASED MODELS
            self.ig = IntegratedGradients(model.model)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Generate Integrated Gradients attributions.

        For classification, computes gradients with respect to predicted class logit.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Feature attributions, shape (n_samples, n_features)
        """
        # CONVERT TO TENSOR
        X_tensor = torch.FloatTensor(X).to(self.model.device)
        X_tensor.requires_grad = True

        # BASELINE ZERO VECTOR FEATURE WISE
        baseline = torch.zeros_like(X_tensor)

        # FOR CLASSIFICATION TARGET IS PREDICTED CLASS
        if self.model.task_type == "classification":
            # GET PREDICTED CLASSES
            self.model.model.eval()
            with torch.no_grad():
                logits = self.model.model(X_tensor)
                predicted_classes = torch.argmax(logits, dim=1)

            # COMPUTE IG FOR PREDICTED CLASS LOGIT
            attributions = self.ig.attribute(
                X_tensor,
                baselines=baseline,
                target=predicted_classes,
                n_steps=self.n_steps
            )
        else:
            # Regression: target=None (single output)
            attributions = self.ig.attribute(
                X_tensor,
                baselines=baseline,
                target=None,
                n_steps=self.n_steps
            )

        # CONVERT BACK TO NUMPY
        return attributions.detach().cpu().numpy()


class SmoothGradExplainer(BaseExplainer):
    """SmoothGrad explainer for gradient-based models (MLP only).

    Averages gradients over multiple noisy copies of the input to reduce
    noise and improve stability compared to vanilla gradients.

    Attributes:
        model: Fitted MLP model instance
        noise_tunnel: Captum NoiseTunnel wrapper for gradient smoothing
        n_samples: Number of noisy samples for averaging
        stdevs: Standard deviation of Gaussian noise (as fraction of input range)
    """

    def __init__(
        self,
        model: BaseModel,
        n_samples: int = 50,
        stdevs: float = 0.1
    ):
        """Initialize SmoothGrad explainer.

        Args:
            model: Fitted BaseModel instance (must support gradients)
            n_samples: Number of noisy samples for gradient averaging. Default: 50.
            stdevs: Standard deviation of Gaussian noise (fraction). Default: 0.1.

        Raises:
            AssertionError: If model does not support gradients
        """
        assert model.supports_gradients, (
            "SmoothGrad requires gradient-based model (MLP only). "
            f"Got {model.__class__.__name__} with supports_gradients={model.supports_gradients}"
        )
        super().__init__(model)
        self.n_samples = n_samples
        self.stdevs = stdevs

        # CAPTUM NOISETUNNEL WRAPS GRADIENT BASED ATTRIBUTION METHODS
        # USE INTEGRATEDGRADIENTS AS BASE METHOD CAN ALSO USE SALIENCY
        if isinstance(model, MLPClassifier):
            base_ig = IntegratedGradients(model.model)
        else:
            base_ig = IntegratedGradients(model.model)

        self.noise_tunnel = NoiseTunnel(base_ig)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Generate SmoothGrad attributions.

        Averages gradients over noisy copies of input for stability.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Feature attributions, shape (n_samples, n_features)
        """
        # CONVERT TO TENSOR
        X_tensor = torch.FloatTensor(X).to(self.model.device)
        X_tensor.requires_grad = True

        # BASELINE FOR INTEGRATEDGRADIENTS USED INTERNALLY
        baseline = torch.zeros_like(X_tensor)

        # FOR CLASSIFICATION TARGET IS PREDICTED CLASS
        if self.model.task_type == "classification":
            self.model.model.eval()
            with torch.no_grad():
                logits = self.model.model(X_tensor)
                predicted_classes = torch.argmax(logits, dim=1)

            # SMOOTHGRAD WITH NOISE TUNNEL
            attributions = self.noise_tunnel.attribute(
                X_tensor,
                baselines=baseline,
                target=predicted_classes,
                nt_type='smoothgrad',
                nt_samples=self.n_samples,
                stdevs=self.stdevs
            )
        else:
            # Regression: target=None
            attributions = self.noise_tunnel.attribute(
                X_tensor,
                baselines=baseline,
                target=None,
                nt_type='smoothgrad',
                nt_samples=self.n_samples,
                stdevs=self.stdevs
            )

        return attributions.detach().cpu().numpy()
