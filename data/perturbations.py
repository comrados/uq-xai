from typing import Literal
import numpy as np

from config.settings import GLOBAL_SEED


class PerturbationGenerator:
    """Generates perturbed versions of data for robustness testing.

    Provides various data perturbation methods including Gaussian noise,
    random masking with imputation, and feature permutation. All methods
    use a seeded random number generator for reproducibility.

    Attributes:
        seed: Random seed for reproducibility.
        rng: NumPy random state generator.
    """

    def __init__(self, seed: int = GLOBAL_SEED):
        """Initialize perturbation generator.

        Args:
            seed: Random seed for reproducibility. Default from config.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def gaussian(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """Add Gaussian noise scaled by feature standard deviation.

        Generates random noise from a standard normal distribution,
        then scales it by each feature's standard deviation and the
        sigma parameter.

        Args:
            X: Input features, shape (n_samples, n_features).
            sigma: Noise level multiplier. Higher values mean more noise.

        Returns:
            Perturbed features, shape (n_samples, n_features).
        """
        noise = self.rng.randn(*X.shape) * X.std(axis=0) * sigma
        return X + noise

    def missing(self, X: np.ndarray, rate: float) -> np.ndarray:
        """Randomly mask features and impute with median values.

        Simulates missing data by randomly masking a fraction of feature
        values and replacing them with the feature-wise median from the
        original data.

        Args:
            X: Input features, shape (n_samples, n_features).
            rate: Fraction of values to mask, between 0 and 1.

        Returns:
            Perturbed features with masked values imputed, shape (n_samples, n_features).
        """
        X_perturbed = X.copy()
        mask = self.rng.rand(*X.shape) < rate
        
        # IMPUTE WITH MEDIAN PER FEATURE
        medians = np.median(X, axis=0)
        X_perturbed[mask] = np.take(medians, np.where(mask)[1])
        
        return X_perturbed
    
    def permute(self, X: np.ndarray, fraction: float) -> np.ndarray:
        """Shuffle a fraction of features across samples.

        Randomly selects features and independently shuffles their values
        across samples, breaking feature-label relationships while preserving
        marginal distributions.

        Args:
            X: Input features, shape (n_samples, n_features).
            fraction: Fraction of features to permute, between 0 and 1.

        Returns:
            Perturbed features with some columns shuffled, shape (n_samples, n_features).
        """
        X_perturbed = X.copy()
        n_features = X.shape[1]
        n_permute = max(1, int(n_features * fraction))

        feature_indices = self.rng.choice(n_features, size=n_permute, replace=False)

        for idx in feature_indices:
            X_perturbed[:, idx] = self.rng.permutation(X_perturbed[:, idx])

        return X_perturbed

    def perturb(self,
                X: np.ndarray,
                method: Literal["gaussian", "missing", "permute"],
                level: float) -> np.ndarray:
        """Apply perturbation using specified method.

        Dispatcher method that routes to the appropriate perturbation
        function based on the method parameter.

        Args:
            X: Input features, shape (n_samples, n_features).
            method: Perturbation method to apply.
            level: Perturbation intensity parameter. Meaning depends on method:
                - gaussian: noise standard deviation multiplier
                - missing: fraction of values to mask
                - permute: fraction of features to shuffle

        Returns:
            Perturbed features, shape (n_samples, n_features).

        Raises:
            ValueError: If method is not recognized.
        """
        if method == "gaussian":
            return self.gaussian(X, level)
        elif method == "missing":
            return self.missing(X, level)
        elif method == "permute":
            return self.permute(X, level)
        else:
            raise ValueError(f"Unknown perturbation method: {method}")

    @staticmethod
    def make_key(method: str, level: float) -> str:
        """Generate cache key for a perturbation configuration.

        Args:
            method: Perturbation method name.
            level: Perturbation level parameter.

        Returns:
            Cache key in format: {method}_{level}.
        """
        return f"{method}_{level}"