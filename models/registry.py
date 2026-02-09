from pathlib import Path
import pickle
from typing import Optional
import numpy as np

from config.settings import MODELS_CACHE_DIR
from models.base import BaseModel


class ModelRegistry:
    """Persistent cache for trained models using pickle serialization.

    Provides a simple interface for saving and loading trained models to disk,
    avoiding redundant training. Models are stored as pickle files with keys
    that typically encode dataset and model configuration information.

    Attributes:
        cache_dir: Directory path where model files are stored.
    """

    def __init__(self, cache_dir: Path = MODELS_CACHE_DIR):
        """Initialize model registry with cache directory.

        Args:
            cache_dir: Path to directory for storing cached models.
                Default from config.
        """
        self.cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        """Generate file path for a cache key.

        Args:
            key: Cache key identifying the model.

        Returns:
            Path to the pickle file for this key.
        """
        return self.cache_dir / f"{key}.pkl"

    def save(self, key: str, model: BaseModel) -> None:
        """Save a trained model to cache.

        Args:
            key: Unique identifier for the model.
            model: Trained model instance to save.
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def load(self, key: str) -> BaseModel:
        """Load a trained model from cache.

        Args:
            key: Unique identifier for the model.

        Returns:
            The loaded model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        with open(self._path(key), "rb") as f:
            return pickle.load(f)

    def exists(self, key: str) -> bool:
        """Check if a model exists in cache.

        Args:
            key: Unique identifier for the model.

        Returns:
            True if the model is cached, False otherwise.
        """
        return self._path(key).exists()

    def get_or_train(self,
                     key: str,
                     model: BaseModel,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None) -> BaseModel:
        """Load model from cache or train and cache it.

        Convenience method that checks cache first, and only trains if
        the model is not found. Newly trained models are automatically saved.

        Args:
            key: Unique identifier for the model.
            model: Untrained model instance to train if not cached.
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features, shape (n_val_samples, n_features).
            y_val: Validation labels, shape (n_val_samples,).

        Returns:
            Trained model (either loaded from cache or freshly trained).
        """
        if self.exists(key):
            return self.load(key)

        model.fit(X_train, y_train, X_val, y_val)
        self.save(key, model)
        return model

    @staticmethod
    def make_key(dataset_name: str, model_name: str) -> str:
        """Generate a cache key from dataset and model names.

        Args:
            dataset_name: Name of the dataset.
            model_name: Name or configuration string of the model.

        Returns:
            Cache key in format: {dataset_name}/{model_name}.
        """
        return f"{dataset_name}/{model_name}"