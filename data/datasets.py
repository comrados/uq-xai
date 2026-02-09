from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    WINE_UCI_ID,
    COVERTYPE_UCI_ID,
    BEAN_UCI_ID,
    IRIS_UCI_ID,
    RICE_UCI_ID,
    ECOLI_UCI_ID,
)


@dataclass
class DatasetInfo:
    """Container for loaded dataset with metadata.

    Encapsulates all information about a dataset including features, labels,
    metadata, and optional perturbation information.

    Attributes:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target labels or values, shape (n_samples,).
        feature_names: List of feature names.
        task_type: Type of machine learning task.
        name: Dataset identifier.
        perturbation: Description of applied perturbation, e.g., "gaussian_0.3",
            "missing_0.1". None indicates clean data.
        class_names: Array of class names for classification tasks. None for
            regression tasks.
    """
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    task_type: Literal["regression", "classification"]
    name: str
    perturbation: Optional[str] = None  # None = clean, "gaussian_0.3", "missing_0.1", etc.
    class_names: Optional[np.ndarray] = None


class BaseDataset(ABC):
    """Abstract base class for all dataset loaders.

    Defines the interface that all dataset implementations must follow.
    Subclasses must implement methods for loading data and generating
    cache keys.
    """

    @abstractmethod
    def load(self) -> DatasetInfo:
        """Load the dataset and return as DatasetInfo.

        Returns:
            DatasetInfo containing features, labels, and metadata.
        """
        pass

    @property
    @abstractmethod
    def cache_key(self) -> str:
        """Generate a unique cache key for this dataset.

        Returns:
            String identifier for caching purposes.
        """
        pass


class UCIDataset(BaseDataset):
    """Generic loader for datasets from the UCI Machine Learning Repository.

    Fetches datasets using the ucimlrepo package and converts them to the
    standard DatasetInfo format. Handles label encoding for classification
    tasks automatically.

    Attributes:
        uci_id: UCI repository dataset ID.
        name: Human-readable dataset name.
        task_type: Type of machine learning task.
    """

    def __init__(self,
                 uci_id: int,
                 name: str,
                 task_type: Literal["regression", "classification"]):
        """Initialize UCI dataset loader.

        Args:
            uci_id: The UCI repository ID for the dataset.
            name: Human-readable name for the dataset.
            task_type: The task type (classification or regression).
        """
        self.uci_id = uci_id
        self.name = name
        self.task_type = task_type

    @property
    def cache_key(self) -> str:
        """Generate cache key for this dataset.

        Returns:
            Cache key in format: {name}_{uci_id}/raw.
        """
        return f"{self.name}_{self.uci_id}/raw"

    def load(self) -> DatasetInfo:
        """Load dataset from UCI repository.

        Fetches the dataset, extracts features and targets, and performs
        label encoding for classification tasks.

        Returns:
            DatasetInfo with features, labels, and metadata.
        """
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=self.uci_id)

        X = data.data.features.values
        y_raw = data.data.targets.values.ravel()

        class_names = None

        if self.task_type == "classification":
            y_raw = np.asarray(y_raw)

            # LabelEncoder -> ints
            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(np.int64)
            class_names = le.classes_

        else:  # regression
            y = np.asarray(y_raw, dtype=float)

        return DatasetInfo(
            X=X,
            y=y,
            feature_names=data.data.features.columns.tolist(),
            task_type=self.task_type,
            name=self.name,
            perturbation=None,
            class_names=class_names,
        )

class WineQualityDataset(UCIDataset):
    """Wine quality dataset as binary classification task.

    Converts the original multi-class wine quality ratings into a binary
    classification problem using a quality threshold. Ratings at or above
    the threshold are labeled as "good", below as "bad".

    Default threshold=6 gives a balanced split:
    - bad (quality 3-5): 36.7% (2,384 samples)
    - good (quality 6-9): 63.3% (4,113 samples)
    Balance ratio: 58% (good trade-off)

    Attributes:
        threshold: Quality score threshold for binary classification.
    """

    def __init__(self, threshold: int = 6):
        """Initialize wine quality dataset loader.

        Args:
            threshold: Quality score threshold. Wines with quality >= threshold
                are labeled as "good" (1), otherwise "bad" (0). Default is 6.
        """
        super().__init__(uci_id=WINE_UCI_ID, name="wine_binary", task_type="classification")
        self.threshold = threshold

    def load(self) -> DatasetInfo:
        """Load and convert wine quality to binary classification.

        Returns:
            DatasetInfo with binary classification labels (0=bad, 1=good).
        """
        # LOAD RAW DATA
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=self.uci_id)

        X = data.data.features.values
        y_raw = data.data.targets.values.ravel().astype(float)

        # Convert to binary: quality >= threshold -> 1 (good), else -> 0 (bad)
        y = (y_raw >= self.threshold).astype(np.int64)
        class_names = np.array(['bad', 'good'])

        return DatasetInfo(
            X=X,
            y=y,
            feature_names=data.data.features.columns.tolist(),
            task_type='classification',
            name=self.name,
            perturbation=None,
            class_names=class_names,
        )


class DryBeanDataset(UCIDataset):
    """Dry bean dataset for multi-class classification.

    Classifies dry bean types based on shape and form features. Contains
    multiple bean variety classes.

    Dataset from UCI ID: Dry Bean.
    """

    def __init__(self):
        """Initialize dry bean dataset loader."""
        super().__init__(uci_id=BEAN_UCI_ID, name="bean", task_type="classification")


class IrisDataset(UCIDataset):
    """Iris flower dataset for multi-class classification."""

    def __init__(self):
        """Initialize iris dataset loader."""
        super().__init__(uci_id=IRIS_UCI_ID, name="iris", task_type="classification")


class RiceDataset(UCIDataset):
    """Rice (Cammeo-Osmancik) dataset for binary classification."""

    def __init__(self):
        """Initialize rice dataset loader."""
        super().__init__(uci_id=RICE_UCI_ID, name="rice", task_type="classification")


class EcoliDataset(UCIDataset):
    """Ecoli protein localization sites dataset for multi-class classification."""

    def __init__(self):
        """Initialize ecoli dataset loader."""
        super().__init__(uci_id=ECOLI_UCI_ID, name="ecoli", task_type="classification")
