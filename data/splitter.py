from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import GLOBAL_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from data.datasets import DatasetInfo


@dataclass
class SplitData:
    """Container for train/validation/test split dataset.

    Holds the complete split dataset with standardized features and
    associated metadata.

    Attributes:
        X_train: Training features, shape (n_train, n_features).
        X_val: Validation features, shape (n_val, n_features).
        X_test: Test features, shape (n_test, n_features).
        y_train: Training labels, shape (n_train,).
        y_val: Validation labels, shape (n_val,).
        y_test: Test labels, shape (n_test,).
        feature_names: List of feature names.
        task_type: Type of machine learning task.
        name: Dataset identifier.
        scaler: Fitted StandardScaler used for feature standardization.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    task_type: Literal["regression", "classification"]
    name: str
    scaler: Optional[StandardScaler] = None


class DataSplitter:
    """Splits datasets into train/validation/test sets with standardization.

    Performs stratified splitting and applies StandardScaler for feature
    normalization. The scaler is fit on training data and applied to all splits.

    Attributes:
        train_size: Proportion of data for training.
        val_size: Proportion of data for validation.
        test_size: Proportion of data for testing.
        seed: Random seed for reproducible splits.
    """

    def __init__(self,
                 train_size: float = TRAIN_RATIO,
                 val_size: float = VAL_RATIO,
                 test_size: float = TEST_RATIO,
                 seed: int = GLOBAL_SEED):
        """Initialize data splitter with split ratios.

        Args:
            train_size: Fraction of data for training. Default from config.
            val_size: Fraction of data for validation. Default from config.
            test_size: Fraction of data for testing. Default from config.
            seed: Random seed for reproducibility. Default from config.

        Raises:
            AssertionError: If split sizes do not sum to 1.0.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-9, \
            "Sizes must sum to 1.0"

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def split(self, info: DatasetInfo) -> SplitData:
        """Split dataset into train/validation/test sets.

        Performs two-stage splitting: first separates training data, then
        splits remainder into validation and test. Applies StandardScaler
        to all features, fit only on training data.

        Args:
            info: DatasetInfo containing the full dataset.

        Returns:
            SplitData with standardized features and preserved labels.
        """
        # FIRST SPLIT TRAIN VS VAL TEST
        X_train, X_temp, y_train, y_temp = train_test_split(
            info.X,
            info.y,
            test_size=(self.val_size + self.test_size),
            random_state=self.seed
        )
        
        # SECOND SPLIT VAL VS TEST
        val_ratio = self.val_size / (self.val_size + self.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=self.seed
        )

        # CENTRALIZED STANDARDSCALER
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        
        return SplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=info.feature_names,
            task_type=info.task_type,
            name=info.name,
            scaler=scaler,
        )
