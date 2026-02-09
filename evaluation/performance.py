from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ClassificationMetrics:
    """Metrics for evaluating classification model performance.

    Provides weighted classification metrics and optionally AUC when
    probability estimates are available.
    """
    
    @staticmethod
    def compute(y_true: np.ndarray, 
                y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute standard classification metrics.

        Args:
            y_true: Ground truth class labels, shape (n_samples,).
            y_pred: Predicted class labels, shape (n_samples,).
            y_proba: Predicted probabilities per class, shape
                (n_samples, n_classes). Optional; used for AUC.

        Returns:
            Dictionary mapping metric names to values. Includes accuracy,
            precision, recall, F1, and AUC (None if probabilities missing or
            AUC computation fails).
        """
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except ValueError:
                metrics['auc'] = None
        
        return metrics
