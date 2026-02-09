import numpy as np


class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality.

    Provides calibration, coverage, sharpness, and proper scoring rules
    for both regression and classification outputs. Lower values generally
    indicate better uncertainty quality unless stated otherwise.
    """
    
    @staticmethod
    def ece(y_true: np.ndarray, 
            y_pred: np.ndarray, 
            uncertainty: np.ndarray, 
            n_bins: int = 10) -> float:
        """Expected Calibration Error for regression.

        Bins predictions by predicted uncertainty and compares the average
        uncertainty to the average absolute error in each bin.

        Args:
            y_true: True target values, shape (n_samples,).
            y_pred: Predicted values, shape (n_samples,).
            uncertainty: Predicted uncertainty per sample, shape (n_samples,).
            n_bins: Number of bins for grouping by uncertainty.

        Returns:
            Expected calibration error where lower values indicate better
            alignment between predicted uncertainty and observed errors.
        """
        # BIN BY UNCERTAINTY
        bin_edges = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
        
        ece = 0.0
        total_samples = len(y_true)
        
        for i in range(n_bins):
            mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (uncertainty >= bin_edges[i]) & (uncertainty <= bin_edges[i + 1])
            
            if mask.sum() == 0:
                continue
            
            # AVERAGE UNCERTAINTY IN BIN
            avg_uncertainty = uncertainty[mask].mean()
            
            # AVERAGE ABSOLUTE ERROR IN BIN
            avg_error = np.abs(y_true[mask] - y_pred[mask]).mean()
            
            # WEIGHTED CONTRIBUTION
            ece += np.abs(avg_uncertainty - avg_error) * mask.sum() / total_samples
        
        return ece
    
    @staticmethod
    def ece_classification(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           n_bins: int = 10) -> float:
        """Expected Calibration Error for classification.

        Groups predictions by confidence to compare mean confidence with
        empirical accuracy per bin.

        Args:
            y_true: True class labels, shape (n_samples,).
            y_proba: Predicted class probabilities, shape
                (n_samples, n_classes).
            n_bins: Number of confidence bins between 0 and 1.

        Returns:
            Expected calibration error where lower values indicate better
            probability calibration.
        """
        # Confidence = max probability
        confidence = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        correct = (predictions == y_true).astype(float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        total_samples = len(y_true)
        
        for i in range(n_bins):
            mask = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
            
            if mask.sum() == 0:
                continue
            
            avg_confidence = confidence[mask].mean()
            avg_accuracy = correct[mask].mean()
            
            ece += np.abs(avg_confidence - avg_accuracy) * mask.sum() / total_samples
        
        return ece
    
    @staticmethod
    def coverage(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 uncertainty: np.ndarray,
                 confidence_level: float = 0.9) -> float:
        """Prediction interval coverage for regression.

        Computes the fraction of true targets lying within a symmetric
        normal interval around predictions.

        Args:
            y_true: True target values, shape (n_samples,).
            y_pred: Predicted values, shape (n_samples,).
            uncertainty: Predicted standard deviation per sample.
            confidence_level: Desired coverage probability (e.g., 0.9).

        Returns:
            Observed coverage proportion; should align with confidence_level
            for well-calibrated uncertainties.
        """
        from scipy.stats import norm
        
        z = norm.ppf((1 + confidence_level) / 2)  # z=1.645 for 90%, z=1.96 for 95%
        
        lower = y_pred - z * uncertainty
        upper = y_pred + z * uncertainty
        
        within = (y_true >= lower) & (y_true <= upper)
        
        return within.mean()
    
    @staticmethod
    def sharpness(uncertainty: np.ndarray) -> float:
        """Average predicted uncertainty.

        Acts as a proxy for interval width. Lower values are preferred when
        coverage remains adequate.

        Args:
            uncertainty: Predicted uncertainty per sample.

        Returns:
            Mean uncertainty value.
        """
        return uncertainty.mean()
    
    @staticmethod
    def nll_gaussian(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_std: np.ndarray) -> float:
        """Negative log-likelihood assuming Gaussian predictive density.

        Proper scoring rule that rewards accurate means and well-calibrated
        variances.

        Args:
            y_true: True target values, shape (n_samples,).
            y_pred: Predicted means, shape (n_samples,).
            y_std: Predicted standard deviations, shape (n_samples,).

        Returns:
            Average negative log-likelihood (lower is better).
        """
        variance = y_std ** 2 + 1e-8  # Avoid division by zero
        nll = 0.5 * (np.log(2 * np.pi * variance) + (y_true - y_pred) ** 2 / variance)
        return nll.mean()
    
    @staticmethod
    def crps(y_true: np.ndarray,
             y_pred: np.ndarray,
             y_std: np.ndarray) -> float:
        """Continuous Ranked Probability Score for Gaussian forecasts.

        Generalizes MAE to probabilistic predictions by integrating the
        squared difference between predicted and empirical CDFs.

        Args:
            y_true: True target values, shape (n_samples,).
            y_pred: Predicted means, shape (n_samples,).
            y_std: Predicted standard deviations, shape (n_samples,).

        Returns:
            Average CRPS value (lower is better).
        """
        from scipy.stats import norm
        
        z = (y_true - y_pred) / (y_std + 1e-8)
        crps_values = y_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        
        return crps_values.mean()
    
    @staticmethod
    def compute_all_classification(y_true: np.ndarray,
                                   y_proba: np.ndarray,
                                   aleatoric: np.ndarray,
                                   epistemic: np.ndarray) -> dict:
        """Compute a suite of classification UQ metrics.

        Args:
            y_true: True class labels, shape (n_samples,).
            y_proba: Predicted class probabilities, shape
                (n_samples, n_classes).
            aleatoric: Aleatoric uncertainty estimates per sample.
            epistemic: Epistemic uncertainty estimates per sample.

        Returns:
            Dictionary containing calibration, sharpness, entropy, and mutual
            information summaries.
        """
        return {
            'ece': UncertaintyMetrics.ece_classification(y_true, y_proba),
            'sharpness_aleatoric': UncertaintyMetrics.sharpness(aleatoric),
            'sharpness_epistemic': UncertaintyMetrics.sharpness(epistemic),
            'mean_entropy': aleatoric.mean(),  # Aleatoric = predictive entropy
            'mean_mi': epistemic.mean()        # Epistemic = mutual information
        }
