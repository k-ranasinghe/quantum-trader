from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np


class DriftDetector:
    """
    Comprehensive drift detection using multiple statistical tests
    """

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)

        # Add small epsilon to avoid log(0)
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi

    @staticmethod
    def calculate_wasserstein(expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Wasserstein distance"""
        return wasserstein_distance(expected, actual)

    @staticmethod
    def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple:
        """Kolmogorov-Smirnov test"""
        statistic, p_value = ks_2samp(expected, actual)
        return statistic, p_value

    @staticmethod
    def detect_drift(
            baseline_data: np.ndarray,
            current_data: np.ndarray,
            threshold_psi: float = 0.2,
            threshold_ks: float = 0.05
    ) -> dict:
        """Comprehensive drift detection"""
        psi = DriftDetector.calculate_psi(baseline_data, current_data)
        wasserstein = DriftDetector.calculate_wasserstein(baseline_data, current_data)
        ks_stat, ks_pval = DriftDetector.ks_test(baseline_data, current_data)

        drift_detected = (psi > threshold_psi) or (ks_pval < threshold_ks)

        return {
            "psi": psi,
            "wasserstein": wasserstein,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "drift_detected": drift_detected
        }