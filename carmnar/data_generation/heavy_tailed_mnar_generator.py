"""
Heavy-Tailed Missing Not At Random (MNAR) Data Generator

This module implements sophisticated MNAR mechanisms that induce missingness
in the tails of data distributions, reflecting scenarios where extreme events
are systematically censored. This is particularly relevant for medical and
financial data where extreme values are often missing due to measurement
limitations or reporting biases.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TailMNARConfig:
    """Configuration class for heavy-tailed MNAR generation."""
    mechanism: str  # 'threshold' or 'parametric'
    target_missing_rate: float
    quantile_threshold: Optional[float] = None  # For threshold-based
    tail_index: Optional[float] = None  # For parametric
    shape_parameter: Optional[float] = None  # GPD shape parameter
    scale_parameter: Optional[float] = None  # GPD scale parameter
    threshold_value: Optional[float] = None  # GPD threshold
    random_seed: Optional[int] = None
    min_tail_samples: int = 50  # Minimum samples in tail for estimation
    max_iterations: int = 1000  # Maximum iterations for parameter optimization

class HeavyTailedMNARGenerator:
    """
    Advanced MNAR generator implementing heavy-tailed missingness mechanisms.
    
    This class provides two primary mechanisms for inducing missingness in
    the tails of data distributions:
    
    1. Threshold-based censoring: Removes data points exceeding specific quantiles
    2. Parametric tail-based censoring: Uses heavy-tailed distributions (GPD)
       to model missingness probability in extreme values
    
    The implementation includes robust parameter estimation, validation metrics,
    and comprehensive reporting capabilities suitable for academic research.
    """
    
    def __init__(self, config: TailMNARConfig):
        """
        Initialize the heavy-tailed MNAR generator.
        
        Args:
            config: Configuration object specifying the MNAR mechanism and parameters
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.estimated_parameters = {}
        self.validation_metrics = {}
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.mechanism not in ['threshold', 'parametric']:
            raise ValueError("Mechanism must be 'threshold' or 'parametric'")
            
        if not 0 < self.config.target_missing_rate < 1:
            raise ValueError("Target missing rate must be between 0 and 1")
            
        if self.config.mechanism == 'threshold' and self.config.quantile_threshold is None:
            raise ValueError("Quantile threshold must be specified for threshold mechanism")
            
        if self.config.mechanism == 'parametric' and self.config.tail_index is None:
            raise ValueError("Tail index must be specified for parametric mechanism")
    
    def generate_missingness_mask(self, data: np.ndarray, 
                                target_variable: str = 'target') -> Tuple[np.ndarray, Dict]:
        """
        Generate missingness mask using heavy-tailed MNAR mechanism.
        
        Args:
            data: Input data array or DataFrame
            target_variable: Name of the target variable for MNAR generation
            
        Returns:
            Tuple of (missing_mask, generation_info)
            - missing_mask: Boolean array indicating missing values
            - generation_info: Dictionary containing generation parameters and metrics
        """
        if isinstance(data, pd.DataFrame):
            values = data[target_variable].values
        else:
            values = data.flatten()
            
        # Remove any existing missing values for clean analysis
        valid_mask = ~np.isnan(values)
        clean_values = values[valid_mask]
        
        if len(clean_values) < self.config.min_tail_samples:
            raise ValueError(f"Insufficient data: need at least {self.config.min_tail_samples} samples")
        
        if self.config.mechanism == 'threshold':
            return self._threshold_based_censoring(clean_values, valid_mask)
        else:
            return self._parametric_tail_censoring(clean_values, valid_mask)
    
    def _threshold_based_censoring(self, values: np.ndarray, 
                                 valid_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Implement threshold-based censoring mechanism.
        
        Removes data points exceeding a specific quantile threshold.
        """
        logger.info(f"Implementing threshold-based censoring at {self.config.quantile_threshold}th percentile")
        
        # Calculate threshold value
        threshold_value = np.percentile(values, self.config.quantile_threshold)
        
        # Generate missingness mask
        missing_mask = np.zeros_like(valid_mask, dtype=bool)
        missing_mask[valid_mask] = values > threshold_value
        
        # Calculate actual missing rate
        actual_missing_rate = np.mean(missing_mask)
        
        # Calculate tail statistics
        tail_values = values[values > threshold_value]
        tail_index = self._estimate_tail_index(tail_values) if len(tail_values) > 10 else None
        
        generation_info = {
            'mechanism': 'threshold',
            'quantile_threshold': self.config.quantile_threshold,
            'threshold_value': threshold_value,
            'target_missing_rate': self.config.target_missing_rate,
            'actual_missing_rate': actual_missing_rate,
            'tail_index': tail_index,
            'tail_samples': len(tail_values),
            'total_samples': len(values)
        }
        
        self.estimated_parameters = generation_info
        return missing_mask, generation_info
    
    def _parametric_tail_censoring(self, values: np.ndarray, 
                                 valid_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Implement parametric tail-based censoring using Generalized Pareto Distribution.
        
        Uses GPD to model the probability of missingness in extreme values.
        """
        logger.info("Implementing parametric tail-based censoring using GPD")
        
        # Estimate GPD parameters if not provided
        if self.config.shape_parameter is None or self.config.scale_parameter is None:
            shape, scale, threshold = self._estimate_gpd_parameters(values)
        else:
            shape = self.config.shape_parameter
            scale = self.config.scale_parameter
            threshold = self.config.threshold_value or np.percentile(values, 95)
        
        # Calculate missingness probabilities using GPD
        excess_values = values - threshold
        excess_mask = excess_values > 0
        
        if np.sum(excess_mask) == 0:
            # No values exceed threshold
            missing_mask = np.zeros_like(valid_mask, dtype=bool)
            actual_missing_rate = 0.0
        else:
            # Calculate GPD-based missingness probabilities
            gpd_probs = self._calculate_gpd_missingness_probability(
                excess_values[excess_mask], shape, scale
            )
            
            # Generate missingness mask
            missing_mask = np.zeros_like(valid_mask, dtype=bool)
            excess_indices = np.where(valid_mask)[0][excess_mask]
            
            # Sample missingness based on GPD probabilities
            missing_indices = self.rng.choice(
                excess_indices, 
                size=int(np.sum(gpd_probs > self.rng.random(len(gpd_probs)))),
                replace=False
            )
            missing_mask[missing_indices] = True
            
            actual_missing_rate = np.mean(missing_mask)
        
        # Calculate tail statistics
        tail_values = values[values > threshold]
        tail_index = self._estimate_tail_index(tail_values) if len(tail_values) > 10 else None
        
        generation_info = {
            'mechanism': 'parametric',
            'gpd_shape': shape,
            'gpd_scale': scale,
            'gpd_threshold': threshold,
            'target_missing_rate': self.config.target_missing_rate,
            'actual_missing_rate': actual_missing_rate,
            'tail_index': tail_index,
            'tail_samples': len(tail_values),
            'total_samples': len(values)
        }
        
        self.estimated_parameters = generation_info
        return missing_mask, generation_info
    
    def _estimate_gpd_parameters(self, values: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate Generalized Pareto Distribution parameters using MLE.
        
        Uses the peaks-over-threshold method to fit GPD to tail data.
        """
        # Use 95th percentile as initial threshold
        initial_threshold = np.percentile(values, 95)
        excess_values = values - initial_threshold
        excess_mask = excess_values > 0
        
        if np.sum(excess_mask) < 10:
            # Fallback to 90th percentile if insufficient tail data
            initial_threshold = np.percentile(values, 90)
            excess_values = values - initial_threshold
            excess_mask = excess_values > 0
        
        if np.sum(excess_mask) < 5:
            # Use default parameters if insufficient data
            return 0.1, 1.0, initial_threshold
        
        # Fit GPD using MLE
        try:
            shape, loc, scale = stats.genpareto.fit(excess_values[excess_mask], floc=0)
            return shape, scale, initial_threshold
        except:
            # Fallback to method of moments
            excess_std = np.std(excess_values[excess_mask])
            excess_mean = np.mean(excess_values[excess_mask])
            shape = 0.5 * (1 - (excess_mean / excess_std) ** 2)
            scale = 0.5 * excess_mean * (1 + (excess_mean / excess_std) ** 2)
            return max(shape, 0.01), max(scale, 0.01), initial_threshold
    
    def _calculate_gpd_missingness_probability(self, excess_values: np.ndarray, 
                                             shape: float, scale: float) -> np.ndarray:
        """
        Calculate missingness probabilities using GPD.
        
        Higher values in the tail have higher probability of being missing.
        """
        if scale <= 0:
            return np.ones_like(excess_values) * 0.5
        
        # Calculate GPD survival function (probability of exceeding value)
        if shape == 0:
            # Exponential case
            survival_probs = np.exp(-excess_values / scale)
        else:
            # General case
            survival_probs = (1 + shape * excess_values / scale) ** (-1 / shape)
        
        # Convert to missingness probability (higher survival = higher missing prob)
        # Normalize to target missing rate
        missing_probs = survival_probs * self.config.target_missing_rate
        
        # Ensure probabilities are in [0, 1]
        missing_probs = np.clip(missing_probs, 0, 1)
        
        return missing_probs
    
    def _estimate_tail_index(self, tail_values: np.ndarray) -> Optional[float]:
        """
        Estimate the tail index (inverse of shape parameter) using Hill estimator.
        
        The tail index characterizes the heaviness of the tail distribution.
        """
        if len(tail_values) < 10:
            return None
        
        try:
            # Sort in descending order
            sorted_values = np.sort(tail_values)[::-1]
            
            # Use top 20% of tail values for estimation
            k = max(5, len(sorted_values) // 5)
            top_values = sorted_values[:k]
            
            # Hill estimator
            log_ratios = np.log(top_values[:-1] / top_values[1:])
            tail_index = np.mean(log_ratios)
            
            return tail_index
        except:
            return None
    
    def validate_missingness_pattern(self, data: np.ndarray, 
                                   missing_mask: np.ndarray) -> Dict[str, float]:
        """
        Validate the generated missingness pattern and calculate quality metrics.
        
        Args:
            data: Original data
            missing_mask: Boolean mask indicating missing values
            
        Returns:
            Dictionary containing validation metrics
        """
        if isinstance(data, pd.DataFrame):
            values = data.iloc[:, 0].values  # Use first column
        else:
            values = data.flatten()
        
        # Calculate basic statistics
        missing_rate = np.mean(missing_mask)
        observed_values = values[~missing_mask]
        missing_values = values[missing_mask]
        
        # Calculate tail statistics
        tail_threshold = np.percentile(values, 95)
        tail_missing_rate = np.mean(missing_mask[values > tail_threshold])
        
        # Calculate correlation between value magnitude and missingness
        value_missingness_corr = np.corrcoef(values, missing_mask.astype(float))[0, 1]
        
        # Calculate tail index for observed and missing data
        observed_tail_index = self._estimate_tail_index(observed_values)
        missing_tail_index = self._estimate_tail_index(missing_values)
        
        validation_metrics = {
            'missing_rate': missing_rate,
            'tail_missing_rate': tail_missing_rate,
            'value_missingness_correlation': value_missingness_corr,
            'observed_tail_index': observed_tail_index,
            'missing_tail_index': missing_tail_index,
            'tail_threshold': tail_threshold,
            'n_observed': len(observed_values),
            'n_missing': len(missing_values)
        }
        
        self.validation_metrics = validation_metrics
        return validation_metrics
    
    def generate_comprehensive_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive report of the MNAR generation process.
        
        Returns:
            Dictionary containing all generation parameters, validation metrics,
            and statistical summaries suitable for academic reporting.
        """
        report = {
            'configuration': {
                'mechanism': self.config.mechanism,
                'target_missing_rate': self.config.target_missing_rate,
                'quantile_threshold': self.config.quantile_threshold,
                'tail_index': self.config.tail_index,
                'random_seed': self.config.random_seed
            },
            'estimated_parameters': self.estimated_parameters,
            'validation_metrics': self.validation_metrics,
            'generation_summary': {
                'mechanism_used': self.config.mechanism,
                'actual_missing_rate': self.estimated_parameters.get('actual_missing_rate', 0),
                'target_achieved': abs(self.estimated_parameters.get('actual_missing_rate', 0) - 
                                     self.config.target_missing_rate) < 0.05,
                'tail_characterization': {
                    'tail_index': self.estimated_parameters.get('tail_index'),
                    'tail_samples': self.estimated_parameters.get('tail_samples', 0)
                }
            }
        }
        
        return report

# Convenience functions for common use cases
def create_threshold_mnar_config(target_missing_rate: float, 
                                quantile_threshold: float,
                                random_seed: Optional[int] = None) -> TailMNARConfig:
    """Create configuration for threshold-based MNAR generation."""
    return TailMNARConfig(
        mechanism='threshold',
        target_missing_rate=target_missing_rate,
        quantile_threshold=quantile_threshold,
        random_seed=random_seed
    )

def create_parametric_mnar_config(target_missing_rate: float,
                                 tail_index: float,
                                 random_seed: Optional[int] = None) -> TailMNARConfig:
    """Create configuration for parametric MNAR generation."""
    return TailMNARConfig(
        mechanism='parametric',
        target_missing_rate=target_missing_rate,
        tail_index=tail_index,
        random_seed=random_seed
    )

def generate_heavy_tailed_mnar(data: Union[np.ndarray, pd.DataFrame],
                              config: TailMNARConfig,
                              target_variable: str = 'target') -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to generate heavy-tailed MNAR data.
    
    Args:
        data: Input data
        config: MNAR configuration
        target_variable: Name of target variable (for DataFrames)
        
    Returns:
        Tuple of (missing_mask, generation_info)
    """
    generator = HeavyTailedMNARGenerator(config)
    missing_mask, generation_info = generator.generate_missingness_mask(data, target_variable)
    
    # Validate the pattern
    validation_metrics = generator.validate_missingness_pattern(data, missing_mask)
    generation_info.update(validation_metrics)
    
    return missing_mask, generation_info

if __name__ == "__main__":
    # Example usage and testing
    print("Heavy-Tailed MNAR Generator - Example Usage")
    
    # Generate sample data with heavy tails
    np.random.seed(42)
    n_samples = 1000
    # Generate data with heavy tails using t-distribution
    data = stats.t.rvs(df=3, size=n_samples) + np.random.normal(0, 0.1, n_samples)
    
    # Test threshold-based censoring
    print("\n=== Threshold-Based Censoring ===")
    config = create_threshold_mnar_config(target_missing_rate=0.2, quantile_threshold=90)
    missing_mask, info = generate_heavy_tailed_mnar(data, config)
    print(f"Missing rate: {info['actual_missing_rate']:.3f}")
    print(f"Tail missing rate: {info['tail_missing_rate']:.3f}")
    print(f"Tail index: {info['tail_index']:.3f}")
    
    # Test parametric censoring
    print("\n=== Parametric Censoring ===")
    config = create_parametric_mnar_config(target_missing_rate=0.2, tail_index=0.5)
    missing_mask, info = generate_heavy_tailed_mnar(data, config)
    print(f"Missing rate: {info['actual_missing_rate']:.3f}")
    print(f"Tail missing rate: {info['tail_missing_rate']:.3f}")
    print(f"Tail index: {info['tail_index']:.3f}")
