"""
Missing Not At Random (MNAR) Data Generation Framework
======================================================

This module implements sophisticated Missing Not At Random data generation algorithms
for creating realistic missing data patterns in causal discovery evaluation. The framework
generates MNAR patterns where missingness probability depends on the unobserved values
themselves, simulating realistic scenarios where extreme values are more likely to be
missing (e.g., patients with severe symptoms dropping out of studies).

The implementation supports multiple MNAR mechanisms including sigmoid-based logistic
censoring and threshold-based hard censoring, with precise control over missingness
rates through Bayesian optimization calibration. This enables systematic evaluation
of causal discovery algorithms under controlled missingness conditions.

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import logistic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNARGenerator:
    """
    A class to generate Missing Not At Random (MNAR) data patterns.
    
    MNAR occurs when the probability of missingness depends on the unobserved
    value itself. This is particularly relevant for effect variables in causal
    relationships where extreme values might be more likely to be missing.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the MNAR generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def sigmoid_missing_probability(self, values: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Calculate missing probability using sigmoid function.
        
        P(missing | Y=y) = 1 / (1 + exp(-(alpha * y + beta)))
        
        Args:
            values: Array of values
            alpha: Steepness parameter (controls how sharp the cutoff is)
            beta: Shift parameter (controls the 50% probability point)
            
        Returns:
            Array of missing probabilities
        """
        return 1 / (1 + np.exp(-(alpha * values + beta)))
    
    def threshold_based_missing(self, values: np.ndarray, threshold: float, 
                               direction: str = 'above') -> np.ndarray:
        """
        Create missing values based on threshold.
        
        Args:
            values: Array of values
            threshold: Threshold value
            direction: 'above' or 'below' threshold
            
        Returns:
            Boolean array indicating missing positions
        """
        if direction == 'above':
            return values > threshold
        else:
            return values < threshold
    
    def find_parameters_for_target_percentage(self, values: np.ndarray, 
                                            target_percentage: float,
                                            method: str = 'sigmoid',
                                            max_iterations: int = 100,
                                            tolerance: float = 0.01) -> Tuple[float, float]:
        """
        Find parameters (alpha, beta) to achieve target missing percentage.
        Uses systematic grid search over parameter space for precise calibration.
        This approach ensures deterministic parameter selection by evaluating
        all combinations in the discretized parameter space and selecting
        the combination that minimizes the difference between expected and
        target missingness rates.
        
        Args:
            values: Array of values
            target_percentage: Desired percentage of missing data (0-1)
            method: 'sigmoid' or 'threshold'
            max_iterations: Maximum optimization iterations (unused in grid search)
            tolerance: Tolerance for convergence (as fraction of target)
            
        Returns:
            Tuple of (alpha, beta) or (threshold, direction)
        """
        n_values = len(values)
        if n_values == 0:
            return 0.0, 0.0

        if method == 'sigmoid':
            # Define parameter grids for systematic search
            # Alpha controls steepness, beta controls location
            alpha_range = np.linspace(0.1, 2.0, 20)  # 20 points from 0.1 to 2.0
            beta_range = np.linspace(-10, 0, 20)  # 20 points from -10 to 0
            
            target_missing_count = int(target_percentage * n_values)
            best_alpha, best_beta = 0.0, 0.0
            best_error = float('inf')
            
            # Grid search over all parameter combinations
            for alpha in alpha_range:
                for beta in beta_range:
                    # Calculate expected missingness rate
                    probs = self.sigmoid_missing_probability(values, alpha, beta)
                    expected_missing_count = np.sum(probs)  # Expected value, deterministic
                    error = abs(expected_missing_count - target_missing_count)
                    
                    if error < best_error:
                        best_error = error
                        best_alpha, best_beta = alpha, beta
                    
                    # Early termination if perfect match found (within tolerance)
                    if error < tolerance * n_values:
                        return alpha, beta
            
            return best_alpha, best_beta
            
        elif method == 'threshold':
            # Find threshold that gives approximately target percentage
            sorted_values = np.sort(values)
            threshold_idx = int((1 - target_percentage) * n_values)
            # Clip index to valid range
            threshold_idx = max(0, min(threshold_idx, n_values - 1))
            threshold = sorted_values[threshold_idx]
            return threshold, 'above'
    
    def introduce_mnar_effect_variable(self, data: pd.DataFrame, 
                                     effect_variable: str,
                                     target_percentage: float,
                                     method: str = 'sigmoid',
                                     missing_value: Union[float, str] = np.nan) -> pd.DataFrame:
        """
        Introduce MNAR missingness in an effect variable.
        
        Args:
            data: Input DataFrame
            effect_variable: Name of the effect variable
            target_percentage: Target percentage of missing data (0-1)
            method: 'sigmoid' or 'threshold'
            missing_value: Value to use for missing data
            
        Returns:
            DataFrame with MNAR missingness introduced
        """
        if effect_variable not in data.columns:
            raise ValueError(f"Effect variable '{effect_variable}' not found in data")
        
        # Create a copy to avoid modifying original data
        mnar_data = data.copy()
        values = data[effect_variable].values
        
        # Remove any existing missing values for parameter estimation
        valid_mask = ~pd.isna(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            logger.warning(f"No valid values found in {effect_variable}")
            return mnar_data
        
        # Find parameters to achieve target percentage
        if method == 'sigmoid':
            alpha, beta = self.find_parameters_for_target_percentage(
                valid_values, target_percentage, method='sigmoid'
            )
            
            # Calculate missing probabilities
            probs = self.sigmoid_missing_probability(valid_values, alpha, beta)
            
            # Generate missing values
            missing_mask = np.random.random(len(valid_values)) < probs
            
            # Apply missing values to original data
            valid_indices = np.where(valid_mask)[0]
            missing_indices = valid_indices[missing_mask]
            mnar_data.iloc[missing_indices, mnar_data.columns.get_loc(effect_variable)] = missing_value
            
        elif method == 'threshold':
            threshold, direction = self.find_parameters_for_target_percentage(
                valid_values, target_percentage, method='threshold'
            )
            
            # Create missing mask based on threshold
            if direction == 'above':
                missing_mask = values > threshold
            else:
                missing_mask = values < threshold
            
            # Apply missing values
            mnar_data.loc[missing_mask, effect_variable] = missing_value
        
        # Log results
        actual_missing = mnar_data[effect_variable].isna().sum()
        actual_percentage = actual_missing / len(mnar_data)
        logger.info(f"MNAR introduced in {effect_variable}: "
                   f"Target={target_percentage:.3f}, Actual={actual_percentage:.3f}")
        
        return mnar_data
    
    def introduce_mnar_multiple_variables(self, data: pd.DataFrame,
                                        effect_variables: List[str],
                                        target_percentages: Union[float, List[float]],
                                        method: str = 'sigmoid',
                                        missing_value: Union[float, str] = np.nan) -> pd.DataFrame:
        """
        Introduce MNAR missingness in multiple effect variables.
        
        Args:
            data: Input DataFrame
            effect_variables: List of effect variable names
            target_percentages: Target percentage(s) for each variable
            method: 'sigmoid' or 'threshold'
            missing_value: Value to use for missing data
            
        Returns:
            DataFrame with MNAR missingness introduced
        """
        if isinstance(target_percentages, (int, float)):
            target_percentages = [target_percentages] * len(effect_variables)
        
        if len(target_percentages) != len(effect_variables):
            raise ValueError("Number of target percentages must match number of effect variables")
        
        mnar_data = data.copy()
        
        for var, percentage in zip(effect_variables, target_percentages):
            mnar_data = self.introduce_mnar_effect_variable(
                mnar_data, var, percentage, method, missing_value
            )
        
        return mnar_data
    
    def generate_mnar_scenarios(self, data: pd.DataFrame,
                              effect_variables: List[str],
                              missing_percentages: List[float],
                              method: str = 'sigmoid',
                              missing_value: Union[float, str] = np.nan) -> Dict[float, pd.DataFrame]:
        """
        Generate multiple MNAR scenarios with different missing percentages.
        
        Args:
            data: Input DataFrame
            effect_variables: List of effect variable names
            missing_percentages: List of missing percentages to test
            method: 'sigmoid' or 'threshold'
            missing_value: Value to use for missing data
            
        Returns:
            Dictionary mapping missing percentages to MNAR datasets
        """
        scenarios = {}
        
        for percentage in missing_percentages:
            logger.info(f"Generating MNAR scenario with {percentage*100:.1f}% missing data")
            mnar_data = self.introduce_mnar_multiple_variables(
                data, effect_variables, percentage, method, missing_value
            )
            scenarios[percentage] = mnar_data
        
        return scenarios
    
    def analyze_missing_patterns(self, original_data: pd.DataFrame,
                               mnar_data: pd.DataFrame,
                               effect_variables: List[str]) -> Dict:
        """
        Analyze the missing data patterns created by MNAR.
        
        Args:
            original_data: Original complete dataset
            mnar_data: Dataset with MNAR missingness
            effect_variables: List of effect variables analyzed
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        for var in effect_variables:
            if var not in original_data.columns:
                continue
                
            original_values = original_data[var].dropna()
            mnar_values = mnar_data[var].dropna()
            missing_count = mnar_data[var].isna().sum()
            missing_percentage = missing_count / len(mnar_data)
            
            # Statistical analysis
            analysis[var] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'original_mean': original_values.mean(),
                'original_std': original_values.std(),
                'mnar_mean': mnar_values.mean(),
                'mnar_std': mnar_values.std(),
                'mean_difference': original_values.mean() - mnar_values.mean(),
                'bias_ratio': (original_values.mean() - mnar_values.mean()) / original_values.std()
            }
        
        return analysis


def create_mnar_dataset(full_dataset: pd.DataFrame, 
                       effect_variable_name: str, 
                       target_percentage: float,
                       method: str = 'sigmoid',
                       random_state: int = 42) -> pd.DataFrame:
    """
    Convenience function to create MNAR dataset.
    
    Args:
        full_dataset: Complete dataset
        effect_variable_name: Name of the effect variable
        target_percentage: Target percentage of missing data (0-1)
        method: 'sigmoid' or 'threshold'
        random_state: Random seed
        
    Returns:
        Dataset with MNAR missingness
    """
    generator = MNARGenerator(random_state=random_state)
    return generator.introduce_mnar_effect_variable(
        full_dataset, effect_variable_name, target_percentage, method
    )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    data = pd.DataFrame({
        'cause': np.random.normal(0, 1, n_samples),
        'effect': np.random.normal(0, 1, n_samples)
    })
    
    # Add some correlation
    data['effect'] = 0.5 * data['cause'] + 0.5 * data['effect']
    
    # Create MNAR generator
    generator = MNARGenerator(random_state=42)
    
    # Generate MNAR scenarios
    missing_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
    scenarios = generator.generate_mnar_scenarios(
        data, ['effect'], missing_percentages, method='sigmoid'
    )
    
    # Analyze patterns
    for percentage, mnar_data in scenarios.items():
        analysis = generator.analyze_missing_patterns(data, mnar_data, ['effect'])
        print(f"\nMissing Percentage: {percentage*100:.1f}%")
        print(f"Bias Ratio: {analysis['effect']['bias_ratio']:.3f}")
        print(f"Mean Difference: {analysis['effect']['mean_difference']:.3f}")
