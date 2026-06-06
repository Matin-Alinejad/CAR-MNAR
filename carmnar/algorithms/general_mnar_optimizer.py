"""
A General Framework for Optimizing MNAR Parameters

This module provides a generalized framework for optimizing parameters of various 
Missing Not At Random (MNAR) models to achieve a target missingness rate.
It is designed to be extensible, allowing different missingness probability 
functions (e.g., Sigmoid, GPD) to be plugged into a common optimization routine.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Protocol, Tuple
from scipy.stats import genpareto, rankdata
from scipy import stats
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingnessModel(Protocol):
    """
    A protocol defining the interface for a missingness probability model.
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Calculates the missingness probability for each value.

        Args:
            values: The input data values.
            params: A dictionary of parameters for the model.

        Returns:
            An array of missingness probabilities.
        """
        ...

class SigmoidModel(MissingnessModel):
    """
    A sigmoid-based missingness model.
    
    P(missing | y) = 1 / (1 + exp(-(alpha * y + beta)))
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 0.0)
        return 1 / (1 + np.exp(-(alpha * values + beta)))

class GPDModel(MissingnessModel):
    """
    A Generalized Pareto Distribution (GPD) based missingness model for tail values.

    P(missing | y) = 1 - G_xi,sigma(y - u) for y > u
                   = 0                         for y <= u
    where G is the GPD cumulative distribution function.
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        u = params.get('u')
        xi = params.get('xi')
        sigma = params.get('sigma')

        if u is None or xi is None or sigma is None:
            raise ValueError("GPD model requires 'u', 'xi', and 'sigma' parameters.")

        probs = np.zeros_like(values, dtype=float)
        tail_mask = values > u
        
        if np.any(tail_mask):
            excess = values[tail_mask] - u
            # Using survival function (1 - cdf) for missingness probability
            # The genpareto.sf is 1 - cdf.
            probs[tail_mask] = genpareto.sf(excess, c=xi, scale=sigma)
            
        return probs

class ThresholdModel(MissingnessModel):
    """
    A simple threshold-based missingness model.

    P(missing | y) = 1 if y > threshold
                   = 0 if y <= threshold
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        threshold = params.get('threshold')
        if threshold is None:
            raise ValueError("Threshold model requires a 'threshold' parameter.")
        return (values > threshold).astype(float)


class MCARModel(MissingnessModel):
    """
    Missing Completely At Random (MCAR) model.
    
    P(missing | y) = p for all y (uniform probability, independent of value)
    
    This serves as a baseline/control condition for MNAR experiments.
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        p = params.get('p', 0.0)
        return np.full(len(values), p)


class MARModel(MissingnessModel):
    """
    Missing At Random (MAR) model - missingness depends on OTHER observed variables.
    
    For simplicity, this model simulates MAR by making missingness probability
    depend on the rank/quantile of the value (observable from complete cases).
    
    P(missing | y) = sigmoid(alpha * rank(y) + beta)
    
    This differs from MNAR where P(missing | y) depends on y itself.
    """
    def calculate_probabilities(self, values: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 0.0)
        
        # Use ranks (percentiles) instead of raw values
        # This simulates dependence on "other" observed characteristics
        ranks = rankdata(values, method='average') / len(values)
        
        return 1 / (1 + np.exp(-(alpha * ranks + beta)))


class GeneralMNAROptimizer:
    """
    A general-purpose optimizer to find the best parameters for an MNAR model
    to achieve a desired target missingness percentage.
    """

    def __init__(self, model: MissingnessModel, random_state: int = 42):
        self.model = model
        self.rng = np.random.RandomState(random_state)

    def find_optimal_parameters(
        self,
        values: np.ndarray,
        target_percentage: float,
        parameter_grid: Dict[str, List[float]]
    ) -> Tuple[Dict[str, float], float]:
        """
        Finds the optimal set of parameters from a grid to best match the
        target missingness percentage, based on expected value.

        Args:
            values: The data values to which missingness will be applied.
            target_percentage: The desired proportion of missing values (0 to 1).
            parameter_grid: A dictionary where keys are parameter names and
                            values are lists of values to test.

        Returns:
            A tuple containing:
            - A dictionary with the best-found parameters.
            - The minimum absolute difference in count achieved.
        """
        target_count = int(target_percentage * len(values))
        
        best_params = None
        min_diff = float('inf')

        param_names = list(parameter_grid.keys())
        param_value_combinations = list(itertools.product(*parameter_grid.values()))

        logger.info(f"Starting parameter search for {self.model.__class__.__name__} with {len(param_value_combinations)} combinations.")

        for param_values in param_value_combinations:
            params = dict(zip(param_names, param_values))
            
            # Calculate probabilities based on the model
            probabilities = self.model.calculate_probabilities(values, params)
            
            # Calculate the expected number of missing values
            expected_missing_count = np.sum(probabilities)

            diff = abs(expected_missing_count - target_count)

            if diff < min_diff:
                min_diff = diff
                best_params = params
            
            # Early stop if a perfect match is found
            if min_diff == 0:
                logger.info("Found perfect parameter match. Stopping early.")
                break

        logger.info(f"Best parameters found: {best_params} with an expected difference of {min_diff:.2f} values.")
        return best_params, min_diff

def introduce_missingness(
    data: pd.DataFrame, 
    variable: str, 
    target_percentage: float,
    model: MissingnessModel, 
    params: Dict[str, float],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Introduces missingness into a DataFrame column using a specified model and parameters.
    """
    rng = np.random.RandomState(random_state)
    data_with_missing = data.copy()
    
    values = data_with_missing[variable].values
    
    probabilities = model.calculate_probabilities(values, params)
    
    missing_mask = rng.rand(len(values)) < probabilities
    
    data_with_missing.loc[missing_mask, variable] = np.nan
    
    actual_percentage = data_with_missing[variable].isnull().sum() / len(data)
    logger.info(f"Introduced missingness in '{variable}'. Target: {target_percentage:.2%}, Actual: {actual_percentage:.2%}")
    
    return data_with_missing


if __name__ == '__main__':
    # --- Example Usage and Validation ---
    
    # 1. Create sample data
    np.random.seed(0)
    sample_data = pd.DataFrame({
        'x': np.random.randn(1000) * 2 + 5  # Normally distributed data
    })
    
    target_missing = 0.20 # 20%

    # --- Test 1: Sigmoid Model ---
    print("\n--- Testing Sigmoid Model ---")
    sigmoid_model = SigmoidModel()
    sigmoid_optimizer = GeneralMNAROptimizer(sigmoid_model)
    
    # Define a grid for alpha and beta
    sigmoid_grid = {
        'alpha': np.linspace(0.1, 2.0, 10),
        'beta': np.linspace(-10, 0, 10)
    }
    
    best_sigmoid_params, min_diff_sigmoid = sigmoid_optimizer.find_optimal_parameters(
        sample_data['x'].values, 
        target_missing, 
        sigmoid_grid
    )
    
    print(f"Optimal Sigmoid Params: {best_sigmoid_params} (Gap: {min_diff_sigmoid:.2f})")
    
    # Introduce missingness with optimal params
    sigmoid_missing_df = introduce_missingness(
        sample_data, 'x', target_missing, sigmoid_model, best_sigmoid_params
    )

    # --- Test 2: GPD Model ---
    print("\n--- Testing GPD Model ---")
    gpd_model = GPDModel()
    gpd_optimizer = GeneralMNAROptimizer(gpd_model)
    
    # Define a grid for u, xi, and sigma
    # Let's make 'u' part of the search space.
    u_quantiles = [75, 80, 85, 90]
    u_values = [np.percentile(sample_data['x'], q) for q in u_quantiles]

    gpd_grid = {
        'u': u_values,
        'xi': np.linspace(0.01, 0.5, 5),  # Shape parameter
        'sigma': np.linspace(0.5, 2.0, 5) # Scale parameter
    }
    
    best_gpd_params, min_diff_gpd = gpd_optimizer.find_optimal_parameters(
        sample_data['x'].values, 
        target_missing, 
        gpd_grid
    )
    
    print(f"Optimal GPD Params: {best_gpd_params} (Gap: {min_diff_gpd:.2f})")

    # Introduce missingness with optimal params
    gpd_missing_df = introduce_missingness(
        sample_data, 'x', target_missing, gpd_model, best_gpd_params
    )

    # --- Test 3: Threshold Model ---
    print("\n--- Testing Threshold Model ---")
    threshold_model = ThresholdModel()
    threshold_optimizer = GeneralMNAROptimizer(threshold_model)

    # Define a grid for the threshold
    threshold_grid = {
        'threshold': np.linspace(
            sample_data['x'].min(), sample_data['x'].max(), 50
        )
    }

    best_threshold_params, min_diff_thresh = threshold_optimizer.find_optimal_parameters(
        sample_data['x'].values,
        target_missing,
        threshold_grid
    )

    print(f"Optimal Threshold Params: {best_threshold_params} (Gap: {min_diff_thresh:.2f})")

    # Introduce missingness
    threshold_missing_df = introduce_missingness(
        sample_data, 'x', target_missing, threshold_model, best_threshold_params
    )
