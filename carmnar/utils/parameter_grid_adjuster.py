"""
Parameter Grid Adjustment Utilities

This module provides utilities for dynamically adjusting parameter grids
based on data statistics, ensuring optimal performance across different datasets.
"""

import numpy as np
from typing import Dict, List, Any


def adjust_parameter_grid_for_data(
    param_grid: Dict[str, List[float]],
    values: np.ndarray,
    mechanism_type: str
) -> Dict[str, List[float]]:
    """
    Adjust parameter grid based on data statistics.
    
    Args:
        param_grid: Original parameter grid
        values: Data values to analyze
        mechanism_type: Type of missingness mechanism
        
    Returns:
        Adjusted parameter grid
    """
    adjusted = param_grid.copy()
    
    value_mean = np.mean(values)
    value_std = np.std(values)
    value_min = np.min(values)
    value_max = np.max(values)
    value_range = value_max - value_min
    
    if mechanism_type == 'sigmoid':
        # Adjust alpha and beta based on data scale
        if 'alpha' in adjusted:
            # Alpha controls steepness - keep reasonable range
            alpha_range = adjusted['alpha']
            if isinstance(alpha_range, list) and len(alpha_range) > 0:
                # Normalize alpha range if values are very large
                if value_std > 10:
                    # Scale down alpha for large value ranges
                    adjusted['alpha'] = [a * 0.1 for a in alpha_range]
                elif value_std < 0.1:
                    # Scale up alpha for small value ranges
                    adjusted['alpha'] = [a * 10 for a in alpha_range]
        
        if 'beta' in adjusted:
            # Beta controls location - adjust based on value mean and range
            beta_range = adjusted['beta']
            if isinstance(beta_range, list) and len(beta_range) > 0:
                # Shift beta to center around appropriate range for the data
                if value_mean > 0:
                    # For positive values, use negative beta
                    adjusted['beta'] = np.linspace(
                        -value_mean * 2,
                        -value_mean * 0.1,
                        len(beta_range)
                    ).tolist()
                else:
                    # For negative or zero-centered values, adjust accordingly
                    adjusted['beta'] = np.linspace(
                        -value_range,
                        0,
                        len(beta_range)
                    ).tolist()
    
    elif mechanism_type == 'threshold':
        # Adjust threshold based on actual value range
        if 'threshold' in adjusted:
            adjusted['threshold'] = np.linspace(
                value_min,
                value_max,
                len(adjusted['threshold'])
            ).tolist()
    
    elif mechanism_type == 'gpd':
        # Adjust GPD parameters
        if 'u' in adjusted:
            u_values = adjusted['u']
            if isinstance(u_values[0], (int, float)):
                # If values are quantiles (0-100), convert to actual values
                if all(0 <= u <= 100 for u in u_values):
                    adjusted['u'] = [np.percentile(values, q) for q in u_values]
                # If values are already in data range, use as is
                elif all(value_min <= u <= value_max for u in u_values):
                    pass  # Use as is
                else:
                    # Use quantile-based approach
                    adjusted['u'] = [np.percentile(values, q) for q in [75, 80, 85, 90]]
        
        # Adjust sigma and xi based on value scale
        if 'sigma' in adjusted:
            sigma_range = adjusted['sigma']

            if 'xi' in adjusted:
                # Broaden xi range to capture extremely heavy tails when std is high
                xi_range = adjusted['xi']
                if value_std > 2 * abs(value_mean):
                    adjusted['xi'] = [min(0.95, xi * 1.5) for xi in xi_range]

            if isinstance(sigma_range, list) and len(sigma_range) > 0:
                # Allow larger scale factors for heavy-tailed targets (>25% MNAR)
                scale_factor = max(0.1, min(20, value_std))
                adjusted['sigma'] = [s * scale_factor for s in sigma_range]
    
    return adjusted


def create_adaptive_parameter_grid(
    mechanism_type: str,
    values: np.ndarray,
    n_alpha: int = 15,
    n_beta: int = 15,
    n_threshold: int = 50
) -> Dict[str, List[float]]:
    """
    Create an adaptive parameter grid based on data statistics.
    
    Args:
        mechanism_type: Type of missingness mechanism
        values: Data values
        n_alpha: Number of alpha values for sigmoid
        n_beta: Number of beta values for sigmoid
        n_threshold: Number of threshold values
        
    Returns:
        Parameter grid optimized for the data
    """
    value_mean = np.mean(values)
    value_std = np.std(values)
    value_min = np.min(values)
    value_max = np.max(values)
    
    if mechanism_type == 'sigmoid':
        # Adaptive alpha range based on data scale
        if value_std > 10:
            alpha_range = np.linspace(0.01, 0.5, n_alpha)
        elif value_std < 0.1:
            alpha_range = np.linspace(1.0, 10.0, n_alpha)
        else:
            alpha_range = np.linspace(0.05, 2.0, n_alpha)
        
        # Adaptive beta range
        if value_mean > 0:
            beta_range = np.linspace(-value_mean * 2.5, -value_mean * 0.1, n_beta)
        else:
            beta_range = np.linspace(value_min, value_max, n_beta)
        
        return {
            'alpha': alpha_range.tolist(),
            'beta': beta_range.tolist()
        }
    
    elif mechanism_type == 'threshold':
        return {
            'threshold': np.linspace(value_min, value_max, n_threshold).tolist()
        }
    
    elif mechanism_type == 'gpd':
        # Use quantile-based u values
        u_quantiles = [60, 65, 70, 75, 80, 85, 90]
        u_values = [np.percentile(values, q) for q in u_quantiles]
        
        # Scale sigma based on data
        sigma_base = max(0.5, value_std / 2)
        sigma_range = np.linspace(sigma_base * 0.5, sigma_base * 3, 10)
        
        return {
            'u': u_values,
            'xi': np.linspace(0.05, 0.7, 10).tolist(),
            'sigma': sigma_range.tolist()
        }
    
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")

