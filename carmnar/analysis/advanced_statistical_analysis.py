"""
Advanced Statistical Analysis Framework for MNAR Mechanism Comparison

This module provides comprehensive statistical analysis capabilities for comparing
different MNAR mechanisms, including Monte Carlo simulations, significance testing,
effect size calculations, and robustness assessment.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.effect_size import effectsize_ttest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StatisticalAnalysisConfig:
    """Configuration for statistical analysis."""
    significance_level: float = 0.05
    confidence_level: float = 0.95
    multiple_comparison_correction: str = 'bonferroni'
    effect_size_threshold: float = 0.2
    power_analysis: bool = True
    bootstrap_samples: int = 1000
    random_seed: int = 42

class AdvancedStatisticalAnalyzer:
    """
    Advanced statistical analyzer for MNAR mechanism comparison.
    
    This class provides comprehensive statistical analysis capabilities including
    significance testing, effect size calculations, power analysis, and robustness
    assessment for comparing different MNAR mechanisms.
    """
    
    def __init__(self, config: StatisticalAnalysisConfig):
        """
        Initialize the statistical analyzer.
        
        Args:
            config: Configuration object for statistical analysis
        """
        self.config = config
        self.analysis_results = {}
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        logger.info("Initialized Advanced Statistical Analyzer")
    
    def perform_comprehensive_analysis(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on experimental data.
        
        Args:
            experimental_data: Dictionary containing experimental results
            
        Returns:
            Dictionary containing all statistical analysis results
        """
        logger.info("Starting comprehensive statistical analysis")
        
        # Initialize analysis results
        self.analysis_results = {
            'descriptive_statistics': {},
            'significance_testing': {},
            'effect_size_analysis': {},
            'power_analysis': {},
            'robustness_assessment': {},
            'multiple_comparison_correction': {},
            'bootstrap_analysis': {},
            'outlier_analysis': {}
        }
        
        # Perform analysis for each dataset
        for dataset_name, dataset_results in experimental_data.items():
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Extract performance data
            performance_data = self._extract_performance_data(dataset_results)
            
            # Perform descriptive statistics
            self.analysis_results['descriptive_statistics'][dataset_name] = \
                self._calculate_descriptive_statistics(performance_data)
            
            # Perform significance testing
            self.analysis_results['significance_testing'][dataset_name] = \
                self._perform_significance_testing(performance_data)
            
            # Calculate effect sizes
            self.analysis_results['effect_size_analysis'][dataset_name] = \
                self._calculate_effect_sizes(performance_data)
            
            # Perform power analysis
            if self.config.power_analysis:
                self.analysis_results['power_analysis'][dataset_name] = \
                    self._perform_power_analysis(performance_data)
            
            # Assess robustness
            self.analysis_results['robustness_assessment'][dataset_name] = \
                self._assess_robustness(performance_data)
            
            # Perform bootstrap analysis
            self.analysis_results['bootstrap_analysis'][dataset_name] = \
                self._perform_bootstrap_analysis(performance_data)
            
            # Analyze outliers
            self.analysis_results['outlier_analysis'][dataset_name] = \
                self._analyze_outliers(performance_data)
        
        # Perform cross-dataset analysis
        self._perform_cross_dataset_analysis(experimental_data)
        
        # Apply multiple comparison correction
        self._apply_multiple_comparison_correction()
        
        logger.info("Statistical analysis completed")
        return self.analysis_results
    
    def _extract_performance_data(self, dataset_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract performance data for statistical analysis."""
        performance_data = {
            'threshold': [],
            'parametric': [],
            'sigmoid': []
        }
        
        # Extract threshold results
        for result in dataset_results['tail_driven_results']['threshold_results'].values():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                performance_data['threshold'].extend(f1_scores)
        
        # Extract parametric results
        for result in dataset_results['tail_driven_results']['parametric_results'].values():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                performance_data['parametric'].extend(f1_scores)
        
        # Extract sigmoid results
        for result in dataset_results['sigmoid_results'].values():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                performance_data['sigmoid'].extend(f1_scores)
        
        return performance_data
    
    def _calculate_descriptive_statistics(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics."""
        descriptive_stats = {}
        
        for mechanism, scores in performance_data.items():
            if scores:
                scores_array = np.array(scores)
                
                descriptive_stats[mechanism] = {
                    'n_observations': len(scores),
                    'mean': np.mean(scores_array),
                    'median': np.median(scores_array),
                    'std': np.std(scores_array, ddof=1),
                    'variance': np.var(scores_array, ddof=1),
                    'min': np.min(scores_array),
                    'max': np.max(scores_array),
                    'range': np.max(scores_array) - np.min(scores_array),
                    'q25': np.percentile(scores_array, 25),
                    'q75': np.percentile(scores_array, 75),
                    'iqr': np.percentile(scores_array, 75) - np.percentile(scores_array, 25),
                    'skewness': stats.skew(scores_array),
                    'kurtosis': stats.kurtosis(scores_array),
                    'coefficient_of_variation': np.std(scores_array) / np.mean(scores_array) if np.mean(scores_array) > 0 else 0
                }
        
        return descriptive_stats
    
    def _perform_significance_testing(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform comprehensive significance testing."""
        significance_results = {}
        
        mechanisms = list(performance_data.keys())
        
        # Perform pairwise comparisons
        for i, mech1 in enumerate(mechanisms):
            for j, mech2 in enumerate(mechanisms[i+1:], i+1):
                if len(performance_data[mech1]) > 1 and len(performance_data[mech2]) > 1:
                    comparison_key = f"{mech1}_vs_{mech2}"
                    
                    try:
                        # Parametric tests
                        t_stat, t_pvalue = ttest_ind(
                            performance_data[mech1], 
                            performance_data[mech2],
                            equal_var=False
                        )
                        
                        # Non-parametric tests
                        u_stat, u_pvalue = mannwhitneyu(
                            performance_data[mech1], 
                            performance_data[mech2],
                            alternative='two-sided'
                        )
                        
                        # Effect size
                        cohens_d = self._calculate_cohens_d(
                            performance_data[mech1], 
                            performance_data[mech2]
                        )
                        
                        significance_results[comparison_key] = {
                            't_test': {
                                'statistic': t_stat,
                                'p_value': t_pvalue,
                                'significant': t_pvalue < self.config.significance_level
                            },
                            'mann_whitney_u': {
                                'statistic': u_stat,
                                'p_value': u_pvalue,
                                'significant': u_pvalue < self.config.significance_level
                            },
                            'cohens_d': cohens_d,
                            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                        }
                        
                    except Exception as e:
                        logger.warning(f"Significance testing failed for {comparison_key}: {e}")
        
        # Perform omnibus tests
        if len(mechanisms) > 2:
            try:
                # Kruskal-Wallis test
                all_scores = []
                groups = []
                for i, (mechanism, scores) in enumerate(performance_data.items()):
                    if scores:
                        all_scores.extend(scores)
                        groups.extend([i] * len(scores))
                
                if all_scores and groups:
                    h_stat, h_pvalue = kruskal(*[performance_data[mech] for mech in mechanisms if performance_data[mech]])
                    
                    significance_results['omnibus_test'] = {
                        'kruskal_wallis': {
                            'statistic': h_stat,
                            'p_value': h_pvalue,
                            'significant': h_pvalue < self.config.significance_level
                        }
                    }
            except Exception as e:
                logger.warning(f"Omnibus testing failed: {e}")
        
        return significance_results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        # Pooled standard deviation
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_effect_sizes(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate comprehensive effect sizes."""
        effect_sizes = {}
        
        mechanisms = list(performance_data.keys())
        
        for mech1 in mechanisms:
            for mech2 in mechanisms:
                if mech1 != mech2 and len(performance_data[mech1]) > 1 and len(performance_data[mech2]) > 1:
                    try:
                        # Cohen's d
                        cohens_d = self._calculate_cohens_d(performance_data[mech1], performance_data[mech2])
                        
                        # Hedges' g (bias-corrected)
                        n1, n2 = len(performance_data[mech1]), len(performance_data[mech2])
                        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
                        hedges_g = cohens_d * correction_factor
                        
                        # Glass's delta
                        glass_delta = (np.mean(performance_data[mech1]) - np.mean(performance_data[mech2])) / np.std(performance_data[mech2])
                        
                        effect_sizes[f"{mech1}_vs_{mech2}"] = {
                            'cohens_d': cohens_d,
                            'hedges_g': hedges_g,
                            'glass_delta': glass_delta,
                            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                        }
                    except Exception as e:
                        logger.warning(f"Effect size calculation failed for {mech1} vs {mech2}: {e}")
        
        return effect_sizes
    
    def _perform_power_analysis(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        power_analysis = {}
        
        mechanisms = list(performance_data.keys())
        
        for i, mech1 in enumerate(mechanisms):
            for j, mech2 in enumerate(mechanisms[i+1:], i+1):
                if len(performance_data[mech1]) > 1 and len(performance_data[mech2]) > 1:
                    try:
                        # Calculate effect size
                        cohens_d = self._calculate_cohens_d(performance_data[mech1], performance_data[mech2])
                        
                        # Sample sizes
                        n1, n2 = len(performance_data[mech1]), len(performance_data[mech2])
                        
                        # Calculate power
                        power = ttest_power(
                            effect_size=abs(cohens_d),
                            nobs1=n1,
                            alpha=self.config.significance_level,
                            alternative='two-sided'
                        )
                        
                        # Calculate required sample size for 80% power
                        required_n = self._calculate_required_sample_size(
                            effect_size=abs(cohens_d),
                            power=0.8,
                            alpha=self.config.significance_level
                        )
                        
                        power_analysis[f"{mech1}_vs_{mech2}"] = {
                            'current_power': power,
                            'effect_size': cohens_d,
                            'current_n1': n1,
                            'current_n2': n2,
                            'required_n_for_80_power': required_n,
                            'power_adequate': power >= 0.8
                        }
                    except Exception as e:
                        logger.warning(f"Power analysis failed for {mech1} vs {mech2}: {e}")
        
        return power_analysis
    
    def _calculate_required_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for desired power."""
        try:
            # Use statsmodels for sample size calculation
            from statsmodels.stats.power import ttest_power
            
            # Binary search for required sample size
            n_low, n_high = 10, 1000
            
            while n_high - n_low > 1:
                n_mid = (n_low + n_high) // 2
                current_power = ttest_power(effect_size, n_mid, alpha, alternative='two-sided')
                
                if current_power < power:
                    n_low = n_mid
                else:
                    n_high = n_mid
            
            return n_high
        except:
            return 100  # Default fallback
    
    def _assess_robustness(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess robustness of different mechanisms."""
        robustness_assessment = {}
        
        for mechanism, scores in performance_data.items():
            if scores:
                scores_array = np.array(scores)
                
                # Calculate robustness metrics
                cv = np.std(scores_array) / np.mean(scores_array) if np.mean(scores_array) > 0 else 0
                
                # Outlier detection using IQR method
                q1, q3 = np.percentile(scores_array, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = scores_array[(scores_array < lower_bound) | (scores_array > upper_bound)]
                
                # Robustness rating
                if cv < 0.05:
                    robustness_rating = "excellent"
                elif cv < 0.1:
                    robustness_rating = "good"
                elif cv < 0.2:
                    robustness_rating = "moderate"
                else:
                    robustness_rating = "poor"
                
                robustness_assessment[mechanism] = {
                    'coefficient_of_variation': cv,
                    'robustness_rating': robustness_rating,
                    'n_outliers': len(outliers),
                    'outlier_percentage': len(outliers) / len(scores_array) * 100,
                    'iqr': iqr,
                    'outlier_bounds': (lower_bound, upper_bound)
                }
        
        return robustness_assessment
    
    def _perform_bootstrap_analysis(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform bootstrap analysis for confidence intervals."""
        bootstrap_results = {}
        
        for mechanism, scores in performance_data.items():
            if scores and len(scores) > 1:
                # Bootstrap sampling
                bootstrap_means = []
                for _ in range(self.config.bootstrap_samples):
                    bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                bootstrap_means = np.array(bootstrap_means)
                
                # Calculate confidence intervals
                alpha = 1 - self.config.confidence_level
                ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
                ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
                
                bootstrap_results[mechanism] = {
                    'bootstrap_mean': np.mean(bootstrap_means),
                    'bootstrap_std': np.std(bootstrap_means),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'confidence_level': self.config.confidence_level
                }
        
        return bootstrap_results
    
    def _analyze_outliers(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze outliers in performance data."""
        outlier_analysis = {}
        
        for mechanism, scores in performance_data.items():
            if scores and len(scores) > 3:
                scores_array = np.array(scores)
                
                # IQR method
                q1, q3 = np.percentile(scores_array, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = scores_array[(scores_array < lower_bound) | (scores_array > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(scores_array))
                z_outliers = scores_array[z_scores > 3]
                
                # Modified Z-score method
                median = np.median(scores_array)
                mad = np.median(np.abs(scores_array - median))
                modified_z_scores = 0.6745 * (scores_array - median) / mad
                modified_z_outliers = scores_array[np.abs(modified_z_scores) > 3.5]
                
                outlier_analysis[mechanism] = {
                    'iqr_outliers': {
                        'count': len(iqr_outliers),
                        'indices': np.where((scores_array < lower_bound) | (scores_array > upper_bound))[0].tolist(),
                        'values': iqr_outliers.tolist()
                    },
                    'z_score_outliers': {
                        'count': len(z_outliers),
                        'indices': np.where(z_scores > 3)[0].tolist(),
                        'values': z_outliers.tolist()
                    },
                    'modified_z_outliers': {
                        'count': len(modified_z_outliers),
                        'indices': np.where(np.abs(modified_z_scores) > 3.5)[0].tolist(),
                        'values': modified_z_outliers.tolist()
                    }
                }
        
        return outlier_analysis
    
    def _perform_cross_dataset_analysis(self, experimental_data: Dict[str, Any]) -> None:
        """Perform cross-dataset statistical analysis."""
        logger.info("Performing cross-dataset analysis")
        
        # Collect all performance data across datasets
        all_performance_data = {'threshold': [], 'parametric': [], 'sigmoid': []}
        
        for dataset_results in experimental_data.values():
            performance_data = self._extract_performance_data(dataset_results)
            for mechanism, scores in performance_data.items():
                all_performance_data[mechanism].extend(scores)
        
        # Perform cross-dataset significance testing
        self.analysis_results['cross_dataset_analysis'] = {
            'descriptive_statistics': self._calculate_descriptive_statistics(all_performance_data),
            'significance_testing': self._perform_significance_testing(all_performance_data),
            'effect_size_analysis': self._calculate_effect_sizes(all_performance_data)
        }
    
    def _apply_multiple_comparison_correction(self) -> None:
        """Apply multiple comparison correction to p-values."""
        logger.info("Applying multiple comparison correction")
        
        # Collect all p-values
        all_p_values = []
        p_value_locations = []
        
        for dataset_name, significance_results in self.analysis_results['significance_testing'].items():
            for comparison, results in significance_results.items():
                if 't_test' in results:
                    all_p_values.append(results['t_test']['p_value'])
                    p_value_locations.append((dataset_name, comparison, 't_test'))
                
                if 'mann_whitney_u' in results:
                    all_p_values.append(results['mann_whitney_u']['p_value'])
                    p_value_locations.append((dataset_name, comparison, 'mann_whitney_u'))
        
        if all_p_values:
            # Apply correction
            if self.config.multiple_comparison_correction == 'bonferroni':
                corrected_p_values = multipletests(all_p_values, method='bonferroni')[1]
            elif self.config.multiple_comparison_correction == 'fdr_bh':
                corrected_p_values = multipletests(all_p_values, method='fdr_bh')[1]
            else:
                corrected_p_values = all_p_values
            
            # Update results with corrected p-values
            for i, (dataset_name, comparison, test_type) in enumerate(p_value_locations):
                if dataset_name in self.analysis_results['significance_testing']:
                    if comparison in self.analysis_results['significance_testing'][dataset_name]:
                        if test_type in self.analysis_results['significance_testing'][dataset_name][comparison]:
                            self.analysis_results['significance_testing'][dataset_name][comparison][test_type]['corrected_p_value'] = corrected_p_values[i]
                            self.analysis_results['significance_testing'][dataset_name][comparison][test_type]['significant_after_correction'] = corrected_p_values[i] < self.config.significance_level
    
    def generate_statistical_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical report."""
        report = {
            'analysis_summary': self._generate_analysis_summary(),
            'key_findings': self._generate_key_findings(),
            'recommendations': self._generate_statistical_recommendations(),
            'methodology': self._describe_methodology()
        }
        
        return report
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate summary of statistical analysis."""
        summary = {
            'total_comparisons': 0,
            'significant_comparisons': 0,
            'effect_size_distribution': {},
            'robustness_ranking': {}
        }
        
        # Count comparisons
        for dataset_results in self.analysis_results['significance_testing'].values():
            for comparison_results in dataset_results.values():
                if 't_test' in comparison_results:
                    summary['total_comparisons'] += 1
                    if comparison_results['t_test'].get('significant', False):
                        summary['significant_comparisons'] += 1
        
        return summary
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from statistical analysis."""
        findings = []
        
        # Analyze significance patterns
        significant_mechanisms = set()
        for dataset_results in self.analysis_results['significance_testing'].values():
            for comparison, results in dataset_results.items():
                if 't_test' in results and results['t_test'].get('significant', False):
                    significant_mechanisms.add(comparison)
        
        if significant_mechanisms:
            findings.append(f"Found {len(significant_mechanisms)} statistically significant differences between mechanisms")
        
        # Analyze effect sizes
        large_effects = 0
        for dataset_results in self.analysis_results['effect_size_analysis'].values():
            for comparison_results in dataset_results.values():
                if comparison_results.get('effect_size_interpretation') == 'large':
                    large_effects += 1
        
        if large_effects > 0:
            findings.append(f"Found {large_effects} comparisons with large effect sizes")
        
        return findings
    
    def _generate_statistical_recommendations(self) -> List[str]:
        """Generate recommendations based on statistical analysis."""
        recommendations = []
        
        recommendations.append("Consider increasing sample size for comparisons with low statistical power")
        recommendations.append("Investigate mechanisms with large effect sizes for practical significance")
        recommendations.append("Use robust statistical methods for datasets with high outlier rates")
        recommendations.append("Apply multiple comparison correction when conducting multiple tests")
        
        return recommendations
    
    def _describe_methodology(self) -> Dict[str, Any]:
        """Describe the statistical methodology used."""
        methodology = {
            'significance_testing': {
                'parametric_test': 'Welch\'s t-test (unequal variances)',
                'non_parametric_test': 'Mann-Whitney U test',
                'omnibus_test': 'Kruskal-Wallis test',
                'significance_level': self.config.significance_level
            },
            'effect_size_measures': {
                'cohens_d': 'Standardized mean difference',
                'hedges_g': 'Bias-corrected Cohen\'s d',
                'glass_delta': 'Effect size using control group standard deviation'
            },
            'multiple_comparison_correction': {
                'method': self.config.multiple_comparison_correction,
                'description': 'Controls family-wise error rate across multiple comparisons'
            },
            'bootstrap_analysis': {
                'n_samples': self.config.bootstrap_samples,
                'confidence_level': self.config.confidence_level
            }
        }
        
        return methodology

# Convenience function for statistical analysis
def perform_advanced_statistical_analysis(
    experimental_data: Dict[str, Any],
    significance_level: float = 0.05,
    confidence_level: float = 0.95,
    multiple_comparison_correction: str = 'bonferroni'
) -> Dict[str, Any]:
    """
    Convenience function for performing advanced statistical analysis.
    
    Args:
        experimental_data: Dictionary containing experimental results
        significance_level: Significance level for hypothesis testing
        confidence_level: Confidence level for interval estimation
        multiple_comparison_correction: Method for multiple comparison correction
        
    Returns:
        Dictionary containing comprehensive statistical analysis results
    """
    config = StatisticalAnalysisConfig(
        significance_level=significance_level,
        confidence_level=confidence_level,
        multiple_comparison_correction=multiple_comparison_correction
    )
    
    analyzer = AdvancedStatisticalAnalyzer(config)
    return analyzer.perform_comprehensive_analysis(experimental_data)

if __name__ == "__main__":
    # Example usage
    print("Advanced Statistical Analyzer - Example Usage")
    print("Ready for comprehensive statistical analysis!")
