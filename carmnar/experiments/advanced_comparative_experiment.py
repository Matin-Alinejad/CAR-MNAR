"""
Advanced Comparative Experimental Framework for MNAR Mechanisms

This module implements a comprehensive comparative analysis framework for evaluating
SM-MVPC's performance under different MNAR mechanisms: tail-driven (threshold and
parametric) versus sigmoid-based MNAR baseline. The framework includes Monte Carlo
simulations, statistical significance testing, and multi-dimensional robustness assessment.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..data_generation.heavy_tailed_mnar_generator import (
    HeavyTailedMNARGenerator, TailMNARConfig,
    create_threshold_mnar_config, create_parametric_mnar_config
)
from ..data_generation.mnar_generator import MNARGenerator, MNARConfig
from ..algorithms.causal_discovery import SMMVPCCausalDiscovery
from ..evaluation.metrics import CausalDiscoveryMetrics
from ..utils.data_loader import load_medical_datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparativeExperimentConfig:
    """Configuration for advanced comparative experiments."""
    # Datasets
    datasets: List[str]
    
    # Missingness proportions (5% to 50%)
    missingness_proportions: List[float]
    
    # Monte Carlo parameters
    n_monte_carlo_runs: int = 100
    random_seeds: List[int] = None
    
    # Sample size variations
    sample_size_factors: List[float] = None
    
    # Tail-driven MNAR parameters
    quantile_thresholds: List[float] = None
    tail_indices: List[float] = None
    
    # Sigmoid-based MNAR parameters
    sigmoid_steepness_values: List[float] = None
    sigmoid_center_values: List[float] = None
    
    # Statistical analysis
    confidence_level: float = 0.95
    significance_level: float = 0.05
    multiple_comparison_correction: str = 'bonferroni'
    
    # Output configuration
    output_dir: str = "results/comparative_analysis"
    save_intermediate_results: bool = True
    generate_plots: bool = True

class AdvancedComparativeExperiment:
    """
    Advanced comparative experimental framework for MNAR mechanism evaluation.
    
    This class implements a comprehensive framework for comparing different MNAR
    mechanisms through rigorous statistical analysis, Monte Carlo simulations,
    and multi-dimensional robustness assessment.
    """
    
    def __init__(self, config: ComparativeExperimentConfig):
        """
        Initialize the advanced comparative experiment.
        
        Args:
            config: Configuration object specifying experimental parameters
        """
        self.config = config
        self.metrics_calculator = CausalDiscoveryMetrics()
        
        # Set default values
        if self.config.random_seeds is None:
            self.config.random_seeds = list(range(self.config.n_monte_carlo_runs))
        
        if self.config.sample_size_factors is None:
            self.config.sample_size_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        if self.config.quantile_thresholds is None:
            self.config.quantile_thresholds = [85, 90, 95, 99]
        
        if self.config.tail_indices is None:
            self.config.tail_indices = [0.1, 0.5, 1.0, 2.0]
        
        if self.config.sigmoid_steepness_values is None:
            self.config.sigmoid_steepness_values = [0.5, 1.0, 2.0, 5.0]
        
        if self.config.sigmoid_center_values is None:
            self.config.sigmoid_center_values = [0.3, 0.5, 0.7]
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.experimental_results = {}
        self.statistical_analysis = {}
        self.robustness_metrics = {}
        
        logger.info(f"Initialized Advanced Comparative Experiment")
        logger.info(f"Datasets: {self.config.datasets}")
        logger.info(f"Missingness proportions: {self.config.missingness_proportions}")
        logger.info(f"Monte Carlo runs: {self.config.n_monte_carlo_runs}")
    
    def run_comprehensive_comparative_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive comparative analysis across all MNAR mechanisms.
        
        Returns:
            Dictionary containing all experimental results and analysis
        """
        logger.info("Starting comprehensive comparative analysis")
        start_time = time.time()
        
        # Load datasets and generate ground truth
        logger.info("Loading datasets and generating ground truth graphs")
        dataset_info = self._load_datasets_and_ground_truth()
        
        # Run experiments for each dataset
        for dataset_name, dataset_data in dataset_info.items():
            logger.info(f"Running comparative experiments for dataset: {dataset_name}")
            
            # Run tail-driven MNAR experiments
            tail_driven_results = self._run_tail_driven_experiments(dataset_name, dataset_data)
            
            # Run sigmoid-based MNAR experiments
            sigmoid_results = self._run_sigmoid_experiments(dataset_name, dataset_data)
            
            # Store results
            self.experimental_results[dataset_name] = {
                'tail_driven_results': tail_driven_results,
                'sigmoid_results': sigmoid_results,
                'dataset_info': dataset_data['info']
            }
        
        # Generate comprehensive statistical analysis
        logger.info("Generating comprehensive statistical analysis")
        self._generate_statistical_analysis()
        
        # Generate robustness assessment
        logger.info("Generating robustness assessment")
        self._generate_robustness_assessment()
        
        # Generate comparative visualizations
        if self.config.generate_plots:
            logger.info("Generating comparative visualizations")
            self._generate_comparative_visualizations()
        
        # Save results
        self._save_experimental_results()
        
        # Generate final report
        final_report = self._generate_final_report()
        
        end_time = time.time()
        logger.info(f"Comparative analysis completed in {end_time - start_time:.2f} seconds")
        
        return final_report
    
    def _load_datasets_and_ground_truth(self) -> Dict[str, Dict]:
        """Load datasets and generate ground truth causal graphs."""
        dataset_info = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            
            # Load dataset
            data = load_medical_datasets()[dataset_name]
            
            # Generate ground truth using PC algorithm
            ground_truth = self._generate_ground_truth_graph(data, dataset_name)
            
            # Calculate dataset statistics
            dataset_stats = self._calculate_dataset_statistics(data)
            
            dataset_info[dataset_name] = {
                'data': data,
                'ground_truth': ground_truth,
                'info': dataset_stats
            }
        
        return dataset_info
    
    def _generate_ground_truth_graph(self, data: pd.DataFrame, dataset_name: str) -> nx.DiGraph:
        """Generate ground truth causal graph using PC algorithm."""
        algorithm = SMMVPCCausalDiscovery()
        missing_mask = np.zeros(len(data), dtype=bool)
        ground_truth = algorithm.discover_causal_structure(data, missing_mask)
        
        logger.info(f"Generated ground truth for {dataset_name}: {ground_truth.number_of_nodes()} nodes, {ground_truth.number_of_edges()} edges")
        return ground_truth
    
    def _calculate_dataset_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        stats = {
            'n_samples': len(data),
            'n_variables': len(data.columns),
            'variable_names': list(data.columns),
            'missing_data_rate': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'data_types': {col: str(data[col].dtype) for col in data.columns},
            'basic_statistics': data.describe().to_dict()
        }
        
        # Calculate tail statistics for each variable
        tail_stats = {}
        for col in data.columns:
            values = data[col].dropna().values
            if len(values) > 0:
                tail_stats[col] = {
                    'tail_index_95': self._estimate_tail_index(values, 95),
                    'tail_index_99': self._estimate_tail_index(values, 99),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                }
        
        stats['tail_statistics'] = tail_stats
        return stats
    
    def _estimate_tail_index(self, values: np.ndarray, percentile: float) -> Optional[float]:
        """Estimate tail index using Hill estimator."""
        if len(values) < 10:
            return None
        
        try:
            threshold = np.percentile(values, percentile)
            tail_values = values[values > threshold]
            
            if len(tail_values) < 5:
                return None
            
            sorted_values = np.sort(tail_values)[::-1]
            k = max(3, len(sorted_values) // 3)
            top_values = sorted_values[:k]
            
            log_ratios = np.log(top_values[:-1] / top_values[1:])
            tail_index = np.mean(log_ratios)
            
            return tail_index
        except:
            return None
    
    def _run_tail_driven_experiments(self, dataset_name: str, dataset_data: Dict) -> Dict[str, Any]:
        """Run tail-driven MNAR experiments (threshold and parametric)."""
        logger.info(f"Running tail-driven experiments for {dataset_name}")
        
        data = dataset_data['data']
        ground_truth = dataset_data['ground_truth']
        
        results = {
            'threshold_results': {},
            'parametric_results': {}
        }
        
        # Run threshold-based experiments
        for missing_pct in self.config.missingness_proportions:
            for quantile in self.config.quantile_thresholds:
                logger.info(f"Testing threshold MNAR: {missing_pct:.1%} missing, {quantile}th percentile")
                
                # Run Monte Carlo simulations
                mc_results = self._run_monte_carlo_simulations(
                    data, ground_truth, 'threshold', missing_pct, quantile=quantile
                )
                
                key = f"missing_{missing_pct:.1f}_quantile_{quantile}"
                results['threshold_results'][key] = mc_results
        
        # Run parametric experiments
        for missing_pct in self.config.missingness_proportions:
            for tail_index in self.config.tail_indices:
                logger.info(f"Testing parametric MNAR: {missing_pct:.1%} missing, tail_index={tail_index}")
                
                # Run Monte Carlo simulations
                mc_results = self._run_monte_carlo_simulations(
                    data, ground_truth, 'parametric', missing_pct, tail_index=tail_index
                )
                
                key = f"missing_{missing_pct:.1f}_tail_index_{tail_index:.2f}"
                results['parametric_results'][key] = mc_results
        
        return results
    
    def _run_sigmoid_experiments(self, dataset_name: str, dataset_data: Dict) -> Dict[str, Any]:
        """Run sigmoid-based MNAR experiments."""
        logger.info(f"Running sigmoid-based experiments for {dataset_name}")
        
        data = dataset_data['data']
        ground_truth = dataset_data['ground_truth']
        
        results = {}
        
        for missing_pct in self.config.missingness_proportions:
            for steepness in self.config.sigmoid_steepness_values:
                for center in self.config.sigmoid_center_values:
                    logger.info(f"Testing sigmoid MNAR: {missing_pct:.1%} missing, steepness={steepness}, center={center}")
                    
                    # Run Monte Carlo simulations
                    mc_results = self._run_monte_carlo_simulations(
                        data, ground_truth, 'sigmoid', missing_pct, 
                        steepness=steepness, center=center
                    )
                    
                    key = f"missing_{missing_pct:.1f}_steepness_{steepness}_center_{center:.1f}"
                    results[key] = mc_results
        
        return results
    
    def _run_monte_carlo_simulations(self, data: pd.DataFrame, ground_truth: nx.DiGraph,
                                   mechanism: str, missing_pct: float, **kwargs) -> Dict[str, Any]:
        """Run Monte Carlo simulations for a specific MNAR mechanism."""
        all_results = []
        
        for run_idx in range(self.config.n_monte_carlo_runs):
            try:
                # Set random seed for reproducibility
                np.random.seed(self.config.random_seeds[run_idx])
                
                # Generate missingness based on mechanism
                if mechanism == 'threshold':
                    config = create_threshold_mnar_config(
                        target_missing_rate=missing_pct,
                        quantile_threshold=kwargs['quantile'],
                        random_seed=self.config.random_seeds[run_idx]
                    )
                    generator = HeavyTailedMNARGenerator(config)
                    missing_mask, generation_info = generator.generate_missingness_mask(data)
                
                elif mechanism == 'parametric':
                    config = create_parametric_mnar_config(
                        target_missing_rate=missing_pct,
                        tail_index=kwargs['tail_index'],
                        random_seed=self.config.random_seeds[run_idx]
                    )
                    generator = HeavyTailedMNARGenerator(config)
                    missing_mask, generation_info = generator.generate_missingness_mask(data)
                
                elif mechanism == 'sigmoid':
                    config = MNARConfig(
                        target_missing_rate=missing_pct,
                        steepness=kwargs['steepness'],
                        center=kwargs['center'],
                        random_seed=self.config.random_seeds[run_idx]
                    )
                    generator = MNARGenerator(config)
                    missing_mask, generation_info = generator.generate_missingness_mask(data)
                
                # Apply missingness to data
                data_with_missing = data.copy()
                data_with_missing.loc[missing_mask, :] = np.nan
                
                # Run algorithm
                algorithm = SMMVPCCausalDiscovery()
                inferred_graph = algorithm.discover_causal_structure(data_with_missing, missing_mask)
                
                # Calculate metrics
                metrics = self.metrics_calculator.comprehensive_evaluation(ground_truth, inferred_graph)
                
                # Add generation info
                metrics.update(generation_info)
                metrics['run_id'] = run_idx
                metrics['mechanism'] = mechanism
                
                all_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Monte Carlo run {run_idx} failed: {e}")
                continue
        
        if not all_results:
            return {}
        
        # Aggregate results
        df_results = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        summary_stats = {}
        for col in df_results.columns:
            if df_results[col].dtype in ['float64', 'int64']:
                summary_stats[f"{col}_mean"] = df_results[col].mean()
                summary_stats[f"{col}_std"] = df_results[col].std()
                summary_stats[f"{col}_min"] = df_results[col].min()
                summary_stats[f"{col}_max"] = df_results[col].max()
                summary_stats[f"{col}_median"] = df_results[col].median()
                summary_stats[f"{col}_q25"] = df_results[col].quantile(0.25)
                summary_stats[f"{col}_q75"] = df_results[col].quantile(0.75)
            else:
                summary_stats[f"{col}_values"] = df_results[col].tolist()
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for col in df_results.columns:
            if df_results[col].dtype in ['float64', 'int64']:
                mean_val = df_results[col].mean()
                std_val = df_results[col].std()
                n = len(df_results[col])
                
                # 95% confidence interval
                se = std_val / np.sqrt(n) if n > 1 else 0
                t_val = stats.t.ppf(0.975, n-1) if n > 1 else 1.96
                ci_lower = mean_val - t_val * se
                ci_upper = mean_val + t_val * se
                
                confidence_intervals[f"{col}_ci_lower"] = ci_lower
                confidence_intervals[f"{col}_ci_upper"] = ci_upper
        
        return {
            'summary_statistics': summary_stats,
            'confidence_intervals': confidence_intervals,
            'raw_results': all_results,
            'n_successful_runs': len(all_results),
            'n_total_runs': self.config.n_monte_carlo_runs
        }
    
    def _generate_statistical_analysis(self) -> None:
        """Generate comprehensive statistical analysis."""
        logger.info("Generating statistical analysis")
        
        # Initialize analysis results
        self.statistical_analysis = {
            'mechanism_comparison': {},
            'significance_testing': {},
            'effect_size_analysis': {},
            'robustness_ranking': {}
        }
        
        # Compare mechanisms across datasets
        for dataset_name, results in self.experimental_results.items():
            logger.info(f"Analyzing statistical significance for {dataset_name}")
            
            # Extract performance metrics for each mechanism
            mechanism_performance = self._extract_mechanism_performance(results)
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(mechanism_performance)
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_sizes(mechanism_performance)
            
            # Store results
            self.statistical_analysis['mechanism_comparison'][dataset_name] = mechanism_performance
            self.statistical_analysis['significance_testing'][dataset_name] = statistical_tests
            self.statistical_analysis['effect_size_analysis'][dataset_name] = effect_sizes
    
    def _extract_mechanism_performance(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract performance metrics for each mechanism."""
        mechanism_performance = {
            'threshold': [],
            'parametric': [],
            'sigmoid': []
        }
        
        # Extract threshold results
        for key, result in results['tail_driven_results']['threshold_results'].items():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                mechanism_performance['threshold'].extend(f1_scores)
        
        # Extract parametric results
        for key, result in results['tail_driven_results']['parametric_results'].items():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                mechanism_performance['parametric'].extend(f1_scores)
        
        # Extract sigmoid results
        for key, result in results['sigmoid_results'].items():
            if 'raw_results' in result:
                f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                mechanism_performance['sigmoid'].extend(f1_scores)
        
        return mechanism_performance
    
    def _perform_statistical_tests(self, mechanism_performance: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        mechanisms = list(mechanism_performance.keys())
        
        # Perform pairwise t-tests
        for i, mech1 in enumerate(mechanisms):
            for j, mech2 in enumerate(mechanisms[i+1:], i+1):
                if len(mechanism_performance[mech1]) > 1 and len(mechanism_performance[mech2]) > 1:
                    try:
                        # Welch's t-test (unequal variances)
                        t_stat, p_value = stats.ttest_ind(
                            mechanism_performance[mech1], 
                            mechanism_performance[mech2],
                            equal_var=False
                        )
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (np.var(mechanism_performance[mech1], ddof=1) + 
                             np.var(mechanism_performance[mech2], ddof=1)) / 2
                        )
                        cohens_d = (np.mean(mechanism_performance[mech1]) - 
                                   np.mean(mechanism_performance[mech2])) / pooled_std
                        
                        tests[f"{mech1}_vs_{mech2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'significant': p_value < self.config.significance_level
                        }
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {mech1} vs {mech2}: {e}")
        
        return tests
    
    def _calculate_effect_sizes(self, mechanism_performance: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate effect sizes for mechanism comparisons."""
        effect_sizes = {}
        
        mechanisms = list(mechanism_performance.keys())
        
        for mech1 in mechanisms:
            for mech2 in mechanisms:
                if mech1 != mech2 and len(mechanism_performance[mech1]) > 1 and len(mechanism_performance[mech2]) > 1:
                    try:
                        # Cohen's d
                        pooled_std = np.sqrt(
                            (np.var(mechanism_performance[mech1], ddof=1) + 
                             np.var(mechanism_performance[mech2], ddof=1)) / 2
                        )
                        cohens_d = (np.mean(mechanism_performance[mech1]) - 
                                   np.mean(mechanism_performance[mech2])) / pooled_std
                        
                        # Hedges' g (bias-corrected)
                        n1, n2 = len(mechanism_performance[mech1]), len(mechanism_performance[mech2])
                        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
                        hedges_g = cohens_d * correction_factor
                        
                        effect_sizes[f"{mech1}_vs_{mech2}"] = {
                            'cohens_d': cohens_d,
                            'hedges_g': hedges_g,
                            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                        }
                    except Exception as e:
                        logger.warning(f"Effect size calculation failed for {mech1} vs {mech2}: {e}")
        
        return effect_sizes
    
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
    
    def _generate_robustness_assessment(self) -> None:
        """Generate comprehensive robustness assessment."""
        logger.info("Generating robustness assessment")
        
        self.robustness_metrics = {
            'missingness_sensitivity': {},
            'mechanism_stability': {},
            'performance_consistency': {},
            'outlier_analysis': {}
        }
        
        # Analyze missingness sensitivity
        for dataset_name, results in self.experimental_results.items():
            self.robustness_metrics['missingness_sensitivity'][dataset_name] = \
                self._analyze_missingness_sensitivity(results)
        
        # Analyze mechanism stability
        for dataset_name, results in self.experimental_results.items():
            self.robustness_metrics['mechanism_stability'][dataset_name] = \
                self._analyze_mechanism_stability(results)
    
    def _analyze_missingness_sensitivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity to missingness proportions."""
        sensitivity_analysis = {}
        
        # Analyze threshold mechanism
        threshold_sensitivity = []
        for key, result in results['tail_driven_results']['threshold_results'].items():
            if 'summary_statistics' in result:
                missing_pct = float(key.split('_')[1])
                f1_mean = result['summary_statistics'].get('edge_f1_score_mean', 0)
                threshold_sensitivity.append((missing_pct, f1_mean))
        
        if threshold_sensitivity:
            threshold_sensitivity.sort()
            sensitivity_analysis['threshold'] = {
                'missingness_levels': [x[0] for x in threshold_sensitivity],
                'performance_levels': [x[1] for x in threshold_sensitivity],
                'degradation_rate': self._calculate_degradation_rate(threshold_sensitivity)
            }
        
        return sensitivity_analysis
    
    def _calculate_degradation_rate(self, sensitivity_data: List[Tuple[float, float]]) -> float:
        """Calculate performance degradation rate."""
        if len(sensitivity_data) < 2:
            return 0.0
        
        # Linear regression to find degradation rate
        missingness = np.array([x[0] for x in sensitivity_data])
        performance = np.array([x[1] for x in sensitivity_data])
        
        slope, _ = np.polyfit(missingness, performance, 1)
        return -slope  # Negative slope indicates degradation
    
    def _analyze_mechanism_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability across different mechanisms."""
        stability_analysis = {}
        
        # Calculate coefficient of variation for each mechanism
        mechanisms = ['threshold', 'parametric', 'sigmoid']
        
        for mechanism in mechanisms:
            if mechanism in ['threshold', 'parametric']:
                mechanism_results = results['tail_driven_results'][f'{mechanism}_results']
            else:
                mechanism_results = results['sigmoid_results']
            
            f1_scores = []
            for result in mechanism_results.values():
                if 'raw_results' in result:
                    f1_scores.extend([r['edge_f1_score'] for r in result['raw_results']])
            
            if f1_scores:
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                cv = std_f1 / mean_f1 if mean_f1 > 0 else 0
                
                stability_analysis[mechanism] = {
                    'mean_performance': mean_f1,
                    'std_performance': std_f1,
                    'coefficient_of_variation': cv,
                    'stability_rating': 'high' if cv < 0.1 else 'medium' if cv < 0.2 else 'low'
                }
        
        return stability_analysis
    
    def _generate_comparative_visualizations(self) -> None:
        """Generate comprehensive comparative visualizations."""
        logger.info("Generating comparative visualizations")
        
        # Create visualization directory
        viz_dir = Path(self.config.output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate mechanism comparison plots
        self._create_mechanism_comparison_plots(viz_dir)
        
        # Generate statistical analysis plots
        self._create_statistical_analysis_plots(viz_dir)
        
        # Generate robustness assessment plots
        self._create_robustness_assessment_plots(viz_dir)
    
    def _create_mechanism_comparison_plots(self, output_dir: Path) -> None:
        """Create mechanism comparison visualization plots."""
        # Implementation for mechanism comparison plots
        pass
    
    def _create_statistical_analysis_plots(self, output_dir: Path) -> None:
        """Create statistical analysis visualization plots."""
        # Implementation for statistical analysis plots
        pass
    
    def _create_robustness_assessment_plots(self, output_dir: Path) -> None:
        """Create robustness assessment visualization plots."""
        # Implementation for robustness assessment plots
        pass
    
    def _save_experimental_results(self) -> None:
        """Save experimental results to files."""
        logger.info("Saving experimental results")
        
        # Save detailed results
        results_file = Path(self.config.output_dir) / "comparative_experimental_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.experimental_results, f, indent=2, default=str)
        
        # Save statistical analysis
        stats_file = Path(self.config.output_dir) / "statistical_analysis.json"
        with open(stats_file, 'w') as f:
            json.dump(self.statistical_analysis, f, indent=2, default=str)
        
        # Save robustness metrics
        robustness_file = Path(self.config.output_dir) / "robustness_metrics.json"
        with open(robustness_file, 'w') as f:
            json.dump(self.robustness_metrics, f, indent=2, default=str)
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        logger.info("Generating final report")
        
        report = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'datasets': self.config.datasets,
                'missingness_proportions': self.config.missingness_proportions,
                'n_monte_carlo_runs': self.config.n_monte_carlo_runs,
                'random_seeds': self.config.random_seeds[:5]  # Show first 5 seeds
            },
            'experimental_results': self.experimental_results,
            'statistical_analysis': self.statistical_analysis,
            'robustness_metrics': self.robustness_metrics,
            'summary_statistics': self._calculate_experiment_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save final report
        report_file = Path(self.config.output_dir) / "final_comparative_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_experiment_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for the entire experiment."""
        summary = {
            'total_conditions_tested': 0,
            'successful_conditions': 0,
            'overall_performance': {},
            'mechanism_ranking': {},
            'statistical_significance': {}
        }
        
        # Calculate overall statistics
        all_f1_scores = []
        mechanism_performance = {'threshold': [], 'parametric': [], 'sigmoid': []}
        
        for dataset_name, results in self.experimental_results.items():
            # Collect threshold results
            for result in results['tail_driven_results']['threshold_results'].values():
                if 'raw_results' in result:
                    f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                    all_f1_scores.extend(f1_scores)
                    mechanism_performance['threshold'].extend(f1_scores)
            
            # Collect parametric results
            for result in results['tail_driven_results']['parametric_results'].values():
                if 'raw_results' in result:
                    f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                    all_f1_scores.extend(f1_scores)
                    mechanism_performance['parametric'].extend(f1_scores)
            
            # Collect sigmoid results
            for result in results['sigmoid_results'].values():
                if 'raw_results' in result:
                    f1_scores = [r['edge_f1_score'] for r in result['raw_results']]
                    all_f1_scores.extend(f1_scores)
                    mechanism_performance['sigmoid'].extend(f1_scores)
        
        # Calculate overall performance
        if all_f1_scores:
            summary['overall_performance'] = {
                'mean_f1_score': np.mean(all_f1_scores),
                'std_f1_score': np.std(all_f1_scores),
                'min_f1_score': np.min(all_f1_scores),
                'max_f1_score': np.max(all_f1_scores)
            }
        
        # Calculate mechanism ranking
        for mechanism, scores in mechanism_performance.items():
            if scores:
                summary['mechanism_ranking'][mechanism] = {
                    'mean_performance': np.mean(scores),
                    'std_performance': np.std(scores),
                    'n_observations': len(scores)
                }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on experimental results."""
        recommendations = []
        
        # Analyze mechanism performance
        if 'mechanism_ranking' in self._calculate_experiment_summary():
            mechanism_ranking = self._calculate_experiment_summary()['mechanism_ranking']
            
            if mechanism_ranking:
                best_mechanism = max(mechanism_ranking.keys(), 
                                   key=lambda x: mechanism_ranking[x]['mean_performance'])
                
                recommendations.append(f"Best performing MNAR mechanism: {best_mechanism}")
                
                # Compare mechanisms
                if len(mechanism_ranking) > 1:
                    sorted_mechanisms = sorted(mechanism_ranking.keys(), 
                                             key=lambda x: mechanism_ranking[x]['mean_performance'], 
                                             reverse=True)
                    recommendations.append(f"Mechanism performance ranking: {' > '.join(sorted_mechanisms)}")
        
        recommendations.append("Consider mechanism-specific optimization for different datasets")
        recommendations.append("Evaluate computational cost vs performance trade-offs")
        recommendations.append("Investigate mechanism robustness under extreme missingness conditions")
        
        return recommendations

# Convenience function for running the experiment
def run_advanced_comparative_experiment(
    datasets: List[str] = None,
    missingness_proportions: List[float] = None,
    n_monte_carlo_runs: int = 100,
    output_dir: str = "results/comparative_analysis"
) -> Dict[str, Any]:
    """
    Convenience function for running the advanced comparative experiment.
    
    Args:
        datasets: List of dataset names to evaluate
        missingness_proportions: List of missingness proportions to test
        n_monte_carlo_runs: Number of Monte Carlo runs
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing comprehensive experimental results
    """
    config = ComparativeExperimentConfig(
        datasets=datasets or ['diabetes', 'heart_disease', 'hepatitis'],
        missingness_proportions=missingness_proportions or [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        n_monte_carlo_runs=n_monte_carlo_runs,
        output_dir=output_dir
    )
    
    experiment = AdvancedComparativeExperiment(config)
    return experiment.run_comprehensive_comparative_analysis()

if __name__ == "__main__":
    # Example usage
    print("Advanced Comparative Experiment - Example Usage")
    
    # Run experiment with default parameters
    results = run_advanced_comparative_experiment(
        datasets=['diabetes', 'heart_disease'],
        missingness_proportions=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_monte_carlo_runs=50
    )
    
    print("Comparative experiment completed successfully!")
