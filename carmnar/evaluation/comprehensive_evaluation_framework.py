"""
Comprehensive Evaluation Framework for Causal Discovery Algorithms

This module provides a comprehensive evaluation framework for causal discovery
algorithms, integrating structural metrics, diagnostic visualizations, and
performance analysis suitable for academic publication.

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

from .advanced_structural_metrics import AdvancedStructuralMetrics, StructuralMetricsConfig
from ..visualization.advanced_diagnostic_visualizer import AdvancedDiagnosticVisualizer
from ..algorithms.causal_discovery import SMMVPCCausalDiscovery
from ..data_generation.heavy_tailed_mnar_generator import HeavyTailedMNARGenerator
from ..data_generation.mnar_generator import MNARGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation framework."""
    # Structural metrics configuration
    structural_metrics_config: StructuralMetricsConfig = None
    
    # Evaluation parameters
    missingness_proportions: List[float] = None
    mechanisms: List[str] = None
    n_repetitions: int = 5
    
    # Output configuration
    output_dir: str = "results/comprehensive_evaluation"
    save_intermediate_results: bool = True
    generate_plots: bool = True
    
    # Statistical parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.structural_metrics_config is None:
            self.structural_metrics_config = StructuralMetricsConfig()
        
        if self.missingness_proportions is None:
            self.missingness_proportions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        if self.mechanisms is None:
            self.mechanisms = ['threshold', 'parametric', 'sigmoid']

class ComprehensiveEvaluationFramework:
    """
    Comprehensive evaluation framework for causal discovery algorithms.
    
    This class provides a complete evaluation framework integrating structural
    metrics, diagnostic visualizations, and performance analysis for academic
    evaluation of causal discovery algorithms under different MNAR mechanisms.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the comprehensive evaluation framework.
        
        Args:
            config: Configuration object for evaluation
        """
        self.config = config
        
        # Initialize components
        self.structural_metrics = AdvancedStructuralMetrics(config.structural_metrics_config)
        self.visualizer = AdvancedDiagnosticVisualizer(f"{config.output_dir}/diagnostic_plots")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.evaluation_results = {}
        self.diagnostic_data = {}
        self.performance_data = {}
        
        logger.info("Initialized Comprehensive Evaluation Framework")
        logger.info(f"Missingness proportions: {self.config.missingness_proportions}")
        logger.info(f"Mechanisms: {self.config.mechanisms}")
        logger.info(f"Number of repetitions: {self.config.n_repetitions}")
    
    def run_comprehensive_evaluation(self, 
                                   datasets: Dict[str, pd.DataFrame],
                                   ground_truths: Dict[str, nx.DiGraph]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all datasets and conditions.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            ground_truths: Dictionary mapping dataset names to ground truth graphs
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        start_time = time.time()
        
        # Run evaluation for each dataset
        for dataset_name, data in datasets.items():
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            ground_truth = ground_truths[dataset_name]
            
            # Run evaluation for this dataset
            dataset_results = self._evaluate_dataset(data, ground_truth, dataset_name)
            self.evaluation_results[dataset_name] = dataset_results
        
        # Generate comprehensive analysis
        logger.info("Generating comprehensive analysis")
        comprehensive_analysis = self._generate_comprehensive_analysis()
        
        # Generate diagnostic visualizations
        if self.config.generate_plots:
            logger.info("Generating diagnostic visualizations")
            self._generate_diagnostic_visualizations()
        
        # Save results
        self._save_evaluation_results()
        
        # Generate final report
        final_report = self._generate_final_report(comprehensive_analysis)
        
        end_time = time.time()
        logger.info(f"Comprehensive evaluation completed in {end_time - start_time:.2f} seconds")
        
        return final_report
    
    def _evaluate_dataset(self, 
                         data: pd.DataFrame, 
                         ground_truth: nx.DiGraph, 
                         dataset_name: str) -> Dict[str, Any]:
        """Evaluate a single dataset across all conditions."""
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        dataset_results = {
            'dataset_info': {
                'name': dataset_name,
                'n_samples': len(data),
                'n_variables': len(data.columns),
                'variable_names': list(data.columns)
            },
            'mechanism_results': {},
            'missingness_analysis': {},
            'performance_analysis': {},
            'diagnostic_data': {}
        }
        
        # Evaluate each mechanism
        for mechanism in self.config.mechanisms:
            logger.info(f"Evaluating mechanism: {mechanism}")
            
            mechanism_results = self._evaluate_mechanism(
                data, ground_truth, mechanism, dataset_name
            )
            dataset_results['mechanism_results'][mechanism] = mechanism_results
        
        # Analyze missingness patterns
        dataset_results['missingness_analysis'] = self._analyze_missingness_patterns(
            data, dataset_name
        )
        
        # Analyze performance patterns
        dataset_results['performance_analysis'] = self._analyze_performance_patterns(
            dataset_results['mechanism_results']
        )
        
        # Collect diagnostic data
        dataset_results['diagnostic_data'] = self._collect_diagnostic_data(
            data, ground_truth, dataset_results['mechanism_results']
        )
        
        return dataset_results
    
    def _evaluate_mechanism(self, 
                           data: pd.DataFrame, 
                           ground_truth: nx.DiGraph, 
                           mechanism: str, 
                           dataset_name: str) -> Dict[str, Any]:
        """Evaluate a specific mechanism across all missingness proportions."""
        mechanism_results = {
            'mechanism': mechanism,
            'missingness_results': {},
            'summary_statistics': {},
            'diagnostic_metrics': {}
        }
        
        # Evaluate each missingness proportion
        for missing_pct in self.config.missingness_proportions:
            logger.info(f"Evaluating {mechanism} mechanism at {missing_pct:.1%} missingness")
            
            missingness_results = self._evaluate_missingness_condition(
                data, ground_truth, mechanism, missing_pct, dataset_name
            )
            mechanism_results['missingness_results'][f"missing_{missing_pct:.1f}"] = missingness_results
        
        # Calculate summary statistics
        mechanism_results['summary_statistics'] = self._calculate_mechanism_summary(
            mechanism_results['missingness_results']
        )
        
        # Calculate diagnostic metrics
        mechanism_results['diagnostic_metrics'] = self._calculate_diagnostic_metrics(
            mechanism_results['missingness_results']
        )
        
        return mechanism_results
    
    def _evaluate_missingness_condition(self, 
                                      data: pd.DataFrame, 
                                      ground_truth: nx.DiGraph, 
                                      mechanism: str, 
                                      missing_pct: float, 
                                      dataset_name: str) -> Dict[str, Any]:
        """Evaluate a specific missingness condition with multiple repetitions."""
        missingness_results = {
            'missingness_proportion': missing_pct,
            'mechanism': mechanism,
            'repetitions': [],
            'summary_metrics': {},
            'diagnostic_data': {}
        }
        
        # Run multiple repetitions
        all_metrics = []
        for rep in range(self.config.n_repetitions):
            logger.debug(f"Repetition {rep + 1}/{self.config.n_repetitions}")
            
            try:
                # Generate missingness
                missing_mask, generation_info = self._generate_missingness(
                    data, mechanism, missing_pct, rep
                )
                
                # Apply missingness to data
                data_with_missing = data.copy()
                data_with_missing.loc[missing_mask, :] = np.nan
                
                # Run algorithm
                algorithm = SMMVPCCausalDiscovery()
                inferred_graph = algorithm.discover_causal_structure(data_with_missing, missing_mask)
                
                # Calculate structural metrics
                metrics = self.structural_metrics.calculate_comprehensive_metrics(
                    ground_truth, inferred_graph, f"{dataset_name}_{mechanism}_{missing_pct:.1f}_{rep}"
                )
                
                # Add repetition metadata
                metrics['repetition'] = rep
                metrics['missingness_proportion'] = missing_pct
                metrics['mechanism'] = mechanism
                metrics['generation_info'] = generation_info
                
                missingness_results['repetitions'].append(metrics)
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Repetition {rep} failed: {e}")
                continue
        
        # Calculate summary metrics
        if all_metrics:
            missingness_results['summary_metrics'] = self._calculate_missingness_summary(all_metrics)
            missingness_results['diagnostic_data'] = self._collect_missingness_diagnostic_data(
                data, missing_mask, all_metrics
            )
        
        return missingness_results
    
    def _generate_missingness(self, 
                            data: pd.DataFrame, 
                            mechanism: str, 
                            missing_pct: float, 
                            repetition: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate missingness based on the specified mechanism."""
        np.random.seed(repetition)
        
        if mechanism == 'threshold':
            # Threshold-based MNAR
            config = {
                'target_missing_rate': missing_pct,
                'quantile_threshold': 95,
                'random_seed': repetition
            }
            generator = HeavyTailedMNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        elif mechanism == 'parametric':
            # Parametric MNAR
            config = {
                'target_missing_rate': missing_pct,
                'tail_index': 1.0,
                'random_seed': repetition
            }
            generator = HeavyTailedMNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        elif mechanism == 'sigmoid':
            # Sigmoid-based MNAR
            config = {
                'target_missing_rate': missing_pct,
                'steepness': 2.0,
                'center': 0.5,
                'random_seed': repetition
            }
            generator = MNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return missing_mask, generation_info
    
    def _calculate_missingness_summary(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for a missingness condition."""
        if not all_metrics:
            return {}
        
        # Extract key metrics
        rel_shd_values = [m['basic_structural']['relative_structural_hamming_distance'] for m in all_metrics]
        f1_scores = [m['adjacency_metrics']['edge_f1_score'] for m in all_metrics]
        precision_scores = [m['adjacency_metrics']['edge_precision'] for m in all_metrics]
        recall_scores = [m['adjacency_metrics']['edge_recall'] for m in all_metrics]
        orientation_accuracy = [m['orientation_metrics']['orientation_accuracy'] for m in all_metrics]
        
        summary = {
            'n_repetitions': len(all_metrics),
            'rel_shd': {
                'mean': np.mean(rel_shd_values),
                'std': np.std(rel_shd_values),
                'min': np.min(rel_shd_values),
                'max': np.max(rel_shd_values),
                'median': np.median(rel_shd_values)
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores),
                'median': np.median(f1_scores)
            },
            'precision': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores),
                'min': np.min(precision_scores),
                'max': np.max(precision_scores),
                'median': np.median(precision_scores)
            },
            'recall': {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores),
                'min': np.min(recall_scores),
                'max': np.max(recall_scores),
                'median': np.median(recall_scores)
            },
            'orientation_accuracy': {
                'mean': np.mean(orientation_accuracy),
                'std': np.std(orientation_accuracy),
                'min': np.min(orientation_accuracy),
                'max': np.max(orientation_accuracy),
                'median': np.median(orientation_accuracy)
            }
        }
        
        return summary
    
    def _calculate_mechanism_summary(self, missingness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for a mechanism."""
        summary = {
            'missingness_conditions': len(missingness_results),
            'overall_performance': {},
            'degradation_analysis': {},
            'stability_analysis': {}
        }
        
        # Collect all metrics across missingness conditions
        all_rel_shd = []
        all_f1_scores = []
        missingness_levels = []
        
        for missing_key, missing_data in missingness_results.items():
            if 'summary_metrics' in missing_data:
                missing_pct = missing_data['missingness_proportion']
                missingness_levels.append(missing_pct)
                
                all_rel_shd.append(missing_data['summary_metrics']['rel_shd']['mean'])
                all_f1_scores.append(missing_data['summary_metrics']['f1_score']['mean'])
        
        # Calculate overall performance
        if all_rel_shd and all_f1_scores:
            summary['overall_performance'] = {
                'mean_rel_shd': np.mean(all_rel_shd),
                'std_rel_shd': np.std(all_rel_shd),
                'mean_f1_score': np.mean(all_f1_scores),
                'std_f1_score': np.std(all_f1_scores)
            }
            
            # Calculate degradation analysis
            if len(missingness_levels) > 1:
                # Linear regression to find degradation rate
                slope_rel_shd, _ = np.polyfit(missingness_levels, all_rel_shd, 1)
                slope_f1, _ = np.polyfit(missingness_levels, all_f1_scores, 1)
                
                summary['degradation_analysis'] = {
                    'rel_shd_degradation_rate': slope_rel_shd,
                    'f1_degradation_rate': -slope_f1,  # Negative because F1 decreases
                    'missingness_levels': missingness_levels,
                    'rel_shd_values': all_rel_shd,
                    'f1_values': all_f1_scores
                }
            
            # Calculate stability analysis
            summary['stability_analysis'] = {
                'rel_shd_cv': np.std(all_rel_shd) / np.mean(all_rel_shd) if np.mean(all_rel_shd) > 0 else 0,
                'f1_cv': np.std(all_f1_scores) / np.mean(all_f1_scores) if np.mean(all_f1_scores) > 0 else 0
            }
        
        return summary
    
    def _calculate_diagnostic_metrics(self, missingness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate diagnostic metrics for a mechanism."""
        diagnostic_metrics = {
            'performance_consistency': {},
            'outlier_analysis': {},
            'convergence_analysis': {}
        }
        
        # Collect all individual repetition results
        all_repetitions = []
        for missing_data in missingness_results.values():
            if 'repetitions' in missing_data:
                all_repetitions.extend(missing_data['repetitions'])
        
        if all_repetitions:
            # Extract metrics
            rel_shd_values = [r['basic_structural']['relative_structural_hamming_distance'] for r in all_repetitions]
            f1_values = [r['adjacency_metrics']['edge_f1_score'] for r in all_repetitions]
            
            # Performance consistency
            diagnostic_metrics['performance_consistency'] = {
                'rel_shd_cv': np.std(rel_shd_values) / np.mean(rel_shd_values) if np.mean(rel_shd_values) > 0 else 0,
                'f1_cv': np.std(f1_values) / np.mean(f1_values) if np.mean(f1_values) > 0 else 0,
                'rel_shd_range': np.max(rel_shd_values) - np.min(rel_shd_values),
                'f1_range': np.max(f1_values) - np.min(f1_values)
            }
            
            # Outlier analysis
            q1_rel_shd, q3_rel_shd = np.percentile(rel_shd_values, [25, 75])
            iqr_rel_shd = q3_rel_shd - q1_rel_shd
            outlier_threshold_rel_shd = 1.5 * iqr_rel_shd
            
            q1_f1, q3_f1 = np.percentile(f1_values, [25, 75])
            iqr_f1 = q3_f1 - q1_f1
            outlier_threshold_f1 = 1.5 * iqr_f1
            
            rel_shd_outliers = [x for x in rel_shd_values if x < q1_rel_shd - outlier_threshold_rel_shd or x > q3_rel_shd + outlier_threshold_rel_shd]
            f1_outliers = [x for x in f1_values if x < q1_f1 - outlier_threshold_f1 or x > q3_f1 + outlier_threshold_f1]
            
            diagnostic_metrics['outlier_analysis'] = {
                'rel_shd_outliers': len(rel_shd_outliers),
                'f1_outliers': len(f1_outliers),
                'rel_shd_outlier_rate': len(rel_shd_outliers) / len(rel_shd_values),
                'f1_outlier_rate': len(f1_outliers) / len(f1_values)
            }
        
        return diagnostic_metrics
    
    def _analyze_missingness_patterns(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyze missingness patterns in the dataset."""
        missingness_analysis = {
            'original_missingness': {
                'total_missing': data.isnull().sum().sum(),
                'missing_rate': data.isnull().sum().sum() / (len(data) * len(data.columns)),
                'per_variable': data.isnull().sum().to_dict()
            },
            'variable_analysis': {},
            'correlation_analysis': {}
        }
        
        # Analyze each variable
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                missingness_analysis['variable_analysis'][col] = {
                    'n_missing': data[col].isnull().sum(),
                    'missing_rate': data[col].isnull().sum() / len(data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
        
        return missingness_analysis
    
    def _analyze_performance_patterns(self, mechanism_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns across mechanisms."""
        performance_analysis = {
            'mechanism_comparison': {},
            'degradation_analysis': {},
            'stability_analysis': {}
        }
        
        # Compare mechanisms
        for mechanism, results in mechanism_results.items():
            if 'summary_statistics' in results:
                summary = results['summary_statistics']
                performance_analysis['mechanism_comparison'][mechanism] = {
                    'mean_rel_shd': summary.get('overall_performance', {}).get('mean_rel_shd', 0),
                    'mean_f1_score': summary.get('overall_performance', {}).get('mean_f1_score', 0),
                    'stability': summary.get('stability_analysis', {})
                }
        
        return performance_analysis
    
    def _collect_diagnostic_data(self, 
                               data: pd.DataFrame, 
                               ground_truth: nx.DiGraph, 
                               mechanism_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect diagnostic data for visualization."""
        diagnostic_data = {
            'dataset_info': {
                'n_samples': len(data),
                'n_variables': len(data.columns),
                'n_edges': ground_truth.number_of_edges()
            },
            'missingness_data': {},
            'performance_data': {}
        }
        
        # Collect missingness data
        for mechanism, results in mechanism_results.items():
            if 'missingness_results' in results:
                missingness_data = {}
                for missing_key, missing_data in results['missingness_results'].items():
                    if 'summary_metrics' in missing_data:
                        missing_pct = missing_data['missingness_proportion']
                        missingness_data[missing_pct] = {
                            'rel_shd': missing_data['summary_metrics']['rel_shd']['mean'],
                            'f1_score': missing_data['summary_metrics']['f1_score']['mean'],
                            'precision': missing_data['summary_metrics']['precision']['mean'],
                            'recall': missing_data['summary_metrics']['recall']['mean']
                        }
                diagnostic_data['missingness_data'][mechanism] = missingness_data
        
        # Collect performance data
        for mechanism, results in mechanism_results.items():
            if 'summary_statistics' in results:
                summary = results['summary_statistics']
                diagnostic_data['performance_data'][mechanism] = {
                    'degradation_analysis': summary.get('degradation_analysis', {}),
                    'stability_analysis': summary.get('stability_analysis', {})
                }
        
        return diagnostic_data
    
    def _collect_missingness_diagnostic_data(self, 
                                           data: pd.DataFrame, 
                                           missing_mask: np.ndarray, 
                                           all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect diagnostic data for a specific missingness condition."""
        diagnostic_data = {
            'missingness_pattern': {
                'missing_rate': np.mean(missing_mask),
                'missing_per_variable': missing_mask.sum(axis=0).tolist(),
                'missing_correlation': np.corrcoef(missing_mask.T) if missing_mask.shape[1] > 1 else np.array([[1.0]])
            },
            'performance_metrics': {
                'rel_shd_values': [m['basic_structural']['relative_structural_hamming_distance'] for m in all_metrics],
                'f1_values': [m['adjacency_metrics']['edge_f1_score'] for m in all_metrics]
            }
        }
        
        return diagnostic_data
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis across all datasets and mechanisms."""
        comprehensive_analysis = {
            'cross_dataset_analysis': {},
            'cross_mechanism_analysis': {},
            'overall_summary': {}
        }
        
        # Cross-dataset analysis
        all_datasets = list(self.evaluation_results.keys())
        comprehensive_analysis['cross_dataset_analysis'] = {
            'datasets': all_datasets,
            'n_datasets': len(all_datasets)
        }
        
        # Cross-mechanism analysis
        all_mechanisms = self.config.mechanisms
        comprehensive_analysis['cross_mechanism_analysis'] = {
            'mechanisms': all_mechanisms,
            'n_mechanisms': len(all_mechanisms)
        }
        
        # Overall summary
        comprehensive_analysis['overall_summary'] = {
            'total_experiments': len(all_datasets) * len(all_mechanisms) * len(self.config.missingness_proportions),
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_analysis
    
    def _generate_diagnostic_visualizations(self) -> None:
        """Generate diagnostic visualizations."""
        logger.info("Generating diagnostic visualizations")
        
        # Collect data for visualization
        experimental_results = self.evaluation_results
        missingness_data = {}
        performance_data = {}
        
        for dataset_name, dataset_results in experimental_results.items():
            missingness_data[dataset_name] = dataset_results.get('missingness_analysis', {})
            performance_data[dataset_name] = dataset_results.get('performance_analysis', {})
        
        # Create visualizations
        plot_paths = self.visualizer.create_comprehensive_diagnostic_plots(
            experimental_results, missingness_data, performance_data
        )
        
        # Store plot paths
        self.diagnostic_plots = plot_paths
        
        logger.info(f"Generated {len(plot_paths)} diagnostic plots")
    
    def _save_evaluation_results(self) -> None:
        """Save evaluation results to files."""
        logger.info("Saving evaluation results")
        
        # Save detailed results
        results_file = Path(self.config.output_dir) / "comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save diagnostic data
        if hasattr(self, 'diagnostic_plots'):
            diagnostic_file = Path(self.config.output_dir) / "diagnostic_plots.json"
            with open(diagnostic_file, 'w') as f:
                json.dump(self.diagnostic_plots, f, indent=2, default=str)
    
    def _generate_final_report(self, comprehensive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        logger.info("Generating final report")
        
        report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'missingness_proportions': self.config.missingness_proportions,
                'mechanisms': self.config.mechanisms,
                'n_repetitions': self.config.n_repetitions,
                'datasets': list(self.evaluation_results.keys())
            },
            'evaluation_results': self.evaluation_results,
            'comprehensive_analysis': comprehensive_analysis,
            'key_findings': self._generate_key_findings(),
            'recommendations': self._generate_recommendations(),
            'methodology': self._describe_methodology()
        }
        
        # Save final report
        report_file = Path(self.config.output_dir) / "comprehensive_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from evaluation results."""
        findings = []
        
        # Analyze mechanism performance
        mechanism_performance = {}
        for dataset_name, dataset_results in self.evaluation_results.items():
            for mechanism, results in dataset_results.get('mechanism_results', {}).items():
                if 'summary_statistics' in results:
                    summary = results['summary_statistics']
                    if mechanism not in mechanism_performance:
                        mechanism_performance[mechanism] = []
                    
                    mechanism_performance[mechanism].append({
                        'mean_rel_shd': summary.get('overall_performance', {}).get('mean_rel_shd', 0),
                        'mean_f1_score': summary.get('overall_performance', {}).get('mean_f1_score', 0)
                    })
        
        # Find best performing mechanism
        if mechanism_performance:
            best_mechanism = None
            best_f1 = 0
            
            for mechanism, performances in mechanism_performance.items():
                avg_f1 = np.mean([p['mean_f1_score'] for p in performances])
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_mechanism = mechanism
            
            if best_mechanism:
                findings.append(f"Best performing mechanism: {best_mechanism} (avg F1: {best_f1:.3f})")
        
        findings.append("Comprehensive evaluation completed successfully")
        findings.append("Structural metrics provide detailed performance assessment")
        findings.append("Diagnostic visualizations enable deep understanding of algorithm behavior")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        recommendations.append("Use relSHD as primary structural similarity metric")
        recommendations.append("Consider adjacency metrics for edge detection evaluation")
        recommendations.append("Evaluate orientation accuracy for causal direction assessment")
        recommendations.append("Monitor performance degradation with increasing missingness")
        recommendations.append("Use diagnostic visualizations for algorithm behavior analysis")
        
        return recommendations
    
    def _describe_methodology(self) -> Dict[str, Any]:
        """Describe the evaluation methodology."""
        methodology = {
            'structural_metrics': {
                'rel_shd': 'Relative Structural Hamming Distance for overall graph dissimilarity',
                'adjacency_metrics': 'Precision, Recall, and F1-score for edge detection',
                'orientation_metrics': 'Accuracy metrics for edge direction assessment'
            },
            'evaluation_design': {
                'missingness_proportions': 'Systematic variation from 5% to 50%',
                'mechanisms': 'Threshold-based, parametric, and sigmoid-based MNAR',
                'repetitions': f'{self.config.n_repetitions} repetitions per condition'
            },
            'diagnostic_visualizations': {
                'missingness_heatmaps': 'Visualization of missing data patterns',
                'qq_plots': 'Quantile-quantile plots for tail behavior analysis',
                'degradation_curves': 'Performance vs missingness curves'
            }
        }
        
        return methodology

# Convenience function for running comprehensive evaluation
def run_comprehensive_evaluation(datasets: Dict[str, pd.DataFrame],
                               ground_truths: Dict[str, nx.DiGraph],
                               missingness_proportions: List[float] = None,
                               mechanisms: List[str] = None,
                               n_repetitions: int = 5,
                               output_dir: str = "results/comprehensive_evaluation") -> Dict[str, Any]:
    """
    Convenience function for running comprehensive evaluation.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        ground_truths: Dictionary mapping dataset names to ground truth graphs
        missingness_proportions: List of missingness proportions to test
        mechanisms: List of mechanisms to test
        n_repetitions: Number of repetitions per condition
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    config = EvaluationConfig(
        missingness_proportions=missingness_proportions,
        mechanisms=mechanisms,
        n_repetitions=n_repetitions,
        output_dir=output_dir
    )
    
    framework = ComprehensiveEvaluationFramework(config)
    return framework.run_comprehensive_evaluation(datasets, ground_truths)

if __name__ == "__main__":
    # Example usage
    print("Comprehensive Evaluation Framework - Example Usage")
    print("Ready for comprehensive causal discovery evaluation!")
