"""
Heavy-Tailed MNAR Robustness Evaluation Framework

This module provides comprehensive evaluation capabilities for assessing the
robustness of causal discovery algorithms under heavy-tailed MNAR mechanisms.
It extends the standard evaluation framework with specialized metrics for
tail behavior analysis and extreme value assessment.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .metrics import CausalDiscoveryMetrics
from ..data_generation.heavy_tailed_mnar_generator import HeavyTailedMNARGenerator, TailMNARConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RobustnessEvaluationConfig:
    """Configuration for robustness evaluation under heavy-tailed MNAR."""
    missing_rates: List[float]
    quantile_thresholds: List[float]
    tail_indices: List[float]
    n_repetitions: int = 5
    confidence_level: float = 0.95
    min_tail_samples: int = 50
    random_seed: Optional[int] = None
    save_plots: bool = True
    output_dir: str = "results/robustness_analysis"

class HeavyTailedRobustnessEvaluator:
    """
    Comprehensive evaluator for algorithm robustness under heavy-tailed MNAR.
    
    This class provides specialized evaluation capabilities for assessing how
    causal discovery algorithms perform when data is missing in the tails of
    distributions, which is particularly relevant for extreme value analysis
    and risk assessment scenarios.
    """
    
    def __init__(self, config: RobustnessEvaluationConfig):
        """
        Initialize the robustness evaluator.
        
        Args:
            config: Configuration object specifying evaluation parameters
        """
        self.config = config
        self.metrics_calculator = CausalDiscoveryMetrics()
        self.results = {}
        self.summary_statistics = {}
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def evaluate_algorithm_robustness(self, 
                                    algorithm_func,
                                    true_graph: nx.DiGraph,
                                    data: pd.DataFrame,
                                    algorithm_name: str = "SM-MVPC") -> Dict[str, Any]:
        """
        Evaluate algorithm robustness under heavy-tailed MNAR mechanisms.
        
        Args:
            algorithm_func: Function that takes (data, missing_mask) and returns inferred graph
            true_graph: Ground truth causal graph
            data: Complete dataset
            algorithm_name: Name of the algorithm being evaluated
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Starting robustness evaluation for {algorithm_name}")
        
        # Initialize results storage
        self.results[algorithm_name] = {
            'threshold_results': {},
            'parametric_results': {},
            'summary_metrics': {},
            'tail_analysis': {}
        }
        
        # Evaluate threshold-based MNAR
        logger.info("Evaluating threshold-based MNAR mechanisms")
        threshold_results = self._evaluate_threshold_mnar(algorithm_func, true_graph, data)
        self.results[algorithm_name]['threshold_results'] = threshold_results
        
        # Evaluate parametric MNAR
        logger.info("Evaluating parametric MNAR mechanisms")
        parametric_results = self._evaluate_parametric_mnar(algorithm_func, true_graph, data)
        self.results[algorithm_name]['parametric_results'] = parametric_results
        
        # Generate comprehensive analysis
        self._generate_robustness_analysis(algorithm_name)
        
        # Save results
        self._save_evaluation_results(algorithm_name)
        
        return self.results[algorithm_name]
    
    def _evaluate_threshold_mnar(self, algorithm_func, true_graph: nx.DiGraph, 
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate algorithm performance under threshold-based MNAR."""
        results = {}
        
        for missing_rate in self.config.missing_rates:
            for quantile in self.config.quantile_thresholds:
                logger.info(f"Evaluating threshold MNAR: {missing_rate:.1%} missing, {quantile}th percentile")
                
                # Generate multiple repetitions
                repetition_results = []
                for rep in range(self.config.n_repetitions):
                    # Create MNAR configuration
                    config = TailMNARConfig(
                        mechanism='threshold',
                        target_missing_rate=missing_rate,
                        quantile_threshold=quantile,
                        random_seed=self.config.random_seed + rep if self.config.random_seed else None
                    )
                    
                    # Generate missingness
                    generator = HeavyTailedMNARGenerator(config)
                    missing_mask, generation_info = generator.generate_missingness_mask(data)
                    
                    # Apply missingness to data
                    data_with_missing = data.copy()
                    data_with_missing.loc[missing_mask, :] = np.nan
                    
                    # Run algorithm
                    try:
                        inferred_graph = algorithm_func(data_with_missing, missing_mask)
                        
                        # Calculate metrics
                        metrics = self.metrics_calculator.comprehensive_evaluation(
                            true_graph, inferred_graph
                        )
                        
                        # Add tail-specific metrics
                        tail_metrics = self._calculate_tail_specific_metrics(
                            data, missing_mask, true_graph, inferred_graph
                        )
                        metrics.update(tail_metrics)
                        
                        # Add generation info
                        metrics.update(generation_info)
                        
                        repetition_results.append(metrics)
                        
                    except Exception as e:
                        logger.warning(f"Algorithm failed for repetition {rep}: {e}")
                        continue
                
                if repetition_results:
                    # Aggregate results
                    key = f"missing_{missing_rate:.1f}_quantile_{quantile}"
                    results[key] = self._aggregate_repetition_results(repetition_results)
                    results[key]['generation_info'] = generation_info
        
        return results
    
    def _evaluate_parametric_mnar(self, algorithm_func, true_graph: nx.DiGraph, 
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate algorithm performance under parametric MNAR."""
        results = {}
        
        for missing_rate in self.config.missing_rates:
            for tail_index in self.config.tail_indices:
                logger.info(f"Evaluating parametric MNAR: {missing_rate:.1%} missing, tail_index={tail_index}")
                
                # Generate multiple repetitions
                repetition_results = []
                for rep in range(self.config.n_repetitions):
                    # Create MNAR configuration
                    config = TailMNARConfig(
                        mechanism='parametric',
                        target_missing_rate=missing_rate,
                        tail_index=tail_index,
                        random_seed=self.config.random_seed + rep if self.config.random_seed else None
                    )
                    
                    # Generate missingness
                    generator = HeavyTailedMNARGenerator(config)
                    missing_mask, generation_info = generator.generate_missingness_mask(data)
                    
                    # Apply missingness to data
                    data_with_missing = data.copy()
                    data_with_missing.loc[missing_mask, :] = np.nan
                    
                    # Run algorithm
                    try:
                        inferred_graph = algorithm_func(data_with_missing, missing_mask)
                        
                        # Calculate metrics
                        metrics = self.metrics_calculator.comprehensive_evaluation(
                            true_graph, inferred_graph
                        )
                        
                        # Add tail-specific metrics
                        tail_metrics = self._calculate_tail_specific_metrics(
                            data, missing_mask, true_graph, inferred_graph
                        )
                        metrics.update(tail_metrics)
                        
                        # Add generation info
                        metrics.update(generation_info)
                        
                        repetition_results.append(metrics)
                        
                    except Exception as e:
                        logger.warning(f"Algorithm failed for repetition {rep}: {e}")
                        continue
                
                if repetition_results:
                    # Aggregate results
                    key = f"missing_{missing_rate:.1f}_tail_index_{tail_index:.2f}"
                    results[key] = self._aggregate_repetition_results(repetition_results)
                    results[key]['generation_info'] = generation_info
        
        return results
    
    def _calculate_tail_specific_metrics(self, data: pd.DataFrame, missing_mask: np.ndarray,
                                       true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate metrics specific to tail behavior analysis."""
        metrics = {}
        
        # Calculate tail statistics for each variable
        tail_threshold = 95  # 95th percentile
        tail_metrics_per_var = []
        
        for col in data.columns:
            values = data[col].values
            missing_col = missing_mask[data.index]
            
            # Calculate tail statistics
            tail_threshold_val = np.percentile(values, tail_threshold)
            tail_values = values[values > tail_threshold_val]
            tail_missing_rate = np.mean(missing_col[values > tail_threshold_val])
            
            # Calculate tail index
            tail_index = self._estimate_tail_index(tail_values) if len(tail_values) > 10 else None
            
            tail_metrics_per_var.append({
                'tail_missing_rate': tail_missing_rate,
                'tail_index': tail_index,
                'tail_threshold': tail_threshold_val,
                'n_tail_samples': len(tail_values)
            })
        
        # Aggregate tail metrics
        metrics['mean_tail_missing_rate'] = np.mean([m['tail_missing_rate'] for m in tail_metrics_per_var])
        metrics['mean_tail_index'] = np.mean([m['tail_index'] for m in tail_metrics_per_var if m['tail_index'] is not None])
        metrics['tail_consistency'] = np.std([m['tail_missing_rate'] for m in tail_metrics_per_var])
        
        # Calculate edge-specific tail metrics
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Focus on edges involving high-degree nodes (often in tails)
        node_degrees = dict(true_graph.degree())
        high_degree_threshold = np.percentile(list(node_degrees.values()), 75)
        high_degree_nodes = [n for n, d in node_degrees.items() if d >= high_degree_threshold]
        
        high_degree_edges = [(u, v) for u, v in true_edges if u in high_degree_nodes or v in high_degree_nodes]
        if high_degree_edges:
            high_degree_precision = len(set(high_degree_edges) & inferred_edges) / len(inferred_edges) if inferred_edges else 0
            high_degree_recall = len(set(high_degree_edges) & inferred_edges) / len(high_degree_edges)
            high_degree_f1 = 2 * high_degree_precision * high_degree_recall / (high_degree_precision + high_degree_recall) if (high_degree_precision + high_degree_recall) > 0 else 0
            
            metrics['high_degree_edge_precision'] = high_degree_precision
            metrics['high_degree_edge_recall'] = high_degree_recall
            metrics['high_degree_edge_f1'] = high_degree_f1
        
        return metrics
    
    def _estimate_tail_index(self, values: np.ndarray) -> Optional[float]:
        """Estimate tail index using Hill estimator."""
        if len(values) < 10:
            return None
        
        try:
            sorted_values = np.sort(values)[::-1]
            k = max(5, len(sorted_values) // 5)
            top_values = sorted_values[:k]
            
            log_ratios = np.log(top_values[:-1] / top_values[1:])
            tail_index = np.mean(log_ratios)
            
            return tail_index
        except:
            return None
    
    def _aggregate_repetition_results(self, repetition_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across repetitions."""
        if not repetition_results:
            return {}
        
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(repetition_results)
        
        # Calculate summary statistics
        summary = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
                summary[f"{col}_min"] = df[col].min()
                summary[f"{col}_max"] = df[col].max()
            else:
                summary[f"{col}_values"] = df[col].tolist()
        
        return summary
    
    def _generate_robustness_analysis(self, algorithm_name: str) -> None:
        """Generate comprehensive robustness analysis."""
        logger.info(f"Generating robustness analysis for {algorithm_name}")
        
        # Analyze threshold-based results
        threshold_analysis = self._analyze_threshold_results(algorithm_name)
        
        # Analyze parametric results
        parametric_analysis = self._analyze_parametric_results(algorithm_name)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(algorithm_name)
        
        # Store analysis results
        self.results[algorithm_name]['tail_analysis'] = {
            'threshold_analysis': threshold_analysis,
            'parametric_analysis': parametric_analysis,
            'comparative_analysis': comparative_analysis
        }
        
        # Generate visualizations
        if self.config.save_plots:
            self._generate_robustness_visualizations(algorithm_name)
    
    def _analyze_threshold_results(self, algorithm_name: str) -> Dict[str, Any]:
        """Analyze threshold-based MNAR results."""
        threshold_results = self.results[algorithm_name]['threshold_results']
        
        analysis = {
            'quantile_sensitivity': {},
            'missing_rate_impact': {},
            'tail_behavior_analysis': {}
        }
        
        # Analyze quantile sensitivity
        for quantile in self.config.quantile_thresholds:
            quantile_data = []
            for missing_rate in self.config.missing_rates:
                key = f"missing_{missing_rate:.1f}_quantile_{quantile}"
                if key in threshold_results:
                    quantile_data.append({
                        'missing_rate': missing_rate,
                        'f1_score': threshold_results[key].get('edge_f1_score_mean', 0),
                        'tail_missing_rate': threshold_results[key].get('mean_tail_missing_rate_mean', 0)
                    })
            
            if quantile_data:
                analysis['quantile_sensitivity'][quantile] = quantile_data
        
        return analysis
    
    def _analyze_parametric_results(self, algorithm_name: str) -> Dict[str, Any]:
        """Analyze parametric MNAR results."""
        parametric_results = self.results[algorithm_name]['parametric_results']
        
        analysis = {
            'tail_index_sensitivity': {},
            'missing_rate_impact': {},
            'gpd_parameter_analysis': {}
        }
        
        # Analyze tail index sensitivity
        for tail_index in self.config.tail_indices:
            tail_index_data = []
            for missing_rate in self.config.missing_rates:
                key = f"missing_{missing_rate:.1f}_tail_index_{tail_index:.2f}"
                if key in parametric_results:
                    tail_index_data.append({
                        'missing_rate': missing_rate,
                        'f1_score': parametric_results[key].get('edge_f1_score_mean', 0),
                        'tail_missing_rate': parametric_results[key].get('mean_tail_missing_rate_mean', 0)
                    })
            
            if tail_index_data:
                analysis['tail_index_sensitivity'][tail_index] = tail_index_data
        
        return analysis
    
    def _generate_comparative_analysis(self, algorithm_name: str) -> Dict[str, Any]:
        """Generate comparative analysis between mechanisms."""
        threshold_results = self.results[algorithm_name]['threshold_results']
        parametric_results = self.results[algorithm_name]['parametric_results']
        
        analysis = {
            'mechanism_comparison': {},
            'robustness_ranking': {},
            'tail_behavior_insights': {}
        }
        
        # Compare mechanisms at similar missing rates
        for missing_rate in self.config.missing_rates:
            threshold_f1 = []
            parametric_f1 = []
            
            # Collect threshold results
            for quantile in self.config.quantile_thresholds:
                key = f"missing_{missing_rate:.1f}_quantile_{quantile}"
                if key in threshold_results:
                    threshold_f1.append(threshold_results[key].get('edge_f1_score_mean', 0))
            
            # Collect parametric results
            for tail_index in self.config.tail_indices:
                key = f"missing_{missing_rate:.1f}_tail_index_{tail_index:.2f}"
                if key in parametric_results:
                    parametric_f1.append(parametric_results[key].get('edge_f1_score_mean', 0))
            
            if threshold_f1 and parametric_f1:
                analysis['mechanism_comparison'][missing_rate] = {
                    'threshold_mean_f1': np.mean(threshold_f1),
                    'parametric_mean_f1': np.mean(parametric_f1),
                    'threshold_std_f1': np.std(threshold_f1),
                    'parametric_std_f1': np.std(parametric_f1)
                }
        
        return analysis
    
    def _generate_robustness_visualizations(self, algorithm_name: str) -> None:
        """Generate comprehensive robustness visualizations."""
        logger.info(f"Generating robustness visualizations for {algorithm_name}")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Robustness Analysis: {algorithm_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Threshold-based performance
        self._plot_threshold_performance(axes[0, 0], algorithm_name)
        
        # Plot 2: Parametric performance
        self._plot_parametric_performance(axes[0, 1], algorithm_name)
        
        # Plot 3: Mechanism comparison
        self._plot_mechanism_comparison(axes[0, 2], algorithm_name)
        
        # Plot 4: Tail behavior analysis
        self._plot_tail_behavior(axes[1, 0], algorithm_name)
        
        # Plot 5: Missing rate impact
        self._plot_missing_rate_impact(axes[1, 1], algorithm_name)
        
        # Plot 6: Robustness summary
        self._plot_robustness_summary(axes[1, 2], algorithm_name)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/{algorithm_name}_robustness_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_performance(self, ax, algorithm_name: str) -> None:
        """Plot threshold-based performance."""
        threshold_results = self.results[algorithm_name]['threshold_results']
        
        # Prepare data for plotting
        quantiles = []
        missing_rates = []
        f1_scores = []
        
        for key, result in threshold_results.items():
            if 'quantile' in key:
                parts = key.split('_')
                quantile = int(parts[-1])
                missing_rate = float(parts[1])
                f1 = result.get('edge_f1_score_mean', 0)
                
                quantiles.append(quantile)
                missing_rates.append(missing_rate)
                f1_scores.append(f1)
        
        if quantiles:
            # Create heatmap
            df = pd.DataFrame({'quantile': quantiles, 'missing_rate': missing_rates, 'f1_score': f1_scores})
            pivot_df = df.pivot(index='missing_rate', columns='quantile', values='f1_score')
            
            sns.heatmap(pivot_df, annot=True, cmap='viridis', ax=ax, cbar_kws={'label': 'F1-Score'})
            ax.set_title('Threshold-Based MNAR Performance')
            ax.set_xlabel('Quantile Threshold')
            ax.set_ylabel('Missing Rate')
    
    def _plot_parametric_performance(self, ax, algorithm_name: str) -> None:
        """Plot parametric performance."""
        parametric_results = self.results[algorithm_name]['parametric_results']
        
        # Prepare data for plotting
        tail_indices = []
        missing_rates = []
        f1_scores = []
        
        for key, result in parametric_results.items():
            if 'tail_index' in key:
                parts = key.split('_')
                tail_index = float(parts[-1])
                missing_rate = float(parts[1])
                f1 = result.get('edge_f1_score_mean', 0)
                
                tail_indices.append(tail_index)
                missing_rates.append(missing_rate)
                f1_scores.append(f1)
        
        if tail_indices:
            # Create heatmap
            df = pd.DataFrame({'tail_index': tail_indices, 'missing_rate': missing_rates, 'f1_score': f1_scores})
            pivot_df = df.pivot(index='missing_rate', columns='tail_index', values='f1_score')
            
            sns.heatmap(pivot_df, annot=True, cmap='plasma', ax=ax, cbar_kws={'label': 'F1-Score'})
            ax.set_title('Parametric MNAR Performance')
            ax.set_xlabel('Tail Index')
            ax.set_ylabel('Missing Rate')
    
    def _plot_mechanism_comparison(self, ax, algorithm_name: str) -> None:
        """Plot mechanism comparison."""
        comparative_analysis = self.results[algorithm_name]['tail_analysis']['comparative_analysis']
        
        missing_rates = []
        threshold_f1 = []
        parametric_f1 = []
        
        for missing_rate, comparison in comparative_analysis['mechanism_comparison'].items():
            missing_rates.append(missing_rate)
            threshold_f1.append(comparison['threshold_mean_f1'])
            parametric_f1.append(comparison['parametric_mean_f1'])
        
        if missing_rates:
            ax.plot(missing_rates, threshold_f1, 'o-', label='Threshold-based', linewidth=2, markersize=8)
            ax.plot(missing_rates, parametric_f1, 's-', label='Parametric', linewidth=2, markersize=8)
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('F1-Score')
            ax.set_title('Mechanism Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_tail_behavior(self, ax, algorithm_name: str) -> None:
        """Plot tail behavior analysis."""
        # This would show how tail-specific metrics change with different mechanisms
        ax.text(0.5, 0.5, 'Tail Behavior Analysis\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Tail Behavior Analysis')
    
    def _plot_missing_rate_impact(self, ax, algorithm_name: str) -> None:
        """Plot missing rate impact analysis."""
        # This would show the impact of different missing rates on performance
        ax.text(0.5, 0.5, 'Missing Rate Impact\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Missing Rate Impact')
    
    def _plot_robustness_summary(self, ax, algorithm_name: str) -> None:
        """Plot robustness summary."""
        # This would show a summary of robustness metrics
        ax.text(0.5, 0.5, 'Robustness Summary\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Robustness Summary')
    
    def _save_evaluation_results(self, algorithm_name: str) -> None:
        """Save evaluation results to files."""
        import json
        
        # Save detailed results
        results_file = f"{self.config.output_dir}/{algorithm_name}_robustness_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(self.results[algorithm_name]), f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_robustness_report(self, algorithm_name: str) -> Dict[str, Any]:
        """Generate a comprehensive robustness report."""
        if algorithm_name not in self.results:
            raise ValueError(f"No results found for algorithm: {algorithm_name}")
        
        results = self.results[algorithm_name]
        
        report = {
            'algorithm_name': algorithm_name,
            'evaluation_config': {
                'missing_rates': self.config.missing_rates,
                'quantile_thresholds': self.config.quantile_thresholds,
                'tail_indices': self.config.tail_indices,
                'n_repetitions': self.config.n_repetitions
            },
            'summary_statistics': self._calculate_summary_statistics(algorithm_name),
            'robustness_insights': self._generate_robustness_insights(algorithm_name),
            'recommendations': self._generate_recommendations(algorithm_name)
        }
        
        return report
    
    def _calculate_summary_statistics(self, algorithm_name: str) -> Dict[str, Any]:
        """Calculate summary statistics for the robustness evaluation."""
        results = self.results[algorithm_name]
        
        # Calculate overall performance metrics
        all_f1_scores = []
        all_tail_metrics = []
        
        # Collect from threshold results
        for result in results['threshold_results'].values():
            if 'edge_f1_score_mean' in result:
                all_f1_scores.append(result['edge_f1_score_mean'])
            if 'mean_tail_missing_rate_mean' in result:
                all_tail_metrics.append(result['mean_tail_missing_rate_mean'])
        
        # Collect from parametric results
        for result in results['parametric_results'].values():
            if 'edge_f1_score_mean' in result:
                all_f1_scores.append(result['edge_f1_score_mean'])
            if 'mean_tail_missing_rate_mean' in result:
                all_tail_metrics.append(result['mean_tail_missing_rate_mean'])
        
        summary = {
            'overall_performance': {
                'mean_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0,
                'std_f1_score': np.std(all_f1_scores) if all_f1_scores else 0,
                'min_f1_score': np.min(all_f1_scores) if all_f1_scores else 0,
                'max_f1_score': np.max(all_f1_scores) if all_f1_scores else 0
            },
            'tail_behavior': {
                'mean_tail_missing_rate': np.mean(all_tail_metrics) if all_tail_metrics else 0,
                'std_tail_missing_rate': np.std(all_tail_metrics) if all_tail_metrics else 0
            },
            'evaluation_coverage': {
                'n_threshold_conditions': len(results['threshold_results']),
                'n_parametric_conditions': len(results['parametric_results']),
                'total_conditions': len(results['threshold_results']) + len(results['parametric_results'])
            }
        }
        
        return summary
    
    def _generate_robustness_insights(self, algorithm_name: str) -> List[str]:
        """Generate insights about algorithm robustness."""
        insights = []
        
        # Add insights based on analysis
        insights.append(f"Algorithm {algorithm_name} was evaluated under {len(self.config.missing_rates)} different missing rates")
        insights.append(f"Threshold-based MNAR was tested with {len(self.config.quantile_thresholds)} quantile thresholds")
        insights.append(f"Parametric MNAR was tested with {len(self.config.tail_indices)} different tail indices")
        
        # Add performance-based insights
        results = self.results[algorithm_name]
        summary = self._calculate_summary_statistics(algorithm_name)
        
        mean_f1 = summary['overall_performance']['mean_f1_score']
        if mean_f1 > 0.8:
            insights.append("Algorithm demonstrates high robustness across all tested conditions")
        elif mean_f1 > 0.6:
            insights.append("Algorithm shows moderate robustness with some performance degradation")
        else:
            insights.append("Algorithm shows limited robustness under heavy-tailed MNAR conditions")
        
        return insights
    
    def _generate_recommendations(self, algorithm_name: str) -> List[str]:
        """Generate recommendations based on robustness evaluation."""
        recommendations = []
        
        results = self.results[algorithm_name]
        summary = self._calculate_summary_statistics(algorithm_name)
        
        mean_f1 = summary['overall_performance']['mean_f1_score']
        
        if mean_f1 > 0.8:
            recommendations.append("Algorithm is suitable for deployment in heavy-tailed MNAR scenarios")
            recommendations.append("Consider using this algorithm for extreme value analysis applications")
        elif mean_f1 > 0.6:
            recommendations.append("Algorithm may require additional robustness improvements for extreme scenarios")
            recommendations.append("Consider ensemble methods or data augmentation for better tail performance")
        else:
            recommendations.append("Algorithm requires significant improvements for heavy-tailed MNAR robustness")
            recommendations.append("Consider developing specialized tail-aware causal discovery methods")
        
        recommendations.append("Further evaluation with real-world extreme value datasets is recommended")
        recommendations.append("Consider incorporating tail-specific regularization in the algorithm")
        
        return recommendations

# Convenience function for quick evaluation
def evaluate_algorithm_robustness(algorithm_func,
                                true_graph: nx.DiGraph,
                                data: pd.DataFrame,
                                missing_rates: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                                quantile_thresholds: List[float] = [90, 95, 99],
                                tail_indices: List[float] = [0.1, 0.5, 1.0, 2.0],
                                n_repetitions: int = 5,
                                algorithm_name: str = "SM-MVPC",
                                output_dir: str = "results/robustness_analysis") -> Dict[str, Any]:
    """
    Convenience function for evaluating algorithm robustness.
    
    Args:
        algorithm_func: Function that takes (data, missing_mask) and returns inferred graph
        true_graph: Ground truth causal graph
        data: Complete dataset
        missing_rates: List of missing rates to test
        quantile_thresholds: List of quantile thresholds for threshold-based MNAR
        tail_indices: List of tail indices for parametric MNAR
        n_repetitions: Number of repetitions per condition
        algorithm_name: Name of the algorithm
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    config = RobustnessEvaluationConfig(
        missing_rates=missing_rates,
        quantile_thresholds=quantile_thresholds,
        tail_indices=tail_indices,
        n_repetitions=n_repetitions,
        output_dir=output_dir
    )
    
    evaluator = HeavyTailedRobustnessEvaluator(config)
    results = evaluator.evaluate_algorithm_robustness(algorithm_func, true_graph, data, algorithm_name)
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Heavy-Tailed MNAR Robustness Evaluator - Example Usage")
    
    # This would be used with actual algorithm and data
    print("Ready for robustness evaluation!")
