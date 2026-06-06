"""
Heavy-Tailed MNAR Robustness Experiment Pipeline

This module implements a comprehensive experimental pipeline for evaluating
causal discovery algorithms under heavy-tailed MNAR mechanisms. It extends
the existing experimental framework with specialized capabilities for
extreme value analysis and tail behavior assessment.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import warnings

from ..data_generation.heavy_tailed_mnar_generator import (
    HeavyTailedMNARGenerator, TailMNARConfig, 
    create_threshold_mnar_config, create_parametric_mnar_config
)
from ..evaluation.heavy_tailed_robustness_evaluator import (
    HeavyTailedRobustnessEvaluator, RobustnessEvaluationConfig
)
from ..algorithms.causal_discovery import SMMVPCCausalDiscovery
from ..utils.data_loader import load_medical_datasets
from ..utils.visualization import create_robustness_plots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeavyTailedRobustnessExperiment:
    """
    Comprehensive experimental pipeline for heavy-tailed MNAR robustness evaluation.
    
    This class orchestrates the entire experimental process, from data generation
    through algorithm evaluation to comprehensive reporting. It provides a
    systematic approach to assessing algorithm robustness under extreme
    missingness scenarios.
    """
    
    def __init__(self, 
                 datasets: List[str] = None,
                 missing_rates: List[float] = None,
                 quantile_thresholds: List[float] = None,
                 tail_indices: List[float] = None,
                 n_repetitions: int = 5,
                 output_dir: str = "results/heavy_tailed_robustness",
                 random_seed: int = 42):
        """
        Initialize the robustness experiment.
        
        Args:
            datasets: List of dataset names to evaluate
            missing_rates: List of missing rates to test
            quantile_thresholds: List of quantile thresholds for threshold-based MNAR
            tail_indices: List of tail indices for parametric MNAR
            n_repetitions: Number of repetitions per condition
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.datasets = datasets or ['diabetes', 'heart_disease', 'hepatitis']
        self.missing_rates = missing_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.quantile_thresholds = quantile_thresholds or [90, 95, 99]
        self.tail_indices = tail_indices or [0.1, 0.5, 1.0, 2.0]
        self.n_repetitions = n_repetitions
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.experiment_results = {}
        self.summary_statistics = {}
        self.robustness_insights = {}
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized Heavy-Tailed Robustness Experiment")
        logger.info(f"Datasets: {self.datasets}")
        logger.info(f"Missing rates: {self.missing_rates}")
        logger.info(f"Quantile thresholds: {self.quantile_thresholds}")
        logger.info(f"Tail indices: {self.tail_indices}")
        logger.info(f"Repetitions: {self.n_repetitions}")
    
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """
        Run the comprehensive robustness experiment.
        
        Returns:
            Dictionary containing all experimental results and analysis
        """
        logger.info("Starting comprehensive heavy-tailed MNAR robustness experiment")
        start_time = time.time()
        
        # Load datasets and generate ground truth graphs
        logger.info("Loading datasets and generating ground truth graphs")
        dataset_info = self._load_datasets_and_ground_truth()
        
        # Run experiments for each dataset
        for dataset_name, dataset_data in dataset_info.items():
            logger.info(f"Running experiments for dataset: {dataset_name}")
            
            # Run threshold-based MNAR experiments
            threshold_results = self._run_threshold_experiments(dataset_name, dataset_data)
            
            # Run parametric MNAR experiments
            parametric_results = self._run_parametric_experiments(dataset_name, dataset_data)
            
            # Store results
            self.experiment_results[dataset_name] = {
                'threshold_results': threshold_results,
                'parametric_results': parametric_results,
                'dataset_info': dataset_data['info']
            }
        
        # Generate comprehensive analysis
        logger.info("Generating comprehensive analysis")
        self._generate_comprehensive_analysis()
        
        # Save results
        self._save_experiment_results()
        
        # Generate final report
        final_report = self._generate_final_report()
        
        end_time = time.time()
        logger.info(f"Experiment completed in {end_time - start_time:.2f} seconds")
        
        return final_report
    
    def _load_datasets_and_ground_truth(self) -> Dict[str, Dict]:
        """Load datasets and generate ground truth causal graphs."""
        dataset_info = {}
        
        for dataset_name in self.datasets:
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
        from ..algorithms.causal_discovery import SMMVPCCausalDiscovery
        
        # Use SM-MVPC with complete data to generate ground truth
        algorithm = SMMVPCCausalDiscovery()
        
        # Create empty missing mask (no missing data)
        missing_mask = np.zeros(len(data), dtype=bool)
        
        # Run algorithm
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
                    'skewness': self._calculate_skewness(values),
                    'kurtosis': self._calculate_kurtosis(values)
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
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of the data."""
        from scipy import stats
        return stats.skew(values)
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        from scipy import stats
        return stats.kurtosis(values)
    
    def _run_threshold_experiments(self, dataset_name: str, dataset_data: Dict) -> Dict[str, Any]:
        """Run threshold-based MNAR experiments."""
        logger.info(f"Running threshold-based experiments for {dataset_name}")
        
        data = dataset_data['data']
        ground_truth = dataset_data['ground_truth']
        
        results = {}
        
        for missing_rate in self.missing_rates:
            for quantile in self.quantile_thresholds:
                logger.info(f"Testing: {missing_rate:.1%} missing, {quantile}th percentile")
                
                # Create configuration
                config = create_threshold_mnar_config(
                    target_missing_rate=missing_rate,
                    quantile_threshold=quantile,
                    random_seed=self.random_seed
                )
                
                # Run repetitions
                repetition_results = []
                for rep in range(self.n_repetitions):
                    try:
                        # Generate missingness
                        generator = HeavyTailedMNARGenerator(config)
                        missing_mask, generation_info = generator.generate_missingness_mask(data)
                        
                        # Apply missingness
                        data_with_missing = data.copy()
                        data_with_missing.loc[missing_mask, :] = np.nan
                        
                        # Run algorithm
                        algorithm = SMMVPCCausalDiscovery()
                        inferred_graph = algorithm.discover_causal_structure(data_with_missing, missing_mask)
                        
                        # Calculate metrics
                        from ..evaluation.metrics import CausalDiscoveryMetrics
                        metrics_calc = CausalDiscoveryMetrics()
                        metrics = metrics_calc.comprehensive_evaluation(ground_truth, inferred_graph)
                        
                        # Add tail-specific metrics
                        tail_metrics = self._calculate_tail_metrics(data, missing_mask, ground_truth, inferred_graph)
                        metrics.update(tail_metrics)
                        
                        # Add generation info
                        metrics.update(generation_info)
                        
                        repetition_results.append(metrics)
                        
                    except Exception as e:
                        logger.warning(f"Repetition {rep} failed: {e}")
                        continue
                
                if repetition_results:
                    # Aggregate results
                    key = f"missing_{missing_rate:.1f}_quantile_{quantile}"
                    results[key] = self._aggregate_repetition_results(repetition_results)
        
        return results
    
    def _run_parametric_experiments(self, dataset_name: str, dataset_data: Dict) -> Dict[str, Any]:
        """Run parametric MNAR experiments."""
        logger.info(f"Running parametric experiments for {dataset_name}")
        
        data = dataset_data['data']
        ground_truth = dataset_data['ground_truth']
        
        results = {}
        
        for missing_rate in self.missing_rates:
            for tail_index in self.tail_indices:
                logger.info(f"Testing: {missing_rate:.1%} missing, tail_index={tail_index}")
                
                # Create configuration
                config = create_parametric_mnar_config(
                    target_missing_rate=missing_rate,
                    tail_index=tail_index,
                    random_seed=self.random_seed
                )
                
                # Run repetitions
                repetition_results = []
                for rep in range(self.n_repetitions):
                    try:
                        # Generate missingness
                        generator = HeavyTailedMNARGenerator(config)
                        missing_mask, generation_info = generator.generate_missingness_mask(data)
                        
                        # Apply missingness
                        data_with_missing = data.copy()
                        data_with_missing.loc[missing_mask, :] = np.nan
                        
                        # Run algorithm
                        algorithm = SMMVPCCausalDiscovery()
                        inferred_graph = algorithm.discover_causal_structure(data_with_missing, missing_mask)
                        
                        # Calculate metrics
                        from ..evaluation.metrics import CausalDiscoveryMetrics
                        metrics_calc = CausalDiscoveryMetrics()
                        metrics = metrics_calc.comprehensive_evaluation(ground_truth, inferred_graph)
                        
                        # Add tail-specific metrics
                        tail_metrics = self._calculate_tail_metrics(data, missing_mask, ground_truth, inferred_graph)
                        metrics.update(tail_metrics)
                        
                        # Add generation info
                        metrics.update(generation_info)
                        
                        repetition_results.append(metrics)
                        
                    except Exception as e:
                        logger.warning(f"Repetition {rep} failed: {e}")
                        continue
                
                if repetition_results:
                    # Aggregate results
                    key = f"missing_{missing_rate:.1f}_tail_index_{tail_index:.2f}"
                    results[key] = self._aggregate_repetition_results(repetition_results)
        
        return results
    
    def _calculate_tail_metrics(self, data: pd.DataFrame, missing_mask: np.ndarray,
                               true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate tail-specific metrics."""
        metrics = {}
        
        # Calculate tail statistics for each variable
        tail_missing_rates = []
        tail_indices = []
        
        for col in data.columns:
            values = data[col].values
            missing_col = missing_mask[data.index]
            
            # Calculate tail statistics
            tail_threshold = np.percentile(values, 95)
            tail_values = values[values > tail_threshold]
            tail_missing_rate = np.mean(missing_col[values > tail_threshold])
            
            # Calculate tail index
            tail_index = self._estimate_tail_index(tail_values, 95)
            
            tail_missing_rates.append(tail_missing_rate)
            if tail_index is not None:
                tail_indices.append(tail_index)
        
        metrics['mean_tail_missing_rate'] = np.mean(tail_missing_rates)
        metrics['std_tail_missing_rate'] = np.std(tail_missing_rates)
        metrics['mean_tail_index'] = np.mean(tail_indices) if tail_indices else None
        metrics['std_tail_index'] = np.std(tail_indices) if tail_indices else None
        
        # Calculate edge-specific tail metrics
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Focus on edges involving high-degree nodes
        node_degrees = dict(true_graph.degree())
        if node_degrees:
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
    
    def _generate_comprehensive_analysis(self) -> None:
        """Generate comprehensive analysis of experimental results."""
        logger.info("Generating comprehensive analysis")
        
        # Analyze each dataset
        for dataset_name, results in self.experiment_results.items():
            logger.info(f"Analyzing results for {dataset_name}")
            
            # Analyze threshold results
            threshold_analysis = self._analyze_threshold_results(results['threshold_results'])
            
            # Analyze parametric results
            parametric_analysis = self._analyze_parametric_results(results['parametric_results'])
            
            # Generate comparative analysis
            comparative_analysis = self._generate_comparative_analysis(
                results['threshold_results'], results['parametric_results']
            )
            
            # Store analysis
            self.robustness_insights[dataset_name] = {
                'threshold_analysis': threshold_analysis,
                'parametric_analysis': parametric_analysis,
                'comparative_analysis': comparative_analysis
            }
        
        # Generate cross-dataset analysis
        self._generate_cross_dataset_analysis()
    
    def _analyze_threshold_results(self, threshold_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threshold-based MNAR results."""
        analysis = {
            'quantile_sensitivity': {},
            'missing_rate_impact': {},
            'performance_trends': {}
        }
        
        # Analyze quantile sensitivity
        for quantile in self.quantile_thresholds:
            quantile_data = []
            for missing_rate in self.missing_rates:
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
    
    def _analyze_parametric_results(self, parametric_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parametric MNAR results."""
        analysis = {
            'tail_index_sensitivity': {},
            'missing_rate_impact': {},
            'gpd_parameter_analysis': {}
        }
        
        # Analyze tail index sensitivity
        for tail_index in self.tail_indices:
            tail_index_data = []
            for missing_rate in self.missing_rates:
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
    
    def _generate_comparative_analysis(self, threshold_results: Dict[str, Any], 
                                     parametric_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between mechanisms."""
        analysis = {
            'mechanism_comparison': {},
            'robustness_ranking': {},
            'tail_behavior_insights': {}
        }
        
        # Compare mechanisms at similar missing rates
        for missing_rate in self.missing_rates:
            threshold_f1 = []
            parametric_f1 = []
            
            # Collect threshold results
            for quantile in self.quantile_thresholds:
                key = f"missing_{missing_rate:.1f}_quantile_{quantile}"
                if key in threshold_results:
                    threshold_f1.append(threshold_results[key].get('edge_f1_score_mean', 0))
            
            # Collect parametric results
            for tail_index in self.tail_indices:
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
    
    def _generate_cross_dataset_analysis(self) -> None:
        """Generate cross-dataset analysis."""
        logger.info("Generating cross-dataset analysis")
        
        # This would contain analysis across all datasets
        # Implementation would go here
        pass
    
    def _save_experiment_results(self) -> None:
        """Save experimental results to files."""
        logger.info("Saving experimental results")
        
        # Save detailed results
        results_file = self.output_dir / "heavy_tailed_robustness_results.json"
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
            
            json.dump(convert_numpy(self.experiment_results), f, indent=2)
        
        # Save analysis results
        analysis_file = self.output_dir / "robustness_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(convert_numpy(self.robustness_insights), f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        logger.info("Generating final report")
        
        report = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'datasets': self.datasets,
                'missing_rates': self.missing_rates,
                'quantile_thresholds': self.quantile_thresholds,
                'tail_indices': self.tail_indices,
                'n_repetitions': self.n_repetitions,
                'random_seed': self.random_seed
            },
            'experiment_results': self.experiment_results,
            'robustness_insights': self.robustness_insights,
            'summary_statistics': self._calculate_experiment_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save final report
        report_file = self.output_dir / "final_robustness_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_experiment_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for the entire experiment."""
        summary = {
            'total_conditions_tested': 0,
            'successful_conditions': 0,
            'overall_performance': {},
            'dataset_performance': {},
            'mechanism_performance': {}
        }
        
        # Calculate overall statistics
        all_f1_scores = []
        
        for dataset_name, results in self.experiment_results.items():
            dataset_f1_scores = []
            
            # Collect from threshold results
            for result in results['threshold_results'].values():
                if 'edge_f1_score_mean' in result:
                    f1_score = result['edge_f1_score_mean']
                    all_f1_scores.append(f1_score)
                    dataset_f1_scores.append(f1_score)
            
            # Collect from parametric results
            for result in results['parametric_results'].values():
                if 'edge_f1_score_mean' in result:
                    f1_score = result['edge_f1_score_mean']
                    all_f1_scores.append(f1_score)
                    dataset_f1_scores.append(f1_score)
            
            # Calculate dataset summary
            if dataset_f1_scores:
                summary['dataset_performance'][dataset_name] = {
                    'mean_f1_score': np.mean(dataset_f1_scores),
                    'std_f1_score': np.std(dataset_f1_scores),
                    'min_f1_score': np.min(dataset_f1_scores),
                    'max_f1_score': np.max(dataset_f1_scores)
                }
        
        # Calculate overall performance
        if all_f1_scores:
            summary['overall_performance'] = {
                'mean_f1_score': np.mean(all_f1_scores),
                'std_f1_score': np.std(all_f1_scores),
                'min_f1_score': np.min(all_f1_scores),
                'max_f1_score': np.max(all_f1_scores)
            }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on experimental results."""
        recommendations = []
        
        # Analyze overall performance
        all_f1_scores = []
        for results in self.experiment_results.values():
            for result in results['threshold_results'].values():
                if 'edge_f1_score_mean' in result:
                    all_f1_scores.append(result['edge_f1_score_mean'])
            for result in results['parametric_results'].values():
                if 'edge_f1_score_mean' in result:
                    all_f1_scores.append(result['edge_f1_score_mean'])
        
        if all_f1_scores:
            mean_f1 = np.mean(all_f1_scores)
            
            if mean_f1 > 0.8:
                recommendations.append("Algorithm demonstrates excellent robustness under heavy-tailed MNAR conditions")
                recommendations.append("Suitable for deployment in extreme value analysis scenarios")
            elif mean_f1 > 0.6:
                recommendations.append("Algorithm shows good robustness with some performance degradation in extreme cases")
                recommendations.append("Consider additional robustness improvements for critical applications")
            else:
                recommendations.append("Algorithm requires significant improvements for heavy-tailed MNAR robustness")
                recommendations.append("Consider developing specialized tail-aware causal discovery methods")
        
        recommendations.append("Further evaluation with real-world extreme value datasets is recommended")
        recommendations.append("Consider incorporating tail-specific regularization techniques")
        recommendations.append("Evaluate algorithm performance on additional heavy-tailed distributions")
        
        return recommendations

# Convenience function for running the experiment
def run_heavy_tailed_robustness_experiment(
    datasets: List[str] = None,
    missing_rates: List[float] = None,
    quantile_thresholds: List[float] = None,
    tail_indices: List[float] = None,
    n_repetitions: int = 5,
    output_dir: str = "results/heavy_tailed_robustness",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Convenience function for running the heavy-tailed robustness experiment.
    
    Args:
        datasets: List of dataset names to evaluate
        missing_rates: List of missing rates to test
        quantile_thresholds: List of quantile thresholds for threshold-based MNAR
        tail_indices: List of tail indices for parametric MNAR
        n_repetitions: Number of repetitions per condition
        output_dir: Directory to save results
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing comprehensive experimental results
    """
    experiment = HeavyTailedRobustnessExperiment(
        datasets=datasets,
        missing_rates=missing_rates,
        quantile_thresholds=quantile_thresholds,
        tail_indices=tail_indices,
        n_repetitions=n_repetitions,
        output_dir=output_dir,
        random_seed=random_seed
    )
    
    return experiment.run_comprehensive_experiment()

if __name__ == "__main__":
    # Example usage
    print("Heavy-Tailed MNAR Robustness Experiment - Example Usage")
    
    # Run experiment with default parameters
    results = run_heavy_tailed_robustness_experiment(
        datasets=['diabetes', 'heart_disease'],
        missing_rates=[0.1, 0.2, 0.3],
        quantile_thresholds=[90, 95],
        tail_indices=[0.5, 1.0],
        n_repetitions=3
    )
    
    print("Experiment completed successfully!")
    print(f"Results saved to: {results['experiment_metadata']['timestamp']}")
