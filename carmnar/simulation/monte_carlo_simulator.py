"""
Monte Carlo Simulation Framework for MNAR Mechanism Evaluation

This module provides a comprehensive Monte Carlo simulation framework for evaluating
MNAR mechanisms with multiple random seeds, varying sample sizes, and statistical
stability assessment.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..data_generation.heavy_tailed_mnar_generator import (
    HeavyTailedMNARGenerator, TailMNARConfig,
    create_threshold_mnar_config, create_parametric_mnar_config
)
from ..data_generation.mnar_generator import MNARGenerator, MNARConfig
from ..algorithms.causal_discovery import SMMVPCCausalDiscovery
from ..evaluation.metrics import CausalDiscoveryMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    n_runs: int = 100
    random_seeds: List[int] = None
    sample_size_factors: List[float] = None
    missingness_proportions: List[float] = None
    mechanisms: List[str] = None
    parallel_processing: bool = True
    n_workers: int = None
    save_intermediate_results: bool = True
    output_dir: str = "results/monte_carlo_simulations"

class MonteCarloSimulator:
    """
    Monte Carlo simulator for MNAR mechanism evaluation.
    
    This class provides comprehensive Monte Carlo simulation capabilities for
    evaluating different MNAR mechanisms with statistical stability assessment,
    multiple random seeds, and varying sample sizes.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            config: Configuration object for Monte Carlo simulations
        """
        self.config = config
        self.metrics_calculator = CausalDiscoveryMetrics()
        
        # Set default values
        if self.config.random_seeds is None:
            self.config.random_seeds = list(range(self.config.n_runs))
        
        if self.config.sample_size_factors is None:
            self.config.sample_size_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        if self.config.missingness_proportions is None:
            self.config.missingness_proportions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        if self.config.mechanisms is None:
            self.config.mechanisms = ['threshold', 'parametric', 'sigmoid']
        
        if self.config.n_workers is None:
            self.config.n_workers = min(mp.cpu_count(), 8)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.simulation_results = {}
        self.stability_analysis = {}
        
        logger.info(f"Initialized Monte Carlo Simulator")
        logger.info(f"Number of runs: {self.config.n_runs}")
        logger.info(f"Sample size factors: {self.config.sample_size_factors}")
        logger.info(f"Missingness proportions: {self.config.missingness_proportions}")
        logger.info(f"Mechanisms: {self.config.mechanisms}")
    
    def run_comprehensive_simulations(self, 
                                    datasets: Dict[str, pd.DataFrame],
                                    ground_truths: Dict[str, nx.DiGraph]) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo simulations across all conditions.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            ground_truths: Dictionary mapping dataset names to ground truth graphs
            
        Returns:
            Dictionary containing all simulation results and analysis
        """
        logger.info("Starting comprehensive Monte Carlo simulations")
        start_time = time.time()
        
        # Run simulations for each dataset
        for dataset_name, data in datasets.items():
            logger.info(f"Running simulations for dataset: {dataset_name}")
            
            ground_truth = ground_truths[dataset_name]
            
            # Run simulations for each mechanism
            dataset_results = {}
            for mechanism in self.config.mechanisms:
                logger.info(f"Running {mechanism} mechanism simulations for {dataset_name}")
                
                mechanism_results = self._run_mechanism_simulations(
                    data, ground_truth, mechanism, dataset_name
                )
                dataset_results[mechanism] = mechanism_results
            
            self.simulation_results[dataset_name] = dataset_results
        
        # Generate stability analysis
        logger.info("Generating stability analysis")
        self._generate_stability_analysis()
        
        # Save results
        self._save_simulation_results()
        
        # Generate final report
        final_report = self._generate_simulation_report()
        
        end_time = time.time()
        logger.info(f"Monte Carlo simulations completed in {end_time - start_time:.2f} seconds")
        
        return final_report
    
    def _run_mechanism_simulations(self, 
                                 data: pd.DataFrame, 
                                 ground_truth: nx.DiGraph,
                                 mechanism: str,
                                 dataset_name: str) -> Dict[str, Any]:
        """Run Monte Carlo simulations for a specific mechanism."""
        mechanism_results = {}
        
        # Run simulations for each missingness proportion
        for missing_pct in self.config.missingness_proportions:
            logger.info(f"Running {mechanism} simulations: {missing_pct:.1%} missing")
            
            missingness_results = {}
            
            # Run simulations for each sample size factor
            for sample_factor in self.config.sample_size_factors:
                logger.info(f"Sample size factor: {sample_factor}")
                
                # Prepare data with sample size factor
                n_samples = int(len(data) * sample_factor)
                if n_samples < 10:
                    logger.warning(f"Sample size too small ({n_samples}), skipping")
                    continue
                
                sampled_data = data.sample(n=n_samples, random_state=42).reset_index(drop=True)
                
                # Run Monte Carlo simulations
                if self.config.parallel_processing:
                    mc_results = self._run_parallel_monte_carlo(
                        sampled_data, ground_truth, mechanism, missing_pct
                    )
                else:
                    mc_results = self._run_sequential_monte_carlo(
                        sampled_data, ground_truth, mechanism, missing_pct
                    )
                
                missingness_results[f"sample_factor_{sample_factor}"] = mc_results
            
            mechanism_results[f"missing_{missing_pct:.1f}"] = missingness_results
        
        return mechanism_results
    
    def _run_parallel_monte_carlo(self, 
                                data: pd.DataFrame,
                                ground_truth: nx.DiGraph,
                                mechanism: str,
                                missing_pct: float) -> Dict[str, Any]:
        """Run Monte Carlo simulations in parallel."""
        logger.info(f"Running parallel Monte Carlo simulations: {mechanism}, {missing_pct:.1%} missing")
        
        # Prepare arguments for parallel execution
        args_list = []
        for run_idx in range(self.config.n_runs):
            args = (
                data.copy(),
                ground_truth,
                mechanism,
                missing_pct,
                self.config.random_seeds[run_idx],
                run_idx
            )
            args_list.append(args)
        
        # Run simulations in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_run = {
                executor.submit(self._single_simulation, *args): args[-1] 
                for args in args_list
            }
            
            for future in as_completed(future_to_run):
                run_idx = future_to_run[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    logger.warning(f"Simulation run {run_idx} failed: {e}")
        
        return self._aggregate_simulation_results(all_results)
    
    def _run_sequential_monte_carlo(self, 
                                  data: pd.DataFrame,
                                  ground_truth: nx.DiGraph,
                                  mechanism: str,
                                  missing_pct: float) -> Dict[str, Any]:
        """Run Monte Carlo simulations sequentially."""
        logger.info(f"Running sequential Monte Carlo simulations: {mechanism}, {missing_pct:.1%} missing")
        
        all_results = []
        for run_idx in range(self.config.n_runs):
            try:
                result = self._single_simulation(
                    data.copy(),
                    ground_truth,
                    mechanism,
                    missing_pct,
                    self.config.random_seeds[run_idx],
                    run_idx
                )
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Simulation run {run_idx} failed: {e}")
        
        return self._aggregate_simulation_results(all_results)
    
    def _single_simulation(self, 
                         data: pd.DataFrame,
                         ground_truth: nx.DiGraph,
                         mechanism: str,
                         missing_pct: float,
                         random_seed: int,
                         run_idx: int) -> Optional[Dict[str, Any]]:
        """Run a single simulation."""
        try:
            # Set random seed
            np.random.seed(random_seed)
            
            # Generate missingness based on mechanism
            missing_mask, generation_info = self._generate_missingness(
                data, mechanism, missing_pct, random_seed
            )
            
            # Apply missingness to data
            data_with_missing = data.copy()
            data_with_missing.loc[missing_mask, :] = np.nan
            
            # Run algorithm
            algorithm = SMMVPCCausalDiscovery()
            inferred_graph = algorithm.discover_causal_structure(data_with_missing, missing_mask)
            
            # Calculate metrics
            metrics = self.metrics_calculator.comprehensive_evaluation(ground_truth, inferred_graph)
            
            # Add simulation metadata
            metrics.update({
                'run_id': run_idx,
                'random_seed': random_seed,
                'mechanism': mechanism,
                'missing_percentage': missing_pct,
                'n_samples': len(data),
                'n_missing': np.sum(missing_mask),
                'actual_missing_rate': np.mean(missing_mask)
            })
            
            # Add generation info
            metrics.update(generation_info)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Single simulation failed: {e}")
            return None
    
    def _generate_missingness(self, 
                            data: pd.DataFrame,
                            mechanism: str,
                            missing_pct: float,
                            random_seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate missingness based on the specified mechanism."""
        if mechanism == 'threshold':
            config = create_threshold_mnar_config(
                target_missing_rate=missing_pct,
                quantile_threshold=95,  # Default quantile
                random_seed=random_seed
            )
            generator = HeavyTailedMNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        elif mechanism == 'parametric':
            config = create_parametric_mnar_config(
                target_missing_rate=missing_pct,
                tail_index=1.0,  # Default tail index
                random_seed=random_seed
            )
            generator = HeavyTailedMNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        elif mechanism == 'sigmoid':
            config = MNARConfig(
                target_missing_rate=missing_pct,
                steepness=2.0,  # Default steepness
                center=0.5,  # Default center
                random_seed=random_seed
            )
            generator = MNARGenerator(config)
            missing_mask, generation_info = generator.generate_missingness_mask(data)
        
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return missing_mask, generation_info
    
    def _aggregate_simulation_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple simulation runs."""
        if not all_results:
            return {}
        
        # Convert to DataFrame for easier analysis
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
                t_val = 1.96  # Approximate for large n
                ci_lower = mean_val - t_val * se
                ci_upper = mean_val + t_val * se
                
                confidence_intervals[f"{col}_ci_lower"] = ci_lower
                confidence_intervals[f"{col}_ci_upper"] = ci_upper
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(df_results)
        
        return {
            'summary_statistics': summary_stats,
            'confidence_intervals': confidence_intervals,
            'stability_metrics': stability_metrics,
            'raw_results': all_results,
            'n_successful_runs': len(all_results),
            'n_total_runs': self.config.n_runs
        }
    
    def _calculate_stability_metrics(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate stability metrics for simulation results."""
        stability_metrics = {}
        
        # Key performance metrics
        key_metrics = ['edge_f1_score', 'edge_precision', 'edge_recall', 'relative_structural_hamming_distance']
        
        for metric in key_metrics:
            if metric in df_results.columns:
                values = df_results[metric].values
                
                # Coefficient of variation
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                
                # Stability rating
                if cv < 0.05:
                    stability_rating = "excellent"
                elif cv < 0.1:
                    stability_rating = "good"
                elif cv < 0.2:
                    stability_rating = "moderate"
                else:
                    stability_rating = "poor"
                
                stability_metrics[metric] = {
                    'coefficient_of_variation': cv,
                    'stability_rating': stability_rating,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': np.max(values) - np.min(values)
                }
        
        return stability_metrics
    
    def _generate_stability_analysis(self) -> None:
        """Generate comprehensive stability analysis."""
        logger.info("Generating stability analysis")
        
        self.stability_analysis = {
            'mechanism_stability': {},
            'missingness_sensitivity': {},
            'sample_size_impact': {},
            'cross_dataset_stability': {}
        }
        
        # Analyze mechanism stability
        for dataset_name, dataset_results in self.simulation_results.items():
            self.stability_analysis['mechanism_stability'][dataset_name] = \
                self._analyze_mechanism_stability(dataset_results)
        
        # Analyze missingness sensitivity
        for dataset_name, dataset_results in self.simulation_results.items():
            self.stability_analysis['missingness_sensitivity'][dataset_name] = \
                self._analyze_missingness_sensitivity(dataset_results)
        
        # Analyze sample size impact
        for dataset_name, dataset_results in self.simulation_results.items():
            self.stability_analysis['sample_size_impact'][dataset_name] = \
                self._analyze_sample_size_impact(dataset_results)
    
    def _analyze_mechanism_stability(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability across different mechanisms."""
        mechanism_stability = {}
        
        for mechanism, mechanism_results in dataset_results.items():
            # Collect all F1 scores for this mechanism
            all_f1_scores = []
            
            for missingness_results in mechanism_results.values():
                for sample_results in missingness_results.values():
                    if 'raw_results' in sample_results:
                        f1_scores = [r['edge_f1_score'] for r in sample_results['raw_results']]
                        all_f1_scores.extend(f1_scores)
            
            if all_f1_scores:
                f1_array = np.array(all_f1_scores)
                
                # Calculate stability metrics
                cv = np.std(f1_array) / np.mean(f1_array) if np.mean(f1_array) > 0 else 0
                
                mechanism_stability[mechanism] = {
                    'mean_f1': np.mean(f1_array),
                    'std_f1': np.std(f1_array),
                    'coefficient_of_variation': cv,
                    'n_observations': len(f1_array),
                    'stability_rating': self._rate_stability(cv)
                }
        
        return mechanism_stability
    
    def _analyze_missingness_sensitivity(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity to missingness proportions."""
        missingness_sensitivity = {}
        
        for mechanism, mechanism_results in dataset_results.items():
            missingness_data = []
            
            for missingness_key, missingness_results in mechanism_results.items():
                if missingness_key.startswith('missing_'):
                    missing_pct = float(missingness_key.split('_')[1])
                    
                    # Collect F1 scores across all sample sizes
                    f1_scores = []
                    for sample_results in missingness_results.values():
                        if 'raw_results' in sample_results:
                            f1_scores.extend([r['edge_f1_score'] for r in sample_results['raw_results']])
                    
                    if f1_scores:
                        missingness_data.append({
                            'missing_percentage': missing_pct,
                            'mean_f1': np.mean(f1_scores),
                            'std_f1': np.std(f1_scores),
                            'n_observations': len(f1_scores)
                        })
            
            if missingness_data:
                missingness_sensitivity[mechanism] = {
                    'data': missingness_data,
                    'degradation_rate': self._calculate_degradation_rate(missingness_data)
                }
        
        return missingness_sensitivity
    
    def _analyze_sample_size_impact(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of sample size on performance."""
        sample_size_impact = {}
        
        for mechanism, mechanism_results in dataset_results.items():
            sample_size_data = []
            
            for missingness_key, missingness_results in mechanism_results.items():
                for sample_key, sample_results in missingness_results.items():
                    if sample_key.startswith('sample_factor_'):
                        sample_factor = float(sample_key.split('_')[2])
                        
                        if 'raw_results' in sample_results:
                            f1_scores = [r['edge_f1_score'] for r in sample_results['raw_results']]
                            
                            sample_size_data.append({
                                'sample_factor': sample_factor,
                                'mean_f1': np.mean(f1_scores),
                                'std_f1': np.std(f1_scores),
                                'n_observations': len(f1_scores)
                            })
            
            if sample_size_data:
                sample_size_impact[mechanism] = {
                    'data': sample_size_data,
                    'sample_size_sensitivity': self._calculate_sample_size_sensitivity(sample_size_data)
                }
        
        return sample_size_impact
    
    def _calculate_degradation_rate(self, missingness_data: List[Dict[str, Any]]) -> float:
        """Calculate performance degradation rate with missingness."""
        if len(missingness_data) < 2:
            return 0.0
        
        # Sort by missing percentage
        sorted_data = sorted(missingness_data, key=lambda x: x['missing_percentage'])
        
        # Linear regression to find degradation rate
        missingness = np.array([x['missing_percentage'] for x in sorted_data])
        performance = np.array([x['mean_f1'] for x in sorted_data])
        
        if len(missingness) > 1:
            slope, _ = np.polyfit(missingness, performance, 1)
            return -slope  # Negative slope indicates degradation
        
        return 0.0
    
    def _calculate_sample_size_sensitivity(self, sample_size_data: List[Dict[str, Any]]) -> float:
        """Calculate sensitivity to sample size changes."""
        if len(sample_size_data) < 2:
            return 0.0
        
        # Sort by sample factor
        sorted_data = sorted(sample_size_data, key=lambda x: x['sample_factor'])
        
        # Linear regression to find sensitivity
        sample_factors = np.array([x['sample_factor'] for x in sorted_data])
        performance = np.array([x['mean_f1'] for x in sorted_data])
        
        if len(sample_factors) > 1:
            slope, _ = np.polyfit(sample_factors, performance, 1)
            return slope  # Positive slope indicates improvement with larger samples
        
        return 0.0
    
    def _rate_stability(self, cv: float) -> str:
        """Rate stability based on coefficient of variation."""
        if cv < 0.05:
            return "excellent"
        elif cv < 0.1:
            return "good"
        elif cv < 0.2:
            return "moderate"
        else:
            return "poor"
    
    def _save_simulation_results(self) -> None:
        """Save simulation results to files."""
        logger.info("Saving simulation results")
        
        # Save detailed results
        results_file = Path(self.config.output_dir) / "monte_carlo_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.simulation_results, f, indent=2, default=str)
        
        # Save stability analysis
        stability_file = Path(self.config.output_dir) / "stability_analysis.json"
        with open(stability_file, 'w') as f:
            json.dump(self.stability_analysis, f, indent=2, default=str)
    
    def _generate_simulation_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report."""
        logger.info("Generating simulation report")
        
        report = {
            'simulation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_runs': self.config.n_runs,
                'sample_size_factors': self.config.sample_size_factors,
                'missingness_proportions': self.config.missingness_proportions,
                'mechanisms': self.config.mechanisms,
                'parallel_processing': self.config.parallel_processing,
                'n_workers': self.config.n_workers
            },
            'simulation_results': self.simulation_results,
            'stability_analysis': self.stability_analysis,
            'summary_statistics': self._calculate_simulation_summary(),
            'recommendations': self._generate_simulation_recommendations()
        }
        
        # Save final report
        report_file = Path(self.config.output_dir) / "simulation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_simulation_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for simulations."""
        summary = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'mechanism_performance': {},
            'stability_ranking': {}
        }
        
        # Calculate overall statistics
        all_f1_scores = []
        mechanism_performance = {}
        
        for dataset_name, dataset_results in self.simulation_results.items():
            for mechanism, mechanism_results in dataset_results.items():
                mechanism_f1_scores = []
                
                for missingness_results in mechanism_results.values():
                    for sample_results in missingness_results.values():
                        if 'raw_results' in sample_results:
                            f1_scores = [r['edge_f1_score'] for r in sample_results['raw_results']]
                            all_f1_scores.extend(f1_scores)
                            mechanism_f1_scores.extend(f1_scores)
                
                if mechanism_f1_scores:
                    if mechanism not in mechanism_performance:
                        mechanism_performance[mechanism] = []
                    mechanism_performance[mechanism].extend(mechanism_f1_scores)
        
        # Calculate mechanism performance
        for mechanism, scores in mechanism_performance.items():
            if scores:
                summary['mechanism_performance'][mechanism] = {
                    'mean_f1': np.mean(scores),
                    'std_f1': np.std(scores),
                    'n_observations': len(scores)
                }
        
        # Calculate overall performance
        if all_f1_scores:
            summary['overall_performance'] = {
                'mean_f1': np.mean(all_f1_scores),
                'std_f1': np.std(all_f1_scores),
                'min_f1': np.min(all_f1_scores),
                'max_f1': np.max(all_f1_scores)
            }
        
        return summary
    
    def _generate_simulation_recommendations(self) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Analyze mechanism performance
        if 'mechanism_performance' in self._calculate_simulation_summary():
            mechanism_performance = self._calculate_simulation_summary()['mechanism_performance']
            
            if mechanism_performance:
                best_mechanism = max(mechanism_performance.keys(), 
                                   key=lambda x: mechanism_performance[x]['mean_f1'])
                
                recommendations.append(f"Best performing mechanism: {best_mechanism}")
                
                # Analyze stability
                for mechanism, perf in mechanism_performance.items():
                    cv = perf['std_f1'] / perf['mean_f1'] if perf['mean_f1'] > 0 else 0
                    if cv < 0.1:
                        recommendations.append(f"{mechanism} shows excellent stability (CV: {cv:.3f})")
                    elif cv > 0.2:
                        recommendations.append(f"{mechanism} shows poor stability (CV: {cv:.3f})")
        
        recommendations.append("Consider increasing sample size for better performance stability")
        recommendations.append("Use parallel processing for large-scale simulations")
        recommendations.append("Monitor convergence of results across multiple runs")
        
        return recommendations

# Convenience function for running simulations
def run_monte_carlo_simulations(
    datasets: Dict[str, pd.DataFrame],
    ground_truths: Dict[str, nx.DiGraph],
    n_runs: int = 100,
    sample_size_factors: List[float] = None,
    missingness_proportions: List[float] = None,
    mechanisms: List[str] = None,
    parallel_processing: bool = True,
    output_dir: str = "results/monte_carlo_simulations"
) -> Dict[str, Any]:
    """
    Convenience function for running Monte Carlo simulations.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        ground_truths: Dictionary mapping dataset names to ground truth graphs
        n_runs: Number of Monte Carlo runs
        sample_size_factors: List of sample size factors to test
        missingness_proportions: List of missingness proportions to test
        mechanisms: List of mechanisms to test
        parallel_processing: Whether to use parallel processing
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing comprehensive simulation results
    """
    config = MonteCarloConfig(
        n_runs=n_runs,
        sample_size_factors=sample_size_factors,
        missingness_proportions=missingness_proportions,
        mechanisms=mechanisms,
        parallel_processing=parallel_processing,
        output_dir=output_dir
    )
    
    simulator = MonteCarloSimulator(config)
    return simulator.run_comprehensive_simulations(datasets, ground_truths)

if __name__ == "__main__":
    # Example usage
    print("Monte Carlo Simulator - Example Usage")
    print("Ready for comprehensive Monte Carlo simulations!")
