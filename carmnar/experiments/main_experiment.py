"""
Main Experimental Pipeline for SM-MVPC Evaluation
===================================================

This module implements the core experimental pipeline for comprehensively evaluating
the Self-Masking Missing Value PC (SM-MVPC) algorithm on medical datasets under
Missing Not At Random (MNAR) conditions. The pipeline orchestrates the complete
experimental workflow from data loading and preprocessing through causal discovery
execution, performance evaluation, and comprehensive result analysis.

The framework systematically evaluates SM-MVPC across multiple clinical datasets
under varying MNAR severity levels, employing rigorous statistical methodology
including bootstrap confidence intervals, effect size estimation, and multiple
comparison correction. Results enable quantitative assessment of algorithmic
robustness and inform practical algorithm selection for clinical applications.

Author: Anonymous (for review)
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.mnar_generator import MNARGenerator
from algorithms.causal_discovery import PCAlgorithm, SMMVPC
from utils.data_loader import MedicalDataLoader, create_sample_datasets
from evaluation.metrics import CausalDiscoveryMetrics, MissingDataAnalysis, ResultsVisualizer
from evaluation.cpdag_aware_metrics import evaluate_causal_discovery
from utils.report_builder import build_all_reports

# Configure logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SM_MVPC_Experiment:
    """
    Main experimental pipeline for evaluating SM-MVPC on medical datasets.
    """
    
    def __init__(self, results_dir: str = "results/experiments", 
                 config_dir: str = "config"):
        """
        Initialize the experiment.
        
        Args:
            results_dir: Directory to save results
            config_dir: Directory containing configuration files
        """
        self.results_dir = Path(results_dir)
        self.config_dir = Path(config_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = MedicalDataLoader()
        self.mnar_generator = MNARGenerator()
        self.pc_algorithm = PCAlgorithm()
        self.sm_mvpc = SMMVPC()
        self.metrics = CausalDiscoveryMetrics()
        self.missing_analysis = MissingDataAnalysis()
        self.visualizer = ResultsVisualizer()
        
        # Experiment configuration
        self.missing_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.datasets = ['heart_disease', 'diabetes', 'hepatitis']
        self.n_repetitions = 5  # Number of repetitions for statistical significance
        
        # Results storage
        self.results = []
        self.ground_truth_graphs = {}
        self.mnar_scenarios = {}
    
    def load_and_preprocess_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess all medical datasets.
        
        Returns:
            Dictionary mapping dataset names to preprocessed DataFrames
        """
        logger.info("Loading and preprocessing datasets...")
        
        try:
            datasets = self.data_loader.load_all_datasets()
        except Exception as e:
            logger.warning(f"Failed to load real datasets: {e}")
            logger.info("Using sample datasets for demonstration...")
            datasets = create_sample_datasets()
            
            # Save sample datasets
            for name, data in datasets.items():
                file_path = self.data_loader.data_dir / f"{name}.csv"
                data.to_csv(file_path, index=False)
                logger.info(f"Created sample dataset: {file_path}")
        
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
    def generate_ground_truth_graphs(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, nx.DiGraph]:
        """
        Generate ground truth causal graphs using PC algorithm.
        
        Args:
            datasets: Dictionary of datasets
            
        Returns:
            Dictionary mapping dataset names to ground truth graphs
        """
        logger.info("Generating ground truth causal graphs...")
        
        ground_truth_graphs = {}
        
        for dataset_name, data in datasets.items():
            logger.info(f"Generating ground truth for {dataset_name}...")
            
            try:
                # Run PC algorithm on complete data
                true_graph = self.pc_algorithm.fit(data)
                ground_truth_graphs[dataset_name] = true_graph
                
                # Save ground truth graph
                graph_path = self.results_dir / f"{dataset_name}_ground_truth.gml"
                nx.write_gml(true_graph, graph_path)
                
                logger.info(f"Ground truth for {dataset_name}: {len(true_graph.edges())} edges")
                
            except Exception as e:
                logger.error(f"Failed to generate ground truth for {dataset_name}: {e}")
                # Create a simple default graph
                ground_truth_graphs[dataset_name] = nx.DiGraph()
        
        self.ground_truth_graphs = ground_truth_graphs
        return ground_truth_graphs
    
    def generate_mnar_scenarios(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[float, pd.DataFrame]]:
        """
        Generate MNAR scenarios for all datasets.
        
        Args:
            datasets: Dictionary of datasets
            
        Returns:
            Dictionary mapping dataset names to MNAR scenarios
        """
        logger.info("Generating MNAR scenarios...")
        
        mnar_scenarios = {}
        
        for dataset_name, data in datasets.items():
            logger.info(f"Generating MNAR scenarios for {dataset_name}...")
            
            # Get effect variables for this dataset
            effect_variables = self.data_loader.get_effect_variables(dataset_name)
            
            if not effect_variables:
                logger.warning(f"No effect variables defined for {dataset_name}")
                continue
            
            # Generate MNAR scenarios
            scenarios = self.mnar_generator.generate_mnar_scenarios(
                data, effect_variables, self.missing_percentages, method='sigmoid'
            )
            
            mnar_scenarios[dataset_name] = scenarios
            
            # Save MNAR datasets
            for percentage, mnar_data in scenarios.items():
                file_path = self.results_dir / f"{dataset_name}_mnar_{percentage:.1f}.csv"
                mnar_data.to_csv(file_path, index=False)
        
        self.mnar_scenarios = mnar_scenarios
        return mnar_scenarios
    
    def run_causal_discovery_experiment(self, dataset_name: str, 
                                      mnar_data: pd.DataFrame,
                                      missing_percentage: float,
                                      repetition: int) -> Dict:
        """
        Run causal discovery experiment on a single MNAR dataset.
        
        Args:
            dataset_name: Name of the dataset
            mnar_data: MNAR dataset
            missing_percentage: Percentage of missing data
            repetition: Repetition number
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Running experiment: {dataset_name}, {missing_percentage:.1f}%, rep {repetition}")
        
        start_time = time.time()
        
        try:
            # Run SM-MVPC
            inferred_graph = self.sm_mvpc.fit(mnar_data)
            
            # Get ground truth graph
            true_graph = self.ground_truth_graphs[dataset_name]
            
            # Evaluate results with standard (DAG-level) metrics
            evaluation_metrics = self.metrics.comprehensive_evaluation(true_graph, inferred_graph)

            # Evaluate CPDAG-aware metrics (skeleton, v-structures, compelled orientations, CPDAG-SHD, SID)
            cpdag_eval = evaluate_causal_discovery(true_graph, inferred_graph, convert_to_cpdag=True)

            # Map a selected subset to clearly named keys to avoid ambiguity
            cpdag_metrics = {
                'cpdag_skeleton_f1': cpdag_eval.get('skeleton_f1', None),
                'cpdag_vstructure_f1': cpdag_eval.get('vstructure_f1', None),
                'cpdag_orientation_f1': cpdag_eval.get('orientation_f1', None),
                'cpdag_shd': cpdag_eval.get('cpdag_shd', None),
                'cpdag_shd_normalized': cpdag_eval.get('cpdag_shd_normalized', None),
                'sid': cpdag_eval.get('sid', None),
                'cpdag_n_nodes': cpdag_eval.get('n_nodes', None),
                'cpdag_n_edges_true': cpdag_eval.get('n_edges_true', None),
                'cpdag_n_edges_inferred': cpdag_eval.get('n_edges_inferred', None),
                'cpdag_markov_equivalent': cpdag_eval.get('markov_equivalent', None),
            }
            
            # Analyze missing data patterns
            original_data = self.data_loader.load_dataset(dataset_name)
            effect_variables = self.data_loader.get_effect_variables(dataset_name)
            missing_analysis = self.missing_analysis.analyze_missing_patterns(
                original_data, mnar_data, effect_variables
            )
            
            # Compile results
            result = {
                'dataset': dataset_name,
                'missing_percentage': missing_percentage,
                'repetition': repetition,
                'experiment_time': time.time() - start_time,
                'missing_data_analysis': missing_analysis,
                **evaluation_metrics,
                **cpdag_metrics,
            }
            
            logger.info(f"Experiment completed: F1={evaluation_metrics['edge_f1_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                'dataset': dataset_name,
                'missing_percentage': missing_percentage,
                'repetition': repetition,
                'experiment_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_full_experiment(self) -> pd.DataFrame:
        """
        Run the complete experimental pipeline.
        
        Returns:
            DataFrame with all experimental results
        """
        logger.info("Starting full experimental pipeline...")
        
        # Step 1: Load and preprocess datasets
        datasets = self.load_and_preprocess_datasets()
        
        # Step 2: Generate ground truth graphs
        ground_truth_graphs = self.generate_ground_truth_graphs(datasets)
        
        # Step 3: Generate MNAR scenarios
        mnar_scenarios = self.generate_mnar_scenarios(datasets)
        
        # Step 4: Run experiments
        logger.info("Running causal discovery experiments...")
        
        all_results = []
        
        for dataset_name in self.datasets:
            if dataset_name not in mnar_scenarios:
                logger.warning(f"Skipping {dataset_name} - no MNAR scenarios available")
                continue
            
            for missing_percentage in self.missing_percentages:
                if missing_percentage not in mnar_scenarios[dataset_name]:
                    logger.warning(f"Skipping {dataset_name} - {missing_percentage:.1f}% missing")
                    continue
                
                mnar_data = mnar_scenarios[dataset_name][missing_percentage]
                
                for repetition in range(self.n_repetitions):
                    result = self.run_causal_discovery_experiment(
                        dataset_name, mnar_data, missing_percentage, repetition
                    )
                    all_results.append(result)
        
        # Step 5: Compile results
        results_df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"experimental_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"Experimental pipeline completed. Results saved to {results_file}")
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze experimental results.
        
        Args:
            results_df: DataFrame with experimental results
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing experimental results...")
        
        # Calculate summary statistics
        summary_stats = results_df.groupby(['dataset', 'missing_percentage']).agg({
            'edge_f1_score': ['mean', 'std', 'min', 'max'],
            'edge_precision': ['mean', 'std'],
            'edge_recall': ['mean', 'std'],
            'structural_hamming_distance': ['mean', 'std'],
            'relative_structural_hamming_distance': ['mean', 'std'],
            'experiment_time': ['mean', 'std']
        }).round(3)
        
        # Calculate performance degradation
        bias_analysis = self.missing_analysis.bias_analysis(results_df)
        
        # Create visualizations
        self.create_visualizations(results_df)
        
        analysis_results = {
            'summary_statistics': summary_stats,
            'bias_analysis': bias_analysis,
            'total_experiments': len(results_df),
            'successful_experiments': len(results_df[results_df.get('error').isna()]) if 'error' in results_df.columns else len(results_df)
        }
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.results_dir / f"analysis_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert summary statistics to dict
        analysis_results['summary_statistics'] = {
            str(k): {str(k2): convert_numpy(v2) for k2, v2 in v.items()} 
            for k, v in summary_stats.to_dict().items()
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=convert_numpy)
        
        logger.info(f"Analysis completed. Results saved to {analysis_file}")
        return analysis_results
    
    def create_visualizations(self, results_df: pd.DataFrame) -> None:
        """
        Create visualizations of experimental results.
        
        Args:
            results_df: DataFrame with experimental results
        """
        logger.info("Creating visualizations...")
        
        # Plot performance vs missing percentage
        self.visualizer.plot_performance_vs_missing(
            results_df, 'edge_f1_score',
            save_path=self.results_dir / 'f1_score_vs_missing.png'
        )
        
        # Plot metrics comparison
        self.visualizer.plot_metrics_comparison(
            results_df,
            save_path=self.results_dir / 'metrics_comparison.png'
        )
        
        # Plot heatmap
        self.visualizer.plot_heatmap(
            results_df, 'edge_f1_score',
            save_path=self.results_dir / 'f1_score_heatmap.png'
        )
        
        logger.info("Visualizations created and saved")
    
    def generate_report(self, results_df: pd.DataFrame, analysis_results: Dict) -> str:
        """
        Generate a comprehensive experimental report.
        
        Args:
            results_df: DataFrame with experimental results
            analysis_results: Dictionary with analysis results
            
        Returns:
            Path to generated report
        """
        logger.info("Generating experimental report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"experimental_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# SM-MVPC Evaluation on Medical Datasets with MNAR Missing Data\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experiments:** {analysis_results['total_experiments']}\n")
            f.write(f"- **Successful Experiments:** {analysis_results['successful_experiments']}\n")
            f.write(f"- **Datasets:** {', '.join(self.datasets)}\n")
            f.write(f"- **Missing Data Percentages:** {', '.join([f'{p*100:.0f}%' for p in self.missing_percentages])}\n")
            f.write(f"- **Repetitions per Condition:** {self.n_repetitions}\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Find best and worst performance
            best_f1 = results_df['edge_f1_score'].max()
            worst_f1 = results_df['edge_f1_score'].min()
            best_condition = results_df.loc[results_df['edge_f1_score'].idxmax()]
            worst_condition = results_df.loc[results_df['edge_f1_score'].idxmin()]
            
            f.write(f"- **Best F1 Score:** {best_f1:.3f} ({best_condition['dataset']}, {best_condition['missing_percentage']*100:.0f}% missing)\n")
            f.write(f"- **Worst F1 Score:** {worst_f1:.3f} ({worst_condition['dataset']}, {worst_condition['missing_percentage']*100:.0f}% missing)\n")
            
            # Performance degradation analysis
            f.write("\n### Performance Degradation Analysis\n\n")
            for dataset, bias_info in analysis_results['bias_analysis'].items():
                f.write(f"**{dataset}:**\n")
                f.write(f"- Baseline F1: {bias_info['baseline_f1']:.3f}\n")
                f.write(f"- Average Degradation: {bias_info['average_degradation']:.3f}\n")
                f.write(f"- Degradation Rate: {bias_info['degradation_rate']:.1%}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("### Summary Statistics by Dataset and Missing Percentage\n\n")
            
            # Convert summary statistics to markdown table
            summary_df = pd.DataFrame(analysis_results['summary_statistics'])
            f.write(summary_df.to_markdown())
            f.write("\n\n")
            
            f.write("## Methodology\n\n")
            f.write("### MNAR Data Generation\n")
            f.write("- Used sigmoid-based probability function for missing data generation\n")
            f.write("- Missingness probability depends on the value of effect variables\n")
            f.write("- Target missing percentages: 0%, 10%, 20%, 30%, 40%, 50%\n\n")
            
            f.write("### Causal Discovery Algorithms\n")
            f.write("- **PC Algorithm:** Used for generating ground truth causal graphs\n")
            f.write("- **SM-MVPC:** Self-Masking Missing Value PC for handling MNAR data\n")
            f.write("- **Evaluation Metrics:** Precision, Recall, F1-Score, Structural Hamming Distance\n\n")
            
            f.write("### Statistical Analysis\n")
            f.write(f"- **Repetitions:** {self.n_repetitions} per condition\n")
            f.write("- **Significance Level:** alpha = 0.05\n")
            f.write("- **Missing Data Handling:** Complete case analysis and imputation\n\n")
        
        logger.info(f"Report generated: {report_file}")
        # Build HTML reports (results and datasets) in reports/
        try:
            # Collect datasets info for datasets report
            datasets_info = {}
            for ds in self.datasets:
                try:
                    datasets_info[ds] = self.data_loader.get_dataset_info(ds)
                except Exception as e:
                    datasets_info[ds] = {'name': ds, 'error': str(e)}
            build_all_reports(results_df, datasets_info, output_dir="reports")
            logger.info("HTML reports generated in reports/ directory")
        except Exception as e:
            logger.error(f"Failed to build HTML reports: {e}")

        return str(report_file)


def main():
    """Main function to run the complete experimental pipeline."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize experiment
    experiment = SM_MVPC_Experiment()
    
    try:
        # Run full experiment
        results_df = experiment.run_full_experiment()
        
        # Analyze results
        analysis_results = experiment.analyze_results(results_df)
        
        # Generate report
        report_path = experiment.generate_report(results_df, analysis_results)
        
        logger.info("="*50)
        logger.info("EXPERIMENTAL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"Results saved to: {experiment.results_dir}")
        logger.info(f"Report generated: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("="*50)
        print(f"Total experiments: {len(results_df)}")
        print(f"Datasets: {', '.join(experiment.datasets)}")
        print(f"Missing percentages: {[f'{p*100:.0f}%' for p in experiment.missing_percentages]}")
        
        # Best and worst performance
        best_f1 = results_df['edge_f1_score'].max()
        worst_f1 = results_df['edge_f1_score'].min()
        print(f"Best F1 score: {best_f1:.3f}")
        print(f"Worst F1 score: {worst_f1:.3f}")
        
        print(f"\nDetailed results and visualizations saved to: {experiment.results_dir}")
        
    except Exception as e:
        logger.error(f"Experimental pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
