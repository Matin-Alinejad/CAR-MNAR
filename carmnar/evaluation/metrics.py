"""
Comprehensive Evaluation Metrics for Causal Discovery
======================================================

This module implements a comprehensive suite of evaluation metrics for assessing
causal discovery algorithm performance, including structural accuracy metrics
(precision, recall, F1-score), distance-based measures (Structural Hamming Distance),
and specialized analyses for missing data patterns. The framework enables rigorous
quantitative assessment of algorithmic performance across multiple evaluation dimensions.

The implementation provides both graph-level and component-level evaluation capabilities,
enabling detailed analysis of skeleton recovery, edge orientation accuracy, and
v-structure detection. Additional functionality includes missing data pattern analysis
and comprehensive result visualization for intuitive performance interpretation.

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalDiscoveryMetrics:
    """
    Comprehensive evaluation metrics for causal discovery algorithms.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def structural_hamming_distance(self, true_graph: nx.DiGraph, 
                                  inferred_graph: nx.DiGraph) -> int:
        """
        Calculate Structural Hamming Distance (SHD) between graphs.
        
        SHD is the minimum number of edge additions, deletions, or reversals
        needed to transform the inferred graph into the true graph.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            SHD value
        """
        # Get all possible edges
        all_nodes = set(true_graph.nodes()) | set(inferred_graph.nodes())
        all_edges = set()
        
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2:
                    all_edges.add((node1, node2))
        
        # Count differences
        differences = 0
        
        for edge in all_edges:
            true_has_edge = true_graph.has_edge(*edge)
            inferred_has_edge = inferred_graph.has_edge(*edge)
            
            if true_has_edge != inferred_has_edge:
                differences += 1
        
        return differences
    
    def relative_structural_hamming_distance(self, true_graph: nx.DiGraph, 
                                           inferred_graph: nx.DiGraph) -> float:
        """
        Calculate Relative Structural Hamming Distance (relSHD) between graphs.
        
        relSHD is the SHD normalized by the maximum possible SHD for graphs
        of the same size, providing a value between 0 and 1 that indicates
        the proportion of differences relative to the maximum possible differences.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            relSHD value (0.0 to 1.0)
        """
        # Calculate SHD
        shd = self.structural_hamming_distance(true_graph, inferred_graph)
        
        # Get all nodes from both graphs
        all_nodes = set(true_graph.nodes()) | set(inferred_graph.nodes())
        n_nodes = len(all_nodes)
        
        # Maximum possible SHD is n*(n-1) for directed graphs
        # This represents the maximum number of possible directed edges
        max_possible_shd = n_nodes * (n_nodes - 1)
        
        # Handle edge case where there are no nodes
        if max_possible_shd == 0:
            return 0.0
        
        # Calculate relative SHD
        rel_shd = shd / max_possible_shd
        
        # Ensure the result is between 0 and 1
        return min(rel_shd, 1.0)
    
    def edge_precision_recall(self, true_graph: nx.DiGraph, 
                            inferred_graph: nx.DiGraph) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for edge detection.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        if len(inferred_edges) == 0:
            precision = 0.0
        else:
            precision = len(true_edges & inferred_edges) / len(inferred_edges)
        
        if len(true_edges) == 0:
            recall = 0.0
        else:
            recall = len(true_edges & inferred_edges) / len(true_edges)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1_score
    
    def skeleton_precision_recall(self, true_graph: nx.DiGraph, 
                                inferred_graph: nx.DiGraph) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for skeleton (undirected edges).
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Convert to undirected graphs
        true_skeleton = true_graph.to_undirected()
        inferred_skeleton = inferred_graph.to_undirected()
        
        true_edges = set(true_skeleton.edges())
        inferred_edges = set(inferred_skeleton.edges())
        
        if len(inferred_edges) == 0:
            precision = 0.0
        else:
            precision = len(true_edges & inferred_edges) / len(inferred_edges)
        
        if len(true_edges) == 0:
            recall = 0.0
        else:
            recall = len(true_edges & inferred_edges) / len(true_edges)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1_score
    
    def orientation_precision_recall(self, true_graph: nx.DiGraph, 
                                   inferred_graph: nx.DiGraph) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for edge orientation.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Get common skeleton edges
        true_skeleton = true_graph.to_undirected()
        inferred_skeleton = inferred_graph.to_undirected()
        
        true_skeleton_edges = set(true_skeleton.edges())
        inferred_skeleton_edges = set(inferred_skeleton.edges())
        common_skeleton_edges = true_skeleton_edges & inferred_skeleton_edges
        
        if len(common_skeleton_edges) == 0:
            return 0.0, 0.0, 0.0
        
        # Count correctly oriented edges
        correctly_oriented = 0
        total_oriented = 0
        
        for edge in common_skeleton_edges:
            u, v = edge
            
            # Check if both graphs have the same orientation
            true_has_u_to_v = true_graph.has_edge(u, v)
            true_has_v_to_u = true_graph.has_edge(v, u)
            inferred_has_u_to_v = inferred_graph.has_edge(u, v)
            inferred_has_v_to_u = inferred_graph.has_edge(v, u)
            
            # Count oriented edges in inferred graph
            if inferred_has_u_to_v or inferred_has_v_to_u:
                total_oriented += 1
                
                # Check if orientation matches
                if (true_has_u_to_v and inferred_has_u_to_v) or \
                   (true_has_v_to_u and inferred_has_v_to_u):
                    correctly_oriented += 1
        
        if total_oriented == 0:
            precision = 0.0
        else:
            precision = correctly_oriented / total_oriented
        
        # For recall, we need to count how many true orientations were captured
        true_oriented = 0
        for edge in common_skeleton_edges:
            u, v = edge
            if true_graph.has_edge(u, v) or true_graph.has_edge(v, u):
                true_oriented += 1
        
        if true_oriented == 0:
            recall = 0.0
        else:
            recall = correctly_oriented / true_oriented
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1_score
    
    def adjacency_precision_recall(self, true_graph: nx.DiGraph, 
                                 inferred_graph: nx.DiGraph) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for adjacency detection.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Get all possible adjacencies
        all_nodes = set(true_graph.nodes()) | set(inferred_graph.nodes())
        
        true_adjacencies = set()
        inferred_adjacencies = set()
        
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2:
                    # Check if nodes are adjacent in true graph
                    if true_graph.has_edge(node1, node2) or true_graph.has_edge(node2, node1):
                        true_adjacencies.add(frozenset([node1, node2]))
                    
                    # Check if nodes are adjacent in inferred graph
                    if inferred_graph.has_edge(node1, node2) or inferred_graph.has_edge(node2, node1):
                        inferred_adjacencies.add(frozenset([node1, node2]))
        
        if len(inferred_adjacencies) == 0:
            precision = 0.0
        else:
            precision = len(true_adjacencies & inferred_adjacencies) / len(inferred_adjacencies)
        
        if len(true_adjacencies) == 0:
            recall = 0.0
        else:
            recall = len(true_adjacencies & inferred_adjacencies) / len(true_adjacencies)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1_score
    
    def comprehensive_evaluation(self, true_graph: nx.DiGraph, 
                               inferred_graph: nx.DiGraph) -> Dict[str, float]:
        """
        Perform comprehensive evaluation of inferred graph.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Edge metrics
        edge_precision, edge_recall, edge_f1 = self.edge_precision_recall(true_graph, inferred_graph)
        
        # Skeleton metrics
        skeleton_precision, skeleton_recall, skeleton_f1 = self.skeleton_precision_recall(true_graph, inferred_graph)
        
        # Orientation metrics
        orientation_precision, orientation_recall, orientation_f1 = self.orientation_precision_recall(true_graph, inferred_graph)
        
        # Adjacency metrics
        adjacency_precision, adjacency_recall, adjacency_f1 = self.adjacency_precision_recall(true_graph, inferred_graph)
        
        # Structural Hamming Distance
        shd = self.structural_hamming_distance(true_graph, inferred_graph)
        
        # Relative Structural Hamming Distance
        rel_shd = self.relative_structural_hamming_distance(true_graph, inferred_graph)
        
        # Additional metrics
        true_edges = len(true_graph.edges())
        inferred_edges = len(inferred_graph.edges())
        correct_edges = len(set(true_graph.edges()) & set(inferred_graph.edges()))
        
        return {
            # Edge metrics
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'edge_f1_score': edge_f1,
            
            # Skeleton metrics
            'skeleton_precision': skeleton_precision,
            'skeleton_recall': skeleton_recall,
            'skeleton_f1_score': skeleton_f1,
            
            # Orientation metrics
            'orientation_precision': orientation_precision,
            'orientation_recall': orientation_recall,
            'orientation_f1_score': orientation_f1,
            
            # Adjacency metrics
            'adjacency_precision': adjacency_precision,
            'adjacency_recall': adjacency_recall,
            'adjacency_f1_score': adjacency_f1,
            
            # Structural metrics
            'structural_hamming_distance': shd,
            'relative_structural_hamming_distance': rel_shd,
            
            # Count metrics
            'true_edges': true_edges,
            'inferred_edges': inferred_edges,
            'correct_edges': correct_edges,
            'false_positive_edges': inferred_edges - correct_edges,
            'false_negative_edges': true_edges - correct_edges
        }


class MissingDataAnalysis:
    """
    Analysis tools for understanding the impact of missing data patterns.
    """
    
    def __init__(self):
        """Initialize the missing data analyzer."""
        pass
    
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
                'original_median': original_values.median(),
                'mnar_mean': mnar_values.mean(),
                'mnar_std': mnar_values.std(),
                'mnar_median': mnar_values.median(),
                'mean_difference': original_values.mean() - mnar_values.mean(),
                'median_difference': original_values.median() - mnar_values.median(),
                'bias_ratio': (original_values.mean() - mnar_values.mean()) / original_values.std(),
                'correlation_with_missing': self._correlation_with_missing(original_data, var, mnar_data)
            }
        
        return analysis
    
    def _correlation_with_missing(self, original_data: pd.DataFrame, 
                                variable: str, mnar_data: pd.DataFrame) -> float:
        """
        Calculate correlation between variable values and missingness.
        
        Args:
            original_data: Original complete dataset
            variable: Variable name
            mnar_data: Dataset with missing values
            
        Returns:
            Correlation coefficient
        """
        original_values = original_data[variable].values
        missing_indicator = mnar_data[variable].isna().astype(int).values
        
        try:
            correlation, _ = pearsonr(original_values, missing_indicator)
            return correlation
        except:
            return 0.0
    
    def bias_analysis(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze bias in causal discovery results across missing data percentages.
        
        Args:
            results_df: DataFrame with experimental results
            
        Returns:
            Dictionary with bias analysis
        """
        bias_analysis = {}
        
        # Group by dataset and missing percentage
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            
            # Calculate performance degradation
            baseline_performance = dataset_results[dataset_results['missing_percentage'] == 0.0]
            if len(baseline_performance) > 0:
                baseline_f1 = baseline_performance['edge_f1_score'].iloc[0]
                
                degradation = []
                for _, row in dataset_results.iterrows():
                    if row['missing_percentage'] > 0:
                        degradation.append(baseline_f1 - row['edge_f1_score'])
                
                bias_analysis[dataset] = {
                    'baseline_f1': baseline_f1,
                    'average_degradation': np.mean(degradation) if degradation else 0,
                    'max_degradation': np.max(degradation) if degradation else 0,
                    'degradation_rate': np.mean(degradation) / baseline_f1 if baseline_f1 > 0 else 0
                }
        
        return bias_analysis


class ResultsVisualizer:
    """
    Visualization tools for experimental results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_performance_vs_missing(self, results_df: pd.DataFrame, 
                                  metric: str = 'edge_f1_score',
                                  save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics vs missing data percentage.
        
        Args:
            results_df: DataFrame with experimental results
            metric: Metric to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            plt.plot(dataset_results['missing_percentage'] * 100, 
                    dataset_results[metric], 
                    marker='o', linewidth=2, label=dataset)
        
        plt.xlabel('Missing Data Percentage (%)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Missing Data Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, results_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple metrics.
        
        Args:
            results_df: DataFrame with experimental results
            save_path: Path to save the plot
        """
        metrics = ['edge_precision', 'edge_recall', 'edge_f1_score', 
                  'skeleton_precision', 'skeleton_recall', 'skeleton_f1_score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for dataset in results_df['dataset'].unique():
                dataset_results = results_df[results_df['dataset'] == dataset]
                ax.plot(dataset_results['missing_percentage'] * 100, 
                       dataset_results[metric], 
                       marker='o', linewidth=2, label=dataset)
            
            ax.set_xlabel('Missing Data Percentage (%)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_heatmap(self, results_df: pd.DataFrame, 
                    metric: str = 'edge_f1_score',
                    save_path: Optional[str] = None) -> None:
        """
        Create heatmap of results.
        
        Args:
            results_df: DataFrame with experimental results
            metric: Metric to plot
            save_path: Path to save the plot
        """
        # Pivot data for heatmap
        pivot_data = results_df.pivot(index='dataset', 
                                    columns='missing_percentage', 
                                    values=metric)
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(pivot_data, annot=True, cmap='viridis', 
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        plt.title(f'{metric.replace("_", " ").title()} Heatmap')
        plt.xlabel('Missing Data Percentage')
        plt.ylabel('Dataset')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create sample graphs
    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
    
    inferred_graph = nx.DiGraph()
    inferred_graph.add_edges_from([('A', 'B'), ('B', 'C')])
    
    # Test metrics
    metrics = CausalDiscoveryMetrics()
    results = metrics.comprehensive_evaluation(true_graph, inferred_graph)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
