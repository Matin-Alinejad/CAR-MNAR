"""
Advanced Structural Metrics Framework for Causal Discovery Evaluation

This module provides comprehensive structural metrics for evaluating causal discovery
algorithms, including relSHD, adjacency metrics, orientation accuracy, and specialized
diagnostic measures suitable for academic publication.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StructuralMetricsConfig:
    """Configuration for structural metrics calculation."""
    include_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    random_seed: int = 42
    detailed_breakdown: bool = True
    edge_weight_analysis: bool = False

class AdvancedStructuralMetrics:
    """
    Advanced structural metrics calculator for causal discovery evaluation.
    
    This class provides comprehensive structural metrics including relSHD,
    adjacency metrics, orientation accuracy, and specialized diagnostic
    measures for academic evaluation.
    """
    
    def __init__(self, config: StructuralMetricsConfig = None):
        """
        Initialize the advanced structural metrics calculator.
        
        Args:
            config: Configuration object for metrics calculation
        """
        self.config = config or StructuralMetricsConfig()
        np.random.seed(self.config.random_seed)
        
        logger.info("Initialized Advanced Structural Metrics Calculator")
    
    def calculate_comprehensive_metrics(self, 
                                      true_graph: nx.DiGraph, 
                                      inferred_graph: nx.DiGraph,
                                      dataset_name: str = "unknown") -> Dict[str, Any]:
        """
        Calculate comprehensive structural metrics for causal discovery evaluation.
        
        Args:
            true_graph: Ground truth causal graph
            inferred_graph: Inferred causal graph
            dataset_name: Name of the dataset for context
            
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.debug(f"Calculating comprehensive metrics for {dataset_name}")
        
        # Ensure graphs have the same nodes
        true_graph, inferred_graph = self._align_graphs(true_graph, inferred_graph)
        
        # Calculate basic structural metrics
        basic_metrics = self._calculate_basic_structural_metrics(true_graph, inferred_graph)
        
        # Calculate adjacency metrics
        adjacency_metrics = self._calculate_adjacency_metrics(true_graph, inferred_graph)
        
        # Calculate orientation metrics
        orientation_metrics = self._calculate_orientation_metrics(true_graph, inferred_graph)
        
        # Calculate specialized metrics
        specialized_metrics = self._calculate_specialized_metrics(true_graph, inferred_graph)
        
        # Calculate graph topology metrics
        topology_metrics = self._calculate_topology_metrics(true_graph, inferred_graph)
        
        # Combine all metrics
        comprehensive_metrics = {
            'dataset_name': dataset_name,
            'basic_structural': basic_metrics,
            'adjacency_metrics': adjacency_metrics,
            'orientation_metrics': orientation_metrics,
            'specialized_metrics': specialized_metrics,
            'topology_metrics': topology_metrics,
            'graph_statistics': self._calculate_graph_statistics(true_graph, inferred_graph)
        }
        
        # Add confidence intervals if requested
        if self.config.include_confidence_intervals:
            comprehensive_metrics['confidence_intervals'] = self._calculate_confidence_intervals(
                comprehensive_metrics
            )
        
        return comprehensive_metrics
    
    def _align_graphs(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Align graphs to have the same nodes for fair comparison."""
        # Get all unique nodes
        all_nodes = set(true_graph.nodes()) | set(inferred_graph.nodes())
        
        # Create aligned graphs
        aligned_true = nx.DiGraph()
        aligned_inferred = nx.DiGraph()
        
        # Add all nodes
        for node in all_nodes:
            aligned_true.add_node(node)
            aligned_inferred.add_node(node)
        
        # Add edges from true graph
        for edge in true_graph.edges():
            aligned_true.add_edge(*edge)
        
        # Add edges from inferred graph
        for edge in inferred_graph.edges():
            aligned_inferred.add_edge(*edge)
        
        return aligned_true, aligned_inferred
    
    def _calculate_basic_structural_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate basic structural metrics including relSHD."""
        # Calculate SHD
        shd = self._calculate_structural_hamming_distance(true_graph, inferred_graph)
        
        # Calculate relSHD
        n_nodes = true_graph.number_of_nodes()
        max_possible_edges = n_nodes * (n_nodes - 1)
        rel_shd = shd / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate edge counts
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Calculate edge differences
        correct_edges = true_edges & inferred_edges
        missing_edges = true_edges - inferred_edges
        extra_edges = inferred_edges - true_edges
        
        # Calculate skeleton differences
        true_skeleton = set(frozenset(edge) for edge in true_edges)
        inferred_skeleton = set(frozenset(edge) for edge in inferred_edges)
        
        correct_skeleton = true_skeleton & inferred_skeleton
        missing_skeleton = true_skeleton - inferred_skeleton
        extra_skeleton = inferred_skeleton - true_skeleton
        
        return {
            'structural_hamming_distance': shd,
            'relative_structural_hamming_distance': rel_shd,
            'n_nodes': n_nodes,
            'n_true_edges': len(true_edges),
            'n_inferred_edges': len(inferred_edges),
            'n_correct_edges': len(correct_edges),
            'n_missing_edges': len(missing_edges),
            'n_extra_edges': len(extra_edges),
            'n_correct_skeleton': len(correct_skeleton),
            'n_missing_skeleton': len(missing_skeleton),
            'n_extra_skeleton': len(extra_skeleton),
            'edge_accuracy': len(correct_edges) / len(true_edges) if len(true_edges) > 0 else 0,
            'skeleton_accuracy': len(correct_skeleton) / len(true_skeleton) if len(true_skeleton) > 0 else 0
        }
    
    def _calculate_structural_hamming_distance(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> int:
        """Calculate Structural Hamming Distance between two graphs."""
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Count missing edges (in true but not in inferred)
        missing_edges = true_edges - inferred_edges
        
        # Count extra edges (in inferred but not in true)
        extra_edges = inferred_edges - true_edges
        
        # Count reversed edges
        reversed_edges = 0
        for edge in true_edges:
            if (edge[1], edge[0]) in inferred_edges and edge not in inferred_edges:
                reversed_edges += 1
        
        # SHD = missing + extra + reversed
        shd = len(missing_edges) + len(extra_edges) + reversed_edges
        
        return shd
    
    def _calculate_adjacency_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate adjacency metrics (precision, recall, F1-score)."""
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Calculate skeleton metrics
        true_skeleton = set(frozenset(edge) for edge in true_edges)
        inferred_skeleton = set(frozenset(edge) for edge in inferred_edges)
        
        # Edge metrics
        edge_precision = len(true_edges & inferred_edges) / len(inferred_edges) if len(inferred_edges) > 0 else 0
        edge_recall = len(true_edges & inferred_edges) / len(true_edges) if len(true_edges) > 0 else 0
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
        
        # Skeleton metrics
        skeleton_precision = len(true_skeleton & inferred_skeleton) / len(inferred_skeleton) if len(inferred_skeleton) > 0 else 0
        skeleton_recall = len(true_skeleton & inferred_skeleton) / len(true_skeleton) if len(true_skeleton) > 0 else 0
        skeleton_f1 = 2 * skeleton_precision * skeleton_recall / (skeleton_precision + skeleton_recall) if (skeleton_precision + skeleton_recall) > 0 else 0
        
        # Adjacency matrix metrics
        adjacency_metrics = self._calculate_adjacency_matrix_metrics(true_graph, inferred_graph)
        
        return {
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'edge_f1_score': edge_f1,
            'skeleton_precision': skeleton_precision,
            'skeleton_recall': skeleton_recall,
            'skeleton_f1_score': skeleton_f1,
            'adjacency_precision': adjacency_metrics['precision'],
            'adjacency_recall': adjacency_metrics['recall'],
            'adjacency_f1_score': adjacency_metrics['f1_score']
        }
    
    def _calculate_adjacency_matrix_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate adjacency matrix-based metrics."""
        nodes = list(true_graph.nodes())
        n_nodes = len(nodes)
        
        # Create adjacency matrices
        true_adj = np.zeros((n_nodes, n_nodes))
        inferred_adj = np.zeros((n_nodes, n_nodes))
        
        # Fill true adjacency matrix
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if true_graph.has_edge(node1, node2):
                    true_adj[i, j] = 1
        
        # Fill inferred adjacency matrix
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if inferred_graph.has_edge(node1, node2):
                    inferred_adj[i, j] = 1
        
        # Calculate metrics
        precision = precision_score(true_adj.flatten(), inferred_adj.flatten(), zero_division=0)
        recall = recall_score(true_adj.flatten(), inferred_adj.flatten(), zero_division=0)
        f1 = f1_score(true_adj.flatten(), inferred_adj.flatten(), zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_orientation_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate orientation accuracy metrics."""
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        # Find common skeleton edges
        true_skeleton = set(frozenset(edge) for edge in true_edges)
        inferred_skeleton = set(frozenset(edge) for edge in inferred_edges)
        common_skeleton = true_skeleton & inferred_skeleton
        
        # Count correctly oriented edges
        correctly_oriented = 0
        total_common_edges = 0
        
        for skeleton_edge in common_skeleton:
            skeleton_list = list(skeleton_edge)
            edge1 = (skeleton_list[0], skeleton_list[1])
            edge2 = (skeleton_list[1], skeleton_list[0])
            
            # Check if both orientations exist in true graph
            true_has_edge1 = edge1 in true_edges
            true_has_edge2 = edge2 in true_edges
            
            # Check if both orientations exist in inferred graph
            inferred_has_edge1 = edge1 in inferred_edges
            inferred_has_edge2 = edge2 in inferred_edges
            
            if true_has_edge1 and inferred_has_edge1:
                correctly_oriented += 1
                total_common_edges += 1
            elif true_has_edge2 and inferred_has_edge2:
                correctly_oriented += 1
                total_common_edges += 1
            elif (true_has_edge1 or true_has_edge2) and (inferred_has_edge1 or inferred_has_edge2):
                total_common_edges += 1
        
        # Calculate orientation metrics
        orientation_accuracy = correctly_oriented / total_common_edges if total_common_edges > 0 else 0
        
        # Calculate orientation precision and recall
        orientation_precision = correctly_oriented / len(inferred_edges) if len(inferred_edges) > 0 else 0
        orientation_recall = correctly_oriented / len(true_edges) if len(true_edges) > 0 else 0
        orientation_f1 = 2 * orientation_precision * orientation_recall / (orientation_precision + orientation_recall) if (orientation_precision + orientation_recall) > 0 else 0
        
        return {
            'orientation_accuracy': orientation_accuracy,
            'orientation_precision': orientation_precision,
            'orientation_recall': orientation_recall,
            'orientation_f1_score': orientation_f1,
            'correctly_oriented_edges': correctly_oriented,
            'total_common_edges': total_common_edges
        }
    
    def _calculate_specialized_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate specialized diagnostic metrics."""
        # Calculate edge weight analysis if requested
        edge_weight_metrics = {}
        if self.config.edge_weight_analysis:
            edge_weight_metrics = self._calculate_edge_weight_metrics(true_graph, inferred_graph)
        
        # Calculate graph similarity metrics
        similarity_metrics = self._calculate_graph_similarity_metrics(true_graph, inferred_graph)
        
        # Calculate structural complexity metrics
        complexity_metrics = self._calculate_structural_complexity_metrics(true_graph, inferred_graph)
        
        return {
            'edge_weight_metrics': edge_weight_metrics,
            'similarity_metrics': similarity_metrics,
            'complexity_metrics': complexity_metrics
        }
    
    def _calculate_edge_weight_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate edge weight analysis metrics."""
        # This would be implemented if edge weights are available
        return {
            'weight_correlation': 0.0,
            'weight_mae': 0.0,
            'weight_rmse': 0.0
        }
    
    def _calculate_graph_similarity_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate graph similarity metrics."""
        # Jaccard similarity
        true_edges = set(true_graph.edges())
        inferred_edges = set(inferred_graph.edges())
        
        intersection = true_edges & inferred_edges
        union = true_edges | inferred_edges
        
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
        
        # Cosine similarity
        nodes = list(true_graph.nodes())
        n_nodes = len(nodes)
        
        true_vector = np.zeros(n_nodes * n_nodes)
        inferred_vector = np.zeros(n_nodes * n_nodes)
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                idx = i * n_nodes + j
                if true_graph.has_edge(node1, node2):
                    true_vector[idx] = 1
                if inferred_graph.has_edge(node1, node2):
                    inferred_vector[idx] = 1
        
        cosine_similarity = np.dot(true_vector, inferred_vector) / (np.linalg.norm(true_vector) * np.linalg.norm(inferred_vector)) if np.linalg.norm(true_vector) > 0 and np.linalg.norm(inferred_vector) > 0 else 0
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'cosine_similarity': cosine_similarity,
            'intersection_size': len(intersection),
            'union_size': len(union)
        }
    
    def _calculate_structural_complexity_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate structural complexity metrics."""
        # Calculate density
        n_nodes = true_graph.number_of_nodes()
        max_possible_edges = n_nodes * (n_nodes - 1)
        
        true_density = true_graph.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0
        inferred_density = inferred_graph.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate clustering coefficient
        true_clustering = nx.average_clustering(true_graph.to_undirected()) if true_graph.number_of_nodes() > 0 else 0
        inferred_clustering = nx.average_clustering(inferred_graph.to_undirected()) if inferred_graph.number_of_nodes() > 0 else 0
        
        # Calculate average degree
        true_avg_degree = np.mean([true_graph.degree(node) for node in true_graph.nodes()]) if true_graph.number_of_nodes() > 0 else 0
        inferred_avg_degree = np.mean([inferred_graph.degree(node) for node in inferred_graph.nodes()]) if inferred_graph.number_of_nodes() > 0 else 0
        
        return {
            'true_density': true_density,
            'inferred_density': inferred_density,
            'density_difference': abs(true_density - inferred_density),
            'true_clustering': true_clustering,
            'inferred_clustering': inferred_clustering,
            'clustering_difference': abs(true_clustering - inferred_clustering),
            'true_avg_degree': true_avg_degree,
            'inferred_avg_degree': inferred_avg_degree,
            'avg_degree_difference': abs(true_avg_degree - inferred_avg_degree)
        }
    
    def _calculate_topology_metrics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate graph topology metrics."""
        # Convert to undirected for topology analysis
        true_undirected = true_graph.to_undirected()
        inferred_undirected = inferred_graph.to_undirected()
        
        # Calculate connectivity metrics
        true_connected_components = nx.number_connected_components(true_undirected)
        inferred_connected_components = nx.number_connected_components(inferred_undirected)
        
        # Calculate path length metrics
        true_avg_path_length = self._calculate_average_path_length(true_undirected)
        inferred_avg_path_length = self._calculate_average_path_length(inferred_undirected)
        
        # Calculate diameter
        true_diameter = nx.diameter(true_undirected) if nx.is_connected(true_undirected) else np.inf
        inferred_diameter = nx.diameter(inferred_undirected) if nx.is_connected(inferred_undirected) else np.inf
        
        return {
            'true_connected_components': true_connected_components,
            'inferred_connected_components': inferred_connected_components,
            'true_avg_path_length': true_avg_path_length,
            'inferred_avg_path_length': inferred_avg_path_length,
            'path_length_difference': abs(true_avg_path_length - inferred_avg_path_length),
            'true_diameter': true_diameter,
            'inferred_diameter': inferred_diameter,
            'diameter_difference': abs(true_diameter - inferred_diameter) if true_diameter != np.inf and inferred_diameter != np.inf else np.inf
        }
    
    def _calculate_average_path_length(self, graph: nx.Graph) -> float:
        """Calculate average path length for a graph."""
        if not nx.is_connected(graph):
            return np.inf
        
        try:
            return nx.average_shortest_path_length(graph)
        except:
            return np.inf
    
    def _calculate_graph_statistics(self, true_graph: nx.DiGraph, inferred_graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        return {
            'true_graph': {
                'n_nodes': true_graph.number_of_nodes(),
                'n_edges': true_graph.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(true_graph),
                'is_connected': nx.is_weakly_connected(true_graph)
            },
            'inferred_graph': {
                'n_nodes': inferred_graph.number_of_nodes(),
                'n_edges': inferred_graph.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(inferred_graph),
                'is_connected': nx.is_weakly_connected(inferred_graph)
            }
        }
    
    def _calculate_confidence_intervals(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence intervals for metrics."""
        # This would implement bootstrap confidence intervals
        # For now, return placeholder
        return {
            'rel_shd_ci': (0.0, 0.0),
            'f1_score_ci': (0.0, 0.0)
        }
    
    def calculate_metrics_summary(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for a list of metrics."""
        if not metrics_list:
            return {}
        
        # Extract key metrics
        rel_shd_values = [m['basic_structural']['relative_structural_hamming_distance'] for m in metrics_list]
        f1_scores = [m['adjacency_metrics']['edge_f1_score'] for m in metrics_list]
        
        summary = {
            'n_experiments': len(metrics_list),
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
            }
        }
        
        return summary

# Convenience function for calculating metrics
def calculate_advanced_structural_metrics(true_graph: nx.DiGraph, 
                                        inferred_graph: nx.DiGraph,
                                        dataset_name: str = "unknown",
                                        config: StructuralMetricsConfig = None) -> Dict[str, Any]:
    """
    Convenience function for calculating advanced structural metrics.
    
    Args:
        true_graph: Ground truth causal graph
        inferred_graph: Inferred causal graph
        dataset_name: Name of the dataset
        config: Configuration object
        
    Returns:
        Dictionary containing comprehensive structural metrics
    """
    calculator = AdvancedStructuralMetrics(config)
    return calculator.calculate_comprehensive_metrics(true_graph, inferred_graph, dataset_name)

if __name__ == "__main__":
    # Example usage
    print("Advanced Structural Metrics Calculator - Example Usage")
    print("Ready for comprehensive structural evaluation!")
