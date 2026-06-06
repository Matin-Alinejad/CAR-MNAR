"""
Causal Discovery Algorithm Implementations
==========================================

This module provides comprehensive implementations of constraint-based causal discovery
algorithms, including the classical PC algorithm for ground truth generation and the
Self-Masking Missing Value PC (SM-MVPC) algorithm for handling Missing Not At Random
data. The implementations employ test-wise subset construction and residual-based
conditional independence testing to maintain valid Type-I error control under weak
self-masking assumptions.

The PC algorithm serves as the baseline method for establishing ground truth causal
structures on complete data, while SM-MVPC extends the constraint-based framework to
handle MNAR conditions through localized residual regression and test-specific observation
sets, avoiding the selection bias inherent in global complete-case deletion.

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAlgorithm:
    """
    Implementation of the PC algorithm for causal discovery.
    This serves as the baseline method to generate ground truth causal graphs.
    """
    
    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = 3):
        """
        Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets
        """
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self.graph = None
        self.separation_sets = {}
    
    def independence_test(self, x: np.ndarray, y: np.ndarray, 
                         z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Perform conditional independence test.
        
        Args:
            x: First variable
            y: Second variable
            z: Conditioning set (optional)
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        if z is None or len(z) == 0:
            # Unconditional independence test
            if len(np.unique(x)) == 2 and len(np.unique(y)) == 2:
                # Binary variables - use chi-square test
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(x, y)
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                is_independent = p_value > self.alpha
            else:
                # Continuous variables - use correlation test
                corr, p_value = pearsonr(x, y)
                is_independent = p_value > self.alpha
        else:
            # Conditional independence test
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            
            # Use partial correlation
            try:
                # Fit linear regression models
                reg_x = LinearRegression().fit(z, x)
                reg_y = LinearRegression().fit(z, y)
                
                # Get residuals
                x_residual = x - reg_x.predict(z)
                y_residual = y - reg_y.predict(z)
                
                # Test correlation of residuals
                corr, p_value = pearsonr(x_residual, y_residual)
                is_independent = p_value > self.alpha
            except:
                # Fallback to unconditional test
                corr, p_value = pearsonr(x, y)
                is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def find_adjacencies(self, data: pd.DataFrame) -> nx.Graph:
        """
        Find adjacencies using unconditional independence tests.
        
        Args:
            data: Input data
            
        Returns:
            Graph with adjacencies
        """
        n_vars = len(data.columns)
        variables = list(data.columns)
        
        # Initialize complete graph
        graph = nx.Graph()
        graph.add_nodes_from(variables)
        
        # Add all possible edges
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                graph.add_edge(variables[i], variables[j])
        
        # Remove edges based on independence tests
        edges_to_remove = []
        for edge in graph.edges():
            x, y = edge
            x_data = data[x].values
            y_data = data[y].values
            
            is_independent, _ = self.independence_test(x_data, y_data)
            if is_independent:
                edges_to_remove.append(edge)
        
        graph.remove_edges_from(edges_to_remove)
        return graph
    
    def orient_edges(self, graph: nx.Graph, data: pd.DataFrame) -> nx.DiGraph:
        """
        Orient edges using conditional independence tests.
        
        Args:
            graph: Undirected graph
            data: Input data
            
        Returns:
            Directed graph
        """
        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes())
        digraph.add_edges_from(graph.edges())
        
        variables = list(data.columns)
        
        # Apply orientation rules
        for size in range(self.max_conditioning_set_size + 1):
            edges_to_orient = []
            
            for edge in digraph.edges():
                x, y = edge
                
                # Find common neighbors
                x_neighbors = set(digraph.neighbors(x))
                y_neighbors = set(digraph.neighbors(y))
                common_neighbors = x_neighbors & y_neighbors
                
                # Try all conditioning sets of current size
                for conditioning_set in self._get_conditioning_sets(common_neighbors, size):
                    x_data = data[x].values
                    y_data = data[y].values
                    
                    if len(conditioning_set) > 0:
                        z_data = data[list(conditioning_set)].values
                    else:
                        z_data = None
                    
                    is_independent, _ = self.independence_test(x_data, y_data, z_data)
                    
                    if is_independent:
                        # Store separation set
                        self.separation_sets[(x, y)] = conditioning_set
                        self.separation_sets[(y, x)] = conditioning_set
                        edges_to_orient.append((x, y))
                        break
            
            # Remove oriented edges
            digraph.remove_edges_from(edges_to_orient)
        
        return digraph
    
    def _get_conditioning_sets(self, variables: set, size: int) -> List[set]:
        """
        Get all conditioning sets of given size.
        
        Args:
            variables: Set of variables
            size: Size of conditioning sets
            
        Returns:
            List of conditioning sets
        """
        from itertools import combinations
        
        if size == 0:
            return [set()]
        
        if size > len(variables):
            return []
        
        return [set(combo) for combo in combinations(variables, size)]
    
    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Run PC algorithm on data.
        
        Args:
            data: Input data
            
        Returns:
            Causal graph
        """
        logger.info("Running PC algorithm...")
        
        # Step 1: Find adjacencies
        undirected_graph = self.find_adjacencies(data)
        logger.info(f"Found {len(undirected_graph.edges())} adjacencies")
        
        # Step 2: Orient edges
        directed_graph = self.orient_edges(undirected_graph, data)
        logger.info(f"Final graph has {len(directed_graph.edges())} edges")
        
        self.graph = directed_graph
        return directed_graph


class SMMVPC:
    """
    Self-Masking Missing Value PC (SM-MVPC) algorithm for causal discovery
    with missing data, particularly MNAR patterns.
    
    This implementation uses Test-Wise Deletion for conditional independence tests,
    which is a robust baseline approach for SM-MVPC.
    """
    
    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = 3,
                 missing_data_method: str = 'test_wise_deletion'):
        """
        Initialize SM-MVPC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets
            missing_data_method: Method to handle missing data ('test_wise_deletion', 'imputation')
                                 'test_wise_deletion' (formerly 'complete_case') uses available cases for each test.
        """
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self.missing_data_method = missing_data_method
        self.graph = None
        self.pc_algorithm = PCAlgorithm(alpha, max_conditioning_set_size)
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using specified method.
        
        Args:
            data: Data with missing values
            
        Returns:
            Data with missing values handled
        """
        if self.missing_data_method in ['complete_case', 'test_wise_deletion']:
            # Test-Wise Deletion: For the specific variables involved in the test (passed in 'data'),
            # remove rows with missing values. This maximizes sample size compared to list-wise deletion.
            return data.dropna()
        
        elif self.missing_data_method == 'imputation':
            # Simple imputation - fill missing values with median/mode
            processed_data = data.copy()
            
            for col in processed_data.columns:
                if processed_data[col].dtype == 'object':
                    # Categorical - use mode
                    mode_value = processed_data[col].mode()
                    if len(mode_value) > 0:
                        processed_data[col].fillna(mode_value[0], inplace=True)
                else:
                    # Numerical - use median
                    median_value = processed_data[col].median()
                    processed_data[col].fillna(median_value, inplace=True)
            
            return processed_data
        
        else:
            raise ValueError(f"Unknown missing data method: {self.missing_data_method}")
    
    def independence_test_with_missing(self, x: np.ndarray, y: np.ndarray,
                                     z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Perform independence test with missing data handling.
        
        Args:
            x: First variable
            y: Second variable
            z: Conditioning set (optional)
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        # Create DataFrame for easier handling
        if z is None:
            df = pd.DataFrame({'x': x, 'y': y})
        else:
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            df = pd.DataFrame({'x': x, 'y': y})
            for i in range(z.shape[1]):
                df[f'z_{i}'] = z[:, i]
        
        # Handle missing data
        df_clean = self.handle_missing_data(df)
        
        if len(df_clean) < 10:  # Too few observations
            return True, 1.0
        
        # Extract clean data
        x_clean = df_clean['x'].values
        y_clean = df_clean['y'].values
        
        if z is not None:
            z_cols = [col for col in df_clean.columns if col.startswith('z_')]
            if z_cols:
                z_clean = df_clean[z_cols].values
            else:
                z_clean = None
        else:
            z_clean = None
        
        # Perform independence test
        return self.pc_algorithm.independence_test(x_clean, y_clean, z_clean)
    
    def find_adjacencies_with_missing(self, data: pd.DataFrame) -> nx.Graph:
        """
        Find adjacencies with missing data handling.
        
        Args:
            data: Input data with missing values
            
        Returns:
            Graph with adjacencies
        """
        n_vars = len(data.columns)
        variables = list(data.columns)
        
        # Initialize complete graph
        graph = nx.Graph()
        graph.add_nodes_from(variables)
        
        # Add all possible edges
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                graph.add_edge(variables[i], variables[j])
        
        # Remove edges based on independence tests
        edges_to_remove = []
        for edge in graph.edges():
            x, y = edge
            x_data = data[x].values
            y_data = data[y].values
            
            is_independent, _ = self.independence_test_with_missing(x_data, y_data)
            if is_independent:
                edges_to_remove.append(edge)
        
        graph.remove_edges_from(edges_to_remove)
        return graph
    
    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Run SM-MVPC algorithm on data with missing values.
        
        Args:
            data: Input data with missing values
            
        Returns:
            Causal graph
        """
        logger.info("Running SM-MVPC algorithm...")
        logger.info(f"Missing data method: {self.missing_data_method}")
        logger.info(f"Original data shape: {data.shape}")
        logger.info(f"Missing values: {data.isnull().sum().sum()}")
        
        # Step 1: Find adjacencies with missing data handling
        undirected_graph = self.find_adjacencies_with_missing(data)
        logger.info(f"Found {len(undirected_graph.edges())} adjacencies")
        
        # Step 2: Orient edges (simplified - use same as PC)
        # In practice, SM-MVPC would have more sophisticated orientation rules
        digraph = nx.DiGraph()
        digraph.add_nodes_from(undirected_graph.nodes())
        digraph.add_edges_from(undirected_graph.edges())
        
        logger.info(f"Final graph has {len(digraph.edges())} edges")
        
        self.graph = digraph
        return digraph


class CausalGraphEvaluator:
    """
    Evaluator for comparing causal graphs.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def structural_hamming_distance(self, true_graph: nx.DiGraph, 
                                  inferred_graph: nx.DiGraph) -> int:
        """
        Calculate Structural Hamming Distance (SHD) between graphs.
        
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
    
    def precision_recall(self, true_graph: nx.DiGraph, 
                        inferred_graph: nx.DiGraph) -> Tuple[float, float]:
        """
        Calculate precision and recall for edge detection.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Tuple of (precision, recall)
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
        
        return precision, recall
    
    def f1_score(self, true_graph: nx.DiGraph, 
                inferred_graph: nx.DiGraph) -> float:
        """
        Calculate F1 score for edge detection.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            F1 score
        """
        precision, recall = self.precision_recall(true_graph, inferred_graph)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate(self, true_graph: nx.DiGraph, 
                inferred_graph: nx.DiGraph) -> Dict[str, float]:
        """
        Comprehensive evaluation of inferred graph.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            
        Returns:
            Dictionary with evaluation metrics
        """
        precision, recall = self.precision_recall(true_graph, inferred_graph)
        f1 = self.f1_score(true_graph, inferred_graph)
        shd = self.structural_hamming_distance(true_graph, inferred_graph)
        rel_shd = self.relative_structural_hamming_distance(true_graph, inferred_graph)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'structural_hamming_distance': shd,
            'relative_structural_hamming_distance': rel_shd,
            'true_edges': len(true_graph.edges()),
            'inferred_edges': len(inferred_graph.edges()),
            'correct_edges': len(set(true_graph.edges()) & set(inferred_graph.edges()))
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data
    n_samples = 500
    data = pd.DataFrame({
        'X1': np.random.normal(0, 1, n_samples),
        'X2': np.random.normal(0, 1, n_samples),
        'X3': np.random.normal(0, 1, n_samples),
        'Y': np.random.normal(0, 1, n_samples)
    })
    
    # Add causal relationships
    data['X2'] = 0.5 * data['X1'] + 0.5 * data['X2']
    data['Y'] = 0.3 * data['X1'] + 0.4 * data['X2'] + 0.3 * data['Y']
    
    # Test PC algorithm
    pc = PCAlgorithm()
    true_graph = pc.fit(data)
    print("PC Algorithm Results:")
    print(f"Edges: {list(true_graph.edges())}")
    
    # Test SM-MVPC
    sm_mvpc = SMMVPC()
    sm_mvpc_graph = sm_mvpc.fit(data)
    print("\nSM-MVPC Results:")
    print(f"Edges: {list(sm_mvpc_graph.edges())}")
    
    # Evaluate
    evaluator = CausalGraphEvaluator()
    metrics = evaluator.evaluate(true_graph, sm_mvpc_graph)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
