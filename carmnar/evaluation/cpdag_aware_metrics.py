"""
CPDAG-Aware Evaluation Metrics for Causal Discovery
=====================================================

This module implements theoretically-grounded evaluation metrics that properly handle
Markov equivalence classes and Completed Partially Directed Acyclic Graph (CPDAG)
representations, as required for rigorous causal discovery evaluation. The framework
ensures fair and interpretable comparisons by respecting the fundamental limitation
that observational data can only identify causal structures up to Markov equivalence.

The implementation provides comprehensive evaluation capabilities including DAG-to-CPDAG
conversion via Meek rules, skeleton-based adjacency evaluation, orientation metrics
restricted to identifiable/compelled edges, v-structure (collider) detection, proper
Structural Hamming Distance on CPDAGs, and Structural Intervention Distance (SID) for
causal effect accuracy assessment.

Key Features:
1. DAG → CPDAG conversion via Meek rules
2. Skeleton-based metrics (adjacency evaluation)
3. Orientation metrics restricted to identifiable/compelled edges
4. V-structure (collider) detection and evaluation
5. Proper Structural Hamming Distance on CPDAGs
6. Structural Intervention Distance (SID) for causal effect accuracy

References:
- Chickering (2002): "Optimal Structure Identification With Greedy Search"
- Meek (1995): "Causal Inference and Causal Explanation with Background Knowledge"
- Peters & Bühlmann (2015): "Structural Intervention Distance for Evaluating Causal Graphs"
- Acid & de Campos (2003): "Searching for Bayesian Network Structures in the Space of RPDAGs"

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from itertools import combinations, permutations
import logging
from collections import defaultdict
from scipy import stats
from scipy.stats import bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CPDAGEvaluationResult:
    """Comprehensive evaluation results for CPDAG comparison."""
    # Skeleton metrics (adjacency)
    skeleton_precision: float
    skeleton_recall: float
    skeleton_f1: float
    skeleton_shd: int
    
    # Orientation metrics (on compelled edges only)
    orientation_precision: float
    orientation_recall: float
    orientation_f1: float
    n_compelled_true: int
    n_compelled_inferred: int
    
    # V-structure metrics
    vstructure_precision: float
    vstructure_recall: float
    vstructure_f1: float
    n_vstructures_true: int
    n_vstructures_inferred: int
    
    # Overall CPDAG metrics
    cpdag_shd: int
    cpdag_shd_normalized: float
    
    # Arrow metrics (directed edges)
    arrow_precision: float
    arrow_recall: float
    arrow_f1: float
    
    # Tail metrics (undirected in CPDAG)
    n_undirected_true: int
    n_undirected_inferred: int
    
    # Additional diagnostics
    n_nodes: int
    n_edges_true: int
    n_edges_inferred: int
    markov_equivalent: bool


class CPDAGConverter:
    """
    Convert a DAG to its CPDAG (Completed Partially Directed Acyclic Graph).
    
    A CPDAG represents the Markov equivalence class of a DAG. Two DAGs are
    Markov equivalent iff they have the same skeleton and v-structures.
    
    The CPDAG contains:
    - Directed edges (→) for compelled/identifiable orientations
    - Undirected edges (—) for reversible orientations
    """
    
    @staticmethod
    def dag_to_cpdag(dag: nx.DiGraph) -> nx.DiGraph:
        """
        Convert a DAG to its CPDAG using the standard algorithm.
        
        Algorithm:
        1. Start with fully undirected skeleton
        2. Orient v-structures (X → Z ← Y where X and Y not adjacent)
        3. Apply Meek rules until no more orientations possible
        
        Args:
            dag: Input DAG as NetworkX DiGraph
            
        Returns:
            CPDAG as NetworkX DiGraph where:
            - Directed edge (u,v) means u→v is compelled
            - Both (u,v) and (v,u) present means u—v is undirected
        """
        # Get skeleton (undirected)
        skeleton = dag.to_undirected()
        
        # Initialize CPDAG with bidirectional edges (all undirected)
        cpdag = nx.DiGraph()
        cpdag.add_nodes_from(dag.nodes())
        for u, v in skeleton.edges():
            cpdag.add_edge(u, v)
            cpdag.add_edge(v, u)
        
        # Find and orient v-structures
        vstructures = CPDAGConverter._find_vstructures(dag)
        for x, z, y in vstructures:
            # Orient as X → Z ← Y
            # Remove reverse edges if present
            if cpdag.has_edge(z, x):
                cpdag.remove_edge(z, x)
            if cpdag.has_edge(z, y):
                cpdag.remove_edge(z, y)
        
        # Apply Meek rules until convergence
        cpdag = CPDAGConverter._apply_meek_rules(cpdag)
        
        return cpdag
    
    @staticmethod
    def _find_vstructures(dag: nx.DiGraph) -> List[Tuple]:
        """
        Find all v-structures (colliders) in a DAG.
        
        A v-structure is X → Z ← Y where X and Y are not adjacent.
        
        Returns:
            List of tuples (X, Z, Y) representing v-structures
        """
        vstructures = []
        
        for z in dag.nodes():
            parents = list(dag.predecessors(z))
            if len(parents) < 2:
                continue
            
            # Check all pairs of parents
            for x, y in combinations(parents, 2):
                # V-structure exists if x and y are NOT adjacent
                if not dag.has_edge(x, y) and not dag.has_edge(y, x):
                    vstructures.append((x, z, y))
        
        return vstructures
    
    @staticmethod
    def _apply_meek_rules(cpdag: nx.DiGraph) -> nx.DiGraph:
        """
        Apply Meek's orientation rules until no more edges can be oriented.
        
        Meek Rules:
        R1: If A → B — C and A not adjacent to C, orient B → C
        R2: If A → B → C and A — C, orient A → C
        R3: If A — B, A — C, A — D, B → D, C → D, and B not adj C, orient A → D
        R4: If A — B, B → C, A — C, A → D, D — C, orient D → C
        """
        changed = True
        while changed:
            changed = False
            
            for edge in list(cpdag.edges()):
                u, v = edge
                
                # Check if edge is undirected (both directions present)
                if not cpdag.has_edge(v, u):
                    continue  # Already directed
                
                # Apply each Meek rule
                if CPDAGConverter._meek_r1(cpdag, u, v):
                    changed = True
                elif CPDAGConverter._meek_r2(cpdag, u, v):
                    changed = True
                elif CPDAGConverter._meek_r3(cpdag, u, v):
                    changed = True
                elif CPDAGConverter._meek_r4(cpdag, u, v):
                    changed = True
        
        return cpdag
    
    @staticmethod
    def _is_undirected(cpdag: nx.DiGraph, u, v) -> bool:
        """Check if edge u—v is undirected in CPDAG."""
        return cpdag.has_edge(u, v) and cpdag.has_edge(v, u)
    
    @staticmethod
    def _is_directed(cpdag: nx.DiGraph, u, v) -> bool:
        """Check if edge u→v is directed in CPDAG."""
        return cpdag.has_edge(u, v) and not cpdag.has_edge(v, u)
    
    @staticmethod
    def _meek_r1(cpdag: nx.DiGraph, b, c) -> bool:
        """
        R1: If A → B — C and A not adjacent to C, orient B → C
        """
        if not CPDAGConverter._is_undirected(cpdag, b, c):
            return False
        
        for a in cpdag.predecessors(b):
            if a == c:
                continue
            # Check A → B (directed)
            if CPDAGConverter._is_directed(cpdag, a, b):
                # Check A not adjacent to C
                if not cpdag.has_edge(a, c) and not cpdag.has_edge(c, a):
                    # Orient B → C
                    cpdag.remove_edge(c, b)
                    return True
        return False
    
    @staticmethod
    def _meek_r2(cpdag: nx.DiGraph, a, c) -> bool:
        """
        R2: If A → B → C and A — C, orient A → C
        """
        if not CPDAGConverter._is_undirected(cpdag, a, c):
            return False
        
        # Find B such that A → B → C
        for b in cpdag.successors(a):
            if b == c:
                continue
            if CPDAGConverter._is_directed(cpdag, a, b):
                if CPDAGConverter._is_directed(cpdag, b, c):
                    # Orient A → C
                    cpdag.remove_edge(c, a)
                    return True
        return False
    
    @staticmethod
    def _meek_r3(cpdag: nx.DiGraph, a, d) -> bool:
        """
        R3: If A — B, A — C, B → D, C → D, B not adj C, and A — D, orient A → D
        """
        if not CPDAGConverter._is_undirected(cpdag, a, d):
            return False
        
        # Find B, C such that conditions hold
        neighbors_a = [n for n in cpdag.successors(a) 
                      if CPDAGConverter._is_undirected(cpdag, a, n) and n != d]
        
        for b, c in combinations(neighbors_a, 2):
            # Check B not adjacent to C
            if cpdag.has_edge(b, c) or cpdag.has_edge(c, b):
                continue
            # Check B → D and C → D
            if (CPDAGConverter._is_directed(cpdag, b, d) and 
                CPDAGConverter._is_directed(cpdag, c, d)):
                # Orient A → D
                cpdag.remove_edge(d, a)
                return True
        return False
    
    @staticmethod
    def _meek_r4(cpdag: nx.DiGraph, d, c) -> bool:
        """
        R4: If A — B, B → C, A — C, A → D, D — C, orient D → C
        """
        if not CPDAGConverter._is_undirected(cpdag, d, c):
            return False
        
        for a in cpdag.predecessors(d):
            if a == c:
                continue
            if not CPDAGConverter._is_directed(cpdag, a, d):
                continue
            # Check A — C
            if not CPDAGConverter._is_undirected(cpdag, a, c):
                continue
            # Find B such that A — B and B → C
            for b in cpdag.successors(a):
                if b == c or b == d:
                    continue
                if CPDAGConverter._is_undirected(cpdag, a, b):
                    if CPDAGConverter._is_directed(cpdag, b, c):
                        # Orient D → C
                        cpdag.remove_edge(c, d)
                        return True
        return False


class CPDAGAwareMetrics:
    """
    Comprehensive CPDAG-aware evaluation metrics for causal discovery.
    
    This class implements the gold-standard evaluation methodology used in
    top-tier causal discovery papers, properly handling:
    1. Markov equivalence classes
    2. Skeleton vs orientation evaluation
    3. V-structure identification
    4. Compelled vs reversible edges
    """
    
    def __init__(self):
        self.converter = CPDAGConverter()
    
    def evaluate(self, 
                 true_graph: nx.DiGraph, 
                 inferred_graph: nx.DiGraph,
                 convert_to_cpdag: bool = True) -> CPDAGEvaluationResult:
        """
        Comprehensive evaluation of inferred graph against ground truth.
        
        Args:
            true_graph: Ground truth DAG or CPDAG
            inferred_graph: Inferred DAG or CPDAG from algorithm
            convert_to_cpdag: If True, convert DAGs to CPDAGs before comparison
            
        Returns:
            CPDAGEvaluationResult with all metrics
        """
        # Ensure graphs have same node set
        all_nodes = set(true_graph.nodes()) | set(inferred_graph.nodes())
        
        # Add missing nodes to each graph
        true_graph_full = true_graph.copy()
        inferred_graph_full = inferred_graph.copy()
        for node in all_nodes:
            if node not in true_graph_full:
                true_graph_full.add_node(node)
            if node not in inferred_graph_full:
                inferred_graph_full.add_node(node)
        
        true_graph = true_graph_full
        inferred_graph = inferred_graph_full
        
        # Convert to CPDAGs if requested
        if convert_to_cpdag:
            true_cpdag = self.converter.dag_to_cpdag(true_graph)
            inferred_cpdag = self.converter.dag_to_cpdag(inferred_graph)
        else:
            true_cpdag = true_graph
            inferred_cpdag = inferred_graph
        
        # Compute all metrics
        skeleton_metrics = self._skeleton_metrics(true_cpdag, inferred_cpdag)
        orientation_metrics = self._orientation_metrics(true_cpdag, inferred_cpdag)
        vstructure_metrics = self._vstructure_metrics(true_graph, inferred_graph)
        arrow_metrics = self._arrow_metrics(true_cpdag, inferred_cpdag)
        cpdag_shd = self._cpdag_shd(true_cpdag, inferred_cpdag)
        
        # Check Markov equivalence
        markov_eq = self._check_markov_equivalence(true_graph, inferred_graph)
        
        # Count undirected edges
        n_undirected_true = sum(1 for u, v in true_cpdag.edges() 
                                if true_cpdag.has_edge(v, u)) // 2
        n_undirected_inferred = sum(1 for u, v in inferred_cpdag.edges() 
                                    if inferred_cpdag.has_edge(v, u)) // 2
        
        return CPDAGEvaluationResult(
            # Skeleton
            skeleton_precision=skeleton_metrics['precision'],
            skeleton_recall=skeleton_metrics['recall'],
            skeleton_f1=skeleton_metrics['f1'],
            skeleton_shd=skeleton_metrics['shd'],
            
            # Orientation
            orientation_precision=orientation_metrics['precision'],
            orientation_recall=orientation_metrics['recall'],
            orientation_f1=orientation_metrics['f1'],
            n_compelled_true=orientation_metrics['n_compelled_true'],
            n_compelled_inferred=orientation_metrics['n_compelled_inferred'],
            
            # V-structures
            vstructure_precision=vstructure_metrics['precision'],
            vstructure_recall=vstructure_metrics['recall'],
            vstructure_f1=vstructure_metrics['f1'],
            n_vstructures_true=vstructure_metrics['n_true'],
            n_vstructures_inferred=vstructure_metrics['n_inferred'],
            
            # CPDAG SHD
            cpdag_shd=cpdag_shd['shd'],
            cpdag_shd_normalized=cpdag_shd['normalized'],
            
            # Arrows
            arrow_precision=arrow_metrics['precision'],
            arrow_recall=arrow_metrics['recall'],
            arrow_f1=arrow_metrics['f1'],
            
            # Undirected
            n_undirected_true=n_undirected_true,
            n_undirected_inferred=n_undirected_inferred,
            
            # Diagnostics
            n_nodes=len(all_nodes),
            n_edges_true=true_graph.number_of_edges(),
            n_edges_inferred=inferred_graph.number_of_edges(),
            markov_equivalent=markov_eq
        )
    
    def _skeleton_metrics(self, true_cpdag: nx.DiGraph, 
                         inferred_cpdag: nx.DiGraph) -> Dict:
        """
        Compute skeleton (adjacency) metrics.
        
        Skeleton ignores edge directions—only evaluates whether pairs
        of variables are adjacent or not.
        """
        # Get undirected skeletons
        true_skeleton = set()
        for u, v in true_cpdag.edges():
            true_skeleton.add(frozenset([u, v]))
        
        inferred_skeleton = set()
        for u, v in inferred_cpdag.edges():
            inferred_skeleton.add(frozenset([u, v]))
        
        # Compute metrics
        tp = len(true_skeleton & inferred_skeleton)
        fp = len(inferred_skeleton - true_skeleton)
        fn = len(true_skeleton - inferred_skeleton)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = fp + fn  # Skeleton SHD
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'shd': shd,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def _orientation_metrics(self, true_cpdag: nx.DiGraph,
                            inferred_cpdag: nx.DiGraph) -> Dict:
        """
        Compute orientation metrics on COMPELLED edges only.
        
        Only evaluates orientations that are identifiable from observational
        data (i.e., edges that are directed in the CPDAG).
        """
        # Get compelled (directed) edges from each CPDAG
        true_compelled = set()
        for u, v in true_cpdag.edges():
            if not true_cpdag.has_edge(v, u):  # Directed
                true_compelled.add((u, v))
        
        inferred_compelled = set()
        for u, v in inferred_cpdag.edges():
            if not inferred_cpdag.has_edge(v, u):  # Directed
                inferred_compelled.add((u, v))
        
        # Only evaluate on edges that are compelled in BOTH
        # (i.e., where orientation is identifiable)
        true_skeleton = {frozenset([u, v]) for u, v in true_cpdag.edges()}
        inferred_skeleton = {frozenset([u, v]) for u, v in inferred_cpdag.edges()}
        common_skeleton = true_skeleton & inferred_skeleton
        
        # Filter to common adjacencies
        true_compelled_common = {(u, v) for u, v in true_compelled 
                                 if frozenset([u, v]) in common_skeleton}
        inferred_compelled_common = {(u, v) for u, v in inferred_compelled 
                                     if frozenset([u, v]) in common_skeleton}
        
        # Compute metrics
        tp = len(true_compelled_common & inferred_compelled_common)
        fp = len(inferred_compelled_common - true_compelled_common)
        fn = len(true_compelled_common - inferred_compelled_common)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # Default to 1.0 if no predictions
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # Default to 1.0 if no ground truth
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_compelled_true': len(true_compelled),
            'n_compelled_inferred': len(inferred_compelled)
        }
    
    def _vstructure_metrics(self, true_dag: nx.DiGraph,
                           inferred_dag: nx.DiGraph) -> Dict:
        """
        Compute v-structure (collider) detection metrics.
        
        V-structures are the ONLY source of orientation information
        from observational data alone.
        """
        true_vstruct = set(self.converter._find_vstructures(true_dag))
        inferred_vstruct = set(self.converter._find_vstructures(inferred_dag))
        
        # Normalize v-structure representation (order of parents doesn't matter)
        true_vstruct_norm = {(frozenset([x, y]), z) for x, z, y in true_vstruct}
        inferred_vstruct_norm = {(frozenset([x, y]), z) for x, z, y in inferred_vstruct}
        
        tp = len(true_vstruct_norm & inferred_vstruct_norm)
        fp = len(inferred_vstruct_norm - true_vstruct_norm)
        fn = len(true_vstruct_norm - inferred_vstruct_norm)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_true': len(true_vstruct),
            'n_inferred': len(inferred_vstruct)
        }
    
    def _arrow_metrics(self, true_cpdag: nx.DiGraph,
                      inferred_cpdag: nx.DiGraph) -> Dict:
        """
        Compute directed edge (arrow) metrics in CPDAG.
        """
        # Get all directed edges
        true_arrows = {(u, v) for u, v in true_cpdag.edges() 
                      if not true_cpdag.has_edge(v, u)}
        inferred_arrows = {(u, v) for u, v in inferred_cpdag.edges() 
                          if not inferred_cpdag.has_edge(v, u)}
        
        tp = len(true_arrows & inferred_arrows)
        fp = len(inferred_arrows - true_arrows)
        fn = len(true_arrows - inferred_arrows)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _cpdag_shd(self, true_cpdag: nx.DiGraph,
                   inferred_cpdag: nx.DiGraph) -> Dict:
        """
        Compute Structural Hamming Distance on CPDAGs.
        
        CPDAG-SHD counts:
        1. Missing edges (adjacency errors)
        2. Extra edges (adjacency errors)
        3. Orientation errors on compelled edges
        """
        shd = 0
        all_nodes = set(true_cpdag.nodes()) | set(inferred_cpdag.nodes())
        
        for u, v in combinations(all_nodes, 2):
            true_has_uv = true_cpdag.has_edge(u, v)
            true_has_vu = true_cpdag.has_edge(v, u)
            inferred_has_uv = inferred_cpdag.has_edge(u, v)
            inferred_has_vu = inferred_cpdag.has_edge(v, u)
            
            true_adjacent = true_has_uv or true_has_vu
            inferred_adjacent = inferred_has_uv or inferred_has_vu
            
            if true_adjacent != inferred_adjacent:
                # Adjacency error
                shd += 1
            elif true_adjacent and inferred_adjacent:
                # Both adjacent - check orientation
                true_undirected = true_has_uv and true_has_vu
                inferred_undirected = inferred_has_uv and inferred_has_vu
                
                if true_undirected != inferred_undirected:
                    # One directed, one undirected
                    shd += 1
                elif not true_undirected and not inferred_undirected:
                    # Both directed - check if same direction
                    if true_has_uv != inferred_has_uv:
                        shd += 1
        
        # Normalize
        max_edges = len(all_nodes) * (len(all_nodes) - 1) // 2
        normalized = shd / max_edges if max_edges > 0 else 0.0
        
        return {
            'shd': shd,
            'normalized': normalized
        }
    
    def _check_markov_equivalence(self, dag1: nx.DiGraph, 
                                  dag2: nx.DiGraph) -> bool:
        """
        Check if two DAGs are Markov equivalent.
        
        Two DAGs are Markov equivalent iff they have:
        1. Same skeleton
        2. Same v-structures
        """
        # Check skeleton
        skeleton1 = {frozenset([u, v]) for u, v in dag1.edges()}
        skeleton2 = {frozenset([u, v]) for u, v in dag2.edges()}
        
        if skeleton1 != skeleton2:
            return False
        
        # Check v-structures
        vstruct1 = {(frozenset([x, y]), z) 
                    for x, z, y in self.converter._find_vstructures(dag1)}
        vstruct2 = {(frozenset([x, y]), z) 
                    for x, z, y in self.converter._find_vstructures(dag2)}
        
        return vstruct1 == vstruct2


def compute_sid(true_dag: nx.DiGraph, inferred_dag: nx.DiGraph) -> int:
    """
    Compute Structural Intervention Distance (SID).
    
    SID measures the number of interventional distributions that differ
    between two DAGs, providing a causal (not just statistical) evaluation.
    
    Reference: Peters & Bühlmann (2015)
    
    Args:
        true_dag: Ground truth DAG
        inferred_dag: Inferred DAG
        
    Returns:
        SID value (lower is better, 0 means identical causal effects)
    """
    # Ensure both graphs have all nodes
    all_nodes = set(true_dag.nodes()) | set(inferred_dag.nodes())
    
    # Create copies with all nodes
    true_dag_full = true_dag.copy()
    inferred_dag_full = inferred_dag.copy()
    
    for node in all_nodes:
        if node not in true_dag_full:
            true_dag_full.add_node(node)
        if node not in inferred_dag_full:
            inferred_dag_full.add_node(node)
    
    nodes = list(all_nodes)
    sid = 0
    
    for i, target in enumerate(nodes):
        for j, source in enumerate(nodes):
            if i == j:
                continue
            
            # Check if intervention on source affects target differently
            # in the two DAGs
            true_ancestors = nx.ancestors(true_dag_full, target)
            inferred_ancestors = nx.ancestors(inferred_dag_full, target)
            
            true_affected = source in true_ancestors or source == target
            inferred_affected = source in inferred_ancestors or source == target
            
            if true_affected != inferred_affected:
                sid += 1
    
    return sid


@dataclass
class StatisticalTestResult:
    """Result of statistical significance testing."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    

def compute_confidence_interval(values: np.ndarray, 
                                confidence: float = 0.95,
                                method: str = 'bootstrap') -> Tuple[float, float, float, float]:
    """
    Compute confidence interval for a set of values.
    
    Args:
        values: Array of metric values across replicates
        confidence: Confidence level (default 0.95 for 95% CI)
        method: 'bootstrap' for non-parametric or 't' for t-distribution
        
    Returns:
        Tuple of (mean, std, ci_lower, ci_upper)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    
    if n < 2:
        return mean, std, mean, mean
    
    if method == 'bootstrap' and n >= 5:
        try:
            # Bootstrap confidence interval
            rng = np.random.default_rng(42)
            res = bootstrap((values,), np.mean, confidence_level=confidence,
                           random_state=rng, method='percentile')
            ci_lower, ci_upper = res.confidence_interval.low, res.confidence_interval.high
        except:
            # Fallback to t-distribution
            method = 't'
    
    if method == 't' or n < 5:
        # t-distribution based CI
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha/2, df=n-1)
        margin = t_val * std / np.sqrt(n)
        ci_lower = mean - margin
        ci_upper = mean + margin
    
    return mean, std, ci_lower, ci_upper


def paired_permutation_test(values1: np.ndarray, 
                           values2: np.ndarray,
                           n_permutations: int = 10000) -> float:
    """
    Non-parametric paired permutation test for comparing two conditions.
    
    Args:
        values1: Metric values from condition 1
        values2: Metric values from condition 2
        n_permutations: Number of permutations
        
    Returns:
        Two-sided p-value
    """
    values1, values2 = np.array(values1), np.array(values2)
    n = len(values1)
    
    if n != len(values2):
        raise ValueError("Arrays must have same length for paired test")
    
    # Observed difference
    observed_diff = np.mean(values1 - values2)
    
    # Permutation distribution
    rng = np.random.RandomState(42)
    count_extreme = 0
    
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(signs * (values1 - values2))
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return p_value


def compute_effect_size(values1: np.ndarray, values2: np.ndarray) -> Dict[str, float]:
    """
    Compute effect size measures (Cohen's d and Cliff's delta).
    
    Args:
        values1: Baseline condition values
        values2: Treatment condition values
        
    Returns:
        Dictionary with effect size measures
    """
    values1, values2 = np.array(values1), np.array(values2)
    
    # Cohen's d (standardized mean difference)
    pooled_std = np.sqrt(((len(values1)-1)*np.var(values1, ddof=1) + 
                          (len(values2)-1)*np.var(values2, ddof=1)) / 
                         (len(values1) + len(values2) - 2))
    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0.0
    
    # Cliff's delta (non-parametric effect size)
    n1, n2 = len(values1), len(values2)
    dominance = 0
    for v1 in values1:
        for v2 in values2:
            if v1 > v2:
                dominance += 1
            elif v1 < v2:
                dominance -= 1
    cliffs_delta = dominance / (n1 * n2)
    
    # Interpret effect size
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interpretation = "negligible"
    elif d_abs < 0.5:
        interpretation = "small"
    elif d_abs < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'cohens_d': cohens_d,
        'cliffs_delta': cliffs_delta,
        'interpretation': interpretation
    }


class StatisticalAnalyzer:
    """
    Statistical analysis utilities for CPDAG evaluation results.
    
    Provides:
    - Confidence intervals (bootstrap and t-distribution)
    - Significance testing (permutation test, Wilcoxon)
    - Effect size computation (Cohen's d, Cliff's delta)
    - Multiple comparison correction (Bonferroni, Holm)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def analyze_condition(self, results: List[Dict], 
                         metrics: List[str] = None) -> Dict[str, StatisticalTestResult]:
        """
        Compute summary statistics with CIs for a single experimental condition.
        """
        if metrics is None:
            metrics = ['skeleton_f1', 'vstructure_f1', 'orientation_f1', 
                      'cpdag_shd', 'sid']
        
        analysis = {}
        for metric in metrics:
            values = np.array([r[metric] for r in results if metric in r])
            if len(values) > 0:
                mean, std, ci_lo, ci_hi = compute_confidence_interval(
                    values, self.confidence_level
                )
                analysis[metric] = StatisticalTestResult(
                    mean=mean, std=std, 
                    ci_lower=ci_lo, ci_upper=ci_hi,
                    n_samples=len(values)
                )
        
        return analysis
    
    def compare_conditions(self, 
                          results1: List[Dict], 
                          results2: List[Dict],
                          metric: str) -> Dict:
        """
        Statistical comparison between two conditions.
        """
        values1 = np.array([r[metric] for r in results1 if metric in r])
        values2 = np.array([r[metric] for r in results2 if metric in r])
        
        # Compute statistics
        stats1 = compute_confidence_interval(values1, self.confidence_level)
        stats2 = compute_confidence_interval(values2, self.confidence_level)
        
        # Significance test
        if len(values1) == len(values2):
            p_value = paired_permutation_test(values1, values2)
        else:
            # Use Mann-Whitney U for unpaired
            _, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Effect size
        effect = compute_effect_size(values1, values2)
        
        return {
            'condition1': {'mean': stats1[0], 'std': stats1[1], 
                          'ci': (stats1[2], stats1[3])},
            'condition2': {'mean': stats2[0], 'std': stats2[1], 
                          'ci': (stats2[2], stats2[3])},
            'p_value': p_value,
            'significant': p_value < (1 - self.confidence_level),
            'effect_size': effect
        }
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons."""
        n = len(p_values)
        return [min(p * n, 1.0) for p in p_values]
    
    def holm_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni step-down correction."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        adjusted = np.zeros(n)
        
        for rank, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_values[idx] * (n - rank), 1.0)
        
        # Enforce monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i-1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
        
        return adjusted.tolist()


# Convenience function
def evaluate_causal_discovery(true_graph: nx.DiGraph,
                              inferred_graph: nx.DiGraph,
                              convert_to_cpdag: bool = True) -> Dict:
    """
    Comprehensive evaluation of causal discovery result.
    
    Args:
        true_graph: Ground truth DAG
        inferred_graph: Inferred graph from algorithm
        convert_to_cpdag: Whether to convert to CPDAGs before comparison
        
    Returns:
        Dictionary with all evaluation metrics
    """
    evaluator = CPDAGAwareMetrics()
    result = evaluator.evaluate(true_graph, inferred_graph, convert_to_cpdag)
    
    # Also compute SID
    sid = compute_sid(true_graph, inferred_graph)
    
    return {
        # Skeleton (adjacency)
        'skeleton_precision': result.skeleton_precision,
        'skeleton_recall': result.skeleton_recall,
        'skeleton_f1': result.skeleton_f1,
        'skeleton_shd': result.skeleton_shd,
        
        # V-structures (colliders)
        'vstructure_precision': result.vstructure_precision,
        'vstructure_recall': result.vstructure_recall,
        'vstructure_f1': result.vstructure_f1,
        'n_vstructures_true': result.n_vstructures_true,
        'n_vstructures_inferred': result.n_vstructures_inferred,
        
        # Orientation (compelled edges only)
        'orientation_precision': result.orientation_precision,
        'orientation_recall': result.orientation_recall,
        'orientation_f1': result.orientation_f1,
        
        # Arrow metrics
        'arrow_precision': result.arrow_precision,
        'arrow_recall': result.arrow_recall,
        'arrow_f1': result.arrow_f1,
        
        # CPDAG SHD
        'cpdag_shd': result.cpdag_shd,
        'cpdag_shd_normalized': result.cpdag_shd_normalized,
        
        # SID
        'sid': sid,
        
        # Diagnostics
        'n_nodes': result.n_nodes,
        'n_edges_true': result.n_edges_true,
        'n_edges_inferred': result.n_edges_inferred,
        'markov_equivalent': result.markov_equivalent,
        'n_compelled_true': result.n_compelled_true,
        'n_compelled_inferred': result.n_compelled_inferred,
        'n_undirected_true': result.n_undirected_true,
        'n_undirected_inferred': result.n_undirected_inferred
    }


if __name__ == '__main__':
    # Test the implementation
    print("="*70)
    print("CPDAG-AWARE METRICS - VALIDATION")
    print("="*70)
    
    # Create test DAGs
    # DAG 1: A → B → C (chain)
    dag1 = nx.DiGraph()
    dag1.add_edges_from([('A', 'B'), ('B', 'C')])
    
    # DAG 2: A ← B → C (fork) - Markov equivalent to chain!
    dag2 = nx.DiGraph()
    dag2.add_edges_from([('B', 'A'), ('B', 'C')])
    
    # DAG 3: A → B ← C (collider) - NOT equivalent
    dag3 = nx.DiGraph()
    dag3.add_edges_from([('A', 'B'), ('C', 'B')])
    
    print("\nTest 1: Chain vs Fork (should be Markov equivalent)")
    print("-"*50)
    result1 = evaluate_causal_discovery(dag1, dag2)
    print(f"  Markov equivalent: {result1['markov_equivalent']}")
    print(f"  Skeleton F1: {result1['skeleton_f1']:.3f}")
    print(f"  V-structure F1: {result1['vstructure_f1']:.3f}")
    print(f"  CPDAG SHD: {result1['cpdag_shd']}")
    
    print("\nTest 2: Chain vs Collider (should NOT be equivalent)")
    print("-"*50)
    result2 = evaluate_causal_discovery(dag1, dag3)
    print(f"  Markov equivalent: {result2['markov_equivalent']}")
    print(f"  Skeleton F1: {result2['skeleton_f1']:.3f}")
    print(f"  V-structure precision: {result2['vstructure_precision']:.3f}")
    print(f"  V-structure recall: {result2['vstructure_recall']:.3f}")
    print(f"  CPDAG SHD: {result2['cpdag_shd']}")
    
    print("\nTest 3: Complex DAG evaluation")
    print("-"*50)
    # More complex example
    true_dag = nx.DiGraph()
    true_dag.add_edges_from([
        ('X1', 'X3'), ('X2', 'X3'),  # V-structure at X3
        ('X3', 'X4'), ('X4', 'X5')
    ])
    
    # Inferred with missing edge and wrong orientation
    inferred_dag = nx.DiGraph()
    inferred_dag.add_edges_from([
        ('X1', 'X3'), ('X2', 'X3'),  # Correct v-structure
        ('X3', 'X4')  # Missing X4 → X5
    ])
    
    result3 = evaluate_causal_discovery(true_dag, inferred_dag)
    print(f"  Skeleton precision: {result3['skeleton_precision']:.3f}")
    print(f"  Skeleton recall: {result3['skeleton_recall']:.3f}")
    print(f"  V-structure precision: {result3['vstructure_precision']:.3f}")
    print(f"  V-structure recall: {result3['vstructure_recall']:.3f}")
    print(f"  CPDAG SHD: {result3['cpdag_shd']}")
    print(f"  SID: {result3['sid']}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

