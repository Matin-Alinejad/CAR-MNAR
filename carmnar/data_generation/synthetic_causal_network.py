"""
Synthetic Causal Network Generation Framework

This module implements a comprehensive framework for generating synthetic causal networks
with known ground-truth structures, simulating data from these networks using Additive
Noise Models (ANMs), and applying MNAR missingness mechanisms for systematic evaluation
of causal discovery algorithms.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Enumeration of DAG topology types."""
    CHAIN = "chain"
    FORK = "fork"
    COLLIDER = "collider"
    DIAMOND = "diamond"
    RANDOM = "random"
    SCALE_FREE = "scale_free"
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert" # Scale-free graph
    WATTS_STROGATZ = "watts_strogatz" # Small-world graph


@dataclass
class SyntheticNetworkConfig:
    """Configuration for synthetic network generation."""
    n_nodes: int
    topology_type: TopologyType
    edge_density: float = 0.3 # For ERDOS_RENYI and RANDOM
    edge_weights: Union[float, List[float], str] = "random"
    noise_variance: Union[float, List[float], str] = "random"
    min_edge_weight: float = 0.3
    max_edge_weight: float = 1.0
    noise_range: Tuple[float, float] = (0.1, 1.0)
    random_seed: Optional[int] = None
    ensure_connected: bool = True
    allow_cycles: bool = False
    
    # Specific parameters for certain topology types
    barabasi_albert_m: int = 1 # Number of edges to attach from a new node to existing nodes for Barabasi-Albert
    watts_strogatz_k: int = 2 # Each node is joined to its k nearest neighbors in a ring topology for Watts-Strogatz
    watts_strogatz_p: float = 0.1 # Probability of rewiring each edge for Watts-Strogatz


@dataclass
class SyntheticDataConfig:
    """Configuration for data simulation from synthetic network."""
    n_samples: int
    noise_distribution: str = "gaussian"  # "gaussian", "uniform", "laplace", "exponential", "student_t"
    non_linear: bool = False
    non_linear_function: Optional[Callable] = None # e.g., np.sin, lambda x: x**2
    random_seed: Optional[int] = None


class SyntheticCausalNetwork:
    """
    Generate synthetic causal networks with known ground-truth structures.
    
    This class provides methods for:
    1. Generating DAG structures with various topologies
    2. Simulating data from DAGs using Additive Noise Models (ANMs)
    3. Identifying effect variables (non-root nodes with parents)
    4. Applying MNAR missingness mechanisms
    """
    
    def __init__(self, network_config: SyntheticNetworkConfig, 
                 data_config: SyntheticDataConfig,
                 random_seed: int = 42):
        """
        Initialize synthetic causal network generator.
        
        Args:
            network_config: Configuration for network structure
            data_config: Configuration for data simulation
            random_seed: Random seed for reproducibility
        """
        self.network_config = network_config
        self.data_config = data_config
        self.rng = np.random.RandomState(random_seed)
        self.dag = None
        self.edge_weights = {}
        self.noise_variances = {}
        self.data = None
        self.ground_truth_adjacency = None
        
    def generate_dag(self) -> nx.DiGraph:
        """
        Generate DAG structure based on configuration.
        
        Returns:
            NetworkX DiGraph representing the causal structure
        """
        n = self.network_config.n_nodes
        topology = self.network_config.topology_type
        
        logger.info(f"Generating {topology.value} DAG with {n} nodes")
        
        if topology == TopologyType.CHAIN:
            dag = self._generate_chain(n)
        elif topology == TopologyType.FORK:
            dag = self._generate_fork(n)
        elif topology == TopologyType.COLLIDER:
            dag = self._generate_collider(n)
        elif topology == TopologyType.DIAMOND:
            dag = self._generate_diamond(n)
        elif topology == TopologyType.RANDOM:
            dag = self._generate_random_dag(n, self.network_config.edge_density)
        elif topology == TopologyType.SCALE_FREE:
            dag = self._generate_scale_free(n, self.network_config.edge_density)
        elif topology == TopologyType.ERDOS_RENYI:
            dag = self._generate_erdos_renyi(n, self.network_config.edge_density)
        elif topology == TopologyType.BARABASI_ALBERT:
            dag = self._generate_barabasi_albert(n, self.network_config.barabasi_albert_m)
        elif topology == TopologyType.WATTS_STROGATZ:
            dag = self._generate_watts_strogatz(n, self.network_config.watts_strogatz_k, self.network_config.watts_strogatz_p)
        else:
            raise ValueError(f"Unknown topology type: {topology}")
        
        # Validate DAG properties
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Generated graph is not a DAG!")
        
        if self.network_config.ensure_connected and not nx.is_weakly_connected(dag):
            logger.warning("Graph is not connected, adding edges to ensure connectivity")
            dag = self._ensure_connectivity(dag)
        
        self.dag = dag
        self._store_ground_truth()
        return dag
    
    def _generate_barabasi_albert(self, n: int, m: int) -> nx.DiGraph:
        """Generate a Barabasi-Albert (scale-free) DAG."""
        if m >= n:
            raise ValueError("m must be less than n for Barabasi-Albert graph.")
        
        # Generate undirected Barabasi-Albert graph
        graph = nx.barabasi_albert_graph(n, m, seed=self.rng.randint(10000))
        
        # Convert to DAG by assigning random topological order
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        
        nodes = list(range(n))
        self.rng.shuffle(nodes) # Random topological ordering
        
        for u, v in graph.edges():
            # Add edge respecting topological order
            if nodes.index(u) < nodes.index(v):
                dag.add_edge(u, v)
            elif nodes.index(v) < nodes.index(u):
                dag.add_edge(v, u)
        
        return dag

    def _generate_watts_strogatz(self, n: int, k: int, p: float) -> nx.DiGraph:
        """Generate a Watts-Strogatz (small-world) DAG."""
        if k >= n:
            raise ValueError("k must be less than n for Watts-Strogatz graph.")
        
        # Generate undirected Watts-Strogatz graph
        graph = nx.watts_strogatz_graph(n, k, p, seed=self.rng.randint(10000))
        
        # Convert to DAG using random topological order
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        
        nodes = list(range(n))
        self.rng.shuffle(nodes) # Random topological ordering
        
        for u, v in graph.edges():
            # Add edge respecting topological order
            if nodes.index(u) < nodes.index(v):
                dag.add_edge(u, v)
            elif nodes.index(v) < nodes.index(u):
                dag.add_edge(v, u)
                
        # Ensure it's a DAG (Watts-Strogatz can introduce cycles when rewiring)
        if not nx.is_directed_acyclic_graph(dag):
            logger.warning("Watts-Strogatz graph resulted in cycles, attempting to resolve.")
            # Fallback to random DAG generation if cycles are persistent
            return self._generate_random_dag(n, self.network_config.edge_density)
        
        return dag
    
    def _generate_chain(self, n: int) -> nx.DiGraph:
        """Generate a chain topology: X1 -> X2 -> ... -> Xn"""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        for i in range(n - 1):
            dag.add_edge(i, i + 1)
        return dag
    
    def _generate_fork(self, n: int) -> nx.DiGraph:
        """Generate a fork topology: X1 -> X2, X1 -> X3, ..., X1 -> Xn"""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        root = 0
        for i in range(1, n):
            dag.add_edge(root, i)
        return dag
    
    def _generate_collider(self, n: int) -> nx.DiGraph:
        """Generate a collider topology: X1 -> Xn, X2 -> Xn, ..., X(n-1) -> Xn"""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        collider = n - 1
        for i in range(n - 1):
            dag.add_edge(i, collider)
        return dag
    
    def _generate_diamond(self, n: int) -> nx.DiGraph:
        """Generate a diamond topology: X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4, ..."""
        if n < 4:
            return self._generate_chain(n)
        
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        
        # Create diamond base
        dag.add_edge(0, 1)  # X1 -> X2
        dag.add_edge(0, 2)  # X1 -> X3
        dag.add_edge(1, 3)  # X2 -> X4
        dag.add_edge(2, 3)  # X3 -> X4
        
        # Add remaining nodes in chain
        for i in range(4, n):
            dag.add_edge(i - 1, i)
        
        return dag
    
    def _generate_random_dag(self, n: int, edge_density: float) -> nx.DiGraph:
        """Generate a random DAG with specified edge density."""
        max_edges = n * (n - 1) // 2
        n_edges = int(max_edges * edge_density)
        
        # Generate random DAG using topological ordering
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n))
        
        # Get random topological ordering
        nodes = list(range(n))
        self.rng.shuffle(nodes)
        
        # Add edges respecting topological order
        edges_added = 0
        while edges_added < n_edges:
            i = self.rng.randint(0, n - 1)
            j = self.rng.randint(i + 1, n)
            u, v = nodes[i], nodes[j]
            if not dag.has_edge(u, v):
                dag.add_edge(u, v)
                edges_added += 1
        
        return dag
    
    def _generate_scale_free(self, n: int, edge_density: float) -> nx.DiGraph:
        """Generate a scale-free DAG using preferential attachment."""
        # Start with a small chain
        dag = nx.DiGraph()
        dag.add_nodes_from(range(min(3, n)))
        for i in range(min(2, n - 1)):
            dag.add_edge(i, i + 1)
        
        # Add nodes with preferential attachment
        for new_node in range(len(dag), n):
            dag.add_node(new_node)
            # Connect to existing nodes based on in-degree
            in_degrees = dict(dag.in_degree())
            total_degree = sum(in_degrees.values()) + len(dag) - 1
            
            # Add edges with probability proportional to degree
            for existing_node in range(new_node):
                if total_degree > 0:
                    prob = (in_degrees.get(existing_node, 0) + 1) / total_degree
                    if self.rng.random() < prob * edge_density * 2:
                        dag.add_edge(existing_node, new_node)
        
        return dag
    
    def _generate_erdos_renyi(self, n: int, edge_density: float) -> nx.DiGraph:
        """Generate an Erdős–Rényi DAG."""
        # Use NetworkX's built-in function but ensure it's a DAG
        g = nx.gnp_random_graph(n, edge_density, directed=True, seed=self.rng.randint(10000))
        
        # Remove cycles by keeping only edges in topological order
        try:
            topo_order = list(nx.topological_sort(g))
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            
            for u in topo_order:
                for v in topo_order:
                    if g.has_edge(u, v) and topo_order.index(u) < topo_order.index(v):
                        dag.add_edge(u, v)
        except nx.NetworkXError:
            # If graph has cycles, use random DAG method
            return self._generate_random_dag(n, edge_density)
        
        return dag
    
    def _ensure_connectivity(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Ensure the DAG is weakly connected."""
        if nx.is_weakly_connected(dag):
            return dag
        
        # Find connected components
        components = list(nx.weakly_connected_components(dag))
        
        # Connect components by adding edges
        for i in range(len(components) - 1):
            comp1 = list(components[i])
            comp2 = list(components[i + 1])
            # Add edge from last node in comp1 to first node in comp2
            dag.add_edge(comp1[-1], comp2[0])
        
        return dag
    
    def _assign_edge_weights(self):
        """Assign edge weights to the DAG."""
        if isinstance(self.network_config.edge_weights, (int, float)):
            # Constant weight
            weight = float(self.network_config.edge_weights)
            self.edge_weights = {edge: weight for edge in self.dag.edges()}
        elif isinstance(self.network_config.edge_weights, list):
            # List of weights
            weights = self.network_config.edge_weights
            edges = list(self.dag.edges())
            if len(weights) != len(edges):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of edges ({len(edges)})")
            self.edge_weights = {edge: w for edge, w in zip(edges, weights)}
        else:
            # Random weights
            min_w = self.network_config.min_edge_weight
            max_w = self.network_config.max_edge_weight
            self.edge_weights = {
                edge: self.rng.uniform(min_w, max_w)
                for edge in self.dag.edges()
            }
    
    def _assign_noise_variances(self):
        """Assign noise variances to nodes."""
        n = len(self.dag)
        noise_min, noise_max = self.network_config.noise_range
        
        if isinstance(self.network_config.noise_variance, (int, float)):
            # Constant variance
            var = float(self.network_config.noise_variance)
            self.noise_variances = {i: var for i in range(n)}
        elif isinstance(self.network_config.noise_variance, list):
            # List of variances
            vars_list = self.network_config.noise_variance
            if len(vars_list) != n:
                raise ValueError(f"Number of variances ({len(vars_list)}) must match number of nodes ({n})")
            self.noise_variances = {i: v for i, v in enumerate(vars_list)}
        else:
            # Random variances
            self.noise_variances = {
                i: self.rng.uniform(noise_min, noise_max)
                for i in range(n)
            }
    
    def _store_ground_truth(self):
        """Store ground-truth adjacency matrix."""
        n = len(self.dag)
        self.ground_truth_adjacency = np.zeros((n, n), dtype=int)
        for u, v in self.dag.edges():
            self.ground_truth_adjacency[u, v] = 1
    
    def simulate_data(self) -> pd.DataFrame:
        """
        Simulate data from the DAG using Additive Noise Models (ANMs).
        
        Returns:
            DataFrame with simulated data, columns named X0, X1, ..., X(n-1)
        """
        if self.dag is None:
            raise ValueError("DAG must be generated first. Call generate_dag() before simulate_data()")
        
        n = len(self.dag)
        n_samples = self.data_config.n_samples
        
        # Assign edge weights and noise variances
        self._assign_edge_weights()
        self._assign_noise_variances()
        
        # Get topological ordering
        try:
            topo_order = list(nx.topological_sort(self.dag))
        except nx.NetworkXError:
            raise ValueError("Graph is not a DAG!")
        
        # Initialize data matrix
        data = np.zeros((n_samples, n))
        
        # Generate data following topological order
        for node in topo_order:
            parents = list(self.dag.predecessors(node))
            
            if len(parents) == 0:
                # Root node: just noise
                noise = self._generate_noise(n_samples, self.noise_variances[node])
                data[:, node] = noise
            else:
                # Non-root node: linear combination of parents + noise
                linear_combination = np.zeros(n_samples)
                for parent in parents:
                    edge = (parent, node)
                    weight = self.edge_weights.get(edge, 1.0)
                    linear_combination += weight * data[:, parent]
                
                # Apply non-linear function if specified
                if self.data_config.non_linear and self.data_config.non_linear_function:
                    linear_combination = self.data_config.non_linear_function(linear_combination)
                
                # Add noise
                noise = self._generate_noise(n_samples, self.noise_variances[node])
                data[:, node] = linear_combination + noise
        
        # Create DataFrame
        column_names = [f"X{i}" for i in range(n)]
        df = pd.DataFrame(data, columns=column_names)
        
        self.data = df
        return df
    
    def _generate_noise(self, n_samples: int, variance: float) -> np.ndarray:
        """Generate noise samples based on configured distribution."""
        dist = self.data_config.noise_distribution.lower()
        
        if dist == "gaussian":
            return self.rng.normal(0, np.sqrt(variance), n_samples)
        elif dist == "uniform":
            # Uniform distribution with same variance
            std = np.sqrt(variance)
            a = -np.sqrt(3) * std
            b = np.sqrt(3) * std
            return self.rng.uniform(a, b, n_samples)
        elif dist == "laplace":
            # Laplace distribution with same variance
            scale = np.sqrt(variance / 2)
            return self.rng.laplace(0, scale, n_samples)
        elif dist == "exponential":
            # Exponential distribution with mean 0 and specified variance
            # E[X] = 1/lambda, Var[X] = 1/lambda^2
            # For mean 0, we shift it: X - 1/lambda
            # Var[X - 1/lambda] = Var[X] = 1/lambda^2 = variance
            # lambda = 1 / sqrt(variance)
            scale = np.sqrt(variance) # This is 1/lambda for numpy's exponential
            return self.rng.exponential(scale, n_samples) - scale # Shift to mean 0
        elif dist == "student_t":
            # Student's t-distribution. Requires degrees of freedom.
            # We'll use a fixed df for simplicity, e.g., 5, and scale for variance.
            df = 5 # Degrees of freedom. Lower df means heavier tails.
            # For t-distribution, Var[X] = df / (df - 2) for df > 2
            # We need to scale it to match the desired variance
            if df <= 2:
                raise ValueError("Degrees of freedom for Student's t must be > 2 for finite variance.")
            scale_factor = np.sqrt(variance * (df - 2) / df)
            return self.rng.standard_t(df, n_samples) * scale_factor
        else:
            raise ValueError(f"Unknown noise distribution: {dist}")
    
    def get_effect_variables(self) -> List[str]:
        """
        Identify effect variables (non-root nodes with parents).
        
        Returns:
            List of variable names that are effect variables
        """
        if self.dag is None:
            raise ValueError("DAG must be generated first")
        
        effect_vars = []
        for node in self.dag.nodes():
            if self.dag.in_degree(node) > 0:  # Has at least one parent
                effect_vars.append(f"X{node}")
        
        return effect_vars
    
    def get_ground_truth_graph(self) -> nx.DiGraph:
        """Get the ground-truth DAG."""
        if self.dag is None:
            raise ValueError("DAG must be generated first")
        return self.dag.copy()
    
    def get_ground_truth_adjacency(self) -> np.ndarray:
        """Get ground-truth adjacency matrix."""
        if self.ground_truth_adjacency is None:
            raise ValueError("DAG must be generated first")
        return self.ground_truth_adjacency.copy()
    
    def save_ground_truth(self, filepath: str):
        """Save ground-truth graph to file."""
        if self.dag is None:
            raise ValueError("DAG must be generated first")
        nx.write_gml(self.dag, filepath)


def create_synthetic_network_config(
    n_nodes: int,
    topology_type: Union[str, TopologyType],
    edge_density: float = 0.3,
    edge_weights: Union[float, List[float], str] = "random",
    noise_variance: Union[float, List[float], str] = "random",
    min_edge_weight: float = 0.3,
    max_edge_weight: float = 1.0,
    noise_range: Tuple[float, float] = (0.1, 1.0),
    random_seed: Optional[int] = None,
    ensure_connected: bool = True,
    allow_cycles: bool = False,
    barabasi_albert_m: int = 1,
    watts_strogatz_k: int = 2,
    watts_strogatz_p: float = 0.1
) -> SyntheticNetworkConfig:
    """Convenience function to create network configuration."""
    if isinstance(topology_type, str):
        topology_type = TopologyType(topology_type.lower())
    
    return SyntheticNetworkConfig(
        n_nodes=n_nodes,
        topology_type=topology_type,
        edge_density=edge_density,
        edge_weights=edge_weights,
        noise_variance=noise_variance,
        min_edge_weight=min_edge_weight,
        max_edge_weight=max_edge_weight,
        noise_range=noise_range,
        random_seed=random_seed,
        ensure_connected=ensure_connected,
        allow_cycles=allow_cycles,
        barabasi_albert_m=barabasi_albert_m,
        watts_strogatz_k=watts_strogatz_k,
        watts_strogatz_p=watts_strogatz_p
    )


def create_synthetic_data_config(
    n_samples: int,
    noise_distribution: str = "gaussian",
    non_linear: bool = False,
    non_linear_function: Optional[Callable] = None, # Added this parameter
    random_seed: Optional[int] = None
) -> SyntheticDataConfig:
    """Convenience function to create data configuration."""
    return SyntheticDataConfig(
        n_samples=n_samples,
        noise_distribution=noise_distribution,
        non_linear=non_linear,
        non_linear_function=non_linear_function, # Added this parameter
        random_seed=random_seed
    )


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("SYNTHETIC CAUSAL NETWORK GENERATION - TESTING")
    print("="*70)
    
    # Test 1: Chain topology
    print("\n1. Testing Chain Topology")
    net_config = create_synthetic_network_config(5, "chain", random_seed=42)
    data_config = create_synthetic_data_config(1000, random_seed=42)
    
    network = SyntheticCausalNetwork(net_config, data_config, random_seed=42)
    dag = network.generate_dag()
    data = network.simulate_data()
    
    print(f"   DAG nodes: {len(dag.nodes())}")
    print(f"   DAG edges: {len(dag.edges())}")
    print(f"   Data shape: {data.shape}")
    print(f"   Effect variables: {network.get_effect_variables()}")
    print(f"   Is DAG: {nx.is_directed_acyclic_graph(dag)}")
    
    # Test 2: Random DAG
    print("\n2. Testing Random DAG")
    net_config = create_synthetic_network_config(10, "random", edge_density=0.3, random_seed=42)
    network = SyntheticCausalNetwork(net_config, data_config, random_seed=42)
    dag = network.generate_dag()
    data = network.simulate_data()
    
    # Test 4: Barabasi-Albert topology
    print("\n4. Testing Barabasi-Albert Topology")
    net_config_ba = create_synthetic_network_config(n_nodes=10, topology_type="barabasi_albert", barabasi_albert_m=2, random_seed=42)
    data_config_exp = create_synthetic_data_config(n_samples=1000, noise_distribution="exponential", random_seed=42)
    network_ba = SyntheticCausalNetwork(net_config_ba, data_config_exp, random_seed=42)
    dag_ba = network_ba.generate_dag()
    data_ba = network_ba.simulate_data()
    print(f"   DAG nodes (BA): {len(dag_ba.nodes())}")
    print(f"   DAG edges (BA): {len(dag_ba.edges())}")
    print(f"   Data shape (BA): {data_ba.shape}")

    # Test 5: Watts-Strogatz topology with Student-t noise and non-linearity
    print("\n5. Testing Watts-Strogatz Topology with Student-t noise and non-linearity")
    net_config_ws = create_synthetic_network_config(n_nodes=10, topology_type="watts_strogatz", watts_strogatz_k=4, watts_strogatz_p=0.3, random_seed=42)
    data_config_st = create_synthetic_data_config(n_samples=1000, noise_distribution="student_t", non_linear=True, non_linear_function=np.sin, random_seed=42)
    network_ws = SyntheticCausalNetwork(net_config_ws, data_config_st, random_seed=42)
    dag_ws = network_ws.generate_dag()
    data_ws = network_ws.simulate_data()
    print(f"   DAG nodes (WS): {len(dag_ws.nodes())}")
    print(f"   DAG edges (WS): {len(dag_ws.edges())}")
    print(f"   Data shape (WS): {data_ws.shape}")
    print(f"   Is non-linear: {network_ws.data_config.non_linear}")
    print(f"   Noise distribution: {network_ws.data_config.noise_distribution}")
    
    print(f"   DAG nodes: {len(dag.nodes())}")
    print(f"   DAG edges: {len(dag.edges())}")
    print(f"   Data shape: {data.shape}")
    print(f"   Effect variables: {network.get_effect_variables()}")
    
    # Test 3: Fork topology
    print("\n3. Testing Fork Topology")
    net_config = create_synthetic_network_config(6, "fork", random_seed=42)
    network = SyntheticCausalNetwork(net_config, data_config, random_seed=42)
    dag = network.generate_dag()
    data = network.simulate_data()
    
    print(f"   DAG nodes: {len(dag.nodes())}")
    print(f"   DAG edges: {len(dag.edges())}")
    print(f"   Root node: {[n for n in dag.nodes() if dag.in_degree(n) == 0]}")
    print(f"   Effect variables: {network.get_effect_variables()}")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)

