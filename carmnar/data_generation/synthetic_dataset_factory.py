"""
Synthetic Dataset Factory for Factorial Experimental Framework

This module creates synthetic dataset configurations that integrate seamlessly
with the existing FactorialExperimentFramework, enabling systematic evaluation
of causal discovery algorithms on synthetic data with known ground truth.

Author: Research Team
Date: 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import sys
from itertools import product
from pathlib import Path
import pandas as pd
import networkx as nx

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from carmnar.data_generation.synthetic_causal_network import (
    SyntheticCausalNetwork,
    SyntheticNetworkConfig,
    SyntheticDataConfig,
    TopologyType,
    create_synthetic_network_config,
    create_synthetic_data_config
)

from carmnar.algorithms.factorial_experiment_framework import DatasetConfig


class SyntheticDatasetFactory:
    """
    Factory class for creating synthetic dataset configurations for factorial experiments.
    
    This class generates synthetic datasets with varying properties (network size,
    topology, edge density, sample size) and integrates them with the existing
    factorial experimental framework.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the synthetic dataset factory.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.generated_networks = {}
        self.ground_truth_graphs = {}
    
    def generate_synthetic_datasets(
        self,
        n_nodes_list: List[int] = [5, 10, 15],
        topology_types: List[str] = ["random", "chain", "fork", "barabasi_albert", "watts_strogatz"],
        edge_densities: List[float] = [0.3],
        edge_weights_list: List[Any] = ["random"],
        sample_sizes: List[int] = [500, 1000],
        noise_distributions: List[str] = ["gaussian", "exponential", "student_t"],
        non_linear_functions: List[Optional[Callable]] = [None],
        barabasi_albert_m_list: List[int] = [1],
        watts_strogatz_k_list: List[int] = [2],
        watts_strogatz_p_list: List[float] = [0.1],
        output_dir: Optional[str] = None
    ) -> List[DatasetConfig]:
        """
        Generate multiple synthetic dataset configurations.
        
        Args:
            n_nodes_list: List of network sizes to generate
            topology_types: List of topology types
            edge_densities: List of edge densities
            edge_weights_list: List of edge weight specifications
            sample_sizes: List of sample sizes
            noise_distributions: List of noise distributions
            output_dir: Directory to save generated datasets and ground truth
            
        Returns:
            List of DatasetConfig objects ready for factorial experiments
        """
        datasets = []
        
        # Generate all combinations
        for n_nodes, topology, edge_density, edge_weights, n_samples, noise_dist, non_linear_func, ba_m, ws_k, ws_p in product(
            n_nodes_list, topology_types, edge_densities, 
            edge_weights_list, sample_sizes, noise_distributions, 
            non_linear_functions, barabasi_albert_m_list, 
            watts_strogatz_k_list, watts_strogatz_p_list
        ):
            # Create unique identifier
            dataset_id = f"Synth_{n_nodes}n_{topology}_{edge_density}ed_{n_samples}s_{noise_dist}"
            if non_linear_func:
                dataset_id += f"_nonlinear_{non_linear_func.__name__}"
            if topology == "barabasi_albert":
                dataset_id += f"_m{ba_m}"
            elif topology == "watts_strogatz":
                dataset_id += f"_k{ws_k}_p{ws_p}"
            
            # Generate network and data
            network_config = create_synthetic_network_config(
                n_nodes=n_nodes,
                topology_type=topology,
                edge_density=edge_density,
                edge_weights=edge_weights,
                random_seed=self.rng.randint(10000),
                barabasi_albert_m=ba_m,
                watts_strogatz_k=ws_k,
                watts_strogatz_p=ws_p
            )
            
            data_config = create_synthetic_data_config(
                n_samples=n_samples,
                noise_distribution=noise_dist,
                non_linear=non_linear_func is not None,
                non_linear_function=non_linear_func,
                random_seed=self.rng.randint(10000)
            )
            
            # Create synthetic network
            network = SyntheticCausalNetwork(
                network_config, 
                data_config, 
                random_seed=self.rng.randint(10000)
            )
            
            # Generate DAG and data
            dag = network.generate_dag()
            data = network.simulate_data()
            
            # Get effect variables (non-root nodes)
            effect_vars = network.get_effect_variables()
            if len(effect_vars) == 0:
                # If no effect variables, skip this configuration
                continue
            
            # Select first effect variable as target
            target_variable = effect_vars[0]
            
            # Save data and ground truth if output directory specified
            gt_path = None
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save data
                data_path = output_path / f"{dataset_id}.csv"
                data.to_csv(data_path, index=False)
                
                # Save ground truth graph
                gt_path = output_path / f"{dataset_id}_ground_truth.gml"
                network.save_ground_truth(str(gt_path))
                
                data_file_path = str(data_path)
            else:
                # In-memory only
                data_file_path = f"memory://{dataset_id}"
            
            # Store network for later retrieval (always)
            self.generated_networks[dataset_id] = network
            self.ground_truth_graphs[dataset_id] = dag
            
            # Create DatasetConfig
            dataset_config = DatasetConfig(
                name=dataset_id,
                path=data_file_path,
                target_variable=target_variable,
                description=f"Synthetic {topology} network: {n_nodes} nodes, {edge_density} density, {n_samples} samples",
                n_variables=n_nodes,
                n_samples=n_samples,
                causal_properties={
                    "topology": topology,
                    "edge_density": edge_density,
                    "n_nodes": n_nodes,
                    "n_edges": len(dag.edges()),
                    "effect_variables": effect_vars,
                    "noise_distribution": noise_dist,
                    "non_linear": non_linear_func is not None,
                    "non_linear_function": non_linear_func.__name__ if non_linear_func else None,
                    "barabasi_albert_m": ba_m if topology == "barabasi_albert" else None,
                    "watts_strogatz_k": ws_k if topology == "watts_strogatz" else None,
                    "watts_strogatz_p": ws_p if topology == "watts_strogatz" else None,
                    "ground_truth_path": str(gt_path) if output_dir else None
                }
            )
            
            datasets.append(dataset_config)
        
        return datasets
    
    def get_ground_truth_graph(self, dataset_id: str) -> Optional[nx.DiGraph]:
        """Get ground-truth graph for a dataset."""
        return self.ground_truth_graphs.get(dataset_id)
    
    def get_network(self, dataset_id: str) -> Optional[SyntheticCausalNetwork]:
        """Get synthetic network object for a dataset."""
        return self.generated_networks.get(dataset_id)
    
    def create_focused_design_datasets(
        self,
        output_dir: Optional[str] = None
    ) -> List[DatasetConfig]:
        """
        Create focused design datasets for initial paper (as per proposal).
        
        This creates a manageable set of synthetic datasets for the first paper:
        - Network sizes: 5, 10, 15
        - Topology: Random (fixed)
        - Edge density: 0.3 (fixed)
        - Edge weights: 0.5, 1.0
        - Sample sizes: 500, 1000
        - Noise: Gaussian (fixed)
        
        Returns:
            List of DatasetConfig objects
        """
        return self.generate_synthetic_datasets(
            n_nodes_list=[5, 10, 15],
            topology_types=["random"],
            edge_densities=[0.3],
            edge_weights_list=[0.5, 1.0],
            sample_sizes=[500, 1000],
            noise_distributions=["gaussian"],
            non_linear_functions=[None],
            output_dir=output_dir
        )
    
    def create_comprehensive_design_datasets(
        self,
        output_dir: Optional[str] = None
    ) -> List[DatasetConfig]:
        """
        Create comprehensive design datasets for full evaluation.
        
        This creates the full experimental design as specified in the proposal.
        
        Returns:
            List of DatasetConfig objects
        """
        return self.generate_synthetic_datasets(
            n_nodes_list=[5, 10, 15, 20],
            topology_types=["random", "chain", "fork", "collider", "barabasi_albert", "watts_strogatz"],
            edge_densities=[0.2, 0.3, 0.4],
            edge_weights_list=[0.3, 0.5, 0.7, 1.0],
            sample_sizes=[200, 500, 1000, 2000],
            noise_distributions=["gaussian", "uniform", "laplace", "exponential", "student_t"],
            non_linear_functions=[None, np.sin, np.square],
            barabasi_albert_m_list=[1, 2],
            watts_strogatz_k_list=[2, 4],
            watts_strogatz_p_list=[0.1, 0.3],
            output_dir=output_dir
        )


def load_synthetic_dataset(
    dataset_config: DatasetConfig,
    factory: SyntheticDatasetFactory
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Load synthetic dataset and its ground-truth graph.
    
    Args:
        dataset_config: Dataset configuration
        factory: Synthetic dataset factory that generated the dataset
        
    Returns:
        Tuple of (data DataFrame, ground-truth DAG)
    """
    # Check if it's a memory dataset
    if dataset_config.path.startswith("memory://"):
        dataset_id = dataset_config.path.replace("memory://", "")
        network = factory.get_network(dataset_id)
        if network is None:
            raise ValueError(f"Dataset {dataset_id} not found in factory")
        return network.data, factory.get_ground_truth_graph(dataset_id)
    else:
        # Load from file
        data = pd.read_csv(dataset_config.path)
        gt_path = dataset_config.causal_properties.get("ground_truth_path")
        if gt_path and Path(gt_path).exists():
            dag = nx.read_gml(gt_path)
            return data, dag
        else:
            raise ValueError(f"Ground truth graph not found for {dataset_config.name}")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SYNTHETIC DATASET FACTORY - TESTING")
    print("="*70)
    
    # Test 1: Chain topology
    print("\n1. Testing Chain Topology")
    factory = SyntheticDatasetFactory(random_seed=42)
    datasets = factory.create_focused_design_datasets(output_dir="results/synthetic_datasets")

    # Test 2: Comprehensive design datasets
    print("\n2. Generating comprehensive design datasets...")
    comprehensive_datasets = factory.create_comprehensive_design_datasets(output_dir="results/synthetic_datasets")
    print(f"\nGenerated {len(comprehensive_datasets)} comprehensive synthetic datasets.")

    # Test 3: Loading a specific dataset (from comprehensive list)
    print("\n" + "="*70)
    print("Testing loading a specific dataset...")
    if len(comprehensive_datasets) > 0:
        test_ds = comprehensive_datasets[0]
        data, gt_graph = load_synthetic_dataset(test_ds, factory)
        print(f"Loaded dataset: {test_ds.name}")
        print(f"Data shape: {data.shape}")
        print(f"Ground truth nodes: {len(gt_graph.nodes())}")
        print(f"Ground truth edges: {len(gt_graph.edges())}")
        print(f"Data columns: {list(data.columns)}")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
