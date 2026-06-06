"""
Synthetic Data Experiments Runner

This module integrates synthetic data generation with the factorial experimental
framework to run comprehensive experiments on synthetic causal networks.

Author: Research Team
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from carmnar.data_generation.synthetic_dataset_factory import (
    SyntheticDatasetFactory,
    load_synthetic_dataset
)
from carmnar.algorithms.factorial_experiment_framework import (
    FactorialExperimentFramework,
    DatasetConfig,
    MissingnessConfig,
    CausalDiscoveryAlgorithm,
    ExperimentalCondition,
    ExperimentalResult
)
from carmnar.algorithms.general_mnar_optimizer import (
    GeneralMNAROptimizer,
    SigmoidModel,
    GPDModel,
    ThresholdModel,
    introduce_missingness
)
from carmnar.evaluation.metrics import calculate_shd, calculate_rel_shd
from carmnar.evaluation.advanced_structural_metrics import AdvancedStructuralMetrics


class SyntheticDataExperimentRunner:
    """
    Runner for synthetic data experiments integrated with factorial framework.
    
    This class extends the factorial framework to handle synthetic datasets
    with known ground-truth graphs, enabling precise evaluation of algorithm
    performance.
    """
    
    def __init__(
        self,
        factory: SyntheticDatasetFactory,
        output_dir: str = "results/synthetic_experiments",
        random_seed: int = 42
    ):
        """
        Initialize synthetic data experiment runner.
        
        Args:
            factory: Synthetic dataset factory
            output_dir: Directory for experiment results
            random_seed: Random seed for reproducibility
        """
        self.factory = factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.results = []
    
    def run_synthetic_experiments(
        self,
        datasets: List[DatasetConfig],
        missingness_mechanisms: List[MissingnessConfig],
        missingness_rates: List[float],
        algorithms: List[CausalDiscoveryAlgorithm],
        n_replicates: int = 20
    ) -> List[ExperimentalResult]:
        """
        Run synthetic data experiments using factorial design.
        
        Args:
            datasets: List of synthetic dataset configurations
            missingness_mechanisms: List of missingness mechanism configurations
            missingness_rates: List of missingness rates
            algorithms: List of causal discovery algorithms
            n_replicates: Number of replicates per condition
            
        Returns:
            List of experimental results
        """
        # Create factorial framework
        framework = FactorialExperimentFramework(
            datasets=datasets,
            missingness_mechanisms=missingness_mechanisms,
            missingness_rates=missingness_rates,
            algorithms=algorithms,
            n_replicates=n_replicates,
            random_seed=self.random_seed
        )
        
        # Override the evaluation method to use ground-truth graphs
        original_evaluate = framework._evaluate_performance
        framework._evaluate_performance = self._evaluate_with_ground_truth
        
        # Store factory reference for ground-truth access
        framework._synthetic_factory = self.factory
        
        # Run all experiments
        results = framework.run_all_experiments()
        
        return results
    
    def _evaluate_with_ground_truth(
        self,
        ground_truth: Any,
        inferred: Any,
        data_complete: pd.DataFrame,
        data_missing: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate performance using ground-truth graph from synthetic data.
        
        This method overrides the default evaluation to use the known
        ground-truth graph from synthetic network generation.
        """
        # Get ground-truth graph from factory
        # Note: This requires access to the condition, which we'll handle differently
        # For now, we'll use a simpler approach
        
        # If ground_truth is already a graph, use it
        if isinstance(ground_truth, nx.DiGraph):
            true_graph = ground_truth
        else:
            # Try to get from condition (this is a workaround)
            # In practice, we'll need to modify the framework to pass dataset info
            return self._evaluate_placeholder(inferred, data_complete, data_missing)
        
        # Convert inferred graph (assuming it's in the same format)
        if isinstance(inferred, nx.DiGraph):
            inferred_graph = inferred
        else:
            # Convert from other formats if needed
            inferred_graph = self._convert_to_graph(inferred, data_complete.columns)
        
        # Calculate metrics using advanced structural metrics
        metrics_calculator = AdvancedStructuralMetrics()
        metrics = metrics_calculator.calculate_comprehensive_metrics(
            true_graph, inferred_graph
        )
        
        return metrics
    
    def _evaluate_placeholder(
        self,
        inferred: Any,
        data_complete: pd.DataFrame,
        data_missing: pd.DataFrame
    ) -> Dict[str, float]:
        """Placeholder evaluation when ground truth is not available."""
        # This is a fallback - in practice, we'll ensure ground truth is always available
        return {
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'shd': float('inf'),
            'n_samples': len(data_complete),
            'n_variables': len(data_complete.columns),
            'actual_missing_rate': data_missing.isnull().any(axis=1).mean()
        }
    
    def _convert_to_graph(self, graph_repr: Any, node_names: List[str]) -> nx.DiGraph:
        """Convert graph representation to NetworkX DiGraph."""
        # This is a placeholder - actual implementation depends on algorithm output format
        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        # Add edges based on graph_repr format
        # Implementation depends on specific algorithm output
        return g


def create_synthetic_missingness_configs() -> List[MissingnessConfig]:
    """Create missingness mechanism configurations for synthetic experiments."""
    from carmnar.algorithms.factorial_experiment_framework import MissingnessConfig
    
    mechanisms = [
        MissingnessConfig(
            name="Sigmoid",
            model_type="sigmoid",
            parameter_grid={
                'alpha': np.linspace(0.05, 1.5, 20),
                'beta': np.linspace(-10, 0, 20)
            },
            description="Sigmoid-based MNAR"
        ),
        MissingnessConfig(
            name="GPD",
            model_type="gpd",
            parameter_grid={
                'u': [75, 80, 85, 90],  # Will be converted to quantiles
                'xi': np.linspace(0.05, 0.7, 15),
                'sigma': np.linspace(0.5, 6.0, 15)
            },
            description="Heavy-tailed MNAR using GPD"
        ),
        MissingnessConfig(
            name="Threshold",
            model_type="threshold",
            parameter_grid={
                'threshold': np.linspace(0, 100, 100)  # Will be adjusted per dataset
            },
            description="Threshold-based censoring"
        )
    ]
    
    return mechanisms


def run_synthetic_pilot_experiments(
    output_dir: str = "results/synthetic_pilot",
    n_replicates: int = 5
):
    """
    Run pilot synthetic data experiments.
    
    This function runs a small-scale pilot to validate the framework.
    """
    print("="*70)
    print("SYNTHETIC DATA PILOT EXPERIMENTS")
    print("="*70)
    
    # Create factory
    factory = SyntheticDatasetFactory(random_seed=42)
    
    # Generate focused design datasets
    print("\n1. Generating synthetic datasets...")
    datasets = factory.create_focused_design_datasets(
        output_dir=f"{output_dir}/datasets"
    )
    print(f"   Generated {len(datasets)} datasets")
    
    # Create missingness mechanisms
    print("\n2. Setting up missingness mechanisms...")
    mechanisms = create_synthetic_missingness_configs()
    print(f"   Configured {len(mechanisms)} mechanisms")
    
    # Missingness rates
    rates = [0.0, 0.1, 0.2, 0.3]
    
    # Mock algorithm (replace with actual SM-MVPC implementation)
    class MockSMMVPC(CausalDiscoveryAlgorithm):
        def learn_structure(self, data: pd.DataFrame):
            # Placeholder - would call actual SM-MVPC
            n_vars = len(data.columns)
            adj = np.random.choice([0, 1], size=(n_vars, n_vars), p=[0.8, 0.2])
            np.fill_diagonal(adj, 0)
            adj = np.triu(adj)  # Make it acyclic
            return {
                'adjacency': adj,
                'variables': list(data.columns)
            }
        
        def get_name(self) -> str:
            return "SM-MVPC"
    
    algorithms = [MockSMMVPC()]
    
    # Create experiment runner
    print("\n3. Running experiments...")
    runner = SyntheticDataExperimentRunner(
        factory=factory,
        output_dir=output_dir,
        random_seed=42
    )
    
    # Run experiments (limited for pilot)
    pilot_datasets = datasets[:3]  # Use first 3 datasets for pilot
    results = runner.run_synthetic_experiments(
        datasets=pilot_datasets,
        missingness_mechanisms=mechanisms[:2],  # Use first 2 mechanisms
        missingness_rates=rates,
        algorithms=algorithms,
        n_replicates=n_replicates
    )
    
    print(f"\n4. Completed {len(results)} experiments")
    
    # Analyze results
    print("\n5. Analyzing results...")
    if results:
        df_results = pd.DataFrame([
            {
                'dataset': r.condition.dataset.name,
                'mechanism': r.condition.missingness_mechanism.name,
                'rate': r.condition.missingness_rate,
                'f1_score': r.metrics.get('f1_score', 0),
                'precision': r.metrics.get('precision', 0),
                'recall': r.metrics.get('recall', 0)
            }
            for r in results
        ])
        
        summary = df_results.groupby(['dataset', 'mechanism', 'rate']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(3)
        
        print("\nResults Summary:")
        print(summary)
    
    print("\n" + "="*70)
    print("Pilot experiments completed!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Run pilot experiments
    results = run_synthetic_pilot_experiments(n_replicates=3)

