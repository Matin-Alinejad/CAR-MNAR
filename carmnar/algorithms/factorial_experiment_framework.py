"""
General Factorial Experimental Framework for Causal Discovery under Missing Data

This module implements a comprehensive factorial design framework for systematically
evaluating causal discovery algorithms under various missing data conditions.
The framework supports arbitrary combinations of datasets, missingness mechanisms,
missingness rates, and causal discovery algorithms.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Protocol, Callable
from dataclasses import dataclass, field
from itertools import product
import json
import logging
from pathlib import Path
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our existing modules
from carmnar.algorithms.general_mnar_optimizer import (
    GeneralMNAROptimizer,
    SigmoidModel,
    GPDModel,
    ThresholdModel,
    MissingnessModel,
    introduce_missingness
)


@dataclass
class DatasetConfig:
    """Configuration for a dataset in the factorial design."""
    name: str
    path: str
    target_variable: str
    description: str
    n_variables: int
    n_samples: int
    causal_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingnessConfig:
    """Configuration for a missingness mechanism."""
    name: str
    model_type: str  # 'sigmoid', 'gpd', 'threshold'
    parameter_grid: Dict[str, List[float]]
    description: str


@dataclass
class ExperimentalFactor:
    """Represents a factor in the factorial design."""
    name: str
    levels: List[Any]
    description: str


@dataclass
class ExperimentalCondition:
    """Represents a single experimental condition (combination of factor levels)."""
    dataset: DatasetConfig
    missingness_mechanism: MissingnessConfig
    missingness_rate: float
    algorithm_name: str
    replicate_id: int
    condition_id: str = field(init=False)
    
    def __post_init__(self):
        """Generate unique condition ID."""
        self.condition_id = f"{self.dataset.name}_{self.missingness_mechanism.name}_{self.missingness_rate:.0%}_{self.algorithm_name}_rep{self.replicate_id}"


@dataclass
class ExperimentalResult:
    """Stores results from a single experimental run."""
    condition: ExperimentalCondition
    metrics: Dict[str, float]
    execution_time: float
    ground_truth_graph: Any  # Placeholder for graph structure
    inferred_graph: Any  # Placeholder for graph structure
    optimization_gap: float
    actual_missingness_rate: float
    timestamp: str


class CausalDiscoveryAlgorithm(Protocol):
    """Protocol for causal discovery algorithms."""
    
    def learn_structure(self, data: pd.DataFrame) -> Any:
        """Learn causal structure from data."""
        ...
    
    def get_name(self) -> str:
        """Get algorithm name."""
        ...


class FactorialExperimentFramework:
    """
    A comprehensive framework for factorial experimental design in causal discovery
    with missing data.
    
    This framework implements a general factorial design where:
    - Factor 1: Datasets (D)
    - Factor 2: Missingness Mechanisms (M)
    - Factor 3: Missingness Rates (R)
    - Factor 4: Causal Discovery Algorithms (A)
    - Replication: Multiple runs per condition
    """
    
    def __init__(self, 
                 datasets: List[DatasetConfig],
                 missingness_mechanisms: List[MissingnessConfig],
                 missingness_rates: List[float],
                 algorithms: List[CausalDiscoveryAlgorithm],
                 n_replicates: int = 5,
                 output_dir: str = "results/factorial_experiments",
                 random_seed: int = 42):
        """
        Initialize the factorial experiment framework.
        
        Args:
            datasets: List of dataset configurations
            missingness_mechanisms: List of missingness mechanism configurations
            missingness_rates: List of target missingness rates (0 to 1)
            algorithms: List of causal discovery algorithms to evaluate
            n_replicates: Number of replications per condition
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.missingness_mechanisms = missingness_mechanisms
        self.missingness_rates = missingness_rates
        self.algorithms = algorithms
        self.n_replicates = n_replicates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Define experimental factors
        self.factors = [
            ExperimentalFactor("Dataset", datasets, "Different causal structures"),
            ExperimentalFactor("Missingness", missingness_mechanisms, "MNAR mechanisms"),
            ExperimentalFactor("Rate", missingness_rates, "Missing data percentages"),
            ExperimentalFactor("Algorithm", algorithms, "Causal discovery methods")
        ]
        
        # Generate all experimental conditions
        self.conditions = self._generate_conditions()
        self.results = []
        
        logger.info(f"Initialized factorial design with {len(self.conditions)} conditions")
        logger.info(f"Total experiments: {len(self.conditions)} × {n_replicates} = {len(self.conditions) * n_replicates}")
    
    def _generate_conditions(self) -> List[ExperimentalCondition]:
        """Generate all experimental conditions from the factorial design."""
        conditions = []
        
        # Generate full factorial design
        for dataset, mechanism, rate, algorithm in product(
            self.datasets, 
            self.missingness_mechanisms, 
            self.missingness_rates,
            self.algorithms
        ):
            for rep in range(self.n_replicates):
                condition = ExperimentalCondition(
                    dataset=dataset,
                    missingness_mechanism=mechanism,
                    missingness_rate=rate,
                    algorithm_name=algorithm.get_name(),
                    replicate_id=rep
                )
                conditions.append(condition)
        
        return conditions
    
    def _create_missingness_model(self, config: MissingnessConfig) -> MissingnessModel:
        """Create appropriate missingness model based on configuration."""
        if config.model_type == 'sigmoid':
            return SigmoidModel()
        elif config.model_type == 'gpd':
            return GPDModel()
        elif config.model_type == 'threshold':
            return ThresholdModel()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def _load_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        """Load dataset from configuration."""
        try:
            data = pd.read_csv(dataset_config.path)
            # Clean data: remove rows with missing or zero values in target variable
            if dataset_config.target_variable in data.columns:
                data = data[data[dataset_config.target_variable] > 0].dropna(
                    subset=[dataset_config.target_variable]
                ).reset_index(drop=True)
            return data
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_config.name}: {e}")
            raise
    
    def _introduce_missingness(self, 
                              data: pd.DataFrame,
                              condition: ExperimentalCondition) -> Tuple[pd.DataFrame, float, float]:
        """
        Introduce missingness according to experimental condition.
        
        Returns:
            Tuple of (data_with_missing, optimization_gap, actual_missing_rate)
        """
        # Create missingness model
        model = self._create_missingness_model(condition.missingness_mechanism)
        
        # If missingness rate is 0, return original data
        if condition.missingness_rate == 0:
            return data.copy(), 0.0, 0.0
        
        # Create optimizer
        optimizer = GeneralMNAROptimizer(model, random_state=self.rng.randint(10000))
        
        # Get target variable values
        target_var = condition.dataset.target_variable
        values = data[target_var].values
        
        # Find optimal parameters
        best_params, gap = optimizer.find_optimal_parameters(
            values,
            condition.missingness_rate,
            condition.missingness_mechanism.parameter_grid
        )
        
        # Introduce missingness
        data_with_missing = introduce_missingness(
            data, 
            target_var,
            condition.missingness_rate,
            model,
            best_params,
            random_state=self.rng.randint(10000)
        )
        
        # Calculate actual missing rate
        actual_rate = data_with_missing[target_var].isnull().mean()
        
        return data_with_missing, gap, actual_rate
    
    def _evaluate_performance(self, 
                            ground_truth: Any,
                            inferred: Any,
                            data_complete: pd.DataFrame,
                            data_missing: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate performance metrics.
        
        Note: This is a placeholder. In practice, you would implement
        graph comparison metrics like F1, precision, recall, SHD, etc.
        """
        # Placeholder metrics
        metrics = {
            'f1_score': self.rng.uniform(0.6, 0.95),  # Placeholder
            'precision': self.rng.uniform(0.5, 1.0),  # Placeholder
            'recall': self.rng.uniform(0.7, 1.0),  # Placeholder
            'shd': self.rng.randint(0, 10),  # Placeholder
            'n_edges_true': 10,  # Placeholder
            'n_edges_inferred': 12,  # Placeholder
        }
        
        # Add data characteristics
        metrics['n_samples'] = len(data_complete)
        metrics['n_variables'] = len(data_complete.columns)
        metrics['actual_missing_rate'] = data_missing.isnull().any(axis=1).mean()
        
        return metrics
    
    def run_single_experiment(self, condition: ExperimentalCondition) -> ExperimentalResult:
        """Run a single experimental condition."""
        logger.info(f"Running experiment: {condition.condition_id}")
        start_time = time.time()
        
        try:
            # Load dataset
            data_complete = self._load_dataset(condition.dataset)
            
            # Learn ground truth from complete data (only if rate > 0)
            algorithm = next(a for a in self.algorithms if a.get_name() == condition.algorithm_name)
            ground_truth = algorithm.learn_structure(data_complete) if condition.missingness_rate > 0 else None
            
            # Introduce missingness
            data_missing, opt_gap, actual_rate = self._introduce_missingness(
                data_complete, condition
            )
            
            # Learn structure from data with missingness
            inferred = algorithm.learn_structure(data_missing)
            
            # Evaluate performance
            metrics = self._evaluate_performance(
                ground_truth, inferred, data_complete, data_missing
            )
            
            # Create result
            result = ExperimentalResult(
                condition=condition,
                metrics=metrics,
                execution_time=time.time() - start_time,
                ground_truth_graph=ground_truth,
                inferred_graph=inferred,
                optimization_gap=opt_gap,
                actual_missingness_rate=actual_rate,
                timestamp=pd.Timestamp.now().isoformat()
            )
            
            logger.info(f"Completed {condition.condition_id}: F1={metrics['f1_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed experiment {condition.condition_id}: {e}")
            # Return failed result
            return ExperimentalResult(
                condition=condition,
                metrics={'error': str(e)},
                execution_time=time.time() - start_time,
                ground_truth_graph=None,
                inferred_graph=None,
                optimization_gap=-1,
                actual_missingness_rate=-1,
                timestamp=pd.Timestamp.now().isoformat()
            )
    
    def run_all_experiments(self, parallel: bool = False) -> List[ExperimentalResult]:
        """Run all experiments in the factorial design."""
        logger.info(f"Starting {len(self.conditions)} experiments...")
        
        if parallel:
            # Parallel execution (requires additional setup)
            logger.warning("Parallel execution not yet implemented, falling back to sequential")
        
        # Sequential execution
        for i, condition in enumerate(self.conditions, 1):
            logger.info(f"Progress: {i}/{len(self.conditions)}")
            result = self.run_single_experiment(condition)
            self.results.append(result)
            
            # Save intermediate results
            if i % 10 == 0:
                self.save_results(f"intermediate_{i}")
        
        # Save final results
        self.save_results("final")
        logger.info("All experiments completed!")
        
        return self.results
    
    def save_results(self, suffix: str = "") -> None:
        """Save experimental results to disk."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"factorial_results_{suffix}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'condition_id': result.condition.condition_id,
                'dataset': result.condition.dataset.name,
                'missingness_mechanism': result.condition.missingness_mechanism.name,
                'missingness_rate': result.condition.missingness_rate,
                'algorithm': result.condition.algorithm_name,
                'replicate': result.condition.replicate_id,
                'metrics': result.metrics,
                'execution_time': result.execution_time,
                'optimization_gap': result.optimization_gap,
                'actual_missingness_rate': result.actual_missingness_rate,
                'timestamp': result.timestamp
            }
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze experimental results and create summary statistics.
        
        Returns:
            DataFrame with aggregated results by experimental condition
        """
        if not self.results:
            logger.warning("No results to analyze")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        records = []
        for result in self.results:
            record = {
                'dataset': result.condition.dataset.name,
                'mechanism': result.condition.missingness_mechanism.name,
                'rate': result.condition.missingness_rate,
                'algorithm': result.condition.algorithm_name,
                'replicate': result.condition.replicate_id,
                **result.metrics,
                'optimization_gap': result.optimization_gap,
                'actual_rate': result.actual_missingness_rate,
                'execution_time': result.execution_time
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Aggregate by condition (excluding replicate)
        grouped = df.groupby(['dataset', 'mechanism', 'rate', 'algorithm'])
        
        # Calculate summary statistics
        summary = grouped.agg({
            'f1_score': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'optimization_gap': 'mean',
            'actual_rate': 'mean',
            'execution_time': 'mean'
        }).round(3)
        
        return summary


# Example usage demonstrating the framework
if __name__ == "__main__":
    
    # Define datasets
    datasets = [
        DatasetConfig(
            name="Diabetes",
            path="data/Raw/Diabetes.csv",
            target_variable="Outcome",
            description="Diabetes prediction dataset",
            n_variables=9,
            n_samples=768
        ),
        DatasetConfig(
            name="Heart",
            path="data/Raw/Heart.csv",
            target_variable="target",
            description="Heart disease dataset",
            n_variables=14,
            n_samples=303
        ),
        DatasetConfig(
            name="Hepatitis",
            path="data/Raw/Hepatitis.csv",
            target_variable="Class",
            description="Hepatitis dataset",
            n_variables=20,
            n_samples=155
        )
    ]
    
    # Define missingness mechanisms
    mechanisms = [
        MissingnessConfig(
            name="Sigmoid",
            model_type="sigmoid",
            parameter_grid={
                'alpha': np.linspace(0.1, 1.0, 10),
                'beta': np.linspace(-10, 0, 10)
            },
            description="Sigmoid-based MNAR"
        ),
        MissingnessConfig(
            name="GPD",
            model_type="gpd",
            parameter_grid={
                'u': [75, 80, 85, 90],  # Will be converted to quantiles
                'xi': np.linspace(0.01, 0.5, 5),
                'sigma': np.linspace(0.5, 2.0, 5)
            },
            description="Heavy-tailed MNAR using GPD"
        ),
        MissingnessConfig(
            name="Threshold",
            model_type="threshold",
            parameter_grid={
                'threshold': np.linspace(0, 100, 50)  # Will be adjusted per dataset
            },
            description="Threshold-based censoring"
        )
    ]
    
    # Define missingness rates
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Placeholder algorithm (would be replaced with actual implementation)
    class DummyAlgorithm(CausalDiscoveryAlgorithm):
        def __init__(self, name: str):
            self.name = name
        
        def learn_structure(self, data: pd.DataFrame):
            # Placeholder - would call actual algorithm
            return f"Graph from {self.name}"
        
        def get_name(self) -> str:
            return self.name
    
    algorithms = [
        DummyAlgorithm("SM-MVPC"),
        # Could add more: DummyAlgorithm("PC"), DummyAlgorithm("GES"), etc.
    ]
    
    # Create and run factorial experiment
    framework = FactorialExperimentFramework(
        datasets=datasets,
        missingness_mechanisms=mechanisms,
        missingness_rates=rates,
        algorithms=algorithms,
        n_replicates=3,  # Reduced for demonstration
        random_seed=42
    )
    
    # Run experiments
    print(f"\n{'='*60}")
    print(f"FACTORIAL EXPERIMENTAL DESIGN")
    print(f"{'='*60}")
    print(f"Factors:")
    print(f"  - Datasets: {len(datasets)} levels")
    print(f"  - Mechanisms: {len(mechanisms)} levels")
    print(f"  - Rates: {len(rates)} levels")
    print(f"  - Algorithms: {len(algorithms)} levels")
    print(f"  - Replicates: {framework.n_replicates}")
    print(f"\nTotal experiments: {len(framework.conditions)}")
    print(f"{'='*60}\n")
    
    # Note: Actual execution would require real algorithm implementations
    # results = framework.run_all_experiments()
    # summary = framework.analyze_results()
    # print(summary)
