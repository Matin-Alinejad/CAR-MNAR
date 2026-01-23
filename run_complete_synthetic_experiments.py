#!/usr/bin/env python3
"""
Complete Synthetic Data Experimental Pipeline
==============================================

This module orchestrates comprehensive synthetic data experiments for evaluating
causal discovery algorithms under controlled conditions with known ground truth
DAGs. The pipeline integrates synthetic causal network generation, mathematically
specified MNAR missingness mechanisms, and rigorous causal discovery evaluation
within a factorial experimental design.

The framework generates diverse synthetic datasets across multiple topologies
(chain, fork, collider, random, scale-free, small-world) and complexity levels,
enabling systematic assessment of algorithmic robustness under varying structural
characteristics and missingness patterns. Results provide critical validation of
framework correctness and enable separation of algorithmic behavior from dataset
idiosyncrasies.

Author: Anonymous (for review)
Date: 2025

Usage:
    python run_complete_synthetic_experiments.py [--pilot] [--full] [--output-dir DIR]
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.data_generation.synthetic_dataset_factory import (
    SyntheticDatasetFactory,
    load_synthetic_dataset
)
from src.algorithms.factorial_experiment_framework import (
    MissingnessConfig
)
from run_synthetic_experiments import (
    SyntheticCausalDiscoveryAlgorithm,
    SyntheticExperimentFramework,
    create_missingness_configs
)


def run_pilot_experiments(output_dir: str = "results/synthetic_pilot"):
    """Run pilot synthetic data experiments (small scale)."""
    print("="*70)
    print("SYNTHETIC DATA PILOT EXPERIMENTS")
    print("="*70)
    
    # Create factory
    factory = SyntheticDatasetFactory(random_seed=42)
    
    # Generate focused design datasets (smaller set for pilot)
    print("\n1. Generating synthetic datasets...")
    datasets = factory.create_focused_design_datasets(
        output_dir=f"{output_dir}/datasets"
    )
    # Use subset for pilot
    pilot_datasets = datasets[:3]
    print(f"   Generated {len(datasets)} datasets, using {len(pilot_datasets)} for pilot")
    
    # Create missingness mechanisms
    print("\n2. Setting up missingness mechanisms...")
    mechanisms = create_missingness_configs()
    pilot_mechanisms = mechanisms[:2]  # Use first 2 for pilot
    print(f"   Configured {len(mechanisms)} mechanisms, using {len(pilot_mechanisms)} for pilot")
    
    # Missingness rates
    rates = [0.0, 0.1, 0.2, 0.3]
    
    # Create algorithms
    print("\n3. Setting up algorithms...")
    algorithms = [
        SyntheticCausalDiscoveryAlgorithm("SM-MVPC")
    ]
    
    # Create experiment framework
    print("\n4. Running experiments...")
    framework = SyntheticExperimentFramework(
        factory=factory,
        output_dir=output_dir,
        random_seed=42
    )
    
    # Run experiments
    n_replicates = 3
    total_experiments = len(pilot_datasets) * len(pilot_mechanisms) * len(rates) * n_replicates
    print(f"   Total experiments: {total_experiments}")
    
    results = []
    exp_count = 0
    
    for dataset in pilot_datasets:
        for mechanism in pilot_mechanisms:
            for rate in rates:
                for rep in range(n_replicates):
                    exp_count += 1
                    if exp_count % 10 == 0:
                        print(f"   Progress: {exp_count}/{total_experiments}")
                    
                    result = framework.run_experiment(
                        dataset, mechanism, rate, algorithms[0], rep
                    )
                    results.append(result)
    
    # Save results
    print("\n5. Saving results...")
    results_file = Path(output_dir) / "pilot_results.json"
    results_data = [
        {
            'condition_id': r.condition.condition_id,
            'dataset': r.condition.dataset.name,
            'mechanism': r.condition.missingness_mechanism.name,
            'rate': r.condition.missingness_rate,
            'algorithm': r.condition.algorithm_name,
            'replicate': r.condition.replicate_id,
            'metrics': r.metrics,
            'optimization_gap': r.optimization_gap,
            'actual_rate': r.actual_missingness_rate,
            'execution_time': r.execution_time,
            'timestamp': r.timestamp
        }
        for r in results
    ]
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"   Results saved to {results_file}")
    
    # Analyze results
    print("\n6. Analyzing results...")
    if results:
        df = pd.DataFrame(results_data)
        valid_results = df[~df['metrics'].apply(lambda x: 'error' in x if isinstance(x, dict) else False)]
        
        if len(valid_results) > 0:
            # Extract metrics
            metrics_data = []
            for _, row in valid_results.iterrows():
                metrics = row['metrics']
                if isinstance(metrics, dict):
                    metrics_data.append({
                        'dataset': row['dataset'],
                        'mechanism': row['mechanism'],
                        'rate': row['rate'],
                        'f1_score': metrics.get('f1_score', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'shd': metrics.get('shd', float('inf')),
                        'optimization_gap': row['optimization_gap'],
                        'actual_rate': row['actual_rate']
                    })
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                summary = df_metrics.groupby(['dataset', 'mechanism', 'rate']).agg({
                    'f1_score': ['mean', 'std'],
                    'precision': 'mean',
                    'recall': 'mean',
                    'optimization_gap': 'mean',
                    'actual_rate': 'mean'
                }).round(3)
                
                print("\nResults Summary:")
                print(summary)
                
                # Save summary
                summary_file = Path(output_dir) / "pilot_summary.csv"
                summary.to_csv(summary_file)
                print(f"\n   Summary saved to {summary_file}")
    
    print("\n" + "="*70)
    print("Pilot experiments completed successfully!")
    print("="*70)
    
    return results


def run_full_experiments(output_dir: str = "results/synthetic_full"):
    """Run full synthetic data experiments (complete factorial design)."""
    print("="*70)
    print("SYNTHETIC DATA FULL EXPERIMENTS")
    print("="*70)
    print("WARNING: This will run a large number of experiments.")
    print("Estimated time: Several hours depending on system performance.")
    print("="*70)
    
    response = input("\nDo you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Create factory
    factory = SyntheticDatasetFactory(random_seed=42)
    
    # Generate full design datasets
    print("\n1. Generating synthetic datasets...")
    datasets = factory.create_focused_design_datasets(
        output_dir=f"{output_dir}/datasets"
    )
    print(f"   Generated {len(datasets)} datasets")
    
    # Create all missingness mechanisms
    print("\n2. Setting up missingness mechanisms...")
    mechanisms = create_missingness_configs()
    print(f"   Configured {len(mechanisms)} mechanisms")
    
    # All missingness rates
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # Create algorithms
    print("\n3. Setting up algorithms...")
    algorithms = [
        SyntheticCausalDiscoveryAlgorithm("SM-MVPC")
    ]
    
    # Create experiment framework
    print("\n4. Running experiments...")
    framework = SyntheticExperimentFramework(
        factory=factory,
        output_dir=output_dir,
        random_seed=42
    )
    
    # Run experiments
    n_replicates = 20
    total_experiments = len(datasets) * len(mechanisms) * len(rates) * n_replicates
    print(f"   Total experiments: {total_experiments}")
    print(f"   Estimated time: {total_experiments * 2 / 60:.1f} minutes (assuming 2 sec/experiment)")
    
    results = []
    exp_count = 0
    
    for dataset in datasets:
        for mechanism in mechanisms:
            for rate in rates:
                for rep in range(n_replicates):
                    exp_count += 1
                    if exp_count % 50 == 0:
                        print(f"   Progress: {exp_count}/{total_experiments} ({exp_count/total_experiments*100:.1f}%)")
                    
                    result = framework.run_experiment(
                        dataset, mechanism, rate, algorithms[0], rep
                    )
                    results.append(result)
                    
                    # Save intermediate results every 100 experiments
                    if exp_count % 100 == 0:
                        intermediate_file = Path(output_dir) / f"intermediate_results_{exp_count}.json"
                        with open(intermediate_file, 'w') as f:
                            json.dump([
                                {
                                    'condition_id': r.condition.condition_id,
                                    'metrics': r.metrics,
                                    'optimization_gap': r.optimization_gap,
                                    'actual_rate': r.actual_missingness_rate
                                }
                                for r in results
                            ], f, indent=2, default=str)
    
    # Save final results
    print("\n5. Saving final results...")
    results_file = Path(output_dir) / "full_results.json"
    results_data = [
        {
            'condition_id': r.condition.condition_id,
            'dataset': r.condition.dataset.name,
            'mechanism': r.condition.missingness_mechanism.name,
            'rate': r.condition.missingness_rate,
            'algorithm': r.condition.algorithm_name,
            'replicate': r.condition.replicate_id,
            'metrics': r.metrics,
            'optimization_gap': r.optimization_gap,
            'actual_rate': r.actual_missingness_rate,
            'execution_time': r.execution_time,
            'timestamp': r.timestamp
        }
        for r in results
    ]
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"   Results saved to {results_file}")
    
    # Generate comprehensive analysis
    print("\n6. Generating comprehensive analysis...")
    # Analysis code would go here
    
    print("\n" + "="*70)
    print("Full experiments completed successfully!")
    print("="*70)
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run synthetic data experiments for causal discovery evaluation"
    )
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Run pilot experiments (small scale, fast)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full experiments (complete factorial design)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/synthetic_experiments',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    if args.pilot:
        run_pilot_experiments(args.output_dir)
    elif args.full:
        run_full_experiments(args.output_dir)
    else:
        # Default to pilot
        print("No mode specified. Running pilot experiments by default.")
        print("Use --pilot for pilot experiments or --full for complete experiments.")
        run_pilot_experiments(args.output_dir)


if __name__ == "__main__":
    main()


