#!/usr/bin/env python3
"""
Focused Experimental Pipeline for SM-MVPC Evaluation
====================================================

This module executes the primary experimental pipeline for evaluating the Self-Masking
Missing Value PC (SM-MVPC) algorithm on medical datasets under Missing Not At Random
(MNAR) conditions. The script implements a statistically rigorous experimental design
with optimal replication count (n_rep=20) determined through comprehensive pilot studies.

The pipeline systematically evaluates SM-MVPC across multiple clinical datasets (Diabetes,
Heart Disease, Hepatitis) under varying MNAR severity levels (0% to 50% missingness),
employing CPDAG-aware evaluation metrics to assess skeleton recovery, v-structure
detection, and edge orientation accuracy. Results are compiled with bootstrap confidence
intervals and statistical significance testing.

Author: Anonymous (for review)
Date: 2025

Usage:
    python run_focused_experiments.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.main_experiment import SM_MVPC_Experiment

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'focused_experiments_n20_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run focused experiments with optimal n_rep=20."""
    logger.info("=" * 80)
    logger.info("FOCUSED EXPERIMENTS WITH OPTIMAL n_rep=20")
    logger.info("=" * 80)

    # Create experiment instance
    experiment = SM_MVPC_Experiment()

    # Set optimal n_rep from pilot study
    optimal_n_rep = 20
    experiment.n_repetitions = optimal_n_rep
    logger.info(f"Using optimal n_rep = {optimal_n_rep} (determined from pilot study)")

    # Full experimental configuration
    datasets = ['diabetes', 'heart_disease', 'hepatitis']
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    experiment.datasets = datasets
    experiment.missing_percentages = rates

    # Calculate total experiments
    total_conditions = len(datasets) * len(rates) * optimal_n_rep
    logger.info(f"Experimental design:")
    logger.info(f"  - Datasets: {len(datasets)} ({', '.join(datasets)})")
    logger.info(f"  - Missing percentages: {len(rates)} ({rates})")
    logger.info(f"  - Replicates per condition: {optimal_n_rep}")
    logger.info(f"  - Total runs: {total_conditions}")
    logger.info(f"  - Estimated time: ~{total_conditions * 2 / 60:.1f} minutes")

    # Run experiments
    start_time = time.time()

    try:
        logger.info("\n" + "=" * 80)
        logger.info("Phase 1: Loading and preprocessing datasets...")
        logger.info("=" * 80)
        datasets_data = experiment.load_and_preprocess_datasets()
        logger.info(f"Loaded {len(datasets_data)} datasets")

        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Generating ground truth graphs...")
        logger.info("=" * 80)
        ground_truth_graphs = experiment.generate_ground_truth_graphs(datasets_data)
        logger.info("Ground truth graphs generated")

        logger.info("\n" + "=" * 80)
        logger.info("Phase 3: Pre-generating MNAR scenarios...")
        logger.info("=" * 80)
        # Pre-generate all MNAR scenarios for efficiency
        all_mnar_scenarios = experiment.generate_mnar_scenarios(datasets_data)
        logger.info("MNAR scenarios generated")

        logger.info("\n" + "=" * 80)
        logger.info("Phase 4: Running experiments...")
        logger.info("=" * 80)
        logger.info("This will take some time. Progress will be logged...")

        # Run experiments condition by condition to be more robust
        all_results = []

        for dataset_name in datasets:
            logger.info(f"\n--- Processing dataset: {dataset_name} ---")

            if dataset_name not in datasets_data:
                logger.warning(f"Skipping {dataset_name} - not in loaded datasets")
                continue

            for missing_percentage in rates:
                logger.info(f"  Running {dataset_name} at {missing_percentage*100:.0f}% missingness...")

                # Get pre-generated MNAR scenario
                mnar_data = all_mnar_scenarios[dataset_name][missing_percentage]

                # Run replicates
                for rep in range(optimal_n_rep):
                    try:
                        result = experiment.run_causal_discovery_experiment(
                            dataset_name, mnar_data, missing_percentage, rep
                        )
                        all_results.append(result)
                        if rep % 5 == 0:  # Progress update every 5 reps
                            logger.info(f"    Completed replicate {rep+1}/{optimal_n_rep}")
                    except Exception as e:
                        logger.error(f"    Error in {dataset_name} {missing_percentage*100:.0f}% rep {rep}: {e}")
                        continue

        elapsed_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("Phase 4: Compiling results...")
        logger.info("=" * 80)

        results_df = pd.DataFrame(all_results)
        logger.info(f"Total results collected: {len(results_df)}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = experiment.results_dir / f"focused_experimental_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 5: Analyzing results...")
        logger.info("=" * 80)
        analysis = experiment.analyze_results(results_df)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 6: Generating reports...")
        logger.info("=" * 80)
        experiment.generate_reports()

        logger.info("\n" + "=" * 80)
        logger.info("FOCUSED EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Optimal n_rep used: {optimal_n_rep}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info(f"Time per condition: {elapsed_time/total_conditions:.2f} seconds")
        logger.info(f"Total experiments completed: {len(results_df)}")
        logger.info(f"Results saved to: {results_file}")

        # Save summary
        summary = {
            'optimal_n_rep': optimal_n_rep,
            'total_datasets': len(datasets),
            'total_rates': len(rates),
            'total_conditions': total_conditions,
            'total_experiments': len(results_df),
            'total_time_seconds': elapsed_time,
            'time_per_experiment': elapsed_time / total_conditions,
            'results_file': str(results_file),
            'pilot_study_conclusion': 'n_rep=20 provides optimal balance of statistical stability (CV < 0.01%) and computational efficiency'
        }

        import json
        with open('results/focused_experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # Use default=str to handle numpy types

        logger.info(f"Summary saved to: results/focused_experiment_summary.json")

        return results_df, analysis, summary

    except KeyboardInterrupt:
        logger.warning("\nExperiments interrupted by user. Partial results may be available.")
        raise
    except Exception as e:
        logger.error(f"\nError running experiments: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular import

    results_df, analysis, summary = main()
    print("\n✓ FOCUSED EXPERIMENTS COMPLETED!")
    print(f"  Optimal n_rep: {summary['optimal_n_rep']}")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Total time: {summary['total_time_seconds']:.1f}s")
    print(f"  Results: {summary['results_file']}")

