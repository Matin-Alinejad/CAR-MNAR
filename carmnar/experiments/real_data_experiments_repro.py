"""
Real-data experiments for SM-MVPC on clinical datasets (Diabetes, Heart, Hepatitis)
under MNAR conditions, using the CAR-MNAR framework.

This script:
- Loads and preprocesses medical datasets via MedicalDataLoader
- Generates sigmoid self-masking MNAR on the clinical target variable
- Runs PC + imputation, SM-MVPC (test-wise deletion), SM-MVPC + imputation
- Evaluates global and local (around target) skeleton F1 vs. a PC-based ground-truth graph
- Writes per-run and aggregated CSVs under results/real_experiments_repro/
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx

# Add repo root so src.* and Workshop scripts are importable
# This script lives at <repo>/carmnar/experiments/, so the package root is 3 levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from carmnar.paths import DATA_DIR, RESULTS_DIR

from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery  # type: ignore
from carmnar.algorithms.mnar_generator import MNARGenerator  # type: ignore
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC  # type: ignore
from carmnar.utils.data_loader import MedicalDataLoader  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def compute_local_skeleton_f1(true_graph: nx.DiGraph, est_graph: nx.DiGraph, target: str) -> float:
    """
    Local skeleton F1 for the adjacency neighborhood of target (undirected).
    """
    # True adjacencies (parents + children)
    true_adj = set(true_graph.predecessors(target)) | set(true_graph.successors(target))
    if est_graph.has_node(target):
        est_adj = set(est_graph.predecessors(target)) | set(est_graph.successors(target))
    else:
        est_adj = set()

    tp = len(true_adj & est_adj)
    fp = len(est_adj - true_adj)
    fn = len(true_adj - est_adj)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1

class RealDataExperiment:
    """
    Run real-data experiments on Diabetes, Heart, Hepatitis with MNAR.
    """

    def __init__(self, results_dir: Path | None = None):
        if results_dir is None:
            results_dir = RESULTS_DIR / "real_experiments_repro"
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Use the root-level data/Raw directory where the clinical CSVs live
        self.data_loader = MedicalDataLoader(data_dir=str(DATA_DIR))
        self.mnar_generator = MNARGenerator(random_state=42)
        self.pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=3)
        self.sm_mvpc_td = SMMVPC(alpha=0.05, max_conditioning_set_size=3, missing_data_method="test_wise_deletion")
        self.sm_mvpc_imp = SMMVPC(alpha=0.05, max_conditioning_set_size=3, missing_data_method="imputation")

        self.datasets: List[str] = ["diabetes", "heart_disease", "hepatitis"]
        # Evaluate a grid of MNAR rates (including 0.0 for complete data)
        self.missing_percentages: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.n_repetitions: int = 5

    def run(self) -> None:
        all_results: List[Dict] = []

        for dataset_name in self.datasets:
            logger.info(f"=== Dataset: {dataset_name} ===")
            try:
                raw = self.data_loader.load_dataset(dataset_name)
                data = self.data_loader.preprocess_dataset(raw, dataset_name)
                effect_vars = self.data_loader.get_effect_variables(dataset_name)
                if not effect_vars:
                    logger.warning(f"No effect/target variables found for {dataset_name}; skipping.")
                    continue
                target = effect_vars[0]
                if target not in data.columns:
                    logger.warning(f"Target {target} not in preprocessed columns for {dataset_name}; skipping.")
                    continue
            except Exception as e:
                logger.error(f"Failed to load/preprocess {dataset_name}: {e}")
                continue

            # Ground-truth graph from PC on complete data
            try:
                true_graph = self.pc.fit(data)
            except Exception as e:
                logger.error(f"Failed to generate PC ground truth for {dataset_name}: {e}")
                true_graph = nx.DiGraph()

            # Save ground-truth graph for reproducibility (PC on complete data)
            try:
                gt_path = self.results_dir / f"{dataset_name}_pc_ground_truth.gml"
                nx.write_gml(true_graph, gt_path)
                # Also save adjacency matrix as CSV (skeleton)
                adj = nx.to_pandas_adjacency(true_graph, dtype=int)
                adj.to_csv(self.results_dir / f"{dataset_name}_pc_adjacency.csv")
            except Exception as e:
                logger.warning(f"Could not save ground-truth graph for {dataset_name}: {e}")

            # Generate and save one MNAR scenario per rate for reproducibility
            try:
                scenarios = self.mnar_generator.generate_mnar_scenarios(
                    data, [target], self.missing_percentages, method="sigmoid"
                )
                for rate, scen_df in scenarios.items():
                    rate_pct = int(round(rate * 100))
                    scen_path = self.results_dir / f"{dataset_name}_mnar_{rate_pct}.csv"
                    scen_df.to_csv(scen_path, index=False)
            except Exception as e:
                logger.warning(f"Could not generate/save MNAR scenarios for {dataset_name}: {e}")

            for missing_rate in self.missing_percentages:
                for rep in range(self.n_repetitions):
                    logger.info(f"{dataset_name} - missing={missing_rate:.1f}, rep={rep}")
                    if missing_rate == 0.0:
                        mnar_data = data.copy()
                    else:
                        # For repeated runs, re-sample MNAR masks to reflect variability
                        try:
                            mnar_data = self.mnar_generator.introduce_mnar_effect_variable(
                                data, effect_variable=target, target_percentage=missing_rate, method="sigmoid"
                            )
                        except Exception as e:
                            logger.error(f"MNAR generation failed for {dataset_name} {missing_rate:.1f}: {e}")
                            continue

                    # Run algorithms
                    alg_results = []
                    for alg_name in ["pc_imputation", "sm_mvpc_td", "sm_mvpc_imp"]:
                        start = time.time()
                        try:
                            if alg_name == "pc_imputation":
                                # Simple imputation, then PC
                                imp_mvpc = SMMVPC(alpha=0.05, max_conditioning_set_size=3, missing_data_method="imputation")
                                imputed = imp_mvpc.handle_missing_data(mnar_data)
                                est_graph = self.pc.fit(imputed)
                            elif alg_name == "sm_mvpc_td":
                                est_graph = self.sm_mvpc_td.fit(mnar_data)
                            elif alg_name == "sm_mvpc_imp":
                                est_graph = self.sm_mvpc_imp.fit(mnar_data)
                            else:
                                continue
                            runtime = time.time() - start
                        except Exception as e:
                            logger.error(f"{alg_name} failed on {dataset_name} {missing_rate:.1f} rep {rep}: {e}")
                            continue

                        # Global metrics
                        try:
                            global_metrics = evaluate_causal_discovery(true_graph, est_graph, convert_to_cpdag=False)
                            g_f1 = global_metrics.get("skeleton_f1", 0.0)
                            g_shd = global_metrics.get("skeleton_shd", 0)
                            n_nodes = global_metrics.get("n_nodes", true_graph.number_of_nodes())
                            n_edges_true = global_metrics.get("n_edges_true", true_graph.number_of_edges())
                            n_edges_inferred = global_metrics.get("n_edges_inferred", est_graph.number_of_edges())
                        except Exception as e:
                            logger.error(f"Global metric evaluation failed for {alg_name} on {dataset_name}: {e}")
                            g_f1 = 0.0
                            g_shd = 0
                            n_nodes = true_graph.number_of_nodes()
                            n_edges_true = true_graph.number_of_edges()
                            n_edges_inferred = est_graph.number_of_edges()

                        # Local metrics (single target variable)
                        try:
                            l_f1 = compute_local_skeleton_f1(true_graph, est_graph, target)
                        except Exception as e:
                            logger.error(f"Local metric evaluation failed for {alg_name} on {dataset_name}: {e}")
                            l_f1 = 0.0

                        res = {
                            "dataset": dataset_name,
                            "missing_rate": missing_rate,
                            "repetition": rep,
                            "algorithm": alg_name,
                            "target": target,
                            "global_skeleton_f1": g_f1,
                            "global_skeleton_shd": g_shd,
                            "local_skeleton_f1": l_f1,
                            "runtime": runtime,
                            "n_nodes": n_nodes,
                            "n_edges_true": n_edges_true,
                            "n_edges_inferred": n_edges_inferred,
                        }
                        all_results.append(res)

        if not all_results:
            logger.warning("No real-data results were generated.")
            return

        df = pd.DataFrame(all_results)
        out_csv = self.results_dir / "real_results.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Saved per-run results to %s", out_csv)

        # Aggregated means/std per dataset, missing_rate, algorithm
        agg = (
            df.groupby(["dataset", "missing_rate", "algorithm"])
            .agg(
                global_mean=("global_skeleton_f1", "mean"),
                global_std=("global_skeleton_f1", "std"),
                local_mean=("local_skeleton_f1", "mean"),
                local_std=("local_skeleton_f1", "std"),
                runtime_mean=("runtime", "mean"),
            )
            .reset_index()
        )
        # Fill NaN std (e.g. when a group has only one repetition) with 0 for reproducibility
        for col in ["global_std", "local_std"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0.0)
        agg_csv = self.results_dir / "real_aggregated.csv"
        agg.to_csv(agg_csv, index=False)
        logger.info("Saved aggregated results to %s", agg_csv)

if __name__ == "__main__":
    exp = RealDataExperiment()
    exp.run()

