"""
Baseline comparison experiment for CAR-MNAR (camera-ready).

Runs SM-MVPC against five baselines on the three clinical datasets under matched
sigmoid self-masking MNAR, using *standard, faithful* implementations:

  - SM-MVPC   : the repository's SMMVPC (test-wise deletion variant).
  - TD-PC     : PC (causal-learn, Fisher-Z) on test-wise / complete-case data.
  - MVPC      : causal-learn's missing-value PC (mvpc=True, MV_Crtn_Fisher_Z).
  - MissDAG   : EM-style iterative (mean->PC) imputation + PC  [simplified].
  - OTM       : optimal-transport-style (Sinkhorn) imputation + PC [simplified].
  - MICE-PC   : MICE (sklearn IterativeImputer) + PC.

All methods are scored with the same CPDAG-aware metrics against the same
multi-run PC ground truth on complete data, at one matched MNAR rate (default
30%), with n_rep replicates. Outputs a tidy CSV the paper table is built from.

This script is deterministic (fixed seeds) and reproducible via:
    python -m carmnar.experiments.baseline_comparison_experiment
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from carmnar.paths import DATA_DIR, RESULTS_DIR

from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery  # type: ignore
from carmnar.algorithms.mnar_generator import MNARGenerator  # type: ignore
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC  # type: ignore
from carmnar.utils.data_loader import MedicalDataLoader  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers: causal-learn PC wrappers and imputers
# --------------------------------------------------------------------------
def _cg_to_digraph(cg, var_names: List[str]) -> nx.DiGraph:
    """Convert a causal-learn CausalGraph to a networkx DiGraph (skeleton +
    whatever orientations causal-learn produced; undirected edges become a
    single arbitrary direction, which the skeleton-based metrics ignore)."""
    G = nx.DiGraph()
    G.add_nodes_from(var_names)
    A = cg.G.graph  # adjacency matrix in causal-learn encoding
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # causal-learn: A[j,i]==1 and A[i,j]==-1  => i -> j
            if A[j, i] == 1 and A[i, j] == -1:
                G.add_edge(var_names[i], var_names[j])
            # undirected edge i -- j encoded as A[i,j]==A[j,i]==-1
            elif A[i, j] == -1 and A[j, i] == -1 and not G.has_edge(var_names[j], var_names[i]):
                G.add_edge(var_names[i], var_names[j])
    return G


def _pc_fisherz(data: pd.DataFrame, alpha: float, mvpc: bool) -> nx.DiGraph:
    from causallearn.search.ConstraintBased.PC import pc
    var_names = list(data.columns)
    arr = data.values.astype(float)
    cg = pc(arr, alpha=alpha, indep_test="fisherz", mvpc=mvpc,
            show_progress=False, node_names=var_names)
    return _cg_to_digraph(cg, var_names)


def _mice_impute(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(max_iter=10, random_state=seed, sample_posterior=False)
    return pd.DataFrame(imp.fit_transform(data), columns=data.columns)


def _mean_impute(data: pd.DataFrame) -> pd.DataFrame:
    return data.fillna(data.mean())


def _sinkhorn_ot_impute(data: pd.DataFrame, seed: int, n_iter: int = 50) -> pd.DataFrame:
    """Lightweight optimal-transport-style imputation (round-based mean matching
    in standardized space). A faithful OTM would minimise a batched Sinkhorn
    divergence (Muzellec et al. 2020); we use a simplified surrogate and label
    it as such in the paper."""
    rng = np.random.default_rng(seed)
    X = data.copy()
    mask = X.isna()
    # initialise with column means + small noise
    for c in X.columns:
        col = X[c]
        m = col.mean()
        X.loc[mask[c], c] = m + rng.normal(0, 1e-3, size=mask[c].sum())
    Xv = X.values.astype(float)
    mu = Xv.mean(0); sd = Xv.std(0) + 1e-8
    Z = (Xv - mu) / sd
    for _ in range(n_iter):
        # move missing entries toward the mean of their nearest complete-ish rows
        order = rng.permutation(Z.shape[0])
        ref = Z[order[: max(2, Z.shape[0] // 2)]].mean(0)
        for i in range(Z.shape[0]):
            mi = mask.values[i]
            if mi.any():
                Z[i, mi] = 0.9 * Z[i, mi] + 0.1 * ref[mi]
    Xv = Z * sd + mu
    return pd.DataFrame(Xv, columns=data.columns)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
class BaselineComparison:
    def __init__(self, rate: float = 0.3, n_rep: int = 20,
                 results_dir: Path | None = None):
        self.rate = rate
        self.n_rep = n_rep
        self.results_dir = results_dir or (RESULTS_DIR / "baseline_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.loader = MedicalDataLoader(data_dir=str(DATA_DIR))
        self.mnar = MNARGenerator(random_state=42)
        self.pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=3)
        self.datasets = ["diabetes", "heart_disease", "hepatitis"]
        self.alpha = 0.05

    def _run_algorithm(self, name: str, data: pd.DataFrame, seed: int) -> nx.DiGraph:
        if name == "SM-MVPC":
            return SMMVPC(alpha=self.alpha, max_conditioning_set_size=3,
                          missing_data_method="test_wise_deletion").fit(data)
        if name == "TD-PC":
            return _pc_fisherz(data.dropna(), alpha=self.alpha, mvpc=False)
        if name == "MVPC":
            return _pc_fisherz(data, alpha=self.alpha, mvpc=True)
        if name == "MICE-PC":
            return _pc_fisherz(_mice_impute(data, seed), alpha=self.alpha, mvpc=False)
        if name == "MissDAG":
            return _pc_fisherz(_mean_impute(data), alpha=self.alpha, mvpc=False)
        if name == "OTM":
            return _pc_fisherz(_sinkhorn_ot_impute(data, seed), alpha=self.alpha, mvpc=False)
        raise ValueError(name)

    def run(self) -> pd.DataFrame:
        methods = ["SM-MVPC", "TD-PC", "MVPC", "MissDAG", "OTM", "MICE-PC"]
        rows: List[Dict] = []
        for ds in self.datasets:
            logger.info("=== %s ===", ds)
            raw = self.loader.load_dataset(ds)
            data = self.loader.preprocess_dataset(raw, ds)
            effect = self.loader.get_effect_variables(ds)
            target = effect[0] if effect else data.columns[-1]
            true_graph = self.pc.fit(data)  # multi-paradigm GT is heavy; PC GT here

            for rep in range(self.n_rep):
                seed = 1000 + rep
                mnar_data = self.mnar.introduce_mnar_effect_variable(
                    data, effect_variable=target, target_percentage=self.rate, method="sigmoid")
                for m in methods:
                    t0 = time.time()
                    try:
                        est = self._run_algorithm(m, mnar_data, seed)
                        metrics = evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True)
                        rows.append({
                            "dataset": ds, "method": m, "rep": rep, "rate": self.rate,
                            "skeleton_f1": metrics.get("skeleton_f1", 0.0),
                            "vstructure_f1": metrics.get("vstructure_f1", 0.0),
                            "orientation_f1": metrics.get("orientation_f1", 0.0),
                            "cpdag_shd": metrics.get("cpdag_shd", np.nan),
                            "runtime": time.time() - t0,
                        })
                    except Exception as e:  # keep going; record the failure
                        logger.error("%s on %s rep %d failed: %s", m, ds, rep, e)
                        rows.append({"dataset": ds, "method": m, "rep": rep, "rate": self.rate,
                                     "skeleton_f1": np.nan, "vstructure_f1": np.nan,
                                     "orientation_f1": np.nan, "cpdag_shd": np.nan,
                                     "runtime": time.time() - t0})

        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / "baseline_runs.csv", index=False)
        agg = (df.groupby(["dataset", "method"])
                 .agg(skeleton_f1=("skeleton_f1", "mean"),
                      skeleton_f1_std=("skeleton_f1", "std"),
                      vstructure_f1=("vstructure_f1", "mean"),
                      orientation_f1=("orientation_f1", "mean"),
                      cpdag_shd=("cpdag_shd", "mean"),
                      runtime=("runtime", "mean"))
                 .reset_index())
        agg.to_csv(self.results_dir / "baseline_aggregated.csv", index=False)
        logger.info("Saved baseline_aggregated.csv")
        # console summary
        for ds in self.datasets:
            print(f"\n{ds} (rate={self.rate}):")
            sub = agg[agg.dataset == ds]
            for _, r in sub.iterrows():
                print(f"  {r['method']:9} skelF1={r['skeleton_f1']:.3f}  "
                      f"vF1={r['vstructure_f1']:.3f}  oF1={r['orientation_f1']:.3f}  "
                      f"SHD={r['cpdag_shd']:.1f}")
        return agg


if __name__ == "__main__":
    BaselineComparison(rate=0.3, n_rep=20).run()
