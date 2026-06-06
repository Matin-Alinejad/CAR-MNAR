"""
Comprehensive CPDAG-aware metrics for SM-MVPC under sigmoid self-masking MNAR
(camera-ready Table 3). For each clinical dataset and missing rate we report the
full-graph skeleton F1, v-structure F1, orientation F1 (compelled edges),
CPDAG-SHD and the deterministic SID proxy, scored against the single-run PC
ground truth on complete data (the same fixed, method-independent reference used
by the baseline-comparison experiment).

Deterministic (fixed seeds); reproducible via
    python -m carmnar.experiments.comprehensive_metrics_experiment
Writes results/comprehensive_metrics_repro/comprehensive_metrics.csv
"""

import sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from carmnar.paths import DATA_DIR, RESULTS_DIR

from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery  # type: ignore
from carmnar.algorithms.mnar_generator import MNARGenerator  # type: ignore
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC  # type: ignore
from carmnar.utils.data_loader import MedicalDataLoader  # type: ignore

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def main(n_rep: int = 20):
    out = RESULTS_DIR / "comprehensive_metrics_repro"
    out.mkdir(parents=True, exist_ok=True)
    loader = MedicalDataLoader(data_dir=str(DATA_DIR))
    mnar = MNARGenerator(random_state=42)
    pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=3)
    smmvpc = SMMVPC(alpha=0.05, max_conditioning_set_size=3,
                    missing_data_method="test_wise_deletion")
    datasets = ["diabetes", "heart_disease", "hepatitis"]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    rows = []
    for ds in datasets:
        data = loader.preprocess_dataset(loader.load_dataset(ds), ds)
        eff = loader.get_effect_variables(ds)
        target = eff[0] if eff else data.columns[-1]
        true_graph = pc.fit(data)
        for rate in rates:
            for rep in range(n_rep):
                if rate == 0.0:
                    md = data.copy()
                else:
                    md = mnar.introduce_mnar_effect_variable(
                        data, effect_variable=target, target_percentage=rate, method="sigmoid")
                est = smmvpc.fit(md)
                m = evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True)
                rows.append({"dataset": ds, "rate": rate, "rep": rep,
                             "skeleton_f1": m.get("skeleton_f1", 0.0),
                             "vstructure_f1": m.get("vstructure_f1", 0.0),
                             "orientation_f1": m.get("orientation_f1", 0.0),
                             "cpdag_shd": m.get("cpdag_shd", np.nan),
                             "sid": m.get("sid", np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "comprehensive_runs.csv", index=False)
    agg = (df.groupby(["dataset", "rate"])
             .agg(skeleton_f1=("skeleton_f1", "mean"),
                  vstructure_f1=("vstructure_f1", "mean"),
                  orientation_f1=("orientation_f1", "mean"),
                  cpdag_shd=("cpdag_shd", "mean"),
                  sid=("sid", "mean")).reset_index())
    agg.to_csv(out / "comprehensive_metrics.csv", index=False)
    for ds in datasets:
        print(f"\n{ds}")
        for _, r in agg[agg.dataset == ds].iterrows():
            print(f"  {r['rate']:.1f}  skel={r['skeleton_f1']:.3f}  v={r['vstructure_f1']:.3f}  "
                  f"o={r['orientation_f1']:.3f}  SHD={r['cpdag_shd']:.1f}  SID={r['sid']:.1f}")
    return agg


if __name__ == "__main__":
    main()
