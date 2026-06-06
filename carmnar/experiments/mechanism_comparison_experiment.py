"""
Mechanism comparison: sigmoid vs threshold self-masking MNAR.
Measures how fast SM-MVPC skeleton and v-structure F1 degrade under each
mechanism on the three clinical datasets.

Deterministic; reproducible via
    python -m carmnar.experiments.mechanism_comparison_experiment
Writes results/mechanism_comparison/mechanism_aggregated.csv
"""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from carmnar.paths import DATA_DIR, RESULTS_DIR

from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery  # type: ignore
from carmnar.algorithms.mnar_generator import MNARGenerator  # type: ignore
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC  # type: ignore
from carmnar.utils.data_loader import MedicalDataLoader  # type: ignore

logging.basicConfig(level=logging.WARNING)


def main(n_rep: int = 20):
    out = RESULTS_DIR / "mechanism_comparison"
    out.mkdir(parents=True, exist_ok=True)
    loader = MedicalDataLoader(data_dir=str(DATA_DIR))
    pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=3)
    smmvpc = SMMVPC(alpha=0.05, max_conditioning_set_size=3,
                    missing_data_method="test_wise_deletion")
    datasets = ["diabetes", "heart_disease", "hepatitis"]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # MNARGenerator.introduce_mnar_effect_variable supports sigmoid + threshold;
    # GPD is specified/implemented separately (mathematically_specified_mnar.py)
    # and is analysed in the extended code, not in this clinical comparison.
    mechanisms = ["sigmoid", "threshold"]

    rows = []
    for ds in datasets:
        data = loader.preprocess_dataset(loader.load_dataset(ds), ds)
        eff = loader.get_effect_variables(ds)
        target = eff[0] if eff else data.columns[-1]
        true_graph = pc.fit(data)
        for mech in mechanisms:
            gen = MNARGenerator(random_state=42)
            for rate in rates:
                for rep in range(n_rep):
                    if rate == 0.0:
                        md = data.copy()
                    else:
                        try:
                            md = gen.introduce_mnar_effect_variable(
                                data, effect_variable=target,
                                target_percentage=rate, method=mech)
                        except Exception:
                            # mechanism may not support this dataset/rate; skip
                            continue
                    try:
                        est = smmvpc.fit(md)
                        m = evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True)
                        rows.append({"dataset": ds, "mechanism": mech, "rate": rate, "rep": rep,
                                     "skeleton_f1": m.get("skeleton_f1", np.nan),
                                     "vstructure_f1": m.get("vstructure_f1", np.nan)})
                    except Exception:
                        continue
    df = pd.DataFrame(rows)
    if df.empty:
        print("No mechanism results (mechanisms may be unsupported).")
        return
    df.to_csv(out / "mechanism_runs.csv", index=False)
    agg = (df.groupby(["dataset", "mechanism", "rate"])
             .agg(skeleton_f1=("skeleton_f1", "mean"),
                  vstructure_f1=("vstructure_f1", "mean")).reset_index())
    agg.to_csv(out / "mechanism_aggregated.csv", index=False)

    # Degradation per mechanism: (F1@0 - F1@50) averaged over datasets
    print("Skeleton-F1 degradation 0%->50% (mean over datasets):")
    for mech in mechanisms:
        drops = []
        for ds in datasets:
            s0 = agg[(agg.dataset == ds) & (agg.mechanism == mech) & (agg.rate == 0.0)]["skeleton_f1"]
            s5 = agg[(agg.dataset == ds) & (agg.mechanism == mech) & (agg.rate == 0.5)]["skeleton_f1"]
            if len(s0) and len(s5):
                drops.append(float(s0.iloc[0]) - float(s5.iloc[0]))
        if drops:
            print(f"  {mech:10} mean drop = {np.mean(drops):.4f}")
    return agg


if __name__ == "__main__":
    main()
