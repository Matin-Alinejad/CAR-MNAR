"""Minimal end-to-end example: inject sigmoid self-masking MNAR on a clinical
target, run SM-MVPC, and score the result with CPDAG-aware metrics.

Run with:  python -m examples.quickstart   (from the repository root)
"""
import logging

from carmnar.utils.data_loader import MedicalDataLoader
from carmnar.algorithms.mnar_generator import MNARGenerator
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC
from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery

logging.disable(logging.INFO)


def main() -> None:
    loader = MedicalDataLoader()
    data = loader.preprocess_dataset(loader.load_dataset("diabetes"), "diabetes")
    target = loader.get_effect_variables("diabetes")[0]

    true_graph = PCAlgorithm().fit(data)
    incomplete = MNARGenerator(random_state=42).introduce_mnar_effect_variable(
        data, effect_variable=target, target_percentage=0.30, method="sigmoid")

    est = SMMVPC(missing_data_method="test_wise_deletion").fit(incomplete)
    metrics = evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True)

    print(f"Target variable        : {target}")
    print(f"Realized missingness   : {incomplete[target].isna().mean():.3f}")
    print(f"Skeleton F1            : {metrics['skeleton_f1']:.3f}")
    print(f"V-structure F1         : {metrics['vstructure_f1']:.3f}")
    print(f"Orientation F1         : {metrics['orientation_f1']:.3f}")
    print(f"CPDAG-SHD              : {metrics['cpdag_shd']}")


if __name__ == "__main__":
    main()
