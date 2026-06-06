"""Smoke tests: the package imports, data loads, and the core pipeline runs."""
import logging

import networkx as nx
import pytest

logging.disable(logging.INFO)


def test_imports():
    import carmnar  # noqa: F401
    from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery  # noqa: F401
    from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC  # noqa: F401
    from carmnar.algorithms.mnar_generator import MNARGenerator  # noqa: F401
    from carmnar.utils.data_loader import MedicalDataLoader  # noqa: F401


@pytest.mark.parametrize("dataset", ["diabetes", "heart_disease", "hepatitis"])
def test_data_loads(dataset):
    from carmnar.utils.data_loader import MedicalDataLoader
    loader = MedicalDataLoader()
    data = loader.preprocess_dataset(loader.load_dataset(dataset), dataset)
    assert len(data) > 0 and data.shape[1] > 1
    # the spurious index column must have been dropped
    assert "no" not in [c.lower() for c in data.columns]


def test_cpdag_metrics_self_consistency():
    from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery
    g = nx.DiGraph([("A", "B"), ("B", "C")])
    m = evaluate_causal_discovery(g, g, convert_to_cpdag=True)
    assert m["skeleton_f1"] == pytest.approx(1.0)


def test_pipeline_runs():
    from carmnar.utils.data_loader import MedicalDataLoader
    from carmnar.algorithms.mnar_generator import MNARGenerator
    from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC
    from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery

    loader = MedicalDataLoader()
    data = loader.preprocess_dataset(loader.load_dataset("diabetes"), "diabetes")
    target = loader.get_effect_variables("diabetes")[0]
    true_graph = PCAlgorithm().fit(data)
    incomplete = MNARGenerator(random_state=42).introduce_mnar_effect_variable(
        data, effect_variable=target, target_percentage=0.30, method="sigmoid")
    est = SMMVPC(missing_data_method="test_wise_deletion").fit(incomplete)
    m = evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True)
    assert 0.0 <= m["skeleton_f1"] <= 1.0
