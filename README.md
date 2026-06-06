# CAR-MNAR

**A unified factorial framework for evaluating causal-discovery robustness under self-masking Missing-Not-At-Random (MNAR) data.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Formally verified](https://img.shields.io/badge/Theorem%201-Lean%204%20verified-success.svg)](formal_verification/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20572905.svg)](https://doi.org/10.5281/zenodo.20572905)

CAR-MNAR (**C**ausal-discovery **A**ssessment and **R**obustness under **MNAR**)
is a reproducible evaluation framework for studying how constraint- and
score-based causal-discovery algorithms behave when data are missing *not at
random* — specifically under **self-masking** MNAR, where a variable's
missingness depends on its own (unobserved) value.

Its contribution is the evaluation **protocol** itself: deterministically
calibrated, mathematically specified missingness mechanisms, multi-algorithm
ground-truth construction, and Markov-equivalence-aware (CPDAG) evaluation
metrics, so that robustness can be compared fairly and reproducibly across
mechanisms and algorithms at equal realized missingness rates.

> The framework underpins the paper *"CAR-MNAR: A Unified Factorial Framework for
> Robust Causal Discovery under Self-Masking MNAR,"* accepted for presentation in
> the **ECML PKDD 2026 Research Track**. Its central identifiability theorem is
> **machine-verified in Lean 4** (see [`formal_verification/`](formal_verification/)).

---

## Why CAR-MNAR

- **Fair, at-equal-rate comparison.** A Bayesian-optimization calibrator matches
  any mechanism to a target missingness rate, so mechanism-shape effects are not
  confounded by incidental rate differences.
- **Five mathematically specified MNAR mechanisms** — sigmoid self-masking,
  heavy-tailed GPD, threshold censoring, self-masking-with-dependencies, and
  2-level hierarchical — each with an explicit identifiability analysis.
- **CPDAG-aware metrics** that respect Markov equivalence: skeleton F1,
  v-structure F1, compelled-edge orientation F1, CPDAG-SHD, and a structural
  intervention-distance proxy. These expose failure modes (e.g. collapsing
  collider recovery) that a single Structural Hamming Distance hides.
- **Honest, reproducible ground truth.** Clinical reference graphs come from a
  six-algorithm consensus (treated as a proxy); synthetic graphs are known by
  construction and are the basis for absolute claims.
- **A machine-checked core theorem.** The identifiability result that justifies
  the residual-independence test under MNAR is formalised and verified in Lean 4.

## Installation

```bash
git clone https://github.com/Matin-Alinejad/CAR-MNAR.git
cd CAR-MNAR
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

The three clinical datasets (Diabetes, Heart Disease, Hepatitis) are bundled under
[`data/raw/`](data/raw/) so the framework runs offline out of the box; see
[`data/README.md`](data/README.md) for sources and licensing.

## Quickstart

```python
from carmnar.utils.data_loader import MedicalDataLoader
from carmnar.algorithms.mnar_generator import MNARGenerator
from carmnar.algorithms.causal_discovery import PCAlgorithm, SMMVPC
from carmnar.evaluation.cpdag_aware_metrics import evaluate_causal_discovery

loader = MedicalDataLoader()
data = loader.preprocess_dataset(loader.load_dataset("diabetes"), "diabetes")
target = loader.get_effect_variables("diabetes")[0]

true_graph = PCAlgorithm().fit(data)                       # reference graph
incomplete = MNARGenerator(random_state=42).introduce_mnar_effect_variable(
    data, effect_variable=target, target_percentage=0.30, method="sigmoid")

est = SMMVPC(missing_data_method="test_wise_deletion").fit(incomplete)
print(evaluate_causal_discovery(true_graph, est, convert_to_cpdag=True))
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Reproducing the paper's tables

All clinical results are regenerated end-to-end by deterministic scripts:

```bash
python -m carmnar.experiments.real_data_experiments_repro      # local vs. full-graph recovery
python -m carmnar.experiments.comprehensive_metrics_experiment # full-graph CPDAG-aware metrics
python -m carmnar.experiments.baseline_comparison_experiment   # SM-MVPC vs. five baselines
python -m carmnar.experiments.mechanism_comparison_experiment  # sigmoid vs. threshold degradation
```

Outputs are written under `results/`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Formal verification

The framework's central theorem — *under weak self-masking, positivity, and
Condition (TWD), the residual-independence test on the test-wise complete subset
preserves conditional independence and controls the Type-I error* — is
formalised and **machine-verified in Lean 4 / Mathlib v4.28.0**, `sorry`-free,
depending only on Lean's three standard axioms. See
[`formal_verification/`](formal_verification/).

## Repository layout

```
carmnar/            core library (algorithms, mechanisms, metrics, experiments)
  algorithms/         SM-MVPC, PC, baselines, MNAR mechanisms + calibrator
  data_generation/    synthetic causal-network / dataset factory
  evaluation/         CPDAG-aware metrics, ground-truth consensus, SID
  experiments/        reproducible experiment entry points
  utils/              data loading, visualization
data/               bundled clinical datasets (+ provenance)
examples/           minimal runnable examples
formal_verification/ Lean 4 proof of Theorem 1
supplementary/      paper supplementary material (PDF + LaTeX source)
tests/              unit / smoke tests
docs/               reproducibility and design notes
```

## Citation

If you use CAR-MNAR, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

## License

MIT — see [`LICENSE`](LICENSE).
