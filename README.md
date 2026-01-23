# CAR-MNAR: Causal Discovery Assessment and Robustness under Missing Not At Random

**IJCAI 2026 Submission #3206**

---

## Overview

This repository contains the complete implementation of the CAR-MNAR framework, a unified experimental platform for assessing causal discovery algorithms under Missing Not At Random (MNAR) conditions. The codebase implements all algorithms, baselines, metrics, and experiments reported in the paper.

---

## Installation

**Requirements**: Python 3.8+

```bash
pip install -r requirements.txt
```

**Dependencies**: NumPy, SciPy, pandas, scikit-learn, NetworkX, scikit-optimize, statsmodels, matplotlib, seaborn, tqdm.

---

## Quick Start

Reproduce key results from Table 1 (SM-MVPC performance evaluation):

```bash
python run_experiments.py \
    --datasets diabetes heart hepatitis \
    --mechanisms sigmoid \
    --rates 0.0 0.1 0.2 0.3 0.4 0.5 \
    --algorithms SM-MVPC \
    --n_rep 20 \
    --random_seed 42
```

For baseline comparisons (Table 2):

```bash
python run_experiments.py \
    --datasets diabetes heart hepatitis \
    --mechanisms sigmoid \
    --rates 0.3 \
    --algorithms SM-MVPC TD-PC MVPC MissDAG OTM MICE-PC \
    --n_rep 20 \
    --random_seed 42
```

---

## Core Implementation

### Algorithms (`src/algorithms/`)

**`causal_discovery.py`** - Causal structure learning algorithms
- `SMMVPC`: Self-Masking Missing Value PC with test-wise deletion and residual-based independence testing
- `PC`: Standard PC algorithm implementation

**`mathematically_specified_mnar.py`** - MNAR missingness mechanisms
- `SigmoidMNAR`: Logistic missingness P(M|Y) = σ(αY + β)
- `GPDMNAR`: Heavy-tailed missingness using Generalized Pareto Distribution
- `ThresholdMNAR`: Hard censoring at quantile thresholds

**`adaptive_mnar_calibrator.py`** - Parameter calibration
- Bayesian optimization using Gaussian process surrogate (Matérn 5/2 kernel)
- Expected improvement acquisition for efficient parameter search
- Targets specified missingness rates with minimal gap

**`comprehensive_baselines.py`** - Baseline implementations
- TD-PC: Test-wise deletion without residuals
- MVPC: Variational approach
- MissDAG: Joint optimization framework
- OTM: Optimal transport matching
- MICE-PC: Multiple imputation with chained equations

**`factorial_experiment_framework.py`** - Experimental framework (Algorithm 1)
- Full factorial design: Dataset × Mechanism × Rate × Algorithm
- Handles real and synthetic data uniformly
- Automated replication and result aggregation

### Evaluation (`src/evaluation/`)

**`cpdag_aware_metrics.py`** - CPDAG-based evaluation metrics
- Skeleton metrics: Precision, recall, F1 for adjacency recovery
- V-structure metrics: Collider identification performance
- Orientation metrics: Directed edge accuracy (compelled edges only)
- CPDAG-SHD: Structural Hamming Distance respecting Markov equivalence
- SID: Structural Intervention Distance
- Bootstrap confidence intervals (1000 resamples)
- Permutation-based significance testing

**`subgraph_analysis.py`** - Local structure robustness analysis
- Incoming-edge subgraph extraction around target variables
- Robustness ratio computation
- Statistical significance testing with negative controls

**`ground_truth_establishment.py`** - Consensus ground truth for real data
- Multi-algorithm ensemble (PC, GES, CAM, NOTEARS, DirectLiNGAM, ICA-LiNGAM)
- Consensus edges with >60% agreement threshold
- Stability scores for confidence assessment

### Data Generation (`src/data_generation/`)

**`synthetic_dataset_factory.py`** - Synthetic dataset generation
- 51 configurations across experimental conditions
- Topologies: random, chain, fork, collider, scale-free, small-world, mixed
- Network sizes: 5, 10, 15, 20 nodes
- Sample sizes: 200, 500, 1000, 2000
- Noise distributions: Gaussian, uniform, Laplace, exponential, Student-t
- Non-linearities: linear, sinusoidal, quadratic

### Statistical Analysis (`src/analysis/`)

**`advanced_statistical_analysis.py`** - Statistical testing
- Bootstrap confidence intervals with bias-corrected acceleration
- Permutation tests for significance assessment
- Effect size computation (Cohen's d, Cliff's delta)
- Multiple comparison correction (Benjamini-Hochberg FDR)
- Statistical power analysis

---

## Datasets

Three clinical datasets from UCI Machine Learning Repository:

- **Diabetes**: Pima Indians Diabetes Database (768 samples, 9 variables)
- **Heart**: Heart Disease Database (303 samples, 14 variables)
- **Hepatitis**: Hepatitis Database (155 samples, 20 variables)

Located in `data/Raw/`.

---

## Experimental Framework

The factorial experimental framework (Algorithm 1, paper Section 7) systematically evaluates:

- **Datasets**: Real (3 clinical) + Synthetic (51 configurations)
- **MNAR Mechanisms**: Sigmoid, GPD, Threshold, Hierarchical, Self-masking
- **Missingness Rates**: 0%, 10%, 20%, 30%, 40%, 50%
- **Algorithms**: SM-MVPC + 5 baselines
- **Replicates**: n=20 per condition

Total experimental conditions: >18,000 individual runs.

---

## Reproducing Paper Results

See `REPRODUCIBILITY.md` for detailed reproduction instructions including:

- Table 1: SM-MVPC performance degradation curves
- Table 2: Baseline method comparisons
- Figure 2: Subgraph robustness analysis
- Synthetic experiments: Full factorial evaluation

All experiments use `random_seed=42` for reproducibility.

---

## Validation

Verify implementation correctness:

```bash
python validation/test_all_modules.py
python validation/comprehensive_validation.py
```

These scripts check:
- Module imports and dependencies
- MNAR mechanisms achieve target rates (within 2% tolerance)
- Metrics compute valid ranges
- Algorithms execute without errors

---

## Code Architecture

```
src/
├── algorithms/        # Causal discovery, MNAR mechanisms, calibration
├── evaluation/        # Metrics, ground truth, subgraph analysis
├── data_generation/   # Synthetic data factory
├── experiments/       # Orchestration, parallel execution
├── analysis/          # Statistical testing, power analysis
├── simulation/        # Monte Carlo utilities
├── utils/             # General utilities
└── visualization/     # Plotting and figure generation
```

**Total**: 39 modules, ~23,000 lines of code.

See `IMPLEMENTATION_GUIDE.md` for detailed architecture documentation.

---

## Key Implementation Details

**Test-wise deletion**: For each independence test X ⊥ Y | Z, we construct test-specific complete subsets containing only observations with no missing values in {X, Y, Z}. This preserves 3-5× more data than complete-case deletion.

**Residual-based testing**: Under MNAR, we compute conditional-specific residuals before independence testing to remove missingness mechanism confounding.

**CPDAG evaluation**: All metrics respect Markov equivalence. We convert learned DAGs to CPDAGs via Meek rules before comparison, distinguishing compelled vs reversible edges.

**Bayesian calibration**: GP-based optimization converges in ~25 iterations versus ~100 for grid search, while maintaining comparable accuracy (<0.5% gap).

**Multi-algorithm consensus**: For real data ground truth, we require >60% agreement across six diverse algorithms (PC, GES, CAM, NOTEARS, DirectLiNGAM, ICA-LiNGAM) to reduce single-algorithm bias.

---

## Implementation Notes

**Computational requirements**:
- Memory: 8GB minimum, 16GB recommended
- CPU: Multi-core beneficial for parallel experiments
- Runtime: Full reproduction ~12-24 hours on modern hardware

**Platform compatibility**:
- Tested on Ubuntu 20.04, macOS 12, Windows 10
- Python 3.8, 3.9, 3.10 verified

**Numerical precision**:
- Results may vary slightly (±0.02 in F1 scores) across platforms due to floating-point differences
- Random seed 42 ensures determinism within platform

---

## Citation

If you use this code, please cite:

```
[Paper citation to be added after acceptance]
```

---

## License

[To be specified after acceptance]

---

**Submission**: IJCAI 2026 #3206
**Code Version**: 1.0
**Last Updated**: January 2026
