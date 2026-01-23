# Submission Manifest

**Conference**: IJCAI 2026
**Submission ID**: #3206
**Title**: CAR-MNAR: Causal Discovery Assessment and Robustness under Missing Not At Random
**Package Version**: 1.0
**Date**: January 2026

---

## Package Contents

### Documentation (3 files)

1. **README.md** (8.0 KB)
   - Framework overview and quick start guide
   - Core implementation summary
   - Installation and usage instructions

2. **REPRODUCIBILITY.md** (13 KB)
   - Complete reproduction instructions for all paper results
   - Table 1: SM-MVPC performance evaluation
   - Table 2: Baseline comparisons
   - Figure 2: Subgraph robustness analysis
   - Validation procedures

3. **IMPLEMENTATION_GUIDE.md** (16 KB)
   - Detailed architecture documentation
   - All 39 modules described
   - Design patterns and algorithmic details
   - Extension guide

### Dependencies

**requirements.txt** (367 bytes)
- 15 core scientific Python packages
- All open-source, standard libraries

### Source Code (src/)

**39 Python modules organized in 8 packages**:

- `algorithms/` (8 modules) - Causal discovery, MNAR mechanisms, calibration, baselines
- `evaluation/` (9 modules) - CPDAG metrics, subgraph analysis, ground truth establishment
- `data_generation/` (4 modules) - Synthetic dataset factory and generators
- `experiments/` (6 modules) - Experimental orchestration and parallel execution
- `analysis/` (3 modules) - Statistical analysis and power calculations
- `simulation/` (1 module) - Monte Carlo utilities
- `utils/` (5 modules) - General utilities and visualization helpers
- `visualization/` (3 modules) - Plotting and figure generation

**Total**: ~23,000 lines of production code

### Datasets (data/Raw/)

**3 clinical datasets from UCI ML Repository**:

- `Diabetes.csv` - Pima Indians Diabetes Database (768 × 9)
- `Heart.csv` - Heart Disease Database (303 × 14)
- `Hepatitis.csv` - Hepatitis Database (155 × 20)

### Experiment Runners (2 scripts)

1. **run_experiments.py** - Main experiment runner for Tables 1 and 2
2. **run_complete_synthetic_experiments.py** - Full synthetic evaluation suite

### Validation Scripts (validation/)

1. **test_all_modules.py** - Module import and basic functionality checks
2. **comprehensive_validation.py** - End-to-end correctness verification

---

## File Organization

```
IJCAI2026_Submission3206_Code/
├── README.md
├── REPRODUCIBILITY.md
├── IMPLEMENTATION_GUIDE.md
├── MANIFEST.md
├── requirements.txt
├── run_experiments.py
├── run_complete_synthetic_experiments.py
├── src/
│   ├── algorithms/
│   ├── evaluation/
│   ├── data_generation/
│   ├── experiments/
│   ├── analysis/
│   ├── simulation/
│   ├── utils/
│   └── visualization/
├── data/
│   └── Raw/
│       ├── Diabetes.csv
│       ├── Heart.csv
│       └── Hepatitis.csv
└── validation/
    ├── test_all_modules.py
    └── comprehensive_validation.py
```

---

## Implementation Coverage

### Paper Algorithms

✓ **Algorithm 1** - Factorial experimental framework (src/algorithms/factorial_experiment_framework.py)
✓ **SM-MVPC** - Self-masking missing value PC (src/algorithms/causal_discovery.py)
✓ **Bayesian Calibrator** - GP-based parameter optimization (src/algorithms/adaptive_mnar_calibrator.py)

### MNAR Mechanisms

✓ **Sigmoid** - Logistic missingness P(M|Y) = σ(αY + β)
✓ **GPD** - Heavy-tailed via Generalized Pareto Distribution
✓ **Threshold** - Hard censoring at quantiles
✓ **Hierarchical** - Multi-level missingness patterns
✓ **Self-masking** - Dependencies between missingness indicators

### Baseline Methods

✓ **TD-PC** - Test-wise deletion without residuals
✓ **MVPC** - Variational approach
✓ **MissDAG** - Joint optimization
✓ **OTM** - Optimal transport matching
✓ **MICE-PC** - Multiple imputation

### Evaluation Metrics

✓ **Skeleton metrics** - Precision, recall, F1 for adjacency
✓ **V-structure metrics** - Collider identification
✓ **Orientation metrics** - Directed edge accuracy
✓ **CPDAG-SHD** - Structural Hamming Distance
✓ **SID** - Structural Intervention Distance
✓ **Bootstrap CI** - 1000 resamples with BCa correction
✓ **Permutation tests** - Significance assessment

---

## Key Results Reproducibility

| Paper Element | Reproduction Command | Expected Output |
|---------------|---------------------|-----------------|
| Table 1 | `python run_experiments.py --datasets diabetes heart hepatitis --mechanisms sigmoid --rates 0.0 0.1 0.2 0.3 0.4 0.5 --algorithms SM-MVPC --n_rep 20` | Skeleton F1: 0.88→0.83 (Diabetes) |
| Table 2 | `python run_experiments.py --datasets diabetes heart hepatitis --mechanisms sigmoid --rates 0.3 --algorithms SM-MVPC TD-PC MVPC MissDAG OTM MICE-PC --n_rep 20` | SM-MVPC outperforms by 10-20% |
| Figure 2 | `from src.evaluation.subgraph_analysis import SubgraphRobustnessAnalyzer; analyzer.analyze_all_datasets(...)` | Robustness ratio: 2.4-7.0× |

See REPRODUCIBILITY.md for complete instructions.

---

## Technical Specifications

**Platform Compatibility**:
- Ubuntu 20.04, macOS 12, Windows 10 tested
- Python 3.8, 3.9, 3.10 verified

**Computational Requirements**:
- Memory: 8GB minimum, 16GB recommended
- CPU: Multi-core beneficial (parallelization supported)
- Storage: 2GB for package + outputs

**Runtime Estimates** (hardware-dependent):
- Quick validation: 5-15 minutes
- Table 1 full: 2-4 hours
- Table 2 full: 4-8 hours
- Complete synthetic: 24-48 hours

**Reproducibility**:
- All experiments use `random_seed=42`
- Results consistent within ±0.02 (platform differences)
- Exact replication requires identical platform/versions

---

## Contact

**During Review**: Use anonymous conference review system

**After Acceptance**: Contact information will be provided

