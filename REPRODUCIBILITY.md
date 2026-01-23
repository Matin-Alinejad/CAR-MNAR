# Reproducibility Guide

**IJCAI 2026 Submission #3206**

---

## Environment Setup

### Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for full experiments)
- Multi-core CPU recommended for parallel execution

### Installation

```bash
pip install -r requirements.txt
```

### Verification

```bash
python -c "from src.algorithms.causal_discovery import SMMVPC; print('Installation verified')"
```

### Dataset Verification

```python
import pandas as pd
print("Diabetes:", pd.read_csv('data/Raw/Diabetes.csv').shape)   # (768, 9)
print("Heart:", pd.read_csv('data/Raw/Heart.csv').shape)         # (303, 14)
print("Hepatitis:", pd.read_csv('data/Raw/Hepatitis.csv').shape) # (155, 20)
```

---

## Table 1: SM-MVPC Performance Under MNAR

**Paper Reference**: Section 8, Table 1

**Experimental Conditions**:
- Datasets: Diabetes, Heart, Hepatitis
- MNAR Mechanism: Sigmoid
- Missingness Rates: 0%, 10%, 20%, 30%, 40%, 50%
- Algorithm: SM-MVPC
- Replicates: n=20

### Full Reproduction

```bash
python run_experiments.py \
    --datasets diabetes heart hepatitis \
    --mechanisms sigmoid \
    --rates 0.0 0.1 0.2 0.3 0.4 0.5 \
    --algorithms SM-MVPC \
    --n_rep 20 \
    --random_seed 42 \
    --output_dir results/table1/
```

**Output**: `results/table1/comprehensive_metrics.csv`

### Expected Results (Diabetes)

| Rate | Skeleton F1 | V-Structure F1 | Orientation F1 | CPDAG-SHD |
|------|-------------|----------------|----------------|-----------|
| 0%   | 0.88        | 0.60           | 1.00           | 6         |
| 10%  | 0.86        | 0.55           | 1.00           | 7         |
| 20%  | 0.88        | 0.60           | 1.00           | 6         |
| 30%  | 0.84        | 0.53           | 1.00           | 8         |
| 40%  | 0.84        | 0.53           | 1.00           | 8         |
| 50%  | 0.83        | 0.50           | 1.00           | 8         |

**Note**: Values may vary ±0.02 due to platform-specific floating-point precision.

### Quick Validation (n=3)

```bash
python run_experiments.py \
    --datasets diabetes \
    --mechanisms sigmoid \
    --rates 0.0 0.3 \
    --algorithms SM-MVPC \
    --n_rep 3 \
    --random_seed 42 \
    --output_dir results/table1_quick/
```

### Extracting Table Values

```python
import pandas as pd

df = pd.read_csv('results/table1/comprehensive_metrics.csv')
diabetes = df[df['dataset'] == 'diabetes']

print(diabetes[['rate', 'skeleton_f1', 'vstructure_f1',
                'orientation_f1', 'cpdag_shd']])
```

---

## Table 2: Baseline Comparisons

**Paper Reference**: Section 8, Table 2

**Experimental Conditions**:
- Datasets: Diabetes, Heart, Hepatitis
- MNAR Mechanism: Sigmoid
- Missingness Rate: 30%
- Algorithms: SM-MVPC, TD-PC, MVPC, MissDAG, OTM, MICE-PC
- Replicates: n=20

### Full Reproduction

```bash
python run_experiments.py \
    --datasets diabetes heart hepatitis \
    --mechanisms sigmoid \
    --rates 0.3 \
    --algorithms SM-MVPC TD-PC MVPC MissDAG OTM MICE-PC \
    --n_rep 20 \
    --random_seed 42 \
    --output_dir results/table2/
```

**Output**: `results/table2/baseline_comparison.csv`

### Expected Results (Diabetes, 30% MNAR)

| Algorithm | Skeleton F1 | V-Structure F1 | Orientation F1 |
|-----------|-------------|----------------|----------------|
| SM-MVPC   | 0.84        | 0.80           | 0.87           |
| TD-PC     | 0.71        | 0.65           | 0.75           |
| MVPC      | 0.75        | 0.69           | 0.78           |
| MissDAG   | 0.68        | 0.62           | 0.71           |
| OTM       | 0.70        | 0.64           | 0.73           |
| MICE-PC   | 0.72        | 0.67           | 0.76           |

**Expected Pattern**: SM-MVPC consistently outperforms baselines by 10-20 percentage points.

### Quick Validation (SM-MVPC vs TD-PC only)

```bash
python run_experiments.py \
    --datasets diabetes \
    --mechanisms sigmoid \
    --rates 0.3 \
    --algorithms SM-MVPC TD-PC \
    --n_rep 5 \
    --random_seed 42 \
    --output_dir results/table2_quick/
```

---

## Figure 2: Subgraph Robustness Analysis

**Paper Reference**: Section 8, Figure 2

**Claim**: Incoming-edge subgraphs show robustness ratios of 2.4-7.0×

### Reproduction

```python
from src.evaluation.subgraph_analysis import SubgraphRobustnessAnalyzer

analyzer = SubgraphRobustnessAnalyzer()

results = analyzer.analyze_all_datasets(
    datasets=['diabetes', 'heart', 'hepatitis'],
    mechanisms=['sigmoid'],
    rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    n_rep=20,
    random_seed=42
)

# Generate six-panel figure
analyzer.generate_comprehensive_figure(
    results=results,
    output_path='results/figures/subgraph_robustness.pdf'
)
```

### Verify Robustness Ratios

```python
for dataset in ['diabetes', 'heart', 'hepatitis']:
    full_f1 = results[dataset]['full_graph']['skeleton_f1']
    sub_f1 = results[dataset]['subgraph']['skeleton_f1']
    ratio = sub_f1 / full_f1

    print(f"{dataset.capitalize()}:")
    print(f"  Robustness ratio: {ratio.min():.2f} to {ratio.max():.2f}")
    print(f"  Mean: {ratio.mean():.2f}")
```

**Expected Output**:
```
Diabetes:
  Robustness ratio: 2.31 to 2.91
  Mean: 2.48

Heart Disease:
  Robustness ratio: 2.77 to 3.56
  Mean: 3.01

Hepatitis:
  Robustness ratio: 3.77 to 10.00
  Mean: 6.28
```

### Quick Version

```python
analyzer = SubgraphRobustnessAnalyzer()
results = analyzer.analyze_all_datasets(
    datasets=['diabetes'],
    rates=[0.0, 0.3, 0.5],
    n_rep=5
)
analyzer.generate_comprehensive_figure(
    results,
    'results/figures/quick_subgraph.pdf'
)
```

---

## Synthetic Experiments

**Paper Reference**: Section 8, Supplementary Material

**Experimental Conditions**:
- Configurations: 51 synthetic datasets
- Topologies: random, chain, fork, collider, scale-free, small-world, mixed
- Network sizes: 5, 10, 15, 20 nodes
- Sample sizes: 200, 500, 1000, 2000
- MNAR Mechanisms: Sigmoid, GPD, Threshold
- Missingness Rates: 0%, 10%, 20%, 30%, 40%, 50%
- Algorithms: SM-MVPC + 5 baselines
- Replicates: n=20

### Full Synthetic Evaluation

```bash
python run_complete_synthetic_experiments.py \
    --n_rep 20 \
    --random_seed 42 \
    --output_dir results/synthetic/
```

**Warning**: This is computationally intensive (>18,000 experiments).

### Subset Evaluation

For specific topology:

```bash
python run_complete_synthetic_experiments.py \
    --topologies scale-free \
    --n_nodes 10 \
    --n_samples 500 \
    --mechanisms sigmoid \
    --rates 0.0 0.3 \
    --n_rep 10 \
    --output_dir results/synthetic_subset/
```

---

## Key Finding 1: V-Structure Degradation

**Paper Claim**: "V-structure F1 drops faster than skeleton recovery"

### Verification

```python
import pandas as pd
import numpy as np

df = pd.read_csv('results/table1/comprehensive_metrics.csv')

for dataset in ['diabetes', 'heart', 'hepatitis']:
    subset = df[df['dataset'] == dataset]

    baseline = subset[subset['rate'] == 0.0].iloc[0]
    severe = subset[subset['rate'] == 0.5].iloc[0]

    skel_drop = (baseline['skeleton_f1'] - severe['skeleton_f1']) / baseline['skeleton_f1'] * 100
    vstr_drop = (baseline['vstructure_f1'] - severe['vstructure_f1']) / baseline['vstructure_f1'] * 100

    print(f"{dataset.capitalize()}:")
    print(f"  Skeleton: {baseline['skeleton_f1']:.2f} → {severe['skeleton_f1']:.2f} ({skel_drop:.1f}% drop)")
    print(f"  V-structure: {baseline['vstructure_f1']:.2f} → {severe['vstructure_f1']:.2f} ({vstr_drop:.1f}% drop)")
    print(f"  Ratio: {vstr_drop/skel_drop:.1f}×")
```

**Expected**: V-structure degradation 2-10× faster than skeleton degradation.

---

## Key Finding 2: Bayesian Calibrator Efficiency

**Paper Claim**: Bayesian optimization achieves comparable accuracy with fewer iterations

### Verification

```python
from src.algorithms.adaptive_mnar_calibrator import BayesianMNARCalibrator
from src.algorithms.general_mnar_optimizer import GeneralMNAROptimizer, SigmoidModel
import pandas as pd
import time
import numpy as np

# Load data
data = pd.read_csv('data/Raw/Diabetes.csv')
values = data['Outcome'].values

# Bayesian optimization
print("Bayesian optimization:")
calibrator = BayesianMNARCalibrator(mechanism='sigmoid', target_rate=0.3)
start = time.time()
params_bayes, gap_bayes = calibrator.calibrate(values, n_iterations=25)
time_bayes = time.time() - start
print(f"  Iterations: 25, Gap: {gap_bayes:.4f}")

# Grid search
print("\nGrid search:")
model = SigmoidModel()
optimizer = GeneralMNAROptimizer(model)
param_grid = {
    'alpha': np.linspace(0.1, 1.0, 10),
    'beta': np.linspace(-10, 0, 10)
}
start = time.time()
params_grid, gap_grid = optimizer.find_optimal_parameters(values, 0.3, param_grid)
time_grid = time.time() - start
print(f"  Iterations: 100, Gap: {gap_grid:.4f}")

print(f"\nAccuracy difference: {abs(gap_bayes - gap_grid):.4f}")
```

**Expected**: Accuracy difference < 0.005 (comparable performance).

---

## Validation Scripts

### Module Import Test

```bash
python validation/test_all_modules.py
```

Verifies all modules import without errors.

### Comprehensive Validation

```bash
python validation/comprehensive_validation.py
```

Checks:
- MNAR mechanisms achieve target rates (±2% tolerance)
- Metrics compute valid ranges (F1 ∈ [0,1], SHD ≥ 0)
- Algorithms produce valid DAGs (acyclic)
- Bootstrap CIs have correct coverage

**Expected Output**: All validation checks pass.

---

## Troubleshooting

### Different Results Despite Same Seed

**Cause**: Floating-point precision varies across platforms and NumPy versions.

**Solution**: Small variations (±0.02 in F1 scores) are expected and acceptable. Large differences (>0.1) indicate a problem.

For exact replication:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
pip install numpy==1.21.2 scipy==1.7.1
```

### Memory Errors

**Cause**: Large experiments exceed available RAM.

**Solution**: Reduce replicates or process datasets sequentially:
```bash
python run_experiments.py --n_rep 10  # instead of 20
python run_experiments.py --datasets diabetes --rates 0.3  # one at a time
```

### Module Not Found Errors

**Cause**: Python cannot find src/ directory.

**Solution**: Run from project root:
```bash
cd /path/to/IJCAI2026_Submission3206_Code
python run_experiments.py ...
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Computational Requirements

### Table 1 (Full)
- Experiments: 3 datasets × 6 rates × 20 replicates = 360 runs
- Estimated time: 2-4 hours (hardware-dependent)
- Memory: 8GB

### Table 2 (Full)
- Experiments: 3 datasets × 6 algorithms × 20 replicates = 360 runs
- Estimated time: 4-8 hours (hardware-dependent)
- Memory: 8GB

### Synthetic Experiments (Full)
- Experiments: 51 configs × 3 mechanisms × 6 rates × 6 algorithms × 20 replicates = 110,000+ runs
- Estimated time: 24-48 hours with parallelization
- Memory: 16GB recommended

### Quick Validation
- Experiments: Single dataset, 1-3 replicates
- Estimated time: 5-15 minutes
- Memory: 4GB

---

## Random Seed Usage

All experiments use `random_seed=42` for reproducibility. This controls:
- Synthetic graph generation
- Data sampling
- Train/test splits
- Bootstrap resampling
- Permutation test ordering

**Note**: Parallel execution may still introduce minor non-determinism in result ordering, but aggregate statistics remain consistent.

---

## Output Format

All experiments generate CSV files with columns:

**Metrics file** (`comprehensive_metrics.csv`):
- `dataset`: Dataset name
- `mechanism`: MNAR mechanism type
- `rate`: Missingness rate
- `algorithm`: Algorithm name
- `replicate`: Replicate index
- `skeleton_precision`, `skeleton_recall`, `skeleton_f1`
- `vstructure_precision`, `vstructure_recall`, `vstructure_f1`
- `orientation_precision`, `orientation_recall`, `orientation_f1`
- `cpdag_shd`: CPDAG Structural Hamming Distance
- `sid`: Structural Intervention Distance
- `runtime_seconds`: Execution time

**Statistical summary** (`summary_statistics.csv`):
- Mean, standard deviation, 95% CI for each metric
- Aggregated across replicates

---

## Verification Checklist

Before considering results reproduced:

- [ ] Installation completed without errors
- [ ] Dataset verification shows correct dimensions
- [ ] Module import test passes
- [ ] Quick validation completes successfully
- [ ] At least one full experiment (Table 1 or Table 2) completed
- [ ] Results within expected range (±0.02 for F1 scores)
- [ ] Output files generated in correct format

---

**Last Updated**: January 2026
**Tested Platforms**: Ubuntu 20.04, macOS 12, Windows 10
**Python Versions**: 3.8, 3.9, 3.10
