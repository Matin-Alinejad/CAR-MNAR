# Implementation Architecture

**IJCAI 2026 Submission #3206**

---

## System Architecture

The codebase implements a layered architecture with clear separation of concerns:

```
Experiment Orchestration Layer (experiments/)
              ↓
Algorithm Layer (algorithms/)
              ↓
Evaluation Layer (evaluation/)
              ↓
Data Generation Layer (data_generation/)
              ↓
Statistical Analysis Layer (analysis/)
```

Data flows downward through the pipeline; results aggregate upward. Each layer has minimal dependencies, enabling independent testing and validation.

---

## Module Inventory

### Core Algorithms (`src/algorithms/`)

**`causal_discovery.py`** (580 lines)
- Standard PC algorithm with Fisher's Z test
- SM-MVPC with test-wise deletion
- Residual-based conditional independence testing
- Skeleton construction via edge removal
- V-structure identification and orientation

**`mathematically_specified_mnar.py`** (920 lines)
- `SigmoidMNAR`: P(M|Y) = 1/(1 + exp(-(αY + β)))
- `GPDMNAR`: Heavy-tailed via Generalized Pareto Distribution
  - P(M|Y) = 1 - (1 + ξ(Y-μ)/σ)^(-1/ξ) for Y > μ
- `ThresholdMNAR`: Hard censoring at quantile q
  - P(M|Y) = 1 if Y > Q(q), else 0
- Identifiability analysis for each mechanism
- Parameter validation and bounds checking

**`adaptive_mnar_calibrator.py`** (650 lines)
- Bayesian optimization framework
- Gaussian process surrogate with Matérn 5/2 kernel
- Expected improvement acquisition function
- Convergence criteria: |rate_achieved - rate_target| < 0.02
- Adaptive sampling with 25 iterations default

**`comprehensive_baselines.py`** (715 lines)
- TD-PC: Test-wise deletion without residual adjustment
- MVPC: Variational approach with evidence lower bound (ELBO)
- MissDAG: Joint optimization of graph and missing data model
- OTM: Optimal transport matching for missing value handling
- MICE-PC: Multiple imputation (m=5) with chained equations

**`factorial_experiment_framework.py`** (549 lines)
- Full factorial design implementation (Algorithm 1)
- Experimental condition generation
- Parallel execution with joblib
- Result aggregation and persistence
- Progress tracking and checkpointing

**`advanced_mnar_mechanisms.py`** (680 lines)
- Hierarchical MNAR with multi-level patterns
- Self-masking with dependency structures
- Complex mechanism composition utilities

**`general_mnar_optimizer.py`** (420 lines)
- Grid search baseline for calibration comparison
- Random search alternative
- Parameter space exploration utilities

### Evaluation Metrics (`src/evaluation/`)

**`cpdag_aware_metrics.py`** (1044 lines)
- **DAG to CPDAG conversion**:
  - Meek Rule R1: Orient i→j→k with no i-k edge as i→j→k
  - Meek Rule R2: If i→j-k and i-k, orient j→k
  - Meek Rule R3: If i-j-k and i-l→k and l-j with no i-l and l-k edges, orient j→k
  - Meek Rule R4: If i-j-k and i-l-k and i-m→k with no m-j and m-k edges, orient j→k

- **Skeleton metrics**:
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)

- **V-structure metrics**:
  - Identifies collider patterns i→j←k
  - Computes precision, recall, F1 for collider recovery

- **Orientation metrics**:
  - Evaluates only compelled edges in CPDAG
  - Excludes reversible edges from orientation accuracy

- **CPDAG-SHD**:
  - Structural Hamming Distance respecting equivalence
  - Counts: added edges + removed edges + incorrectly oriented compelled edges

- **SID (Structural Intervention Distance)**:
  - Measures difference in interventional distributions
  - Computes missing and extra edges weighted by causal paths

- **Bootstrap confidence intervals**:
  - 1000 resamples with replacement
  - Bias-corrected accelerated (BCa) method
  - 95% confidence level default

**`subgraph_analysis.py`** (720 lines)
- Incoming-edge subgraph extraction: G_X = {Y → X | Y ∈ Pa(X)}
- Robustness ratio: R = F1_subgraph / F1_full
- Negative control generation via random subgraph matching
- Permutation tests for statistical significance (1000 permutations)

**`ground_truth_establishment.py`** (480 lines)
- Multi-algorithm ensemble approach:
  - PC: Constraint-based via conditional independence
  - GES: Score-based greedy equivalence search
  - CAM: Causal additive models with generalized additive models
  - NOTEARS: Continuous optimization with acyclicity constraint
  - DirectLiNGAM: Direct learning of linear non-Gaussian acyclic models
  - ICA-LiNGAM: Independent component analysis-based LiNGAM
- Consensus: Edges with >60% agreement across algorithms
- Stability score: Proportion of algorithms agreeing on each edge
- Confidence threshold: Include only edges with stability >0.78

**`negative_controls.py`** (390 lines)
- Random subgraph generation matching size distribution
- Density-matched control selection
- Permutation-based null distribution construction
- False discovery rate estimation

### Data Generation (`src/data_generation/`)

**`synthetic_dataset_factory.py`** (650 lines)
- **Topology generation**:
  - Random: Erdős-Rényi G(n, p) with edge probability p
  - Chain: Linear sequence v1 → v2 → ... → vn
  - Fork: Common cause v1 → {v2, ..., vn}
  - Collider: Common effect {v1, ..., vn-1} → vn
  - Scale-free: Barabási-Albert preferential attachment
  - Small-world: Watts-Strogatz with rewiring probability
  - Mixed: Combination of patterns

- **Data generation process**:
  1. Generate DAG structure according to topology
  2. Assign random edge weights from [-1, -0.5] ∪ [0.5, 1]
  3. Sample exogenous noise from specified distribution
  4. Compute values via structural equations: Xi = Σ wij·Xj + εi
  5. Apply non-linearity if specified (sin, square)
  6. Normalize to zero mean, unit variance

- **51 configurations**:
  - 7 topologies × 4 network sizes × varying sample sizes × 3 noise types × 3 non-linearities
  - Systematic coverage of experimental space

**`enhanced_synthetic_dataset_factory.py`** (820 lines)
- Extended configurations with clinical data templates
- Complex interaction patterns (moderation, mediation)
- Time-varying structures for longitudinal data

### Experiment Orchestration (`src/experiments/`)

**`main_experiment.py`** (680 lines)
- Main experimental pipeline
- Progress tracking with tqdm
- Intermediate result saving every 50 experiments
- Result aggregation and summary statistics
- Error handling and logging

**`distributed_experiment_runner.py`** (490 lines)
- Parallel execution using joblib.Parallel
- Load balancing across available cores
- Fault tolerance with error capture
- Resource monitoring and adaptive scheduling

### Statistical Analysis (`src/analysis/`)

**`advanced_statistical_analysis.py`** (920 lines)
- **Bootstrap confidence intervals**:
  - Standard percentile method
  - Bias-corrected accelerated (BCa) method
  - 1000 resamples default

- **Permutation tests**:
  - Exact permutation for small samples
  - Monte Carlo approximation for large samples
  - Two-sided p-values

- **Effect sizes**:
  - Cohen's d: (μ1 - μ2) / σpooled
  - Cliff's delta: P(X1 > X2) - P(X2 > X1)
  - Interpretation thresholds: small (0.2), medium (0.5), large (0.8)

- **Multiple comparison correction**:
  - Benjamini-Hochberg FDR control
  - Bonferroni correction for conservative control
  - Holm-Bonferroni step-down procedure

**`statistical_power_analysis.py`** (580 lines)
- Power calculations for different metrics
- Sample size requirements for target power (0.80)
- Sensitivity analysis for effect size detection
- Minimum detectable effect size computation

---

## Design Patterns

### Protocol-Based Interfaces

Instead of rigid inheritance hierarchies, we use structural typing (PEP 544):

```python
class MissingnessModel(Protocol):
    def calculate_probabilities(self, values: np.ndarray,
                                params: Dict[str, float]) -> np.ndarray:
        """Return missingness probabilities for each value."""
        ...
```

Any class implementing this interface works seamlessly with the framework.

### Factory Pattern

Dataset generation uses factories for configuration management:

```python
factory = SyntheticDatasetFactory()
dataset = factory.generate(
    topology='scale-free',
    n_nodes=10,
    n_samples=500,
    noise_distribution='gaussian'
)
```

### Strategy Pattern

Algorithm selection and execution via strategy:

```python
calibrator = MNARCalibrator(strategy='bayesian')  # or 'grid', 'random'
params = calibrator.calibrate(data, target_rate=0.3)
```

### Composition Over Inheritance

The experimental framework composes components:

```python
framework = FactorialExperimentFramework(
    datasets=dataset_configs,
    mechanisms=mechanism_configs,
    rates=missingness_rates,
    algorithms=algorithm_instances
)
```

---

## Key Algorithmic Details

### Test-Wise Deletion

For independence test X ⊥ Y | Z:

1. Identify variables involved: V = {X, Y} ∪ Z
2. Find observations with complete data on V
3. Perform test on this subset
4. Different tests use different subsets

Contrast with complete-case deletion (CCD):
- CCD: Remove any observation with any missing value
- Test-wise: Remove observation only if missing in relevant variables
- Typical preservation: 60-80% of data vs 20-40% for CCD

### Residual-Based Independence

Under MNAR, missingness depends on variable values, creating confounding. Our approach:

1. For test X ⊥ Y | Z under MNAR:
2. Compute residuals: r_X = X - E[X|M_X, Z], r_Y = Y - E[Y|M_Y, Z]
3. Test: r_X ⊥ r_Y | Z
4. Removes missingness mechanism confounding

### CPDAG Construction

Conversion from DAG to CPDAG (Completed Partially Directed Acyclic Graph):

1. Start with skeleton (undirected graph)
2. Identify v-structures: i→j←k where i and k not adjacent
3. Orient these as compelled
4. Apply Meek rules iteratively until convergence:
   - R1: Avoid new v-structures
   - R2: Avoid cycles
   - R3, R4: Propagate orientations
5. Remaining undirected edges are reversible (Markov equivalent)

### Bayesian Calibration

Gaussian process optimization for MNAR parameters:

1. Initialize: Latin hypercube sampling (5 points)
2. Build GP surrogate: μ(θ) and σ(θ) for objective f(θ) = |rate(θ) - target|
3. Acquisition: α(θ) = (μ_min - μ(θ)) / σ(θ) × Φ((μ_min - μ(θ)) / σ(θ))
4. Optimize acquisition: θ_next = argmax α(θ)
5. Evaluate: f(θ_next), update GP
6. Repeat until convergence or max iterations (25)

Kernel: Matérn 5/2 for twice-differentiable functions
- k(θ, θ') = σ² (1 + √5r + 5r²/3) exp(-√5r)
- r = ||θ - θ'|| / ℓ (length scale)

---

## Module Dependencies

```
experiments/ → algorithms/, evaluation/, data_generation/
algorithms/ → (no internal dependencies)
evaluation/ → algorithms/ (for graph structures)
analysis/ → (no internal dependencies)
data_generation/ → (no internal dependencies)
```

Clean layering enables independent testing and modification.

External dependencies:
- Core: NumPy, SciPy, pandas
- Graphs: NetworkX
- ML: scikit-learn
- Optimization: scikit-optimize
- Stats: statsmodels
- Viz: matplotlib, seaborn
- Utils: tqdm

---

## Performance Characteristics

**Computational complexity**:
- PC algorithm: O(n × p³) where n=samples, p=variables
- MNAR calibration: O(T × n) where T=iterations
- Bootstrap CI: O(B × C) where B=resamples, C=metric cost
- CPDAG conversion: O(p²) via Meek rules

**Memory usage**:
- Typical: 4-8GB for clinical datasets
- Peak: 12GB for large synthetic experiments (n=2000, p=20)
- Dominated by independence test covariance matrices

**Parallelization**:
- Experiments across conditions: Embarrassingly parallel
- Within-experiment: Independence tests parallelizable but not implemented
- Speedup: Near-linear up to number of conditions

**Bottlenecks**:
- Independence testing (70% of runtime)
- Bootstrap resampling (15% for expensive metrics)
- GP surrogate fitting (10% during calibration)

---

## Validation Strategy

Rather than comprehensive unit tests, we provide validation scripts:

- `test_all_modules.py`: Import checks and basic functionality
- `comprehensive_validation.py`: End-to-end correctness verification
  - MNAR mechanisms hit target rates (±2%)
  - Metrics compute valid ranges (F1 ∈ [0,1], SHD ≥ 0)
  - Algorithms produce valid DAGs (acyclic, proper edge types)
  - Bootstrap CIs contain true values in synthetic data
  - Permutation tests have correct null distribution (uniform p-values)

For research code, validation scripts provide adequate correctness checking without the overhead of comprehensive test suites.

---

## Extension Guide

### Adding a New MNAR Mechanism

1. Implement the protocol:
```python
class CustomMNAR:
    def calculate_probabilities(self, values: np.ndarray,
                                params: Dict[str, float]) -> np.ndarray:
        # Return P(missing|value) for each value
        return custom_function(values, params)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        # Return valid parameter ranges
        return {'param1': (0.0, 1.0), 'param2': (-10.0, 10.0)}
```

2. Register and use:
```python
from src.algorithms.advanced_mnar_mechanisms import register_mechanism
register_mechanism('custom', CustomMNAR)

# Use in experiments
mechanism_config = MissingnessConfig(
    name='Custom',
    model_type='custom',
    parameter_grid={'param1': [0.5], 'param2': [-5.0]}
)
```

### Adding a New Algorithm

1. Implement structure learning interface:
```python
class NewAlgorithm:
    def learn_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        # Return learned causal graph
        graph = nx.DiGraph()
        # ... learning logic ...
        return graph

    def get_name(self) -> str:
        return "NewAlgorithm"
```

2. Use in framework:
```python
algorithms = [SMMVPC(), NewAlgorithm()]
framework = FactorialExperimentFramework(
    datasets=datasets,
    mechanisms=mechanisms,
    rates=rates,
    algorithms=algorithms
)
```

### Adding a New Metric

1. Extend evaluation result dataclass:
```python
from dataclasses import dataclass, asdict
from src.evaluation.cpdag_aware_metrics import CPDAGEvaluationResult

@dataclass
class ExtendedResult(CPDAGEvaluationResult):
    custom_metric: float
```

2. Extend evaluator:
```python
class ExtendedEvaluator(CPDAGEvaluator):
    def evaluate(self, true_cpdag: nx.DiGraph,
                 inferred_cpdag: nx.DiGraph) -> ExtendedResult:
        base_result = super().evaluate(true_cpdag, inferred_cpdag)
        custom_value = self._compute_custom_metric(true_cpdag, inferred_cpdag)
        return ExtendedResult(**asdict(base_result), custom_metric=custom_value)
```

---

## Known Limitations

**Scale**: Tested up to 30-node graphs. Larger graphs (100+ nodes) require algorithmic optimization (approximate independence tests, parallelization).

**Platform**: Results vary slightly (±0.02 in F1) across platforms due to floating-point precision. Use consistent platform for exact replication.

**Memory**: Large synthetic experiments (p=20, n=2000, 1000 bootstrap resamples) can require 12-16GB RAM. Reduce batch size or resamples if memory-constrained.

**Runtime**: Full reproduction (18,000+ experiments) requires 12-24 hours on modern hardware. Quick validation runs complete in minutes.

---

## Code Quality Metrics

**Documentation**:
- Docstring coverage: ~90%
- Most modules include usage examples
- All public APIs documented

**Type hints**:
- Extensive coverage in newer modules
- Gradual typing approach

**Naming conventions**:
- PEP 8 compliance
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE

**Line length**:
- Flexible 80-100 character limit
- Readability prioritized over strict limits

---

**Code Version**: 1.0
**Implementation**: ~23,000 lines across 39 modules
**Last Updated**: January 2026
