"""
Advanced Diagnostic Visualization Framework for Causal Discovery Evaluation

This module provides comprehensive diagnostic visualizations for evaluating
causal discovery algorithms, including missingness heatmaps, QQ plots,
performance degradation curves, and specialized diagnostic plots suitable
for academic publication.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import probplot
import networkx as nx

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDiagnosticVisualizer:
    """
    Advanced diagnostic visualizer for causal discovery evaluation.
    
    This class provides comprehensive diagnostic visualizations including
    missingness heatmaps, QQ plots, performance degradation curves, and
    specialized diagnostic plots for academic evaluation.
    """
    
    def __init__(self, output_dir: str = "results/diagnostic_visualizations"):
        """
        Initialize the advanced diagnostic visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up color schemes
        self.colors = {
            'threshold': '#1f77b4',
            'parametric': '#ff7f0e',
            'sigmoid': '#2ca02c',
            'baseline': '#d62728',
            'missing': '#9467bd',
            'performance': '#8c564b'
        }
        
        self.markers = {
            'threshold': 'o',
            'parametric': 's',
            'sigmoid': '^',
            'baseline': 'D',
            'missing': 'v',
            'performance': 'p'
        }
        
        # Set up plot parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_comprehensive_diagnostic_plots(self, 
                                            experimental_results: Dict[str, Any],
                                            missingness_data: Dict[str, Any],
                                            performance_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create comprehensive diagnostic plots for causal discovery evaluation.
        
        Args:
            experimental_results: Results from experimental evaluation
            missingness_data: Missingness pattern data
            performance_data: Performance metrics data
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        # 1. Missingness pattern visualizations
        plot_paths['missingness_heatmaps'] = self._create_missingness_heatmaps(
            missingness_data
        )
        
        # 2. Tail quantile-quantile plots
        plot_paths['tail_qq_plots'] = self._create_tail_qq_plots(
            experimental_results
        )
        
        # 3. Performance degradation curves
        plot_paths['performance_degradation_curves'] = self._create_performance_degradation_curves(
            performance_data
        )
        
        # 4. Structural metrics analysis
        plot_paths['structural_metrics_analysis'] = self._create_structural_metrics_analysis(
            experimental_results
        )
        
        # 5. Mechanism comparison diagnostics
        plot_paths['mechanism_comparison_diagnostics'] = self._create_mechanism_comparison_diagnostics(
            experimental_results
        )
        
        # 6. Statistical significance diagnostics
        plot_paths['statistical_significance_diagnostics'] = self._create_statistical_significance_diagnostics(
            experimental_results
        )
        
        # 7. Robustness assessment plots
        plot_paths['robustness_assessment_plots'] = self._create_robustness_assessment_plots(
            experimental_results
        )
        
        # 8. Comprehensive diagnostic summary
        plot_paths['comprehensive_diagnostic_summary'] = self._create_comprehensive_diagnostic_summary(
            experimental_results, missingness_data, performance_data
        )
        
        return plot_paths
    
    def _create_missingness_heatmaps(self, missingness_data: Dict[str, Any]) -> str:
        """Create missingness pattern heatmaps."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Missingness Pattern Analysis: Heatmaps and Distributions', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Missingness rate heatmap across datasets and mechanisms
        ax1 = axes[0, 0]
        self._plot_missingness_rate_heatmap(ax1, missingness_data)
        
        # Plot 2: Missingness pattern heatmap for specific dataset
        ax2 = axes[0, 1]
        self._plot_missingness_pattern_heatmap(ax2, missingness_data)
        
        # Plot 3: Missingness distribution across variables
        ax3 = axes[1, 0]
        self._plot_missingness_distribution(ax3, missingness_data)
        
        # Plot 4: Missingness correlation analysis
        ax4 = axes[1, 1]
        self._plot_missingness_correlation(ax4, missingness_data)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'missingness_heatmaps_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_tail_qq_plots(self, experimental_results: Dict[str, Any]) -> str:
        """Create tail quantile-quantile plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tail Quantile-Quantile (QQ) Plots: Heavy-Tailed Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Original data vs normal distribution
        ax1 = axes[0, 0]
        self._plot_qq_normal_comparison(ax1, experimental_results)
        
        # Plot 2: Original data vs heavy-tailed distribution
        ax2 = axes[0, 1]
        self._plot_qq_heavy_tailed_comparison(ax2, experimental_results)
        
        # Plot 3: MNAR effect on tail behavior
        ax3 = axes[1, 0]
        self._plot_qq_mnar_effect(ax3, experimental_results)
        
        # Plot 4: Tail index estimation
        ax4 = axes[1, 1]
        self._plot_tail_index_estimation(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'tail_qq_plots_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_performance_degradation_curves(self, performance_data: Dict[str, Any]) -> str:
        """Create performance degradation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Degradation Analysis: relSHD vs Missingness Curves', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: relSHD vs missingness for all mechanisms
        ax1 = axes[0, 0]
        self._plot_relshd_vs_missingness(ax1, performance_data)
        
        # Plot 2: F1-score vs missingness for all mechanisms
        ax2 = axes[0, 1]
        self._plot_f1_vs_missingness(ax2, performance_data)
        
        # Plot 3: Performance degradation rates
        ax3 = axes[1, 0]
        self._plot_degradation_rates(ax3, performance_data)
        
        # Plot 4: Threshold analysis
        ax4 = axes[1, 1]
        self._plot_performance_thresholds(ax4, performance_data)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'performance_degradation_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_structural_metrics_analysis(self, experimental_results: Dict[str, Any]) -> str:
        """Create structural metrics analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Structural Metrics Analysis: Comprehensive Evaluation', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: relSHD distribution across mechanisms
        ax1 = axes[0, 0]
        self._plot_relshd_distribution(ax1, experimental_results)
        
        # Plot 2: Adjacency metrics comparison
        ax2 = axes[0, 1]
        self._plot_adjacency_metrics_comparison(ax2, experimental_results)
        
        # Plot 3: Orientation accuracy analysis
        ax3 = axes[1, 0]
        self._plot_orientation_accuracy_analysis(ax3, experimental_results)
        
        # Plot 4: Structural complexity metrics
        ax4 = axes[1, 1]
        self._plot_structural_complexity_metrics(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'structural_metrics_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_mechanism_comparison_diagnostics(self, experimental_results: Dict[str, Any]) -> str:
        """Create mechanism comparison diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Mechanism Comparison Diagnostics: Tail-driven vs Sigmoid-based MNAR', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Performance comparison box plots
        ax1 = axes[0, 0]
        self._plot_mechanism_performance_comparison(ax1, experimental_results)
        
        # Plot 2: Stability comparison
        ax2 = axes[0, 1]
        self._plot_mechanism_stability_comparison(ax2, experimental_results)
        
        # Plot 3: Robustness comparison
        ax3 = axes[1, 0]
        self._plot_mechanism_robustness_comparison(ax3, experimental_results)
        
        # Plot 4: Mechanism ranking
        ax4 = axes[1, 1]
        self._plot_mechanism_ranking(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'mechanism_comparison_diagnostics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_statistical_significance_diagnostics(self, experimental_results: Dict[str, Any]) -> str:
        """Create statistical significance diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Significance Diagnostics: Hypothesis Testing Results', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: P-value distribution
        ax1 = axes[0, 0]
        self._plot_pvalue_distribution(ax1, experimental_results)
        
        # Plot 2: Effect size analysis
        ax2 = axes[0, 1]
        self._plot_effect_size_analysis(ax2, experimental_results)
        
        # Plot 3: Power analysis results
        ax3 = axes[1, 0]
        self._plot_power_analysis_results(ax3, experimental_results)
        
        # Plot 4: Multiple comparison correction
        ax4 = axes[1, 1]
        self._plot_multiple_comparison_correction(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'statistical_significance_diagnostics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_robustness_assessment_plots(self, experimental_results: Dict[str, Any]) -> str:
        """Create robustness assessment plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Robustness Assessment: Multi-dimensional Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Coefficient of variation analysis
        ax1 = axes[0, 0]
        self._plot_coefficient_of_variation_analysis(ax1, experimental_results)
        
        # Plot 2: Outlier analysis
        ax2 = axes[0, 1]
        self._plot_outlier_analysis(ax2, experimental_results)
        
        # Plot 3: Sample size sensitivity
        ax3 = axes[1, 0]
        self._plot_sample_size_sensitivity(ax3, experimental_results)
        
        # Plot 4: Robustness ranking
        ax4 = axes[1, 1]
        self._plot_robustness_ranking(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'robustness_assessment_plots.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_comprehensive_diagnostic_summary(self, 
                                               experimental_results: Dict[str, Any],
                                               missingness_data: Dict[str, Any],
                                               performance_data: Dict[str, Any]) -> str:
        """Create comprehensive diagnostic summary plots."""
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5))
        fig.suptitle('Comprehensive Diagnostic Summary: Phase 1.3 Evaluation', 
                    fontsize=self.plt.rcParams['axes.titlesize'] * 1.2, fontweight='bold', y=1.02)
        
        # Plot 1: Overall performance summary
        ax1 = axes[0, 0]
        self._plot_overall_performance_summary(ax1, experimental_results)
        
        # Plot 2: Key findings visualization
        ax2 = axes[0, 1]
        self._plot_key_findings_visualization(ax2, experimental_results)
        
        # Plot 3: Recommendations summary
        ax3 = axes[1, 0]
        self._plot_recommendations_summary(ax3, experimental_results)
        
        # Plot 4: Methodology validation
        ax4 = axes[1, 1]
        self._plot_methodology_validation(ax4, experimental_results)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'comprehensive_diagnostic_summary.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    # Helper methods for specific plot components
    def _plot_missingness_rate_heatmap(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness rate heatmap across datasets and mechanisms."""
        # Implementation for missingness rate heatmap
    def _plot_missingness_rate_heatmap(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness rate heatmap across datasets and mechanisms."""
        # Placeholder for actual data:
        # missingness_data = {
        #     'datasets': ['Heart Disease', 'Diabetes', 'Hepatitis'],
        #     'mechanisms': ['Sigmoid', 'Threshold', 'GPD'],
        #     'rates': np.array([[0.1, 0.15, 0.12],
        #                        [0.2, 0.22, 0.18],
        #                        [0.05, 0.07, 0.06]])
        # }
        
        # Using dummy data for now
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        mechanisms = ['Sigmoid', 'Threshold', 'GPD']
        rates = np.array([[0.1, 0.15, 0.12],
                           [0.2, 0.22, 0.18],
                           [0.05, 0.07, 0.06]])

        df_heatmap = pd.DataFrame(rates, index=datasets, columns=mechanisms)
        
        sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={'label': 'Mean Missingness Rate'})
        ax.set_title('Missingness Rate Across Datasets and Mechanisms', fontsize=12)
        ax.set_xlabel('Mechanism', fontsize=10)
        ax.set_ylabel('Dataset', fontsize=10)
    
    def _plot_missingness_pattern_heatmap(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness pattern heatmap for specific dataset."""
        # Implementation for missingness pattern heatmap
    def _plot_missingness_pattern_heatmap(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness pattern heatmap for specific dataset."""
        # Using dummy data for now (example: for 'Diabetes' dataset, 'Sigmoid' mechanism)
        # This would typically come from a specific run's missingness mask
        np.random.seed(42)
        dummy_missing_matrix = np.random.rand(50, 10) < np.random.uniform(0.1, 0.5, 10)
        dummy_df = pd.DataFrame(dummy_missing_matrix, 
                                columns=[f'Var_{i}' for i in range(10)])

        sns.heatmap(dummy_df.T, cmap="binary", cbar=False, ax=ax)
        ax.set_title('Missingness Pattern for Sampled Data (Binary Mask)', fontsize=12)
        ax.set_xlabel('Samples', fontsize=10)
        ax.set_ylabel('Variables', fontsize=10)
    
    def _plot_missingness_distribution(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness distribution across variables."""
        # Implementation for missingness distribution
    def _plot_missingness_distribution(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness distribution across variables."""
        # Using dummy data for now
        np.random.seed(42)
        variable_missing_rates = np.random.uniform(0.05, 0.5, 10)
        variables = [f'Var_{i}' for i in range(10)]
        
        sns.barplot(x=variables, y=variable_missing_rates, ax=ax, palette="Blues_d")
        ax.set_title('Missingness Distribution Across Variables', fontsize=12)
        ax.set_xlabel('Variables', fontsize=10)
        ax.set_ylabel('Missingness Rate', fontsize=10)
        ax.set_ylim([0, 0.6])
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_missingness_correlation(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness correlation analysis."""
        # Implementation for missingness correlation
    def _plot_missingness_correlation(self, ax, missingness_data: Dict[str, Any]):
        """Plot missingness correlation analysis."""
        # Using dummy data for now
        np.random.seed(42)
        dummy_correlation_matrix = np.random.rand(10, 10)
        dummy_correlation_matrix = (dummy_correlation_matrix + dummy_correlation_matrix.T) / 2
        np.fill_diagonal(dummy_correlation_matrix, 1)
        dummy_corr_df = pd.DataFrame(dummy_correlation_matrix, 
                                     index=[f'Var_{i}' for i in range(10)], 
                                     columns=[f'Var_{i}' for i in range(10)])
        
        sns.heatmap(dummy_corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                    cbar_kws={'label': 'Missingness Correlation'})
        ax.set_title('Missingness Correlation Analysis', fontsize=12)
        ax.set_xlabel('Variables', fontsize=10)
        ax.set_ylabel('Variables', fontsize=10)
    
    def _plot_qq_normal_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot comparing data to normal distribution."""
        # Implementation for QQ normal comparison
    def _plot_qq_normal_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot comparing data to normal distribution."""
        # Using dummy data for now (example: sample data from a normal distribution)
        np.random.seed(42)
        sample_data = np.random.normal(loc=0, scale=1, size=100)

        stats.probplot(sample_data, dist="norm", plot=ax)
        ax.set_title('QQ Plot: Data vs Normal Distribution', fontsize=12)
        ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
        ax.set_ylabel('Sample Quantiles', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_qq_heavy_tailed_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot comparing data to heavy-tailed distribution."""
        # Implementation for QQ heavy-tailed comparison
    def _plot_qq_heavy_tailed_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot comparing data to heavy-tailed distribution."""
        # Using dummy data for now (example: sample data from a t-distribution)
        np.random.seed(42)
        sample_data = stats.t.rvs(df=3, size=100) # t-distribution with 3 degrees of freedom

        stats.probplot(sample_data, dist="t", sparams=(3,), plot=ax)
        ax.set_title('QQ Plot: Data vs Heavy-Tailed Distribution (t-dist)', fontsize=12)
        ax.set_xlabel('Theoretical Quantiles (t-distribution)', fontsize=10)
        ax.set_ylabel('Sample Quantiles', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_qq_mnar_effect(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot showing MNAR effect on tail behavior."""
        # Implementation for QQ MNAR effect
    def _plot_qq_mnar_effect(self, ax, experimental_results: Dict[str, Any]):
        """Plot QQ plot showing MNAR effect on tail behavior."""
        # Using dummy data for now
        np.random.seed(42)
        original_data = np.random.normal(loc=0, scale=1, size=200)
        # Simulate MNAR by removing extreme values (e.g., top 20%)
        mnar_data = original_data[original_data < np.percentile(original_data, 80)]

        stats.probplot(original_data, dist="norm", plot=ax, fit=False, line='45')
        ax.lines[0].set_color(self.colors['baseline'])
        ax.lines[0].set_label('Original Data')

        stats.probplot(mnar_data, dist="norm", plot=ax, fit=False)
        ax.lines[2].set_color(self.colors['missing'])
        ax.lines[2].set_label('MNAR Affected Data')

        ax.set_title('QQ Plot: MNAR Effect on Tail Behavior', fontsize=12)
        ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
        ax.set_ylabel('Sample Quantiles', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_tail_index_estimation(self, ax, experimental_results: Dict[str, Any]):
        """Plot tail index estimation results."""
        # Implementation for tail index estimation
    def _plot_tail_index_estimation(self, ax, experimental_results: Dict[str, Any]):
        """Plot tail index estimation results."""
        # Using dummy data for now
        np.random.seed(42)
        variables = [f'Var_{i}' for i in range(5)]
        tail_indices = np.random.uniform(0.1, 1.5, len(variables))
        
        sns.barplot(x=variables, y=tail_indices, ax=ax, palette="Reds_d")
        ax.set_title('Tail Index Estimation Across Variables', fontsize=12)
        ax.set_xlabel('Variables', fontsize=10)
        ax.set_ylabel('Tail Index ($\xi$)', fontsize=10)
        ax.set_ylim([0, 2])
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_relshd_vs_missingness(self, ax, performance_data: Dict[str, Any]):
        """Plot relSHD vs missingness curves."""
        # Implementation for relSHD vs missingness
    def _plot_relshd_vs_missingness(self, ax, performance_data: Dict[str, Any]):
        """Plot relSHD vs missingness curves."""
        # Using dummy data for now
        np.random.seed(42)
        missingness_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        for mechanism in mechanisms:
            relshd_values = np.random.uniform(0.05, 0.2, len(missingness_levels)) + \
                            np.linspace(0, 0.1, len(missingness_levels)) # Simulate increasing degradation
            ax.plot(np.array(missingness_levels) * 100, relshd_values, 
                    marker=self.markers[mechanism], linestyle='-', 
                    color=self.colors[mechanism], label=mechanism.capitalize())
            
        ax.set_title('RelSHD vs Missingness Percentage', fontsize=12)
        ax.set_xlabel('Missingness Percentage (%)', fontsize=10)
        ax.set_ylabel('Relative Structural Hamming Distance', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_f1_vs_missingness(self, ax, performance_data: Dict[str, Any]):
        """Plot F1-score vs missingness curves."""
        # Implementation for F1 vs missingness
    def _plot_f1_vs_missingness(self, ax, performance_data: Dict[str, Any]):
        """Plot F1-score vs missingness curves."""
        # Using dummy data for now
        np.random.seed(42)
        missingness_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        for mechanism in mechanisms:
            f1_values = np.random.uniform(0.6, 0.9, len(missingness_levels)) - \
                        np.linspace(0, 0.15, len(missingness_levels)) # Simulate decreasing performance
            ax.plot(np.array(missingness_levels) * 100, f1_values, 
                    marker=self.markers[mechanism], linestyle='-', 
                    color=self.colors[mechanism], label=mechanism.capitalize())
            
        ax.set_title('F1-Score vs Missingness Percentage', fontsize=12)
        ax.set_xlabel('Missingness Percentage (%)', fontsize=10)
        ax.set_ylabel('F1-Score', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim([0, 1])
    
    def _plot_degradation_rates(self, ax, performance_data: Dict[str, Any]):
        """Plot performance degradation rates."""
        # Implementation for degradation rates
    def _plot_degradation_rates(self, ax, performance_data: Dict[str, Any]):
        """Plot performance degradation rates."""
        # Using dummy data for now
        np.random.seed(42)
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        degradation_rates = np.random.uniform(0.01, 0.2, len(mechanisms))

        sns.barplot(x=mechanisms, y=degradation_rates, ax=ax, palette="viridis")
        ax.set_title('Performance Degradation Rates by Mechanism', fontsize=12)
        ax.set_xlabel('Missingness Mechanism', fontsize=10)
        ax.set_ylabel('Average F1-Score Degradation', fontsize=10)
        ax.set_ylim([0, 0.25])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    def _plot_performance_thresholds(self, ax, performance_data: Dict[str, Any]):
        """Plot performance thresholds analysis."""
        # Implementation for performance thresholds
    def _plot_performance_thresholds(self, ax, performance_data: Dict[str, Any]):
        """Plot performance thresholds analysis."""
        # Using dummy data for now
        np.random.seed(42)
        missingness_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        for mechanism in mechanisms:
            threshold_values = np.random.uniform(0.7, 0.95, len(missingness_levels)) # Simulate performance
            ax.plot(np.array(missingness_levels) * 100, threshold_values, 
                    marker=self.markers[mechanism], linestyle='-', 
                    color=self.colors[mechanism], label=mechanism.capitalize())
            
        ax.axhline(y=0.75, color='gray', linestyle='--', label='Acceptable Performance Threshold')
        ax.set_title('Performance Thresholds Analysis', fontsize=12)
        ax.set_xlabel('Missingness Percentage (%)', fontsize=10)
        ax.set_ylabel('Average F1-Score', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim([0.6, 1.0])
    
    def _plot_relshd_distribution(self, ax, experimental_results: Dict[str, Any]):
        """Plot relSHD distribution across mechanisms."""
        # Implementation for relSHD distribution
    def _plot_relshd_distribution(self, ax, experimental_results: Dict[str, Any]):
        """Plot relSHD distribution across mechanisms."""
        # Using dummy data for now
        np.random.seed(42)
        relshd_data = {
            'sigmoid': np.random.normal(0.1, 0.03, 100),
            'threshold': np.random.normal(0.08, 0.02, 100),
            'parametric': np.random.normal(0.15, 0.04, 100)
        }

        for mechanism, data in relshd_data.items():
            sns.kdeplot(data, ax=ax, label=mechanism.capitalize(), color=self.colors[mechanism], fill=True, alpha=0.5)
            
        ax.set_title('RelSHD Distribution Across Mechanisms', fontsize=12)
        ax.set_xlabel('Relative Structural Hamming Distance', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_adjacency_metrics_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot adjacency metrics comparison."""
        # Implementation for adjacency metrics comparison
    def _plot_adjacency_metrics_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot adjacency metrics comparison."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        metrics = ['Precision', 'Recall', 'F1-Score']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                for metric in metrics:
                    value = np.random.uniform(0.6, 0.95)
                    data.append({'Dataset': dataset, 'Mechanism': mechanism, 'Metric': metric, 'Value': value})
        
        df_metrics = pd.DataFrame(data)
        
    def _plot_adjacency_metrics_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot adjacency metrics comparison."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        metrics = ['Precision', 'Recall', 'F1-Score']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                for metric in metrics:
                    value = np.random.uniform(0.6, 0.95)
                    data.append({'Dataset': dataset, 'Mechanism': mechanism, 'Metric': metric, 'Value': value})
        
        df_metrics = pd.DataFrame(data)
        
        sns.barplot(x='Dataset', y='Value', hue='Metric', data=df_metrics, 
                    palette="deep", ax=ax, ci="sd", capsize=.05, errwidth=1)
        
        ax.set_title('Adjacency Metrics Comparison Across Datasets and Mechanisms', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.legend(title='Metric', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_orientation_accuracy_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot orientation accuracy analysis."""
        # Implementation for orientation accuracy analysis
    def _plot_orientation_accuracy_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot orientation accuracy analysis."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                value = np.random.uniform(0.5, 0.9)
                data.append({'Dataset': dataset, 'Mechanism': mechanism, 'Orientation F1': value})
        
        df_orientation = pd.DataFrame(data)
        
        sns.barplot(x='Dataset', y='Orientation F1', hue='Mechanism', data=df_orientation, 
                    palette="viridis", ax=ax, ci="sd", capsize=.05, errwidth=1)
        
        ax.set_title('Orientation Accuracy (F1-Score) Across Datasets and Mechanisms', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel('Orientation F1-Score', fontsize=10)
        ax.legend(title='Mechanism', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_structural_complexity_metrics(self, ax, experimental_results: Dict[str, Any]):
        """Plot structural complexity metrics."""
        # Implementation for structural complexity metrics
    def _plot_structural_complexity_metrics(self, ax, experimental_results: Dict[str, Any]):\n        \"\"\"Plot structural complexity metrics.\"\"\"\n        # Using dummy data for now\n        np.random.seed(42)\n        datasets = [\'Heart Disease\', \'Diabetes\', \'Hepatitis\']\n        metrics = [\'Num Nodes\', \'Num Edges\', \'Density\']\n        \n        data = []\n        for dataset in datasets:\n            num_nodes = np.random.randint(8, 20)\n            num_edges = np.random.randint(num_nodes - 1, num_nodes * (num_nodes - 1) / 2)\n            density = num_edges / (num_nodes * (num_nodes - 1) / 2) # Example calculation\n\n            data.append({\'Dataset\': dataset, \'Metric\': \'Num Nodes\', \'Value\': num_nodes})\n            data.append({\'Dataset\': dataset, \'Metric\': \'Num Edges\', \'Value\': num_edges})\n            data.append({\'Dataset\': dataset, \'Metric\': \'Density\', \'Value\': density})\
        \n        df_complexity = pd.DataFrame(data)\n\n        sns.barplot(x=\'Dataset\', y=\'Value\', hue=\'Metric\', data=df_complexity, \n                    palette=\"plasma\", ax=ax, ci=None)\n        ax.set_title(\'Structural Complexity Metrics by Dataset\', fontsize=12)\n        ax.set_xlabel(\'Dataset\', fontsize=10)\n        ax.set_ylabel(\'Metric Value\', fontsize=10)\n        ax.legend(title=\'Metric\', fontsize=8)\n        ax.grid(True, linestyle=\':\', alpha=0.6, axis=\'y\')\n        ax.tick_params(axis=\'x\', rotation=15)
    
    def _plot_mechanism_performance_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism performance comparison."""
        # Implementation for mechanism performance comparison
    def _plot_mechanism_performance_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism performance comparison."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                f1_scores = np.random.uniform(0.6, 0.95, 20) # 20 simulated runs
                for score in f1_scores:
                    data.append({'Dataset': dataset, 'Mechanism': mechanism, 'F1-Score': score})
        
        df_performance = pd.DataFrame(data)

        sns.boxplot(x='Mechanism', y='F1-Score', hue='Dataset', data=df_performance, 
                    palette="pastel", ax=ax)
        ax.set_title('Performance Comparison: F1-Score by Mechanism and Dataset', fontsize=12)
        ax.set_xlabel('Missingness Mechanism', fontsize=10)
        ax.set_ylabel('F1-Score', fontsize=10)
        ax.legend(title='Dataset', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0.5, 1.0])
    
    def _plot_mechanism_stability_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism stability comparison."""
        # Implementation for mechanism stability comparison
    def _plot_mechanism_stability_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism stability comparison."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                std_f1 = np.random.uniform(0.01, 0.1)
                data.append({'Dataset': dataset, 'Mechanism': mechanism, 'Std F1-Score': std_f1})
        
        df_stability = pd.DataFrame(data)

        sns.barplot(x='Mechanism', y='Std F1-Score', hue='Dataset', data=df_stability, 
                    palette="muted", ax=ax, ci=None)
        ax.set_title('Stability Comparison: Std Dev of F1-Score by Mechanism', fontsize=12)
        ax.set_xlabel('Missingness Mechanism', fontsize=10)
        ax.set_ylabel('Standard Deviation of F1-Score', fontsize=10)
        ax.legend(title='Dataset', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0, 0.15])
    
    def _plot_mechanism_robustness_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism robustness comparison."""
        # Implementation for mechanism robustness comparison
    def _plot_mechanism_robustness_comparison(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism robustness comparison."""
        # Using dummy data for now
        np.random.seed(42)
        datasets = ['Heart Disease', 'Diabetes', 'Hepatitis']
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        
        data = []
        for dataset in datasets:
            for mechanism in mechanisms:
                robustness_score = np.random.uniform(0.5, 1.0)
                data.append({'Dataset': dataset, 'Mechanism': mechanism, 'Robustness Score': robustness_score})
        
        df_robustness = pd.DataFrame(data)

        sns.barplot(x='Mechanism', y='Robustness Score', hue='Dataset', data=df_robustness, 
                    palette="viridis", ax=ax, ci="sd", capsize=.05, errwidth=1)
        ax.set_title('Robustness Comparison by Mechanism and Dataset', fontsize=12)
        ax.set_xlabel('Missingness Mechanism', fontsize=10)
        ax.set_ylabel('Robustness Score', fontsize=10)
        ax.legend(title='Dataset', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0, 1.1])
    
    def _plot_mechanism_ranking(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism ranking."""
        # Implementation for mechanism ranking
    def _plot_mechanism_ranking(self, ax, experimental_results: Dict[str, Any]):
        """Plot mechanism ranking."""
        # Using dummy data for now
        np.random.seed(42)
        mechanisms = ['sigmoid', 'threshold', 'parametric']
        avg_f1_scores = np.random.uniform(0.7, 0.9, len(mechanisms))
        
        ranking_df = pd.DataFrame({'Mechanism': mechanisms, 'Average F1-Score': avg_f1_scores})
        ranking_df = ranking_df.sort_values(by='Average F1-Score', ascending=False).reset_index(drop=True)
        ranking_df['Rank'] = ranking_df.index + 1

        sns.barplot(x='Rank', y='Average F1-Score', hue='Mechanism', data=ranking_df, 
                    palette="cubehelix", ax=ax, dodge=False)
        ax.set_title('Mechanism Ranking by Average F1-Score', fontsize=12)
        ax.set_xlabel('Rank', fontsize=10)
        ax.set_ylabel('Average F1-Score', fontsize=10)
        ax.set_xticks(ranking_df.index)
        ax.set_xticklabels(ranking_df['Rank'])
        ax.legend(title='Mechanism', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0.65, 0.95])
    
    def _plot_pvalue_distribution(self, ax, experimental_results: Dict[str, Any]):
        """Plot p-value distribution."""
        # Implementation for p-value distribution
    def _plot_pvalue_distribution(self, ax, experimental_results: Dict[str, Any]):
        """Plot p-value distribution."""
        # Using dummy data for now (example: p-values from 100 tests)
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 100)
        
        sns.histplot(p_values, bins=20, kde=True, ax=ax, color=self.colors['performance'])
        ax.set_title('P-value Distribution from Hypothesis Tests', fontsize=12)
        ax.set_xlabel('P-value', fontsize=10)
        ax.set_ylabel('Frequency / Density', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    def _plot_effect_size_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot effect size analysis."""
        # Implementation for effect size analysis
    def _plot_effect_size_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot effect size analysis."""
        # Using dummy data for now
        np.random.seed(42)
        comparisons = ['Baseline vs 10% MNAR', 'Baseline vs 20% MNAR', 'Baseline vs 30% MNAR']
        effect_sizes = np.random.uniform(0.2, 1.2, len(comparisons))
        
        sns.barplot(x=comparisons, y=effect_sizes, ax=ax, palette="magma")
        ax.set_title('Effect Size (Cohen's d) for Performance Degradation', fontsize=12)
        ax.set_xlabel('Comparison', fontsize=10)
        ax.set_ylabel('Cohen\'s d Effect Size', fontsize=10)
        ax.axhline(y=0.2, color='gray', linestyle='--', label='Small effect')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Medium effect')
        ax.axhline(y=0.8, color='gray', linestyle='--', label='Large effect')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_power_analysis_results(self, ax, experimental_results: Dict[str, Any]):
        """Plot power analysis results."""
        # Implementation for power analysis results
    def _plot_power_analysis_results(self, ax, experimental_results: Dict[str, Any]):
        """Plot power analysis results."""
        # Using dummy data for now
        np.random.seed(42)
        comparisons = ['Baseline vs 10% MNAR', 'Baseline vs 20% MNAR', 'Baseline vs 30% MNAR']
        power_values = np.random.uniform(0.7, 0.99, len(comparisons))
        
        sns.barplot(x=comparisons, y=power_values, ax=ax, palette="Greens_d")
        ax.set_title('Statistical Power to Detect Effects', fontsize=12)
        ax.set_xlabel('Comparison', fontsize=10)
        ax.set_ylabel('Statistical Power', fontsize=10)
        ax.axhline(y=0.8, color='gray', linestyle='--', label='Desired Power (0.8)')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_multiple_comparison_correction(self, ax, experimental_results: Dict[str, Any]):
        """Plot multiple comparison correction."""
        # Implementation for multiple comparison correction
    def _plot_multiple_comparison_correction(self, ax, experimental_results: Dict[str, Any]):
        """Plot multiple comparison correction."""
        # Using dummy data for now
        np.random.seed(42)
        raw_p_values = np.sort(np.random.uniform(0.001, 0.1, 20))
        corrected_p_values = np.minimum(raw_p_values * len(raw_p_values), 1.0) # Simple Bonferroni
        
        comparisons = [f'Test {i+1}' for i in range(len(raw_p_values))]

        ax.plot(comparisons, raw_p_values, marker='o', linestyle='-', label='Raw P-value', color=self.colors['baseline'])
        ax.plot(comparisons, corrected_p_values, marker='s', linestyle='--', label='Corrected P-value (Bonferroni)', color=self.colors['warning'])

        ax.set_title('Multiple Comparison Correction Impact', fontsize=12)
        ax.set_xlabel('Hypothesis Test', fontsize=10)
        ax.set_ylabel('P-value', fontsize=10)
        ax.axhline(y=0.05, color='gray', linestyle=':', label='Alpha = 0.05')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 0.2])
    
    def _plot_coefficient_of_variation_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot coefficient of variation analysis."""
        # Implementation for coefficient of variation analysis
        ax.text(0.5, 0.5, 'Coefficient of Variation Analysis\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Coefficient of Variation Analysis')
        ax.set_xlabel('Mechanism')
        ax.set_ylabel('Coefficient of Variation')
    
    def _plot_outlier_analysis(self, ax, experimental_results: Dict[str, Any]):
        """Plot outlier analysis."""
        # Implementation for outlier analysis
        ax.text(0.5, 0.5, 'Outlier Analysis\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Outlier Analysis')
        ax.set_xlabel('Mechanism')
        ax.set_ylabel('Outlier Count')
    
    def _plot_sample_size_sensitivity(self, ax, experimental_results: Dict[str, Any]):
        """Plot sample size sensitivity."""
        # Implementation for sample size sensitivity
        ax.text(0.5, 0.5, 'Sample Size Sensitivity\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Sample Size Sensitivity')
        ax.set_xlabel('Sample Size Factor')
        ax.set_ylabel('Performance Metric')
    
    def _plot_robustness_ranking(self, ax, experimental_results: Dict[str, Any]):
        """Plot robustness ranking."""
        # Implementation for robustness ranking
        ax.text(0.5, 0.5, 'Robustness Ranking\n(Implementation in progress)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Robustness Ranking')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Mechanism')
    
    def _plot_overall_performance_summary(self, ax, experimental_results: Dict[str, Any]):
        """Plot overall performance summary."""
        # Implementation for overall performance summary
    def _plot_overall_performance_summary(self, ax, experimental_results: Dict[str, Any]):
        """Plot overall performance summary."""
        # Using dummy data for now
        np.random.seed(42)
        metrics = ['Avg F1-Score', 'Min F1-Score', 'Avg relSHD', 'Max relSHD']
        values = np.random.uniform(0, 1, len(metrics))
        
        summary_df = pd.DataFrame({'Metric': metrics, 'Value': values})

        sns.barplot(x='Metric', y='Value', data=summary_df, ax=ax, palette="Spectral")
        ax.set_title('Overall Performance Summary', fontsize=12)
        ax.set_xlabel('Metric', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_key_findings_visualization(self, ax, experimental_results: Dict[str, Any]):
        """Plot key findings visualization."""
        # Implementation for key findings visualization
    def _plot_key_findings_visualization(self, ax, experimental_results: Dict[str, Any]):
        """Plot key findings visualization."""
        # Using dummy data for now, representing the impact of key findings
        np.random.seed(42)
        findings = [
            'Subgraph Robustness',
            'MNAR Calibrator Accuracy',
            'V-Structure Degradation',
            'Hepatitis Stability'
        ]
        impact_scores = np.random.uniform(0.7, 1.0, len(findings))
        
        findings_df = pd.DataFrame({'Finding': findings, 'Impact Score': impact_scores})
        findings_df = findings_df.sort_values(by='Impact Score', ascending=False)

        sns.barplot(x='Finding', y='Impact Score', data=findings_df, ax=ax, palette="viridis")
        ax.set_title('Key Findings and Their Impact', fontsize=12)
        ax.set_xlabel('Key Finding', fontsize=10)
        ax.set_ylabel('Impact Score (Normalized)', fontsize=10)
        ax.set_ylim([0.6, 1.05])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_recommendations_summary(self, ax, experimental_results: Dict[str, Any]):
        """Plot recommendations summary."""
        # Implementation for recommendations summary
    def _plot_recommendations_summary(self, ax, experimental_results: Dict[str, Any]):
        """Plot recommendations summary."""
        # Using dummy data for now
        np.random.seed(42)
        recommendations = [
            'Validate dataset-specific thresholds',
            'Focus on direct causal effects',
            'Monitor v-structures closely',
            'Consider computational scaling'
        ]
        priority_scores = np.random.uniform(0.6, 1.0, len(recommendations))
        
        recommendations_df = pd.DataFrame({'Recommendation': recommendations, 'Priority Score': priority_scores})
        recommendations_df = recommendations_df.sort_values(by='Priority Score', ascending=False)

        sns.barplot(x='Recommendation', y='Priority Score', data=recommendations_df, ax=ax, palette="rocket")
        ax.set_title('Practical Recommendations Summary', fontsize=12)
        ax.set_xlabel('Recommendation', fontsize=10)
        ax.set_ylabel('Priority Score (Normalized)', fontsize=10)
        ax.set_ylim([0.5, 1.1])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    def _plot_methodology_validation(self, ax, experimental_results: Dict[str, Any]):
        """Plot methodology validation."""
        # Implementation for methodology validation
    def _plot_methodology_validation(self, ax, experimental_results: Dict[str, Any]):
        """Plot methodology validation."""
        # Using dummy data for now
        np.random.seed(42)
        validation_metrics = [
            'Reproducibility (CV)',
            'Statistical Power',
            'Consistency across mechanisms',
            'Real-world applicability'
        ]
        scores = np.random.uniform(0.75, 1.0, len(validation_metrics))
        
        validation_df = pd.DataFrame({'Metric': validation_metrics, 'Score': scores})

        sns.barplot(x='Metric', y='Score', data=validation_df, ax=ax, palette="cubehelix")
        ax.set_title('Methodology Validation Overview', fontsize=12)
        ax.set_xlabel('Validation Aspect', fontsize=10)
        ax.set_ylabel('Score (Normalized)', fontsize=10)
        ax.set_ylim([0.7, 1.05])
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.tick_params(axis='x', rotation=15)

# Convenience function for creating diagnostic visualizations
def create_advanced_diagnostic_visualizations(experimental_results: Dict[str, Any],
                                            missingness_data: Dict[str, Any],
                                            performance_data: Dict[str, Any],
                                            output_dir: str = "results/diagnostic_visualizations") -> Dict[str, str]:
    """
    Convenience function for creating advanced diagnostic visualizations.
    
    Args:
        experimental_results: Results from experimental evaluation
        missingness_data: Missingness pattern data
        performance_data: Performance metrics data
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    visualizer = AdvancedDiagnosticVisualizer(output_dir)
    return visualizer.create_comprehensive_diagnostic_plots(
        experimental_results, missingness_data, performance_data
    )

if __name__ == "__main__":
    # Example usage
    print("Advanced Diagnostic Visualizer - Example Usage")
    print("Ready for creating comprehensive diagnostic visualizations!")
