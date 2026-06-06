"""
Subgraph Visualization Tools
===========================

This module provides comprehensive visualization tools for subgraph analysis,
including robustness comparisons, age variable patterns, and structural degradation.

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

        # Set up plot parameters for professional look
        plt.rcParams['figure.figsize'] = (10, 8)  # Slightly smaller for two-column
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6


class SubgraphVisualizer:
    """
    Comprehensive visualization tools for subgraph analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = {
            'primary': '#0D47A1',
            'secondary': '#1976D2',
            'accent': '#FFC107',
            'success': '#4CAF50',
            'warning': '#F44336',
            'diabetes': '#2196F3',
            'heart_disease': '#FF5722',
            'hepatitis': '#4CAF50'
        }
    
    def plot_subgraph_vs_full_robustness(self, df: pd.DataFrame,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of subgraph vs full graph robustness.
        
        Args:
            df: DataFrame with analysis results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        datasets = df['dataset'].unique()
        missing_pcts = sorted(df['missing_percentage'].unique())
        
        # Aggregate by dataset and missing percentage
        agg_df = df.groupby(['dataset', 'missing_percentage']).agg({
            'edge_f1_score': 'mean',
            'subgraph_f1_score': 'mean',
            'relative_structural_hamming_distance': 'mean',
            'subgraph_rel_shd': 'mean'
        }).reset_index()
        
        # Plot 1: F1-Score Comparison
        ax1 = axes[0, 0]
        for dataset in datasets:
            dataset_data = agg_df[agg_df['dataset'] == dataset]
            # Use a more distinct color for full vs subgraph
            color_idx = list(datasets).index(dataset)
            full_color = sns.color_palette("dark", len(datasets))[color_idx]
            sub_color = sns.color_palette("pastel", len(datasets))[color_idx]

            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['edge_f1_score'],
                    marker='o', linestyle='--', alpha=0.7,
                    label=f'{dataset} (Full)', color=full_color)
            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['subgraph_f1_score'],
                    marker='s', linestyle='-',
                    label=f'{dataset} (Subgraph)', color=sub_color)
        
        ax1.set_xlabel('Missing Data Percentage (%)')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('(A) F1-Score: Subgraph vs Full Graph', loc='left', fontweight='bold')
        ax1.legend(loc='lower left', frameon=True, shadow=False)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_ylim([0, 1])
        
        # Plot 2: relSHD Comparison
        ax2 = axes[0, 1]
        for dataset in datasets:
            dataset_data = agg_df[agg_df['dataset'] == dataset]
            # Use a more distinct color for full vs subgraph
            color_idx = list(datasets).index(dataset)
            full_color = sns.color_palette("dark", len(datasets))[color_idx]
            sub_color = sns.color_palette("pastel", len(datasets))[color_idx]

            ax2.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['relative_structural_hamming_distance'],
                    marker='o', linestyle='--', alpha=0.7,
                    label=f'{dataset} (Full)', color=full_color)
            ax2.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['subgraph_rel_shd'],
                    marker='s', linestyle='-',
                    label=f'{dataset} (Subgraph)', color=sub_color)
        
        ax2.set_xlabel('Missing Data Percentage (%)', fontsize=12)
        ax2.set_ylabel('Relative SHD', fontsize=12)
        ax2.set_title('(B) Structural Similarity: Subgraph vs Full Graph', loc='left', fontweight='bold')
        ax2.legend(loc='upper left', frameon=True, shadow=False)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 3: Robustness Ratio
        ax3 = axes[1, 0]
        for dataset in datasets:
            dataset_data = agg_df[agg_df['dataset'] == dataset]
            robustness_ratio = (dataset_data['subgraph_f1_score'] / 
                              dataset_data['edge_f1_score']).replace([np.inf, -np.inf], np.nan)
            color_idx = list(datasets).index(dataset)
            line_color = sns.color_palette("deep", len(datasets))[color_idx]
            ax3.plot(dataset_data['missing_percentage'] * 100,
                    robustness_ratio,
                    marker='o', label=dataset, color=line_color)
        
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Robustness')
        ax3.set_xlabel('Missing Data Percentage (%)')
        ax3.set_ylabel('Robustness Ratio (Subgraph F1 / Full F1)')
        ax3.set_title('(C) Subgraph Robustness Ratio', loc='left', fontweight='bold')
        ax3.legend(loc='upper left', frameon=True, shadow=False)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.set_ylim([0.5, 7.0]) # Adjusted based on expected range
        
        # Plot 4: Incoming Edge Counts
        ax4 = axes[1, 1]
        for dataset in datasets:
            dataset_data = df[df['dataset'] == dataset]
            edge_counts = dataset_data.groupby('missing_percentage')['correct_incoming_edges'].mean()
            color_idx = list(datasets).index(dataset)
            line_color = sns.color_palette("dark", len(datasets))[color_idx]
            ax4.plot(edge_counts.index * 100,
                    edge_counts.values,
                    marker='o', label=dataset, color=line_color)
        
        ax4.set_xlabel('Missing Data Percentage (%)')
        ax4.set_ylabel('Number of Correct Incoming Edges')
        ax4.set_title('(D) Incoming Edge Preservation', loc='left', fontweight='bold')
        ax4.legend(loc='lower left', frameon=True, shadow=False)
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_age_variable_error_analysis(self, df: pd.DataFrame,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error analysis around age variable.
        
        Args:
            df: DataFrame with analysis results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Filter datasets with age variable
        age_datasets = df[df['age_variable_found'] == True]['dataset'].unique()
        
        if len(age_datasets) == 0:
            logger.warning("No datasets with age variable found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract age error data
        age_error_data = []
        for _, row in df.iterrows():
            if row['age_variable_found'] and isinstance(row.get('age_errors'), dict):
                errors = row['age_errors']
                age_error_data.append({
                    'dataset': row['dataset'],
                    'missing_percentage': row['missing_percentage'],
                    'false_positives': errors.get('false_positives_involving', 0),
                    'false_negatives': errors.get('false_negatives_involving', 0),
                    'total_errors': errors.get('total_errors_involving', 0)
                })
        
        if not age_error_data:
            logger.warning("No age error data found")
            return None
        
        age_df = pd.DataFrame(age_error_data)
        agg_age_df = age_df.groupby(['dataset', 'missing_percentage']).agg({
            'false_positives': 'mean',
            'false_negatives': 'mean',
            'total_errors': 'mean'
        }).reset_index()
        
        # Plot 1: Total Errors Around Age
        ax1 = axes[0, 0]
        for dataset in age_datasets:
            dataset_data = agg_age_df[agg_age_df['dataset'] == dataset]
            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['total_errors'],
                    marker='o', linewidth=2, label=dataset)
        
        ax1.set_xlabel('Missing Data Percentage (%)')
        ax1.set_ylabel('Total Errors Involving Age')
        ax1.set_title('(A) Error Clustering Around Age Variable', loc='left', fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, shadow=False)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2: False Positives vs False Negatives
        ax2 = axes[0, 1]
        for dataset in age_datasets:
            dataset_data = agg_age_df[agg_age_df['dataset'] == dataset]
            ax2.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['false_positives'],
                    marker='o', linewidth=2, label=f'{dataset} (FP)',
                    linestyle='--')
            ax2.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['false_negatives'],
                    marker='s', linewidth=2, label=f'{dataset} (FN)',
                    linestyle='-')
        
        ax2.set_xlabel('Missing Data Percentage (%)')
        ax2.set_ylabel('Error Count')
        ax2.set_title('(B) False Positives vs False Negatives Around Age', loc='left', fontweight='bold')
        ax2.legend(loc='upper left', frameon=True, shadow=False)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 3: Error Rate Heatmap
        ax3 = axes[1, 0]
        pivot_data = age_df.pivot_table(
            index='dataset',
            columns='missing_percentage',
            values='total_errors',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Mean Total Errors'})
        ax3.set_title('(C) Error Rate Heatmap: Age Variable', loc='left', fontweight='bold')
        ax3.set_xlabel('Missing Data Percentage')
        ax3.set_ylabel('Dataset')
        
        # Plot 4: Error Proportion
        ax4 = axes[1, 1]
        for dataset in age_datasets:
            dataset_data = df[df['dataset'] == dataset]
            total_errors = dataset_data.groupby('missing_percentage')['false_positive_edges'].sum() + \
                          dataset_data.groupby('missing_percentage')['false_negative_edges'].sum()
            age_errors = age_df[age_df['dataset'] == dataset].groupby('missing_percentage')['total_errors'].sum()
            error_proportion = (age_errors / total_errors).fillna(0)
            
            ax4.plot(error_proportion.index * 100,
                    error_proportion.values,
                    marker='o', linewidth=2, label=dataset)
        
        ax4.set_xlabel('Missing Data Percentage (%)')
        ax4.set_ylabel('Proportion of Errors Involving Age')
        ax4.set_title('(D) Age Variable Error Proportion', loc='left', fontweight='bold')
        ax4.legend(loc='upper left', frameon=True, shadow=False)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_incoming_edge_degradation(self, df: pd.DataFrame,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot incoming edge count degradation across missingness levels.
        
        Args:
            df: DataFrame with analysis results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] / 1.5)) # Adjust figsize for 2 subplots
        fig.suptitle('Incoming Edge Degradation Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        datasets = df['dataset'].unique()
        missing_pcts = sorted(df['missing_percentage'].unique())
        
        # Aggregate data
        agg_df = df.groupby(['dataset', 'missing_percentage']).agg({
            'true_incoming_edges': 'mean',
            'inferred_incoming_edges': 'mean',
            'correct_incoming_edges': 'mean',
            'false_positive_incoming_edges': 'mean',
            'false_negative_incoming_edges': 'mean'
        }).reset_index()
        
        # Plot 1: Edge Counts Over Missingness
        ax1 = axes[0]
        for dataset in datasets:
            dataset_data = agg_df[agg_df['dataset'] == dataset]
            # Use distinct colors for true, inferred, and correct edges
            color_map = sns.color_palette("deep", 3)

            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['true_incoming_edges'],
                    marker='o', linestyle='--', alpha=0.7,
                    label=f'{dataset} (True)', color=color_map[0])
            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['inferred_incoming_edges'],
                    marker='s', linestyle='-',
                    label=f'{dataset} (Inferred)', color=color_map[1])
            ax1.plot(dataset_data['missing_percentage'] * 100,
                    dataset_data['correct_incoming_edges'],
                    marker='^', linestyle='-.',
                    label=f'{dataset} (Correct)', color=color_map[2])
        
        ax1.set_xlabel('Missing Data Percentage (%)')
        ax1.set_ylabel('Number of Incoming Edges')
        ax1.set_title('(A) Incoming Edge Count Dynamics', loc='left', fontweight='bold')
        ax1.legend(loc='lower left', ncol=2, frameon=True, shadow=False)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2: Error Breakdown
        ax2 = axes[1]
        x = np.arange(len(missing_pcts))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            dataset_data = agg_df[agg_df['dataset'] == dataset]
            fp = dataset_data['false_positive_incoming_edges'].values
            fn = dataset_data['false_negative_incoming_edges'].values
            offset = (i - 1) * width
            
            # Use distinct colors for FP and FN
            fp_color = sns.color_palette("Reds_d", len(datasets))[i]
            fn_color = sns.color_palette("Blues_d", len(datasets))[i]

            ax2.bar(x + offset, fp, width, label=f'{dataset} (FP)', alpha=0.8, color=fp_color)
            ax2.bar(x + offset, fn, width, bottom=fp, label=f'{dataset} (FN)', alpha=0.8, color=fn_color)
        
        ax2.set_xlabel('Missing Data Percentage (%)')
        ax2.set_ylabel('Error Count')
        ax2.set_title('(B) Incoming Edge Error Breakdown', loc='left', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{p*100:.0f}%' for p in missing_pcts])
        ax2.legend(loc='upper left', ncol=2, frameon=True, shadow=False)
        ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_comprehensive_subgraph_analysis(self, df: pd.DataFrame,
                                            output_dir: str = "paper_conference_version") -> List[str]:
        """
        Generate all subgraph analysis visualizations.
        
        Args:
            df: DataFrame with analysis results
            output_dir: Directory to save figures
            
        Returns:
            List of saved figure paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_figures = []
        
        # Plot 1: Subgraph vs Full Robustness
        fig1 = self.plot_subgraph_vs_full_robustness(
            df,
            save_path=str(output_path / "subgraph_robustness_comparison.png")
        )
        saved_figures.append("subgraph_robustness_comparison.png")
        
        # Plot 2: Age Variable Error Analysis
        fig2 = self.plot_age_variable_error_analysis(
            df,
            save_path=str(output_path / "age_variable_error_analysis.png")
        )
        if fig2:
            saved_figures.append("age_variable_error_analysis.png")
        
        # Plot 3: Incoming Edge Degradation
        fig3 = self.plot_incoming_edge_degradation(
            df,
            save_path=str(output_path / "incoming_edge_degradation.png")
        )
        saved_figures.append("incoming_edge_degradation.png")
        
        logger.info(f"Generated {len(saved_figures)} visualization figures")
        
        return saved_figures

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'dataset': ['diabetes'] * 30 + ['heart_disease'] * 30,
        'missing_percentage': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] * 10,
        'edge_f1_score': np.random.uniform(0.7, 0.9, 60),
        'subgraph_f1_score': np.random.uniform(0.75, 0.95, 60),
        'relative_structural_hamming_distance': np.random.uniform(0.05, 0.2, 60),
        'subgraph_rel_shd': np.random.uniform(0.03, 0.15, 60),
        'correct_incoming_edges': np.random.randint(2, 5, 60),
        'age_variable_found': [True] * 30 + [True] * 30
    }
    
    df = pd.DataFrame(sample_data)
    
    visualizer = SubgraphVisualizer()
    visualizer.plot_comprehensive_subgraph_analysis(df)

