"""
Advanced Visualization Utilities
===============================

This module provides advanced visualization tools for causal discovery
experiments and results analysis.

Author: Anonymous (for review)
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CausalGraphVisualizer:
    """
    Advanced visualization tools for causal graphs.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_causal_graph(self, graph: nx.DiGraph, 
                         title: str = "Causal Graph",
                         layout: str = 'spring',
                         save_path: Optional[str] = None) -> None:
        """
        Plot a causal graph with enhanced visualization.
        
        Args:
            graph: NetworkX directed graph
            title: Plot title
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        else:
            pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, 
                              node_color='lightblue',
                              node_size=1000,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos,
                               font_size=10,
                               font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_graph_comparison(self, true_graph: nx.DiGraph, 
                            inferred_graph: nx.DiGraph,
                            title: str = "Graph Comparison",
                            save_path: Optional[str] = None) -> None:
        """
        Plot comparison between true and inferred graphs.
        
        Args:
            true_graph: True causal graph
            inferred_graph: Inferred causal graph
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot true graph
        pos = nx.spring_layout(true_graph, k=3, iterations=50)
        
        nx.draw_networkx_nodes(true_graph, pos, 
                              node_color='lightgreen',
                              node_size=1000,
                              alpha=0.8,
                              ax=ax1)
        nx.draw_networkx_edges(true_graph, pos,
                              edge_color='green',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              alpha=0.8,
                              ax=ax1)
        nx.draw_networkx_labels(true_graph, pos,
                               font_size=10,
                               font_weight='bold',
                               ax=ax1)
        
        ax1.set_title("True Causal Graph", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Plot inferred graph
        nx.draw_networkx_nodes(inferred_graph, pos, 
                              node_color='lightcoral',
                              node_size=1000,
                              alpha=0.8,
                              ax=ax2)
        nx.draw_networkx_edges(inferred_graph, pos,
                              edge_color='red',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              alpha=0.8,
                              ax=ax2)
        nx.draw_networkx_labels(inferred_graph, pos,
                               font_size=10,
                               font_weight='bold',
                               ax=ax2)
        
        ax2.set_title("Inferred Causal Graph", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_graph(self, graph: nx.DiGraph, 
                             title: str = "Interactive Causal Graph") -> None:
        """
        Create an interactive plotly visualization of the causal graph.
        
        Args:
            graph: NetworkX directed graph
            title: Plot title
        """
        # Get layout
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=2, color='gray'),
                               hoverinfo='none',
                               mode='lines')
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               marker=dict(size=20,
                                         color='lightblue',
                                         line=dict(width=2, color='black')))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title=title,
                                      titlefont_size=16,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      annotations=[ dict(
                                          text="Interactive Causal Graph",
                                          showarrow=False,
                                          xref="paper", yref="paper",
                                          x=0.005, y=-0.002,
                                          xanchor='left', yanchor='bottom',
                                          font=dict(color="black", size=12)
                                      )],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        fig.show()


class ResultsVisualizer:
    """
    Advanced visualization tools for experimental results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_performance_heatmap(self, results_df: pd.DataFrame,
                               metric: str = 'edge_f1_score',
                               save_path: Optional[str] = None) -> None:
        """
        Create a detailed heatmap of performance metrics.
        
        Args:
            results_df: DataFrame with experimental results
            metric: Metric to plot
            save_path: Path to save the plot
        """
        # Pivot data for heatmap
        pivot_data = results_df.pivot(index='dataset', 
                                    columns='missing_percentage', 
                                    values=metric)
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap with annotations
        sns.heatmap(pivot_data, 
                   annot=True, 
                   cmap='RdYlBu_r',
                   center=0.5,
                   cbar_kws={'label': f'{metric.replace("_", " ").title()}'},
                   fmt='.3f')
        
        plt.title(f'{metric.replace("_", " ").title()} Heatmap', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Missing Data Percentage', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_degradation(self, results_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot performance degradation across missing data percentages.
        
        Args:
            results_df: DataFrame with experimental results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['edge_f1_score', 'edge_precision', 'edge_recall', 'relative_structural_hamming_distance']
        metric_labels = ['F1 Score', 'Precision', 'Recall', 'Relative Structural Hamming Distance']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            for dataset in results_df['dataset'].unique():
                dataset_results = results_df[results_df['dataset'] == dataset]
                
                # Calculate mean and std across repetitions
                mean_values = dataset_results.groupby('missing_percentage')[metric].mean()
                std_values = dataset_results.groupby('missing_percentage')[metric].std()
                
                ax.errorbar(mean_values.index * 100, mean_values.values,
                           yerr=std_values.values,
                           marker='o', linewidth=2, label=dataset,
                           capsize=5, capthick=2)
            
            ax.set_xlabel('Missing Data Percentage (%)', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f'{label} vs Missing Data Percentage', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Degradation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistical_significance(self, results_df: pd.DataFrame,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot statistical significance analysis.
        
        Args:
            results_df: DataFrame with experimental results
            save_path: Path to save the plot
        """
        # Calculate confidence intervals
        confidence_data = []
        
        for dataset in results_df['dataset'].unique():
            for missing_pct in results_df['missing_percentage'].unique():
                subset = results_df[(results_df['dataset'] == dataset) & 
                                  (results_df['missing_percentage'] == missing_pct)]
                
                if len(subset) > 1:
                    mean_f1 = subset['edge_f1_score'].mean()
                    std_f1 = subset['edge_f1_score'].std()
                    n = len(subset)
                    
                    # 95% confidence interval
                    ci = 1.96 * (std_f1 / np.sqrt(n))
                    
                    confidence_data.append({
                        'dataset': dataset,
                        'missing_percentage': missing_pct,
                        'mean_f1': mean_f1,
                        'ci_lower': mean_f1 - ci,
                        'ci_upper': mean_f1 + ci,
                        'std': std_f1
                    })
        
        conf_df = pd.DataFrame(confidence_data)
        
        plt.figure(figsize=self.figsize)
        
        for dataset in conf_df['dataset'].unique():
            dataset_conf = conf_df[conf_df['dataset'] == dataset]
            
            plt.errorbar(dataset_conf['missing_percentage'] * 100,
                        dataset_conf['mean_f1'],
                        yerr=[dataset_conf['mean_f1'] - dataset_conf['ci_lower'],
                              dataset_conf['ci_upper'] - dataset_conf['mean_f1']],
                        marker='o', linewidth=2, label=dataset,
                        capsize=5, capthick=2)
        
        plt.xlabel('Missing Data Percentage (%)', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_missing_data_analysis(self, results_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot missing data pattern analysis.
        
        Args:
            results_df: DataFrame with experimental results
            save_path: Path to save the plot
        """
        # Extract missing data analysis information
        missing_analysis_data = []
        
        for _, row in results_df.iterrows():
            if 'missing_data_analysis' in row and isinstance(row['missing_data_analysis'], dict):
                for var, analysis in row['missing_data_analysis'].items():
                    missing_analysis_data.append({
                        'dataset': row['dataset'],
                        'missing_percentage': row['missing_percentage'],
                        'variable': var,
                        'bias_ratio': analysis.get('bias_ratio', 0),
                        'correlation_with_missing': analysis.get('correlation_with_missing', 0)
                    })
        
        if not missing_analysis_data:
            logger.warning("No missing data analysis information found")
            return
        
        missing_df = pd.DataFrame(missing_analysis_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot bias ratio
        for dataset in missing_df['dataset'].unique():
            dataset_data = missing_df[missing_df['dataset'] == dataset]
            mean_bias = dataset_data.groupby('missing_percentage')['bias_ratio'].mean()
            
            ax1.plot(mean_bias.index * 100, mean_bias.values,
                    marker='o', linewidth=2, label=dataset)
        
        ax1.set_xlabel('Missing Data Percentage (%)', fontsize=12)
        ax1.set_ylabel('Bias Ratio', fontsize=12)
        ax1.set_title('Bias Ratio vs Missing Data Percentage', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot correlation with missing
        for dataset in missing_df['dataset'].unique():
            dataset_data = missing_df[missing_df['dataset'] == dataset]
            mean_corr = dataset_data.groupby('missing_percentage')['correlation_with_missing'].mean()
            
            ax2.plot(mean_corr.index * 100, mean_corr.values,
                    marker='s', linewidth=2, label=dataset)
        
        ax2.set_xlabel('Missing Data Percentage (%)', fontsize=12)
        ax2.set_ylabel('Correlation with Missingness', fontsize=12)
        ax2.set_title('Correlation with Missingness vs Missing Data Percentage', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, results_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            results_df: DataFrame with experimental results
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score vs Missing Percentage', 
                          'Precision vs Missing Percentage',
                          'Recall vs Missing Percentage',
                          'Structural Hamming Distance vs Missing Percentage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['edge_f1_score', 'edge_precision', 'edge_recall', 'relative_structural_hamming_distance']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            for dataset in results_df['dataset'].unique():
                dataset_results = results_df[results_df['dataset'] == dataset]
                
                # Calculate mean across repetitions
                mean_values = dataset_results.groupby('missing_percentage')[metric].mean()
                
                fig.add_trace(
                    go.Scatter(x=mean_values.index * 100,
                              y=mean_values.values,
                              mode='lines+markers',
                              name=f'{dataset} - {metric.replace("_", " ").title()}',
                              line=dict(width=2)),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title_text="SM-MVPC Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Missing Data Percentage (%)", row=i, col=j)
                fig.update_yaxes(title_text="Score", row=i, col=j)
        
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            logger.info(f"Interactive dashboard saved to {save_path}")
        else:
            fig.show()


class ReportGenerator:
    """
    Generate comprehensive reports with visualizations.
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph_visualizer = CausalGraphVisualizer()
        self.results_visualizer = ResultsVisualizer()
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame,
                                    ground_truth_graphs: Dict[str, nx.DiGraph],
                                    inferred_graphs: Dict[str, nx.DiGraph]) -> str:
        """
        Generate a comprehensive report with all visualizations.
        
        Args:
            results_df: DataFrame with experimental results
            ground_truth_graphs: Dictionary of ground truth graphs
            inferred_graphs: Dictionary of inferred graphs
            save_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        logger.info("Generating comprehensive visualization report...")
        
        # Create all visualizations
        self.results_visualizer.plot_performance_heatmap(
            results_df, save_path=self.output_dir / 'performance_heatmap.png'
        )
        
        self.results_visualizer.plot_performance_degradation(
            results_df, save_path=self.output_dir / 'performance_degradation.png'
        )
        
        self.results_visualizer.plot_statistical_significance(
            results_df, save_path=self.output_dir / 'statistical_significance.png'
        )
        
        self.results_visualizer.plot_missing_data_analysis(
            results_df, save_path=self.output_dir / 'missing_data_analysis.png'
        )
        
        # Create interactive dashboard
        self.results_visualizer.create_interactive_dashboard(
            results_df, save_path=self.output_dir / 'interactive_dashboard.html'
        )
        
        # Plot causal graphs
        for dataset_name in ground_truth_graphs.keys():
            if dataset_name in inferred_graphs:
                self.graph_visualizer.plot_graph_comparison(
                    ground_truth_graphs[dataset_name],
                    inferred_graphs[dataset_name],
                    title=f"Causal Graph Comparison - {dataset_name}",
                    save_path=self.output_dir / f'graph_comparison_{dataset_name}.png'
                )
        
        logger.info(f"Comprehensive report generated in {self.output_dir}")
        return str(self.output_dir)


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    results_data = []
    
    for dataset in ['heart_disease', 'diabetes', 'hepatitis']:
        for missing_pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for rep in range(3):
                results_data.append({
                    'dataset': dataset,
                    'missing_percentage': missing_pct,
                    'repetition': rep,
                    'edge_f1_score': np.random.uniform(0.3, 0.9),
                    'edge_precision': np.random.uniform(0.4, 0.8),
                    'edge_recall': np.random.uniform(0.3, 0.7),
                    'relative_structural_hamming_distance': np.random.uniform(0, 1)
                })
    
    results_df = pd.DataFrame(results_data)
    
    # Test visualizations
    visualizer = ResultsVisualizer()
    visualizer.plot_performance_heatmap(results_df)
    visualizer.plot_performance_degradation(results_df)
    visualizer.create_interactive_dashboard(results_df)
    
    print("Visualization tests completed successfully!")
