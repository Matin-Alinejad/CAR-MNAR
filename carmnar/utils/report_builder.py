"""
HTML Report Builder
===================

Generates HTML reports from experimental results, including summary tables,
per-dataset metrics, and links to figures produced by the pipeline.

Author: Anonymous (for review)
Date: 2025
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import json


BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
FONTAWESOME = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"


def _write_html(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _card(title: str, body_html: str, icon: str = "fa-chart-bar") -> str:
    return f"""
    <div class="card mb-4">
      <div class="card-body">
        <h5 class="card-title"><i class="fas {icon} me-2"></i>{title}</h5>
        {body_html}
      </div>
    </div>
    """


def _table_html(df: pd.DataFrame, classes: str = "table table-striped table-hover table-sm") -> str:
    return df.to_html(classes=classes, index=False, border=0)


def build_results_report(results_df: pd.DataFrame, output_dir: str = "reports", figures_dir: str = "results") -> str:
    """
    Build results.html with summary statistics and links to figures.
    """
    output_path = Path(output_dir) / "results.html"

    # Aggregate metrics by dataset and missing percentage
    agg = results_df.groupby(["dataset", "missing_percentage"]).agg({
        'edge_precision': 'mean',
        'edge_recall': 'mean',
        'edge_f1_score': 'mean',
        'skeleton_precision': 'mean',
        'skeleton_recall': 'mean',
        'skeleton_f1_score': 'mean',
        'structural_hamming_distance': 'mean',
        'relative_structural_hamming_distance': 'mean',
        'experiment_time': 'mean'
    }).reset_index()

    # Round for presentation
    rounded = agg.copy()
    for col in rounded.columns:
        if col not in ["dataset", "missing_percentage"]:
            rounded[col] = rounded[col].astype(float).round(3)

    # Best/worst rows
    best_idx = results_df['edge_f1_score'].idxmax()
    worst_idx = results_df['edge_f1_score'].idxmin()
    best_row = results_df.loc[best_idx]
    worst_row = results_df.loc[worst_idx]

    # Build HTML
    header = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Results & Analysis - SM-MVPC</title>
      <link href="{BOOTSTRAP}" rel="stylesheet">
      <link href="{FONTAWESOME}" rel="stylesheet">
    </head>
    <body>
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
        <div class="container">
          <a class="navbar-brand" href="index.html"><i class="fas fa-brain me-2"></i>SM-MVPC Results</a>
        </div>
      </nav>
      <div class="container">
        <h2 class="mb-4">Results & Analysis</h2>
    """

    # Summary cards
    summary_html = """
    <div class="row">
      <div class="col-md-6">{best_card}</div>
      <div class="col-md-6">{worst_card}</div>
    </div>
    """.format(
        best_card=_card(
            "Best Condition",
            f"<p class='mb-1'><strong>Dataset:</strong> {best_row['dataset']}</p>"
            f"<p class='mb-1'><strong>Missing %:</strong> {best_row['missing_percentage']*100:.0f}%</p>"
            f"<p class='mb-0'><strong>F1 Score:</strong> {best_row['edge_f1_score']:.3f}</p>",
            "fa-trophy"
        ),
        worst_card=_card(
            "Worst Condition",
            f"<p class='mb-1'><strong>Dataset:</strong> {worst_row['dataset']}</p>"
            f"<p class='mb-1'><strong>Missing %:</strong> {worst_row['missing_percentage']*100:.0f}%</p>"
            f"<p class='mb-0'><strong>F1 Score:</strong> {worst_row['edge_f1_score']:.3f}</p>",
            "fa-triangle-exclamation"
        )
    )

    # Aggregated table per dataset
    tables_html = ""
    for ds in rounded['dataset'].unique():
        ds_df = rounded[rounded['dataset'] == ds].copy()
        ds_df.rename(columns={'missing_percentage': 'Missing %'}, inplace=True)
        ds_df['Missing %'] = (ds_df['Missing %'] * 100).round(0).astype(int)
        tables_html += _card(f"Summary Metrics - {ds}", _table_html(ds_df))

    # Link to figures if exist
    figs = []
    figs_dir = Path(figures_dir)
    for fname in [
        'f1_score_vs_missing.png',
        'metrics_comparison.png',
        'f1_score_heatmap.png',
        'performance_degradation.png',
        'statistical_significance.png',
        'missing_data_analysis.png'
    ]:
        # figures may be saved under results/experiments
        for base in [figs_dir, figs_dir / 'experiments', Path('results/experiments')]:
            p = base / fname
            if p.exists():
                figs.append(p)
                break

    figs_html = "<div class='row'>"
    for p in figs:
        figs_html += f"""
        <div class="col-md-6 mb-4">
          <div class="card">
            <img src="{p.as_posix()}" class="card-img-top" alt="{p.name}">
            <div class="card-body"><small class="text-muted">{p.name}</small></div>
          </div>
        </div>
        """
    figs_html += "</div>"

    footer = """
      </div>
    </body>
    </html>
    """

    html = header + summary_html + tables_html + _card("Figures", figs_html, "fa-image") + footer
    _write_html(output_path, html)
    return str(output_path)


def build_datasets_report(datasets_info: Dict[str, dict], output_dir: str = "reports") -> str:
    """
    Build datasets.html summarizing dataset shapes, targets, and preprocessing.
    """
    output_path = Path(output_dir) / "datasets.html"
    rows = []
    for name, info in datasets_info.items():
        if 'error' in info:
            rows.append({
                'Dataset': name,
                'Status': 'Error',
                'Message': info['error']
            })
        else:
            rows.append({
                'Dataset': name,
                'Description': info.get('description', ''),
                'Original Shape': f"{info.get('original_shape', ('?','?'))}",
                'Processed Shape': f"{info.get('processed_shape', ('?','?'))}",
                'Target Column': info.get('target_column', ''),
                'Missing (Original)': info.get('missing_values_original', 0),
                'Missing (Processed)': info.get('missing_values_processed', 0)
            })

    df = pd.DataFrame(rows)
    header = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Datasets - SM-MVPC</title>
      <link href="{BOOTSTRAP}" rel="stylesheet">
      <link href="{FONTAWESOME}" rel="stylesheet">
    </head>
    <body>
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
        <div class="container">
          <a class="navbar-brand" href="index.html"><i class="fas fa-database me-2"></i>Datasets</a>
        </div>
      </nav>
      <div class="container">
        <h2 class="mb-4">Datasets Summary</h2>
    """
    footer = """
      </div>
    </body>
    </html>
    """
    html = header + _table_html(df) + footer
    _write_html(output_path, html)
    return str(output_path)


def build_all_reports(results_df: pd.DataFrame, datasets_info: Dict[str, dict], output_dir: str = "reports") -> Dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    paths['results'] = build_results_report(results_df, output_dir=output_dir)
    paths['datasets'] = build_datasets_report(datasets_info, output_dir=output_dir)
    return paths


