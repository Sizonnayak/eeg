"""
Generate Comprehensive Comparison Report for All EEGNet Models
================================================================
Analyzes all result JSONs and creates a detailed comparison PDF.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 8.5)  # Letter size
plt.rcParams['font.size'] = 10


def load_results(results_dir):
    """Load all available result JSONs."""
    results_dir = Path(results_dir)

    models = {
        'Raw EEGNet (Baseline)': 'eegnet_loso_results.json',
        'xDAWN-4 + EEGNet': 'eegnet_xdawn4_loso_results.json',
        'EEG-Inception': 'eeginception_loso_results.json',
        'Optimized EEGNet (F1=16)': 'eegnet_optimized_loso_results.json',
        'Undersampled (1:2)': 'eegnet_undersampled_1to2_results.json',
        'Ensemble': 'eegnet_ensemble_results.json',
    }

    loaded = {}
    for name, filename in models.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                loaded[name] = json.load(f)

    return loaded


def extract_metrics(results_dict):
    """Extract key metrics from results."""
    metrics = {}
    for model_name, data in results_dict.items():
        agg = data['aggregate']
        metrics[model_name] = {
            'bacc_mean': agg['balanced_accuracy']['mean'],
            'bacc_std': agg['balanced_accuracy']['std'],
            'f1_mean': agg['f1']['mean'],
            'f1_std': agg['f1']['std'],
            'auc_roc_mean': agg['auc_roc']['mean'],
            'auc_roc_std': agg['auc_roc']['std'],
            'precision_mean': agg['precision']['mean'],
            'precision_std': agg['precision']['std'],
            'recall_mean': agg['recall']['mean'],
            'recall_std': agg['recall']['std'],
            'bacc_per_subject': agg['balanced_accuracy']['per_subject'],
        }
    return metrics


def create_title_page(pdf, results_dict):
    """Create title page."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.axis('off')

    title_text = "P300 Classification Performance Report\n8-Subject LOSO Cross-Validation"
    ax.text(0.5, 0.7, title_text, ha='center', va='center',
            fontsize=24, fontweight='bold')

    info_text = f"""
    Dataset: 8 subjects, 33,489 epochs
    Target rate: 16.6% (class imbalanced)
    Evaluation: Leave-One-Subject-Out CV

    Models Compared: {len(results_dict)}

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """
    ax.text(0.5, 0.4, info_text, ha='center', va='top',
            fontsize=12, family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_summary_table(pdf, metrics):
    """Create summary comparison table."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Model Performance Summary', ha='center',
            fontsize=16, fontweight='bold')

    # Table data
    headers = ['Model', 'BAcc', 'F1', 'AUC-ROC', 'Precision', 'Recall']
    table_data = []

    for model_name in sorted(metrics.keys()):
        m = metrics[model_name]
        row = [
            model_name,
            f"{m['bacc_mean']:.3f} ± {m['bacc_std']:.3f}",
            f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
            f"{m['auc_roc_mean']:.3f} ± {m['auc_roc_std']:.3f}",
            f"{m['precision_mean']:.3f} ± {m['precision_std']:.3f}",
            f"{m['recall_mean']:.3f} ± {m['recall_std']:.3f}",
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='left', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.75])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best values
    best_bacc_idx = max(range(len(table_data)),
                       key=lambda i: metrics[table_data[i][0]]['bacc_mean'])
    table[(best_bacc_idx + 1, 1)].set_facecolor('#C6E0B4')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_bacc_comparison(pdf, metrics):
    """Create BAcc comparison bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

    models = list(metrics.keys())
    baccs = [metrics[m]['bacc_mean'] for m in models]
    stds = [metrics[m]['bacc_std'] for m in models]

    colors = ['#4472C4' if bacc >= 0.65 else '#ED7D31' for bacc in baccs]

    bars = ax.barh(models, baccs, xerr=stds, color=colors, alpha=0.7, capsize=5)

    # Add value labels
    for i, (bacc, std) in enumerate(zip(baccs, stds)):
        ax.text(bacc + std + 0.01, i, f'{bacc:.3f}', va='center', fontsize=9)

    ax.axvline(0.65, color='green', linestyle='--', linewidth=2,
               label='Target (65%)')
    ax.axvline(0.605, color='red', linestyle='--', linewidth=1,
               label='Traditional ML Baseline')

    ax.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Balanced Accuracy Comparison (LOSO CV)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.55, max(baccs) + 0.05)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_per_subject_heatmap(pdf, metrics):
    """Create per-subject performance heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

    # Prepare data
    models = list(metrics.keys())
    subjects = list(range(1, 9))

    data = []
    for model in models:
        data.append(metrics[model]['bacc_per_subject'])

    data = np.array(data)

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.55, vmax=0.75)

    # Set ticks
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'S{i}' for i in subjects])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Add values
    for i in range(len(models)):
        for j in range(8):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_title('Per-Subject Balanced Accuracy',
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('BAcc', rotation=270, labelpad=20, fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_precision_recall_scatter(pdf, metrics):
    """Create precision-recall scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

    models = list(metrics.keys())
    precisions = [metrics[m]['precision_mean'] for m in models]
    recalls = [metrics[m]['recall_mean'] for m in models]
    baccs = [metrics[m]['bacc_mean'] for m in models]

    # Size by BAcc
    sizes = [(bacc - 0.55) * 2000 for bacc in baccs]

    scatter = ax.scatter(recalls, precisions, s=sizes, alpha=0.6,
                        c=baccs, cmap='viridis', edgecolors='black', linewidth=1.5)

    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (recalls[i], precisions[i]),
                   fontsize=8, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off (bubble size = BAcc)',
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('BAcc', rotation=270, labelpad=20, fontweight='bold')

    ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_improvement_analysis(pdf, metrics):
    """Create improvement analysis from baseline."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

    # Find baseline
    baseline_name = 'Raw EEGNet (Baseline)'
    if baseline_name not in metrics:
        baseline_name = list(metrics.keys())[0]

    baseline_bacc = metrics[baseline_name]['bacc_mean']

    models = [m for m in metrics.keys() if m != baseline_name]
    improvements = [(metrics[m]['bacc_mean'] - baseline_bacc) * 100 for m in models]

    colors = ['green' if imp > 0 else 'red' for imp in improvements]

    bars = ax.barh(models, improvements, color=colors, alpha=0.7)

    # Add value labels
    for i, imp in enumerate(improvements):
        x_pos = imp + (0.3 if imp > 0 else -0.3)
        ax.text(x_pos, i, f'{imp:+.1f}%', va='center',
               ha='left' if imp > 0 else 'right', fontsize=9)

    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Improvement Analysis\n(Baseline: {baseline_name} = {baseline_bacc:.3f})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_recommendation_page(pdf, metrics):
    """Create recommendations page."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.axis('off')

    # Find best model
    best_model = max(metrics.keys(), key=lambda m: metrics[m]['bacc_mean'])
    best_bacc = metrics[best_model]['bacc_mean']

    ax.text(0.5, 0.95, 'Recommendations & Insights', ha='center',
            fontsize=16, fontweight='bold')

    recommendations = f"""
    BEST PERFORMING MODEL:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {best_model}
    Balanced Accuracy: {best_bacc:.3f} ± {metrics[best_model]['bacc_std']:.3f}


    KEY FINDINGS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Best approach: {"Ensemble" if "Ensemble" in best_model else "Single model optimization"}

    2. Improvement over traditional ML (60.5%): +{(best_bacc - 0.605)*100:.1f}%

    3. Target achievement: {"✓ ACHIEVED" if best_bacc >= 0.67 else "Approaching target (67%)"}


    DEPLOYMENT RECOMMENDATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    if 'Ensemble' in metrics and metrics['Ensemble']['bacc_mean'] >= 0.67:
        recommendations += """
    ✓ Use ENSEMBLE model for production
    ✓ Achieves 67%+ BAcc (production-ready)
    ✓ Robust across subjects (reduced variance)
    """
    elif best_bacc >= 0.65:
        recommendations += f"""
    ✓ Use {best_model}
    ✓ Solid performance (65%+ BAcc)
    ✓ Consider ensemble for +2-3% boost
    """
    else:
        recommendations += """
    → Further optimization needed
    → Try: More epochs, ensemble, or data augmentation
    """

    ax.text(0.05, 0.85, recommendations, ha='left', va='top',
            fontsize=10, family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    """Generate comprehensive comparison PDF."""
    print("=" * 70)
    print("  Generating P300 Classification Comparison Report")
    print("=" * 70)

    results_dir = Path(__file__).parent / 'results'

    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)
    print(f"  Found {len(results)} models:")
    for name in results.keys():
        print(f"    - {name}")

    if len(results) == 0:
        print("\n❌ No result files found!")
        return

    # Extract metrics
    print("\nExtracting metrics...")
    metrics = extract_metrics(results)

    # Generate PDF
    output_path = results_dir / 'P300_Model_Comparison_Report.pdf'
    print(f"\nGenerating PDF: {output_path}")

    with PdfPages(output_path) as pdf:
        print("  Creating title page...")
        create_title_page(pdf, results)

        print("  Creating summary table...")
        create_summary_table(pdf, metrics)

        print("  Creating BAcc comparison...")
        create_bacc_comparison(pdf, metrics)

        print("  Creating per-subject heatmap...")
        create_per_subject_heatmap(pdf, metrics)

        print("  Creating precision-recall scatter...")
        create_precision_recall_scatter(pdf, metrics)

        print("  Creating improvement analysis...")
        create_improvement_analysis(pdf, metrics)

        print("  Creating recommendations page...")
        create_recommendation_page(pdf, metrics)

        # Metadata
        d = pdf.infodict()
        d['Title'] = 'P300 Classification Performance Report'
        d['Author'] = 'EEG Analysis Pipeline'
        d['Subject'] = '8-Subject LOSO Cross-Validation Comparison'
        d['CreationDate'] = datetime.now()

    print(f"\n✓ Report generated: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
