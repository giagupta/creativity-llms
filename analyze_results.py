#!/usr/bin/env python3
"""
Analysis script for divergent-convergent experiment results.
Generates statistical comparisons, visualizations, and detailed breakdowns.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_all_results(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all result JSON files organized by prompt and condition."""
    results = {}
    
    for prompt_dir in runs_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
            
        prompt_id = prompt_dir.name
        results[prompt_id] = {}
        
        for json_file in prompt_dir.glob("*.json"):
            condition = json_file.stem
            with open(json_file, 'r') as f:
                results[prompt_id][condition] = json.load(f)
    
    return results


def extract_scores(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Extract scores into a DataFrame for analysis."""
    rows = []
    
    for prompt_id, conditions in results.items():
        for condition, data in conditions.items():
            # Selected candidate scores
            selected = data['result']['selected']['scores']
            
            row = {
                'prompt_id': prompt_id,
                'condition': condition,
                'type': 'selected',
                'constraint': selected['constraint_satisfaction'],
                'usefulness': selected['usefulness'],
                'breakthrough': selected['breakthrough_potential'],
                'final_score': selected['final_score']
            }
            
            # Add breakthrough dimension scores if available
            if selected.get('breakthrough_scores'):
                bs = selected['breakthrough_scores']
                row.update({
                    'generative_power': bs.get('generative_power', np.nan),
                    'explanatory_depth': bs.get('explanatory_depth', np.nan),
                    'non_obviousness': bs.get('non_obviousness', np.nan),
                    'scalability': bs.get('scalability_of_impact', np.nan),
                    'principle_over_impl': bs.get('principle_over_implementation', np.nan)
                })
            
            rows.append(row)
            
            # All candidates (for distribution analysis)
            for cand in data['result']['candidates']:
                if cand['judge'].get('discard', False):
                    continue
                    
                cand_row = {
                    'prompt_id': prompt_id,
                    'condition': condition,
                    'type': 'candidate',
                    'mode': cand.get('mode', 'unknown'),
                    'constraint': cand['judge']['constraint_satisfaction'],
                    'usefulness': cand['judge']['usefulness'],
                    'breakthrough': cand['judge']['breakthrough_potential'],
                    'final_score': cand['judge']['final_score']
                }
                
                if cand['judge'].get('breakthrough_scores'):
                    bs = cand['judge']['breakthrough_scores']
                    cand_row.update({
                        'generative_power': bs.get('generative_power', np.nan),
                        'explanatory_depth': bs.get('explanatory_depth', np.nan),
                        'non_obviousness': bs.get('non_obviousness', np.nan),
                        'scalability': bs.get('scalability_of_impact', np.nan),
                        'principle_over_impl': bs.get('principle_over_implementation', np.nan)
                    })
                
                rows.append(cand_row)
    
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistical comparisons between conditions."""
    stats_results = {}
    
    # Compare selected candidates only
    selected = df[df['type'] == 'selected']
    
    conditions = selected['condition'].unique()
    if len(conditions) != 2:
        print(f"Warning: Expected 2 conditions, found {len(conditions)}")
        return stats_results
    
    cond1, cond2 = sorted(conditions)
    
    metrics = ['constraint', 'usefulness', 'breakthrough', 'final_score']
    
    for metric in metrics:
        group1 = selected[selected['condition'] == cond1][metric].dropna()
        group2 = selected[selected['condition'] == cond2][metric].dropna()
        
        # Paired t-test (same prompts across conditions)
        if len(group1) == len(group2) and len(group1) > 1:
            t_stat, p_value = stats.ttest_rel(group1, group2)
            
            # Effect size (Cohen's d for paired samples)
            diff = group1 - group2
            cohens_d = diff.mean() / diff.std()
            
            stats_results[metric] = {
                f'{cond1}_mean': group1.mean(),
                f'{cond1}_std': group1.std(),
                f'{cond2}_mean': group2.mean(),
                f'{cond2}_std': group2.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
        else:
            # Unpaired t-test if sample sizes differ
            t_stat, p_value = stats.ttest_ind(group1, group2)
            cohens_d = (group2.mean() - group1.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
            
            stats_results[metric] = {
                f'{cond1}_mean': group1.mean(),
                f'{cond1}_std': group1.std(),
                f'{cond2}_mean': group2.mean(),
                f'{cond2}_std': group2.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
    
    # Breakthrough dimensions comparison
    breakthrough_dims = ['generative_power', 'explanatory_depth', 'non_obviousness', 
                         'scalability', 'principle_over_impl']
    
    stats_results['breakthrough_dimensions'] = {}
    for dim in breakthrough_dims:
        group1 = selected[selected['condition'] == cond1][dim].dropna()
        group2 = selected[selected['condition'] == cond2][dim].dropna()
        
        if len(group1) > 0 and len(group2) > 0:
            if len(group1) == len(group2) and len(group1) > 1:
                t_stat, p_value = stats.ttest_rel(group1, group2)
                diff = group1 - group2
                cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
            else:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
                cohens_d = (group2.mean() - group1.mean()) / pooled_std if pooled_std > 0 else 0
            
            stats_results['breakthrough_dimensions'][dim] = {
                f'{cond1}_mean': group1.mean(),
                f'{cond2}_mean': group2.mean(),
                'difference': group2.mean() - group1.mean(),
                'cohens_d': cohens_d,
                'p_value': p_value
            }
    
    return stats_results


def analyze_mode_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance by generation mode (for divergent-convergent condition)."""
    dc_candidates = df[(df['condition'].str.contains('divergent_convergent')) & 
                       (df['type'] == 'candidate')]
    
    if dc_candidates.empty:
        return {}
    
    mode_stats = {}
    for mode in dc_candidates['mode'].unique():
        mode_data = dc_candidates[dc_candidates['mode'] == mode]
        
        mode_stats[mode] = {
            'count': len(mode_data),
            'breakthrough_mean': mode_data['breakthrough'].mean(),
            'breakthrough_std': mode_data['breakthrough'].std(),
            'non_obviousness_mean': mode_data['non_obviousness'].mean(),
            'generative_power_mean': mode_data['generative_power'].mean(),
            'final_score_mean': mode_data['final_score'].mean()
        }
    
    return mode_stats


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate visualizations of results."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 1. Selected candidates: Breakthrough comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    selected = df[df['type'] == 'selected']
    
    sns.boxplot(data=selected, x='condition', y='breakthrough', ax=ax)
    sns.swarmplot(data=selected, x='condition', y='breakthrough', color='black', 
                  alpha=0.5, size=8, ax=ax)
    
    ax.set_title('Breakthrough Potential by Condition (Selected Candidates)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Breakthrough Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'breakthrough_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. All metrics comparison (selected only)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['constraint', 'usefulness', 'breakthrough', 'final_score']
    titles = ['Constraint Satisfaction', 'Usefulness', 'Breakthrough Potential', 'Final Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        sns.barplot(data=selected, x='condition', y=metric, ax=ax, 
                   errorbar='sd', capsize=0.1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1.1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.suptitle('Performance Metrics Comparison (Selected Candidates)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Breakthrough dimensions breakdown
    breakthrough_dims = ['generative_power', 'explanatory_depth', 'non_obviousness', 
                        'scalability', 'principle_over_impl']
    dim_labels = ['Generative\nPower', 'Explanatory\nDepth', 'Non-\nObviousness', 
                  'Scalability', 'Principle over\nImplementation']
    
    selected_dims = selected[['condition'] + breakthrough_dims].copy()
    selected_dims_long = selected_dims.melt(id_vars='condition', 
                                            var_name='dimension', 
                                            value_name='score')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=selected_dims_long, x='dimension', y='score', 
               hue='condition', ax=ax, errorbar='sd', capsize=0.1)
    
    ax.set_title('Breakthrough Dimensions by Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(dim_labels)
    ax.legend(title='Condition', fontsize=10)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(output_dir / 'breakthrough_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of all candidates (not just selected)
    candidates = df[df['type'] == 'candidate']
    
    if not candidates.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Violin plot
        sns.violinplot(data=candidates, x='condition', y='breakthrough', ax=axes[0])
        axes[0].set_title('Distribution of All Candidate Breakthrough Scores', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Condition', fontsize=10)
        axes[0].set_ylabel('Breakthrough Score', fontsize=10)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Histogram
        for condition in candidates['condition'].unique():
            cond_data = candidates[candidates['condition'] == condition]['breakthrough']
            axes[1].hist(cond_data, alpha=0.6, label=condition, bins=15)
        
        axes[1].set_title('Histogram of Breakthrough Scores', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Breakthrough Score', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'candidate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Mode comparison (divergent-convergent only)
    dc_candidates = df[(df['condition'].str.contains('divergent_convergent')) & 
                       (df['type'] == 'candidate')]
    
    if not dc_candidates.empty and 'mode' in dc_candidates.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=dc_candidates, x='mode', y='breakthrough', ax=ax)
        sns.swarmplot(data=dc_candidates, x='mode', y='breakthrough', 
                     color='black', alpha=0.3, size=4, ax=ax)
        
        ax.set_title('Breakthrough Potential by Generation Mode', fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation Mode', fontsize=12)
        ax.set_ylabel('Breakthrough Score', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'mode_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(results: Dict[str, Any], stats: Dict[str, Any], 
                   mode_stats: Dict[str, Any], output_path: Path):
    """Generate a text report summarizing findings."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DIVERGENT-CONVERGENT EXPERIMENT ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL PERFORMANCE COMPARISON (Selected Candidates)")
    report_lines.append("-" * 80)
    
    for metric, data in stats.items():
        if metric == 'breakthrough_dimensions':
            continue
            
        report_lines.append(f"\n{metric.upper().replace('_', ' ')}:")
        
        cond_keys = [k for k in data.keys() if k.endswith('_mean')]
        for key in cond_keys:
            cond = key.replace('_mean', '')
            mean = data[key]
            std = data.get(f'{cond}_std', 0)
            report_lines.append(f"  {cond}: {mean:.4f} (±{std:.4f})")
        
        if 'p_value' in data:
            sig_marker = "***" if data['p_value'] < 0.001 else \
                        "**" if data['p_value'] < 0.01 else \
                        "*" if data['p_value'] < 0.05 else "ns"
            
            report_lines.append(f"  t-statistic: {data['t_statistic']:.4f}")
            report_lines.append(f"  p-value: {data['p_value']:.4f} {sig_marker}")
            report_lines.append(f"  Cohen's d: {data['cohens_d']:.4f}")
            
            # Interpret effect size
            abs_d = abs(data['cohens_d'])
            if abs_d < 0.2:
                effect = "negligible"
            elif abs_d < 0.5:
                effect = "small"
            elif abs_d < 0.8:
                effect = "medium"
            else:
                effect = "large"
            report_lines.append(f"  Effect size: {effect}")
    
    # Breakthrough dimensions
    if 'breakthrough_dimensions' in stats:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BREAKTHROUGH DIMENSIONS BREAKDOWN")
        report_lines.append("-" * 80)
        
        for dim, data in stats['breakthrough_dimensions'].items():
            report_lines.append(f"\n{dim.upper().replace('_', ' ')}:")
            
            cond_keys = [k for k in data.keys() if k.endswith('_mean')]
            for key in cond_keys:
                cond = key.replace('_mean', '')
                report_lines.append(f"  {cond}: {data[key]:.4f}")
            
            report_lines.append(f"  Difference: {data['difference']:.4f}")
            report_lines.append(f"  Cohen's d: {data['cohens_d']:.4f}")
            
            sig_marker = "***" if data['p_value'] < 0.001 else \
                        "**" if data['p_value'] < 0.01 else \
                        "*" if data['p_value'] < 0.05 else "ns"
            report_lines.append(f"  p-value: {data['p_value']:.4f} {sig_marker}")
    
    # Mode performance
    if mode_stats:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("GENERATION MODE ANALYSIS (Divergent-Convergent)")
        report_lines.append("-" * 80)
        
        for mode, data in mode_stats.items():
            report_lines.append(f"\n{mode.upper()}:")
            report_lines.append(f"  Count: {data['count']}")
            report_lines.append(f"  Breakthrough: {data['breakthrough_mean']:.4f} (±{data['breakthrough_std']:.4f})")
            report_lines.append(f"  Non-obviousness: {data['non_obviousness_mean']:.4f}")
            report_lines.append(f"  Generative power: {data['generative_power_mean']:.4f}")
            report_lines.append(f"  Final score: {data['final_score_mean']:.4f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))


def main():
    root = Path(__file__).parent
    runs_dir = root / "runs"
    
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist. Run experiment first.")
        return
    
    # Load results
    print("Loading results...")
    results = load_all_results(runs_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for {len(results)} prompts")
    
    # Extract scores
    print("Extracting scores...")
    df = extract_scores(results)
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(df)
    
    # Analyze mode performance
    print("Analyzing generation modes...")
    mode_stats = analyze_mode_performance(df)
    
    # Create output directory
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    viz_dir = analysis_dir / "visualizations"
    create_visualizations(df, viz_dir)
    
    # Save data
    print("Saving data...")
    df.to_csv(analysis_dir / 'all_scores.csv', index=False)
    
    with open(analysis_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    with open(analysis_dir / 'mode_statistics.json', 'w') as f:
        json.dump(mode_stats, f, indent=2)
    
    # Generate report
    print("Generating report...")
    generate_report(results, stats, mode_stats, analysis_dir / 'report.txt')
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}/")
    print(f"  - all_scores.csv: Raw data")
    print(f"  - statistics.json: Statistical comparisons")
    print(f"  - mode_statistics.json: Mode-specific analysis")
    print(f"  - report.txt: Human-readable summary")
    print(f"  - visualizations/: Plots and charts")


if __name__ == "__main__":
    main()