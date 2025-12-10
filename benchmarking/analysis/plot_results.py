"""
Plot and analyze benchmark results

This script generates publication-quality plots and summary tables
from benchmark JSON files.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_results(data_dir, pattern):
    """Load all JSON files matching pattern from data directory."""
    results = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob(f"*{pattern}*.json"):
        if "metadata" not in json_file.name:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
    
    return results


def plot_cost_vs_qubits(results, output_dir, title="Cost vs Number of Qubits"):
    """Plot partitioning cost vs number of qubits."""
    df = pd.DataFrame(results)
    
    if 'num_partitions' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, num_parts in zip(axes, sorted(df['num_partitions'].unique())):
            subset = df[df['num_partitions'] == num_parts]
            
            if 'fraction' in subset.columns:
                for frac in sorted(subset['fraction'].unique()):
                    frac_data = subset[subset['fraction'] == frac]
                    grouped = frac_data.groupby('num_qubits')['mean_cost'].mean()
                    ax.plot(grouped.index, grouped.values, marker='o', label=f'fraction={frac}')
            else:
                grouped = subset.groupby('num_qubits')['mean_cost'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', linewidth=2)
            
            ax.set_xlabel('Number of Qubits')
            ax.set_ylabel('Mean Communication Cost')
            ax.set_title(f'{num_parts} Partitions')
            ax.grid(True, alpha=0.3)
            if 'fraction' in subset.columns:
                ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped = df.groupby('num_qubits')['mean_cost'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2)
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Mean Communication Cost')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "cost_vs_qubits.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_time_vs_qubits(results, output_dir, title="Runtime vs Number of Qubits"):
    """Plot runtime vs number of qubits."""
    df = pd.DataFrame(results)
    
    if 'num_partitions' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, num_parts in zip(axes, sorted(df['num_partitions'].unique())):
            subset = df[df['num_partitions'] == num_parts]
            
            if 'fraction' in subset.columns:
                for frac in sorted(subset['fraction'].unique()):
                    frac_data = subset[subset['fraction'] == frac]
                    grouped = frac_data.groupby('num_qubits')['mean_time'].mean()
                    ax.plot(grouped.index, grouped.values, marker='o', label=f'fraction={frac}')
            else:
                grouped = subset.groupby('num_qubits')['mean_time'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', linewidth=2)
            
            ax.set_xlabel('Number of Qubits')
            ax.set_ylabel('Mean Runtime (seconds)')
            ax.set_title(f'{num_parts} Partitions')
            ax.grid(True, alpha=0.3)
            if 'fraction' in subset.columns:
                ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped = df.groupby('num_qubits')['mean_time'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2)
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Mean Runtime (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "time_vs_qubits.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_circuit_comparison(data_dir, output_dir):
    """Compare different circuit types."""
    circuit_types = ["QAOA", "QFT", "QV"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for circuit_type in circuit_types:
        results = load_results(data_dir, f"means_MLFM-R_{circuit_type}")
        if not results:
            continue
        
        df = pd.DataFrame(results)
        
        # Plot cost comparison
        for num_parts in sorted(df['num_partitions'].unique()):
            subset = df[df['num_partitions'] == num_parts]
            grouped = subset.groupby('num_qubits')['mean_cost'].mean()
            axes[0].plot(grouped.index, grouped.values, marker='o', label=f'{circuit_type} ({num_parts} parts)')
        
        # Plot time comparison
        for num_parts in sorted(df['num_partitions'].unique()):
            subset = df[df['num_partitions'] == num_parts]
            grouped = subset.groupby('num_qubits')['mean_time'].mean()
            axes[1].plot(grouped.index, grouped.values, marker='o', label=f'{circuit_type} ({num_parts} parts)')
    
    axes[0].set_xlabel('Number of Qubits')
    axes[0].set_ylabel('Mean Communication Cost')
    axes[0].set_title('Cost Comparison Across Circuit Types')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of Qubits')
    axes[1].set_ylabel('Mean Runtime (seconds)')
    axes[1].set_title('Runtime Comparison Across Circuit Types')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "circuit_type_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def generate_summary_table(results, output_file):
    """Generate a markdown summary table."""
    df = pd.DataFrame(results)
    
    # Group by configuration and compute statistics
    group_cols = [col for col in ['circuit_type', 'num_qubits', 'num_partitions', 'fraction'] 
                  if col in df.columns]
    
    if group_cols:
        summary = df.groupby(group_cols).agg({
            'mean_cost': ['mean', 'std'],
            'mean_time': ['mean', 'std']
        }).round(3)
        
        # Save to markdown
        with open(output_file, 'w') as f:
            f.write("# Benchmark Results Summary\n\n")
            f.write(summary.to_markdown())
            f.write("\n")
        
        print(f"Saved summary table: {output_file}")


def main():
    """Generate all plots and analysis."""
    data_dir = "../data"
    output_dir = "../results/plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Generating Benchmark Plots and Analysis")
    print("="*70)
    
    # CP Large Scale plots
    print("\nProcessing CP Large Scale results...")
    cp_large_results = load_results(data_dir, "means_MLFM-R_CP_large")
    if cp_large_results:
        plot_cost_vs_qubits(cp_large_results, output_dir, "CP Large Scale: Cost vs Qubits")
        plot_time_vs_qubits(cp_large_results, output_dir, "CP Large Scale: Runtime vs Qubits")
        generate_summary_table(cp_large_results, os.path.join(output_dir, "../cp_large_summary.md"))
    
    # CP Scaling plots
    print("\nProcessing CP Scaling results...")
    cp_scaling_results = load_results(data_dir, "means_MLFM-R_CP_scaling")
    if cp_scaling_results:
        plot_cost_vs_qubits(cp_scaling_results, output_dir, "CP Scaling: Cost vs Qubits")
        plot_time_vs_qubits(cp_scaling_results, output_dir, "CP Scaling: Runtime vs Qubits")
        generate_summary_table(cp_scaling_results, os.path.join(output_dir, "../cp_scaling_summary.md"))
    
    # Circuit comparison plots
    print("\nProcessing standard circuit comparison...")
    plot_circuit_comparison(data_dir, output_dir)
    
    # Individual circuit plots
    for circuit_type in ["QAOA", "QFT", "QV"]:
        print(f"\nProcessing {circuit_type} results...")
        results = load_results(data_dir, f"means_MLFM-R_{circuit_type}")
        if results:
            plot_cost_vs_qubits(results, output_dir, f"{circuit_type}: Cost vs Qubits")
            plot_time_vs_qubits(results, output_dir, f"{circuit_type}: Runtime vs Qubits")
            generate_summary_table(results, os.path.join(output_dir, f"../{circuit_type.lower()}_summary.md"))
    
    print("\n" + "="*70)
    print(f"All plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
