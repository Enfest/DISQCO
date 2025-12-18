"""
Calculate E-bit Fractions from Benchmark Results

This script processes benchmark JSON files to calculate cost and time fractions
(normalized by the number of two-qubit gates). It generates a summary report
showing the average cost and time per two-qubit gate for each benchmark.

Usage:
    python calculate_ebit_fractions.py --input_dir data --output_file ebit_fractions.dat
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_fractions_for_file(filepath):
    """
    Calculate average cost and time fractions for a single benchmark file.
    
    Args:
        filepath: Path to JSON benchmark results file
        
    Returns:
        dict with 'cost_fraction' and 'time_fraction', or None if file has issues
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    if not data:
        return None
    
    # Check if data has required fields
    if not all('num_two_qubit_gates' in entry and 'cost' in entry and 'time' in entry 
               for entry in data):
        print(f"Warning: {filepath} missing required fields (num_two_qubit_gates, cost, time)")
        return None
    
    # Calculate fractions for each entry
    cost_fractions = []
    time_fractions = []
    
    for entry in data:
        num_gates = entry['num_two_qubit_gates']
        
        # Skip entries with zero gates to avoid division by zero
        if num_gates == 0:
            continue
            
        cost_fractions.append(entry['cost'] / num_gates)
        time_fractions.append(entry['time'] / num_gates)
    
    if not cost_fractions:
        return None
    
    # Return average fractions
    return {
        'cost_fraction': sum(cost_fractions) / len(cost_fractions),
        'time_fraction': sum(time_fractions) / len(time_fractions),
        'num_samples': len(cost_fractions)
    }


def extract_circuit_type(filename):
    """
    Extract circuit type from benchmark filename.
    
    Args:
        filename: Benchmark filename (e.g., 'benchmark_results_MLFM-R_QAOA.json')
        
    Returns:
        Circuit type string (e.g., 'QAOA')
    """
    # Remove prefix and suffix
    name = filename.replace('benchmark_results_', '').replace('.json', '')
    # Split by underscore and take everything after the method name
    parts = name.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return name


def process_benchmark_directory(input_dir, output_file):
    """
    Process all JSON files in a directory and generate summary report.
    
    Args:
        input_dir: Directory containing benchmark JSON files
        output_file: Output file path for the summary report
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find only detailed results files (not means)
    json_files = list(input_path.glob("benchmark_results_*.json"))
    
    if not json_files:
        print(f"No benchmark results JSON files found in {input_dir}")
        return
    
    print(f"Processing {len(json_files)} benchmark files from {input_dir}")
    
    # Calculate fractions for each file, grouped by circuit type
    circuit_type_results = defaultdict(list)
    
    for filepath in sorted(json_files):
        fractions = calculate_fractions_for_file(filepath)
        if fractions:
            circuit_type = extract_circuit_type(filepath.name)
            circuit_type_results[circuit_type].append(fractions)
    
    if not circuit_type_results:
        print("No valid results to process")
        return
    
    # Aggregate results by circuit type
    circuit_type_averages = {}
    for circuit_type, results_list in circuit_type_results.items():
        avg_cost = sum(r['cost_fraction'] for r in results_list) / len(results_list)
        avg_time = sum(r['time_fraction'] for r in results_list) / len(results_list)
        total_samples = sum(r['num_samples'] for r in results_list)
        
        circuit_type_averages[circuit_type] = {
            'cost_fraction': avg_cost,
            'time_fraction': avg_time,
            'num_samples': total_samples,
            'num_files': len(results_list)
        }
    
    # Calculate overall averages
    overall_avg_cost = sum(data['cost_fraction'] for data in circuit_type_averages.values()) / len(circuit_type_averages)
    overall_avg_time = sum(data['time_fraction'] for data in circuit_type_averages.values()) / len(circuit_type_averages)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("E-bit Fraction Analysis by Circuit Type\n")
        f.write("=" * 80 + "\n\n")
        
        # Write table header
        f.write(f"{'Circuit Type':<20} {'Cost Fraction':>15} {'Time Fraction':>15} {'Num Samples':>12}\n")
        f.write("-" * 80 + "\n")
        
        # Write circuit type results (sorted alphabetically)
        for circuit_type in sorted(circuit_type_averages.keys()):
            data = circuit_type_averages[circuit_type]
            f.write(f"{circuit_type:<20} {data['cost_fraction']:>15.6f} {data['time_fraction']:>15.6f} {data['num_samples']:>12}\n")
        
        # Write separator and average
        f.write("-" * 80 + "\n")
        f.write(f"{'Average':<20} {overall_avg_cost:>15.6f} {overall_avg_time:>15.6f} {'':<12}\n")
        f.write("=" * 80 + "\n")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("E-bit Fraction Analysis by Circuit Type")
    print(f"{'='*80}\n")
    print(f"{'Circuit Type':<20} {'Cost Fraction':>15} {'Time Fraction':>15} {'Num Samples':>12}")
    print("-" * 80)
    
    for circuit_type in sorted(circuit_type_averages.keys()):
        data = circuit_type_averages[circuit_type]
        print(f"{circuit_type:<20} {data['cost_fraction']:>15.6f} {data['time_fraction']:>15.6f} {data['num_samples']:>12}")
    
    print("-" * 80)
    print(f"{'Average':<20} {overall_avg_cost:>15.6f} {overall_avg_time:>15.6f} {'':<12}")
    print("=" * 80)
    
    print(f"\nResults written to {output_file}")
    print(f"Total circuit types processed: {len(circuit_type_averages)}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate e-bit fractions from benchmark results"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing benchmark JSON files (default: data)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ebit_fractions_MLFM-R.dat",
        help="Output file for results (default: ebit_fractions_MLFM-R.dat)"
    )
    
    args = parser.parse_args()
    
    process_benchmark_directory(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
