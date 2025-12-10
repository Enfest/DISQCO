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
    
    # Calculate fractions for each file
    results = {}
    for filepath in sorted(json_files):
        fractions = calculate_fractions_for_file(filepath)
        if fractions:
            results[filepath.name] = fractions
    
    if not results:
        print("No valid results to process")
        return
    
    # Write results to output file
    with open(output_file, 'w') as f:
        f.write("Circuit name, Cost fraction, Time fraction, Num samples\n")
        
        cumulative_cost = 0
        cumulative_time = 0
        total_samples = 0
        
        for filename, data in results.items():
            f.write(f"{filename}, {data['cost_fraction']:.6f}, {data['time_fraction']:.6f}, {data['num_samples']}\n")
            cumulative_cost += data['cost_fraction']
            cumulative_time += data['time_fraction']
            total_samples += 1
        
        # Calculate and write averages
        avg_cost = cumulative_cost / total_samples
        avg_time = cumulative_time / total_samples
        f.write(f"\nAverage, {avg_cost:.6f}, {avg_time:.6f}, {total_samples}\n")
    
    print(f"\nResults written to {output_file}")
    print(f"Average cost fraction: {avg_cost:.6f}")
    print(f"Average time fraction: {avg_time:.6f}")
    print(f"Total benchmarks processed: {total_samples}")


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
