"""
Generate DAT Files from Benchmark Results

This script processes benchmark JSON files and generates .dat files with
mean, min, and max values for cost and time metrics. Generates one cost
file and one time file per unique configuration in the data.

Usage:
    python generate_dat_files.py --input_dir data --output_dir analysis/dat_files
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def process_json_file(filepath, is_scaling=False):
    """
    Process a single JSON benchmark file and organize data by configuration.
    
    Args:
        filepath: Path to JSON benchmark results file
        is_scaling: If True (CP scaling), group by fraction only (num_partitions scales with num_qubits).
                   If False (other benchmarks), group by num_qubits and fraction.
        
    Returns:
        Dictionary organized by configuration parameters
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    if not data:
        return None
    
    # Organize data by configuration
    configs = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        if is_scaling:
            # For CP scaling: group by fraction only, x-axis is num_qubits
            fraction = entry.get('fraction')
            config_key = ('fraction', fraction)
            num_qubits = entry['num_qubits']
            
            # Store cost and time for this qubit size
            if 'cost' in entry:
                configs[config_key][num_qubits].append(entry['cost'])
            if 'time' in entry:
                configs[config_key][('time', num_qubits)].append(entry['time'])
        else:
            # For other benchmarks: group by fraction and num_qubits, x-axis is num_partitions
            fraction = entry.get('fraction')
            num_qubits = entry.get('num_qubits')
            num_partitions = entry.get('num_partitions')
            config_key = ('fraction', fraction, 'num_qubits', num_qubits)
            
            # Store cost and time for this partition count
            if 'cost' in entry:
                configs[config_key][num_partitions].append(entry['cost'])
            if 'time' in entry:
                configs[config_key][('time', num_partitions)].append(entry['time'])
    
    return configs


def write_dat_file(output_path, data_dict, metric_name='cost', x_axis='num_qubits'):
    """
    Write a .dat file with mean, min, max values.
    
    Args:
        output_path: Path to output .dat file
        data_dict: Dictionary mapping x-axis values to list of values
        metric_name: Name of the metric (for column naming)
        x_axis: Name of the x-axis column ('num_qubits' or 'num_partitions')
    """
    # Get sorted list of x-axis values
    x_values = sorted([k for k in data_dict.keys() if isinstance(k, int)])
    
    if not x_values:
        return
    
    with open(output_path, 'w') as f:
        # Write header
        prefix = metric_name[0] if metric_name else 'r'
        f.write(f"{x_axis} {prefix}_mean {prefix}_min {prefix}_max\n")
        
        # Write data for each x value
        for x_val in x_values:
            values = data_dict[x_val]
            
            if not values:
                continue
            
            mean_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Format based on metric type
            if metric_name == 'time':
                f.write(f"{x_val} {mean_val:.4f} {min_val:.4f} {max_val:.4f}\n")
            else:
                f.write(f"{x_val} {mean_val:.1f} {min_val:.1f} {max_val:.1f}\n")


def generate_filename_from_config(config_tuple, base_name, metric, is_scaling=False):
    """
    Generate a descriptive filename from configuration parameters.
    
    Args:
        config_tuple: Tuple of configuration parameters
        base_name: Base name from the JSON file
        metric: 'cost' or 'time'
        is_scaling: If True (CP scaling), use fraction only in filename
        
    Returns:
        Filename string
    """
    config_dict = dict(zip(config_tuple[::2], config_tuple[1::2]))
    
    parts = []
    
    if is_scaling:
        # For CP scaling: fraction_X_{cost/time}.dat
        if config_dict.get('fraction') is not None:
            parts.append(f"fraction_{config_dict['fraction']}")
    else:
        # For other benchmarks: Xq_fraction_Y_{cost/time}.dat
        if config_dict.get('num_qubits') is not None:
            parts.append(f"{config_dict['num_qubits']}q")
        if config_dict.get('fraction') is not None:
            parts.append(f"fraction_{config_dict['fraction']}")
    
    # If no specific params, use base name
    if not parts:
        parts.append(base_name.replace('benchmark_results_MLFM-R_', '').replace('.json', ''))
    
    # Add metric
    parts.append(metric)
    
    return "_".join(parts) + ".dat"


def process_benchmark_directory(input_dir, output_dir):
    """
    Process all benchmark_results_*.json files in input directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to write .dat files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all benchmark_results_*.json files
    json_files = list(input_path.glob("benchmark_results_*.json"))
    
    if not json_files:
        print(f"No benchmark_results_*.json files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} benchmark result files")
    
    total_dat_files = 0
    
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        
        # Determine if this is a scaling benchmark
        is_scaling = 'scaling' in json_file.name.lower()
        
        configs = process_json_file(json_file, is_scaling=is_scaling)
        
        if not configs:
            print(f"  Skipped: no valid data")
            continue
        
        base_name = json_file.stem
        
        # Determine x-axis based on benchmark type
        x_axis = 'num_qubits' if is_scaling else 'num_partitions'
        
        # Process each configuration
        for config_key, data in configs.items():
            # Separate cost and time data
            cost_data = {k: v for k, v in data.items() if not isinstance(k, tuple)}
            time_data = {k[1]: v for k, v in data.items() if isinstance(k, tuple) and k[0] == 'time'}
            
            # Generate cost .dat file
            if cost_data:
                cost_filename = generate_filename_from_config(config_key, base_name, 'cost', is_scaling=is_scaling)
                cost_path = output_path / cost_filename
                write_dat_file(cost_path, cost_data, 'cost', x_axis=x_axis)
                print(f"  Created: {cost_filename}")
                total_dat_files += 1
            
            # Generate time .dat file
            if time_data:
                time_filename = generate_filename_from_config(config_key, base_name, 'time', is_scaling=is_scaling)
                time_path = output_path / time_filename
                write_dat_file(time_path, time_data, 'time', x_axis=x_axis)
                print(f"  Created: {time_filename}")
                total_dat_files += 1
    
    print(f"\n{'='*70}")
    print(f"Generated {total_dat_files} .dat files in {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DAT files from benchmark JSON results"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing benchmark_results_*.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/dat_files",
        help="Directory to save generated .dat files"
    )
    
    args = parser.parse_args()
    
    process_benchmark_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
