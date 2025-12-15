"""
Generate DAT Files from Benchmark Results

This script processes benchmark JSON files and generates .dat files with
mean, min, and max values for cost and time metrics.

Usage:
    python generate_dat_files.py --input_dir data --output_dir analysis/dat_files
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def determine_benchmark_type(filename):
    """Determine the type of benchmark from filename."""
    name_lower = filename.lower()
    if 'scaling' in name_lower:
        return 'scaling'
    elif 'qasm' in name_lower:
        return 'qasm'
    else:
        return 'standard'


def process_scaling_benchmark(data):
    """Process CP scaling benchmarks - group by fraction, x-axis is num_qubits."""
    configs = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        fraction = entry.get('fraction')
        num_qubits = entry['num_qubits']
        
        config_key = f"fraction_{fraction}"
        
        if 'cost' in entry:
            configs[config_key][('cost', num_qubits)].append(entry['cost'])
        if 'time' in entry:
            configs[config_key][('time', num_qubits)].append(entry['time'])
    
    return configs


def process_qasm_benchmark(data, base_name):
    """Process QASM benchmarks - single file with circuit_name, num_qubits and num_partitions."""
    configs = defaultdict(lambda: defaultdict(list))
    
    config_key = base_name  # Use base name as config key
    
    for entry in data:
        circuit_name = entry.get('circuit_name', 'unknown')
        num_qubits = entry.get('num_qubits')
        num_partitions = entry.get('num_partitions')
        
        if 'cost' in entry:
            configs[config_key][('cost', circuit_name, num_qubits, num_partitions)].append(entry['cost'])
        if 'time' in entry:
            configs[config_key][('time', circuit_name, num_qubits, num_partitions)].append(entry['time'])
    
    return configs


def process_standard_benchmark(data):
    """Process standard benchmarks (QAOA, QFT, QV, CP_large) - group by num_partitions, x-axis is num_qubits."""
    configs = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        num_qubits = entry.get('num_qubits')
        num_partitions = entry.get('num_partitions')
        
        config_key = f"{num_partitions}partitions"
        
        if 'cost' in entry:
            configs[config_key][('cost', num_qubits)].append(entry['cost'])
        if 'time' in entry:
            configs[config_key][('time', num_qubits)].append(entry['time'])
    
    return configs


def write_standard_dat_file(output_path, data_dict, metric_name):
    """Write a standard .dat file with num_qubits as x-axis."""
    # Get sorted list of qubit values
    qubit_values = sorted(set(k[1] for k in data_dict.keys() if k[0] == metric_name))
    
    if not qubit_values:
        return
    
    with open(output_path, 'w') as f:
        # Write header
        prefix = 'r'
        f.write(f"num_qubits {prefix}_mean {prefix}_min {prefix}_max\n")
        
        # Write data for each qubit value
        for num_qubits in qubit_values:
            key = (metric_name, num_qubits)
            if key in data_dict:
                values = data_dict[key]
                
                if not values:
                    continue
                
                mean_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Format based on metric type
                if metric_name == 'time':
                    f.write(f"{num_qubits} {mean_val:.4f} {min_val:.4f} {max_val:.4f}\n")
                else:
                    f.write(f"{num_qubits} {mean_val:.1f} {min_val:.1f} {max_val:.1f}\n")


def write_qasm_dat_file(output_path, data_dict, metric_name):
    """Write a QASM .dat file with num_qubits and num_partitions."""
    # Get all (circuit_name, num_qubits, num_partitions) combinations
    combinations = sorted(set((k[1], k[2], k[3]) for k in data_dict.keys() if k[0] == metric_name))
    
    if not combinations:
        return
    
    with open(output_path, 'w') as f:
        # Write header
        prefix = 'r'
        f.write(f"circuit_name num_qubits num_partitions {prefix}_mean {prefix}_min {prefix}_max\n")
        
        # Write data for each combination
        for circuit_name, num_qubits, num_partitions in combinations:
            key = (metric_name, circuit_name, num_qubits, num_partitions)
            if key in data_dict:
                values = data_dict[key]
                
                if not values:
                    continue
                
                mean_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Format based on metric type
                if metric_name == 'time':
                    f.write(f"{circuit_name} {num_qubits} {num_partitions} {mean_val:.4f} {min_val:.4f} {max_val:.4f}\n")
                else:
                    f.write(f"{circuit_name} {num_qubits} {num_partitions} {mean_val:.1f} {min_val:.1f} {max_val:.1f}\n")


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
    
    for json_file in sorted(json_files):
        print(f"\nProcessing {json_file.name}...")
        
        # Load data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error reading {json_file}: {e}")
            continue
        
        if not data:
            print(f"  Skipped: no valid data")
            continue
        
        # Determine benchmark type
        benchmark_type = determine_benchmark_type(json_file.name)
        base_name = json_file.stem.replace('benchmark_results_MLFM-R_', '')
        
        # Process based on type
        if benchmark_type == 'scaling':
            configs = process_scaling_benchmark(data)
        elif benchmark_type == 'qasm':
            configs = process_qasm_benchmark(data, base_name)
        else:
            configs = process_standard_benchmark(data)
        
        # Generate DAT files for each configuration
        for config_key, data_dict in configs.items():
            if benchmark_type == 'qasm':
                # QASM: Create one cost and one time file
                cost_filename = f"{config_key}_cost.dat"
                cost_path = output_path / cost_filename
                write_qasm_dat_file(cost_path, data_dict, 'cost')
                print(f"  Created: {cost_filename}")
                total_dat_files += 1
                
                time_filename = f"{config_key}_time.dat"
                time_path = output_path / time_filename
                write_qasm_dat_file(time_path, data_dict, 'time')
                print(f"  Created: {time_filename}")
                total_dat_files += 1
            else:
                # Standard and scaling: Create separate cost and time files
                cost_filename = f"{base_name}_{config_key}_cost.dat"
                cost_path = output_path / cost_filename
                write_standard_dat_file(cost_path, data_dict, 'cost')
                print(f"  Created: {cost_filename}")
                total_dat_files += 1
                
                time_filename = f"{base_name}_{config_key}_time.dat"
                time_path = output_path / time_filename
                write_standard_dat_file(time_path, data_dict, 'time')
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
