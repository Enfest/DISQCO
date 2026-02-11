#!/usr/bin/env python3
"""
EEDQC Benchmark Runner
Energy-Efficient Distributed Quantum Computing Paper

This script runs all benchmarks defined in benchmark_config.yaml for
heterogeneous network partitioning experiments.
"""

import os
import sys
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.benchmark_cp_hetero import run_cp_hetero_benchmark
from scripts.benchmark_qasm_hetero import run_qasm_hetero_benchmark
from scripts.benchmark_net_coarsened import run_net_coarsened_benchmark


def load_config(config_path="benchmark_config.yaml"):
    """Load benchmark configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_metadata(output_dir, config):
    """Save benchmark run metadata."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "paper": "EEDQC - Energy-Efficient Distributed Quantum Computing",
        "config": config,
        "python_version": sys.version,
    }
    
    # Try to get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        metadata["git_commit"] = git_hash
    except:
        metadata["git_commit"] = "unknown"
    
    metadata_file = os.path.join(output_dir, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    return metadata


def main():
    """Run all enabled benchmarks."""
    print("="*70)
    print("EEDQC Benchmarking Suite")
    print("Energy-Efficient Distributed Quantum Computing")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = config.get('output_dir', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata = save_metadata(output_dir, config)
    
    start_time = time.time()
    
    # Run CP Heterogeneous Network Benchmarks
    if config.get('cp_hetero', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running CP Heterogeneous Network Benchmarks")
        print("="*70)
        run_cp_hetero_benchmark(config, output_dir)
    
    # Run QASMBench Heterogeneous Network Benchmarks
    if config.get('qasm_hetero', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running QASMBench Heterogeneous Network Benchmarks")
        print("="*70)
        qasmbench_path = config.get('qasm_hetero', {}).get('qasmbench_path', '../QASMBench')
        run_qasm_hetero_benchmark(config, output_dir, qasmbench_path)
    
    # Run Net-Coarsened Partitioning Benchmarks
    if config.get('net_coarsened', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running Net-Coarsened Partitioning Benchmarks")
        print("="*70)
        run_net_coarsened_benchmark(config, output_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"All EEDQC benchmarks completed in {total_time/60:.2f} minutes")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
