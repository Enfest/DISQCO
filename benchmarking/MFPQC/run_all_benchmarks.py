#!/usr/bin/env python3
"""
Consolidated Benchmark Runner for MLFM-R Algorithm

This script runs all benchmarks defined in benchmark_config.yaml and saves
results to the data/ directory. Individual benchmark modules are in scripts/.
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

from scripts.benchmark_cp_large import run_cp_large_benchmark
from scripts.benchmark_cp_scaling import run_cp_scaling_benchmark
from scripts.benchmark_qaoa import run_qaoa_benchmark
from scripts.benchmark_qft import run_qft_benchmark
from scripts.benchmark_qv import run_qv_benchmark
from scripts.benchmark_qasm_50 import run_qasm_benchmark_50
from scripts.benchmark_qasm_100 import run_qasm_benchmark_100
from scripts.benchmark_coarsener_comparison import run_coarsener_comparison


def load_config(config_path="benchmark_config.yaml"):
    """Load benchmark configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_metadata(output_dir, config):
    """Save benchmark run metadata."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
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
    print("MLFM-R Benchmarking Suite")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = config.get('output_dir', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata = save_metadata(output_dir, config)
    
    start_time = time.time()
    
    # Run CP Scaling Benchmarks
    if config.get('cp_scaling', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running CP Scaling Benchmarks (2-12 partitions)")
        print("="*70)
        run_cp_scaling_benchmark(config, output_dir)
    
    # Run CP Large Scale Benchmarks
    if config.get('cp_large', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running CP Large Scale Benchmarks (2-4 partitions)")
        print("="*70)
        run_cp_large_benchmark(config, output_dir)
    
    # Run Standard Circuit Benchmarks
    if config.get('standard_circuits', {}).get('enabled', False):
        circuit_types = config['standard_circuits'].get('circuit_types', [])
        
        if 'QAOA' in circuit_types:
            print("\n" + "="*70)
            print("Running QAOA Benchmarks")
            print("="*70)
            run_qaoa_benchmark(config, output_dir)
        
        if 'QFT' in circuit_types:
            print("\n" + "="*70)
            print("Running QFT Benchmarks")
            print("="*70)
            run_qft_benchmark(config, output_dir)
        
        if 'QV' in circuit_types:
            print("\n" + "="*70)
            print("Running Quantum Volume Benchmarks")
            print("="*70)
            run_qv_benchmark(config, output_dir)
    
    # Run QASMBench Benchmarks
    if config.get('qasm_circuits', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running QASMBench Benchmarks")
        print("="*70)
        qasmbench_path = config.get('qasm_circuits', {}).get('qasmbench_path', 'QASMBench')
        run_qasm_benchmark_50(config, output_dir, qasmbench_path)
        run_qasm_benchmark_100(config, output_dir, qasmbench_path)
    
    # Run Coarsener Comparison Benchmarks
    if config.get('coarsener_comparison', {}).get('enabled', False):
        print("\n" + "="*70)
        print("Running Coarsener Comparison Benchmarks")
        print("="*70)
        run_coarsener_comparison(config, output_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"All benchmarks completed in {total_time/60:.2f} minutes")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
