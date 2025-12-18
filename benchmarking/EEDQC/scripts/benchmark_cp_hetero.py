"""
CP Circuit Benchmarking with Heterogeneous Networks

This script benchmarks the MLFM-R algorithm on CP fraction circuits
with heterogeneous quantum networks (linear, grid, random topologies).

Usage:
    python benchmark_cp_hetero.py --config benchmark_config.yaml
"""

import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from multiprocessing import Pool

from qiskit import transpile

from disqco.circuits.cp_fraction import cp_fraction
from disqco import QuantumCircuitHyperGraph
from disqco import QuantumNetwork
from disqco.graphs.quantum_network import linear_coupling, grid_coupling, random_coupling
from disqco import set_initial_partition_assignment
from disqco.parti import FiducciaMattheyses
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


def _run_single_iteration(args):
    """Helper function to run a single iteration (for multiprocessing)."""
    (num_qubits, num_partitions, fraction, network_type, 
     iteration, passes_per_level, basis_gates, stochastic) = args
    
    # Create circuit
    circuit = cp_fraction(num_qubits, num_qubits, fraction=fraction)
    circuit = transpile(circuit, basis_gates=basis_gates)
    num_two_qubit_gates = circuit.count_ops().get('cp', 0)
    
    # Create heterogeneous network
    qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
    
    if network_type == 'linear':
        coupling = linear_coupling(num_partitions)
    elif network_type == 'grid':
        coupling = grid_coupling(num_partitions)
    elif network_type == 'random':
        coupling = random_coupling(num_partitions)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    network = QuantumNetwork(qpu_info, coupling)
    
    # Create hypergraph and initial assignment
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    initial_assignment = set_initial_partition_assignment(hypergraph, network)
    
    # Run MLFM-R with recursive coarsening
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    partitioner = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=hypergraph)
    
    start_time = time.time()
    results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=passes_per_level,
        stochastic=stochastic
    )
    elapsed_time = time.time() - start_time
    
    return {
        "num_qubits": num_qubits,
        "num_partitions": num_partitions,
        "fraction": fraction,
        "network_type": network_type,
        "iteration": iteration,
        "num_two_qubit_gates": num_two_qubit_gates,
        "cost": results['best_cost'],
        "time": elapsed_time
    }


def run_cp_hetero_benchmark(config, output_dir):
    """
    Run CP heterogeneous network benchmarks.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    cp_config = config.get('cp_hetero', {})
    algo_config = config['algorithm']
    num_iterations = config.get('num_iterations', 5)
    num_processes = config.get('num_processes', 1)
    
    sizes = cp_config.get('sizes', range(112, 257, 16))
    num_partitions_list = cp_config.get('num_partitions', [4, 6, 8])
    fraction = cp_config.get('fraction', 0.5)
    network_types = cp_config.get('network_types', ['linear', 'grid', 'random'])
    
    passes_per_level = algo_config.get('passes_per_level', 10)
    basis_gates = algo_config.get('basis_gates', ['u', 'cp'])
    stochastic = algo_config.get('stochastic', True)
    
    detailed_filename = os.path.join(output_dir, "benchmark_results_EEDQC_CP_hetero.json")
    means_filename = os.path.join(output_dir, "benchmark_means_EEDQC_CP_hetero.json")
    
    # Load existing results if available
    if os.path.exists(detailed_filename):
        with open(detailed_filename, "r") as f:
            detailed_results = json.load(f)
    else:
        detailed_results = []
    
    if os.path.exists(means_filename):
        with open(means_filename, "r") as f:
            mean_results = json.load(f)
    else:
        mean_results = []
    
    # Build configuration list
    configs_to_run = []
    for network_type in network_types:
        for num_partitions in num_partitions_list:
            for num_qubits in sizes:
                configs_to_run.append((num_qubits, num_partitions, fraction, network_type))
    
    print(f"\nRunning CP Heterogeneous Network Benchmarks")
    print(f"Network types: {network_types}")
    print(f"Configurations: {len(configs_to_run)}, Iterations per config: {num_iterations}")
    
    # Run benchmarks
    for num_qubits, num_partitions, fraction, network_type in tqdm(configs_to_run, desc="CP Hetero Configs"):
        
        # Prepare arguments for all iterations
        iter_args = [
            (num_qubits, num_partitions, fraction, network_type, 
             iteration, passes_per_level, basis_gates, stochastic)
            for iteration in range(num_iterations)
        ]
        
        # Run iterations (parallel or serial)
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                iteration_results = list(tqdm(
                    pool.imap(_run_single_iteration, iter_args),
                    total=num_iterations,
                    desc=f"  {num_qubits}q/{num_partitions}p/{network_type}",
                    leave=False
                ))
        else:
            iteration_results = []
            for args in tqdm(iter_args, desc=f"  {num_qubits}q/{num_partitions}p/{network_type}", leave=False):
                iteration_results.append(_run_single_iteration(args))
        
        # Store detailed results
        detailed_results.extend(iteration_results)
        with open(detailed_filename, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        # Compute and store means
        costs = [r['cost'] for r in iteration_results]
        times = [r['time'] for r in iteration_results]
        
        mean_entry = {
            "num_qubits": num_qubits,
            "num_partitions": num_partitions,
            "fraction": fraction,
            "network_type": network_type,
            "num_two_qubit_gates": iteration_results[0]['num_two_qubit_gates'],
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times))
        }
        
        mean_results.append(mean_entry)
        with open(means_filename, "w") as f:
            json.dump(mean_results, f, indent=2)
        
        # Log results
        tqdm.write(f"\n{num_qubits}q/{num_partitions}p/{network_type}:")
        tqdm.write(f"  Cost: {mean_entry['mean_cost']:.1f} ± {mean_entry['std_cost']:.1f}")
        tqdm.write(f"  Time: {mean_entry['mean_time']:.3f} ± {mean_entry['std_time']:.3f}s")
    
    print(f"\nResults saved to {detailed_filename}")
    print(f"Means saved to {means_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run CP Heterogeneous Network Benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to benchmark configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save benchmark results"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    run_cp_hetero_benchmark(config, args.output_dir)


if __name__ == "__main__":
    main()
