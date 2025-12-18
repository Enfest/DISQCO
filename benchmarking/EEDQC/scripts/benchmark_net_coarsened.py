"""
Net-Coarsened Partitioning Benchmarking

This script benchmarks net-coarsened partitioning on circuits with
heterogeneous quantum networks. Net coarsening enables efficient
partitioning of large networks with complex topologies.

Usage:
    python benchmark_net_coarsened.py --config benchmark_config.yaml
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
from qiskit.circuit.library import QFT, QuantumVolume

from disqco.circuits.cp_fraction import cp_fraction
from disqco import QuantumCircuitHyperGraph
from disqco import QuantumNetwork
from disqco.graphs.quantum_network import linear_coupling, grid_coupling, random_coupling
from disqco.parti import FiducciaMattheyses
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


def _run_single_iteration(args):
    """Helper function to run a single iteration (for multiprocessing)."""
    (circuit_type, num_qubits, num_qpus, network_type, coarsening_factor,
     iteration, passes_per_level, basis_gates, use_multiprocessing) = args
    
    # Create circuit
    if circuit_type == 'cp_fraction':
        circuit = cp_fraction(num_qubits, depth=num_qubits, fraction=0.5)
    elif circuit_type == 'QFT':
        circuit = QFT(num_qubits, do_swaps=False)
    elif circuit_type == 'QV':
        circuit = QuantumVolume(num_qubits, depth=num_qubits)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    # Transpile circuit
    circuit = transpile(circuit, basis_gates=basis_gates)
    num_two_qubit_gates = circuit.count_ops().get('cp', 0)
    
    # Create heterogeneous network
    qpu_capacity = int(np.ceil(num_qubits / num_qpus)) + 1
    qpu_sizes = [qpu_capacity] * num_qpus
    
    if network_type == 'linear':
        connectivity = linear_coupling(num_qpus)
    elif network_type == 'grid':
        connectivity = grid_coupling(num_qpus)
    elif network_type == 'random':
        connectivity = random_coupling(num_qpus, p=0.5)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    network = QuantumNetwork(qpu_sizes, connectivity)
    
    # Create hypergraph
    hypergraph = QuantumCircuitHyperGraph(circuit)
    
    # Run net-coarsened partitioning
    partitioner = FiducciaMattheyses(circuit, network=network)
    hypergraph_coarsener = HypergraphCoarsener().coarsen_recursive_subgraph_batch
    
    start_time = time.time()
    results = partitioner.net_coarsened_partition(
        coarsening_factor=coarsening_factor,
        hypergraph_coarsener=hypergraph_coarsener,
        passes_per_level=passes_per_level,
        use_multiprocessing=use_multiprocessing
    )
    elapsed_time = time.time() - start_time
    
    return {
        "circuit_type": circuit_type,
        "num_qubits": num_qubits,
        "num_qpus": num_qpus,
        "network_type": network_type,
        "coarsening_factor": coarsening_factor,
        "iteration": iteration,
        "num_two_qubit_gates": num_two_qubit_gates,
        "cost": results['best_cost'],
        "time": elapsed_time
    }


def run_net_coarsened_benchmark(config, output_dir):
    """
    Run net-coarsened partitioning benchmarks.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    nc_config = config.get('net_coarsened', {})
    algo_config = config['algorithm']
    num_iterations = config.get('num_iterations', 5)
    num_processes = config.get('num_processes', 1)
    
    circuit_types = nc_config.get('circuit_types', ['cp_fraction', 'QFT'])
    sizes = nc_config.get('sizes', [64, 96, 128])
    num_qpus_list = nc_config.get('num_qpus', [8, 16])
    network_types = nc_config.get('network_types', ['linear', 'grid'])
    coarsening_factors = nc_config.get('coarsening_factors', [2, 4])
    
    passes_per_level = algo_config.get('passes_per_level', 10)
    basis_gates = algo_config.get('basis_gates', ['u', 'cp'])
    use_multiprocessing = nc_config.get('use_multiprocessing', True)
    
    detailed_filename = os.path.join(output_dir, "benchmark_results_EEDQC_net_coarsened.json")
    means_filename = os.path.join(output_dir, "benchmark_means_EEDQC_net_coarsened.json")
    
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
    for circuit_type in circuit_types:
        for network_type in network_types:
            for num_qpus in num_qpus_list:
                for num_qubits in sizes:
                    for cf in coarsening_factors:
                        configs_to_run.append((circuit_type, num_qubits, num_qpus, network_type, cf))
    
    print(f"\nRunning Net-Coarsened Partitioning Benchmarks")
    print(f"Circuit types: {circuit_types}, Network types: {network_types}")
    print(f"Configurations: {len(configs_to_run)}, Iterations per config: {num_iterations}")
    
    # Run benchmarks
    for circuit_type, num_qubits, num_qpus, network_type, cf in tqdm(configs_to_run, desc="Net-Coarsened Configs"):
        
        # Prepare arguments for all iterations
        iter_args = [
            (circuit_type, num_qubits, num_qpus, network_type, cf,
             iteration, passes_per_level, basis_gates, use_multiprocessing)
            for iteration in range(num_iterations)
        ]
        
        # Run iterations (parallel or serial)
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                iteration_results = list(tqdm(
                    pool.imap(_run_single_iteration, iter_args),
                    total=num_iterations,
                    desc=f"  {circuit_type}/{num_qubits}q/{num_qpus}qpus/{network_type}/cf{cf}",
                    leave=False
                ))
        else:
            iteration_results = []
            for args in tqdm(iter_args, desc=f"  {circuit_type}/{num_qubits}q/{num_qpus}qpus/{network_type}/cf{cf}", leave=False):
                iteration_results.append(_run_single_iteration(args))
        
        # Store detailed results
        detailed_results.extend(iteration_results)
        with open(detailed_filename, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        # Compute and store means
        costs = [r['cost'] for r in iteration_results]
        times = [r['time'] for r in iteration_results]
        
        mean_entry = {
            "circuit_type": circuit_type,
            "num_qubits": num_qubits,
            "num_qpus": num_qpus,
            "network_type": network_type,
            "coarsening_factor": cf,
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
        tqdm.write(f"\n{circuit_type}/{num_qubits}q/{num_qpus}qpus/{network_type}/cf{cf}:")
        tqdm.write(f"  Cost: {mean_entry['mean_cost']:.1f} ± {mean_entry['std_cost']:.1f}")
        tqdm.write(f"  Time: {mean_entry['mean_time']:.3f} ± {mean_entry['std_time']:.3f}s")
    
    print(f"\nResults saved to {detailed_filename}")
    print(f"Means saved to {means_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run Net-Coarsened Partitioning Benchmarks")
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
    run_net_coarsened_benchmark(config, args.output_dir)


if __name__ == "__main__":
    main()
