"""
QAOA Circuit Benchmarking Module

Benchmarks for QAOA circuits.
"""

import os
import json
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from qiskit import transpile
from disqco.circuits.QAOA import QAOA_random
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
from disqco.parti.FM.fiduccia import FiducciaMattheyses
from disqco.parti.FM.FM_methods import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


def _run_single_iteration(args):
    """
    Run a single iteration of the benchmark.
    Helper function for multiprocessing.
    """
    iteration, num_qubits, num_partitions, algo_config, qaoa_config = args
    
    # Create network
    qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
    network = QuantumNetwork(qpu_info)
    
    # Create QAOA circuit
    circuit = QAOA_random(
        num_qubits,
        prob=qaoa_config.get('prob', 0.5),
        reps=qaoa_config.get('reps', 1)
    )
    
    circuit = transpile(circuit, basis_gates=algo_config['basis_gates'])
    num_two_qubit_gates = circuit.count_ops().get('cp', 0)
    base_graph = QuantumCircuitHyperGraph(circuit, group_gates=algo_config['group_gates'])
    initial_assignment = set_initial_partition_assignment(base_graph, network)
    
    # Run MLFM-R
    recursive_coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    passes_per_level = algo_config['passes_per_level']
    
    partitioner = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=base_graph)
    start = time.time()
    results = partitioner.multilevel_partition(
        coarsener=recursive_coarsener,
        passes_per_level=int(passes_per_level)
    )
    elapsed = time.time() - start
    
    return {
        "circuit_type": "QAOA",
        "num_qubits": num_qubits,
        "num_partitions": num_partitions,
        "iteration": iteration,
        "num_two_qubit_gates": num_two_qubit_gates,
        "cost": results['best_cost'],
        "time": elapsed,
    }


def run_qaoa_benchmark(config, output_dir):
    """
    Run benchmarks for QAOA circuits.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    circuit_config = config['standard_circuits']
    algo_config = config['algorithm']
    num_iterations = config['num_iterations']
    num_processes = config.get('num_processes', 1)
    qaoa_config = config.get('qaoa', {})
    
    circuit_type = "QAOA"
    
    detailed_filename = os.path.join(output_dir, f"benchmark_results_MLFM-R_{circuit_type}.json")
    means_filename = os.path.join(output_dir, f"benchmark_means_MLFM-R_{circuit_type}.json")
    
    # Load existing results if available
    detailed_results = []
    mean_results = []
    if os.path.exists(detailed_filename):
        with open(detailed_filename, "r") as f:
            detailed_results = json.load(f)
    if os.path.exists(means_filename):
        with open(means_filename, "r") as f:
            mean_results = json.load(f)
    
    sizes = circuit_config['sizes']
    num_partitions_list = circuit_config['num_partitions']
    
    # Build list of all configurations
    configs_to_run = [
        (num_qubits, num_partitions)
        for num_partitions in num_partitions_list
        for num_qubits in sizes
    ]
    
    # Run benchmarks with progress bar
    for num_qubits, num_partitions in tqdm(configs_to_run, desc=f"{circuit_type} Benchmarks"):
        # Prepare arguments for parallel execution
        iteration_args = [
            (iteration, num_qubits, num_partitions, algo_config, qaoa_config)
            for iteration in range(num_iterations)
        ]
        
        # Run iterations in parallel or serial
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                iteration_data = list(tqdm(
                    pool.imap(_run_single_iteration, iteration_args),
                    total=num_iterations,
                    desc=f"  {num_qubits}q/{num_partitions}p",
                    leave=False
                ))
        else:
            iteration_data = [
                _run_single_iteration(args)
                for args in tqdm(
                    iteration_args,
                    desc=f"  {num_qubits}q/{num_partitions}p",
                    leave=False
                )
            ]
        
        # Store results
        detailed_results.extend(iteration_data)
        
        # Compute means
        mean_cost = float(np.mean([x["cost"] for x in iteration_data]))
        mean_time = float(np.mean([x["time"] for x in iteration_data]))
        
        # Log results (using tqdm.write to avoid interfering with progress bars)
        tqdm.write(f"\n{circuit_type} {num_qubits}q/{num_partitions}p: Mean Cost={mean_cost:.3f}, Mean Time={mean_time:.3f}s")
        tqdm.write(f"  Individual costs: {[f'{x['cost']:.1f}' for x in iteration_data]}")
        
        mean_entry = {
            "circuit_type": circuit_type,
            "num_qubits": num_qubits,
            "num_partitions": num_partitions,
            "num_two_qubit_gates": iteration_data[0]["num_two_qubit_gates"],
            "mean_cost": mean_cost,
            "mean_time": mean_time,
        }
        mean_results.append(mean_entry)
        
        # Save results
        with open(detailed_filename, "w") as f:
            json.dump(detailed_results, f, indent=2)
        with open(means_filename, "w") as f:
            json.dump(mean_results, f, indent=2)
    
    print(f"\nResults saved to {detailed_filename} and {means_filename}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import yaml

    parser = argparse.ArgumentParser(description="Run QAOA Scaling Circuit Benchmarks")
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
    run_qaoa_benchmark(config, args.output_dir)