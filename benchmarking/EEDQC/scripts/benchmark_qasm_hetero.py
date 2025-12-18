"""
QASMBench Benchmarking with Heterogeneous Networks

This script benchmarks the MLFM-R algorithm on QASMBench circuits
with heterogeneous quantum networks (linear, grid, random topologies).

Usage:
    python benchmark_qasm_hetero.py --config benchmark_config.yaml
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

from disqco import QuantumCircuitHyperGraph
from disqco import QuantumNetwork
from disqco.graphs.quantum_network import linear_coupling, grid_coupling, random_coupling
from disqco import set_initial_partition_assignment
from disqco.parti import FiducciaMattheyses
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


def _run_single_iteration(args):
    """Helper function to run a single iteration (for multiprocessing)."""
    (circuit, circuit_name, num_qubits, num_partitions, network_type,
     iteration, passes_per_level, basis_gates, stochastic) = args
    
    try:
        # Transpile circuit
        circuit = transpile(circuit, basis_gates=basis_gates)
        num_two_qubit_gates = circuit.count_ops().get('cp', 0)
        
        # Create heterogeneous network
        qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
        
        if network_type == 'linear':
            coupling = linear_coupling(num_partitions)
        elif network_type == 'grid':
            coupling = grid_coupling(num_partitions)
        elif network_type == 'random':
            coupling = random_coupling(num_partitions, p=0.5)
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
            "circuit_name": circuit_name,
            "num_qubits": num_qubits,
            "num_partitions": num_partitions,
            "network_type": network_type,
            "iteration": iteration,
            "num_two_qubit_gates": num_two_qubit_gates,
            "cost": results['best_cost'],
            "time": elapsed_time,
            "success": True
        }
    except Exception as e:
        return {
            "circuit_name": circuit_name,
            "num_qubits": num_qubits,
            "num_partitions": num_partitions,
            "network_type": network_type,
            "iteration": iteration,
            "error": str(e),
            "success": False
        }


def run_qasm_hetero_benchmark(config, output_dir, qasmbench_path):
    """
    Run QASMBench heterogeneous network benchmarks.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
        qasmbench_path: Path to QASMBench directory
    """
    from QASMBench.interface.qiskit import QASMBenchmark
    
    qasm_config = config.get('qasm_hetero', {})
    algo_config = config['algorithm']
    num_iterations = config.get('num_iterations', 5)
    num_processes = config.get('num_processes', 1)
    
    category = qasm_config.get('category', 'large')
    max_qubits = qasm_config.get('max_qubits', 100)
    max_depth = qasm_config.get('max_depth', 1000)
    skip_circuits = set(qasm_config.get('skip', []))
    num_partitions_list = qasm_config.get('num_partitions', [6, 8])
    network_types = qasm_config.get('network_types', ['linear', 'grid'])
    
    passes_per_level = algo_config.get('passes_per_level', 10)
    basis_gates = algo_config.get('basis_gates', ['u', 'cp'])
    stochastic = algo_config.get('stochastic', True)
    
    detailed_filename = os.path.join(output_dir, "benchmark_results_EEDQC_QASM_hetero.json")
    means_filename = os.path.join(output_dir, "benchmark_means_EEDQC_QASM_hetero.json")
    
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
    
    # Load QASMBench circuits
    print(f"Loading QASMBench circuits from {qasmbench_path}...")
    num_qubits_list = list(range(1, max_qubits + 1))
    bm = QASMBenchmark(
        qasmbench_path,
        category,
        num_qubits_list=num_qubits_list,
        do_transpile=True,
        basis_gates=basis_gates,
        remove_final_measurements=True
    )
    
    # Filter circuits
    circuit_list = []
    for circ_name in bm.circ_name_list:
        if any(skip in circ_name for skip in skip_circuits):
            continue
        circ = bm.get(circ_name)
        if circ.depth() < max_depth:
            circuit_list.append((circ_name, circ, circ.num_qubits))
    
    print(f"\nRunning QASMBench Heterogeneous Network Benchmarks")
    print(f"Circuits: {len(circuit_list)}, Network types: {network_types}")
    print(f"Partitions: {num_partitions_list}, Iterations: {num_iterations}")
    
    # Run benchmarks
    for circ_name, circuit, num_qubits in tqdm(circuit_list, desc="QASM Circuits"):
        for network_type in network_types:
            for num_partitions in num_partitions_list:
                
                # Prepare arguments for all iterations
                iter_args = [
                    (circuit, circ_name, num_qubits, num_partitions, network_type,
                     iteration, passes_per_level, basis_gates, stochastic)
                    for iteration in range(num_iterations)
                ]
                
                # Run iterations (parallel or serial)
                if num_processes > 1:
                    with Pool(processes=num_processes) as pool:
                        iteration_results = list(tqdm(
                            pool.imap(_run_single_iteration, iter_args),
                            total=num_iterations,
                            desc=f"  {circ_name}/{num_partitions}p/{network_type}",
                            leave=False
                        ))
                else:
                    iteration_results = []
                    for args in tqdm(iter_args, desc=f"  {circ_name}/{num_partitions}p/{network_type}", leave=False):
                        iteration_results.append(_run_single_iteration(args))
                
                # Filter successful results
                successful_results = [r for r in iteration_results if r.get('success', False)]
                
                if not successful_results:
                    tqdm.write(f"Skipped {circ_name}/{num_partitions}p/{network_type}: all iterations failed")
                    continue
                
                # Store detailed results
                detailed_results.extend(successful_results)
                with open(detailed_filename, "w") as f:
                    json.dump(detailed_results, f, indent=2)
                
                # Compute and store means
                costs = [r['cost'] for r in successful_results]
                times = [r['time'] for r in successful_results]
                
                mean_entry = {
                    "circuit_name": circ_name,
                    "num_qubits": num_qubits,
                    "num_partitions": num_partitions,
                    "network_type": network_type,
                    "num_two_qubit_gates": successful_results[0]['num_two_qubit_gates'],
                    "mean_cost": float(np.mean(costs)),
                    "std_cost": float(np.std(costs)),
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "num_successful": len(successful_results)
                }
                
                mean_results.append(mean_entry)
                with open(means_filename, "w") as f:
                    json.dump(mean_results, f, indent=2)
                
                # Log results
                tqdm.write(f"{circ_name}/{num_partitions}p/{network_type}: "
                          f"Cost={mean_entry['mean_cost']:.1f}±{mean_entry['std_cost']:.1f}, "
                          f"Time={mean_entry['mean_time']:.3f}±{mean_entry['std_time']:.3f}s")
    
    print(f"\nResults saved to {detailed_filename}")
    print(f"Means saved to {means_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run QASMBench Heterogeneous Network Benchmarks")
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
    parser.add_argument(
        "--qasmbench_path",
        type=str,
        default="QASMBench",
        help="Path to QASMBench directory"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    run_qasm_hetero_benchmark(config, args.output_dir, args.qasmbench_path)


if __name__ == "__main__":
    main()
