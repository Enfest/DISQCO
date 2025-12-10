"""
QASMBench Circuit Benchmarking Module

Benchmarks for circuits from QASMBench suite.
"""

import os
import json
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from qiskit import transpile
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.FM.fiduccia import FiducciaMattheyses
from disqco.parti.FM.FM_methods import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
import re


def _run_single_iteration(args):
    """
    Run a single iteration of the benchmark.
    Helper function for multiprocessing.
    """
    iteration, circuit, circ_name, num_partitions, algo_config = args
    
    num_qubits = circuit.num_qubits
    
    # Create network
    qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
    network = QuantumNetwork(qpu_info)
    
    # Skip if depth is too large
    if circuit.depth() >= algo_config.get('max_depth', 1000):
        return None
    
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
        "circuit_name": circ_name,
        "num_qubits": num_qubits,
        "depth": circuit.depth(),
        "num_partitions": num_partitions,
        "iteration": iteration,
        "num_two_qubit_gates": num_two_qubit_gates,
        "cost": results['best_cost'],
        "time": elapsed,
    }


def run_qasm_benchmark_100(config, output_dir, qasmbench_path="QASMBench"):
    """
    Run benchmarks for QASMBench circuits.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
        qasmbench_path: Path to QASMBench directory
    """
    try:
        import sys
        sys.path.append(f'./')
        from QASMBench.interface.qiskit import QASMBenchmark
    except ImportError:
        print("Error: QASMBench not found. Please install QASMBench.")
        print("Clone from: https://github.com/pnnl/QASMBench")
        return
    
    qasm_config = config.get('qasm_circuits', {})
    algo_config = config['algorithm']
    num_iterations = config['num_iterations']
    num_processes = config.get('num_processes', 1)
    
    # QASMBench configuration
    category = qasm_config.get('category', 'large')
    min_qubits = 50
    max_qubits = 100
    max_depth = qasm_config.get('max_depth', 1000)
    
    # Initialize QASMBenchmark
    bm = QASMBenchmark(
        qasmbench_path,
        category,
        num_qubits_list=list(range(min_qubits,max_qubits)),
        do_transpile=True,
        basis_gates=algo_config['basis_gates'],
        remove_final_measurements=True
    )
    
    circ_name_list = bm.circ_name_list
    print(f"Found {len(circ_name_list)} circuits in QASMBench category '{category}' with up to {max_qubits} qubits.")
    
    detailed_filename = os.path.join(output_dir, f"benchmark_results_MLFM-R_QASM_{category}.json")
    means_filename = os.path.join(output_dir, f"benchmark_means_MLFM-R_QASM_{category}.json")
    
    # Load existing results if available
    detailed_results = []
    mean_results = []
    if os.path.exists(detailed_filename):
        with open(detailed_filename, "r") as f:
            detailed_results = json.load(f)
    if os.path.exists(means_filename):
        with open(means_filename, "r") as f:
            mean_results = json.load(f)
    
    num_partition_list = qasm_config.get('num_partitions', [2, 3, 4])
    
    # Build list of all circuit configurations
    configs_to_run = []
   
    for circ_name in circ_name_list:
        # Check if circuit name matches any skip pattern
        skip_patterns = qasm_config.get('skip', [])
        if any(re.match(pattern, circ_name) for pattern in skip_patterns):
            print(f"Skipping circuit {circ_name} due to skip pattern.")
            continue
        circ = bm.get(circ_name)
        if circ.depth() >= max_depth:
            continue
        circuit = transpile(circ, basis_gates=algo_config['basis_gates'])
        for num_partitions in num_partition_list:
            print(f"Loading circuit {circ_name}...")
            # Skip circuits that are too deep
            configs_to_run.append((circ_name, circ, num_partitions))
    
    # Run benchmarks with progress bar
    for circ_name, circuit, num_partitions in tqdm(configs_to_run, desc=f"QASMBench {category} circuits"):
        num_qubits = circuit.num_qubits
        
        # Prepare arguments for parallel execution
        iteration_args = [
            (iteration, circuit, circ_name, num_partitions, algo_config)
            for iteration in range(num_iterations)
        ]
        
        # Run iterations in parallel or serial
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                iteration_data = list(tqdm(
                    pool.imap(_run_single_iteration, iteration_args),
                    total=num_iterations,
                    desc=f"  {circ_name}",
                    leave=False
                ))
        else:
            iteration_data = [
                _run_single_iteration(args)
                for args in tqdm(
                    iteration_args,
                    desc=f"  {circ_name}",
                    leave=False
                )
            ]
        
        # Filter out None results (skipped circuits)
        iteration_data = [x for x in iteration_data if x is not None]
        
        if not iteration_data:
            continue
        
        # Store results
        detailed_results.extend(iteration_data)
        
        # Compute means
        mean_cost = float(np.mean([x["cost"] for x in iteration_data]))
        mean_time = float(np.mean([x["time"] for x in iteration_data]))
        
        # Log results (using tqdm.write to avoid interfering with progress bars)
        tqdm.write(f"\n{circ_name} ({num_qubits}q/{num_partitions}p): Mean Cost={mean_cost:.3f}, Mean Time={mean_time:.3f}s")
        tqdm.write(f"  Individual costs: {[f'{x['cost']:.1f}' for x in iteration_data]}")
        
        mean_entry = {
            "circuit_name": circ_name,
            "num_qubits": num_qubits,
            "num_partitions": num_partitions,
            "depth": iteration_data[0]["depth"],
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

    parser = argparse.ArgumentParser(description="Run QASM Circuit Benchmarks")
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
    run_qasm_benchmark(config, args.output_dir)