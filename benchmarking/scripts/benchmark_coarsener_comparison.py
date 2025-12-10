"""
Coarsener Comparison Benchmarking Script

Compares different coarsening strategies:
- Fine-grained (f): No coarsening
- Window-based (w): Window coarsening
- Block-based (b): Block coarsening  
- Recursive (r): Recursive coarsening (batches mapped)

Usage:
    python benchmark_coarsener_comparison.py --config benchmark_config.yaml
"""

import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

from qiskit import transpile
from qiskit.circuit.library import QFT, QuantumVolume

from disqco.circuits.cp_fraction import cp_fraction
from disqco.circuits.QAOA import QAOA_random
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.FM.fiduccia import FiducciaMattheyses
from disqco.parti.FM.FM_methods import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


def run_coarsener_comparison(config, output_dir):
    """
    Run benchmarks comparing different coarsening strategies.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    comp_config = config.get('coarsener_comparison', {})
    algo_config = config['algorithm']
    num_iterations = config.get('num_iterations', 5)
    
    # Get circuit parameters
    circuit_type = comp_config.get('circuit_type', 'cp_fraction')
    sizes = comp_config.get('sizes', [32])
    num_partitions_list = comp_config.get('num_partitions', [4])
    passes_per_level = algo_config.get('passes_per_level', 10)
    stochastic = algo_config.get('stochastic', True)
    
    detailed_filename = os.path.join(output_dir, "benchmark_results_coarsener_comparison.json")
    
    # Load existing results if available
    if os.path.exists(detailed_filename):
        with open(detailed_filename, "r") as f:
            detailed_results = json.load(f)
    else:
        detailed_results = []
    
    # Circuit-specific parameters
    if circuit_type == 'cp_fraction':
        fractions = comp_config.get('fractions', [0.5])
    else:
        fractions = [None]  # Not applicable for other circuits
    
    # Build configuration list
    configs_to_run = []
    for num_qubits in sizes:
        for num_partitions in num_partitions_list:
            for fraction in fractions:
                configs_to_run.append((num_qubits, num_partitions, fraction))
    
    print(f"\nComparing coarsening strategies: Fine-grained, Window, Block, Recursive")
    print(f"Circuit type: {circuit_type}")
    print(f"Configurations: {len(configs_to_run)}, Iterations per config: {num_iterations}")
    
    # Run benchmarks with progress bar
    for num_qubits, num_partitions, fraction in tqdm(configs_to_run, desc="Coarsener Comparison"):
        
        # Create network
        qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
        network = QuantumNetwork(qpu_info)
        
        for iteration in tqdm(range(num_iterations), desc=f"  {num_qubits}q/{num_partitions}p", leave=False):
            
            # Create circuit based on type
            if circuit_type == 'cp_fraction':
                circuit = cp_fraction(num_qubits, num_qubits, fraction=fraction)
            elif circuit_type == 'QFT':
                circuit = QFT(num_qubits, do_swaps=False)
            elif circuit_type == 'QV':
                circuit = QuantumVolume(num_qubits, num_qubits)
            elif circuit_type == 'QAOA':
                circuit = QAOA_random(num_qubits, prob=0.5, reps=1)
            else:
                raise ValueError(f"Unknown circuit type: {circuit_type}")
            
            # Transpile and create hypergraph
            circuit = transpile(circuit, basis_gates=algo_config['basis_gates'])
            num_two_qubit_gates = circuit.count_ops().get('cp', 0)
            base_graph = QuantumCircuitHyperGraph(circuit, group_gates=algo_config['group_gates'])
            initial_assignment = set_initial_partition_assignment(base_graph, network)
            
            # Calculate number of levels for multilevel approaches
            depth = base_graph.depth
            num_levels = int(np.ceil(np.log2(depth)))
            
            # -------------------------
            # 1. Fine-grained (no coarsening)
            # -------------------------
            partitioner_f = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=base_graph)
            start = time.time()
            results_f = partitioner_f.multilevel_partition(
                coarsener=None,  # No coarsening
                passes_per_level=passes_per_level * (num_levels + 1),
                stochastic=stochastic
            )
            time_f = time.time() - start
            cost_f = results_f['best_cost']
            
            # -------------------------
            # 2. Window-based coarsening
            # -------------------------
            coarsener_w = HypergraphCoarsener().coarsen_window
            partitioner_w = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=base_graph)
            start = time.time()
            results_w = partitioner_w.multilevel_partition(
                coarsener=coarsener_w,
                passes_per_level=passes_per_level,
                stochastic=stochastic
            )
            time_w = time.time() - start
            cost_w = results_w['best_cost']
            
            # -------------------------
            # 3. Block-based coarsening
            # -------------------------
            coarsener_b = HypergraphCoarsener().coarsen_blocks
            partitioner_b = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=base_graph)
            start = time.time()
            results_b = partitioner_b.multilevel_partition(
                coarsener=coarsener_b,
                passes_per_level=passes_per_level,
                stochastic=stochastic
            )
            time_b = time.time() - start
            cost_b = results_b['best_cost']
            
            # -------------------------
            # 4. Recursive coarsening (batches mapped)
            # -------------------------
            coarsener_r = HypergraphCoarsener().coarsen_recursive_batches_mapped
            partitioner_r = FiducciaMattheyses(circuit, network, initial_assignment, hypergraph=base_graph)
            start = time.time()
            results_r = partitioner_r.multilevel_partition(
                coarsener=coarsener_r,
                passes_per_level=passes_per_level,
                stochastic=stochastic
            )
            time_r = time.time() - start
            cost_r = results_r['best_cost']
            
            # Store results
            result_entry = {
                "circuit_type": circuit_type,
                "num_qubits": num_qubits,
                "num_partitions": num_partitions,
                "fraction": fraction,
                "num_two_qubit_gates": num_two_qubit_gates,
                "iteration": iteration,
                "cost_f": cost_f,
                "cost_w": cost_w,
                "cost_b": cost_b,
                "cost_r": cost_r,
                "time_f": time_f,
                "time_w": time_w,
                "time_b": time_b,
                "time_r": time_r,
            }
            
            detailed_results.append(result_entry)
            
            # Save after each iteration
            with open(detailed_filename, "w") as f:
                json.dump(detailed_results, f, indent=2)
            
            # Log results
            tqdm.write(f"\n{num_qubits}q/{num_partitions}p iter {iteration}:")
            tqdm.write(f"  Costs - F: {cost_f:.1f}, W: {cost_w:.1f}, B: {cost_b:.1f}, R: {cost_r:.1f}")
            tqdm.write(f"  Times - F: {time_f:.2f}s, W: {time_w:.2f}s, B: {time_b:.2f}s, R: {time_r:.2f}s")
    
    # Compute summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    # Group by configuration
    from collections import defaultdict
    stats = defaultdict(lambda: {'costs_f': [], 'costs_w': [], 'costs_b': [], 'costs_r': [],
                                   'times_f': [], 'times_w': [], 'times_b': [], 'times_r': []})
    
    for entry in detailed_results:
        key = (entry['num_qubits'], entry['num_partitions'], entry.get('fraction'))
        stats[key]['costs_f'].append(entry['cost_f'])
        stats[key]['costs_w'].append(entry['cost_w'])
        stats[key]['costs_b'].append(entry['cost_b'])
        stats[key]['costs_r'].append(entry['cost_r'])
        stats[key]['times_f'].append(entry['time_f'])
        stats[key]['times_w'].append(entry['time_w'])
        stats[key]['times_b'].append(entry['time_b'])
        stats[key]['times_r'].append(entry['time_r'])
    
    for key, data in stats.items():
        nq, nparts, frac = key
        print(f"\nConfiguration: {nq} qubits, {nparts} partitions" + 
              (f", fraction={frac}" if frac else ""))
        print(f"  Mean Costs - F: {np.mean(data['costs_f']):.1f}, W: {np.mean(data['costs_w']):.1f}, "
              f"B: {np.mean(data['costs_b']):.1f}, R: {np.mean(data['costs_r']):.1f}")
        print(f"  Mean Times - F: {np.mean(data['times_f']):.2f}s, W: {np.mean(data['times_w']):.2f}s, "
              f"B: {np.mean(data['times_b']):.2f}s, R: {np.mean(data['times_r']):.2f}s")
    
    print(f"\nResults saved to {detailed_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run Coarsener Comparison Benchmarks")
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
    run_coarsener_comparison(config, args.output_dir)


if __name__ == "__main__":
    main()
