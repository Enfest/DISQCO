from qiskit import QuantumCircuit
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti.FM.FM_hetero import run_FM_sparse
from disqco.parti.FM.partition_and_build_subgraph import partition_and_build_subgraph
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.FM.FM_methods import calculate_full_cost_hetero
import numpy as np

def run_full_net_coarsened_FM(
    circuit : QuantumCircuit,
    num_qubits : int,
    network : QuantumNetwork,
    coarsening_factor : int = 2,
    passes_per_level : int = 10,
    use_multiprocessing : bool = False,
    hypergraph_coarsener : callable = HypergraphCoarsener().coarsen_recursive_subgraph_batch,
    num_processes : int | None = None,
    group_gates : bool = True,
    ML_internal_level_limit : int = 6
):
    """
    Outer wrapper to build the initial hypergraph, network, coarsened network, and run multilevel partitioning.
    """
    from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph, SubGraphManager
    from disqco.graphs.coarsening.network_coarsener import NetworkCoarsener
    
    # Build hypergraph
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=group_gates)
    
    # Build initial network

    # Coarsen network
    net_coarsener = NetworkCoarsener(network)
    net_coarsener.coarsen_network_recursive(l=coarsening_factor)
    network_level_list = []
    final_network = net_coarsener.network_coarse_list[-1]
    final_network.active_nodes = set(final_network.qpu_graph.nodes)
    network_level_list.append({None: final_network})
    for i in range(len(net_coarsener.network_coarse_list) - 1):
        prev_networks = network_level_list[i]
        new_networks = net_coarsener.cut_network(prev_networks, level=i)
        network_level_list.append(new_networks)
    
    # Initial partitioning on coarsest network
    from disqco.parti.FM.FM_methods import set_initial_partitions
    coarsest_networks = network_level_list[0]
    coarsest_network = list(coarsest_networks.values())[0]
    coarsest_active_nodes = coarsest_network.active_nodes
    coarsest_node_map = {list(coarsest_active_nodes)[i]: i for i in range(len(coarsest_active_nodes))}
    initial_assignment_coarse = set_initial_partitions(
        network=coarsest_network,
        num_qubits=num_qubits,
        depth=hypergraph.depth,
        node_map=coarsest_node_map,
    )
    from disqco.parti.FM.fiduccia import FiducciaMattheyses
    FM_partitioner = FiducciaMattheyses(circuit=circuit,
                                        network=coarsest_network,
                                        initial_assignment=initial_assignment_coarse,
                                        sparse=True )

    results = FM_partitioner.partition(coarsener=hypergraph_coarsener, sparse=True, level_limit=ML_internal_level_limit)
    optimized_assignment_coarse = results['best_assignment']
    
    # Build initial subgraphs for first level
    sub_graph_manager = SubGraphManager(hypergraph)
    subgraphs = sub_graph_manager.build_partition_subgraphs(
        graph=hypergraph,
        assignment=optimized_assignment_coarse,
        current_network=coarsest_network,
        new_networks=network_level_list[1],
        old_dummy_nodes=set()
    )
    # Run multilevel partitioning
    partitioning_results = multilevel_network_partitioning(
        initial_subgraphs=subgraphs,
        initial_assignment=optimized_assignment_coarse,
        network_level_list=network_level_list,
        sub_graph_manager=sub_graph_manager,
        num_qubits=num_qubits,
        hypergraph=hypergraph,
        initial_network=network,
        use_multiprocessing=use_multiprocessing,
        num_processes=num_processes,
        ML_internal_level_limit=ML_internal_level_limit,
        passes_per_level=passes_per_level
    )
    return partitioning_results

import multiprocessing as mp
from functools import partial
from disqco.parti.FM.net_coarsened_FM import stitch_solution_sparse

def multilevel_network_partitioning(
    initial_subgraphs,
    initial_assignment,
    network_level_list,
    sub_graph_manager,
    num_qubits,
    hypergraph,
    initial_network,
    hypergraph_coarsener=HypergraphCoarsener().coarsen_recursive_subgraph_batch,
    use_multiprocessing=False,
    num_processes=None,
    ML_internal_level_limit=6,
    passes_per_level=10
):
    print(sub_graph_manager)
    all_level_assignments = []
    current_subgraphs = initial_subgraphs
    parent_assignment = initial_assignment
    for level_idx in range(len(network_level_list) - 1):
        build_next_level = (level_idx + 2 < len(network_level_list))
        args_list = [
            (source_node, subgraph, level_idx, network_level_list, parent_assignment, num_qubits, hypergraph, initial_network, hypergraph_coarsener, sub_graph_manager, build_next_level, ML_internal_level_limit, passes_per_level)
            for source_node, subgraph in current_subgraphs.items()
        ]
        if use_multiprocessing and len(args_list) > 1:
            if num_processes is None:
                num_processes = min(mp.cpu_count(), len(args_list))
            with mp.Pool(processes=num_processes) as pool:
                results = pool.starmap(partition_and_build_subgraph, args_list)
        else:
            results = [partition_and_build_subgraph(*args) for args in args_list]
        level_assignments = []
        next_subgraphs = {}
        for result in results:
            if result['success']:
                level_assignments.append(result['assignment'])
                next_subgraphs.update(result['next_level_subgraphs'])
            else:
                raise ValueError(f"Partitioning failed for source node {result['source_node']}: {result['error']}")
        all_level_assignments.append(level_assignments)
        parent_assignment = stitch_solution_sparse(
            current_subgraphs, level_assignments, num_qubits, depth=hypergraph.depth
        )
        
        if build_next_level:
            current_subgraphs = next_subgraphs
    final_cost = calculate_full_cost_hetero(
        hypergraph, parent_assignment, len(initial_network.qpu_sizes), {}, initial_network
    )
    return {
        'all_level_assignments': all_level_assignments,
        'final_assignment': parent_assignment,
        'final_cost': final_cost,
        'num_levels': len(all_level_assignments)
    }


def check_assignment_validity(assignment, qpu_sizes, subgraph):
    """
    Check if an assignment is valid (all qubits assigned to valid partitions within capacity).
    """
    # Count assignments to each partition
    partition_counts = [{qpu: 0 for qpu in qpu_sizes.keys()} for t in range(assignment.shape[0])]
    
    # Handle both 2D arrays and lists of lists
    for node in subgraph.nodes:
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
            node_partition = assignment[t][q]
            partition_counts[t][node_partition] += 1

    # Check capacity constraints
    for t in range(assignment.shape[0]):
        for partition in partition_counts[t]:
            if partition_counts[t][partition] > qpu_sizes[partition]:
                print(f"Layer {t}, partition {partition} exceeds capacity: {partition_counts[t][partition]} > {qpu_sizes[partition]}")
                return False
    
    return True
