from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti.FM.FM_hetero import run_FM_sparse
from disqco.parti.FM.fiduccia import FiducciaMattheyses
import numpy as np

def refine_assignment_sparse(level, num_levels, assignment, mapping_list, subgraph, next_graph, qpu_sizes):
    new_assignment = assignment
    unassigned_nodes = {}
    if level < num_levels - 1:
        for super_node_t in mapping_list[level]:
            for t in mapping_list[level][super_node_t]:
                for q in range(len(assignment[0])):
                    if (q,t) in subgraph.nodes and (q, super_node_t) in subgraph.nodes:
                        new_assignment[t][q] = assignment[super_node_t][q]
                    elif (q, t) in subgraph.nodes and (q, super_node_t) not in subgraph.nodes:
                        target_partition = assignment[super_node_t][q]
                        unassigned_nodes[(q, t)] = target_partition



    partition_counts = [{qpu: 0 for qpu in qpu_sizes.keys()} for t in range(assignment.shape[0])]
    for node in set(subgraph.nodes) - unassigned_nodes.keys():
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
            node_partition = assignment[t][q]
            partition_counts[t][node_partition] += 1

    for node in unassigned_nodes.keys():
        q, t = node
        # Find the first partition with available space
        for partition, size in qpu_sizes.items():
            if partition_counts[t][partition] < size:
                new_assignment[t][q] = partition
                partition_counts[t][partition] += 1
                break

    return new_assignment

def set_initial_partitions_sparse(assignment, active_nodes, qpu_sizes, subgraph):
    sparse_assignment = assignment.copy()
    depth = len(assignment)
    spaces = [] 
    for i in range(depth):
        spaces_layer = []
        for qpu in qpu_sizes:
            spaces_layer_qpu = [qpu for q in range(qpu_sizes[qpu])]
            # print(f"Layer {i}, QPU {qpu}: {len(spaces_layer_qpu)} spaces")
            spaces_layer += spaces_layer_qpu
        # print(f"Layer {i} spaces: {spaces_layer}")
        spaces.append(spaces_layer)

    for i, layer in enumerate(sparse_assignment):
        for idx, qpu in enumerate(layer):
            if qpu in active_nodes:
                # print(f"Expanding QPU {qpu} in layer {i}")
                index = spaces[i].pop(0)
                # print(f"Assigning index {index} to QPU {qpu} in layer {i}")
                sparse_assignment[i][idx] = index
    return sparse_assignment

def partition_and_build_subgraph(
    source_node, subgraph, level_idx, 
    network_level_list, parent_assignment, 
    num_qubits, hypergraph,
    hypergraph_coarsener,
    sub_graph_manager, build_next_level, 
    ML_internal_level_limit: int | None = None, passes_per_level : int = 10
):
    try:
        sub_network = network_level_list[level_idx][source_node]
        active_nodes = sub_network.active_nodes
        qpu_sizes = {node: sub_network.qpu_graph.nodes[node]['size'] for node in active_nodes}
        node_map = {node: idx for idx, node in enumerate(active_nodes)}
        
        sparse_assignment = set_initial_partitions_sparse(
            assignment=parent_assignment,
            active_nodes=active_nodes,
            qpu_sizes=qpu_sizes,
            subgraph=subgraph
        )

        partitioner = FiducciaMattheyses(
            circuit=None,
            initial_assignment=sparse_assignment,
            hypergraph=subgraph,
            qpu_info=qpu_sizes,
            num_partitions=len(active_nodes),
            active_nodes=active_nodes,
            limit=num_qubits,
            max_gain=4*hypergraph.depth,
            passes=passes_per_level,
            stochastic=True,
            network=sub_network,
            node_map=node_map,
            sparse=True
        )
        results = partitioner.partition(coarsener=hypergraph_coarsener, 
                                        sparse=True, level_limit=ML_internal_level_limit, 
                                        passes_per_level=passes_per_level)

        sparse_assignment = results['best_assignment']
        final_cost = results['best_cost']
        
        next_level_subgraphs = {}
        if build_next_level and level_idx + 1 < len(network_level_list):
            existing_dummy_nodes = set()
            for node in subgraph.nodes:
                if isinstance(node, tuple) and len(node) > 0 and node[0] == 'dummy':
                    existing_dummy_nodes.add(node)
            
            next_level_subgraphs = sub_graph_manager.build_partition_subgraphs(
                graph=subgraph,
                assignment=sparse_assignment,
                current_network=sub_network,
                new_networks=network_level_list[level_idx + 1],
                old_dummy_nodes=existing_dummy_nodes
            )
        return {
            'source_node': source_node,
            'assignment': sparse_assignment,
            'next_level_subgraphs': next_level_subgraphs,
            'success': True,
            'final_cost': final_cost
        }
    except Exception as e:
        return {
            'source_node': source_node,
            'success': False,
            'error': str(e)
        }
