from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti.FM.FM_hetero import run_FM_sparse

def refine_assignment_sparse(level, num_levels, assignment, mapping_list, subgraph, next_graph, qpu_sizes):
    new_assignment = assignment
    # First check assignment validity
    # if not check_assignment_validity(new_assignment, qpu_sizes, subgraph):
    #     print(f"Invalid assignment at level {level}. Cannot refine.")
    #     raise ValueError("Invalid assignment")
    unassigned_nodes = set()
    if level < num_levels - 1:
        for super_node_t in mapping_list[level]:
            # print(f"Refining assignment for super node {super_node_t} at level {level}")
            # print(f"  Contained time steps: {mapping_list[level][super_node_t]}")
            for t in mapping_list[level][super_node_t]:
                for q in range(len(assignment[0])):
                    # super_node_t = mapping_list[level][super_node_t]  # Get the first time step in the super node
                    
                    if (q,t) in next_graph.nodes and (q, super_node_t) in next_graph.nodes:
                        # print(f"  Assigning qubit {(q,t)} to super node {(q,super_node_t)}")
                        # if t < len(new_assignment) and super_node_t < len(assignment):
                        new_assignment[t][q] = assignment[super_node_t][q]
                    elif (q, t) in next_graph.nodes and (q, super_node_t) not in next_graph.nodes:
                        # print(f"New qubit has entered, assign where space is available after others")
                        unassigned_nodes.add((q, t))

    partition_counts = [{qpu: 0 for qpu in qpu_sizes.keys()} for t in range(assignment.shape[0])]
    
    # Handle both 2D arrays and lists of lists
    for node in set(subgraph.nodes) - unassigned_nodes:
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
            node_partition = assignment[t][q]
            partition_counts[t][node_partition] += 1
    
    for node in unassigned_nodes:
        q, t = node
        # Find the first partition with available space
        for partition, size in qpu_sizes.items():
            if partition_counts[t][partition] < size:
                new_assignment[t][q] = partition
                partition_counts[t][partition] += 1
                # print(f"Assigned unassigned node {(q,t)} to partition {partition}")
                break
        # else:
        #     print(f"No available partition for unassigned node {(q,t)}")

    return new_assignment

# def refine_assignment_sparse(level, num_levels, assignment, mapping_list, subgraph):
#     """
#     Refine the assignment after each coarsening level using the mapping.
#     """
#     new_assignment = assignment.copy()
#     if level < num_levels - 1:
#         mapping = mapping_list[level]
#         for super_node_t in mapping:
#             for t in mapping[super_node_t]:
#                 for q in range(len(assignment[0]) if len(assignment) > 0 else 0):
#                     if (q, t) in subgraph.nodes and (q, super_node_t) in subgraph.nodes:
#                         if t < len(new_assignment) and super_node_t < len(assignment):
#                             new_assignment[t][q] = assignment[super_node_t][q]
#     return new_assignment


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
    source_node, subgraph, level_idx, network_level_list, parent_assignment, num_qubits, hypergraph, initial_network, sub_graph_manager, build_next_level
):
    try:
        print(f"🔄 Processing Level {level_idx + 1}/{len(network_level_list) - 1} for source node {source_node}")
        sub_network = network_level_list[level_idx + 1][source_node]
        active_nodes = sub_network.active_nodes
        qpu_sizes = {node: sub_network.qpu_graph.nodes[node]['size'] for node in active_nodes}
        node_map = {node: idx for idx, node in enumerate(active_nodes)}
        dummy_nodes = set()
        for node in subgraph.nodes:
            if isinstance(node, tuple) and len(node) >= 3 and node[0] == 'dummy':
                dummy_nodes.add(node)
                qpu = node[2]
                if qpu not in node_map:
                    node_map[qpu] = len(node_map)
        sparse_assignment = set_initial_partitions_sparse(
            assignment=parent_assignment,
            active_nodes=active_nodes,
            qpu_sizes=qpu_sizes,
            subgraph=subgraph
        )
        print(f"Initial sparse assignment: {sparse_assignment}")
        coarsener = HypergraphCoarsener()
        graph_list, mapping_list = coarsener.coarsen_recursive_subgraph_batch(subgraph)
        # Assignment refinement after each coarsening level
        for level, (graph, mapping) in enumerate(zip(graph_list[::-1], mapping_list[::-1])):
            final_cost, final_assignment, _ = run_FM_sparse(
                hypergraph=graph,
                initial_assignment=sparse_assignment,
                qpu_info=qpu_sizes,
                num_partitions=len(active_nodes),
                active_nodes=active_nodes,
                limit=num_qubits,
                max_gain=4*hypergraph.depth,
                passes=10,
                stochastic=True,
                log=False,
                network=sub_network,
                node_map=node_map,
                dummy_nodes=dummy_nodes
            )
            sparse_assignment = refine_assignment_sparse(
                level=level,
                num_levels=len(graph_list),
                assignment=final_assignment,
                mapping_list=mapping_list[::-1],
                subgraph=subgraph,
                next_graph=graph_list[::-1][level + 1] if level + 1 < len(graph_list) else None,
                qpu_sizes=qpu_sizes
            )
        next_level_subgraphs = {}
        if build_next_level and level_idx + 2 < len(network_level_list):
            existing_dummy_nodes = set()
            for node in subgraph.nodes:
                if isinstance(node, tuple) and len(node) > 0 and node[0] == 'dummy':
                    existing_dummy_nodes.add(node)
            next_level_subgraphs = sub_graph_manager.build_partition_subgraphs(
                graph=subgraph,
                assignment=sparse_assignment,
                current_network=sub_network,
                new_networks=network_level_list[level_idx + 2],
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
