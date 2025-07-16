import time
from disqco.graphs.hypergraph_methods import *
from disqco.parti.FM.FM_methods import *
from networkx import diameter

def FM_pass_hetero(hypergraph,
            max_gain,
            assignment,
            num_partitions,
            qpu_info, 
            costs, 
            limit, 
            active_nodes,
            network = None,
            node_map = None,
            assignment_map = None,
            dummy_nodes = {}):
        
        num_qubits = hypergraph.num_qubits
        depth = hypergraph.depth

        spaces = find_spaces(num_qubits, depth, assignment,qpu_info)
        hypergraph = map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, network, costs, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)

        lock_dict = {node: False for node in active_nodes}
  
        max_time = 0
        for node in active_nodes:
            if node[1] > max_time:
                max_time = node[1]

        array = find_all_gains_hetero(hypergraph,active_nodes,assignment,num_partitions,costs, network=network,node_map=node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
        buckets = fill_buckets(array,max_gain)
        gain_list = []
        gain_list.append(0)
        assignment_list = []
        assignment_list.append(assignment)
        cumulative_gain = 0
        action = 0
        h = 0
        while h < limit:
            action, gain = find_action(buckets,lock_dict,spaces,max_gain)
            if action == None:
                break
            cumulative_gain += gain
            gain_list.append(cumulative_gain)
            node = (action[1],action[0])
            destination = action[2]
            if assignment_map is not None:
                sub_node = assignment_map[node]
                source = assignment[sub_node[1]][sub_node[0]]
            else:
                source = assignment[node[1]][node[0]]

            assignment_new, array, buckets = take_action_and_update_hetero(hypergraph,
                                                                           node,
                                                                           destination,
                                                                           array,
                                                                           buckets,
                                                                           num_partitions,
                                                                           lock_dict, 
                                                                           assignment, 
                                                                           costs, 
                                                                           network = network, 
                                                                           node_map = node_map, 
                                                                           assignment_map=assignment_map)

            update_spaces(node,source,destination,spaces)
            lock_dict = lock_node(node,lock_dict)

            
            assignment = assignment_new
            assignment_list.append(assignment)
            h += 1
        return assignment_list, gain_list

def run_FM_hetero(
    hypergraph,
    initial_assignment,
    qpu_info,
    num_partitions,
    limit = None,
    max_gain=None,
    passes=100,
    stochastic=True,
    active_nodes=None,
    log = False,
    add_initial = False,
    costs = None,
    network = None,
    node_map = None,
    assignment_map = None,
    dummy_nodes = {}
):  
    if network is None:
        network = QuantumNetwork(qpu_info)
    
    if active_nodes is None:
        active_nodes = hypergraph.nodes

    if costs is None and num_partitions < 12:
        configs = get_all_configs(num_partitions, hetero=True)
        costs, edge_trees = get_all_costs_hetero(network, configs, node_map=node_map)
    else:
        costs = {}

    if limit is None:
        limit = len(active_nodes) * 0.125

    initial_assignment = np.array(initial_assignment)
    initial_cost = calculate_full_cost_hetero(hypergraph, initial_assignment, num_partitions, costs, network = network, node_map = node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)

    if active_nodes is not None:
        active_nodes = hypergraph.nodes
    
    if log:
        print("Initial cost:", initial_cost)
    
    if max_gain is None:
        max_gain = 4*diameter(network.qpu_graph)

    if isinstance(qpu_info, dict):
        qpu_info = {i: size for i, size in qpu_info.items()}
    else:
        qpu_info = {i: qpu_info[i] for i in range(len(qpu_info))}


    cost = initial_cost
    cost_list = []
    best_assignments = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)
    # print("Starting FM passes...")
    for n in range(passes):
        # print(f"Pass number: {n}")
        assignment_list, gain_list = FM_pass_hetero(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, 
            active_nodes = active_nodes, network = network, 
            node_map = node_map, assignment_map=assignment_map,
            dummy_nodes=dummy_nodes
        )

        # Decide how to pick new assignment depending on stochastic or not
        if stochastic:
            if n % 2 == 0:
                # Exploratory approach
                initial_assignment = assignment_list[-1]
                cost += gain_list[-1]
            else:
                # Exploitative approach
                idx_best = np.argmin(gain_list)
                initial_assignment = assignment_list[idx_best]
                cost += min(gain_list)
        else:
            # purely pick the best
            idx_best = np.argmin(gain_list)
            initial_assignment = assignment_list[idx_best]
            cost += min(gain_list)

        # print(f"Running cost after pass {n}:", cost)
        cost_list.append(cost)
        best_assignments.append(initial_assignment)

    # 5) Identify best assignment across all passes
    idx_global_best = np.argmin(cost_list)
    final_assignment = best_assignments[idx_global_best]
    final_cost = cost_list[idx_global_best]

    if log:
        print("All passes complete.")
        print("Final cost:", final_cost)

    return final_cost, final_assignment, cost_list

def FM_pass_sparse(hypergraph,
                   max_gain,
                   assignment,
                   num_partitions,
                   qpu_info, 
                   costs, 
                   limit, 
                   active_nodes,
                   network=None,
                   node_map=None,
                   dummy_nodes=set()):
    """
    Single FM pass using sparse assignment with node mapping.
    
    Args:
        hypergraph: The hypergraph
        max_gain: Maximum gain value for bucketing
        assignment: Sparse assignment matrix [depth][num_qubits] -> partition
        num_partitions: Number of partitions
        qpu_info: QPU size information
        costs: Cost dictionary for edge configurations
        limit: Maximum number of moves per pass
        active_nodes: Set of (q, t) nodes that are active
        network: Target network
        node_map: Mapping from partition IDs to network positions
        dummy_nodes: Set of dummy nodes to exclude
        
    Returns:
        tuple: (assignment_list, gain_list)
    """
    from disqco.parti.FM.FM_methods import find_spaces_sparse, find_all_gains_hetero_sparse, fill_buckets, find_action, update_spaces, lock_node, take_action_and_update_hetero_sparse
    from disqco.graphs.hypergraph_methods import map_counts_and_configs_hetero
    
    # Calculate available spaces using sparse method
    spaces = find_spaces_sparse(assignment=assignment, 
                                qpu_sizes=qpu_info, 
                                graph=hypergraph)
    
    # Map counts and configs to hypergraph using sparse method with node_map
    hypergraph = map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, 
                                              network, costs,
                                              node_map=node_map, dummy_nodes=dummy_nodes)

    # Initialize lock dictionary for all active nodes
    lock_dict = {node: False for node in hypergraph.nodes}
    # Lock dummy nodes
    lock_dict.update({node: True for node in dummy_nodes})

    # Calculate gains for all possible moves
    array = find_all_gains_hetero_sparse(hypergraph, hypergraph.nodes, assignment, 
                                  num_partitions, costs, network=network, active_nodes=network.active_nodes,
                                  node_map=node_map, dummy_nodes=dummy_nodes)
    
    # Fill buckets based on gains
    buckets = fill_buckets(array, max_gain)
    
    # Track gains and assignments
    gain_list = [0]
    assignment_list = [assignment.copy()]
    cumulative_gain = 0
    
    # Perform moves
    h = 0
    while h < limit:
        action, gain = find_action(buckets, lock_dict, spaces, max_gain)
        if action is None:
            break
            
        cumulative_gain += gain
        gain_list.append(cumulative_gain)
        
        # Extract move details
        t, q, destination = action
        node = (q, t)
        source = assignment[t][q]
        
        # Apply the move
        assignment_new, array, buckets = take_action_and_update_hetero_sparse(
            hypergraph, node, destination, array, buckets, num_partitions,
            lock_dict, assignment, costs, network=network, 
            node_map=node_map
        )
        
        # Update spaces and lock the node
        update_spaces(node, source, destination, spaces)
        lock_dict = lock_node(node, lock_dict)
        
        assignment = assignment_new
        assignment_list.append(assignment.copy())
        h += 1
        
    return assignment_list, gain_list

def run_FM_sparse(hypergraph,
                  initial_assignment,
                  qpu_info,
                  num_partitions,
                  active_nodes,
                  limit=None,
                  max_gain=None,
                  passes=100,
                  stochastic=True,
                  log=False,
                  add_initial=False,
                  costs=None,
                  network=None,
                  node_map=None,
                  dummy_nodes=set()):
    """
    Run FM algorithm using sparse assignment with node mapping.
    
    Args:
        hypergraph: The hypergraph
        initial_assignment: Initial sparse assignment matrix
        qpu_info: QPU size information
        num_partitions: Number of partitions
        active_nodes: Set of (q, t) nodes that are active
        limit: Maximum moves per pass
        max_gain: Maximum gain for bucketing
        passes: Number of FM passes
        stochastic: Whether to use stochastic approach
        log: Whether to log progress
        add_initial: Whether to include initial cost in results
        costs: Cost dictionary
        network: Target network
        node_map: Mapping from partition IDs to network positions
        dummy_nodes: Set of dummy nodes
        
    Returns:
        tuple: (final_cost, final_assignment, cost_list)
    """
    from disqco.parti.FM.FM_methods import calculate_full_cost_hetero
    from networkx import diameter
    import numpy as np
    
    if network is None:
        from disqco.graphs.quantum_network import QuantumNetwork
        network = QuantumNetwork(qpu_info)
    
    # if costs is None and num_partitions < 12:
    #     from disqco.parti.FM.FM_methods import get_all_configs, get_all_costs_hetero
    #     configs = get_all_configs(num_partitions, hetero=True)
    #     costs, edge_trees = get_all_costs_hetero(network, configs, node_map=node_map)
    # else:
    costs = {}

    if limit is None:
        limit = len(active_nodes) * 0.125

    initial_assignment = np.array(initial_assignment)
    initial_cost = calculate_full_cost_hetero(hypergraph, initial_assignment, num_partitions, 
                                            costs, network=network, node_map=node_map, 
                                            assignment_map=None, dummy_nodes=dummy_nodes)
    
    if log:
        print("Initial cost:", initial_cost)
    
    if max_gain is None:
        max_gain = 4 * diameter(network.qpu_graph)

    if isinstance(qpu_info, dict):
        qpu_info = {i: size for i, size in qpu_info.items()}
    else:
        qpu_info = {i: qpu_info[i] for i in range(len(qpu_info))}

    cost = initial_cost
    cost_list = []
    best_assignments = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)

    # Run FM passes
    for n in range(passes):
        assignment_list, gain_list = FM_pass_sparse(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, 
            active_nodes, network=network, node_map=node_map,
            dummy_nodes=dummy_nodes
        )

        # Decide how to pick new assignment
        if stochastic:
            if n % 2 == 0:
                # Exploratory approach
                initial_assignment = assignment_list[-1]
                cost += gain_list[-1]
            else:
                # Exploitative approach
                idx_best = np.argmin(gain_list)
                initial_assignment = assignment_list[idx_best]
                cost += min(gain_list)
        else:
            # Purely pick the best
            idx_best = np.argmin(gain_list)
            initial_assignment = assignment_list[idx_best]
            cost += min(gain_list)

        cost_list.append(cost)
        best_assignments.append(initial_assignment)

    # Identify best assignment across all passes
    idx_global_best = np.argmin(cost_list)
    final_assignment = best_assignments[idx_global_best]
    final_cost = cost_list[idx_global_best]

    if log:
        print("All passes complete.")
        print("Final cost:", final_cost)

    return final_cost, final_assignment, cost_list
