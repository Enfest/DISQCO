from itertools import product
import numpy as np
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph

def get_all_configs(num_partitions : int, hetero = False, sparse = False) -> list[set[int]]:
    """
    Generates all possible configurations for a given number of partitions."
    """
    # Each configuration is represented as a tuple of 0s and 1s, where 1 indicates
    # that at least one qubit in the edge is assigned to the current partition.
    configs = set(product((0,1),repeat=num_partitions))
    if hetero:
        configs = configs - set([(0,)*num_partitions])
    
    if sparse:
        configs_sets = set()
        for config in list(configs):
            config_set = set([idx for idx, val in enumerate(config) if val == 1])
            configs_sets.add(config_set)
        configs = configs_sets

    return list(configs)

def config_to_cost(config : tuple[int], ) -> int:
    """
    Converts a configuration tuple to its corresponding cost (assuming all to all connectivity)."
    """
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    return cost

def get_all_costs_hetero(network, 
                         configs : list[tuple[int]], 
                         node_map = None
                         ) -> tuple[dict[tuple[tuple[int],tuple[int]] : int], 
                                    dict[tuple[tuple[int],tuple[int]]] : list[tuple[int]]]:
    """
    Computes the costs and edge forests for all configurations using the provided network."
    """
    costs = {}
    edge_trees = {}
    for root_config in configs:
        print("Root config:", root_config)
        for rec_config in configs:
            print("Receiver config:", rec_config)
            edges, cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = cost
            edge_trees[(root_config, rec_config)] = edges
    return costs, edge_trees

def get_all_costs(configs : list[tuple[int]]
                  ) -> dict[tuple[int] : int]:
    """
    Computes the costs for all configurations given all-to-all connectivity.
    """

    # costs = np.zeros(len(configs)+1)

    # for config in configs:
    #     cost = config_to_cost(config)
    #     integer = int("".join(map(str, config)), 2)
    #     costs[integer] = cost
    costs = {}
    for config in configs:
        cost = config_to_cost(config)
        costs[tuple(config)] = cost

    return costs

def get_cost(config : tuple[int], costs : np.array) -> int:
        config = list(config)
        config = [str(x) for x in config]
        config = "".join(config)
        config = int(config, 2)
        return costs[config]

def hedge_k_counts(hypergraph,
                   hedge,
                   assignment,
                   num_partitions, 
                   set_attrs = False, 
                   assignment_map = None, 
                   node_map = None, 
                   dummy_nodes = {}):
    # root_counts = np.zeros(num_partitions + len(dummy_nodes), dtype=int)
    # rec_counts = np.zeros(num_partitions + len(dummy_nodes), dtype=int)   
    root_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
    rec_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
    info = hypergraph.hyperedges[hedge]
    root_set = info['root_set']
    receiver_set = info['receiver_set']

    if dummy_nodes == {}:
        for root_node in root_set:
            if assignment_map is not None:
                root_node = assignment_map[root_node]

            partition_root = assignment[root_node[1]][root_node[0]]
            if node_map is not None:
                partition_root = node_map[partition_root]
            # partition_root = assignment[root_node]

            root_counts[partition_root] += 1
        for rec_node in receiver_set:
            if assignment_map is not None:
                rec_node = assignment_map[rec_node]
            

            partition_rec = assignment[rec_node[1]][rec_node[0]]

            if node_map is not None:
                partition_rec = node_map[partition_rec]

            # partition_rec = assignment[rec_node]
            rec_counts[partition_rec] += 1
        
        # root_counts = tuple(root_counts)
        # rec_counts = tuple(rec_counts)
        if set_attrs:
            hypergraph.set_hyperedge_attribute(hedge, 'root_counts', root_counts)
            hypergraph.set_hyperedge_attribute(hedge, 'rec_counts', rec_counts)
    else:
        for root_node in root_set:
            print("Root node:", root_node)
            if root_node not in hypergraph.nodes:
                continue
            
            if root_node in dummy_nodes:
                # print("Dummy node root", root_node)
                partition_root = num_partitions + root_node[3]
                # print("Partition dummy root", partition_root)
                root_counts[partition_root] += 1
                continue

            if assignment_map is not None:
                root_node = assignment_map[root_node]
            try:
                partition_root = assignment[root_node[1]][root_node[0]]
            except Exception:
                raise ValueError
            # partition_root = assignment[root_node]
            print("Partition root unmapped:", partition_root)
            if node_map is not None:
                print("Node map:", node_map)
                try:
                    partition_root = node_map[partition_root]
                except Exception as e:
                    print("Assignment:", assignment
                          )
                    raise e
                
            else:
                print("No node map")
            print("Partition root:", partition_root)
            root_counts[partition_root] += 1
        for rec_node in receiver_set:
            print("Receiver node:", rec_node)

            if rec_node not in hypergraph.nodes:
                continue
            if rec_node in dummy_nodes:
                # print("Dummy node rec", rec_node)
                partition_rec = num_partitions + rec_node[3]
                # print("Partition dummy rec", partition_rec)
                rec_counts[partition_rec] += 1
                continue

            if assignment_map is not None:
                rec_node = assignment_map[rec_node]
                
            try:
                partition_rec = assignment[rec_node[1]][rec_node[0]]
            except Exception:
                # print("Rec node", rec_node)
                # print("Assignment", assignment)
                # print("Assignment shape", np.shape(assignment))
                # print("Assignment map", assignment_map)
                # print("Root set", root_set)
                # print("Receiver set", receiver_set)
                # for node in hypergraph.nodes:
                #     print("Node", node)
                #     print("Mapped node", assignment_map[node] if assignment_map is not None else node)
                raise ValueError
            # partition_rec = assignment[rec_node]
            print("Partition rec unmapped:", partition_rec)
            if node_map is not None:
                print("Node map:", node_map)
                try:
                    partition_rec = node_map[partition_rec]
                except Exception as e:
                    print("Assignment:", assignment)
                    raise e
            print("Partition rec:", partition_rec)
            rec_counts[partition_rec] += 1
        
        # root_counts = tuple(root_counts)
        # rec_counts = tuple(rec_counts)
        if set_attrs:
            hypergraph.set_hyperedge_attribute(hedge, 'root_counts', root_counts)
            hypergraph.set_hyperedge_attribute(hedge, 'rec_counts', rec_counts)

    
    return root_counts, rec_counts

def counts_to_configs(root_counts : tuple[int], rec_counts : tuple[int]) -> tuple[tuple[int], tuple[int]]:
    """
    Converts the counts of nodes in each partition to root and rec config tuples."
    """
    root_config = []
    rec_config = []
    for x,y in zip(root_counts,rec_counts):
        if x > 0:
            root_config.append(1)
        else:
            root_config.append(0)
        if y > 0:
            rec_config.append(1)
        else:
            rec_config.append(0)
    return tuple(root_config), tuple(rec_config)

def full_config_from_counts(root_counts : tuple[int], 
                       rec_counts : tuple[int]
                       ) -> tuple[int]:
    """
    Converts the counts of nodes in each partition to full configuration tuple.
    """
    config = []
    for x,y in zip(root_counts,rec_counts):
        if y > 0 and x < 1:
            config.append(1)
        else:
            config.append(0)
    return config

def map_hedge_to_config(hypergraph : QuantumCircuitHyperGraph, 
                          hedge : tuple, 
                          assignment : dict[tuple[int,int]], 
                          num_partitions : int
                          ) -> tuple[int]:
    
    """
    Maps a hyperedge to its full configuration based on the current assignment.
    Uses config_from_counts to skip the intermediate step of counts_to_configs.
    """
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
    config = full_config_from_counts(root_counts,rec_counts)

    return config

def map_hedge_to_configs(hypergraph,hedge,assignment,num_partitions,assignment_map = None, node_map=None, dummy_nodes = {}):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False,assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)
    root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    # config = config_from_counts(root_counts,rec_counts)
    return root_config,rec_config

def get_full_config(root_config : tuple[int], rec_config : tuple[int]) -> tuple[int]:
    """
    Converts the root and receiver configurations to a full configuration tuple."
    """
    config = list(rec_config)
    for i, element in enumerate(root_config):
        if rec_config[i] == 1:
            config[i] -= element
    return config

def hedge_to_cost(hypergraph : QuantumCircuitHyperGraph, 
                   hedge : tuple, 
                   assignment : dict[tuple[int,int]], 
                   num_partitions : int, 
                   costs : dict[tuple] = {}) -> int:
    """
    Computes the cost of a hyperedge based on its configuration and the current assignment.
    """ 
    config = map_hedge_to_config(hypergraph, hedge, assignment, num_partitions)

    # if config not in costs:
    #     cost = config_to_cost(config)
    #     configint = list(config)
    #     configint = [str(x) for x in configint]
    #     configint = "".join(configint)
    #     costs[config] = cost
    # else:
        # cost = costs[config]
    cost = get_cost(config, costs)
    return cost

def hedge_to_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                         hedge : tuple, 
                         assignment : dict[tuple[int,int]], 
                         num_partitions : int, 
                         costs : dict[tuple] = {},
                         network = None,
                         assignment_map = None,
                            dummy_nodes = {}
                         ) -> int:
    """"
    Computes the cost of a hyperedge based on its configuration and the current assignment."
    """
    root_config, rec_config = map_hedge_to_configs(hypergraph, hedge, assignment, num_partitions, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
    if (root_config, rec_config) not in costs:
        edges, cost = network.steiner_forest(root_config, rec_config)
        costs[(root_config, rec_config)] = cost
    else:
        cost = costs[(root_config, rec_config)]
    return cost

def map_current_costs(hypergraph : QuantumCircuitHyperGraph, 
                      assignment : dict[tuple[int,int]], 
                      num_partitions : int, 
                      costs: dict
                      ) -> None:
    """
    Maps the current costs of all hyperedges to hyperedge attributes based on the current assignment.
    """
    for edge in hypergraph.hyperedges:
        hypergraph.set_hyperedge_attribute(edge, 'cost', hedge_to_cost(hypergraph,edge,assignment,num_partitions,costs))
    return
        
def map_counts_and_configs(hypergraph : QuantumCircuitHyperGraph, 
                            assignment : dict[tuple[int,int]], 
                            num_partitions : int, 
                            costs: dict,
                            **kwargs) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    """
    hetero = kwargs.get('hetero', False)
    if hetero:
        network = kwargs.get('network', None)
        node_map = kwargs.get('node_map', None)
        assignment_map = kwargs.get('assignment_map', None)
        dummy_nodes = kwargs.get('dummy_nodes', {})
        return map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, network, costs=costs, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)
    else:
        return map_counts_and_configs_homo(hypergraph, assignment, num_partitions, costs=costs)

def map_counts_and_configs_homo(hypergraph : QuantumCircuitHyperGraph, 
                            assignment : dict[tuple[int,int]], 
                            num_partitions : int, 
                            costs: dict) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    """
    for edge in hypergraph.hyperedges:
        # print("Edge", edge)
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)

        config = full_config_from_counts(root_counts, rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'config', config)
        if tuple(config) not in costs:
            cost = config_to_cost(config)
            costs[tuple(config)] = cost
        else:
            cost = costs[tuple(config)]

        hypergraph.set_hyperedge_attribute(edge, 'cost', cost)
    return hypergraph

def map_counts_and_configs_hetero(hypergraph : QuantumCircuitHyperGraph,
                                  assignment : dict[tuple[int,int]],
                                  num_partitions : int,
                                  network,
                                  costs: dict = {},
                                  node_map = None,
                                  dummy_nodes = {}) -> QuantumCircuitHyperGraph:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True, node_map=node_map, dummy_nodes=dummy_nodes)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        root_config, rec_config = counts_to_configs(root_counts,rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
        if (root_config, rec_config) not in costs:
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = edge_cost
        else:
            edge_cost = costs[(root_config, rec_config)]
        hypergraph.set_hyperedge_attribute(edge, 'cost', edge_cost)
    return hypergraph

# def map_counts_and_configs_hetero_sparse(hypergraph : QuantumCircuitHyperGraph,
#                                   assignment : dict[tuple[int,int]],
#                                   num_partitions : int,
#                                   network,
#                                   costs: dict = {},
#                                   assignment_map = None,
#                                   node_map = None,
#                                   dummy_nodes = {}) -> QuantumCircuitHyperGraph:
#     """
#     Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
#     For heterogeneous networks, it uses the network to compute the costs.
#     """
#     for edge in hypergraph.hyperedges:
#         root_counts, rec_counts = hedge_k_counts_sparse_dict(hypergraph,edge,assignment,num_partitions, dummy_nodes=dummy_nodes)
#         hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
#         hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
#         root_config, rec_config = counts_to_config_sets(root_counts,rec_counts)
#         hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
#         hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
#         if (root_config, rec_config) not in costs:
#             edges, edge_cost = network.steiner_forest_from_sets(root_config, rec_config, node_map=node_map)
#             costs[(root_config, rec_config)] = edge_cost
#         else:
#             edge_cost = costs[(root_config, rec_config)]
#         hypergraph.set_hyperedge_attribute(edge, 'cost', edge_cost)
#     return hypergraph

def calculate_full_cost(hypergraph : QuantumCircuitHyperGraph,
                        assignment,
                        num_partitions : int,
                        costs: dict = {},
                        **kwargs) -> int:
    """
    Wrapper function for computing full cost under either homogeneous (fully connected) or heterogeneous (not fully connected) networks.
    """
    hetero = kwargs.get('hetero', False)
    if hetero:
        network = kwargs.get('network', None)
        node_map = kwargs.get('node_map', None)
        assignment_map = kwargs.get('assignment_map', None)
        dummy_nodes = kwargs.get('dummy_nodes', {})
        return calculate_full_cost_hetero(  hypergraph, 
                                            assignment, 
                                            num_partitions, 
                                            costs=costs, 
                                            network=network,
                                            node_map=node_map, 
                                            assignment_map=assignment_map, 
                                            dummy_nodes=dummy_nodes )
    else:  
        return calculate_full_cost_homo(    hypergraph, 
                                            assignment, 
                                            num_partitions, 
                                            costs=costs )

def calculate_full_cost_homo(hypergraph : QuantumCircuitHyperGraph,
                        assignment : dict[tuple[int,int]],
                        num_partitions : int,
                        costs: dict = {}) -> int:
    """
    Computes the total cost of the hypergraph based on the current assignment and the costs of each configuration.
    """
    cost = 0
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions)
        config = full_config_from_counts(root_counts,rec_counts)
        conf = tuple(config)
        if conf not in costs:
            edge_cost = config_to_cost(config)
            costs[conf] = edge_cost
        else:
            edge_cost = costs[conf]
        cost += edge_cost
    return cost

def calculate_full_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                               assignment : dict[tuple[int,int]],
                               num_partitions : int,
                               costs: dict = {},
                               network = None, 
                               node_map: dict = None,
                               assignment_map = None,
                               dummy_nodes = {}) -> int:
    """
    Computes the total cost of the hypergraph based on the current assignment and the costs of each configuration.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    cost = 0
    for edge in hypergraph.hyperedges:
        print("Edge", edge)
        print("Assignment:", assignment_map)
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)
        print(f"Edge {edge}: Root counts: {root_counts}, Rec counts: {rec_counts}")
        root_config, rec_config = counts_to_configs(root_counts, rec_counts)
        print(f"Root config: {root_config}, Rec config: {rec_config}")

        if (root_config, rec_config) in costs:
            edge_cost = costs[(root_config, rec_config)]
        else:
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = edge_cost
        cost += edge_cost
    
    return cost

# def map_hedge_to_configs_sparse(hypergraph, hedge, assignment, num_partitions, dummy_nodes=set()):
#     """
#     Map a hyperedge to its root and receiver configurations using sparse assignment.
#     This version works directly with sparse assignment matrix without node mapping.
    
#     Args:
#         hypergraph: The hypergraph
#         hedge: Hyperedge ID
#         assignment: Sparse assignment matrix [depth][num_qubits] -> partition
#         num_partitions: Number of partitions
#         dummy_nodes: Set of dummy nodes
        
#     Returns:
#         tuple: (root_config, rec_config)
#     """
#     root_counts, rec_counts = hedge_k_counts_sparse(
#         hypergraph, hedge, assignment, num_partitions, dummy_nodes=dummy_nodes
#     )
#     root_config, rec_config = counts_to_configs(root_counts, rec_counts)
#     return root_config, rec_config

# def hedge_k_counts_sparse(hypergraph, hedge, assignment, num_partitions, dummy_nodes=set()):
#     """
#     Count nodes in each partition for a hyperedge using sparse assignment.
    
#     Args:
#         hypergraph: The hypergraph
#         hedge: Hyperedge ID
#         assignment: Sparse assignment matrix [depth][num_qubits] -> partition
#         num_partitions: Number of partitions
#         dummy_nodes: Set of dummy nodes
        
#     Returns:
#         tuple: (root_counts, rec_counts) - lists of counts per partition
#     """
#     root_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
#     rec_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
    
#     info = hypergraph.hyperedges[hedge]
#     root_set = info['root_set']
#     receiver_set = info['receiver_set']
    
#     # Count root nodes
#     for root_node in root_set:
#         if root_node not in hypergraph.nodes:
#             continue
            
#         if root_node in dummy_nodes:
#             # Handle dummy nodes
#             partition_root = num_partitions + root_node[3]
#             root_counts[partition_root] += 1
#             continue
            
#         q, t = root_node
#         partition_root = assignment[t][q]
        
#         # Skip nodes that aren't assigned in sparse assignment
#         if partition_root == -1:
#             continue
            
#         root_counts[partition_root] += 1
    
#     # Count receiver nodes
#     for rec_node in receiver_set:
#         if rec_node not in hypergraph.nodes:
#             continue
            
#         if rec_node in dummy_nodes:
#             # Handle dummy nodes
#             partition_rec = num_partitions + rec_node[3]
#             rec_counts[partition_rec] += 1
#             continue
            
#         q, t = rec_node
#         partition_rec = assignment[t][q]
        
#         # Skip nodes that aren't assigned in sparse assignment
#         if partition_rec == -1:
#             continue
            
#         rec_counts[partition_rec] += 1
    
#     return root_counts, rec_counts

# Set-based sparse config functions for efficient representation

def counts_to_config_sets(root_counts, rec_counts):
    """
    Convert counts to set-based configs for efficient sparse representation.
    
    Args:
        root_counts: List/dict of counts per partition for root nodes
        rec_counts: List/dict of counts per partition for receiver nodes
        
    Returns:
        tuple: (root_config_set, rec_config_set) - frozensets of partition indices with non-zero nodes
    """
    if isinstance(root_counts, dict):
        root_config_set = frozenset({i for i, count in root_counts.items() if count > 0})
        rec_config_set = frozenset({i for i, count in rec_counts.items() if count > 0})
    else:
        root_config_set = frozenset({i for i, count in enumerate(root_counts) if count > 0})
        rec_config_set = frozenset({i for i, count in enumerate(rec_counts) if count > 0})

    return root_config_set, rec_config_set

def config_set_to_cost(config_set):
    """
    Calculate cost from a set-based config (much more efficient than tuple-based).
    
    Args:
        config_set: Set of partition indices with nodes
        
    Returns:
        int: Cost (number of partitions involved)
    """
    return len(config_set)

def full_config_set_from_configs(root_config_set, rec_config_set):
    """
    Get the full config set (union of root and receiver partitions).
    
    Args:
        root_config_set: Set of root partition indices
        rec_config_set: Set of receiver partition indices
        
    Returns:
        set: Union of both config sets
    """
    return root_config_set.union(rec_config_set)

def sparse_counts_dict(num_partitions, dummy_count=0):
    """
    Create a sparse counts dictionary for efficient counting.
    Only allocates entries that are actually used.
    
    Args:
        num_partitions: Total number of partitions
        dummy_count: Number of dummy nodes
        
    Returns:
        dict: Empty sparse counts dictionary
    """
    return {}

def hedge_k_counts_sparse_dict(hypergraph, hedge, assignment, num_partitions, dummy_nodes=set()):
    """
    Count nodes using sparse dictionaries for better performance with many partitions.
    
    Args:
        hypergraph: The hypergraph
        hedge: Hyperedge ID
        assignment: Sparse assignment matrix [depth][num_qubits] -> partition
        num_partitions: Number of partitions
        dummy_nodes: Set of dummy nodes
        
    Returns:
        tuple: (root_counts_dict, rec_counts_dict) - sparse dictionaries
    """
    root_counts = {}
    rec_counts = {}
    
    info = hypergraph.hyperedges[hedge]
    root_set = info['root_set']
    receiver_set = info['receiver_set']
    
    # Count root nodes
    for root_node in root_set:
        if root_node not in hypergraph.nodes:
            continue
            
        if root_node in dummy_nodes:
            partition_root = num_partitions + root_node[3]
            root_counts[partition_root] = root_counts.get(partition_root, 0) + 1
            continue
            
        q, t = root_node
        partition_root = assignment[t][q]
        
        if partition_root == -1:
            continue
            
        root_counts[partition_root] = root_counts.get(partition_root, 0) + 1
    
    # Count receiver nodes
    for rec_node in receiver_set:
        if rec_node not in hypergraph.nodes:
            continue
            
        if rec_node in dummy_nodes:
            partition_rec = num_partitions + rec_node[3]
            rec_counts[partition_rec] = rec_counts.get(partition_rec, 0) + 1
            continue
            
        q, t = rec_node
        partition_rec = assignment[t][q]
        
        if partition_rec == -1:
            continue
            
        rec_counts[partition_rec] = rec_counts.get(partition_rec, 0) + 1
    
    return root_counts, rec_counts

def map_hedge_to_config_sets_sparse(hypergraph, hedge, assignment, num_partitions, dummy_nodes=set()):
    """
    Map a hyperedge to set-based configs using sparse assignment for maximum efficiency.
    
    Args:
        hypergraph: The hypergraph
        hedge: Hyperedge ID
        assignment: Sparse assignment matrix [depth][num_qubits] -> partition
        num_partitions: Number of partitions
        dummy_nodes: Set of dummy nodes
        
    Returns:
        tuple: (root_config_set, rec_config_set) - sets of involved partition indices
    """
    root_counts, rec_counts = hedge_k_counts_sparse_dict(
        hypergraph, hedge, assignment, num_partitions, dummy_nodes=dummy_nodes
    )
    root_config_set, rec_config_set = counts_to_config_sets(root_counts, rec_counts)
    return root_config_set, rec_config_set

def map_counts_and_configs_sparse_sets(hypergraph: QuantumCircuitHyperGraph,
                                      assignment: list,
                                      num_partitions: int,
                                      network=None,
                                      costs: dict = {},
                                      node_map=None,
                                      dummy_nodes=set()) -> QuantumCircuitHyperGraph:
    """
    Map counts and configs using set-based sparse representation for maximum efficiency.
    
    This is the most efficient version:
    - Uses sparse dictionaries for counting (only stores non-zero entries)
    - Uses sets for configs (only stores partition indices with nodes)
    - Eliminates tuple/list iteration overhead
    - Optimal for large partition counts with sparse occupancy
    
    Args:
        hypergraph: The quantum circuit hypergraph
        assignment: Sparse assignment matrix [depth][num_qubits] -> partition (-1 for inactive)
        num_partitions: Total number of partitions
        network: Network topology for cost calculation (optional)
        costs: Cache for computed costs (now keyed by set pairs)
        node_map: Map from global partition IDs to network positions
        dummy_nodes: Set of dummy nodes
        
    Returns:
        Updated hypergraph with config/cost attributes
    """
    for edge in hypergraph.hyperedges:
        # Get set-based configs using sparse method
        root_config_set, rec_config_set = map_hedge_to_config_sets_sparse(
            hypergraph, edge, assignment, num_partitions, dummy_nodes=dummy_nodes
        )
        
        # Store configurations as sets (more efficient)
        hypergraph.set_hyperedge_attribute(edge, 'root_config_set', root_config_set)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config_set', rec_config_set)
        
        # Calculate cost
        if network is not None:
            # For heterogeneous networks, use set-based steiner forest
            config_key = (frozenset(root_config_set), frozenset(rec_config_set))
            if config_key not in costs:
                edges, edge_cost = network.steiner_forest_from_sets(
                    root_config_set, rec_config_set, node_map=node_map
                )
                costs[config_key] = edge_cost
            else:
                edge_cost = costs[config_key]
        else:
            # For homogeneous networks, use set-based cost (much faster)
            full_config_set = full_config_set_from_configs(root_config_set, rec_config_set)
            config_key = frozenset(full_config_set)
            if config_key not in costs:
                edge_cost = config_set_to_cost(full_config_set)
                costs[config_key] = edge_cost
            else:
                edge_cost = costs[config_key]
        
        hypergraph.set_hyperedge_attribute(edge, 'cost', edge_cost)
    
    return hypergraph