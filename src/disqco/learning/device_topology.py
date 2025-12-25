import networkx as nx
from qiskit.transpiler import CouplingMap
import matplotlib.pyplot as plt

def generate_distributed_heavy_hex(num_qpus=2, distance=3, connection_strategy='linear'):
    """
    Generates a graph representing a distributed system of IBM Heavy Hex QPUs.
    """
    # 1. Generate the base Heavy Hex Unit
    # We use Qiskit to get the edge list, then build a standard NetworkX graph
    unit_coupling = CouplingMap.from_heavy_hex(distance)
    
    # FIX: Explicitly build a NetworkX graph from the CouplingMap
    # This avoids the 'rustworkx' Attribute Error
    unit_graph = nx.Graph()
    unit_graph.add_nodes_from(range(unit_coupling.size())) # Ensure all qubits are added
    unit_graph.add_edges_from(unit_coupling.get_edges())   # Add the connections
    
    qubits_per_qpu = unit_coupling.size()
    print(f"Base QPU Size: {qubits_per_qpu} qubits (Heavy Hex d={distance})")
    
    # 2. Create the composite System Graph
    system_graph = nx.Graph()
    
    # Loop to duplicate the QPU structure for each module
    for qpu_id in range(num_qpus):
        offset = qpu_id * qubits_per_qpu
        
        # Add nodes and internal edges
        for u, v in unit_graph.edges():
            global_u = u + offset
            global_v = v + offset
            
            # Add nodes with metadata
            # We add nodes explicitly to ensure attributes are set even for isolated ones
            system_graph.add_node(global_u, qpu=qpu_id, local_id=u, type='data')
            system_graph.add_node(global_v, qpu=qpu_id, local_id=v, type='data')
            
            # Add LOCAL edges (Intra-QPU) -> Cheap SWAPs allowed
            system_graph.add_edge(global_u, global_v, type='intra', weight=1.0)

    # 3. Add Distributed Connections (Inter-QPU Edges)
    if connection_strategy == 'linear':
        for i in range(num_qpus - 1):
            qpu_a_offset = i * qubits_per_qpu
            qpu_b_offset = (i + 1) * qubits_per_qpu
            
            # Define Ports: For d=3 Heavy Hex, standard ports are roughly at ends.
            # We heuristically pick the first and last indexed qubits.
            # (In a real hardware map, you would look up specific indices).
            port_a_local = 0  
            port_b_local = qubits_per_qpu - 1 
            
            global_u = qpu_a_offset + port_b_local # Last qubit of QPU A
            global_v = qpu_b_offset + port_a_local # First qubit of QPU B
            
            # Add INTER-QPU edge -> Teleportation only
            system_graph.add_edge(global_u, global_v, type='inter', weight=10.0)
            
            # Mark the nodes as "communication" nodes
            system_graph.nodes[global_u]['type'] = 'comm'
            system_graph.nodes[global_v]['type'] = 'comm'

    return system_graph, qubits_per_qpu

# Usage
if __name__ == "__main__":
    dist_graph, size = generate_distributed_heavy_hex(num_qpus=2, distance=3)

    # Visualization to verify "Inter" vs "Intra"
    edges = dist_graph.edges(data=True)
    intra_edges = [(u, v) for u, v, d in edges if d['type'] == 'intra']
    inter_edges = [(u, v) for u, v, d in edges if d['type'] == 'inter']

    pos = nx.spring_layout(dist_graph) # Use spring layout for visualization
    nx.draw_networkx_nodes(dist_graph, pos, node_size=20)
    nx.draw_networkx_edges(dist_graph, pos, edgelist=intra_edges, edge_color='blue', label='Intra-QPU')
    nx.draw_networkx_edges(dist_graph, pos, edgelist=inter_edges, edge_color='red', width=2.0, label='Inter-QPU')
    plt.legend()
    plt.savefig("distributed_heavy_hex_topology.png")