import torch
from torch_geometric.data import HeteroData, Batch
import networkx as nx
import matplotlib.pyplot as plt

class DistributedGraphData(HeteroData):
    """
    Custom HeteroData class that knows how to batch our specific
    decoder indices (intra_edge_index, inter_edge_index).
    """
    def __inc__(self, key, value, *args, **kwargs):
        # 1. Standard Handling (let PyG handle default edge_index)
        if key == 'edge_index' or 'edge_index' in key:
            return super().__inc__(key, value, *args, **kwargs)
        
        # 2. Custom Handling: Shift decoder indices by number of Physical Nodes
        if key == 'intra_edge_index':
            return self['physical'].num_nodes
        
        if key == 'inter_edge_index':
            return self['physical'].num_nodes
        
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Concatenate indices along columns (dim 1)
        if key == 'intra_edge_index' or key == 'inter_edge_index':
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

def graph_collate_fn(data_list):
    """
    Merges a list of DistributedGraphData into one Batch object.
    Used by the Buffer's internal loader.
    """
    return Batch.from_data_list(data_list)

def save_env_snapshot(env, filename="env_snapshot.png", title=None):
    """
    Visualizes the current state of the Distributed Quantum Environment.
    
    Nodes:
        - White: Empty Physical Qubit
        - Blue:  Occupied by PRIMARY Logical Qubit (L_x)
        - Red:   Occupied by GHOST Logical Qubit (Entangled Copy)
    
    Edges:
        - Black: Intra-QPU Connection
        - Gray (Dashed): Inactive Optical Link
        - Green (Thick): ACTIVE Entanglement Link
    """
    G = env.phys_graph
    
    # 1. Setup Layout (Compute once if possible, but here we do it dynamic)
    # Kamada-Kawai is good for hex/grid meshes
    pos = nx.kamada_kawai_layout(G) 
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # --- DRAW EDGES ---
    # Split edges by type
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'intra']
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'inter']
    
    # Draw Standard Intra-Chip connections
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='black', width=1.5, alpha=0.3)
    
    # Draw Inter-Chip Links (Dynamic Status)
    active_links = []
    inactive_links = []
    
    for u, v in inter_edges:
        # Check Link Status in Env
        # Try both directions just in case
        try: idx = env.inter_edges.index((u, v))
        except: idx = env.inter_edges.index((v, u))
            
        if env.link_status[idx] == 1:
            active_links.append((u, v))
        else:
            inactive_links.append((u, v))
            
    # Inactive = Dashed Gray
    nx.draw_networkx_edges(G, pos, edgelist=inactive_links, edge_color='gray', width=1.0, style='dashed', alpha=0.5)
    # Active = Thick Green
    nx.draw_networkx_edges(G, pos, edgelist=active_links, edge_color='#32CD32', width=4.0)

    # --- DRAW NODES ---
    node_colors = []
    labels = {}
    
    for n in G.nodes():
        l_id = env.phys_to_logical[n]
        
        # Case 1: Empty
        if l_id == -1:
            node_colors.append('white')
            labels[n] = f"{n}"
            
        # Case 2: Occupied
        else:
            # Is it Primary or Ghost?
            is_primary = (env.mapping[l_id] == n)
            
            if is_primary:
                node_colors.append('#87CEEB') # Sky Blue (Primary)
                labels[n] = f"{n}\nL{l_id}"
            else:
                node_colors.append('#FF6347') # Tomato Red (Ghost)
                labels[n] = f"{n}\n(L{l_id})"

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=600)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')

    # --- INFO BOX ---
    # Add a small text box with step info
    info_text = (
        f"Step: {env.steps}\n"
        f"Phase: {'Placement' if env.current_phase == 0 else 'Routing'}\n"
        f"Executed Gates: {len(env.tracker.executed_gates)} / {env.tracker.total_gates}"
    )
    plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- FINISH ---
    if title: plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"📸 Snapshot saved to {filename}")

def save_full_state(env, filename="full_state.png"):
    """
    Saves a side-by-side visualization:
    [Physical Device State] | [Circuit Window Graph]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # =========================================================================
    # LEFT PLOT: PHYSICAL DEVICE
    # =========================================================================
    G_phys = env.phys_graph
    pos_phys = nx.kamada_kawai_layout(G_phys)
    
    # 1. Edges
    intra_edges = [(u, v) for u, v, d in G_phys.edges(data=True) if d['type'] == 'intra']
    inter_edges = [(u, v) for u, v, d in G_phys.edges(data=True) if d['type'] == 'inter']
    
    nx.draw_networkx_edges(G_phys, pos_phys, edgelist=intra_edges, ax=ax1, edge_color='black', width=1.5, alpha=0.3)
    
    # Active Links (Green) vs Inactive (Gray)
    active_links, inactive_links = [], []
    for u, v in inter_edges:
        # Find index safely
        try: idx = env.inter_edges.index((u, v))
        except: idx = env.inter_edges.index((v, u))
        
        if env.link_status[idx] == 1: active_links.append((u, v))
        else: inactive_links.append((u, v))

    nx.draw_networkx_edges(G_phys, pos_phys, edgelist=inactive_links, ax=ax1, edge_color='gray', style='dashed', alpha=0.5)
    nx.draw_networkx_edges(G_phys, pos_phys, edgelist=active_links, ax=ax1, edge_color='limegreen', width=4.0)

    # 2. Nodes
    colors, labels = [], {}
    for n in G_phys.nodes():
        l_id = env.phys_to_logical[n]
        if l_id == -1:
            colors.append('white')
            labels[n] = f"{n}"
        else:
            # Check if Primary or Ghost
            if env.mapping[l_id] == n:
                colors.append('#87CEEB') # Blue (Primary)
                labels[n] = f"{n}\nL{l_id}"
            else:
                colors.append('#FF6347') # Red (Ghost)
                labels[n] = f"{n}\n(L{l_id})"
    
    nx.draw_networkx_nodes(G_phys, pos_phys, node_color=colors, edgecolors='black', node_size=600, ax=ax1)
    nx.draw_networkx_labels(G_phys, pos_phys, labels=labels, font_size=9, ax=ax1)
    ax1.set_title(f"Physical Device (Step {env.steps})")
    ax1.axis('off')

    # =========================================================================
    # RIGHT PLOT: WINDOW GRAPH (The Circuit)
    # =========================================================================
    # Get the graph directly from the environment
    # Note: We need to regenerate it or access the cached one if stored
    G_win = env.tracker.get_window_graph(env.window_depth)
    
    if G_win.number_of_nodes() == 0:
        ax2.text(0.5, 0.5, "Circuit Complete!", ha='center', fontsize=15)
        ax2.axis('off')
    else:
        # Hierarchy Layout (Topological sort usually looks best for DAGs)
        # Fallback to spring if generic
        try:
            pos_win = nx.multipartite_layout(G_win, subset_key="layer") # If layers exist
        except:
            pos_win = nx.spring_layout(G_win, k=0.5)

        # for n, d in G_win.nodes(data=True):
        #     print(n, d)

        # 1. Logical Qubit Nodes (Anchors)
        qubit_nodes = [n for n, d in G_win.nodes(data=True) if d.get('type', None) == 'qubit']
        gate_nodes = [n for n, d in G_win.nodes(data=True) if d.get('type', None) == 'gate']
        
        nx.draw_networkx_nodes(G_win, pos_win, nodelist=qubit_nodes, node_color='lightgrey', node_shape='s', node_size=500, ax=ax2, label="Qubits")
        
        # 2. Gate Nodes (Color by Readiness)
        ready_gates = [n for n in gate_nodes if G_win.nodes[n].get('ready', False)]
        blocked_gates = [n for n in gate_nodes if not G_win.nodes[n].get('ready', False)]
        
        nx.draw_networkx_nodes(G_win, pos_win, nodelist=ready_gates, node_color='gold', node_size=700, ax=ax2, label="Ready")
        nx.draw_networkx_nodes(G_win, pos_win, nodelist=blocked_gates, node_color='lightgrey', node_size=400, ax=ax2, label="Blocked")
        
        # 3. Edges
        # Dependencies (Gate->Gate) vs Participation (Qubit->Gate)
        dep_edges = [(u, v) for u, v, d in G_win.edges(data=True) if d.get('type') == 'dependency']
        part_edges = [(u, v) for u, v, d in G_win.edges(data=True) if d.get('type') != 'dependency']
        
        nx.draw_networkx_edges(G_win, pos_win, edgelist=dep_edges, ax=ax2, edge_color='black', arrows=True)
        nx.draw_networkx_edges(G_win, pos_win, edgelist=part_edges, ax=ax2, edge_color='blue', style='dotted', alpha=0.5)
        
        # Labels
        nx.draw_networkx_labels(G_win, pos_win, font_size=8, ax=ax2)
        ax2.set_title("Circuit Window (Next Operations)")
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"📸 Full State Snapshot saved to {filename}")