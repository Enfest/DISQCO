import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, AttentionalAggregation, GlobalAttention
from torch_geometric.data import Data

# ==============================================================================
# 1. CUSTOM DATA STRUCTURE (Handling Custom Batching)
# ==============================================================================
class DistributedGraphData(HeteroData):
    def __inc__(self, key, value, *args, **kwargs):
        # Allow batching of custom indices by shifting them by num_physical nodes
        if key == 'intra_edge_index' or key == 'inter_edge_index':
            return self['physical'].num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'intra_edge_index' or key == 'inter_edge_index':
            return 1 # Concat along columns
        return super().__cat_dim__(key, value, *args, **kwargs)

# ==============================================================================
# 2. DATA CONVERTER (Obs -> PyG)
# ==============================================================================

def obs_to_pyg(obs, env):
    """
    Converts Observation -> PyTorch Geometric HeteroData.
    Robustly handles circuit graph parsing.
    """
    
    data = HeteroData()
    
    # Unpack
    phys_graph = obs['phys_graph']
    window_graph = obs['window_graph']
    mapping = obs['mapping']
    phys_to_logical = obs['phys_to_logical']
    phys_is_entangled = obs['phys_is_entangled']
    qubit_clock = obs['qubit_clock']

    # =========================================================================
    # PART A: PHYSICAL NODES (Hardware + GPS)
    # =========================================================================
    node_features = []
    node_targets = {} 

    for p_node in range(env.num_phys):
        l_qubit = phys_to_logical[p_node]
        
        # GPS Defaults
        norm_dist = 0.0 
        is_remote = 0.0
        
        if l_qubit != -1:
            partner_l = env.tracker.get_partner(l_qubit)
            if partner_l != -1:
                p_partner = mapping[partner_l]
                if p_partner != -1:
                    raw_dist = env.dist_matrix[p_node][p_partner]
                    norm_dist = raw_dist / env.diameter
                    node_targets[p_node] = p_partner 
                    if raw_dist > 2: is_remote = 1.0

        feat = [
            1.0 if l_qubit != -1 else 0.0,
            float(phys_is_entangled[p_node]),
            qubit_clock[p_node] / 100.0, 
            norm_dist,
            is_remote
        ]
        node_features.append(feat)

    data['physical'].x = torch.tensor(node_features, dtype=torch.float)
    data['physical'].num_nodes = env.num_phys

    # =========================================================================
    # PART B: LOGICAL NODES (Software Qubits)
    # =========================================================================
    data['logical'].x = torch.ones((env.num_logical, 1), dtype=torch.float)
    # NEW (Universal): Constant Feature '1.0' for all qubits
    # Shape is [Num_Logical, 1]
    # data['logical'].x = torch.ones((env.num_logical, 1), dtype=torch.float)
    data['logical'].num_nodes = env.num_logical

    # =========================================================================
    # PART C: GATE NODES (Circuit Ops)
    # =========================================================================
    gate_nodes_nx = [n for n, d in window_graph.nodes(data=True) if d.get('type') == 'gate']
    
    gate_vectors = []
    nx_to_gate_idx = {} 
    
    gate_map = {'h':0, 't':1, 's':2, 'x':3, 'y':4, 'z':5, 'rx':6, 'ry':7, 'rz':8, 'cx':9, 'cz':10, 'swap':11}
    
    for idx, node_id in enumerate(gate_nodes_nx):
        nx_to_gate_idx[node_id] = idx
        attrs = window_graph.nodes[node_id]
        
        g_type = float(gate_map.get(attrs.get('gate', 'cx'), 9))
        is_ready = 1.0 if attrs.get('ready', False) else 0.0
        
        gate_vectors.append([g_type, is_ready])
        
    if gate_vectors:
        data['gate'].x = torch.tensor(gate_vectors, dtype=torch.float)
    else:
        data['gate'].x = torch.empty((0, 2), dtype=torch.float)

    # =========================================================================
    # PART D: EDGES (Connecting Everything)
    # =========================================================================
    
    # 1. PHYSICAL (Intra/Inter)
    intra_src, intra_dst, intra_attr = [], [], []
    inter_src, inter_dst, inter_attr = [], [], []
    
    for u, v, d in phys_graph.edges(data=True):
        for src, dst in [(u, v), (v, u)]:
            is_towards = 0.0
            if src in node_targets:
                target = node_targets[src]
                if src != target and env.dist_matrix[dst][target] < env.dist_matrix[src][target]:
                    is_towards = 1.0 

            if d['type'] == 'intra':
                intra_src.append(src); intra_dst.append(dst); intra_attr.append([1.0, is_towards])
            elif d['type'] == 'inter':
                inter_src.append(src); inter_dst.append(dst); inter_attr.append([1.0, is_towards])

    data['physical', 'intra', 'physical'].edge_index = torch.tensor([intra_src, intra_dst], dtype=torch.long)
    data['physical', 'intra', 'physical'].edge_attr = torch.tensor(intra_attr, dtype=torch.float)
    data['physical', 'inter', 'physical'].edge_index = torch.tensor([inter_src, inter_dst], dtype=torch.long)
    data['physical', 'inter', 'physical'].edge_attr = torch.tensor(inter_attr, dtype=torch.float)

    # 2. MAPPING ('logical' -> 'physical')
    map_src, map_dst = [], []
    for l_idx, p_loc in enumerate(mapping):
        if p_loc != -1: map_src.append(l_idx); map_dst.append(p_loc)
    
    if map_src:
        data['logical', 'mapped_to', 'physical'].edge_index = torch.tensor([map_src, map_dst], dtype=torch.long)
    else:
        data['logical', 'mapped_to', 'physical'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3. PARTICIPATION ('logical' -> 'gate') & DEPENDENCY ('gate' -> 'gate')
    part_src, part_dst = [], []
    dep_src, dep_dst = [], []
    
    for u, v, d in window_graph.edges(data=True):
        u_attrs = window_graph.nodes[u]
        v_attrs = window_graph.nodes[v]
        
        u_type = u_attrs.get('type')
        v_type = v_attrs.get('type')
        
        # --- FIX STARTS HERE ---
        if u_type == 'qubit' and v_type == 'gate':
            # Try to get index from attributes, fallback to node ID if it's an int
            l_idx = u_attrs.get('index')
            if l_idx is None and isinstance(u, int):
                l_idx = u
            
            if l_idx is not None and v in nx_to_gate_idx:
                part_src.append(l_idx)
                part_dst.append(nx_to_gate_idx[v])
        # --- FIX ENDS HERE ---
        
        elif u_type == 'gate' and v_type == 'gate':
            if u in nx_to_gate_idx and v in nx_to_gate_idx:
                dep_src.append(nx_to_gate_idx[u])
                dep_dst.append(nx_to_gate_idx[v])

    if part_src:
        data['logical', 'participates', 'gate'].edge_index = torch.tensor([part_src, part_dst], dtype=torch.long)
    else:
        data['logical', 'participates', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    if dep_src:
        data['gate', 'dependency', 'gate'].edge_index = torch.tensor([dep_src, dep_dst], dtype=torch.long)
    else:
        data['gate', 'dependency', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

# ==============================================================================
# 3. GNN ARCHITECTURE
# ==============================================================================
class DistributedQCompilerGNN(torch.nn.Module):
    # CHANGED: Added 'action_dim' to arguments
    def __init__(self, num_phys_nodes, num_logical_qubits, action_dim, hidden_dim=64):
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        
        # 1. ENCODERS
        self.phys_encoder = nn.Linear(5, hidden_dim)
       # --- CHANGE THIS LINE ---
        # OLD (Bad): Tied to specific qubit ID logic
        # self.logical_encoder = nn.Linear(num_logical_qubits, hidden_dim)
        
        # NEW (Correct): Universal "I am a Qubit" feature
        # The GNN will distinguish them based on their connections to Gates.
        self.logical_encoder = nn.Linear(1, hidden_dim)

        # NEW (Universal): Just a "I am a Qubit" signal
        # self.logical_encoder = nn.Linear(1, hidden_dim)
        self.gate_encoder = nn.Linear(2, hidden_dim)
        
        # 2. CONVOLUTIONS
        self.conv1 = HeteroConv({
            ('physical', 'intra', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('physical', 'inter', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('logical', 'mapped_to', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('logical', 'participates', 'gate'): SAGEConv(hidden_dim, hidden_dim),
            ('gate', 'dependency', 'gate'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('physical', 'intra', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('physical', 'inter', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('logical', 'mapped_to', 'physical'): SAGEConv(hidden_dim, hidden_dim),
            ('logical', 'participates', 'gate'): SAGEConv(hidden_dim, hidden_dim),
            ('gate', 'dependency', 'gate'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        # 3. HEADS
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        
        # Actor Head (Policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # CHANGED: Use dynamic 'action_dim' instead of hardcoded 1000
            nn.Linear(hidden_dim, action_dim) 
        )
        
        # Critic Head (Value)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # 1. Encode
        x_dict = {}
        x_dict['physical'] = F.relu(self.phys_encoder(data['physical'].x))
        x_dict['logical'] = F.relu(self.logical_encoder(data['logical'].x))
        x_dict['gate'] = F.relu(self.gate_encoder(data['gate'].x))
        
        # 2. Message Passing (With Static Node Preservation)
        out_dict1 = self.conv1(x_dict, data.edge_index_dict)
        for key in x_dict.keys():
            if key not in out_dict1:
                out_dict1[key] = x_dict[key]
            else:
                out_dict1[key] = F.relu(out_dict1[key])
        
        out_dict2 = self.conv2(out_dict1, data.edge_index_dict)
        x_dict_final = {k: F.relu(v) for k, v in out_dict2.items()}
        
        # 3. Pool & Predict
        graph_embedding = self.pool(x_dict_final['physical'], data['physical'].batch)
        
        logits = self.actor_head(graph_embedding)
        value = self.critic_head(graph_embedding)
        
        return logits, value