import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
# IMPORT GATv2Conv HERE
from torch_geometric.nn import HeteroConv, SAGEConv, AttentionalAggregation, GATv2Conv

# ==============================================================================
# 1. DATA CONVERTER (Obs -> PyG)
# ==============================================================================
def obs_to_pyg(obs, env):
    data = HeteroData()
    
    # Unpack
    phys_graph = obs['phys_graph']
    window_graph = obs['window_graph']
    mapping = obs['mapping']
    phys_to_logical = obs['phys_to_logical']
    phys_is_entangled = obs['phys_is_entangled']
    qubit_clock = obs['qubit_clock']

    # --- A. PHYSICAL NODES ---
    node_features = []
    node_targets = {} 

    for p_node in range(env.num_phys):
        l_qubit = phys_to_logical[p_node]
        
        # GPS Feature Calculation
        norm_dist = 0.0 
        is_remote = 0.0
        if l_qubit != -1:
            partner_l = env.tracker.get_partner(l_qubit)
            if partner_l != -1:
                p_partner = mapping[partner_l]
                if p_partner != -1:
                    raw_dist = env.dist_matrix[p_node][p_partner]
                    norm_dist = raw_dist / (env.diameter + 1e-5)
                    node_targets[p_node] = p_partner 
                    if raw_dist > 2: is_remote = 1.0

        feat = [
            1.0 if l_qubit != -1 else 0.0,    # Is Occupied?
            float(phys_is_entangled[p_node]), # Is Entangled?
            qubit_clock[p_node] * 0.01,       # Normalized Clock
            norm_dist,                        # Distance to Partner
            is_remote                         # Is Partner Far?
        ]
        node_features.append(feat)

    data['physical'].x = torch.tensor(node_features, dtype=torch.float)
    data['physical'].num_nodes = env.num_phys

    # --- B. LOGICAL NODES (UNIVERSAL FIX) ---
    # Constant feature [1.0] for every logical qubit.
    data['logical'].x = torch.ones((env.num_logical, 1), dtype=torch.float)
    data['logical'].num_nodes = env.num_logical

    # --- C. GATE NODES ---
    gate_nodes_nx = [n for n, d in window_graph.nodes(data=True) if d.get('type') == 'gate']
    gate_vectors = []
    nx_to_gate_idx = {} 
    
    # Simple ID mapping for gates: 0=H, 1=T ... 9=CNOT
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

    # --- D. EDGES ---
    # 1. PHYSICAL (Intra/Inter)
    intra_src, intra_dst = [], []
    inter_src, inter_dst = [], []
    
    for u, v, d in phys_graph.edges(data=True):
        for src, dst in [(u, v), (v, u)]:
            if d['type'] == 'intra':
                intra_src.append(src); intra_dst.append(dst)
            elif d['type'] == 'inter':
                inter_src.append(src); inter_dst.append(dst)

    data['physical', 'intra', 'physical'].edge_index = torch.tensor([intra_src, intra_dst], dtype=torch.long)
    data['physical', 'inter', 'physical'].edge_index = torch.tensor([inter_src, inter_dst], dtype=torch.long)

    # 2. MAPPING
    map_src, map_dst = [], []
    for l_idx, p_loc in enumerate(mapping):
        if p_loc != -1: map_src.append(l_idx); map_dst.append(p_loc)
    
    if map_src:
        data['logical', 'mapped_to', 'physical'].edge_index = torch.tensor([map_src, map_dst], dtype=torch.long)
    else:
        data['logical', 'mapped_to', 'physical'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3. INTERACTION & DEPENDENCY
    part_src, part_dst = [], []
    dep_src, dep_dst = [], []
    
    for u, v, d in window_graph.edges(data=True):
        u_type = window_graph.nodes[u].get('type')
        v_type = window_graph.nodes[v].get('type')
        
        # Logical -> Gate
        if u_type == 'qubit' and v_type == 'gate':
            l_idx = window_graph.nodes[u].get('index', u if isinstance(u, int) else -1)
            if l_idx != -1 and v in nx_to_gate_idx:
                part_src.append(l_idx)
                part_dst.append(nx_to_gate_idx[v])
        
        # Gate -> Gate
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
# 2. GNN ARCHITECTURE (UPGRADED TO GATv2)
# ==============================================================================
class DistributedQCompilerGNN(torch.nn.Module):
    def __init__(self, num_phys_nodes, num_logical_qubits, action_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim 
        
        # 1. ENCODERS
        self.phys_encoder = nn.Linear(5, hidden_dim)
        self.logical_encoder = nn.Linear(1, hidden_dim) # Universal Feature
        self.gate_encoder = nn.Linear(2, hidden_dim)
        
        # 2. CONVOLUTIONS
        # CRITICAL FIX: Set add_self_loops=False for cross-type edges!
        
        self.conv1 = HeteroConv({
            # Physical -> Physical (Same type: Self-loops OK)
            ('physical', 'intra', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
            ('physical', 'inter', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
            
            # Logical -> Physical (Different types: NO Self-loops)
            ('logical', 'mapped_to', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=False),
            
            # Logical -> Gate (Different types: NO Self-loops)
            ('logical', 'participates', 'gate'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=False),
            
            # Gate -> Gate (Same type: Self-loops OK)
            ('gate', 'dependency', 'gate'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('physical', 'intra', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
            ('physical', 'inter', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
            
            # Fix applied here too
            ('logical', 'mapped_to', 'physical'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=False),
            ('logical', 'participates', 'gate'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=False),
            
            ('gate', 'dependency', 'gate'): GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False),
        }, aggr='sum')
        
        # 3. HEADS
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # ... (Forward logic remains exactly the same) ...
        # 1. Encode
        x_dict = {}
        x_dict['physical'] = F.relu(self.phys_encoder(data['physical'].x))
        x_dict['logical'] = F.relu(self.logical_encoder(data['logical'].x))
        x_dict['gate'] = F.relu(self.gate_encoder(data['gate'].x))
        
        # 2. Message Passing
        out_dict1 = self.conv1(x_dict, data.edge_index_dict)
        for key in x_dict.keys():
            if key not in out_dict1: out_dict1[key] = x_dict[key]
            else: out_dict1[key] = F.relu(out_dict1[key])
        
        out_dict2 = self.conv2(out_dict1, data.edge_index_dict)
        x_dict_final = {k: F.relu(v) for k, v in out_dict2.items()}
        
        # 3. Pool & Predict
        graph_embedding = self.pool(x_dict_final['physical'], data['physical'].batch)
        
        logits = self.actor_head(graph_embedding)
        value = self.critic_head(graph_embedding)
        
        return logits, value