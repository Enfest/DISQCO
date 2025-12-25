import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import CouplingMap

# --- IMPORT CONFIG ---
# We need this to ensure consistency if needed, though mostly handled dynamically now
from disqco.learning.config import model_config

# ==============================================================================
# 1. HARDWARE GENERATORS (Unchanged)
# ==============================================================================
def example_device():
    # add 6 node graph
    G = nx.Graph()
    for i in range(10):
        G.add_node(i, type='data', id=i)
    # G.add_edge(0, 1, type='intra', weight=1.0)
    # G.add_edge(1, 2, type='intra', weight=1.0)
    # G.add_edge(2, 3, type='intra', weight=1.0)
    # G.add_edge(3, 5, type='intra', weight=1.0)
    # G.add_edge(5, 8, type='intra', weight=1.0)
    # G.add_edge(8, 9, type='intra', weight=1.0)
    # G.add_edge(8, 11, type='intra', weight=1.0)
    # G.add_edge(11, 14, type='intra', weight=1.0)
    # G.add_edge(14, 13, type='intra', weight=1.0)
    # G.add_edge(13, 12, type='intra', weight=1.0)
    # G.add_edge(12, 15, type='intra', weight=1.0)
    # G.add_edge(12, 10, type='intra', weight=1.0)
    # G.add_edge(10, 7, type='intra', weight=1.0)
    # G.add_edge(7, 6, type='intra', weight=1.0)
    # G.add_edge(7, 4, type='intra', weight=1.0)
    # G.add_edge(1, 4, type='intra', weight=1.0)
    # G.add_edge(15, 16, type='inter', weight=1.0)
    # G.nodes[15]['type'] = 'comm'
    # G.nodes[16]['type'] = 'comm'
    # G.add_edge(16, 17, type='intra', weight=1.0)
    # G.add_edge(17, 18, type='intra', weight=1.0)
    # G.add_edge(18, 19, type='intra', weight=1.0)
    # G.add_edge(19, 21, type='intra', weight=1.0)
    # G.add_edge(21, 24, type='intra', weight=1.0)
    # G.add_edge(24, 25, type='intra', weight=1.0)
    # G.add_edge(24, 27, type='intra', weight=1.0)
    # G.add_edge(27, 30, type='intra', weight=1.0)
    # G.add_edge(30, 29, type='intra', weight=1.0)
    # G.add_edge(29, 28, type='intra', weight=1.0)
    # G.add_edge(28, 31, type='intra', weight=1.0)
    # G.add_edge(28, 26, type='intra', weight=1.0)
    # G.add_edge(26, 23, type='intra', weight=1.0)
    # G.add_edge(23, 22, type='intra', weight=1.0)
    # G.add_edge(23, 20, type='intra', weight=1.0)
    # G.add_edge(17, 20, type='intra', weight=1.0)
    
    G.add_edge(0, 1, type='intra', weight=1.0)
    G.add_edge(1, 2, type='intra', weight=1.0)
    G.add_edge(2, 3, type='intra', weight=1.0)
    G.add_edge(3, 4, type='intra', weight=1.0)
    G.add_edge(0, 4, type='intra', weight=1.0)
    G.add_edge(4, 5, type='inter', weight=1.0)
    G.nodes[4]['type'] = 'comm'
    G.nodes[5]['type'] = 'comm'
    G.add_edge(5, 6, type='intra', weight=1.0)
    G.add_edge(6, 7, type='intra', weight=1.0)
    G.add_edge(7, 8, type='intra', weight=1.0)
    G.add_edge(8, 9, type='intra', weight=1.0)
    G.add_edge(9, 5, type='intra', weight=1.0)
    return G

# ==============================================================================
# 2. CIRCUIT TRACKER (With Safety Fix)
# ==============================================================================
class CircuitTracker:
    def __init__(self, qiskit_circuit, window_depth=5):
        self.dag = circuit_to_dag(qiskit_circuit)
        self.front_layer = set(self.dag.front_layer())
        self.executed_gates = set()
        self.total_gates = self.dag.size()
        self.logical_qubits = qiskit_circuit.qubits
        self.window_depth = window_depth
       
        # Map Gate Object -> Unique Integer ID for GNN
        self.node_to_id = {node: i for i, node in enumerate(self.dag.op_nodes())}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}

    def get_window_graph(self, depth=5):
        G = nx.DiGraph()
        depth = self.window_depth
        
        # 1. Add Logical Qubit Nodes
        for i in range(len(self.logical_qubits)):
            G.add_node(f"Q_{i}", type='qubit', id=i)

        # 2. BFS to find next 'depth' gates
        queue = [(node, 0) for node in self.front_layer]
        visited = set(n for n, _ in queue)
        
        while queue:
            node, d = queue.pop(0)
            if d >= depth: continue
            
            gate_id = f"G_{self.node_to_id[node]}"
            G.add_node(gate_id, type='gate', ready=(d==0))
            
            # Interaction Edges (Qubit -> Gate)
            weight = 1.0 / (d + 1.0)
            is_cnot = (node.name in ['cx', 'ecr', 'cz'])
            
            for idx, q in enumerate(node.qargs):
                q_idx = self.logical_qubits.index(q) # Safe index lookup
                role = 2 
                if len(node.qargs) == 2 and is_cnot:
                    if idx == 0: role = 0 
                    elif idx == 1: role = 1 
                
                G.add_edge(f"Q_{q_idx}", gate_id, weight=weight, role=role)

            # Dependency Edges
            for child in self.dag.successors(node):
                if isinstance(child, DAGOpNode):
                    child_id = f"G_{self.node_to_id[child]}"
                    G.add_edge(gate_id, child_id, type='dependency')
                    
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, d + 1))
        return G

    def execute_gate(self, gate_id):
        node = self.id_to_node[gate_id]
        if node not in self.front_layer: return
        
        self.front_layer.remove(node)
        self.executed_gates.add(node)
        
        # Unlock successors
        for child in self.dag.successors(node):
            if isinstance(child, DAGOpNode):
                ready = True
                for p in self.dag.predecessors(child):
                    if isinstance(p, DAGOpNode) and p not in self.executed_gates:
                        ready = False; break
                if ready: self.front_layer.add(child)

    def is_done(self):
        return len(self.executed_gates) == self.total_gates
    
    def get_partner(self, logical_idx):
        """Returns the logical index of the qubit 'logical_idx' needs to interact with."""
        for node in self.front_layer:
            # FIX: Use list lookup instead of private ._index
            qubits = [self.logical_qubits.index(q) for q in node.qargs] 
            
            if len(qubits) == 2:
                if qubits[0] == logical_idx: return qubits[1]
                if qubits[1] == logical_idx: return qubits[0]
        return -1

# ==============================================================================
# 3. MAIN ENVIRONMENT (Corrected)
# ==============================================================================
class AutoExecDistributedEnv(gym.Env):
    def __init__(self, qiskit_circuit, num_qpus=2, hex_distance=3, window_depth=6):
        super().__init__()
        
        # --- 1. SETUP ---
        self.phys_graph = example_device()
        self.num_phys = self.phys_graph.number_of_nodes()
        self.window_depth = window_depth
        
        # Edge Caching (Vectorized)
        intra_edges = [(u,v) for u,v,d in self.phys_graph.edges(data=True) if d['type']=='intra']
        self.intra_edges = intra_edges
        self.intra_u = np.array([u for u,v in intra_edges], dtype=int)
        self.intra_v = np.array([v for u,v in intra_edges], dtype=int)
        
        inter_edges = [(u,v) for u,v,d in self.phys_graph.edges(data=True) if d['type']=='inter']
        self.inter_edges = inter_edges
        self.inter_u = np.array([u for u,v in inter_edges], dtype=int)
        self.inter_v = np.array([v for u,v in inter_edges], dtype=int)
        
        # GPS System
        self.dist_matrix = dict(nx.all_pairs_shortest_path_length(self.phys_graph))
        self.diameter = 0
        for src in self.dist_matrix:
            curr_max = max(self.dist_matrix[src].values())
            if curr_max > self.diameter: self.diameter = curr_max

        # --- 2. LOAD CIRCUIT & DEFINE ACTIONS ---
        # We must call set_circuit to initialize logical_num, depth, and action space
        self.set_circuit(qiskit_circuit, window_depth)
        self.last_swap_edge = -1

    def set_circuit(self, qiskit_circuit, window_depth=None):
        """Hot-swap circuit, update dimensions, and rebuild interactions."""
        self.circuit_source = qiskit_circuit
        self.num_logical = len(qiskit_circuit.qubits)
        self.circuit_depth = qiskit_circuit.depth()
        
        # Update Window Depth
        wd = window_depth if window_depth else self.window_depth
        self.tracker = CircuitTracker(qiskit_circuit, window_depth=wd)
        
        # --- A. REBUILD INTERACTION GRAPH (For Placement Reward) ---
        self.interaction_adj = {i: [] for i in range(self.num_logical)}
        for gate in qiskit_circuit.data:
            qubits = [qiskit_circuit.qubits.index(q) for q in gate.qubits]
            if len(qubits) == 2:
                u, v = qubits[0], qubits[1]
                if v not in self.interaction_adj[u]: self.interaction_adj[u].append(v)
                if u not in self.interaction_adj[v]: self.interaction_adj[v].append(u)

        # --- B. DEFINE ACTION SPACE ---
        # 0..Place | Place..Swap | Swap..Inter
        self.limit_place = self.num_logical * self.num_phys
        self.limit_swap = self.limit_place + len(self.intra_edges)
        self.limit_inter = self.limit_swap + (3 * len(self.inter_edges))
        
        self.action_space = spaces.Discrete(self.limit_inter)
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.tracker = CircuitTracker(self.circuit_source, self.window_depth)
        
        # State Maps
        self.mapping = np.full(self.num_logical, -1, dtype=int)
        self.phys_to_logical = np.full(self.num_phys, -1, dtype=int)
        
        self.link_status = np.zeros(len(self.inter_edges), dtype=int) 
        self.phys_is_entangled = np.zeros(self.num_phys, dtype=int)
        self.qubit_clock = np.zeros(self.num_phys, dtype=float)
        
        self.current_phase = 0
        self.placed_count = 0
        self.steps = 0

        # NEW: Track the last SWAP edge index to detect reversals
        # We use -1 to mean "No previous swap"
        self.last_swap_edge = -1
        
        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # Dynamic Timeout (Prevents Zombie)
        allowed_steps = self.circuit_depth * model_config["num_qubits"]*1.2
        truncated = (self.steps > allowed_steps)

        if truncated:
             # Major penalty for timeout
            return self._get_observation(), -1000.0, False, True, {"is_success": False}

        # --- PHASE 0: PLACEMENT ---
        if self.current_phase == 0:
            if action >= self.limit_place: 
                return self._get_observation(), -1000.0, False, truncated, {} # Invalid
            
            l_q = action // self.num_phys
            p_q = action % self.num_phys
            
            # Check validity (Spot already taken?)
            if self.phys_to_logical[p_q] != -1:
                 return self._get_observation(), -self.circuit_depth, False, truncated, {} 

            self._update_location(l_q, p_q)
            self.placed_count += 1
            
            # === PLACEMENT REWARD ===
            # Bonus for clustering interacting qubits
            placement_bonus = 0.0
            partners = self.interaction_adj[l_q]
            for index, partner_l in enumerate(partners):
                if index >= 5: break # Only consider first 5 partners to limit computation
                partner_p = self.mapping[partner_l]
                if partner_p != -1:
                    dist = self.dist_matrix[p_q][partner_p]
                    # Higher reward for closer placement
                    placement_bonus += 1.0 / (dist + 0.1) / len(partners)
            
            reward += placement_bonus * 0.1
            
            # Transition to Routing?
            if self.placed_count == self.num_logical:
                self.current_phase = 1
                r, done = self._auto_execute_gates()
                reward += r
                terminated = done
                
        # --- PHASE 1: ROUTING ---
        else:
            cost = self._step_routing(action)
            reward += cost
            
            exec_r, done = self._auto_execute_gates()
            reward += exec_r
            terminated = done
            
        info = {"action_mask": self.get_action_mask()}
        if terminated: info["is_success"] = True
            
        return self._get_observation(), reward, terminated, truncated, info

    def _step_routing(self, action):
        # 1. SNAPSHOT DISTANCE
        prev_dist = self._get_front_layer_sum_distance()
        
        # Validate Phase
        if action < self.limit_place: return -self.circuit_depth # Penalty 

        # A. SWAP
        if action < self.limit_swap:

            idx = action - self.limit_place

            if idx == self.last_swap_edge:
                # PENALTY!
                # We return a large negative reward to say "This was stupid."
                # We also typically do NOT execute the swap to save time, 
                # or we execute it but punish heavily.
                # Let's execute it (so state changes) but punish hard.
                self.last_swap_edge = -1 # Reset so we don't punish 3 times in a row weirdly
                return -self.circuit_depth
            
            # If not redundant, update the tracker
            self.last_swap_edge = idx

            u, v = self.intra_edges[idx]
            l_u, l_v = self.phys_to_logical[u], self.phys_to_logical[v]
            
            # Update Clock
            start_time = max(self.qubit_clock[u], self.qubit_clock[v])
            finish_time = start_time + 3.0 # SWAP Cost
            self.qubit_clock[u] = finish_time
            self.qubit_clock[v] = finish_time
            
            duration = 3.0 # For reward calculation

            # Swap Logic
            self.phys_to_logical[u], self.phys_to_logical[v] = l_v, l_u
            if l_u != -1 and self.mapping[l_u] == u: self.mapping[l_u] = v
            if l_v != -1 and self.mapping[l_v] == v: self.mapping[l_v] = u

        # B. INTER-OPS
        elif action < self.limit_inter:

            self.last_swap_edge = -1

            offset = action - self.limit_swap
            link_idx = offset // 3
            op = offset % 3
            u, v = self.inter_edges[link_idx]
            l_u, l_v = self.phys_to_logical[u], self.phys_to_logical[v]
            
            duration = 0.0
            
            if op == 0: # Entangle
                duration = 10.0
                self.link_status[link_idx] = 1
                self.phys_is_entangled[u] = 1; self.phys_is_entangled[v] = 1
                # Ghost Creation
                if l_u != -1 and l_v == -1: self.phys_to_logical[v] = l_u
                elif l_v != -1 and l_u == -1: self.phys_to_logical[u] = l_v

            else: # Consume (Teleport/Disentangle)
                duration = 2.0
                self.link_status[link_idx] = 0
                self.phys_is_entangled[u] = 0; self.phys_is_entangled[v] = 0
                
                if op == 1: src, dst, l_src = v, u, l_v
                else:       src, dst, l_src = u, v, l_u
                
                if l_src != -1:
                    # Move or Merge
                    if self.phys_to_logical[dst] == -1 or self.phys_to_logical[dst] == l_src:
                        self._update_location(l_src, dst)
                        self.phys_to_logical[src] = -1

            # Update Clock
            t = max(self.qubit_clock[u], self.qubit_clock[v]) + duration
            self.qubit_clock[u] = t; self.qubit_clock[v] = t

        else:
            return -1000.0 # Out of bounds

        # --- REWARD CALCULATION ---
        # 1. Distance Improvement Bonus
        curr_dist = self._get_front_layer_sum_distance()
        dist_diff = prev_dist - curr_dist
        
        # 2. Combined Reward
        # Penalty is proportional to DURATION, not absolute time.
        # Bonus for distance reduction.
        reward = -(duration * 0.1) + (dist_diff * 0.02)
        
        return reward

    def _auto_execute_gates(self):
        total_r = 0.0
        while True:
            ran_any = False
            ready_gates = sorted(list(self.tracker.front_layer), key=lambda x: self.tracker.node_to_id[x])
            
            for node in ready_gates:
                self.last_swap_edge = -1
                qubits = [self.circuit_source.qubits.index(q) for q in node.qargs]
                
                # Single Qubit
                if len(qubits) == 1:
                    p_q = self.mapping[qubits[0]]
                    self.qubit_clock[p_q] += 1.0
                    self.tracker.execute_gate(self.tracker.node_to_id[node])
                    total_r += - (1.0 * 0.01) # Small cost
                    ran_any = True

                # CNOT
                elif len(qubits) == 2:
                    l_c, l_t = qubits[0], qubits[1]
                    p_c, p_t = self.mapping[l_c], self.mapping[l_t]
                    
                    if self.phys_is_entangled[p_t]: continue
                    
                    # Logic: Check if ANY valid control location is neighbor to Target
                    control_locs = np.where(self.phys_to_logical == l_c)[0]
                    valid_pc = -1
                    can_run = False
                    
                    for pc_cand in control_locs:
                        if self.phys_graph.has_edge(pc_cand, p_t):
                            if self.phys_graph[pc_cand][p_t]['type'] == 'intra':
                                valid_pc = pc_cand; can_run = True; break
                                
                    if can_run:
                        duration = 1.0
                        t = max(self.qubit_clock[valid_pc], self.qubit_clock[p_t]) + duration
                        self.qubit_clock[valid_pc] = t
                        self.qubit_clock[p_t] = t
                        
                        total_r += - (duration * 0.01) + 1.0 # +1.0 Bonus for executing gate
                        self.tracker.execute_gate(self.tracker.node_to_id[node])
                        ran_any = True
            
            if not ran_any: break
        return total_r, self.tracker.is_done()

    def _get_front_layer_sum_distance(self):
        total = 0.0
        for node in self.tracker.front_layer:
            qs = [self.circuit_source.qubits.index(q) for q in node.qargs]
            if len(qs) == 2:
                p1, p2 = self.mapping[qs[0]], self.mapping[qs[1]]
                if p1 != -1 and p2 != -1: total += self.dist_matrix[p1][p2]
        return total

    def _update_location(self, l_q, p_q):
        old_p = self.mapping[l_q]
        if old_p != -1: self.phys_to_logical[old_p] = -1
        self.mapping[l_q] = p_q
        self.phys_to_logical[p_q] = l_q

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.current_phase == 0:
            log_free = (self.mapping == -1)
            phys_free = (self.phys_to_logical == -1)
            # Flatten 2D mask (Num_Logical x Num_Phys)
            mask[:self.limit_place] = np.outer(log_free, phys_free).flatten()
            
        else:
            # 1. SWAP
            u_occ = self.phys_to_logical[self.intra_u] != -1
            v_occ = self.phys_to_logical[self.intra_v] != -1
            mask[self.limit_place : self.limit_swap] = (u_occ | v_occ)

            # 2. INTER-OPS
            is_entangled = (self.link_status == 1)
            l_u = self.phys_to_logical[self.inter_u]
            l_v = self.phys_to_logical[self.inter_v]
            
            can_spread_u = (l_u != -1) & (l_v == -1)
            can_spread_v = (l_v != -1) & (l_u == -1)
            mask_entangle = (~is_entangled) & (can_spread_u | can_spread_v)
            
            dest_u_safe = (l_u == -1) | (l_u == l_v)
            mask_v_to_u = is_entangled & (l_v != -1) & dest_u_safe
            dest_v_safe = (l_v == -1) | (l_v == l_u)
            mask_u_to_v = is_entangled & (l_u != -1) & dest_v_safe
            
            base = self.limit_swap
            mask[base + 0 :: 3] = mask_entangle
            mask[base + 1 :: 3] = mask_v_to_u
            mask[base + 2 :: 3] = mask_u_to_v

        return mask

    def _get_observation(self):
        return {
            "phys_graph": self.phys_graph,
            "window_graph": self.tracker.get_window_graph(self.window_depth),
            "mapping": self.mapping.copy(),
            "phys_to_logical": self.phys_to_logical.copy(),
            "phys_is_entangled": self.phys_is_entangled.copy(),
            "qubit_clock": self.qubit_clock.copy(),
            "phase": self.current_phase
        }