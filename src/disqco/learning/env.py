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
            # Check gate type for simple visualization
            is_cnot = (node.name in ['cx', 'ecr', 'cz', 'cp'])
            
            for idx, q in enumerate(node.qargs):
                q_idx = self.logical_qubits.index(q)
                role = 2 # Default role
                if len(node.qargs) == 2 and is_cnot:
                    role = idx # 0=Control, 1=Target
                
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
            qubits = [self.logical_qubits.index(q) for q in node.qargs]
            if len(qubits) == 2:
                if qubits[0] == logical_idx: return qubits[1]
                if qubits[1] == logical_idx: return qubits[0]
        return -1

# ==============================================================================
# 3. MAIN ENVIRONMENT (Pure Negative Reward System)
# ==============================================================================
class AutoExecDistributedEnv(gym.Env):
    def __init__(self, qiskit_circuit, num_qpus=2, hex_distance=3, window_depth=6):
        super().__init__()
        
        # --- 1. SETUP ---
        self.phys_graph = example_device()
        self.num_phys = self.phys_graph.number_of_nodes()
        self.window_depth = window_depth
        
        # Edge Caching
        intra_edges = [(u,v) for u,v,d in self.phys_graph.edges(data=True) if d['type']=='intra']
        self.intra_edges = intra_edges
        self.intra_u = np.array([u for u,v in intra_edges], dtype=int)
        self.intra_v = np.array([v for u,v in intra_edges], dtype=int)
        
        inter_edges = [(u,v) for u,v,d in self.phys_graph.edges(data=True) if d['type']=='inter']
        self.inter_edges = inter_edges
        self.inter_u = np.array([u for u,v in inter_edges], dtype=int)
        self.inter_v = np.array([v for u,v in inter_edges], dtype=int)
        
        self.dist_matrix = dict(nx.all_pairs_shortest_path_length(self.phys_graph))
        self.diameter = 0
        for src in self.dist_matrix:
            curr_max = max(self.dist_matrix[src].values())
            if curr_max > self.diameter: self.diameter = curr_max

        # --- 2. LOAD CIRCUIT & DEFINE ACTIONS ---
        self.set_circuit(qiskit_circuit, window_depth)
        self.last_swap_edge = -1

    def set_circuit(self, qiskit_circuit, window_depth=None):
        self.circuit_source = qiskit_circuit
        self.num_logical = len(qiskit_circuit.qubits)
        self.circuit_depth = qiskit_circuit.depth()
        
        wd = window_depth if window_depth else self.window_depth
        self.tracker = CircuitTracker(qiskit_circuit, window_depth=wd)
        
        # Interaction Map for Placement
        self.interaction_adj = {i: [] for i in range(self.num_logical)}
        for gate in qiskit_circuit.data:
            qubits = [qiskit_circuit.qubits.index(q) for q in gate.qubits]
            if len(qubits) == 2:
                u, v = qubits[0], qubits[1]
                if v not in self.interaction_adj[u]: self.interaction_adj[u].append(v)
                if u not in self.interaction_adj[v]: self.interaction_adj[v].append(u)

        # Action Space
        self.limit_place = self.num_logical * self.num_phys
        self.limit_swap = self.limit_place + len(self.intra_edges)
        self.limit_inter = self.limit_swap + (3 * len(self.inter_edges))
        
        self.action_space = spaces.Discrete(self.limit_inter)
        self.reset()

    def reset(self, seed=None, use_heuristic_placement=False):
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
        self.last_swap_edge = -1
        
        # Optional: Heuristic Warmup
        if use_heuristic_placement:
            # ... (Existing heuristic logic) ...
            self.current_phase = 1
            self.placed_count = self.num_logical

        # --- NEW: INITIALIZE COMPASS ---
        # We calculate the starting "Stress" (Total Distance)
        self.prev_dist = self._get_front_layer_sum_distance()
        
        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # --- SAFETY TIMEOUT (Massive Penalty) ---
        # If the agent fails to finish, it gets hit hard (-1000).
        # This makes "finishing slowly" (-50) infinitely better than "timeout" (-1000).
        max_duration = max(50, self.circuit_depth * 8)
        truncated = (self.steps > max_duration) 
        if truncated:
             return self._get_observation(), -1000.0, False, True, {"is_success": False}

        # --- PHASE 0: PLACEMENT ---
        if self.current_phase == 0:
            if action >= self.limit_place: 
                return self._get_observation(), -1.0, False, truncated, {} 
            
            l_q = action // self.num_phys
            p_q = action % self.num_phys
            
            if self.phys_to_logical[p_q] != -1: 
                 return self._get_observation(), -1.0, False, truncated, {} 

            self._update_location(l_q, p_q)
            self.placed_count += 1
            
            # REWARD: Always 0.0 per step
            reward = 0.0
            
            # END OF PHASE EVALUATION
            if self.placed_count == self.num_logical:
                self.current_phase = 1
                
                # Calculate "Stress" (Average Distance of Front Layer)
                stress = 0.0
                count = 0
                for node in self.tracker.front_layer:
                    qubits = [self.circuit_source.qubits.index(q) for q in node.qargs]
                    if len(qubits) == 2:
                        p1, p2 = self.mapping[qubits[0]], self.mapping[qubits[1]]
                        stress += self.dist_matrix[p1][p2]
                        count += 1
                
                avg_dist = stress / max(1, count)
                
                # REWARD: Pure Negative Cost
                # Good placement (dist 1.0) -> Reward -1.0
                # Bad placement (dist 4.0) -> Reward -4.0
                reward = -avg_dist

        # --- PHASE 1: ROUTING (Pure Negative) ---
        else:
            # 1. Action Penalty (Based on Makespan Delay)
            step_penalty = self._step_routing(action)
            reward += step_penalty
            
            # 2. Execute Gates (No Bonus)
            # We run gates to advance the state, but give ZERO reward points.
            # The "Reward" is implicit: The penalties STOP accumulating when we finish.
            _, done = self._auto_execute_gates()
            
            terminated = done
            
            # 3. Completion (No Bonus)
            # If done, we simply return. The accumulated reward is frozen.
            if terminated:
                pass # reward += 0.0

        info = {"action_mask": self.get_action_mask()}
        if terminated: info["is_success"] = True
            
        return self._get_observation(), reward, terminated, truncated, info

    def _step_routing(self, action):
        # ======================================================================
        # 1. SNAPSHOT STATE (The Baseline)
        # ======================================================================
        old_makespan = np.max(self.qubit_clock) if len(self.qubit_clock) > 0 else 0
        old_dist = self.prev_dist # Uses the cached value from previous step
        
        # Safety Check
        if action < self.limit_place: return -1.0 

        # ======================================================================
        # 2. EXECUTE ACTION (Physics Simulation)
        # ======================================================================
        
        # --- A. SWAP (INTRA-CHIP) ---
        if action < self.limit_swap:
            idx = action - self.limit_place
            
            # Redundancy Check (Stop "Ping-Pong" moves)
            if idx == self.last_swap_edge:
                self.last_swap_edge = -1
                return -5.0 # Immediate penalty for stupidity
            
            self.last_swap_edge = idx
            u, v = self.intra_edges[idx]
            l_u, l_v = self.phys_to_logical[u], self.phys_to_logical[v]
            
            # Physics: Update Clocks (Parallelism aware)
            # A SWAP waits for both qubits to be free, then takes 3.0ns
            start_time = max(self.qubit_clock[u], self.qubit_clock[v])
            finish_time = start_time + 3.0 
            self.qubit_clock[u] = finish_time
            self.qubit_clock[v] = finish_time
            
            # Logic: Swap locations
            self.phys_to_logical[u], self.phys_to_logical[v] = l_v, l_u
            if l_u != -1: self.mapping[l_u] = v
            if l_v != -1: self.mapping[l_v] = u

        # --- B. INTER-OPS (CROSS-CHIP) ---
        elif action < self.limit_inter:
            self.last_swap_edge = -1 # Reset redundancy
            
            offset = action - self.limit_swap
            link_idx = offset // 3
            op = offset % 3
            u, v = self.inter_edges[link_idx]
            l_u, l_v = self.phys_to_logical[u], self.phys_to_logical[v]
            
            # Physics: Durations vary by operation
            duration = 10.0 if op == 0 else 2.0 
            
            # Logic: Entanglement State Machine
            if op == 0: # Entangle
                self.link_status[link_idx] = 1
                self.phys_is_entangled[u] = 1; self.phys_is_entangled[v] = 1
                # "Ghost" Logic (Optional: Pre-assign destination)
                if l_u != -1 and l_v == -1: self.phys_to_logical[v] = l_u
                elif l_v != -1 and l_u == -1: self.phys_to_logical[u] = l_v
            
            else: # Consume (Teleport / Move)
                self.link_status[link_idx] = 0
                self.phys_is_entangled[u] = 0; self.phys_is_entangled[v] = 0
                
                # Move Logic
                if op == 1: # Move V -> U
                    if l_v != -1: self._update_location(l_v, u); self.phys_to_logical[v] = -1
                else:       # Move U -> V
                    if l_u != -1: self._update_location(l_u, v); self.phys_to_logical[u] = -1

            # Update Clocks
            t = max(self.qubit_clock[u], self.qubit_clock[v]) + duration
            self.qubit_clock[u] = t; self.qubit_clock[v] = t

        else:
            return -10.0 # Out of bounds error

        # ======================================================================
        # 3. CALCULATE HYBRID REWARD
        # ======================================================================
        
        # --- COMPONENT A: MAKESPAN (The Stick) ---
        # Did we push the finish line further away?
        new_makespan = np.max(self.qubit_clock)
        delay = new_makespan - old_makespan
        
        # Penalty is heavy (-0.5 per ns) to encourage speed/parallelism.
        # If delay is 0 (parallel move), this is 0.
        time_penalty = -(delay * 0.5) 
        
        # --- COMPONENT B: DISTANCE SHAPING (The Compass) ---
        # Did we move interacting qubits closer together?
        curr_dist = self._get_front_layer_sum_distance()
        dist_improvement = old_dist - curr_dist 
        self.prev_dist = curr_dist # Update baseline for next step
        
        # Bonus is small (+0.1 per unit) to guide, but not overpower the Stick.
        # This prevents the "Flatline" issue by giving constant feedback.
        shaping_reward = (dist_improvement * 0.1)
        
        # --- COMPONENT C: EXISTENCE TAX (The Leash) ---
        # A tiny cost to ensure the agent prefers doing nothing over doing useless things.
        action_tax = -0.01 
        
        # Final Sum
        # Typical Bad Move: -1.5 (Time) + 0.0 (Shape) - 0.01 = -1.51
        # Typical Good Move: -1.5 (Time) + 0.5 (Shape) - 0.01 = -1.01 (Preferred!)
        return time_penalty + shaping_reward + action_tax

    # Helper function required for the Compass
    def _get_front_layer_sum_distance(self):
        total = 0.0
        # Iterate only through the "Front Layer" (gates currently available)
        for node in self.tracker.front_layer:
            qubits = [self.circuit_source.qubits.index(q) for q in node.qargs]
            # Only care about multi-qubit gates (CNOTs)
            if len(qubits) == 2:
                p1, p2 = self.mapping[qubits[0]], self.mapping[qubits[1]]
                if p1 != -1 and p2 != -1:
                    total += self.dist_matrix[p1][p2]
        return total

    def _auto_execute_gates(self):
        count = 0
        while True:
            ran_any = False
            ready_gates = sorted(list(self.tracker.front_layer), key=lambda x: self.tracker.node_to_id[x])
            
            for node in ready_gates:
                self.last_swap_edge = -1 
                qubits = [self.circuit_source.qubits.index(q) for q in node.qargs]
                
                if len(qubits) == 1:
                    p_q = self.mapping[qubits[0]]
                    self.qubit_clock[p_q] += 1.0
                    self.tracker.execute_gate(self.tracker.node_to_id[node])
                    ran_any = True; count += 1

                elif len(qubits) == 2:
                    l_c, l_t = qubits[0], qubits[1]
                    p_c, p_t = self.mapping[l_c], self.mapping[l_t]
                    
                    if self.phys_graph.has_edge(p_c, p_t) and self.phys_graph[p_c][p_t]['type'] == 'intra':
                        duration = 2.0 
                        t = max(self.qubit_clock[p_c], self.qubit_clock[p_t]) + duration
                        self.qubit_clock[p_c] = t
                        self.qubit_clock[p_t] = t
                        self.tracker.execute_gate(self.tracker.node_to_id[node])
                        ran_any = True; count += 1
            
            if not ran_any: break
        return count, self.tracker.is_done()

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
            mask[:self.limit_place] = np.outer(log_free, phys_free).flatten()
            
        else:
            u_occ = self.phys_to_logical[self.intra_u] != -1
            v_occ = self.phys_to_logical[self.intra_v] != -1
            mask[self.limit_place : self.limit_swap] = (u_occ | v_occ)

            if self.last_swap_edge != -1:
                mask[self.limit_place + self.last_swap_edge] = False
                
            # (Simplified Inter-Op mask logic assumes valid for now)
            pass

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