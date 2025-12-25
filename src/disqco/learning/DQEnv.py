import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from disqco.learning.CircuitTracker import CircuitTracker
from disqco.learning.device_topology import generate_distributed_heavy_hex

# Assumptions: You have imported or defined the helper classes from previous steps
# from circuit_tracker import CircuitTracker
# from topology_gen import generate_distributed_heavy_hex

class DistributedHeavyHexEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, qiskit_circuit, num_qpus=2, hex_distance=3, window_depth=5):
        super().__init__()
        
        # --- 1. Infrastructure Setup ---
        self.num_qpus = num_qpus
        self.window_depth = window_depth
        
        # Generate Hardware Graph (Nodes have 'type': data/comm, Edges have 'type': intra/inter)
        self.phys_graph, self.qubits_per_qpu = generate_distributed_heavy_hex(num_qpus, hex_distance)
        self.num_phys_qubits = self.phys_graph.number_of_nodes()
        self.phys_edges = list(self.phys_graph.edges(data=True)) # Store for SWAP validation
        
        # Circuit Tracker (Manages dependencies)
        self.circuit_source = qiskit_circuit
        self.tracker = None # Initialized in reset()
        self.num_logical = len(qiskit_circuit.qubits)

        # --- 2. Action Space Definition (Flattened Discrete) ---
        # We calculate the offsets for our "Mega-Action" integer
        self.limit_place = self.num_logical * self.num_phys_qubits
        self.limit_swap = self.limit_place + len(self.phys_edges)
        self.limit_teleport = self.limit_swap + (self.num_logical * self.num_qpus)
        self.limit_execute = self.limit_teleport + 10 # Window size for gate selection
        
        self.action_space = spaces.Discrete(self.limit_execute)

        # --- 3. Observation Space (Dict for GNN) ---
        # We return raw data; the Agent is responsible for turning this into PyG Data objects
        # This acts as a placeholder to satisfy Gym API checks
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=0, high=1, shape=(self.num_logical + 50, 5), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=1000, shape=(2, 200), dtype=np.int64),
            "edge_weights": spaces.Box(low=0, high=1, shape=(200,), dtype=np.float32),
            "mapping": spaces.Box(low=-1, high=self.num_phys_qubits, shape=(self.num_logical,), dtype=np.int64),
            "phase": spaces.Discrete(2) # 0=Placement, 1=Execution
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize fresh tracker
        self.tracker = CircuitTracker(self.circuit_source)
        
        # Reset State
        self.mapping = np.full(self.num_logical, -1, dtype=int) # -1 = Unplaced
        self.placed_qubits_count = 0
        self.current_phase = 0 # PLACEMENT PHASE
        self.steps_taken = 0
        
        # Clear hardware state (optional: track occupied slots if needed)
        self.phys_occupancy = np.zeros(self.num_phys_qubits, dtype=int)

        return self._get_observation(), {}

    def step(self, action):
        self.steps_taken += 1
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- PHASE 1: PLACEMENT ---
        if self.current_phase == 0:
            reward = self._step_placement(action)
            
            # Check Phase Transition
            if self.placed_qubits_count == self.num_logical:
                self.current_phase = 1 # Transition to EXECUTION
                # Optional: Give small bonus for completing placement validly
                
        # --- PHASE 2: EXECUTION ---
        else:
            reward, terminated = self._step_execution(action)

        # Check Truncation (Time limit)
        if self.steps_taken > 1000: # Safety break
            truncated = True

        return self._get_observation(), reward, terminated, truncated, info

    # =========================================================================
    #                               LOGIC HANDLERS
    # =========================================================================

    def _step_placement(self, action):
        """Validates and applies initial qubit placement."""
        # 1. Decode
        if action >= self.limit_place:
            return -10.0 # Invalid: Tried to Execute/Swap during placement
            
        logical_q = action // self.num_phys_qubits
        phys_q = action % self.num_phys_qubits
        
        # 2. Validate
        if self.mapping[logical_q] != -1:
            return -10.0 # Penalty: Qubit already placed
        
        if self.phys_occupancy[phys_q] >= 1:
            return -10.0 # Penalty: Physical slot taken
            
        # 3. Apply
        self.mapping[logical_q] = phys_q
        self.phys_occupancy[phys_q] = 1
        self.placed_qubits_count += 1
        
        return 0.0 # No immediate reward (Delayed gratification)

    def _step_execution(self, action):
        """Handles SWAP, Teleport, and Gate Execution."""
        # 1. Reject Placement Actions
        if action < self.limit_place:
            return -5.0, False # Penalty: Tried to place during execution

        # 2. Decode Action
        if action < self.limit_swap:
            return self._handle_swap(action)
        elif action < self.limit_teleport:
            return self._handle_teleport(action)
        elif action < self.limit_execute:
            return self._handle_gate_execution(action)
        
        return -1.0, False # Fallback

    def _handle_swap(self, action):
        idx = action - self.limit_place
        u, v, data = self.phys_edges[idx] # Get edge data
        
        # COST FUNCTION: Heavy Hex Constraints
        if data['type'] == 'inter':
            return -100.0, False # ILLEGAL: Cannot SWAP across optical link (must teleport)
        
        # Logic: Swap logical qubits at u and v
        l_u = self._get_logical_at(u)
        l_v = self._get_logical_at(v)
        
        # Apply Mapping Update
        if l_u is not None: self.mapping[l_u] = v
        if l_v is not None: self.mapping[l_v] = u
        
        # Standard SWAP Cost (Depth + Noise)
        return -3.0, False 

    def _handle_gate_execution(self, action):
        idx = action - self.limit_teleport
        
        # Get ready gates from tracker
        ready_gates = list(self.tracker.front_layer)
        # Sort by ID to ensure deterministic action mapping
        ready_gates.sort(key=lambda n: self.tracker.node_to_id[n])
        
        if idx >= len(ready_gates):
            return -1.0, False # No-Op (Index out of bounds)
            
        gate_node = ready_gates[idx]
        qubits = [q.index for q in gate_node.qargs]
        
        # 1-Qubit Gate: Always Free
        if len(qubits) == 1:
            self.tracker.execute_gate(self.tracker.node_to_id[gate_node])
            return 0.1, self.tracker.is_done()
            
        # 2-Qubit Gate: Check Connectivity
        phys_q1 = self.mapping[qubits[0]]
        phys_q2 = self.mapping[qubits[1]]
        
        if self.phys_graph.has_edge(phys_q1, phys_q2):
            # Success!
            self.tracker.execute_gate(self.tracker.node_to_id[gate_node])
            return 5.0, self.tracker.is_done() # High reward for clearing a gate
        else:
            # Failure: Qubits not adjacent
            return -2.0, False # Penalty for trying to execute too early

    def _handle_teleport(self, action):
        # Implementation of "Teleport State"
        # Moves logical qubit to target QPU if capacity allows
        # Costs 1 e-bit (-10 reward approx)
        # (Simplified for brevity)
        return -10.0, False

    # =========================================================================
    #                               OBSERVATION
    # =========================================================================

    def _get_observation(self):
        """
        Converts internal NetworkX graph + Mapping into Numeric Tensors.
        """
        # 1. Get Graph from Tracker (Dynamic Weights!)
        # We look ahead 5 layers
        window_graph = self.tracker.get_window_graph(depth=self.window_depth)
        
        # 2. Extract Features for GNN
        # Nodes: [is_qubit, is_gate, is_ready, current_qpu_id, -1]
        # Edges: [source, target, weight]
        
        # (This section usually requires conversion logic specific to PyTorch Geometric)
        # For now, we return the raw graph wrapper, or you can implement the matrix conversion here.
        
        return {
            "graph": window_graph, # Pass the NX object directly if using a custom collator
            "mapping": self.mapping.copy(),
            "phase": self.current_phase
        }

    def _get_logical_at(self, phys_node):
        """Reverse lookup: Find logical qubit at physical node."""
        results = np.where(self.mapping == phys_node)[0]
        return results[0] if len(results) > 0 else None