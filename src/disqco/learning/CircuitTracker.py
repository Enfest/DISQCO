import networkx as nx
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

class CircuitTracker:
    def __init__(self, qiskit_circuit):
        """
        Parses a Qiskit circuit into a dynamic dependency graph.
        """
        # 1. Convert to DAG (Captures all dependencies)
        self.dag = circuit_to_dag(qiskit_circuit)
        
        # 2. Track Execution State
        # 'front_layer' contains gates whose dependencies are met and are ready to run.
        # We use a set for O(1) lookups.
        self.front_layer = set(self.dag.front_layer()) 
        self.executed_gates = set()
        self.total_gates = self.dag.size()
        
        # 3. ID Mapping
        # We assign a stable integer ID to every gate node for the RL agent.
        self.node_to_id = {node: i for i, node in enumerate(self.dag.op_nodes())}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}
        
        # Track logical qubits to create "Anchor Nodes" in the graph
        self.logical_qubits = qiskit_circuit.qubits

    def get_window_graph(self, depth=5):
        """
        Generates the RL Observation Graph for the current moment.
        
        Returns:
            nx.DiGraph: Nodes = Qubits + Gates. Edges = Dependencies + Proximity.
        """
        window_graph = nx.DiGraph()
        
        # --- A. Add Logical Qubit Nodes (Anchors) ---
        # These represent the qubit lines. The GNN uses them to understand "location".
        for q_idx, q in enumerate(self.logical_qubits):
            window_graph.add_node(f"Q_{q_idx}", type='qubit', index=q_idx)

        # --- B. BFS Traversal to find 'Visible' Gates ---
        # Queue stores: (DAGNode, current_depth_from_front)
        queue = [(node, 0) for node in self.front_layer]
        visited = set(node for node, _ in queue)
        
        while queue:
            node, d = queue.pop(0)
            
            if d >= depth:
                continue
            
            # 1. Add Gate Node
            gate_id = self.node_to_id[node]
            node_name = f"G_{gate_id}"
            
            # Feature: 'ready' tells the GNN this gate is executable NOW (depth 0)
            window_graph.add_node(
                node_name, 
                type='gate', 
                op=node.name, 
                ready=(d == 0) 
            )
            
            # 2. Add WEIGHTED Edges (The Sliding Window Logic)
            # Connect this gate to the Logical Qubits it acts on.
            # Weight is high (1.0) if gate is immediate, low if far in future.
            proximity_weight = 1.0 / (d + 1.0)
            
            for q in node.qargs:
                q_idx = self.logical_qubits.index(q)
                qubit_node_name = f"Q_{q_idx}"
                
                # Edge: Logical Qubit <--> Gate
                # Attribute 'type=proximity' helps the GNN treat these differently from dependencies
                window_graph.add_edge(
                    qubit_node_name, 
                    node_name, 
                    type='proximity', 
                    weight=proximity_weight
                )

            # 3. Add Dependency Edges (Gate -> Gate)
            # Find children in the DAG
            for child in self.dag.successors(node):
                if isinstance(child, DAGOpNode):
                    child_id = self.node_to_id[child]
                    child_name = f"G_{child_id}"
                    
                    # Add edge only if child is within window (or will be visited)
                    # We add the edge definition here; the node is added when popped from queue.
                    window_graph.add_edge(node_name, child_name, type='dependency')
                    
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, d + 1))
        
        return window_graph

    def execute_gate(self, gate_id):
        """
        Commits a gate execution.
        1. Marks gate as executed.
        2. Updates 'front_layer' to include newly unlocked children.
        """
        if gate_id not in self.id_to_node:
            raise ValueError(f"Invalid Gate ID: {gate_id}")

        node = self.id_to_node[gate_id]
        
        if node not in self.front_layer:
            # This prevents the agent from executing future gates out of order
            raise ValueError(f"Gate {gate_id} is not in the front layer!")
            
        # 1. Remove from front layer & mark done
        self.front_layer.remove(node)
        self.executed_gates.add(node)
        
        # 2. Check Successors
        for child in self.dag.successors(node):
            if isinstance(child, DAGOpNode):
                # Only add child if ALL its parents are done
                if self._all_parents_executed(child):
                    self.front_layer.add(child)
    
    def _all_parents_executed(self, node):
        """Helper: Checks if all dependencies of a node are satisfied."""
        for parent in self.dag.predecessors(node):
            if isinstance(parent, DAGOpNode) and parent not in self.executed_gates:
                return False
        return True

    def is_done(self):
        """Returns True if all gates have been executed."""
        return len(self.executed_gates) == self.total_gates
    
    def get_partner(self, logical_idx):
        """Returns the logical index of the qubit 'logical_idx' needs to interact with in the front layer."""
        for node in self.front_layer:
            qubits = [q.index for q in node.qargs] # q.index for Qiskit
            if len(qubits) == 2:
                if qubits[0] == logical_idx: return qubits[1]
                if qubits[1] == logical_idx: return qubits[0]
        return -1 # No partner (idle or single-qubit gate)