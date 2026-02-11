from disqco.circuits.cp_fraction import cp_fraction
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
from qiskit import transpile
from disqco.parti.FM.FM_methods import set_initial_partition_assignment, calculate_full_cost
from qiskit.circuit.library import QFT, QuantumVolume, EfficientSU2
from disqco.circuits.QAOA import QAOA_random
import numpy as np
import time
from disqco.graphs.quantum_network import QuantumNetwork

num_qubits = 16
num_partitions = 4
qpu_size = int(num_qubits / num_partitions) + 1
qpu_sizes = [qpu_size] * num_partitions

# qpu_sizes[-1] += 1

network = QuantumNetwork(qpu_sizes)
print("Network: ")
print(network)


circuit = cp_fraction(  num_qubits=num_qubits,
                        depth=2*num_qubits,
                        fraction= 0.5,
                        seed=42)

# circuit = QuantumVolume(num_qubits, depth=num_qubits)

circuit = transpile(circuit, basis_gates = ['cp', 'u'])
depth = circuit.depth()

print("Circuit: ")
print(circuit)

graph = QuantumCircuitHyperGraph(circuit)

print("Graph: ")
print(graph)

assignment = set_initial_partition_assignment(graph, network)

initial_cost = calculate_full_cost(graph, assignment, num_partitions)
