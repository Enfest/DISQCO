"""
Circuits module - creation of circuits used for partitioning in DisQCO.

This module provides the functions to creae some instances of circuits that are not already in the Qiskit library.
"""

from disqco.circuits.cp_fraction import cp_fraction, cz_fraction
from disqco.circuits.IQP import build_IQP
from disqco.circuits.QAOA import QAOA_random
from disqco.circuits.square import build_square_circuit

__all__ = ['cp_fraction', 'cz_fraction', 'build_IQP', 'QAOA_random', 'build_square_circuit']
