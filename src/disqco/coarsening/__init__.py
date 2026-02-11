"""
Coarsening Module - Hypergraph and Network Coarsening

This module provides coarsening algorithms for quantum circuit hypergraphs
and quantum network topologies.
"""

from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.graphs.coarsening.network_coarsener import NetworkCoarsener

__all__ = ['HypergraphCoarsener', 'NetworkCoarsener']
