"""
Test suite for network coarsening
"""

import pytest
from disqco import QuantumNetwork
from disqco.coarsening import NetworkCoarsener


@pytest.fixture
def test_network():
    """Create an all-to-all network with 4 QPUs"""
    qpu_sizes = {0: 4, 1: 4, 2: 4, 3: 4}
    network = QuantumNetwork(qpu_sizes)
    return network


def test_network_coarsener_import():
    """Test that NetworkCoarsener can be imported from disqco.coarsening"""
    from disqco.coarsening import NetworkCoarsener
    assert NetworkCoarsener is not None


def test_network_coarsener_instantiation(test_network):
    """Test creating an instance of NetworkCoarsener"""
    coarsener = NetworkCoarsener(test_network)
    assert coarsener is not None
    assert coarsener.initial_network == test_network


def test_network_coarsen(test_network):
    """Test network coarsening"""
    coarsener = NetworkCoarsener(test_network)
    
    # Coarsen the network to desired size (returns tuple: network, mapping)
    coarsened_network, mapping = coarsener.coarsen_network(
        network=test_network,
        desired_size=2
    )
    
    assert coarsened_network is not None
    assert mapping is not None
    # After coarsening to size 2, we should have at most 2 QPUs
    assert len(coarsened_network.qpu_sizes) <= 2
    print(f"\nNetwork coarsening: {len(test_network.qpu_sizes)} QPUs -> {len(coarsened_network.qpu_sizes)} QPUs")


def test_network_coarsener_in_star_import():
    """Test that NetworkCoarsener is in __all__ for star imports"""
    from disqco import coarsening
    assert 'NetworkCoarsener' in coarsening.__all__
