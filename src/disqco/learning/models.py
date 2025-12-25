import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool

class GraphActorCritic(nn.Module):
    def __init__(self, gnn_model):
        super().__init__()
        self.gnn = gnn_model
        
    def forward(self, data):
        # The GNN already returns (logits, value)
        # We just pass it through directly.
        return self.gnn(data)