import torch
import numpy as np
from torch_geometric.data import Batch

class GraphRolloutBuffer:
    def __init__(self, buffer_size, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.graphs = []        # List of Graph Data Objects
        self.actions = []       # List of scalars
        self.log_probs = []     # List of scalars
        self.rewards = []       # List of scalars
        self.dones = []         # List of booleans
        self.values = []        # List of scalars
        self.masks = []         # List of 1D Bool Tensors
        self.ptr = 0

    def add(self, graph, action, log_prob, reward, done, value, mask):
        if self.ptr < self.buffer_size:
            # Append references (Python handles object overhead efficiently)
            self.graphs.append(graph) 
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.masks.append(mask)
            self.ptr += 1

    def compute_gae(self, last_value):
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        values = torch.tensor(self.values + [last_value], dtype=torch.float)
        dones = torch.tensor(self.dones, dtype=torch.float)
        
        self.advantages = torch.zeros_like(rewards)
        self.returns = torch.zeros_like(rewards)
        
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            
        self.returns = self.advantages + values[:-1]

    def get_batches(self, batch_size):
        indices = np.arange(len(self.graphs))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.graphs), batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            
            # 1. Collate Graphs
            batch_graphs = [self.graphs[i] for i in batch_idx]
            batched_data = Batch.from_data_list(batch_graphs)
            
            # 2. Stack Simple Tensors
            b_actions = torch.tensor([self.actions[i] for i in batch_idx], dtype=torch.long)
            b_log_probs = torch.tensor([self.log_probs[i] for i in batch_idx], dtype=torch.float)
            b_returns = self.returns[batch_idx]
            b_adv = self.advantages[batch_idx]
            b_vals = torch.tensor([self.values[i] for i in batch_idx], dtype=torch.float)
            
            # 3. Ragged Masks (Keep as list)
            b_masks = [self.masks[i] for i in batch_idx]
            
            yield batched_data, b_actions, b_log_probs, b_returns, b_adv, b_vals, b_masks