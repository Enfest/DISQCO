import torch
import numpy as np
from disqco.learning.data_utils import graph_collate_fn

class GraphRolloutBuffer:
    def __init__(self, buffer_size, gamma=0.99, lam=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        """Clears the buffer."""
        self.obs_buf = [] # List to store PyG HeteroData objects
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.term_buf = [] # Terminated (Success/Fail)
        self.val_buf = []  # Value estimates
        self.mask_buf = []
        self.ptr = 0

    def add(self, obs, act, logp, rew, term, val, mask):
        """
        Store one step of data.
        """
        if self.ptr < self.buffer_size:
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.logp_buf.append(logp)
            self.rew_buf.append(rew)
            self.term_buf.append(term)
            self.val_buf.append(val)
            self.mask_buf.append(mask)
            self.ptr += 1

    def get(self):
        """
        Prepare the data for PPO update.
        This is where GNN Batching happens.
        """
        assert self.ptr == self.buffer_size, "Buffer not full yet!"
        
        # 1. Batch the Graph Data (Crucial for GNNs)
        # We use your existing collate function to merge list of graphs into one Batch
        obs_batch = graph_collate_fn(self.obs_buf)
        
        # 2. Convert lists to tensors
        act_batch = torch.tensor(self.act_buf, dtype=torch.long)
        logp_batch = torch.tensor(self.logp_buf, dtype=torch.float32)
        val_batch = torch.tensor(self.val_buf, dtype=torch.float32)
        mask_batch = torch.stack(self.mask_buf) # Stack boolean masks

        # 3. Advantage Normalization (The "Magic" of PPO)
        # We assume the "Advantages" were already computed by compute_gae()
        # and stored in self.adv_buf and self.ret_buf
        adv_batch = torch.tensor(self.adv_buf, dtype=torch.float32)
        ret_batch = torch.tensor(self.ret_buf, dtype=torch.float32)

        # Return a dictionary for clean unpacking
        return {
            'obs': obs_batch,
            'act': act_batch,
            'ret': ret_batch,
            'adv': adv_batch,
            'logp': logp_batch,
            'mask': mask_batch
        }

    def compute_gae(self, last_val):
        """
        Generalized Advantage Estimation (GAE-Lambda).
        This smooths the reward signal to reduce variance.
        """
        rews = np.array(self.rew_buf)
        vals = np.array(self.val_buf + [last_val]) # Append value of Next State
        terms = np.array(self.term_buf)
        
        # GAE Calculation
        deltas = rews + self.gamma * vals[1:] * (1 - terms) - vals[:-1]
        
        self.adv_buf = np.zeros_like(rews, dtype=np.float32)
        last_gae_lam = 0
        
        # Walk backwards
        for t in reversed(range(len(rews))):
            last_gae_lam = deltas[t] + self.gamma * self.lam * (1 - terms[t]) * last_gae_lam
            self.adv_buf[t] = last_gae_lam
            
        # Returns = Advantage + Value (The Target for the Critic)
        self.ret_buf = self.adv_buf + vals[:-1]