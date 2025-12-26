import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO_Graph_Agent:
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, ent_coef=0.01, grad_clip=0.5):
        """
        Args:
            model: The GraphActorCritic model (GNN + Heads).
            lr: Learning Rate.
            gamma: Discount factor (used in buffer, kept here for config reference).
            clip_ratio: PPO clipping parameter (0.1 to 0.2 is standard).
            value_coef: Weight for value function loss (0.5 is standard).
            ent_coef: Entropy coefficient (Higher = More exploration).
            grad_clip: Max norm for gradient clipping (Prevents GNN explosion).
        """
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.grad_clip = grad_clip
        
        # Number of times to reuse the data batch for training (PPO standard is 3-10)
        self.train_epochs = 4 

    def policy(self, batch_input):
        """
        Forward pass wrapper for inference/sampling.
        Ensures model is in eval mode (disables Dropout/BatchNorm update).
        """
        self.model.eval() 
        return self.model(batch_input)

    def update(self, buffer, logger=None):
        """
        Main PPO Update Loop.
        """
        # 1. Get Batch Data from Buffer
        data = buffer.get()
        
        # Unpack tensors (All on the correct device already if buffer handles it)
        # obs_batch is a PyG Batch object containing many graphs
        obs_batch = data['obs'] 
        act_batch = data['act']
        ret_batch = data['ret']
        adv_batch = data['adv']
        logp_old = data['logp']
        mask_batch = data['mask']

        # 2. Normalize Advantages (CRITICAL STABILITY FIX)
        # This keeps gradients roughly the same scale regardless of reward magnitude (-1000 vs -1)
        adv_mean = adv_batch.mean()
        adv_std = adv_batch.std()
        adv_batch = (adv_batch - adv_mean) / (adv_std + 1e-8)

        # 3. Training Loop (Multi-Epoch)
        self.model.train() # Switch to train mode (Enables Dropout if present)
        
        total_loss_p = 0
        total_loss_v = 0
        total_ent = 0
        
        for _ in range(self.train_epochs):
            # A. Forward Pass (Re-evaluate obs with current policy)
            # We must re-run the model on the graph data to generate the gradient graph.
            flat_logits, values = self.model(obs_batch)
            
            # Mask Logits (Same as inference)
            # Use a large negative number to effectively zero out probability of invalid actions
            masked_logits = flat_logits.masked_fill(~mask_batch, -1e9)
            
            # Calculate new Log Probabilities & Entropy
            dist = torch.distributions.Categorical(logits=masked_logits)
            logp = dist.log_prob(act_batch)
            entropy = dist.entropy().mean()

            # B. Ratio Calculation (Pi_new / Pi_old)
            ratio = torch.exp(logp - logp_old)
            
            # C. Surrogate Loss (The PPO Clipping Trick)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch
            
            # We take the minimum, then negate (because optimizers minimize, but we want to maximize reward)
            policy_loss = -torch.min(surr1, surr2).mean() 
            
            # D. Value Loss (MSE)
            # Predicting the 'Return' (Actual Reward Sum)
            value_loss = 0.5 * ((ret_batch - values.squeeze()) ** 2).mean()
            
            # E. Total Loss
            # Loss = Policy_Loss + c1 * Value_Loss - c2 * Entropy
            # (Subtracting entropy maximizes it, encouraging exploration)
            loss = policy_loss + (self.value_coef * value_loss) - (self.ent_coef * entropy)

            # F. Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # G. Gradient Clipping (CRITICAL FOR GNNs)
            # GNN gradients can sometimes spike massively; this cuts them off at 0.5
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate stats for logging
            total_loss_p += policy_loss.item()
            total_loss_v += value_loss.item()
            total_ent += entropy.item()

        # 4. Logging & Diagnostics
        avg_p_loss = total_loss_p / self.train_epochs
        avg_v_loss = total_loss_v / self.train_epochs
        avg_ent = total_ent / self.train_epochs
        
        # Approx KL Divergence (Did the policy change too much?)
        # Useful for tuning clip_ratio or learning_rate
        with torch.no_grad():
            approx_kl = (logp_old - logp).mean().item()

        if logger:
            logger.log({
                "Train/Policy_Loss": avg_p_loss,
                "Train/Value_Loss": avg_v_loss,
                "Train/Entropy": avg_ent,
                "Train/Approx_KL": approx_kl
            })

        print(f"DEBUG: Adv Mean: {adv_batch.mean():.4f} | Adv Std: {adv_batch.std():.4f}")
        print(f"DEBUG: Batch Size: {len(adv_batch)}")
            
        return avg_p_loss