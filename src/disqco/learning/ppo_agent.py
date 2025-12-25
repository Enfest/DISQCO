import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

class PPO_Graph_Agent:
    def __init__(self, actor_critic, lr=3e-4, clip_ratio=0.2, ent_coef=0.01):
        self.policy = actor_critic
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()

    def update(self, buffer, batch_size=32, epochs=4, logger=None):
        # Normalize Advantages
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)
        
        total_loss_log = 0

        # Track averages for this update
        avg_policy_loss = 0
        avg_val_loss = 0
        avg_entropy = 0
        updates_count = 0
        
        for _ in range(epochs):
            generator = buffer.get_batches(batch_size)
            
            for g_batch, actions, old_log_probs, returns, advs, old_vals, masks in generator:
                
                # 1. Forward Pass (Get Flat Logits)
                flat_logits, current_values = self.policy(g_batch)
                
                # 2. Ragged Batching: Split & Pad
                # Split flat logits back into per-graph slices using mask sizes
                sizes = [m.size(0) for m in masks]
                # NEW (Correct for Pooled Graphs)
                # We split by 1 because there is exactly 1 prediction vector per graph in the batch
                logits_list = torch.split(flat_logits, 1)
                
                # Determine max action space in this specific batch
                max_len = max(sizes)
                batch_size_curr = len(masks)
                
                # Create Padded Tensor filled with -inf
                padded_logits = torch.full((batch_size_curr, max_len), -1e9)
                
                for i, logit in enumerate(logits_list):
                    # Apply Mask to the valid portion
                    masked_logit = logit.masked_fill(~masks[i], -1e9)
                    # Copy to padded container
                    padded_logits[i, :sizes[i]] = masked_logit
                
                # 3. Probability Calculation
                dist = Categorical(logits=padded_logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # 4. Losses
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.mse_loss(current_values.squeeze(), returns)
                
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                
                # 5. Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss_log += loss.item()

                # ACCUMULATE STATS
                avg_policy_loss += policy_loss.item()
                avg_val_loss += value_loss.item()
                avg_entropy += entropy.item()
                updates_count += 1
        
        # LOG AVERAGES ONCE PER UPDATE CYCLE
        if logger:
            logger.log({
                "Train/Policy_Loss": avg_policy_loss / updates_count,
                "Train/Value_Loss": avg_val_loss / updates_count,
                "Train/Entropy": avg_entropy / updates_count,
                "Train/Approx_KL": (old_log_probs - new_log_probs).mean().item() # Optional: Watch for divergence
            })
                
        return total_loss_log / epochs