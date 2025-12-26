import torch
import numpy as np
import os
import wandb
import gymnasium as gym

from disqco.learning.data_utils import graph_collate_fn
from disqco.learning.env import AutoExecDistributedEnv
from disqco.learning.gnn import DistributedQCompilerGNN, obs_to_pyg
from disqco.learning.models import GraphActorCritic
from disqco.learning.buffer import GraphRolloutBuffer
from disqco.learning.ppo_agent import PPO_Graph_Agent
from disqco.learning.config import model_config
from disqco.circuits.cp_fraction import cp_fraction

# ==============================================================================
# 1. THE EXAM FUNCTION (Strict Evaluation)
# ==============================================================================
def run_level_up_exam(agent, env_config, num_qubits, depth, n_episodes=10):
    """
    Runs a strict exam using Greedy (Deterministic) actions on FRESH circuits.
    Returns: Exam Success Rate (0.0 to 1.0)
    """
    print(f"\n📝 TRIGGERED EXAM: Depth {depth} | {n_episodes} Episodes...")
    
    # Create a temporary environment just for the exam
    # (Prevents messing up the training environment's state/history)
    temp_qc = cp_fraction(num_qubits, depth, 0.5, seed=None)
    eval_env = AutoExecDistributedEnv(temp_qc, 
                                      num_qpus=env_config['num_qpus'], 
                                      hex_distance=env_config['hex_dist'])
    
    success_count = 0
    
    for i in range(n_episodes):
        # IMPORTANT: seed=None ensures we test generalization, not memorization
        qc = cp_fraction(num_qubits=num_qubits, depth=depth, fraction=0.5, seed=None)
        eval_env.set_circuit(qc, model_config['window_depth'])
        
        obs, _ = eval_env.reset()
        pyg_data = obs_to_pyg(obs, eval_env)
        done = False
        
        while not done:
            with torch.no_grad():
                # Prepare Batch
                batch_input = graph_collate_fn([pyg_data])
                
                # Get Policy Output
                flat_logits, _ = agent.policy(batch_input)
                
                # Apply Mask
                mask = torch.tensor(eval_env.get_action_mask(), dtype=torch.bool)
                masked_logits = flat_logits.masked_fill(~mask, -1e9)
                
                # GREEDY ACTION (Argmax) - Test the agent's *best* behavior
                action = masked_logits.argmax().item()
            
            # Step
            obs, _, term, trunc, info = eval_env.step(action)
            pyg_data = obs_to_pyg(obs, eval_env)
            done = term or trunc
            
            if info.get('is_success', False):
                success_count += 1
    
    score = success_count / n_episodes
    print(f"📝 EXAM RESULT: {success_count}/{n_episodes} Passed ({score*100:.0f}%)")
    return score

# ==============================================================================
# 2. CURRICULUM TRAINING LOOP
# ==============================================================================
def train_curriculum():
    # 1. SETUP
    run = wandb.init(project="distributed-quantum-compiler", config=model_config)
    config = wandb.config

    # --- CURRICULUM CONFIG ---
    current_depth = config.min_depth
    success_history = []    # Rolling window of TRAINING wins
    TRIGGER_THRESHOLD = 0.85 # Trigger exam if training looks 85% good
    PASS_MARK = 0.8         # Must score 80% on Exam to level up
    
    # 2. INITIALIZE
    qc = cp_fraction(config.num_qubits, current_depth, 0.5, seed=42)
    env = AutoExecDistributedEnv(qc, num_qpus=config.num_qpus, hex_distance=config.hex_dist)
    
    gnn_backbone = DistributedQCompilerGNN(
        num_phys_nodes=env.num_phys,
        num_logical_qubits=env.num_logical,
        action_dim=env.action_space.n,  
        hidden_dim=config.hidden_dim
    )
    
    model = GraphActorCritic(gnn_backbone)
    agent = PPO_Graph_Agent(model, lr=config.lr, clip_ratio=config.clip_ratio, ent_coef=config.ent_coef)
    buffer = GraphRolloutBuffer(buffer_size=config.batch_size)
    
    obs, _ = env.reset()
    pyg_data = obs_to_pyg(obs, env)

    print(f"🚀 Starting Curriculum at Depth {current_depth}")

    current_ep_reward = 0
    current_ep_steps = 0
    
    # 3. MAIN LOOP
    for step in range(config.total_steps): 
        
        # --- EXPLORATION (Stochastic) ---
        with torch.no_grad():
            batch_input = graph_collate_fn([pyg_data]) 
            flat_logits, value = agent.policy(batch_input)
            
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            masked_logits = flat_logits.masked_fill(~mask, -1e9)
            
            # Sample from distribution (Training = Exploration)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # --- STEP ---
        next_obs, reward, term, trunc, info = env.step(action.item())
        buffer.add(pyg_data, action.item(), log_prob.item(), reward, term, value.item(), mask)

        current_ep_reward += reward
        current_ep_steps += 1
        
        # --- EPISODE END ---
        if term or trunc:
            is_success = 1.0 if term else 0.0 # term=True means success
            
            # Log Metrics
            wandb.log({
                "Episode/Reward": current_ep_reward,
                "Episode/Length": env.steps,
                "Episode/Success": is_success,
                "Curriculum/Depth": current_depth,
                "global_step": step
            })
            
            # Track Training History
            success_history.append(is_success)
            if len(success_history) > 40: success_history.pop(0) # Keep last 40 episodes
            
            # === CURRICULUM CHECK LOGIC ===
            # 1. Do we have enough data? (At least 20 episodes)
            if len(success_history) >= 20:
                train_win_rate = sum(success_history) / len(success_history)
                print(f"📊 Training Win Rate over last {len(success_history)} episodes: {train_win_rate*100:.1f}%")
                
                # 2. Does the agent look ready? (Trigger Exam)
                if train_win_rate >= TRIGGER_THRESHOLD:
                    
                    # 3. RUN THE EXAM
                    exam_score = run_level_up_exam(
                        agent, 
                        {"num_qpus": config.num_qpus, "hex_dist": config.hex_dist},
                        config.num_qubits, 
                        current_depth, 
                        n_episodes=10
                    )
                    
                    # 4. DID IT PASS?
                    if exam_score >= PASS_MARK:
                        print(f"🎉 PASSED EXAM! Leveling up to Depth {current_depth + 1}")
                        
                        # Save checkpoint
                        torch.save(model.state_dict(), os.path.join(config.save_dir, f"grad_depth_{current_depth}.pt"))
                        
                        # Level Up
                        current_depth += 1
                        wandb.log({"Curriculum/LevelUp": current_depth, "global_step": step})
                        
                        # Hard Reset History (New level is scary, don't trust old wins)
                        success_history = [] 
                        
                    else:
                        print(f"⚠️ FAILED EXAM ({exam_score:.1f}). Staying at Depth {current_depth}.")
                        # Soft Reset: Remove top 10 oldest records to force it to prove itself again
                        # This prevents an infinite loop of exam-taking if the win rate stays high but exam fails.
                        success_history = success_history[10:]
            
            # === REFRESH ENVIRONMENT ===
            # Use seed=None for randomness in training too
            new_qc = cp_fraction(config.num_qubits, current_depth, 0.5, seed=None)
            env.set_circuit(new_qc, model_config['window_depth'])
            
            obs, _ = env.reset()
            pyg_data = obs_to_pyg(obs, env)
            current_ep_reward = 0
            current_ep_steps = 0
            
        else:
            obs = next_obs
            pyg_data = obs_to_pyg(obs, env)
            
        # --- PPO UPDATE ---
        if buffer.ptr == buffer.buffer_size:
            with torch.no_grad():
                batch_input = graph_collate_fn([pyg_data])
                _, next_val = agent.policy(batch_input)
            buffer.compute_gae(next_val.item())
            loss = agent.update(buffer, logger=run)
            buffer.reset()

        # Periodic refresh just in case an episode gets stuck too long
        if step % model_config["switch_frequency"] == 0:
            env.set_circuit(cp_fraction(config.num_qubits, current_depth, 0.5, seed=None), model_config['window_depth'])

    # FINISH
    torch.save(model.state_dict(), os.path.join(config.save_dir, "final_model.pt"))
    wandb.finish()

if __name__ == "__main__":
    train_curriculum()