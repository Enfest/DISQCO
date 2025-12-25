import torch
from qiskit import QuantumCircuit
from disqco.learning.data_utils import graph_collate_fn, save_env_snapshot, save_full_state
from disqco.learning.env import AutoExecDistributedEnv
from disqco.learning.gnn import DistributedQCompilerGNN, obs_to_pyg
from disqco.learning.models import GraphActorCritic
from disqco.learning.buffer import GraphRolloutBuffer
from disqco.learning.ppo_agent import PPO_Graph_Agent
import wandb
import os  
from disqco.learning.config import model_config
from disqco.circuits.cp_fraction import cp_fraction
import gymnasium as gym

def generate_task(num_qubits, depth):
    """Generates a random circuit of a specific difficulty."""
    # max_operands=2 ensures we get CNOTs, which are the hard part
    return cp_fraction(
        num_qubits=num_qubits,
        depth=depth,
        fraction=0.5,
        seed=42
    )

def evaluate_agent(agent, env, num_qubits, depth, n_episodes=10):
    """Runs inference to check success rate on fresh circuits."""
    wins = 0
    for _ in range(n_episodes):
        # Generate a test circuit for this specific depth
        qc = generate_task(num_qubits, depth)
        env.set_circuit(qc, model_config['window_depth']) 
        
        obs, _ = env.reset()
        pyg_data = obs_to_pyg(obs, env)
        done = False
        
        while not done:
            with torch.no_grad():
                # Deterministic Action (Greedy) for Evaluation
                batch_input = graph_collate_fn([pyg_data])
                flat_logits, _ = agent.policy(batch_input)
                mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
                masked_logits = flat_logits.masked_fill(~mask, -1e9)
                action = masked_logits.argmax().item()
            
            obs, _, term, trunc, info = env.step(action)
            pyg_data = obs_to_pyg(obs, env)
            done = term or trunc
            
            if info.get('is_success', False):
                wins += 1
                
    return wins / n_episodes

def train_curriculum():

    # 1. INITIALIZE WANDB
    run = wandb.init(
        project="distributed-quantum-compiler",
        config=model_config
    )
    config = wandb.config

    # --- CURRICULUM VARIABLES ---
    # Start depth same as your original code
    current_depth = config.min_depth
    success_history = []  # Rolling window to track win rate
    SUCCESS_THRESHOLD = 0.9 # Level up when 90% of last 20 episodes are wins
    
    # 2. Setup Initial Environment
    # We generate the first circuit
    qc = cp_fraction(num_qubits=config.num_qubits,
                     depth=current_depth,
                     fraction=0.5,
                     seed=42)

    print(f"Training Initial Circuit (Depth {current_depth}):")
    
    env = AutoExecDistributedEnv(qc, num_qpus=config.num_qpus, hex_distance=config.hex_dist)
    
    # Initialize GNN 
    gnn_backbone = DistributedQCompilerGNN(
        num_phys_nodes=env.num_phys,
        num_logical_qubits=env.num_logical,
        action_dim=env.action_space.n,  
        hidden_dim=config.hidden_dim
    )
    
    model = GraphActorCritic(gnn_backbone)
    agent = PPO_Graph_Agent(
                model,
                lr=config.lr,
                clip_ratio=config.clip_ratio,
                ent_coef=config.ent_coef
            )
    buffer = GraphRolloutBuffer(buffer_size=config.batch_size)
    
    # Training Init
    obs, _ = env.reset()
    pyg_data = obs_to_pyg(obs, env)

    print("Starting Curriculum Training...")

    current_ep_reward = 0
    current_ep_steps = 0
    best_avg_reward = -float('inf') 
    recent_rewards = [] 
    level_improvement_steps = 0
    
    for step in range(config.total_steps): 
        
        # --- COLLECT DATA ---
        with torch.no_grad():
            batch_input = graph_collate_fn([pyg_data]) 
            flat_logits, value = agent.policy(batch_input)
            
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            masked_logits = flat_logits.masked_fill(~mask, -1e9)
            
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # --- EXECUTE ---
        next_obs, reward, term, trunc, info = env.step(action.item())
        
        # --- STORE ---
        buffer.add(pyg_data, action.item(), log_prob.item(), reward, term, value.item(), mask)

        current_ep_reward += reward
        current_ep_steps += 1
        
        # --- RESET/UPDATE ---
        if term or trunc:
            # 1. Standard Logging
            is_success = 1.0 if not trunc else 0.0 # term=True means success, trunc=True means timeout
            
            wandb.log({
                "Episode/Reward": current_ep_reward,
                "Episode/Length": env.steps,
                "Episode/Success": 0.0 if trunc else 1.0,
                "Curriculum/Depth": current_depth, # Track difficulty
                "global_step": step
            })

            # 2. Track Best Model (Legacy Logic)
            recent_rewards.append(current_ep_reward)
            if len(recent_rewards) > 10: recent_rewards.pop(0)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            if avg_reward > best_avg_reward and len(recent_rewards) >= 10:
                best_avg_reward = avg_reward
                if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)
                save_path = os.path.join(config.save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"New Record ({best_avg_reward:.2f})! Saved best_model.pt")

            # ==========================================================
            # 3. CURRICULUM LOGIC (The New Part)
            # ==========================================================
            success_history.append(is_success)
            if len(success_history) > 20: success_history.pop(0)
            
            # Check if we are ready to Level Up
            current_win_rate = sum(success_history) / len(success_history) if len(success_history) > 0 else 0
            
            if len(success_history) >= 10 and current_win_rate >= SUCCESS_THRESHOLD and avg_reward > -(current_depth*2*config.num_qubits):

                # LEVEL UP!
                print(f"🎉 Level Up! Win Rate {current_win_rate:.2f} >= {SUCCESS_THRESHOLD}")
                print(f"   Increasing Depth: {current_depth} -> {current_depth + 1}")
                
                current_depth += 1 # Increase difficulty (adjust step size as needed)
                success_history = [] # Reset history for new level
                best_avg_reward = -float('inf') # Reset best reward for new level
                recent_rewards = []
                level_improvement_steps = 0
                
                # Save checkpoint for this level
                level_path = os.path.join(config.save_dir, f"model_depth_{current_depth}.pt")
                torch.save(model.state_dict(), level_path)
            # elif  len(success_history) >= 10 and current_win_rate >= SUCCESS_THRESHOLD :
            #     level_improvement_steps += 1
            else:
                print(f"Current Depth {current_depth} | Win Rate: {current_win_rate:.2f}")

            # 4. REFRESH CIRCUIT
            # Always generate a fresh circuit for the next episode.
            # This prevents overfitting to a single file, which is crucial for generalization.
            new_qc = cp_fraction(num_qubits=config.num_qubits,
                                 depth=current_depth,
                                 fraction=0.5,
                                 seed=None) # Seed=None ensures randomness!
            
            # Inject new circuit into environment
            env.set_circuit(new_qc, model_config['window_depth'])
            # ==========================================================

            # Reset Metrics
            current_ep_reward = 0
            current_ep_steps = 0

            # Env was implicitly reset by set_circuit, but calling reset() again is safe
            obs, _ = env.reset()
            pyg_data = obs_to_pyg(obs, env)
        else:
            obs = next_obs
            pyg_data = obs_to_pyg(obs, env)
            
        # --- PPO UPDATE TRIGGER ---
        if buffer.ptr == buffer.buffer_size:
            print(f"Update at step {step} | Depth: {current_depth}")
            with torch.no_grad():
                batch_input = graph_collate_fn([pyg_data])
                _, next_val = agent.policy(batch_input)
                
            buffer.compute_gae(next_val.item())
            loss = agent.update(buffer, logger=run)
            buffer.reset()
            # print(f"Loss: {loss:.4f}")

        if step % model_config["switch_frequency"] == 0:
            new_qc = cp_fraction(num_qubits=config.num_qubits,
                                 depth=current_depth,
                                 fraction=0.5,
                                 seed=None) # Seed=None ensures randomness!
            
            # Inject new circuit into environment
            env.set_circuit(new_qc, model_config['window_depth'])

    # 4. FINAL SAVE
    final_path = os.path.join(config.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    print("Training Complete. Final model saved.")
    wandb.finish()

def train():

    # 1. INITIALIZE WANDB
    # We define the "Default" hyperparameters here.
    run = wandb.init(
        project="distributed-quantum-compiler",
        config=model_config
    )
    
    # IMPORTANT: Access the config object!
    # This allows WandB Sweeps to override these values automatically later.
    config = wandb.config

    # 1. Setup Environment & Agent
    qc = cp_fraction(  num_qubits=config.num_qubits,
                        depth=config.num_qubits,
                        fraction= 0.5,
                        seed=42)

    # print the circuit
    print("Training Circuit:")
    print(qc)
    
    env = AutoExecDistributedEnv(qc, num_qpus=config.num_qpus, hex_distance=config.hex_dist)
    
    # Initialize GNN (Dimensions must match your feature lists)
    gnn_backbone = DistributedQCompilerGNN(
        num_phys_nodes=env.num_phys,
        num_logical_qubits=env.num_logical,
        action_dim=env.action_space.n,  # <--- PASS THE CORRECT SIZE HERE (33)
        hidden_dim=config.hidden_dim    # <--- From Config
    )
    
    model = GraphActorCritic(gnn_backbone)
    agent = PPO_Graph_Agent(
                model,
                lr=config.lr,
                clip_ratio=config.clip_ratio,
                ent_coef=config.ent_coef
            )
    buffer = GraphRolloutBuffer(buffer_size=config.batch_size)
    # 2. Training Init
    obs, _ = env.reset()
    pyg_data = obs_to_pyg(obs, env)

    # save initial environment snapshot
    # save_full_state(env, "initial_env_snapshot")
    
    print("Starting Training...")

    current_ep_reward = 0
    current_ep_steps = 0
    best_avg_reward = -float('inf') # Track best performance
    recent_rewards = [] # For calculating moving average
    
    for step in range(config.total_steps): # Total Steps
        
        # --- COLLECT DATA ---
        with torch.no_grad():
            # Create a "fake" batch of 1 to run through network
            batch_input = graph_collate_fn([pyg_data]) 
            
            # Model returns flat logits for the whole batch
            flat_logits, value = agent.policy(batch_input)
            
            # Masking for Sampling
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            masked_logits = flat_logits.masked_fill(~mask, -1e9)
            
            # Sample Action
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # --- EXECUTE ---
        next_obs, reward, term, trunc, info = env.step(action.item())
        # save_full_state(env, f"step_{step}_env_snapshot")
        
        # --- STORE ---
        # Note: We store 'value.item()' to strip tensor wrapper
        buffer.add(pyg_data, action.item(), log_prob.item(), reward, term, value.item(), mask)

        current_ep_reward += reward
        current_ep_steps += 1
        
        # --- RESET/UPDATE ---
        if term or trunc:
            wandb.log({
                "Episode/Reward": current_ep_reward,
                "Episode/Length": current_ep_steps,
                "Episode/Success": 1.0 if term else 0.0, # 1 if finished, 0 if timed out
                "global_step": step
            })

            # Track Best Model
            recent_rewards.append(current_ep_reward)
            if len(recent_rewards) > 50: recent_rewards.pop(0)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            # avg_reward = current_ep_reward  # Use current episode reward for record
            
            if avg_reward > best_avg_reward and len(recent_rewards) >= 10:
                best_avg_reward = avg_reward
                # 4. SAVE BEST MODEL
                # check if save_dir exists
                if not os.path.exists(config.save_dir):
                    os.makedirs(config.save_dir)
                save_path = os.path.join(config.save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"New Record ({best_avg_reward:.2f})! Saved best_model.pt")

            # Reset Metrics
            current_ep_reward = 0
            current_ep_steps = 0

            obs, _ = env.reset()
            pyg_data = obs_to_pyg(obs, env)
        else:
            obs = next_obs
            pyg_data = obs_to_pyg(obs, env)
            
        # --- PPO UPDATE TRIGGER ---
        if buffer.ptr == buffer.buffer_size:
            print(f"Update at step {step}")
            # Calculate GAE
            with torch.no_grad():
                batch_input = graph_collate_fn([pyg_data])
                _, next_val = agent.policy(batch_input)
                
            buffer.compute_gae(next_val.item())
            loss = agent.update(buffer, logger=run)
            buffer.reset()
            print(f"Loss: {loss:.4f}")

        if step % model_config["switch_frequency"] == 0:
            new_qc = cp_fraction(num_qubits=config.num_qubits,
                                 depth=config.num_qubits,
                                 fraction=0.5,
                                 seed=None) # Seed=None ensures randomness!
            
            # Inject new circuit into environment
            env.set_circuit(new_qc, model_config['window_depth'])

    # 4. FINAL SAVE
    final_path = os.path.join(config.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path) # Upload to cloud
    print("Training Complete. Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    train_curriculum()
    # train()