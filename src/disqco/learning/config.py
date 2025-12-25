model_config = {
    "lr": 1e-4,              # Learning Rate
    "batch_size": 2048,      # Rollout Buffer Size
    "clip_ratio": 0.2,       # PPO Clip
    "ent_coef": 0.01,        # Entropy Coefficient (Exploration) 
    "num_qubits": 6,        # Circuit Size
    "num_qpus": 2,           # Env Topology
    "hex_dist": 3,           # Env Size
    "hidden_dim": 16,        # GNN Size
    "total_steps": 10000000,     # Duration
    "save_dir": "./models",    # Directory to save models
    "max_episode_length": 500,  # Max Steps per Episode
    "max_depth": 8,    # Max Circuit Depth
    "min_depth": 5,   # Min Circuit Depth
    "success_threshold": 0.9,  # Success Rate to consider training complete
    "window_depth": 8,      # Execution Window Depth
    "switch_frequency": 20      # Frequency of switching circuit
}