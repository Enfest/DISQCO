import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import networkx as nx

# --- PROJECT IMPORTS ---
from disqco.circuits.cp_fraction import cp_fraction
from disqco.learning.config import model_config
from disqco.learning.env import AutoExecDistributedEnv
from disqco.learning.gnn import DistributedQCompilerGNN, obs_to_pyg
from disqco.learning.models import GraphActorCritic
from disqco.learning.data_utils import graph_collate_fn

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================
def decode_action(action, env):
    """Decodes integer action into human-readable string."""
    if action < env.limit_place:
        l_q = action // env.num_phys
        p_q = action % env.num_phys
        return f"🟦 PLACE Q_L{l_q} -> Node P_{p_q}"
    elif action < env.limit_swap:
        idx = action - env.limit_place
        u, v = env.intra_edges[idx]
        return f"🔄 SWAP {u}<->{v}"
    else:
        offset = action - env.limit_swap
        link_idx = offset // 3
        op = offset % 3
        u, v = env.inter_edges[link_idx]
        ops = ["Entangle", "Move(V->U)", "Move(U->V)"]
        return f"✨ {ops[op]} Link({u}-{v})"

# ==============================================================================
# 2. INFERENCE ENGINE (STOCHASTIC)
# ==============================================================================
def run_inference_stochastic(circuit, model_path, num_shots=10, max_steps=1000):
    """
    Runs inference multiple times (Monte Carlo) to escape local optima.
    Returns: (Success, Best History Log, Best Action Sequence)
    """
    print("\n" + "="*40)
    print("       TARGET QUANTUM CIRCUIT")
    print("="*40)
    print(circuit)
    print("="*40 + "\n")

    print(f"Initializing Env for {len(circuit.qubits)} Qubits...")
    
    # Validation
    if len(circuit.qubits) != model_config['num_qubits']:
        print(f"⛔ ERROR: Model expects {model_config['num_qubits']} qubits, but circuit has {len(circuit.qubits)}.")
        return False, [], []

    # Setup Base Environment
    env_base = AutoExecDistributedEnv(
        circuit, 
        num_qpus=model_config['num_qpus'], 
        hex_distance=model_config['hex_dist']
    )
    
    # Load Model
    try:
        gnn = DistributedQCompilerGNN(
            num_phys_nodes=env_base.num_phys,
            num_logical_qubits=env_base.num_logical,
            action_dim=env_base.action_space.n,
            hidden_dim=model_config['hidden_dim']
        )
        model = GraphActorCritic(gnn)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    except Exception as e:
        print(f"⛔ Model Load Error: {e}")
        return False, [], []

    # Variables to track best solution
    best_len = float('inf')
    best_history = []
    best_actions = []
    success_count = 0
    
    print(f"🚀 Running {num_shots} Stochastic Shots to find MINIMUM solution...\n")

    for shot in range(num_shots):
        # Fresh Env for every shot
        env = copy.deepcopy(env_base)
        obs, _ = env.reset()
        pyg_data = obs_to_pyg(obs, env)
        
        history = []
        actions = []
        shot_success = False
        
        for step in range(max_steps):
            with torch.no_grad():
                batch = graph_collate_fn([pyg_data])
                flat_logits, _ = model(batch)
                
                # Apply Mask
                mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
                masked_logits = flat_logits.masked_fill(~mask, -1e9)
                
                # SAMPLING (Stochastic Policy) to break loops
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample().item()
            
            # Execute
            next_obs, reward, term, trunc, info = env.step(action)
            
            # Record
            actions.append(action)
            phase = "PLACE" if env.current_phase == 0 else "ROUTE"
            action_str = decode_action(action, env)
            history.append(f"{step:03d} [{phase}] {action_str}")
            
            if term:
                shot_success = True
                break
            if trunc:
                break
                
            pyg_data = obs_to_pyg(next_obs, env)
            
        # Shot Result
        status = "✅ Success" if shot_success else "❌ Failed"
        len_str = f"{len(history)} steps" if shot_success else "-"
        print(f"   Shot {shot+1:02d}: {status} | Length: {len_str}")
        
        if shot_success:
            success_count += 1
            if len(history) < best_len:
                best_len = len(history)
                best_history = list(history)
                best_actions = list(actions)

    # Summary
    print("\n" + "="*40)
    print("       MINIMUM SOLUTION FOUND")
    print("="*40)
    
    if success_count > 0:
        print(f"🏆 Best Length: {best_len} steps")
        print(f"🎯 Success Rate: {success_count}/{num_shots}")
        print("-" * 40)
        for line in best_history:
            print(line)
        print("-" * 40)
        return True, best_history, best_actions
    else:
        print("💀 No solution found in any shot.")
        return False, [], []

# ==============================================================================
# 3. VERIFICATION & VISUALIZATION
# ==============================================================================
def visualize_hardware_state(env):
    """
    Draws the Heavy Hex graph with Logical Qubits mapped to their final positions.
    """
    G = env.phys_graph
    pos = nx.spring_layout(G, seed=42)  # Standard layout
    
    # Separate Nodes by Type
    empty_nodes = []
    qubit_nodes = []
    qubit_labels = {}
    
    for p_node in G.nodes():
        l_qubit = env.phys_to_logical[p_node]
        if l_qubit != -1:
            qubit_nodes.append(p_node)
            qubit_labels[p_node] = f"Q{l_qubit}"
        else:
            empty_nodes.append(p_node)

    # Edges
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'intra']
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'inter']

    plt.figure(figsize=(10, 6))
    
    # Draw Hardware
    nx.draw_networkx_nodes(G, pos, nodelist=empty_nodes, node_color='lightgrey', node_size=300)
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='black', width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='red', style='dashed', width=2.0, label='Optical Link')
    
    # Draw Qubits
    nx.draw_networkx_nodes(G, pos, nodelist=qubit_nodes, node_color='#4CAF50', node_size=500, label='Logical Qubit')
    nx.draw_networkx_labels(G, pos, labels=qubit_labels, font_color='white', font_weight='bold')
    
    plt.title("Final Compilation State (Green = Logical Qubits)")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("📊 Visualization Generated.")

def verify_and_visualize(original_circuit, actions):
    """
    1. Replays the action sequence on a fresh environment to verify constraints.
    2. Visualizes the final qubit mapping on the hardware graph.
    """
    print("\n" + "="*40)
    print("       VERIFICATION & VISUALIZATION")
    print("="*40)

    # 1. SETUP FRESH ENV
    env = AutoExecDistributedEnv(
        original_circuit, 
        num_qpus=model_config['num_qpus'], 
        hex_distance=model_config['hex_dist']
    )
    obs, _ = env.reset()
    
    print(f"🔍 Replaying {len(actions)} actions to check validity...")
    
    valid = True
    error_msg = ""
    
    # 2. STEP-BY-STEP REPLAY
    for i, action in enumerate(actions):
        # Check Mask BEFORE stepping (Strict Rule Check)
        mask = env.get_action_mask()
        if not mask[action]:
            valid = False
            error_msg = f"Step {i}: Action {action} was INVALID (Masked Out)."
            break
            
        # Execute
        obs, reward, term, trunc, info = env.step(action)
        
        if trunc:
            valid = False
            error_msg = f"Step {i}: Environment Triggered Timeout."
            break
            
    # 3. FINAL VALIDATION
    if valid and not info.get('is_success', False):
        valid = False
        error_msg = "Finished actions but Circuit is NOT done (Gates remaining)."
    elif valid:
        print("✅ LOGIC CHECK PASSED: All actions valid, circuit finished.")
    else:
        print(f"❌ VERIFICATION FAILED: {error_msg}")
        return

    # 4. VISUALIZATION
    visualize_hardware_state(env)

def analyze_efficiency(original_circuit, actions):
    """
    Replays the solution to calculate 'Wasted Movement'.
    Metrics:
    1. Total SWAPs per Qubit vs. Net Displacement (Start -> End).
    2. Immediate Reversals (SWAP u-v followed by SWAP u-v).
    """
    print("\n" + "="*40)
    print("       EFFICIENCY ANALYSIS")
    print("="*40)

    # 1. SETUP REPLAY
    env = AutoExecDistributedEnv(
        original_circuit, 
        num_qpus=model_config['num_qpus'], 
        hex_distance=model_config['hex_dist']
    )
    env.reset()
    
    # Trackers
    qubit_path_length = {i: 0 for i in range(env.num_logical)} # Total hops
    qubit_start_pos = {} # Where did it settle after Placement?
    
    redundant_swaps = 0
    total_swaps = 0
    last_swap_link = None
    
    print("⚙️ Replaying trajectory...")

    # 2. REPLAY
    for idx, action in enumerate(actions):
        # Decode Action
        is_swap = (env.limit_place <= action < env.limit_swap)
        swap_link = None
        
        if is_swap:
            total_swaps += 1
            edge_idx = action - env.limit_place
            u, v = env.intra_edges[edge_idx]
            swap_link = tuple(sorted((u, v))) # Sort to handle (u,v) == (v,u)
            
            # CHECK 1: IMMEDIATE REVERSAL (The "Stupid" Move)
            if swap_link == last_swap_link:
                print(f"   ⚠️ REDUNDANCY DETECTED at Step {idx}: Immediate Reversal on Link {swap_link}")
                redundant_swaps += 1
            
            # Track movement for involved qubits
            l_u, l_v = env.phys_to_logical[u], env.phys_to_logical[v]
            if l_u != -1: qubit_path_length[l_u] += 1
            if l_v != -1: qubit_path_length[l_v] += 1
            
        last_swap_link = swap_link
        
        # Execute to update state
        env.step(action)
        
        # Snapshot positions right after Placement phase ends
        if env.current_phase == 1 and not qubit_start_pos:
            for l_q, p_node in enumerate(env.mapping):
                if p_node != -1: qubit_start_pos[l_q] = p_node

    # 3. CALCULATE METRICS
    print("\n--- Per-Qubit Efficiency ---")
    print(f"{'Qubit':<6} | {'Steps':<6} | {'Displacement':<12} | {'Efficiency'}")
    print("-" * 45)
    
    total_eff = 0
    count = 0
    
    for l_q in range(env.num_logical):
        if l_q not in qubit_start_pos: continue # Was never placed?
        
        start = qubit_start_pos[l_q]
        end = env.mapping[l_q]
        
        # Calculate Net Displacement (Shortest path on hardware)
        if start == -1 or end == -1:
            displacement = 0
        else:
            displacement = nx.shortest_path_length(env.phys_graph, start, end)
        
        traveled = qubit_path_length[l_q]
        
        # Efficiency Ratio: Displacement / Traveled
        # 1.0 = Perfect Straight Line
        # 0.0 = Moved but stayed in same spot (Circle)
        if traveled > 0:
            eff = displacement / traveled
        elif displacement == 0 and traveled == 0:
            eff = 1.0 # Didn't need to move, and didn't move. Perfect.
        else:
            eff = 0.0
            
        total_eff += eff
        count += 1
        
        print(f"Q_{l_q:<4} | {traveled:<6} | {displacement:<12} | {eff:.2f}")

    avg_eff = total_eff / max(1, count)
    print("-" * 45)
    print(f"🌍 SYSTEM AVERAGE EFFICIENCY: {avg_eff:.2f}")
    print(f"🔄 Redundant Swaps (Reversals): {redundant_swaps}/{total_swaps}")
    
    if redundant_swaps > 0:
        print("\n❌ VERDICT: Agent contains redundancies. Increase 'Entropy' in training or train longer.")
    elif avg_eff < 0.5:
        print("\n⚠️ VERDICT: Agent is 'wandering'. It solves the problem but takes long paths.")
    else:
        print("\n✅ VERDICT: Agent is efficient.")

def reconstruct_timeline(original_circuit, actions):
    """
    Replays the agent's actions and interweaves them with the 
    Logical Gates (CNOTs) that get executed automatically.
    
    Returns a chronological list of events:
    [
      {'type': 'SWAP', 'qubits': (0, 1)}, 
      {'type': 'GATE', 'name': 'cx', 'qubits': (0, 2)},
      {'type': 'SWAP', 'qubits': (0, 1)}
    ]
    """
    # 1. Setup Env
    env = AutoExecDistributedEnv(
        original_circuit, 
        num_qpus=model_config['num_qpus'], 
        hex_distance=model_config['hex_dist']
    )
    env.reset()
    
    timeline = []
    
    # Track which gates have been executed (Set of IDs)
    previously_executed = set()
    
    print("⚙️ Reconstructing Circuit Timeline...")
    
    for action in actions:
        # A. Record the Routing Action
        if env.limit_place <= action < env.limit_swap:
            idx = action - env.limit_place
            u, v = env.intra_edges[idx]
            # Sort tuple for consistency (0,1) == (1,0)
            phys_edge = tuple(sorted((u, v)))
            timeline.append({'type': 'SWAP', 'qubits': phys_edge, 'action_id': action})
            
        elif env.limit_swap <= action < env.limit_inter:
            # Add Inter-Op logic if needed, treating like a Move
            timeline.append({'type': 'INTER', 'action_id': action})
            
        # B. Execute Step
        env.step(action)
        
        # C. Check for New Logical Gates
        # We look at the tracker to see what finished in this step
        current_executed = set(env.tracker.executed_gates) # Copy the set
        new_gates = current_executed - previously_executed
        
        # We need to sort them to maintain deterministic order (by ID)
        # We use the internal node_to_id from tracker
        new_gates_sorted = sorted(list(new_gates), key=lambda n: env.tracker.node_to_id[n])
        
        for node in new_gates_sorted:
            # We need to know WHICH physical qubits were involved.
            # We use the CURRENT mapping (env.mapping) because gates execute in-place.
            
            # Get Logical Indices
            log_qubits = [original_circuit.qubits.index(q) for q in node.qargs]
            
            if len(log_qubits) == 2:
                # Map Logical -> Physical
                p0 = env.mapping[log_qubits[0]]
                p1 = env.mapping[log_qubits[1]]
                
                # Record Gate Event
                timeline.append({
                    'type': 'GATE', 
                    'name': node.name, 
                    'qubits': tuple(sorted((p0, p1))) # Physical Qubits
                })
        
        previously_executed = current_executed

    return timeline

def check_smart_redundancy(timeline):
    """
    Scans the timeline for SWAP A -> [Intermediate] -> SWAP A patterns.
    It marks SWAP A as redundant ONLY IF the 'Intermediate' events 
    did not use the swapped qubits.
    """
    print("\n" + "="*40)
    print("       SMART REDUNDANCY CHECK")
    print("="*40)
    
    redundant_indices = set()
    
    # We scan for pairs of identical SWAPs
    for i in range(len(timeline)):
        event_a = timeline[i]
        
        if event_a['type'] != 'SWAP': continue
        if i in redundant_indices: continue # Already marked
        
        # Look ahead for the matching reversal
        for j in range(i + 1, len(timeline)):
            event_b = timeline[j]
            
            # Stop if we hit a SWAP that touches our qubits (complex interference)
            # For simplicity, we look for the *exact same link* swap.
            if event_b['type'] == 'SWAP':
                if event_b['qubits'] == event_a['qubits']:
                    # FOUND A REVERSAL CANDIDATE at index j!
                    
                    # Check the "Sandwich": events between i and j
                    is_useful = False
                    swapped_q1, swapped_q2 = event_a['qubits']
                    
                    for k in range(i + 1, j):
                        mid_event = timeline[k]
                        if mid_event['type'] == 'GATE':
                            # Did this gate use either of our swapped qubits?
                            g_q1, g_q2 = mid_event['qubits']
                            if swapped_q1 in (g_q1, g_q2) or swapped_q2 in (g_q1, g_q2):
                                is_useful = True
                                break
                    
                    if not is_useful:
                        # No gate used the qubits! It's a useless Loop.
                        redundant_indices.add(i)
                        redundant_indices.add(j)
                        print(f"❌ FOUND REDUNDANCY: Steps {i} and {j} (Swap {event_a['qubits']}) cancel out with no useful gates in between.")
                    else:
                        print(f"✅ VALID MOVEMENT: Steps {i} and {j} (Swap {event_a['qubits']}) enabled a Gate.")
                        
                    # Stop looking for pair for 'i' once we found 'j'
                    break
    
    # Filter Actions
    # We need to extract just the actions that are NOT redundant
    optimized_actions = []
    actions_kept = 0
    actions_removed = 0
    
    for idx, event in enumerate(timeline):
        if idx in redundant_indices:
            actions_removed += 1
            continue
            
        if 'action_id' in event: # It's an agent action
            optimized_actions.append(event['action_id'])
            actions_kept += 1
            
    print("-" * 40)
    print(f"Total Ops Analyzed: {len(timeline)}")
    print(f"Redundant Ops Removed: {actions_removed}")
    print(f"Final Action Count: {actions_kept}")
    
    return optimized_actions

# --- UPDATED MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define Circuit
    qc = cp_fraction(model_config['num_qubits'], depth=5, fraction=0.5, seed=42)
    model_path = os.path.join(model_config['save_dir'], "best_model.pt")
    
    if os.path.exists(model_path):
        # 2. Inference
        success, log, best_actions = run_inference_stochastic(qc, model_path, num_shots=20)
        
        if success:
            # 3. Reconstruct & Analyze
            timeline = reconstruct_timeline(qc, best_actions)
            
            # 4. Filter Redundancy
            opt_actions = check_smart_redundancy(timeline)
            
            # 5. Verify the Optimized Result works
            print("\n🔍 Verifying Optimized Sequence...")
            verify_and_visualize(qc, best_actions)
            
            # Calculate new efficiency
            analyze_efficiency(qc, best_actions)
            
    else:
        print(f"Model file not found at {model_path}")