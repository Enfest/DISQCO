"""
Plot and analyze benchmark results

This script generates publication-quality plots and summary tables
from benchmark DAT files.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path


def ensure_dat_files_exist(data_dir, dat_dir):
    """
    Check if DAT files exist, and if not, generate them by calling generate_dat_files.py.
    
    Args:
        data_dir: Directory containing benchmark JSON files
        dat_dir: Directory where DAT files should be located
    """
    dat_path = Path(dat_dir)
    
    # Check if dat_files directory exists and has files
    if dat_path.exists() and list(dat_path.glob("*.dat")):
        print(f"Using existing DAT files from {dat_dir}")
        return
    
    print(f"DAT files not found in {dat_dir}. Generating them now...")
    
    # Call generate_dat_files.py script
    script_path = Path(__file__).parent / "generate_dat_files.py"
    
    try:
        result = subprocess.run(
            ["python", str(script_path), "--input_dir", data_dir, "--output_dir", dat_dir],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("DAT files generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error generating DAT files: {e}")
        print(e.stderr)
        raise


def load_dat_file(filepath):
    """
    Load a single DAT file and return as a dictionary.
    
    Args:
        filepath: Path to .dat file
        
    Returns:
        Dictionary with column names as keys and lists of values
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Parse header
        header = lines[0].strip().split()
        
        # Initialize data dictionary
        data = {col: [] for col in header}
        
        # Parse data lines
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split()
                for col, val in zip(header, values):
                    # Try to convert to appropriate type
                    try:
                        if '.' in val:
                            data[col].append(float(val))
                        else:
                            data[col].append(int(val))
                    except ValueError:
                        data[col].append(val)
        
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def get_all_dat_files(dat_dir):
    """Get all DAT files from dat directory."""
    dat_path = Path(dat_dir)
    return list(dat_path.glob("*.dat"))


def plot_dat_file(dat_file_path, output_dir):
    """Plot a single DAT file."""
    data = load_dat_file(dat_file_path)
    
    if data is None or not data:
        return
    
    # Determine metric type from filename
    filename = dat_file_path.stem
    is_cost = 'cost' in filename
    is_time = 'time' in filename
    
    # Determine x-axis column
    if 'circuit_name' in data:
        # QASM files - use circuit_name on x-axis, separate bars for num_partitions
        x_col = 'circuit_name'
        x_label = 'Circuit Name'
        group_col = 'num_partitions'
    elif 'num_qubits' in data:
        x_col = 'num_qubits'
        x_label = 'Number of Qubits'
        group_col = None
    elif 'num_partitions' in data:
        x_col = 'num_partitions'
        x_label = 'Number of Partitions'
        group_col = None
    else:
        print(f"Skipping {filename}: No recognizable x-axis column")
        return
    
    # Check for r_mean column
    if 'r_mean' not in data:
        print(f"Skipping {filename}: No r_mean column found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if group_col:
        # Plot grouped bar chart for each group (e.g., num_partitions in QASM files)
        unique_groups = sorted(set(data[group_col]))
        unique_x = sorted(set(data[x_col]))
        
        # Calculate bar width and positions
        bar_width = 0.8 / len(unique_groups)
        x_positions = np.arange(len(unique_x))
        
        for i, group_val in enumerate(unique_groups):
            # Filter data for this group
            indices = [idx for idx, v in enumerate(data[group_col]) if v == group_val]
            
            # Create mapping from unique_x to y_vals
            x_to_y = {}
            for idx in indices:
                x_val = data[x_col][idx]
                x_to_y[x_val] = {
                    'y': data['r_mean'][idx],
                    'ymin': data['r_min'][idx] if 'r_min' in data else None,
                    'ymax': data['r_max'][idx] if 'r_max' in data else None
                }
            
            # Build y_vals in same order as unique_x
            y_vals = [x_to_y[x]['y'] for x in unique_x if x in x_to_y]
            
            # Calculate error bars if min/max available
            if 'r_min' in data and 'r_max' in data:
                r_min_vals = [x_to_y[x]['ymin'] for x in unique_x if x in x_to_y]
                r_max_vals = [x_to_y[x]['ymax'] for x in unique_x if x in x_to_y]
                yerr = [[y - ymin for y, ymin in zip(y_vals, r_min_vals)],
                        [ymax - y for y, ymax in zip(y_vals, r_max_vals)]]
            else:
                yerr = None
            
            label = f"{group_col.replace('num_', '')} = {group_val}"
            offset = (i - len(unique_groups)/2 + 0.5) * bar_width
            bar_positions = x_positions[:len(y_vals)] + offset
            ax.bar(bar_positions, y_vals, bar_width, label=label, yerr=yerr, capsize=3)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_x, rotation=45, ha='right')
    else:
        # Single bar chart
        x_vals = data[x_col]
        y_vals = data['r_mean']
        
        # Calculate error bars if min/max available
        if 'r_min' in data and 'r_max' in data:
            yerr = [[y - ymin for y, ymin in zip(y_vals, data['r_min'])],
                    [ymax - y for y, ymax in zip(y_vals, data['r_max'])]]
        else:
            yerr = None
        
        # Use positions for bars
        x_positions = np.arange(len(x_vals))
        ax.bar(x_positions, y_vals, yerr=yerr, capsize=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_vals)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    
    if is_cost:
        ax.set_ylabel('Entanglement cost')
        title = filename.replace('_cost', '').replace('_', ' ').title()
        title += ' - Cost'
    elif is_time:
        ax.set_ylabel('Time taken (s)')
        title = filename.replace('_time', '').replace('_', ' ').title()
        title += ' - Runtime'
    else:
        ax.set_ylabel('Mean Value')
        title = filename.replace('_', ' ').title()
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if group_col:
        ax.legend()
    
    plt.tight_layout()
    
    # Save with same name as dat file but as png
    output_file = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()





def main():
    """Generate all plots and analysis."""
    data_dir = "../data"
    dat_dir = "./dat_files"
    output_dir = "./plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Generating Benchmark Plots")
    print("="*70)
    
    # Ensure DAT files exist (generate if needed)
    ensure_dat_files_exist(data_dir, dat_dir)
    
    # Get all DAT files
    dat_files = get_all_dat_files(dat_dir)
    
    if not dat_files:
        print(f"No DAT files found in {dat_dir}")
        return
    
    print(f"\nFound {len(dat_files)} DAT files to plot")
    
    # Plot each DAT file
    for dat_file in sorted(dat_files):
        print(f"Processing {dat_file.name}...")
        plot_dat_file(dat_file, output_dir)
    
    print("\n" + "="*70)
    print(f"All plots saved to: {output_dir}")
    print(f"Total plots generated: {len(dat_files)}")
    print("="*70)


if __name__ == "__main__":
    main()
