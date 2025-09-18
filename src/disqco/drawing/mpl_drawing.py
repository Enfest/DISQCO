import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
from disqco.drawing.map_positions import find_node_layout_sparse
import networkx as nx
from disqco.parti.FM.FM_methods_nx import calculate_cut_size




def draw_hypergraph_mpl(
    H,
    assignment,
    qpu_info,
    xscale=None,
    yscale=None,
    show_labels=True,
    invert_colors=False,
    fill_background=True,
    assignment_map=None,
    remove_intermediate_roots=False,
    ax=None,
    figsize=(10, 6),
    save_path=None,
    dpi=150,
):
    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    depth = getattr(H, 'depth', 0)
    num_qubits = getattr(H, 'num_qubits', 0)
    num_qubits_phys = sum(qpu_sizes)

    if xscale is None:
        xscale = 10.0 / depth if depth else 1
    if yscale is None:
        yscale = 6.0 / num_qubits if num_qubits else 1

    node_scale = min(0.6, max(0.3, 1.0 / (max(depth, num_qubits) ** 0.5)))
    gate_node_scale = node_scale * 1.2
    small_node_scale = node_scale * 0.5

    # space_map = space_mapping(qpu_sizes, depth)
    # print(space_map)
    # print(assignment)
    # pos_list = get_pos_list(H, num_qubits, assignment, space_map)

    pos_list = find_node_layout_sparse(H, assignment, qpu_sizes)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if fill_background:
        ax.set_facecolor('black' if invert_colors else 'white')
    else:
        ax.set_facecolor('none')

    node_colors = {
        'black': 'black' if not invert_colors else 'white',
        'white': 'white',
        'grey': 'grey',
        'dummy': 'blue',
        'invisible': 'none',
    }
    edge_color = 'black' if not invert_colors else 'white'
    boundary_color = 'black' if not invert_colors else 'white'

    def pick_position(node):
        if isinstance(node, tuple) and len(node) == 3 and node[0] == "dummy":
            _, p, pprime = node
            x = (depth/len(qpu_sizes) *(pprime-1)) * xscale * 1.2
            y = (-2) * yscale * 0.8
            return (x, y)
        if isinstance(node, tuple) and len(node) == 2:
            if assignment_map is not None:
                q, t = assignment_map[node]
            else:
                q, t = node
            x = t * xscale
            y = (num_qubits_phys - pos_list[node]) * yscale
            return (x, y)
        return (0, 0)

    def pick_style(node):
        if hasattr(H, 'get_node_attribute') and H.get_node_attribute(node, 'dummy', False):
            return 'dummy'
        node_type = H.get_node_attribute(node, 'type', None) if hasattr(H, 'get_node_attribute') else None
        if node_type in ("group", "two-qubit"):
            if hasattr(H, 'node_attrs') and H.node_attrs[node].get('name') == "target":
                return 'white'
            else:
                return 'black'
        elif node_type == "root_t":
            if remove_intermediate_roots:
                return 'invisible'
            else:
                return 'black'
        elif node_type == "measure":
            # Draw measurements as black nodes (like control)
            return 'black'
        elif node_type == "single-qubit":
            params = H.get_node_attribute(node, 'params', None) if hasattr(H, 'get_node_attribute') else None
            if params is not None and len(params) > 0:
                params_sum = sum(abs(x) for x in params)
                if params_sum < 1e-10:
                    return 'invisible'
            return 'grey'
        else:
            return 'invisible'

    node_pos = {}
    for n in H.nodes:
        x, y = pick_position(n)
        node_pos[n] = (x, y)
        style = pick_style(n)
        color = node_colors.get(style, 'grey')
        if style == 'invisible':
            continue
        size = 200 * (gate_node_scale if (isinstance(n, tuple) and len(n) == 3) else node_scale)
        ax.scatter(x, y, s=size, c=color, edgecolors='k', zorder=3)
        if show_labels and style != 'dummy':
            if isinstance(n, tuple) and len(n) == 2:
                q, t = n
                # Build label with classical bit info for measure or classically controlled nodes
                extra = ""
                n_type = H.get_node_attribute(n, 'type', None) if hasattr(H, 'get_node_attribute') else None
                if n_type == 'measure':
                    cbit = H.get_node_attribute(n, 'measurement_bit', None)
                    if cbit is not None:
                        extra = f"c{cbit}"
                elif hasattr(H, 'get_node_attribute') and H.get_node_attribute(n, 'classically_controlled', False):
                    reg = H.get_node_attribute(n, 'control_register', None)
                    val = H.get_node_attribute(n, 'control_val', None)
                    if reg is not None and val is not None:
                        extra = f"{reg}=={val}"
                    else:
                        cbit = H.get_node_attribute(n, 'control_bit', None)
                        if cbit is not None:
                            extra = f"c{cbit}"
                # Draw main coords above and condition label below to avoid overlap
                ax.text(x, y+0.12, f"({q},{t})", fontsize=8, ha='center', va='bottom', color='k' if not invert_colors else 'w', zorder=4)
                if extra:
                    ax.text(x, y-0.12, f"{extra}", fontsize=8, ha='center', va='top', color='k' if not invert_colors else 'w', zorder=4)

    for edge_id, edge_info in getattr(H, 'hyperedges', {}).items() if hasattr(H, 'hyperedges') else []:
        roots = list(edge_info['root_set'])
        receivers = list(edge_info['receiver_set'])
        all_nodes = roots + receivers
        if len(all_nodes) == 2:
            n1, n2 = all_nodes
            x1, y1 = node_pos.get(n1, (0, 0))
            x2, y2 = node_pos.get(n2, (0, 0))
            # Only bend if q indices are different
            if (
                isinstance(n1, tuple) and isinstance(n2, tuple)
                and len(n1) == 2 and len(n2) == 2
                and n1[0] != n2[0]
            ):
                dx, dy = x2 - x1, y2 - y1
                norm = np.hypot(dx, dy)
                if norm == 0:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    px, py = -dy / norm, dx / norm
                    bend = 0.3 * yscale
                    mx, my = (x1 + x2) / 2 + px * bend, (y1 + y2) / 2 + py * bend
                path = np.array([[x1, y1], [mx, my], [x2, y2]])
                verts = [tuple(path[0]), tuple(path[1]), tuple(path[2])]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1.5, zorder=2)
                ax.add_patch(bezier)
            else:
                # Draw straight line for same q index
                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1.5, zorder=2)
        elif roots and receivers:
            # Central node logic
            root_times = [n[1] for n in roots if isinstance(n, tuple) and len(n) == 2]
            rec_times = [n[1] for n in receivers if isinstance(n, tuple) and len(n) == 2]
            if root_times and rec_times:
                min_time = min(root_times + rec_times)
                max_time = max(root_times + rec_times)
                central_x = (min_time + max_time) / 2 * xscale
            else:
                central_x = 0
            rec_ys = [node_pos[n][1] for n in receivers if n in node_pos]
            root_y = node_pos[roots[0]][1] if roots and roots[0] in node_pos else 0
            if rec_ys:
                avg_rec_y = sum(rec_ys) / len(rec_ys)
                if avg_rec_y > root_y:
                    central_y = root_y + 0.5 * yscale
                else:
                    central_y = root_y - 0.5 * yscale
            else:
                central_y = root_y
            # Roots to central node (bend up)
            for root in roots:
                x0, y0 = node_pos.get(root, (0, 0))
                mx, my = (x0 + central_x) / 2, (y0 + central_y) / 2 + 0.5 * yscale
                path = np.array([[x0, y0], [mx, my], [central_x, central_y]])
                verts = [tuple(path[0]), tuple(path[1]), tuple(path[2])]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1.5, zorder=2)
                ax.add_patch(bezier)
            # Central node to receivers (bend down)
            for rec in receivers:
                x1, y1 = node_pos.get(rec, (0, 0))
                mx, my = (central_x + x1) / 2, (central_y + y1) / 2 - 0.5 * yscale
                path = np.array([[central_x, central_y], [mx, my], [x1, y1]])
                verts = [tuple(path[0]), tuple(path[1]), tuple(path[2])]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1.5, zorder=2)
                ax.add_patch(bezier)

    # buffer_left_time = -1
    # buffer_right_time = depth
    # for qubit in range(num_qubits):
    #     left_x = buffer_left_time * xscale
    #     left_y_val = pos_list[(qubit, 0)]
    #     if left_y_val is None:
    #         left_y_val = qubit
    #     left_y = (num_qubits_phys - int(left_y_val)) * yscale
    #     right_x = buffer_right_time * xscale
    #     right_y_val = pos_list[(qubit, depth-1)]
    #     if right_y_val is None:
    #         right_y_val = qubit
    #     right_y = (num_qubits_phys - int(right_y_val)) * yscale
    #     ax.scatter(left_x, left_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
    #     # Only draw right boundary node if the final time-edge exists
    #     has_tail = True
    #     if depth >= 2 and hasattr(H, 'hyperedges'):
    #         edge_id = ((qubit, depth - 2), (qubit, depth - 1))
    #         has_tail = edge_id in getattr(H, 'hyperedges', {})
    #     if has_tail:
    #         ax.scatter(right_x, right_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
    #     ax.plot([left_x, node_pos.get((qubit, 0), (left_x, left_y))[0]], [left_y, node_pos.get((qubit, 0), (left_x, left_y))[1]], color=edge_color, lw=1, zorder=2)
    #     if has_tail:
    #         ax.plot([right_x, node_pos.get((qubit, depth-1), (right_x, right_y))[0]], [right_y, node_pos.get((qubit, depth-1), (right_x, right_y))[1]], color=edge_color, lw=1, zorder=2)
    #     # Always show q_i labels at the start
    #     ax.text(left_x-0.4, left_y, f"$q_{{{qubit}}}$", fontsize=11, ha='right', va='center', color='k' if not invert_colors else 'w', zorder=4)

    # Draw partition boundaries and Q_i labels
    for i in range(1, len(qpu_sizes)):
        boundary = sum(qpu_sizes[:i])+1
        line_y = (num_qubits_phys - boundary + 0.5) * yscale
        ax.axhline(line_y, color=boundary_color, linestyle='--', lw=1, zorder=10)
        # Add Q_i label above the boundary line on the far right, starting from Q_0
        ax.text((depth + 1.5) * xscale, line_y + 0.5 * yscale, f"$Q_{{{i-1}}}$", fontsize=18, ha='left', va='bottom', color=boundary_color, zorder=20)

    # Add a final QPU label for the last partition (Q_{N-1})
    final_boundary = sum(qpu_sizes)
    final_line_y = (num_qubits_phys - final_boundary + 0.5) * yscale
    ax.text((depth + 1.5) * xscale, final_line_y + 0.5 * yscale, f"$Q_{{{len(qpu_sizes)-1}}}$", fontsize=18, ha='left', va='bottom', color=boundary_color, zorder=20)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if ax is None:
        plt.show()
    return ax

def draw_partitioned_graph_nx(graph, num_partitions, assignment, title="Graph Partitioning"):
    """
    Draw a graph with nodes positioned in boxes on the circumference of a circle, one box per QPU
    """
    if len(graph.nodes()) > 64:
        print(f"Graph too large to visualize effectively ({len(graph.nodes())} nodes). Skipping visualization.")
        return
        
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_partitions))
    
    # Group nodes by partition
    partition_nodes = {}
    for node_idx, node in enumerate(graph.nodes()):
        partition = assignment[node] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[node_idx]
        if partition not in partition_nodes:
            partition_nodes[partition] = []
        partition_nodes[partition].append(node)
    
    pos = {}
    
    # Parameters for box layout on circumference
    circle_radius = 1.0  # Radius of the main circle
    box_width = 0.6      # Width of each box
    box_height = 0.6     # Height of each box
    
    # For each partition, create a box on the circumference with spring layout inside
    for partition in range(num_partitions):
        if partition not in partition_nodes or not partition_nodes[partition]:
            continue
            
        nodes_in_partition = partition_nodes[partition]
        
        # Calculate angle for this partition's box position
        angle = (partition / num_partitions) * 2 * np.pi
        
        # Position the box center on the circumference
        box_center_x = circle_radius * np.cos(angle)
        box_center_y = circle_radius * np.sin(angle)
        
        # Create subgraph for this partition (only internal edges)
        partition_subgraph = graph.subgraph(nodes_in_partition)
        
        # Generate spring layout for the subgraph
        if len(nodes_in_partition) == 1:
            # Single node - place at box center
            pos[nodes_in_partition[0]] = (box_center_x, box_center_y)
        else:
            # Multiple nodes - use spring layout within the box
            subgraph_pos = nx.spring_layout(
                partition_subgraph, 
                k=1/np.sqrt(len(nodes_in_partition)), 
                iterations=50,
                seed=42 + partition  # Different seed for each partition
            )
            
            # Transform the spring layout to fit within the box
            if subgraph_pos:
                # Scale factor to fit within the box
                scale_factor_x = box_width * 0.8 / 2  # Leave some margin
                scale_factor_y = box_height * 0.8 / 2
                
                # Transform each node position
                for node in nodes_in_partition:
                    # Get normalized position from spring layout
                    local_x, local_y = subgraph_pos[node]
                    
                    # Scale and translate to box region
                    final_x = box_center_x + local_x * scale_factor_x
                    final_y = box_center_y + local_y * scale_factor_y
                    
                    pos[node] = (final_x, final_y)
    
    # Draw boxes on the circumference for each partition
    for partition in range(num_partitions):
        if partition not in partition_nodes or not partition_nodes[partition]:
            continue
            
        # Calculate angle and box position
        angle = (partition / num_partitions) * 2 * np.pi
        box_center_x = circle_radius * np.cos(angle)
        box_center_y = circle_radius * np.sin(angle)
        
        # Calculate box boundaries
        left = box_center_x - box_width / 2
        right = box_center_x + box_width / 2
        bottom = box_center_y - box_height / 2
        top = box_center_y + box_height / 2
        
        # Draw box boundary
        box_x = [left, right, right, left, left]
        box_y = [bottom, bottom, top, top, bottom]
        ax.plot(box_x, box_y, 'k-', alpha=0.7, linewidth=2)
        
        # Add QPU label above the box
        label_x = box_center_x
        label_y = top + 0.08
        ax.text(label_x, label_y, f'QPU {partition}', 
                fontsize=10, fontweight='bold', 
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[partition], alpha=0.8))
    
    
    # Draw edges
    edge_colors = []
    for u, v in graph.edges():
        u_partition = assignment[u] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[list(graph.nodes()).index(u)]
        v_partition = assignment[v] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[list(graph.nodes()).index(v)]
        
        if u_partition == v_partition:
            edge_colors.append('gray')  # Internal edges
        else:
            edge_colors.append('red')   # Cut edges
    
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7, width=1.5, ax=ax)
    
    # Draw nodes colored by partition
    for partition in range(num_partitions):
        nodes_in_partition = []
        for node_idx, node in enumerate(graph.nodes()):
            node_partition = assignment[node] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[node_idx]
            if node_partition == partition:
                nodes_in_partition.append(node)
        
        if nodes_in_partition:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes_in_partition, 
                                 node_color=[colors[partition]], 
                                 node_size=400, alpha=0.9, edgecolors='black', linewidths=1, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold', ax=ax)
    
    # Draw edge weights if they exist
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6, ax=ax)
    
    ax.set_xlim(-1.8, 1.8)  # Larger to accommodate boxes on circumference
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    cut_edges = sum(1 for u, v in graph.edges() 
                   if (assignment[u] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[list(graph.nodes()).index(u)]) != 
                      (assignment[v] if isinstance(assignment, (dict, np.ndarray)) and len(assignment) > max(graph.nodes()) else assignment[list(graph.nodes()).index(v)]))
    total_edges = len(graph.edges())
    
    # Calculate cut size properly
    try:
        cut_size = calculate_cut_size(graph, assignment)
        ax.text(0, -1.6, f'Cut edges: {cut_edges}/{total_edges} (Cut size: {cut_size})', 
                ha='center', fontsize=10, fontweight='bold')
    except:
        ax.text(0, -1.6, f'Cut edges: {cut_edges}/{total_edges}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()