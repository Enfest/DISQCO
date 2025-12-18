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
    display_width=None,
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
            
            # Check if this is a state edge (same qubit, different time)
            is_state_edge = (
                isinstance(n1, tuple) and isinstance(n2, tuple)
                and len(n1) == 2 and len(n2) == 2
                and n1[0] == n2[0]  # Same qubit index
            )
            
            if is_state_edge:
                # State edges should be perfectly straight
                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1, zorder=2)
            elif (
                isinstance(n1, tuple) and isinstance(n2, tuple)
                and len(n1) == 2 and len(n2) == 2
                and n1[0] != n2[0]  # Different qubit indices (gate edge)
            ):
                # Gate edges: curve in +x direction with y-offset for visibility
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                x_offset = 0.3 * max(dx, dy * 0.5)
                y_offset = 0.1 * dy
                mx = (x1 + x2) / 2 + x_offset
                my = (y1 + y2) / 2 + (y_offset if y2 > y1 else -y_offset)
                verts = [(x1, y1), (mx, my), (x2, y2)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)
        elif roots and receivers:
            # Hyperedge: place central node at +0.5 offset from earliest root node
            # Find the root with minimum time (earliest in the circuit)
            earliest_root = min(roots, key=lambda n: node_pos.get(n, (float('inf'), 0))[0])
            if earliest_root in node_pos:
                first_root_x = node_pos[earliest_root][0]
                central_x = first_root_x + 0.5 * xscale
            else:
                central_x = 0
            rec_ys = [node_pos[n][1] for n in receivers if n in node_pos]
            root_y = node_pos[earliest_root][1] if earliest_root in node_pos else 0
            if rec_ys:
                avg_rec_y = sum(rec_ys) / len(rec_ys)
                if avg_rec_y > root_y:
                    central_y = root_y + 0.5 * yscale
                else:
                    central_y = root_y - 0.5 * yscale
            else:
                central_y = root_y
            # Draw curves from roots to central node and central node to receivers
            for root in roots:
                x0, y0 = node_pos.get(root, (0, 0))
                x_offset = 0.3 * abs(central_x - x0)
                mx = (x0 + central_x) / 2 + x_offset
                my = (y0 + central_y) / 2
                verts = [(x0, y0), (mx, my), (central_x, central_y)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)
            
            for rec in receivers:
                x1, y1 = node_pos.get(rec, (0, 0))
                x_offset = 0.3 * abs(x1 - central_x)
                mx = (central_x + x1) / 2 + x_offset
                my = (central_y + y1) / 2
                verts = [(central_x, central_y), (mx, my), (x1, y1)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)

<<<<<<< HEAD
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
=======
    buffer_left_time = -1
    buffer_right_time = depth
    for qubit in range(num_qubits):
        left_x = buffer_left_time * xscale
        left_y_val = pos_list[0][qubit]
        if left_y_val is None:
            left_y_val = qubit
        left_y = (num_qubits_phys - int(left_y_val)) * yscale
        right_x = buffer_right_time * xscale
        right_y_val = pos_list[depth-1][qubit]
        if right_y_val is None:
            right_y_val = qubit
        right_y = (num_qubits_phys - int(right_y_val)) * yscale
        ax.scatter(left_x, left_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
        # Only draw right boundary node if the final time-edge exists
        has_tail = True
        if depth >= 2 and hasattr(H, 'hyperedges'):
            edge_id = ((qubit, depth - 2), (qubit, depth - 1))
            has_tail = edge_id in getattr(H, 'hyperedges', {})
        if has_tail:
            ax.scatter(right_x, right_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
        ax.plot([left_x, node_pos.get((qubit, 0), (left_x, left_y))[0]], [left_y, node_pos.get((qubit, 0), (left_x, left_y))[1]], color=edge_color, lw=1, zorder=2)
        if has_tail:
            ax.plot([right_x, node_pos.get((qubit, depth-1), (right_x, right_y))[0]], [right_y, node_pos.get((qubit, depth-1), (right_x, right_y))[1]], color=edge_color, lw=1, zorder=2)
        # Always show q_i labels at the start
        ax.text(left_x-0.4, left_y, f"$q_{{{qubit}}}$", fontsize=11, ha='right', va='center', color='k' if not invert_colors else 'w', zorder=4)
>>>>>>> clean_benchmarking

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
        if display_width is not None:
            # Save to buffer and return HTML img tag with specified width
            from IPython.display import HTML
            from io import BytesIO
            import base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
            buf.seek(0)
            plt.close(fig)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            return HTML(f'<img src="data:image/png;base64,{img_data}" width="{display_width}" />')
        else:
            plt.show()
    return ax


def draw_subgraph_mpl(
    H,
    assignment,
    qpu_info,
    network=None,
    node_map=None,
    xscale=None,
    yscale=None,
    show_labels=True,
    invert_colors=False,
    fill_background=True,
    remove_intermediate_roots=False,
    ax=None,
    figsize=None,
    save_path=None,
    dpi=150,
    display_width=None,
):
    """
    Draw a subgraph with dummy nodes using matplotlib.
    Similar to draw_hypergraph_mpl but handles dummy nodes specially.
    """
    from disqco.drawing.map_positions import find_node_layout_sparse
    
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
    dummy_node_scale = node_scale * 2.0

    # Use sparse layout for subgraphs - returns dict mapping node -> y_position
    node_positions_dict = find_node_layout_sparse(
        graph=H,
        assignment=assignment,
        qpu_sizes=qpu_info,
        node_map=node_map
    )

    # Find time bounds
    if H.nodes:
        max_time = max(n[1] for n in H.nodes if isinstance(n, tuple) and len(n) == 2 and n[0] != "dummy")
        min_time = min(n[1] for n in H.nodes if isinstance(n, tuple) and len(n) == 2 and n[0] != "dummy")
    else:
        max_time = depth - 1
        min_time = 0

    if ax is None:
        # Calculate adaptive figure size based on circuit dimensions
        if figsize is None:
            # Width based on time steps, height based on qubits
            width = max(4, min(16, (max_time - min_time + 3) * 0.8))
            height = max(3, min(10, num_qubits_phys * 0.6))
            figsize = (width, height)
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
        'dummy': 'red',  # Dummy nodes in red
        'invisible': 'none',
    }
    edge_color = 'black' if not invert_colors else 'white'
    boundary_color = 'black' if not invert_colors else 'white'

    def pick_position(node):
        # Handle dummy nodes - place on right boundary
        if isinstance(node, tuple) and len(node) > 2 and node[0] == "dummy":
            dummy_vertical_shift = 3.0 * yscale
            partition = node[2]
            x = (max_time + 2.0) * xscale
            base_y = node_map[partition] if node_map else partition
            y = base_y * yscale + dummy_vertical_shift
            return (x, y)
        
        # Regular nodes - use the dictionary returned by find_node_layout_sparse
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
            x = t * xscale
            # Get y position from dictionary, with fallback
            y_pos = node_positions_dict.get(node, q)  # Fallback to qubit index if not found
            y = (num_qubits_phys - y_pos) * yscale
            return (x, y)
        return (0, 0)

    def pick_style(node):
        # Dummy nodes
        if isinstance(node, tuple) and len(node) >= 2 and node[0] == "dummy":
            return 'dummy'
        
        # Regular nodes
        if hasattr(H, 'get_node_attribute'):
            node_type = H.get_node_attribute(node, 'type', None)
            if node_type in ("group", "two-qubit"):
                if H.node_attrs.get(node, {}).get('name') == "target":
                    return 'white'
                return 'black'
            elif node_type == "root_t":
                if remove_intermediate_roots:
                    return 'invisible'
                return 'black'
            elif node_type == "measure":
                return 'black'
            elif node_type == "single-qubit":
                params = H.get_node_attribute(node, 'params', None)
                if params is not None and len(params) > 0:
                    params_sum = sum(abs(x) for x in params)
                    if params_sum < 1e-10:
                        return 'invisible'
                return 'grey'
        return 'invisible'

    # Build network node positions if network is provided
    qpu_node_pos = {}
    if network is not None and hasattr(network, 'qpu_graph'):
        qpu_x = (max_time + 3.0) * xscale
        # Calculate y positions for QPU nodes
        # Boundaries are at (boundary - 0.5), so QPU nodes should be positioned
        # such that boundaries are at midpoints between them
        prev_boundary = 0
        for qpu_node in network.active_nodes:
            i = node_map[qpu_node] if node_map is not None else qpu_node
            next_boundary = sum(qpu_sizes[:i+1]) if i < len(qpu_sizes) - 1 else num_qubits_phys
            # Position QPU at center of partition region
            y = (num_qubits_phys - (prev_boundary + next_boundary)/2) * yscale
            qpu_node_pos[qpu_node] = (qpu_x, y)
            prev_boundary = next_boundary
        
        # Position inactive QPU nodes outside the main region
        for qpu_node in network.qpu_graph.nodes:
            if qpu_node not in network.active_nodes:
                if qpu_node > max(network.active_nodes):
                    y = - num_qubits_phys / len(network.active_nodes) * yscale
                elif qpu_node < min(network.active_nodes):
                    y = (num_qubits_phys + num_qubits_phys / len(network.active_nodes)) * yscale
                else:
                    y = 0
                qpu_node_pos[qpu_node] = (qpu_x, y)
    
    node_pos = {}
    for n in H.nodes:
        # Handle dummy nodes - map to network QPU positions
        if isinstance(n, tuple) and len(n) > 2 and n[0] == "dummy":
            partition = n[2]
            if network is not None and partition in qpu_node_pos:
                # Use QPU node position for dummy nodes
                node_pos[n] = qpu_node_pos[partition]
            else:
                # Fallback to old position
                x, y = pick_position(n)
                node_pos[n] = (x, y)
            continue  # Don't draw dummy nodes separately - they'll be part of network
        
        x, y = pick_position(n)
        node_pos[n] = (x, y)
        style = pick_style(n)
        color = node_colors.get(style, 'grey')
        if style == 'invisible':
            continue
        
        # Different sizes for different node types
        if isinstance(n, tuple) and len(n) == 3:
            size = 200 * gate_node_scale
            marker = 'o'
        else:
            size = 200 * node_scale
            marker = 'o'
        
        ax.scatter(x, y, s=size, c=color, edgecolors='k', marker=marker, zorder=3)
        
        if show_labels and style != 'dummy':
            if isinstance(n, tuple) and len(n) == 2:
                q, t = n
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
                ax.text(x, y+0.12, f"({q},{t})", fontsize=8, ha='center', va='bottom', color='k' if not invert_colors else 'w', zorder=4)
                if extra:
                    ax.text(x, y-0.12, f"{extra}", fontsize=8, ha='center', va='top', color='k' if not invert_colors else 'w', zorder=4)

    # Draw edges (same as regular version)
    for edge_id, edge_info in getattr(H, 'hyperedges', {}).items() if hasattr(H, 'hyperedges') else []:
        roots = list(edge_info['root_set'])
        receivers = list(edge_info['receiver_set'])
        all_nodes = roots + receivers
        if len(all_nodes) == 2:
            n1, n2 = all_nodes
            x1, y1 = node_pos.get(n1, (0, 0))
            x2, y2 = node_pos.get(n2, (0, 0))
            
            # Check if this is a state edge (same qubit, including dummy nodes)
            # For edges involving dummy nodes, check if the non-dummy node's qubit matches
            # by checking if both nodes have the same first element (qubit index)
            is_state_edge = False
            
            n1_is_dummy = isinstance(n1, tuple) and len(n1) > 2 and n1[0] == "dummy"
            n2_is_dummy = isinstance(n2, tuple) and len(n2) > 2 and n2[0] == "dummy"
            
            if not n1_is_dummy and not n2_is_dummy:
                # Both are regular nodes - check if same qubit
                if (isinstance(n1, tuple) and isinstance(n2, tuple) 
                    and len(n1) == 2 and len(n2) == 2 
                    and n1[0] == n2[0]):
                    is_state_edge = True
            elif n1_is_dummy or n2_is_dummy:
                # One or both are dummy nodes
                # State edges to dummy nodes are those where the edge ID suggests same qubit continuation
                # Check if edge_id is a simple tuple like ((q,t1), (q,t2)) pattern
                if isinstance(edge_id, tuple) and len(edge_id) == 2:
                    id1, id2 = edge_id
                    if (isinstance(id1, tuple) and isinstance(id2, tuple) 
                        and len(id1) == 2 and len(id2) == 2
                        and id1[0] == id2[0]):  # Same qubit in edge ID
                        is_state_edge = True
            
            if is_state_edge:
                # State edges should be straight lines
                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1, zorder=2)
            elif (
                isinstance(n1, tuple) and isinstance(n2, tuple)
                and len(n1) == 2 and len(n2) == 2
                and n1[0] != n2[0]
            ):
                # Gate edges between different qubits (both regular nodes)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                x_offset = 0.3 * max(dx, dy * 0.5)
                y_offset = 0.1 * dy
                mx = (x1 + x2) / 2 + x_offset
                my = (y1 + y2) / 2 + (y_offset if y2 > y1 else -y_offset)
                verts = [(x1, y1), (mx, my), (x2, y2)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)
        elif roots and receivers:
            earliest_root = min(roots, key=lambda n: node_pos.get(n, (float('inf'), 0))[0])
            if earliest_root in node_pos:
                first_root_x = node_pos[earliest_root][0]
                central_x = first_root_x + 0.5 * xscale
            else:
                central_x = 0
            rec_ys = [node_pos[n][1] for n in receivers if n in node_pos]
            root_y = node_pos[earliest_root][1] if earliest_root in node_pos else 0
            if rec_ys:
                avg_rec_y = sum(rec_ys) / len(rec_ys)
                if avg_rec_y > root_y:
                    central_y = root_y + 0.5 * yscale
                else:
                    central_y = root_y - 0.5 * yscale
            else:
                central_y = root_y
            
            for root in roots:
                x0, y0 = node_pos.get(root, (0, 0))
                x_offset = 0.3 * abs(central_x - x0)
                mx = (x0 + central_x) / 2 + x_offset
                my = (y0 + central_y) / 2
                verts = [(x0, y0), (mx, my), (central_x, central_y)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)
            
            for rec in receivers:
                x1, y1 = node_pos.get(rec, (0, 0))
                x_offset = 0.3 * abs(x1 - central_x)
                mx = (central_x + x1) / 2 + x_offset
                my = (central_y + y1) / 2
                verts = [(central_x, central_y), (mx, my), (x1, y1)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', ec=edge_color, lw=1, zorder=2)
                ax.add_patch(bezier)

    # Draw boundary nodes at left and right edges
    buffer_left_time = min_time - 1
    buffer_right_time = max_time + 1
    
    # Get all qubits that appear in the subgraph
    subgraph_qubits = set()
    for n in H.nodes:
        if isinstance(n, tuple) and len(n) == 2 and n[0] != "dummy":
            subgraph_qubits.add(n[0])
    
    # Track which qubits exist at boundary time steps
    qubits_at_min_time = set()
    qubits_at_max_time = set()
    for n in H.nodes:
        if isinstance(n, tuple) and len(n) == 2 and n[0] != "dummy":
            q, t = n
            if t == min_time:
                qubits_at_min_time.add(q)
            if t == max_time:
                qubits_at_max_time.add(q)
    
    for qubit in sorted(subgraph_qubits):
        # Left boundary node - only if qubit exists at min_time
        if qubit in qubits_at_min_time:
            left_x = buffer_left_time * xscale
            left_y_pos = node_positions_dict.get((qubit, min_time), qubit)
            left_y = (num_qubits_phys - left_y_pos) * yscale
            ax.scatter(left_x, left_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
            # Label q_i on the left
            ax.text(left_x-0.4, left_y, f"$q_{{{qubit}}}$", fontsize=11, ha='right', va='center', 
                   color='k' if not invert_colors else 'w', zorder=4)
            # Connect to first node
            if (qubit, min_time) in node_pos:
                first_node_x, first_node_y = node_pos[(qubit, min_time)]
                ax.plot([left_x, first_node_x], [left_y, first_node_y], color=edge_color, lw=1, zorder=2)
        
        # Right boundary node - only if qubit exists at max_time
        if qubit in qubits_at_max_time:
            right_x = buffer_right_time * xscale
            right_y_pos = node_positions_dict.get((qubit, max_time), qubit)
            right_y = (num_qubits_phys - right_y_pos) * yscale
            ax.scatter(right_x, right_y, s=100*small_node_scale, c='w', edgecolors='k', zorder=3)
            # Connect to last node
            if (qubit, max_time) in node_pos:
                last_node_x, last_node_y = node_pos[(qubit, max_time)]
                ax.plot([last_node_x, right_x], [last_node_y, right_y], color=edge_color, lw=1, zorder=2)

    # Draw partition boundaries
    for i in range(1, len(qpu_sizes)):
        boundary = sum(qpu_sizes[:i])
        line_y = (num_qubits_phys - boundary - 0.5) * yscale
        ax.axhline(line_y, color=boundary_color, linestyle='--', lw=1, zorder=10)

    # Draw mini QPU network graph (replaces Q labels)
    if network is not None and hasattr(network, 'qpu_graph'):
        qpu_graph = network.qpu_graph
        
        # Calculate QPU node size based on figure dimensions and circuit size
        # Get figure size
        fig_width, fig_height = fig.get_size_inches()
        # Scale factor based on figure size (smaller figures need smaller nodes)
        fig_scale = min(fig_width / 10.0, fig_height / 6.0)
        qpu_node_size_base = min(1.5, max(0.6, 8.0 / num_qubits)) * fig_scale
        
        for qpu_node in network.qpu_graph.nodes:
            if qpu_node in qpu_node_pos:
                qpu_x, qpu_y = qpu_node_pos[qpu_node]
                # Color based on active/inactive
                if qpu_node in network.active_nodes:
                    node_color = 'royalblue'
                else:
                    node_color = 'lightgray'
                
                # Draw QPU node with size scaled to figure
                qpu_size = 1600 * qpu_node_size_base
                ax.scatter(qpu_x, qpu_y, s=qpu_size, c=node_color, edgecolors='k', 
                          marker='o', linewidths=1.5, zorder=20)
                
                # Add label with size scaled to figure
                label_fontsize = max(8, min(20, 150 / num_qubits * fig_scale))
                label_color = 'k' if not invert_colors else 'w'
                ax.text(qpu_x, qpu_y, f"$Q_{{{qpu_node}}}$", fontsize=label_fontsize, 
                       ha='center', va='center', color=label_color, weight='bold', zorder=21)
        
        # Draw edges between QPU nodes
        for idx, (src, tgt) in enumerate(qpu_graph.edges()):
            if src in qpu_node_pos and tgt in qpu_node_pos:
                x1, y1 = qpu_node_pos[src]
                x2, y2 = qpu_node_pos[tgt]
                
                # Draw curved edges with alternating bends
                dx = x2 - x1
                dy = y2 - y1
                
                # Alternate bend direction
                if idx % 2 == 0:
                    bend_factor = 0.2
                else:
                    bend_factor = -0.2
                
                # Control point for bezier curve
                mx = (x1 + x2) / 2 + bend_factor * abs(dy)
                my = (y1 + y2) / 2
                
                # Draw bezier curve
                verts = [(x1, y1), (mx, my), (x2, y2)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                bezier = patches.PathPatch(MplPath(verts, codes), fc='none', 
                                          ec='black', lw=1.5, zorder=19)
                ax.add_patch(bezier)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    if display_width is not None:
        # Save to buffer and return HTML img tag with specified width
        from IPython.display import HTML
        from io import BytesIO
        import base64
        fig = ax.figure
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        return HTML(f'<img src="data:image/png;base64,{img_data}" width="{display_width}" />')
    else:
        plt.show()
<<<<<<< HEAD
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
=======
    
    return ax
>>>>>>> clean_benchmarking
