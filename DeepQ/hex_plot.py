import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

plt.rcParams['figure.figsize'] = [10, 10]


def pointy_hex_to_pixel(axial_coords, radius):
    x = radius * (math.sqrt(3) * axial_coords[0]  +  math.sqrt(3)/2 * axial_coords[1])
    y = radius * (3./2 * axial_coords[1])
    return (x, y)

def pixel_to_pointy_hex(point, radius):
    q = int((math.sqrt(3)/3 * point[0]  -  1./3 * point[1]) / radius)
    r = int((2./3 * point[1]) / radius)
    return (q, r)

def flat_hex_to_pixel(axial_coords, radius):
    x = radius * (3./2 * axial_coords[0])
    y = radius * (math.sqrt(3)/2 * axial_coords[0]  +  math.sqrt(3) * axial_coords[1])
    return (x, y)

def pixel_to_flat_hex(point, radius):
    q = int(( 2./3 * point[0]) / radius)
    r = int((-1./3 * point[0]  +  math.sqrt(3)/3 * point[1]) / radius)
    return (q, r)

def diverging_red_white_green(n):
    """
    n colors from red → white → green.
    """
    half = n // 2
    
    # red → white
    r1 = np.linspace(1, 1, half)
    g1 = np.linspace(0, 1, half)
    b1 = np.linspace(0, 1, half)

    # white → green
    r2 = np.linspace(1, 0, n - half)
    g2 = np.linspace(1, 1, n - half)
    b2 = np.linspace(1, 0, n - half)

    r = np.concatenate([r1, r2])
    g = np.concatenate([g1, g2])
    b = np.concatenate([b1, b2])

    return list(zip(r, g, b))

def white_to_blue(n):
    """
    n colors from white → dark blue.
    """
    # White (1,1,1)
    r_start, g_start, b_start = 1.0, 1.0, 1.0
    
    # Dark blue (0,0,0.4)   # adjust as needed
    r_end, g_end, b_end = 0.0, 0.0, 0.4
    
    r = np.linspace(r_start, r_end, n)
    g = np.linspace(g_start, g_end, n)
    b = np.linspace(b_start, b_end, n)
    
    return list(zip(r, g, b))


def plot_hex(loc, color, alphas, radius, fig, ax, orientation = "pointy", text_labels = False, highlight_hex = None):

    # "option": function which executes the option
    orientations = {
        "flat": plot_hex_flat_top,
        "pointy": plot_hex_pointy_top
    }
    if orientation not in orientations.keys():
        raise ValueError(f'Invalid orientation. Valid orientations include {orientations.keys()}')
    orientations[orientation](loc, color, alphas, radius, fig, ax, text_labels, highlight_hex)

def plot_hex_flat_top(locations, colors, alphas, radius, fig, ax, text_labels, highlight_hex = None):
    # loc: list of hex locations in cube coordinates
    for loc, c, a in zip(locations, colors, alphas):
        if highlight_hex is not None and loc in highlight_hex.keys():
            c = highlight_hex[loc]
        x, y = flat_hex_to_pixel(loc, radius)
        hex = RegularPolygon((x, y), numVertices=6, radius=radius, 
                                 orientation=np.radians(30), 
                                 facecolor=c, alpha=a, edgecolor='k')
        ax.add_patch(hex)
        label_offset = radius * 0.5
        q,r = loc
        s = -q-r
        if text_labels:
            ax.text(x+label_offset, y, str(q), ha='center', va='center', size=14, c="red")
            ax.text(x-label_offset, y+label_offset, str(r), ha='center', va='center', size=14, c="green")
            ax.text(x-label_offset, y-label_offset, str(s), ha='center', va='center', size=14, c="blue")

def plot_hex_pointy_top(locations, colors, alphas, radius, fig, ax, text_labels, highlight_hex):
    for loc, c, a in zip(locations, colors, alphas):
        if highlight_hex is not None and loc in highlight_hex.keys():
            c = highlight_hex[loc]
        x, y = pointy_hex_to_pixel(loc, radius)
        hex = RegularPolygon((x, y), numVertices=6, radius=radius, 
                                 orientation=np.radians(0), 
                                 facecolor=c, alpha=a, edgecolor='k')
        ax.add_patch(hex)
        label_offset = radius * 0.5
        q,r = loc
        s = -q-r
        if text_labels:
            ax.text(x+label_offset, y-label_offset, str(q), ha='center', va='center', size=14, c="red")
            ax.text(x, y+label_offset, str(r), ha='center', va='center', size=14, c="green")
            ax.text(x-label_offset, y-label_offset, str(s), ha='center', va='center', size=14, c="blue")
            # ax.text(x, y, f'({z})', ha='center', va='center', size=14, c="black")

def plot_map(hexes, obstacles_locs, agent_start_loc, goal_loc, max_radius, hex_size, velocities=None, h_values=None, plt_filename = None):
    axis_range = math.ceil((1+max_radius) * hex_size * 1.86)
    axis_x_range = [-axis_range, axis_range]
    axis_y_range = [-axis_range, axis_range]
    
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlim(axis_x_range)
    ax.set_ylim(axis_y_range)
    ax.axis('off')
    
    obstacle_idx = [1 if v in obstacles_locs else 0 for v in hexes]
    agent_idx = hexes.index(agent_start_loc)
    goal_idx =  hexes.index(goal_loc)
    
    colors = ["black" if c else "white" for c in obstacle_idx]
    colors[agent_idx] = "blue"
    colors[goal_idx] = "green"

    alphas = [1]*len(hexes)    
    
    plot_hex(hexes, colors, alphas, hex_size, fig, ax, text_labels=False)    
    if plt_filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")


def plot_problem(problem, solution, h_values = None, plt_title=None, plt_filename = None):

    # Get the solution path from the problem, including velocities:
    current_node = solution    
    solution_path = []
    while current_node.parent is not None:
        v = current_node.state
        solution_path.append(v)
        current_node = current_node.parent

    # Store the velocities in a dictionary keyed by the hex coordinates
    velocities = {k:v for k,v in solution_path}

    max_velocity = max([s[1][0] for s in solution_path])

    axis_range = math.ceil((1+problem.hex_radius) * problem.hex_size * 1.66)
    axis_x_range = [-axis_range, axis_range]
    axis_y_range = [-axis_range, axis_range]
    
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlim(axis_x_range)
    ax.set_ylim(axis_y_range)
    ax.axis('off')
    
    if plt_title is not None: plt.title(plt_title)

    obstacle_idx = [1 if v in problem.obstacle_map else 0 for v in problem.hex_map]
    agent_idx = problem.hex_map.index(problem.root.state[0])
    goal_idx =  problem.hex_map.index(problem.goal_loc)
    path_idx = [1 if v in [s[0] for s in solution_path] else 0 for v in problem.hex_map]
    
    
    alphas_idx = [velocities[v][0] / max_velocity  if v in velocities else 1 for v in problem.hex_map]
    
    colors = ["black" if c else "white" for c in obstacle_idx]
    colors = ["red" if c else colors[i] for i, c in enumerate(path_idx)]
    colors[agent_idx] = "blue"
    colors[goal_idx] = "green"
    
    
    plot_hex(problem.hex_map, colors, alphas_idx, problem.hex_size, fig, ax, text_labels=False)

def plot_q_values(problem, q_values, q_tracker, plt_title=None, plt_counts=False, text_labels=False, plt_filename = None):

    axis_range = math.ceil((1+problem.hex_radius) * problem.hex_size * 1.66)
    axis_x_range = [-axis_range, axis_range]
    axis_y_range = [-axis_range, axis_range]
    
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlim(axis_x_range)
    ax.set_ylim(axis_y_range)
    ax.axis('off')
    
    if plt_title is not None: plt.title(plt_title, fontsize=20)
    
    obstacle_idx = [1 if v in problem.obstacle_map else 0 for v in problem.hex_map]
    agent_idx = problem.hex_map.index(problem.root.state[0])
    goal_idx =  problem.hex_map.index(problem.goal_loc)
    colors = ["black" if c else "white" for c in obstacle_idx]

    q_avgs = {s:sum([q_values[(s, problem.goal_loc, d)] for d in list(range(6))])/6 # Average the 6 directional h values for each cell
        for s in problem.hex_map} # for all hexes in the map
    max_q = max(q_avgs.values())

    q_counts = {s:sum([q_tracker[(s, problem.goal_loc, d)] for d in list(range(6))])/6
                for s in problem.hex_map}
    max_q_count = max(q_counts.values())
    scale_size = 100
    color_scale = diverging_red_white_green(scale_size)
    color_scale2 = white_to_blue(scale_size)
    if plt_counts:
        colors = ["black" if v in problem.obstacle_map else color_scale2[int((scale_size-1) * q_counts[v] / max_q_count)] for v in problem.hex_map]
    else:
        colors = ["black" if v in problem.obstacle_map else color_scale[-int((scale_size-1) * q_avgs[v] / max_q)] for v in problem.hex_map]
    
    # min_alpha = 0.05
    # alphas_idx = [1 if v in problem.obstacle_map else  max(min_alpha, q_counts[v] / max_q_count) for v in problem.hex_map]
    alphas_idx = [1 for v in problem.hex_map]

    
    
    # colors[agent_idx] = "blue"
    colors[goal_idx] = "magenta"
    plot_hex(problem.hex_map, colors, alphas_idx, problem.hex_size, fig, ax, text_labels=text_labels)

    if plt_filename is not None:
        plt.savefig(plt_filename, dpi=300, bbox_inches="tight")

def plot_benchmarks(depths, states, nodes=None):

    b = 1.8

    exp_ref = [pow(b, d) for d in depths]
    print(exp_ref)
    
    lin_ref = [d* b for d in depths]
    print(lin_ref)

    plt.figure(figsize = (12,6))
    plt.semilogy(depths, states, color='purple', linestyle='dashed', marker='o', 
             markerfacecolor='yellow', markersize=10, label='States expanded')
    plt.semilogy(depths, nodes, color='orange', linestyle='solid', marker='s', 
             markerfacecolor='blue', markersize=10, label='Nodes generated')
    plt.semilogy(sorted(depths), sorted(exp_ref), color='black', linestyle='solid', marker='.', 
             markerfacecolor='black', markersize=10, label='O(b^d)')
    plt.semilogy(sorted(depths), sorted(lin_ref), color='black', linestyle='solid', marker='.', 
             markerfacecolor='black', markersize=10, label='O(b*d)')
    plt.legend()
    plt.title('States expanded by depth')
    plt.xlabel('Depth')
    plt.ylabel('States Expanded')

def plot_benchmark_comparison(depths, nodes_h, nodes_a):
    r = range(10, max(depths))
    
    b = 1.8

    exp_ref = [pow(b, d) for d in r][:20]
    #print(exp_ref)

    nlogn_ref = [b * d * math.log(d) for d in r]
    print(nlogn_ref)
    quad_ref = [b * d * d for d in r]
    print(quad_ref)
    lin_ref = [d* b for d in r]
    #print(lin_ref)

    plt.figure(figsize = (12,6))
    plt.semilogy(depths, nodes_h, color='purple', linestyle='solid', marker='o', 
             markerfacecolor='yellow', markersize=5, label='Nodes generated: H*')
    plt.semilogy(depths, nodes_a, color='orange', linestyle='solid', marker='s', 
             markerfacecolor='blue', markersize=5, label='Nodes generated: A*', alpha=0.7)
    plt.semilogy(r[:20], exp_ref, color='gray', linestyle='dashed', label='O(2^n)')
    plt.semilogy(r, quad_ref, color='gray', linestyle='solid', label='O(n^2)')
    plt.legend()
    plt.title('Nodes generated by depth')
    plt.xlabel('Depth')
    plt.ylabel('Nodes generated')

def plot_benchmark_comparison_old(depths, nodes_h, nodes_a):

    b = 1.8

    exp_ref = [pow(b, d) for d in depths]
    print(exp_ref)
    
    lin_ref = [d* b for d in depths]
    print(lin_ref)

    plt.figure(figsize = (12,6))
    plt.semilogy(depths, states_h, color='purple', linestyle='dashed', marker='o', 
             markerfacecolor='yellow', markersize=10, label='States expanded: H* (Time to goal heuristic')
    plt.semilogy(depths, states_a, color='orange', linestyle='solid', marker='s', 
             markerfacecolor='blue', markersize=10, label='Nodes generated: A* (Manhattan distance to goal heuristic)')
    plt.semilogy(depths, nodes_h, color='red', linestyle='dashed', marker='o', 
             markerfacecolor='white', markersize=10, label='States expanded: H* (Time to goal heuristic')
    plt.semilogy(depths, nodes_a, color='green', linestyle='solid', marker='s', 
             markerfacecolor='black', markersize=10, label='Nodes generated: A* (Manhattan distance to goal heuristic)')
    plt.legend()
    plt.title('States expanded and Nodes Generated by depth')
    plt.xlabel('Depth')
    plt.ylabel('N')

def plot_eff_branching_factor(depths, branching):
    plt.figure(figsize = (12,6))
    colors = ['red', 'blue', 'green']
    markers = ['o','s','.']
    for h,c,m in zip(branching, colors, markers):
        depths, branches = zip(*sorted(zip(depths, branching[h])))
        plt.plot(depths, branches, color=c, linestyle='dashed', marker=m, 
             markerfacecolor=c, markersize=10, label=h)
    plt.legend()
    plt.title('Effective Branching Factor by Depth')
    plt.xlabel('Depth')
    plt.ylabel('Effective Branching Factor')
