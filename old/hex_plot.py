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

def plot_hex(loc, color, radius, fig, ax, orientation = "pointy", text_labels = False, highlight_hex = None):

    # "option": function which executes the option
    orientations = {
        "flat": plot_hex_flat_top,
        "pointy": plot_hex_pointy_top
    }
    if orientation not in orientations.keys():
        raise ValueError(f'Invalid orientation. Valid orientations include {orientations.keys()}')
    orientations[orientation](loc, color, radius, fig, ax, text_labels, highlight_hex)

def plot_hex_flat_top(locations, colors, radius, fig, ax, text_labels, highlight_hex = None):
    # loc: list of hex locations in cube coordinates
    for loc, c in zip(locations, colors):
        if highlight_hex is not None and loc in highlight_hex.keys():
            c = highlight_hex[loc]
        x, y = flat_hex_to_pixel(loc, radius)
        hex = RegularPolygon((x, y), numVertices=6, radius=radius, 
                                 orientation=np.radians(30), 
                                 facecolor=c, alpha=0.6, edgecolor='k')
        ax.add_patch(hex)
        label_offset = radius * 0.5
        q,r = loc
        s = -q-r
        if text_labels:
            ax.text(x+label_offset, y, str(q), ha='center', va='center', size=14, c="red")
            ax.text(x-label_offset, y+label_offset, str(r), ha='center', va='center', size=14, c="green")
            ax.text(x-label_offset, y-label_offset, str(s), ha='center', va='center', size=14, c="blue")

def plot_hex_pointy_top(locations, colors, radius, fig, ax, text_labels, highlight_hex):
    for loc, c in zip(locations, colors):
        if highlight_hex is not None and loc in highlight_hex.keys():
            c = highlight_hex[loc]
        x, y = pointy_hex_to_pixel(loc, radius)
        alpha = 1
        if c == "red":
            alpha = 0.4
        hex = RegularPolygon((x, y), numVertices=6, radius=radius, 
                                 orientation=np.radians(0), 
                                 facecolor=c, alpha=alpha, edgecolor='k')
        ax.add_patch(hex)
        label_offset = radius * 0.5
        q,r = loc
        s = -q-r
        if text_labels:
            ax.text(x+label_offset, y-label_offset, str(q), ha='center', va='center', size=14, c="red")
            ax.text(x, y+label_offset, str(r), ha='center', va='center', size=14, c="green")
            ax.text(x-label_offset, y-label_offset, str(s), ha='center', va='center', size=14, c="blue")
            # ax.text(x, y, f'({z})', ha='center', va='center', size=14, c="black")

def plot_map(hexes, obstacles_locs, agent_start_loc, goal_loc, max_radius, hex_size):
    axis_range = math.ceil((1+max_radius) * hex_size * 1.66)
    axis_x_range = [-axis_range, axis_range]
    axis_y_range = [-axis_range, axis_range]
    
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlim(axis_x_range)
    ax.set_ylim(axis_y_range)
    #ax.axis('off')
    
    obstacle_idx = [1 if v in obstacles_locs else 0 for v in hexes]
    agent_idx = hexes.index(agent_start_loc)
    goal_idx =  hexes.index(goal_loc)
    
    colors = ["black" if c else "white" for c in obstacle_idx]
    colors[agent_idx] = "blue"
    colors[goal_idx] = "green"
    
    
    plot_hex(hexes, colors, hex_size, fig, ax, text_labels=False)    


def plot_problem(problem, solution_path):

    axis_range = math.ceil((1+problem.hex_radius) * problem.hex_size * 1.66)
    axis_x_range = [-axis_range, axis_range]
    axis_y_range = [-axis_range, axis_range]
    
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlim(axis_x_range)
    ax.set_ylim(axis_y_range)
    ax.axis('off')
    
    obstacle_idx = [1 if v in problem.obstacle_map else 0 for v in problem.hex_map]
    agent_idx = problem.hex_map.index(problem.root.state[0])
    goal_idx =  problem.hex_map.index(problem.goal_loc)
    path_idx = [1 if v in solution_path else 0 for v in problem.hex_map]
    
    colors = ["black" if c else "white" for c in obstacle_idx]
    colors = ["red" if c else colors[i] for i, c in enumerate(path_idx)]
    colors[agent_idx] = "blue"
    colors[goal_idx] = "green"
    
    
    plot_hex(problem.hex_map, colors, problem.hex_size, fig, ax, text_labels=False)