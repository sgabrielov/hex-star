import math
from typing import Iterable, Optional, Set, List, Tuple, Dict, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from collections.abc import Mapping
import numpy as np

from hexgeometry import hdist



from hexgrid import HexCoord, make_grid_with_obstacles, hex_disk


# If HexCoord is defined in your module, import it and remove this local definition.
try:
    HexCoord
except NameError:
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class HexCoord:
        q: int
        r: int


def axial_to_xy(q: int, r: int, hex_size: float) -> Tuple[float, float]:
    """
    Axial (q, r) -> Cartesian (x, y) for a pointy-top hex grid.
    x = sqrt(3) * hex_size * (q + r/2)
    y = 1.5     * hex_size * r
    """
    x = math.sqrt(3) * hex_size * (q + r / 2.0)
    y = 1.5 * hex_size * r
    return x, y


def axial_ring(center: HexCoord, radius: int) -> Iterable[HexCoord]:
    """
    Generate all axial coordinates in the hex 'disk' of given radius around center.
    """
    cq, cr = center.q, center.r
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
            yield HexCoord(cq + dq, cr + dr)

def plot_map_config(
    config: Dict,
    *,
    figsize=(8, 8),
    save_path: Optional[str] = None,
):
    """
    Visualize a scenario config using plot_hex_grid_2.

    Parameters
    ----------
    config : Dict
        Scenario config from generate_scenarios()
    hex_size : float
        Hex size passed to grid + plotting
    show_path : bool
        If True and planner_fn provided, overlay computed path
    figsize : tuple
        Figure size
    save_path : Optional[str]
        If provided, saves image

    Returns
    -------
    ax : matplotlib Axes
    """
    # --- Build grid ---
    grid = make_grid_with_obstacles(
        hex_size=config['hex_size'],
        center=config["center"],
        radius=config["radius"],
        types=config["types"],
        exclude=[config["start"], config["goal"]],
        seed=config.get("seed", None),
    )

    obstacles = set(grid.obstacles.keys())

    # --- Plot ---
    ax = plot_hex_grid_2(
        obstacles=obstacles,
        start=config["start"],
        goal=config["goal"],
        center=config["center"],
        radius=config["radius"],
        hex_size=config['hex_size'],
        figsize=figsize,
        plt_title=f"[{config.get('name','scenario')}]",
        save_path=save_path,
    )

    return ax
    
def plot_hex_grid_2(
    obstacles: Iterable[HexCoord],
    start: HexCoord,
    center: HexCoord,
    radius: int,
    hex_size: float = 1.0,
    goal: Optional[HexCoord] = None,
    path: Optional[Iterable[HexCoord] | Mapping[HexCoord, float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 8),
    # ---- Labels & colors ----
    show_coords: bool = False,
    label_two_lines: bool = True,
    q_color: str = "#2d6cdf",
    r_color: str = "#e67e22",
    facecolor_free: str = "#f2f6ff",
    facecolor_obstacle: str = "#444444",
    facecolor_start: str = "#2d6cdf",
    facecolor_goal: str = "#2eb82e",
    facecolor_path: str = "#e74c3c",
    edgecolor: str = "#555555",
    linewidth: float = 0.8,
    # ---- Axis arrows & legend ----
    draw_axes: bool = True,
    axis_corner: str = "lower left",
    show_axis_legend: bool = True,
    plt_title: str = "",
    # ---- Saving ----
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Fast PolyCollection-based hex grid plotter.

    `path` may be either:
      - Iterable[HexCoord]: all path cells drawn with full opacity
      - Mapping[HexCoord, float]: values are normalized to [0, 1] and used as alpha

    Normalization behavior for dict-valued path:
      - If all values are equal, all path cells get alpha = 1.0
      - Otherwise uses min-max normalization
      - If numerical edge cases prevent max(alpha) from reaching 1.0, all alphas are
        shifted upward by (1 - max_alpha), then clipped to [0, 1]
    """

    def _normalize_path_alpha(
        path_obj: Optional[Iterable[HexCoord] | Mapping[HexCoord, float]]
    ) -> dict[HexCoord, float]:
        if path_obj is None:
            return {}

        # Case 1: dict-like path -> normalize values to alpha
        if isinstance(path_obj, Mapping):
            if not path_obj:
                return {}

            raw = {cell: float(val) for cell, val in path_obj.items()}
            vals = np.array(list(raw.values()), dtype=float)

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            # Constant-valued case: make all visible at full opacity
            if np.isclose(vmax, vmin):
                return {cell: 1.0 for cell in raw}

            # Min-max normalize
            denom = vmax - vmin
            norm = {cell: (val - vmin) / denom for cell, val in raw.items()}

            # If for any reason max isn't 1.0, shift everything upward
            max_norm = max(norm.values()) if norm else 0.0
            if not np.isclose(max_norm, 1.0):
                shift = 1.0 - max_norm
                norm = {cell: val + shift for cell, val in norm.items()}

            # Clamp to [0, 1]
            norm = {cell: float(np.clip(val, 0.0, 1.0)) for cell, val in norm.items()}
            return norm

        # Case 2: set/list/iterable path -> all full opacity
        return {cell: 1.0 for cell in path_obj}

    obstacle_set = set(obstacles or [])
    path_alpha = _normalize_path_alpha(path)
    path_set = set(path_alpha.keys())

    #cells = list(axial_ring(center, radius))
    cells = list(hex_disk(center, radius))
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        ax.cla()

    # --- Precompute hex template (pointy-top) ---
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    unit_hex = np.stack([
        np.cos(angles),
        np.sin(angles)
    ], axis=1)
    
    hex_template = unit_hex * hex_size
    path_template = unit_hex * (hex_size * 0.82)

    polys = []
    colors = []

    path_polys = []
    path_facecolors = []

    xs = []
    ys = []

    path_rgb = mcolors.to_rgb(facecolor_path)

    for cell in cells:
        x, y = axial_to_xy(cell.q, cell.r, hex_size)
        center_xy = np.array([x, y])

        xs.append(x)
        ys.append(y)

        poly = hex_template + center_xy
        polys.append(poly)

        # --- Main cell face color ---
        if cell == start:
            colors.append(facecolor_start)
        elif goal is not None and cell == goal:
            colors.append(facecolor_goal)
        elif cell in obstacle_set:
            colors.append(facecolor_obstacle)
        else:
            colors.append(facecolor_free)

        # --- Path overlay with alpha from dict/set ---
        if cell in path_set and cell not in obstacle_set and cell != start and cell != goal:
            alpha = path_alpha[cell]
            path_polys.append(path_template + center_xy)
            path_facecolors.append((*path_rgb, alpha))

        # --- Optional labels ---
        if show_coords:
            if label_two_lines:
                ax.text(
                    x, y + 0.16 * hex_size,
                    f"q={cell.q}",
                    ha="center", va="center",
                    fontsize=8, color=q_color, zorder=3,
                )
                ax.text(
                    x, y - 0.16 * hex_size,
                    f"r={cell.r}",
                    ha="center", va="center",
                    fontsize=8, color=r_color, zorder=3,
                )
            else:
                ax.text(
                    x, y,
                    f"(q={cell.q}, r={cell.r})",
                    ha="center", va="center",
                    fontsize=8, color=q_color, zorder=3,
                )

    # --- Main grid collection ---
    main_collection = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=1,
    )
    ax.add_collection(main_collection)

    # --- Path overlay collection ---
    if path_polys:
        path_collection = PolyCollection(
            path_polys,
            facecolors=path_facecolors,
            edgecolors="none",
            linewidths=0,
            zorder=2,
        )
        ax.add_collection(path_collection)

    # --- Bounds ---
    if xs and ys:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        margin = 2.0 * hex_size
        ax.set_xlim(xs.min() - margin, xs.max() + margin)
        ax.set_ylim(ys.min() - margin, ys.max() + margin)
    else:
        margin = 2.0 * hex_size
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

    # --- Axes styling ---
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", color="#333333")
    ax.set_ylabel("y", color="#333333")
    ax.set_title(
        f"Hex grid (radius={radius}) centered at ({center.q},{center.r}) {plt_title}"
    )

    # ---- Corner-anchored axis arrows ----
    if draw_axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if axis_corner == "upper right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymax - margin * 0.5
            ha_q, va_q = "right", "top"
            ha_r, va_r = "right", "top"
        elif axis_corner == "upper left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymax - margin * 0.5
            ha_q, va_q = "left", "top"
            ha_r, va_r = "left", "top"
        elif axis_corner == "lower left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymin + margin * 0.5
            ha_q, va_q = "left", "bottom"
            ha_r, va_r = "left", "bottom"
        elif axis_corner == "lower right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymin + margin * 0.5
            ha_q, va_q = "right", "bottom"
            ha_r, va_r = "right", "bottom"
        else:
            raise ValueError(
                "axis_corner must be one of: upper right, upper left, lower left, lower right"
            )

        q_unit_dx, q_unit_dy = axial_to_xy(1, 0, hex_size)
        r_unit_dx, r_unit_dy = axial_to_xy(0, 1, hex_size)

        scale = max(2, radius // 3)

        ax.arrow(
            anchor_x, anchor_y,
            q_unit_dx * scale, q_unit_dy * scale,
            head_width=0.4 * hex_size, head_length=0.6 * hex_size,
            length_includes_head=True,
            fc=q_color, ec=q_color, lw=2.0, zorder=4,
        )
        ax.arrow(
            anchor_x, anchor_y,
            r_unit_dx * scale, r_unit_dy * scale,
            head_width=0.4 * hex_size, head_length=0.6 * hex_size,
            length_includes_head=True,
            fc=r_color, ec=r_color, lw=2.0, zorder=4,
        )

        ax.text(
            anchor_x + q_unit_dx * (scale + 0.2),
            anchor_y + q_unit_dy * (scale + 0.2),
            "q-axis", color=q_color, fontsize=9, fontweight="bold",
            ha=ha_q, va=va_q, zorder=5,
        )
        ax.text(
            anchor_x + r_unit_dx * (scale + 0.2),
            anchor_y + r_unit_dy * (scale + 0.2),
            "r-axis", color=r_color, fontsize=9, fontweight="bold",
            ha=ha_r, va=va_r, zorder=5,
        )

    # ---- Legend ----
    if show_axis_legend:
        legend_handles = [
            Line2D([0], [0], color=q_color, lw=3, label="q-axis (axial)"),
            Line2D([0], [0], color=r_color, lw=3, label="r-axis (axial)"),
            Line2D([0], [0], color=edgecolor, lw=1, label="cell boundary"),
            Line2D([0], [0], color=facecolor_obstacle, lw=0, marker="s", markersize=8,
                   label="obstacle", markerfacecolor=facecolor_obstacle),
            Line2D([0], [0], color=facecolor_start, lw=0, marker="s", markersize=8,
                   label="start", markerfacecolor=facecolor_start),
            Line2D([0], [0], color=facecolor_goal, lw=0, marker="s", markersize=8,
                   label="goal", markerfacecolor=facecolor_goal),
            Line2D([0], [0], color=facecolor_path, lw=0, marker="s", markersize=8,
                   label="path overlay", markerfacecolor=facecolor_path),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)

    if created_fig:
        plt.tight_layout()

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
def plot_hex_grid(
    obstacles: Iterable[HexCoord],
    start: HexCoord,
    center: HexCoord,
    radius: int,
    hex_size: float = 1.0,
    goal: Optional[HexCoord] = None,
    path: Optional[Iterable[HexCoord]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8,8),
    # ---- Labels & colors ----
    show_coords: bool = False,
    label_two_lines: bool = True,
    q_color: str = "#2d6cdf",
    r_color: str = "#e67e22",
    facecolor_free: str = "#f2f6ff",
    facecolor_obstacle: str = "#444444",
    facecolor_start: str = "#2d6cdf",
    facecolor_goal: str = "#2eb82e",
    facecolor_path: str = "#e74c3c",
    edgecolor: str = "#555555",
    linewidth: float = 0.8,
    # ---- Axis arrows & legend ----
    draw_axes: bool = True,
    axis_corner: str = "lower left",
    show_axis_legend: bool = True,
    plt_title: str = "",
    # ---- Saving ----
    save_path: Optional[str] = None,
) -> plt.Axes:

    """
    Plot a pointy-top hex grid around 'start' with given 'radius'.
    - Corner-anchored colored axis arrows (q, r).
    - Optional color-coded labels for each hex's (q, r).

    Parameters
    ----------
    obstacles : Iterable[HexCoord]
        Set/list of blocked axial cells.
    start : HexCoord
        Start location for the Hex Star agent
    center : HexCoord
        Center of the plotted disk.
    radius : int
        Plot radius in hex steps around 'start'.
    hex_size : float
        RegularPolygon center-to-vertex radius.
    goal : Optional[HexCoord]
        Goal cell to highlight.
    path : Optional[Iterable[HexCoord]]
        Path overlay cells.
    ax : Optional[plt.Axes]
        Target axes; new figure/axes is created if None.
    show_coords : bool
        Annotate each cell with q/r values.
    label_two_lines : bool
        Show "q=.." (q_color) and "r=.." (r_color) on separate lines if True.
    q_color, r_color : str
        Colors for axes and labels.
    draw_axes : bool
        Draw corner-anchored axis arrows for q & r.
    axis_corner : str
        One of {"upper right", "upper left", "lower left", "lower right"}.
    show_axis_legend : bool
        Add a legend indicating axis colors.
    """

    # Materialize iterables (protect against exhausted generators)
    obstacle_list: List[HexCoord] = list(obstacles) if obstacles is not None else []
    obstacle_set: Set[HexCoord] = set(obstacle_list)
    path_set: Set[HexCoord] = set(path) if path is not None else set()

    cells: List[HexCoord] = list(axial_ring(center, radius))



    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        # Important when reusing the same axes:
        ax.cla()


    # Draw each cell
    for cell in cells:
        q, r = cell.q, cell.r
        x, y = axial_to_xy(q, r, hex_size)

        fc = facecolor_free
        if cell in obstacle_set:
            fc = facecolor_obstacle
        if cell == start:
            fc = facecolor_start
        elif goal is not None and cell == goal:
            fc = facecolor_goal

        # Main hex
        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_size,
            orientation=math.radians(0),  # pointy-top
            facecolor=fc,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1,
        )
        ax.add_patch(hex_patch)

        # Path overlay
        if cell in path_set and cell not in obstacle_set and cell!= start and cell!=goal:
            path_patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_size * 0.82,
                orientation=math.radians(0),
                facecolor=facecolor_path,
                edgecolor=None,
                linewidth=0,
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(path_patch)

        # Optional q/r label
        if show_coords:
            if label_two_lines:
                ax.text(
                    x, y + 0.16 * hex_size,
                    f"q={q}",
                    ha="center", va="center",
                    fontsize=8, color=q_color, zorder=3,
                )
                ax.text(
                    x, y - 0.16 * hex_size,
                    f"r={r}",
                    ha="center", va="center",
                    fontsize=8, color=r_color, zorder=3,
                )
            else:
                ax.text(
                    x, y,
                    f"(q={q}, r={r})",
                    ha="center", va="center",
                    fontsize=8, color=q_color, zorder=3,
                )

    # Equal aspect & bounds
    ax.set_aspect("equal")
    margin = 2.0 * hex_size
    xs, ys = zip(*(axial_to_xy(c.q, c.r, hex_size) for c in cells)) if cells else ([0], [0])
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Hide ticks; keep labels neutral
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", color="#333333")
    ax.set_ylabel("y", color="#333333")
    ax.set_title(f"Hex grid (radius={radius}) centered at center=({center.q},{center.r}) {plt_title}")

    # ---- Corner-anchored axis arrows ----
    if draw_axes:
        # Compute corner anchor position in data coordinates
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if axis_corner == "upper right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymax - margin * 0.5
            ha_q, va_q = "right", "top"
            ha_r, va_r = "right", "top"
        elif axis_corner == "upper left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymax - margin * 0.5
            ha_q, va_q = "left", "top"
            ha_r, va_r = "left", "top"
        elif axis_corner == "lower left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymin + margin * 0.5
            ha_q, va_q = "left", "bottom"
            ha_r, va_r = "left", "bottom"
        elif axis_corner == "lower right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymin + margin * 0.5
            ha_q, va_q = "right", "bottom"
            ha_r, va_r = "right", "bottom"
        else:
            raise ValueError("axis_corner must be one of: upper right, upper left, lower left, lower right")

        # Determine axis direction vectors using axial unit steps.
        # For pointy-top axial mapping:
        #  - q-axis unit direction: (dq, dr) = (1, 0)
        #  - r-axis unit direction: (dq, dr) = (0, 1)
        # Convert these unit steps to XY deltas using hex_size.
        q_unit_dx, q_unit_dy = axial_to_xy(1, 0, hex_size)
        r_unit_dx, r_unit_dy = axial_to_xy(0, 1, hex_size)

        # These are absolute positions; we need pure direction vectors (deltas).
        # axial_to_xy(1, 0) gives the center of that unit step from (0,0) -> use as vector.
        # Scale the arrows for visibility based on plot radius
        scale = max(2, radius // 3)
        ax.arrow(
            anchor_x, anchor_y,
            q_unit_dx * scale, q_unit_dy * scale,
            head_width=0.4 * hex_size, head_length=0.6 * hex_size,
            length_includes_head=True,
            fc=q_color, ec=q_color, lw=2.0, zorder=4,
        )
        ax.arrow(
            anchor_x, anchor_y,
            r_unit_dx * scale, r_unit_dy * scale,
            head_width=0.4 * hex_size, head_length=0.6 * hex_size,
            length_includes_head=True,
            fc=r_color, ec=r_color, lw=2.0, zorder=4,
        )

        # Axis labels near arrow tips
        ax.text(
            anchor_x + q_unit_dx * (scale + 0.2),
            anchor_y + q_unit_dy * (scale + 0.2),
            "q-axis", color=q_color, fontsize=9, fontweight="bold",
            ha=ha_q, va=va_q, zorder=5,
        )
        ax.text(
            anchor_x + r_unit_dx * (scale + 0.2),
            anchor_y + r_unit_dy * (scale + 0.2),
            "r-axis", color=r_color, fontsize=9, fontweight="bold",
            ha=ha_r, va=va_r, zorder=5,
        )

    # ---- Legend indicating axis colors ----
    if show_axis_legend:
        legend_handles = [
            Line2D([0], [0], color=q_color, lw=3, label="q-axis (axial)"),
            Line2D([0], [0], color=r_color, lw=3, label="r-axis (axial)"),
            Line2D([0], [0], color=edgecolor, lw=1, label="cell boundary"),
            Line2D([0], [0], color=facecolor_obstacle, lw=0, marker='s', markersize=8,
                   label="obstacle", markerfacecolor=facecolor_obstacle),
            Line2D([0], [0], color=facecolor_start, lw=0, marker='s', markersize=8,
                   label="start", markerfacecolor=facecolor_start),
            Line2D([0], [0], color=facecolor_goal, lw=0, marker='s', markersize=8,
                   label="goal", markerfacecolor=facecolor_goal),
            Line2D([0], [0], color=facecolor_path, lw=0, marker='s', markersize=8,
                   label="path overlay", markerfacecolor=facecolor_path),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)

    if created_fig:
        plt.tight_layout()

    # ---- Optional save ----
    if save_path is not None:
        # If we created the figure, save that; otherwise save the parent figure
        fig = ax.figure
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


    return ax

def plot_hex_grid_fast(
    obstacles: Iterable[HexCoord],
    start: HexCoord,
    center: HexCoord,
    radius: int,
    hex_size: float = 1.0,
    goal: Optional[HexCoord] = None,
    path: Optional[Iterable[HexCoord]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8,8),
    # ---- Labels & colors ----
    show_coords: bool = False,
    label_two_lines: bool = True,
    q_color: str = "#2d6cdf",
    r_color: str = "#e67e22",
    facecolor_free: str = "#f2f6ff",
    facecolor_obstacle: str = "#444444",
    facecolor_start: str = "#2d6cdf",
    facecolor_goal: str = "#2eb82e",
    facecolor_path: str = "#e74c3c",
    edgecolor: str = "#555555",
    linewidth: float = 0.8,
    # ---- Axis arrows & legend ----
    draw_axes: bool = True,
    axis_corner: str = "lower left",
    show_axis_legend: bool = True,
    plt_title: str = "",
    # ---- Saving ----
    save_path: Optional[str] = None,
) -> plt.Axes:

    obstacle_set = set(obstacles or [])
    path_set = set(path or [])
    cells = list(axial_ring(center, radius))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        ax.cla()

    # --- Precompute hex template (pointy-top) ---
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_template = np.stack([np.cos(angles), np.sin(angles)], axis=1) * hex_size

    polys = []
    colors = []
    path_polys = []

    xs = []
    ys = []

    for cell in cells:
        x, y = axial_to_xy(cell.q, cell.r, hex_size)
        xs.append(x)
        ys.append(y)

        poly = hex_template + np.array([x, y])
        polys.append(poly)

        # --- Face colors ---
        if cell == start:
            colors.append(facecolor_start)
        elif goal is not None and cell == goal:
            colors.append(facecolor_goal)
        elif cell in obstacle_set:
            colors.append(facecolor_obstacle)
        else:
            colors.append(facecolor_free)

        # --- Path overlay ---
        if cell in path_set and cell not in obstacle_set and cell != start and cell != goal:
            path_polys.append(
                hex_template * (hex_size * 0.82) + np.array([x, y])
            )

        # --- Labels (still per-cell, optional) ---
        if show_coords:
            if label_two_lines:
                ax.text(x, y + 0.16 * hex_size, f"q={cell.q}",
                        ha="center", va="center", fontsize=8, color=q_color)
                ax.text(x, y - 0.16 * hex_size, f"r={cell.r}",
                        ha="center", va="center", fontsize=8, color=r_color)
            else:
                ax.text(x, y, f"(q={cell.q}, r={cell.r})",
                        ha="center", va="center", fontsize=8, color=q_color)

    # --- Draw collections ---
    main_collection = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=edgecolor,
        linewidths=linewidth
    )
    ax.add_collection(main_collection)

    if path_polys:
        path_collection = PolyCollection(
            path_polys,
            facecolors=facecolor_path,
            edgecolors="none",
            alpha=0.85
        )
        ax.add_collection(path_collection)

    # --- Bounds ---
    xs = np.array(xs)
    ys = np.array(ys)
    margin = 2.0 * hex_size

    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", color="#333333")
    ax.set_ylabel("y", color="#333333")
    ax.set_title(f"Hex grid (radius={radius}) centered at ({center.q},{center.r}) {plt_title}")

    # ---- Axis arrows ----
    if draw_axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if axis_corner == "upper right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymax - margin * 0.5
        elif axis_corner == "upper left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymax - margin * 0.5
        elif axis_corner == "lower left":
            anchor_x, anchor_y = xmin + margin * 0.5, ymin + margin * 0.5
        elif axis_corner == "lower right":
            anchor_x, anchor_y = xmax - margin * 0.5, ymin + margin * 0.5
        else:
            raise ValueError("Invalid axis_corner")

        q_dx, q_dy = axial_to_xy(1, 0, hex_size)
        r_dx, r_dy = axial_to_xy(0, 1, hex_size)

        scale = max(2, radius // 3)

        ax.arrow(anchor_x, anchor_y,
                 q_dx * scale, q_dy * scale,
                 head_width=0.4 * hex_size,
                 head_length=0.6 * hex_size,
                 fc=q_color, ec=q_color, lw=2.0)

        ax.arrow(anchor_x, anchor_y,
                 r_dx * scale, r_dy * scale,
                 head_width=0.4 * hex_size,
                 head_length=0.6 * hex_size,
                 fc=r_color, ec=r_color, lw=2.0)

        ax.text(anchor_x + q_dx * (scale + 0.2),
                anchor_y + q_dy * (scale + 0.2),
                "q-axis", color=q_color, fontsize=9, fontweight="bold")

        ax.text(anchor_x + r_dx * (scale + 0.2),
                anchor_y + r_dy * (scale + 0.2),
                "r-axis", color=r_color, fontsize=9, fontweight="bold")

    # ---- Legend ----
    if show_axis_legend:
        legend_handles = [
            Line2D([0], [0], color=q_color, lw=3, label="q-axis (axial)"),
            Line2D([0], [0], color=r_color, lw=3, label="r-axis (axial)"),
            Line2D([0], [0], color=edgecolor, lw=1, label="cell boundary"),
            Line2D([0], [0], color=facecolor_obstacle, marker='s', lw=0, label="obstacle"),
            Line2D([0], [0], color=facecolor_start, marker='s', lw=0, label="start"),
            Line2D([0], [0], color=facecolor_goal, marker='s', lw=0, label="goal"),
            Line2D([0], [0], color=facecolor_path, marker='s', lw=0, label="path"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if created_fig:
        plt.tight_layout()

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax

def draw_path_edges(path, hex_size=1.0, ax=None, color="black", lw=2.0, linestyle="-", zorder=5):
    """Draw lines between consecutive hex centers."""
    if ax is None:
        ax = plt.gca()

    pts = [axial_to_xy(p.q, p.r, hex_size) for p in path]

    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=linestyle, zorder=zorder)


def label_nodes(path, label_map=None, ax=None, hex_size=1.0, color="black"):
    """Annotate nodes with optional labels."""
    if ax is None:
        ax = plt.gca()

    for i, p in enumerate(path):
        x, y = axial_to_xy(p.q, p.r, hex_size)
        label = label_map.get(i, str(i)) if label_map else str(i)
        ax.text(x, y + 0.3, label, ha="center", fontsize=10, color=color, zorder=6)


def draw_direction_arrows(path, ax=None, hex_size=1.0, color="red"):
    """Draw small arrows to show direction."""
    if ax is None:
        ax = plt.gca()

    for p1, p2 in zip(path[:-1], path[1:]):
        x1, y1 = axial_to_xy(p1.q, p1.r, hex_size)
        x2, y2 = axial_to_xy(p2.q, p2.r, hex_size)

        dx = (x2 - x1) * 0.5
        dy = (y2 - y1) * 0.5

        ax.arrow(
            x1, y1,
            dx, dy,
            head_width=0.2,
            head_length=0.3,
            fc=color,
            ec=color,
            length_includes_head=True,
            zorder=6,
        )

def annotate_state(ax, node, text, hex_size=1.0, color="black", y_offset=0.0):
    x, y = axial_to_xy(node.q, node.r, hex_size)
    ax.text(
        x,
        y + y_offset,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=color,
        zorder=7,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

def annotate_heading_vmax(ax, node, theta, vmax, hex_size=1.0, color="black"):
    x, y = axial_to_xy(node.q, node.r, hex_size)

    theta_str = "∅" if theta is None else str(theta)

    ax.text(
        x, y,
        f"θ={theta_str}\nv={vmax:.1f}",
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        zorder=7,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

def annotate_theta_only(ax, node, theta, hex_size=1.0, color="black"):
    x, y = axial_to_xy(node.q, node.r, hex_size)

    theta_str = "∅" if theta is None else str(theta)

    ax.text(
        x, y,
        f"Path Heading={theta_str}",
        ha="center",
        va="center",
        fontsize=12,
        color=color,
        weight="bold",
        zorder=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )


def annotate_radius_only(ax, node, radius, hex_size=1.0, color="black"):
    if radius is None:
        return

    x, y = axial_to_xy(node.q, node.r, hex_size)

    ax.text(
        x, y,
        f"R={radius}",
        ha="center",
        va="center",
        fontsize=12,
        color=color,
        weight="bold",
        zorder=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

def velocity_profile_xy(path: Sequence) -> Tuple[List[float], List[float], List[int]]:
    """
    Returns:
      x_cum: cumulative hex distance traveled (same length as path)
      vels : velocity magnitudes (same length as path)
      segd : per-segment hex distances between consecutive points (len = len(path)-1)
    """
    if not path:
        return [], [], []

    vels = [n.velocity.magnitude for n in path]

    x_cum = [0.0]
    segd = []
    total = 0

    for prev, nxt in zip(path[:-1], path[1:]):
        d = hdist(prev.location, nxt.location)
        segd.append(d)
        total += d
        x_cum.append(float(total))

    return x_cum, vels, segd

def plot_velocity_profile(
    path,
    ax=None,
    marker="o",
    lw=2.0,
    color="#1f77b4",
    title="Velocity vs Path Distance",
    show=True,
    savepath=None,          # <-- NEW: e.g. "vel_profile.png" or Path(...)
    dpi=150,                # <-- NEW: file resolution
    bbox_inches="tight",    # <-- NEW: trims whitespace in saved image
    close=False             # <-- NEW: close figure after saving/showing (useful in batch)
):
    x, v, segd = velocity_profile_xy(path)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    ax.plot(x, v, marker=marker, lw=lw, color=color)
    ax.set_xlabel("Cumulative hex distance traveled (steps)")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Optional: annotate segment distances (can be noisy; keep off by default)
    # for i, d in enumerate(segd, start=1):
    #     ax.text(x[i], v[i], str(d), fontsize=8, ha="left", va="bottom")

    # Layout before save/show so both look good
    if fig is not None:
        fig.tight_layout()

    # Optional save
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)

    # Optional show
    if show:
        plt.show()

    # Optional close (handy for loops / notebooks to avoid too many open figures)
    if close and fig is not None:
        plt.close(fig)

    return ax