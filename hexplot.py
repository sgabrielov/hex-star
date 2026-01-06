import math
from typing import Iterable, Optional, Set, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D

from hexgrid import HexCoord


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


def plot_hex_grid(
    obstacles: Iterable[HexCoord],
    start: HexCoord,
    radius: int,
    hex_size: float = 1.0,
    goal: Optional[HexCoord] = None,
    path: Optional[Iterable[HexCoord]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8,8),
    # ---- Labels & colors ----
    show_coords: bool = False,
    label_two_lines: bool = True,
    q_color: str = "#2d6cdf",   # q-axis blue
    r_color: str = "#e67e22",   # r-axis orange
    facecolor_free: str = "#f2f6ff",
    facecolor_obstacle: str = "#444444",
    facecolor_start: str = "#2d6cdf",
    facecolor_goal: str = "#2eb82e",
    facecolor_path: str = "#e74c3c",
    edgecolor: str = "#555555",
    linewidth: float = 0.8,
    # ---- Axis arrows & legend ----
    draw_axes: bool = True,
    axis_corner: str = "lower left",   # "upper right" | "upper left" | "lower left" | "lower right"
    show_axis_legend: bool = True,
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
    obstacle_set: Set[HexCoord] = set(obstacles) if obstacles is not None else set()
    path_set: Set[HexCoord] = set(path) if path is not None else set()
    cells: List[HexCoord] = list(axial_ring(start, radius))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

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
        if cell in path_set and cell not in obstacle_set:
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
    ax.set_title(f"Hex grid (radius={radius}) centered at start=({start.q},{start.r})")

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

    return ax
