from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set
import math


# ------------------------------
# Core data structures
# ------------------------------

@dataclass(frozen=True)
class HexCoord:
    """
    Axial coordinates (q, r) for hex grid.
    s = -q - r (implicit).
    """
    q: int
    r: int


@dataclass(frozen=True)
class VelocityState:
    """
    Velocity vector at a node, including magnitude and direction.
    direction is an integer encoding from 0-5; None at the start node by convention.
    """
    magnitude: float      # speed
    direction: int        # heading angle in radians


@dataclass
class Node:
    """
    Search node for H* on a hex grid.
    """
    location: HexCoord
    velocity: VelocityState
    parent: Optional["Node"] = None

    # Costs (optional, can be maintained externally)
    g_cost: float = 0.0  # actual travel time accumulated
    h_cost: float = 0.0  # heuristic travel time
    f_cost: float = 0.0  # total estimated cost


# ------------------------------
# Problem / environment model
# ------------------------------

class HexGrid:
    """
    Hexagonal grid environment using axial coordinates.
    Responsible for neighbor generation and obstacle queries.
    """

    def __init__(self, hex_size: float, obstacles: Optional[Dict[HexCoord, bool]] = None) -> None:
        """
        :param hex_size: size parameter for hex cells (distance center-to-center basis).
        :param obstacles: map of HexCoord -> True if blocked.
        """
        self.hex_size = hex_size
        self.obstacles = obstacles or {}
        self.DIRECTIONS: Tuple[Tuple[int, int], ...] = (
            (1, 0),    # East
            (1, -1),   # North-East
            (0, -1),   # North-West
            (-1, 0),   # West
            (-1, 1),   # South-West
            (0, 1),    # South-East
        )

    def is_blocked(self, coord: HexCoord) -> bool:
        """Return True if the cell is blocked or out of bounds."""
        return coord in self.obstacles



    def neighbors(self, coord: HexCoord) -> List[HexCoord]:
        """
        Return the six adjacent axial coordinates (filtering out blocked ones).

        Uses pointy-top axial directions:
            (+1,  0), (+1, -1), (0, -1),
            (-1,  0), (-1, +1), (0, +1)

        Any neighbor for which `is_blocked(nei)` returns True will be excluded.
        """


        q0, r0 = coord.q, coord.r
        result: List[HexCoord] = []
        for dq, dr in self.DIRECTIONS:
            nei = HexCoord(q0 + dq, r0 + dr)
            if not self.is_blocked(nei):
                result.append(nei)
        return result

# ------------------------------
# Hex Geometry Functions
# ------------------------------        

def add_coords(a: HexCoord, step: Tuple[int, int]) -> HexCoord:
    """
    Element-wise addition of axial coordinate and step.
    """
    dq, dr = step
    return HexCoord(a.q+dq, a.r+dr)

def get_neighbors_at_radius(center: "HexCoord", radius: int) -> Set["HexCoord"]:
    """
    Return the set of axial hex coordinates lying exactly at a given hex radius
    (i.e., the cells on the 'ring' around the center at distance = radius).

    This uses axial coordinates for a pointy-top hex grid. The algorithm:
      1) Move from `center` to the starting corner (center + radius * dir0).
      2) Walk the six sides of the hexagon ring, each for `radius` steps,
         using the six axial direction vectors.

    Parameters
    ----------
    center : HexCoord
        The axial center coordinate (q, r).
    radius : int
        Non-negative integer distance. If 0, returns {center}.

    Returns
    -------
    Set[HexCoord]
        The set of coordinates at exactly `radius` steps from `center`.

    Notes
    -----
    - Axial direction deltas (pointy-top):
        E  = ( 1,  0)
        NE = ( 1, -1)
        NW = ( 0, -1)
        W  = (-1,  0)
        SW = (-1,  1)
        SE = ( 0,  1)

    - This method does NOT filter obstacles or bounds; it only computes geometry.
      If you need filtering, post-process the returned set with the `is_blocked`
      or map bounds logic.

    Examples
    --------
    >>> ring0 = get_neighbors_at_radius(HexCoord(0,0), 0)
    >>> ring1 = get_neighbors_at_radius(HexCoord(0,0), 1)
    >>> len(ring0) == 1 and HexCoord(0,0) in ring0
    True
    >>> len(ring1) == 6
    True
    """
    if radius < 0:
        raise ValueError("radius must be a non-negative integer")

    # Radius 0: by definition, the ring is just the center.
    if radius == 0:
        return {center}

    # Six axial directions (pointy-top). Your original list missed (0, 1).
    directions: Tuple[Tuple[int, int], ...] = (
        ( 1,  0),  # E
        ( 1, -1),  # SE
        ( 0, -1),  # SW
        (-1,  0),  # W
        (-1,  1),  # NW
        ( 0,  1),  # NE
    )

    # Start from the "east" corner: center + radius * NE
    current = add_coords(center, (radius * directions[4][0], radius * directions[4][1]))

    ring: Set["HexCoord"] = set()
    # Walk the six sides; each side has exactly `radius` steps
    for dir_dq, dir_dr in directions:
        for _ in range(radius):
            ring.add(current)
            current = add_coords(current, (dir_dq, dir_dr))

    return ring



def axial_to_cube(coord: HexCoord) -> Tuple[int, int, int]:
    """
    Convert axial (q, r) to cube (x, y, z).
    For the standard pointy-top axial convention:
        x = q
        z = r
        y = -x - z = -q - r
    Returns:
        (x, y, z) integers satisfying x + y + z = 0
    """
    x = coord.q
    z = coord.r
    y = -x - z
    return (x, y, z)
