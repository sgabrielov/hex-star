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



DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    ( 1,  0),  # 0: E
    ( 1, -1),  # 1: SE
    ( 0, -1),  # 2: SW
    (-1,  0),  # 3: W
    (-1,  1),  # 4: NW
    ( 0,  1),  # 5: NE
)



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
        for dq, dr in DIRECTIONS:
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

def get_direction(prev: HexCoord, nxt: HexCoord) -> int:
    """
    Return the direction index (0..5) of the unit step from `prev` to `nxt`,
    according to the DIRECTIONS tuple.

    Raises:
        ValueError: if `nxt` is not an immediate neighbor of `prev` (i.e., the
                    delta doesn't match any unit direction).
    """
    dq = nxt.q - prev.q
    dr = nxt.r - prev.r
    try:
        return DIRECTIONS.index((dq, dr))
    except ValueError as e:
        raise ValueError(
            f"{prev} -> {nxt} is not a single-step neighbor; delta=({dq},{dr})"
        ) from e


def get_path_direction(prev: HexCoord, current: HexCoord, nxt: HexCoord) -> float:
    """
    Returns the path direction *at* `current` as a float index into DIRECTIONS,
    allowing half-step indices (e.g., 0.5) for zigzag straight-line approximations.

    Semantics
    ---------
    - Straight (same incoming and outgoing direction):
        return i  (i in {0,1,2,3,4,5})
    - Zigzag (adjacent unit directions, i.e., ±1 mod 6):
        return i ± 0.5  (midpoint between the two indices on the circle)
    - Turn (120°/180° change, i.e., ±2 or 3 mod 6):
        return i_out  (use the new, outgoing direction at `current`)

    Raises:
        ValueError: if any pair (prev,current) or (current,nxt) is not single-step adjacent.
    """
    if current == prev or current == nxt:
        raise ValueError("`current` must differ from `prev` and `nxt`")

    # Get unit direction indices for the incoming and outgoing steps.
    i_in = get_direction(prev, current)      # prev -> current
    i_out = get_direction(current, nxt)      # current -> nxt

    # Straight: keep the same integer index
    if i_in == i_out:
        return float(i_out)

    # Work on the circle Z_6
    diff = (i_out - i_in) % 6  # in {1,2,3,4,5}

    # Zigzag (60° left/right): midpoint between adjacent directions.
    # diff == 1 means rotate +60°; diff == 5 means rotate -60° (equivalently +300°).
    if diff == 1:
        # midpoint forward (i + 0.5), wrap to [0,6)
        return (i_in + 0.5) % 6.0
    if diff == 5:
        # midpoint backward (i - 0.5) == (i + 5.5) mod 6
        return (i_in - 0.5) % 6.0

    # Sharp turn (120°) or U-turn (180°): use outgoing heading (integer index).
    # diff == 2 or 4 -> 120°, diff == 3 -> 180° (backtrack)
    return float(i_out)



def get_neighbors_at_radius(center: "HexCoord", radius: int) -> Set["HexCoord"]:
    if radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if radius == 0:
        return {center}

    # start at the 'NE corner' (index 5) or any fixed corner; consistency matters more than which corner
    start = add_coords(center, (radius * DIRECTIONS[5][0], radius * DIRECTIONS[5][1]))
    ring: Set["HexCoord"] = set()
    curr = start

    for side in range(6):
        dq, dr = DIRECTIONS[side]  # walk sides in the same order
        for _ in range(radius):
            ring.add(curr)
            curr = add_coords(curr, (dq, dr))
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

def hex_step_distance(hex_size: float) -> float:
    """Center-to-center distance for one axial step (all 6 neighbors)."""
    return math.sqrt(3.0) * hex_size  # pointy-top tiling


def idx_to_step(i: int) -> Tuple[int, int]:
    return DIRECTIONS[i % 6]


# --- Obstacle map builders (extensible) --------------------------------------
from typing import Dict, Set, Iterable, Optional, List, Tuple, Callable
import random
import math

# If HexCoord is defined above in this file:
# @dataclass(frozen=True)
# class HexCoord:
#     q: int
#     r: int

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def hex_disk(center: "HexCoord", radius: int) -> List["HexCoord"]:
    """
    Return all axial hex coordinates within (<=) the given radius from `center`.
    This is the standard axial 'disk' (not just the ring).

    Notes
    -----
    - radius == 0 -> [center]
    - Uses the axial constraints to stay in the hex-shaped bounds.

    Based on the common axial iteration pattern:
      for dq in [-r..+r]:
        for dr in [max(-r, -dq-r) .. min(+r, -dq+r)]:
            (q,r) = (cq + dq, cr + dr)
    """
    cq, cr = center.q, center.r
    cells: List["HexCoord"] = []
    for dq in range(-radius, radius + 1):
        rmin = max(-radius, -dq - radius)
        rmax = min(radius,  -dq + radius)
        for dr in range(rmin, rmax + 1):
            cells.append(HexCoord(cq + dq, cr + dr))
    return cells


def hex_step_distance(hex_size: float) -> float:
    """
    Center-to-center distance for one axial step on a pointy-top hex grid.
    """
    return math.sqrt(3.0) * hex_size


# -----------------------------------------------------------------------------
# Obstacle generator registry
# -----------------------------------------------------------------------------
# Signature each generator should follow:
#   gen(center, radius, *, rng, exclude, **kwargs) -> Set[HexCoord]
ObstacleGen = Callable[[ "HexCoord", int], Set["HexCoord"]]

# We'll maintain a registry mapping names -> callables that accept (center, radius, *, rng, exclude, **kwargs)
_OBSTACLE_GENERATORS: Dict[str, Callable[..., Set["HexCoord"]]] = {}

def register_obstacle_generator(name: str, func: Callable[..., Set["HexCoord"]]) -> None:
    """
    Register a new obstacle generator under a short name.

    Each generator must have the signature:
        func(center: HexCoord, radius: int, *, rng: random.Random, exclude: Set[HexCoord], **kwargs) -> Set[HexCoord]
    """
    _OBSTACLE_GENERATORS[name] = func


# -----------------------------------------------------------------------------
# Generator #1: random_n  (uniform random obstacles in the disk)
# -----------------------------------------------------------------------------
def gen_random_obstacles(
    center: "HexCoord",
    radius: int,
    *,
    rng: random.Random,
    exclude: Iterable["HexCoord"] = (),
    n: int,
) -> Set["HexCoord"]:
    """
    Place `n` random obstacles uniformly within the hex disk (<= radius),
    excluding any coordinates passed via `exclude`.

    Parameters
    ----------
    center : HexCoord
        Disk center.
    radius : int
        Disk radius in hex steps.
    rng : random.Random
        RNG instance (for reproducibility).
    exclude : Iterable[HexCoord]
        Coordinates to keep free (e.g., start, goal, reserved path).
    n : int
        Number of obstacles to place (actual placed count may be less if space is limited).

    Returns
    -------
    Set[HexCoord]
        A set with up to `n` random blocked cells.
    """
    excl: Set["HexCoord"] = set(exclude) if exclude else set()
    candidates = [c for c in hex_disk(center, radius) if c not in excl]
    if not candidates or n <= 0:
        return set()

    k = min(n, len(candidates))
    # random.Random.sample requires a sequence
    chosen = rng.sample(candidates, k)
    return set(chosen)


# Register the first generator
register_obstacle_generator("random_n", gen_random_obstacles)


from typing import Iterable, Set
# assumes HexCoord, hex_disk (or _hex_disk), and register_obstacle_generator are already defined

# -----------------------------------------------------------------------------
# Generator #2: fill all  (fill all hexes as obstacles in radius)
# -----------------------------------------------------------------------------
def gen_fill_all(
    center: "HexCoord",
    radius: int,
    *,
    rng,                              # required by the registry; not used here
    exclude: Iterable["HexCoord"] = (),
) -> Set["HexCoord"]:
    """
    Fill the entire allowed disk with cells.

    Intended use:
    - Use with op="add" to make the whole map blocked.
    - Then call other generators with op="carve" (e.g., 'track') to open corridors.

    Parameters
    ----------
    center : HexCoord
        Disk center.
    radius : int
        Disk radius in hex steps.
    rng : random.Random
        Unused; present for signature consistency.
    exclude : Iterable[HexCoord]
        Cells to keep free (e.g., start/goal or reserved anchors).

    Returns
    -------
    Set[HexCoord]
        All cells in the (center, radius) disk, minus 'exclude'.
    """
    # If you already have _hex_disk(center, radius), you can use that instead of hex_disk(...)
    all_cells = set(hex_disk(center, radius))
    return all_cells.difference(set(exclude or []))

# Register it
register_obstacle_generator("fill_all", gen_fill_all)

# -----------------------------------------------------------------------------
# Generator #2: track  (Build a road from start to goal)
# -----------------------------------------------------------------------------

# --- Track generator: 60° turns only, no self-intersection -----------------------
def gen_track_road(
    center: "HexCoord",
    radius: int,
    *,
    rng: random.Random,
    exclude: Iterable["HexCoord"] = (),
    start: "HexCoord",
    goal: "HexCoord",
    # Enforce at least this many 60° turns before connecting to goal
    min_turns: int = 4,
    # Back-compat alias if old code passes "turns"
    turns: Optional[int] = None,
    # Corridor thickness (hex radius): 0 => single-cell line
    thickness: int = 1,
    # Random straight segment length between 60° turns during turn-building phase
    segment_len_min: int = 2,
    segment_len_max: int = 6,
    # Safety caps
    max_steps_build: int = 10000,
    max_steps_connect: int = 10000,
) -> Set["HexCoord"]:
    """
    Build a 'track' (road corridor) from `start` to `goal` that:
      - uses only 60° turns (heading changes of ±1 on the 0..5 ring),
      - never self-intersects (centerline cells are all unique),
      - enforces at least `min_turns` heading changes before attempting to connect to `goal`.

    The route is *not* shortest; it is composed of random-length straight segments
    with 60° direction changes. After achieving `min_turns`, a constrained connection
    phase (using only keep/±60° per step) finishes the path.

    Returns a set of cells to be *carved* (kept free). Use with op="carve".
    """
    # --- parameter normalization ---
    if turns is not None:
        min_turns = turns
    min_turns = max(0, int(min_turns))
    Lmin = max(1, int(segment_len_min))
    Lmax = max(Lmin, int(segment_len_max))
    pad  = max(0, int(thickness))

    # --- bounds & trivial cases ---
    allowed = _hex_disk(center, radius)
    if start not in allowed or goal not in allowed:
        return set()  # endpoints outside bounds; nothing to carve

    if start == goal:
        return _hex_disk(start, pad) & allowed

    # --- centerline construction with constraints ---
    centerline: list["HexCoord"] = [start]
    visited: Set["HexCoord"] = {start}   # prevent self-intersection

    # Pick an initial heading (0..5). If you prefer biasing toward goal, you can
    # seed this with a direction that reduces distance from start.
    heading = rng.randrange(6)
    turns_done = 0
    steps_done = 0

    # ------------------
    # Phase 1: build at least `min_turns` 60° turns
    # ------------------
    while turns_done < min_turns and steps_done < max_steps_build:
        # Walk a straight segment of random length L in the *current* heading.
        L = rng.randint(Lmin, Lmax)
        for _ in range(L):
            nxt = _step(centerline[-1], heading)

            # If next step violates bounds or revisits a cell, try steering ±60°
            if nxt not in allowed or nxt in visited:
                candidates = [_dir_left(heading), _dir_right(heading)]
                rng.shuffle(candidates)  # randomize left/right preference
                chosen = None
                for h2 in candidates:
                    t = _step(centerline[-1], h2)
                    if t in allowed and t not in visited:
                        chosen = h2
                        nxt = t
                        break
                if chosen is None:
                    # Can't proceed further in this phase; give up building more here
                    break
                # We turned by ±60° (compliant) and used a step; count it as a turn
                heading = chosen
                turns_done += 1

            # Commit the step
            centerline.append(nxt)
            visited.add(nxt)
            steps_done += 1

            if steps_done >= max_steps_build:
                break

        if steps_done >= max_steps_build:
            break

        # Enforce a 60° turn between segments (if we can move at least 1 step)
        # Choose left or right randomly; only ±1 is allowed
        candidates = [_dir_left(heading), _dir_right(heading)]
        rng.shuffle(candidates)
        turned = False
        for h2 in candidates:
            # Turning itself doesn't move; we’ll move in Phase 1’s next segment loop.
            # But we also need to ensure the *first* step after turn will be legal.
            t = _step(centerline[-1], h2)
            if t in allowed and t not in visited:
                heading = h2
                turns_done += 1
                turned = True
                break
        # If we couldn't commit to a turn heading now, we’ll let the next loop
        # attempt to steer on the first step anyway (still 60° only).

    # ------------------
    # Phase 2: connect to goal using only keep/±60° per step, no self-intersections
    # ------------------
    it = 0
    while centerline[-1] != goal and it < max_steps_connect:
        it += 1
        cur = centerline[-1]

        # Candidates limited to keep/±60°
        candidates = [heading, _dir_left(heading), _dir_right(heading)]

        # Score candidates: prefer those that reduce distance to goal; otherwise equal distance
        cur_d = _hdist(cur, goal)
        scored: list[tuple[int, int]] = []  # (priority, dir)
        for h2 in candidates:
            nxt = _step(cur, h2)
            if nxt not in allowed or nxt in visited:
                continue  # must stay in bounds and avoid self-intersection
            d = _hdist(nxt, goal)
            if d < cur_d:
                scored.append((0, h2))   # best: reduces distance
            elif d == cur_d:
                scored.append((1, h2))   # neutral: equal distance
            else:
                scored.append((2, h2))   # worse: increases distance (allowed if needed)

        if not scored:
            # Stuck: try a gentle random wobble (still keep/±60°), but we already filtered those
            break

        # Pick a candidate with the best priority, break ties randomly
        best_pri = min(p for p, _ in scored)
        options  = [d for p, d in scored if p == best_pri]
        rng.shuffle(options)
        chosen = options[0]

        # Count a turn if we changed heading by ±1
        if chosen != heading:
            # Because candidates are only keep/±60°, any change is a 60° turn
            turns_done += 1

        heading = chosen
        nxt = _step(cur, heading)
        centerline.append(nxt)
        visited.add(nxt)

    # Attempt a final hop if we are adjacent to goal and haven’t stepped on it yet
    if centerline[-1] != goal:
        cur = centerline[-1]
        for i in range(6):
            if _step(cur, i) == goal and goal in allowed and goal not in visited:
                # Only accept if i is keep/±60° relative to current heading
                if i in (heading, _dir_left(heading), _dir_right(heading)):
                    centerline.append(goal)
                    visited.add(goal)
                break

    # --- Thicken the centerline into a corridor and clip to the allowed disk ---
    road_set: Set["HexCoord"] = set()
    if pad == 0:
        for c in centerline:
            if c in allowed:
                road_set.add(c)
    else:
        for c in centerline:
            road_set |= (_hex_disk(c, pad) & allowed)

    # Ensure endpoints included (thickened)
    road_set |= (_hex_disk(start, pad) & allowed)
    road_set |= (_hex_disk(goal,  pad) & allowed)

    return road_set


# Register
register_obstacle_generator("track", gen_track_road)


# -----------------------------------------------------------------------------
# Main builder: call a sequence of generator "types" and merge their outputs
# -----------------------------------------------------------------------------


def build_obstacle_map(
    center: "HexCoord",
    radius: int,
    *,
    types: Iterable[tuple] = (),
    exclude: Iterable["HexCoord"] = (),
    seed: Optional[int] = None,
) -> Dict["HexCoord", bool]:
    """
    Build an obstacle map (Dict[HexCoord, True]) by applying a sequence of
    named obstacle generators.

    Each entry in `types` can be:
        (name, params)                    # op defaults to "add"
      or (name, params, op)               # op in {"add","carve"}

    Or put "op" into params: {"op": "carve", ...} (we will strip it).
    """
    rng = random.Random(seed)
    keep_free: Set["HexCoord"] = set(exclude) if exclude else set()
    blocked: Set["HexCoord"] = set()

    for entry in types:
        # Normalize (name, params, op)
        if len(entry) == 2:
            name, params = entry
            op = (params or {}).get("op", "add")
        elif len(entry) == 3:
            name, params, op = entry
        else:
            raise ValueError("Each 'types' entry must be (name, params) or (name, params, op)")

        if name not in _OBSTACLE_GENERATORS:
            raise KeyError(f"Unknown obstacle generator '{name}'. "
                           f"Available: {sorted(_OBSTACLE_GENERATORS.keys())}")

        gen = _OBSTACLE_GENERATORS[name]

        # --- DO NOT forward 'op' to the generator ---
        params = dict(params or {})
        params.pop("op", None)

        # Respect current exclusions and already-blocked cells
        local_exclude = keep_free.union(blocked)

        produced = gen(center, radius, rng=rng, exclude=local_exclude, **params)

        if op == "add":
            blocked |= produced
        elif op == "carve":
            blocked -= produced
            keep_free |= produced
        else:
            raise ValueError(f"Unsupported op='{op}'. Use 'add' or 'carve'.")

    return {c: True for c in blocked}


# -----------------------------------------------------------------------------
# Misc Helper Functions
# -----------------------------------------------------------------------------
# --- Utilities ---------------------------------------------------------------
def _hdist(a: "HexCoord", b: "HexCoord") -> int:
    """Axial hex Manhattan distance (# of steps)."""
    dq = abs(a.q - b.q)
    dr = abs(a.r - b.r)
    ds = abs((-a.q - a.r) - (-b.q - b.r))
    return (dq + dr + ds) // 2

def _hex_disk(center: "HexCoord", r: int) -> Set["HexCoord"]:
    """Closed disk (<= r) around center in axial coords."""
    if r < 0:
        return set()
    cq, cr = center.q, center.r
    out: Set["HexCoord"] = set()
    for dq in range(-r, r + 1):
        rmin = max(-r, -dq - r)
        rmax = min(r,  -dq + r)
        for dr in range(rmin, rmax + 1):
            out.add(HexCoord(cq + dq, cr + dr))
    return out

def _step(a: "HexCoord", dir_idx: int) -> "HexCoord":
    dq, dr = DIRECTIONS[dir_idx % 6]
    return HexCoord(a.q + dq, a.r + dr)

def _dir_left(h: int) -> int:  return (h - 1) % 6  # -60°
def _dir_right(h: int) -> int: return (h + 1) % 6  # +60°


# -----------------------------------------------------------------------------
# Optional convenience: build a HexGrid directly from generators
# -----------------------------------------------------------------------------
def make_grid_with_obstacles(
    hex_size: float,
    center: "HexCoord",
    radius: int,
    *,
    types: Iterable[Tuple[str, Dict]] = (),
    exclude: Iterable["HexCoord"] = (),
    seed: Optional[int] = None,
) -> "HexGrid":
    """
    Convenience: create a HexGrid with obstacles built via the given generator
    sequence. You can still add/remove obstacles later on the grid if needed.
    """
    obstacles = build_obstacle_map(center, radius, types=types, exclude=exclude, seed=seed)
    return HexGrid(hex_size=hex_size, obstacles=obstacles)


