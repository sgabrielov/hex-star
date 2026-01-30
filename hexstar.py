from hexgrid import HexCoord, VelocityState, Node, HexGrid, get_neighbors_at_radius, add_coords, get_direction, get_path_direction, hex_step_distance, idx_to_step, build_obstacle_map
from typing import Iterable, Optional, Set, List, Tuple, Callable, Dict
from queue import PriorityQueue
from itertools import count
import math

class HStarProblem:
    """
    Encapsulates the pathfinding problem for H*:
    - Grid definition
    - Start & goal
    - Robot kinematic limits
    """

    def __init__(
        self,
        grid: HexGrid,
        start: HexCoord,
        goal: HexCoord,
        a_max: float,  # maximum acceleration (for acceleration phase)
        a_min: float,  # maximum deceleration (positive value; used with sign as needed)
        start_v: VelocityState = VelocityState(0, None),   # Start velocity
        goal_v: VelocityState = VelocityState(None, None), # Target goal velocity for BiH*
        ay_window_ms: int = 1000,  # window for lateral acceleration averaging
        collision_radius: int = 0, # How far away an obstacle must be from the agent's location to cause a collision
        search_direction: str = "forward",
    ) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal
        self.a_max = a_max
        self.a_min = a_min
        self.ay_window_ms = ay_window_ms
        self.collision_radius = collision_radius
        self.start_v = start_v
        self.goal_v = goal_v
        self.search_direction = search_direction


    def is_goal(self, location: HexCoord) -> bool:
        """Return True when the agent is exactly at the goal hex."""
        return location == self.goal

    
    def collision_clear(self, coord: HexCoord) -> bool:
        """
        True if 'coord' is collision-free given self.collision_radius.
        Uses grid.is_blocked for the cell itself and all cells within the
        specified hex ring radius around it.
        """
        r = getattr(self, "collision_radius", 0) or 0
        if r <= 0:
            return not self.grid.is_blocked(coord)

        if self.grid.is_blocked(coord):
            return False

        for rad in range(1, r + 1):
            for c in get_neighbors_at_radius(coord, rad):
                if self.grid.is_blocked(c):
                    return False
        return True

    def candidate_dir_indices(self, node: Node) -> list[int]:
        """
        Directions for expansion:
        - If start with unknown heading (parent is None AND node.velocity.direction is None) -> all 6.
        - Else -> 3 directions centered on the *last move heading* (integer):
            keep, +60°, -60° relative to that integer.
        """
        start_unknown_heading = (node.parent is None and node.velocity.direction is None)
        if start_unknown_heading:
            return list(range(6))
    
        # Determine integer base heading from the last move:
        if node.parent is not None:
            base_dir = get_direction(node.parent.location, node.location)  # integer 0..5
        else:
            base_dir = int(node.velocity.direction) % 6
    
        return [base_dir, (base_dir + 1) % 6, (base_dir - 1) % 6]

        
    def actions(self, node: "Node") -> List["HexCoord"]:
        """
        Return only the **new states** (successor locations) that are reachable from `node`,
        honoring the start-unknown-heading rule and collision clearance.
        """
        successors: List[HexCoord] = []
        for di in self.candidate_dir_indices(node):
            dq, dr = idx_to_step(di)
            nxt = add_coords(node.location, (dq, dr))
            if self.collision_clear(nxt):
                successors.append(nxt)
        return successors

    def _effective_step_distance(self, node: "Node", curr: "HexCoord", nxt: "HexCoord") -> float:
        """
        Symmetric per-step travel distance for curr -> nxt:
          - Normal neighbor step:  sqrt(3) * hex_size
          - Zig-zag (adjacent 60°): 1.5 * hex_size
    
        Uses the node’s stored incoming heading when available:
            i_in  = node.velocity.direction      (incoming at `curr`)
            i_out = get_direction(curr, nxt)     (outgoing from `curr`)
        Zig-zag iff (i_out - i_in) % 6 in {1, 5}.
    
        Falls back to parent-based incoming heading only if i_in is None.
        """
    
        d_normal = hex_step_distance(self.grid.hex_size)  # = sqrt(3) * hex_size
        # --- incoming heading from the node’s stored state (preferred) ---
        i_in = node.velocity.direction if node.velocity and node.velocity.direction is not None else None
    
        # Fallback once (root) if no stored heading yet:
        if i_in is None:
            if node.parent is None:
                return d_normal
            try:
                i_in = get_direction(node.parent.location, curr)
            except Exception:
                return d_normal
    
        # Outgoing heading from geometry:
        try:
            i_out = get_direction(curr, nxt)
        except Exception:
            return d_normal
    
        diff = (i_out - i_in) % 6
        return (1.5 * self.grid.hex_size) if diff in (1, 5) else d_normal


    # 
    def result(self, node: "Node", action: "HexCoord") -> "Node":
        curr = node.location
        nxt  = action
        if not self.collision_clear(nxt):
            return node
    
        s_eff = self._effective_step_distance(node, curr, nxt)  # existing
        v_here = float(node.velocity.magnitude or 0.0)
    
        out_dir_idx = get_direction(curr, nxt)
    
        # Provisional kinematics
        if self.search_direction == "reverse":
            v_next_prov = prev_velocity_straight(v_child=v_here, a_max=self.a_max, delta_d=s_eff)
            # Edge time will use upstream entry speed in reverse later
        else:
            v_next_prov = next_velocity_straight(v_parent=v_here, a_max=self.a_max, delta_d=s_eff)
    
        # --- INSERT turning-radius enforcement here ---
        v_parent_adj, v_child_adj = _enforce_turning_backtrack(
            self, node, out_dir_idx, s_eff, v_here, v_next_prov
        )
    
        # Use adjusted speeds for edge time
        if self.search_direction == "reverse":
            t_step = travel_time(u=v_child_adj, a=self.a_max, s=s_eff)  # reverse uses downstream in your code
            v_next = v_child_adj
        else:
            t_step = travel_time(u=v_parent_adj, a=self.a_max, s=s_eff)
            v_next = v_child_adj
    
        child = Node(
            location=nxt,
            velocity=VelocityState(magnitude=v_next, direction=out_dir_idx),
            parent=node
        )
        child.g_cost = node.g_cost + t_step
        child.h_cost = self.heuristic(self, child) if hasattr(self, "heuristic") and callable(self.heuristic) else h_cost_travel_time(self, child)
        child.f_cost = f_cost(child.g_cost, child.h_cost)
        return child
    
        
    




    def action_cost(self, prev: Node, action: HexCoord) -> float:
        """
        Travel time from `prev` to `action` (a neighbor location), using:
            s = √3 * hex_size
            t = (-u + sqrt(u^2 + 2*a*s)) / a
        Note: this uses the parent's speed `u` and problem's `a_max` (acceleration phase).
        """
        d_step = hex_step_distance(self.grid.hex_size)
        u = prev.velocity.magnitude
        a = self.a_max
        return travel_time(u, a, d_step)

    def reverse(
        self,
        goal_speed_mag: float,
        goal_heading_idx: Optional[int] = None,
    ) -> "HStarProblem":
        """
        Build a cost-symmetric reverse problem for BiH*:
          - start <- original goal
          - goal  <- original start
          - start_v <- known velocity at the (original) goal
          - goal_v  <- original start's velocity target (if any)
        Acceleration/deceleration magnitudes remain positive to keep the same
        travel-time model in both directions.

        Parameters
        ----------
        goal_speed_mag : float
            Known |v| at the original goal (now the reversed start).
        goal_heading_idx : Optional[int]
            Optional discrete heading (0..5). If None, reversed search will
            treat the start heading as unknown (allowed to expand all 6).
        """
        # Seed reversed start with the known goal velocity (magnitude; heading optional)
        rev_start_v = VelocityState(magnitude=max(0.0, float(goal_speed_mag)),
                                    direction=goal_heading_idx if goal_heading_idx is not None else None)

        # Mirror problem geometry & limits; keep a_max/a_min positive
        reversed_problem = HStarProblem(
            grid=self.grid,
            start=self.goal,     # swap endpoints
            goal=self.start,
            a_max=self.a_max,    # keep positive magnitudes (do NOT negate)
            a_min=self.a_min,
            start_v=rev_start_v, # start speed at reversed start
            goal_v=self.start_v, # target end-state matches original start
            ay_window_ms=self.ay_window_ms,
            collision_radius=self.collision_radius,
            search_direction="reverse",

        )

        # If you store a callable heuristic on the instance, carry it over
        if hasattr(self, "heuristic") and callable(getattr(self, "heuristic")):
            reversed_problem.heuristic = self.heuristic

        # If you added any effective-distance logic (e.g., zig-zag straightening),
        # keep using the same code paths in result()/action_cost().
        return reversed_problem

            



# ------------------------------
# Geometry & kinematics utilities
# ------------------------------

def hex_manhattan_distance(a: HexCoord, b: HexCoord) -> int:
    """
    Hex Manhattan distance (axial form).
    Returns the number of steps on the hex grid between a and b.
    """
    q1, r1 = (a.q, a.r)
    q2, r2 = (b.q, b.r)

    dq = abs(q1-q2)
    dr = abs(r1-r2)
    ds = abs((q2 + r2) - (q1 + r1))

    return (dq+dr+ds)//2

def convert_direction(direction: int) -> float:
    """
    Return the direction in radians of a direction index input
    Can also be used to calculate the difference of two angles in radians
        by passing the difference of indices as input
    """
    if direction > 5:
        raise ValueError(f'Direction out of bounds: direction={direction}, acceptable range is [0,5]')
    return direction * math.pi / 3  

def detect_turn(node: Node) -> int:
    """
    Return the turn *magnitude* at this node as an index:
        0 -> 0° (straight)
        1 -> 60° (smooth/zigzag)
        2 -> 120° (sharp)
        3 -> 180° (U-turn/backtrack)

    Raises:
        ValueError if parent or grandparent is missing (no turn can be defined).
    """
    if node.parent is None or node.parent.parent is None:
        raise ValueError("Need parent and grandparent to detect a turn")

    gp = node.parent.parent.location
    p  = node.parent.location
    c  = node.location

    i_in  = get_direction(gp, p)       # gp -> p
    i_out = get_direction(p, c)        # p  -> c

    diff = (i_out - i_in) % 6          # 0..5
    # Smallest absolute diff in {0,1,2,3}
    turn_idx = diff if diff <= 3 else 6 - diff
    return turn_idx


# --- Add to hexstar.py (utilities section) ---
from typing import Optional

def _compute_turn_category(node: "Node", out_dir_idx: int) -> int:
    """
    Return turn magnitude category at `node` when taking outgoing dir `out_dir_idx`:
      0 -> straight (0°)
      1 -> gentle (±60°)
      2 -> sharp  (±120°)
      3 -> U-turn (180°)
    Requires node.parent; if missing, returns 0 (no turn).
    """
    if node.parent is None:  # no incoming heading
        return 0
    gp = node.parent.location
    p  = node.location
    # incoming heading index i_in = gp->p, outgoing i_out = p->child
    i_in  = get_direction(gp, p)
    i_out = out_dir_idx
    diff = (i_out - i_in) % 6
    turn_idx = diff if diff <= 3 else 6 - diff
    return turn_idx  # 0..3

def _turn_radius_for_category(hex_size: float, turn_idx: int) -> Optional[float]:
    """
    Map discrete turn category to geometric turning radius r_node (paper §III-B.1.b):
      - straight neighbor step: radius is undefined for pure straight; return None
      - zig-zag (handled elsewhere by s_eff=1.5*size) -> radius ~ size / sqrt(3) per Eq.(5)
      - ±60° turn -> radius ~ size
      - ±120° turn -> radius ~ size / 2
      - 180° U-turn -> treat as very small radius; effectively requires very low speed.
    Adjust if you use a different convention. Returns None if no turn constraint applies.
    """
    if turn_idx == 0:
        return None  # straight; no turn cap
    if turn_idx == 1:
        return hex_size  # 60°
    if turn_idx == 2:
        return hex_size / 2.0  # 120°
    if turn_idx == 3:
        return hex_size / 4.0  # 180° (aggressive cap; can tune)
    return None

def _enforce_turning_backtrack(
    problem: "HStarProblem",
    node: "Node",              # parent (current) node
    out_dir_idx: int,          # heading for edge node->child
    s_eff: float,              # effective step distance for this edge
    v_here: float,             # parent speed before the edge
    v_next_provisional: float  # computed by next/prev_velocity_straight
) -> tuple[float, float]:
    """
    Apply turning-radius enforcement for edge (node -> child):
      - detect if this edge is part of a turn and compute r_node
      - compute vmax_turn from ay_window and r_node
      - clamp v_next to <= vmax_turn
      - if clamped, backtrack and reduce ancestor velocities as needed, honoring a_min,
        so reaching this edge with the clamped velocity is dynamically feasible.
      - return (v_parent_adjusted, v_child_adjusted)

    NOTE: This routine mutates velocity magnitudes stored in nodes along the current branch,
    and recomputes any intermediate edge times if you cache them. If you only store g_cost
    on each Node, recompute g_cost for the child from scratch using the adjusted speeds.
    """
    # 1) Detect turn and compute geometric radius
    turn_cat = _compute_turn_category(node, out_dir_idx)  # 0..3
    r_node = _turn_radius_for_category(problem.grid.hex_size, turn_cat)
    if r_node is None:
        return (v_here, v_next_provisional)  # no turn-induced limit

    # 2) Compute vmax allowed by lateral accel window (Eq. 8)
    # You might have ay_window_ms and a measurement or bound for |ay|. If you don't yet
    # have ay data, set a conservative constant such as problem.ay_max or similar.
    ay = getattr(problem, "ay_max", None)
    if ay is None:
        # Fallback: derive from a_min to keep units coherent; tune as needed.
        ay = problem.a_min
    v_turn_max = math.sqrt(max(ay * r_node, 0.0))

    v_child = min(v_next_provisional, v_turn_max)

    if v_child >= v_next_provisional - 1e-12:
        # No clamping needed; return unchanged
        return (v_here, v_child)

    # 3) Backtrack deceleration through ancestors to make v_child reachable
    # Work backward: ensure arriving at this edge with entry speed v_here' such that
    # after moving s_eff with +a_max we end at v_child in forward search.
    # Or in reverse search, ensure symmetry by using your reverse kinematics.
    a_max = problem.a_max
    direction_mode = getattr(problem, "search_direction", "forward")

    if direction_mode == "reverse":
        # In reverse, parent velocity should be >= child velocity,
        # but here we still need to ensure feasibility given the symmetric model.
        # We'll map to the forward analogy for simplicity.
        pass  # (optional: keep same adjustment path as forward)

    # Desired parent speed so that after +a_max over s_eff, we end at v_child:
    # v_child^2 = v_parent'^2 + 2 a_max s_eff  =>  v_parent' = sqrt(max(v_child^2 - 2 a_max s_eff, 0))
    v_parent_req = math.sqrt(max(v_child*v_child - 2.0*a_max*s_eff, 0.0))

    # If current parent speed is higher than required, try to decelerate earlier using a_min
    if v_here <= v_parent_req + 1e-12:
        return (v_here, v_child)  # already feasible

    # Walk ancestors, reducing speeds so that each step can decelerate to the next.
    # We do not know per-edge distances here; if you store them, use them.
    # Otherwise assume standard hex step distance for ancestor edges.
    hex_d = hex_step_distance(problem.grid.hex_size)

    cur_child_speed = v_parent_req
    cur = node
    while cur.parent is not None and cur.velocity.magnitude > cur_child_speed + 1e-12:
        # Given child target speed `cur_child_speed` at end of this edge,
        # compute allowable parent speed using deceleration a_min over distance hex_d:
        # cur_child_speed^2 = v_parent'^2 - 2 a_min * hex_d  =>  v_parent' = sqrt(cur_child_speed^2 + 2 a_min * hex_d)
        v_parent_cap = math.sqrt(max(cur_child_speed*cur_child_speed + 2.0*problem.a_min*hex_d, 0.0))
        if cur.velocity.magnitude <= v_parent_cap + 1e-12:
            break
        # Reduce parent's stored speed
        cur.velocity = VelocityState(magnitude=v_parent_cap, direction=cur.velocity.direction)
        # Move up one edge; the new "child target" is this parent's capped speed,
        # and the next parent edge length is hex_d (or your recorded s_eff per edge if stored).
        cur_child_speed = v_parent_cap
        cur = cur.parent

    # Return adjusted current-edge entry and exit speeds
    v_here_adj = min(v_here, math.sqrt(max(v_child*v_child - 2.0*a_max*s_eff, 0.0)))
    return (v_here_adj, v_child)


# ------------------------------
# Cost functions (time-based)
# ------------------------------




# helpers: put these once (if not already defined with these exact formulas)
def next_velocity_straight(v_parent: float, a_max: float, delta_d: float) -> float:
    return math.sqrt(max(v_parent*v_parent + 2.0*a_max*delta_d, 0.0))  # Eq.(7)

def prev_velocity_straight(v_child: float, a_max: float, delta_d: float) -> float:
    return math.sqrt(max(v_child*v_child - 2.0*a_max*delta_d, 0.0))     # inverse of Eq.(7)

def travel_time(u: float, a: float, s: float) -> float:
    # t = (-u + sqrt(u^2 + 2 a s)) / a   (note **2 a s**, not 4 a s)
    if a == 0: raise ValueError("Acceleration cannot be zero")
    disc = u*u + 2.0*a*s
    if disc < 0: return math.inf
    return (-u + math.sqrt(disc)) / a




def travel_time_one_hex(u: float, a: float, hex_size: float) -> float:
    """Travel time for exactly one axial step on a pointy-top hex grid."""
    return travel_time(u, a, hex_step_distance(hex_size))


def g_cost(problem: HStarProblem, node: Node) -> float:
    """
    Actual cost (accumulated travel time) from start to 'node'.
    Uses velocity at parent and hex step distance.
    """
    return node.g_cost


def h_cost_travel_time(problem: HStarProblem, node: Node) -> float:
    """
    Heuristic: estimated time from node to goal assuming highest safe velocity
    and straight-line (manhattan) hex distance.
    """
    steps = hex_manhattan_distance(node.location, problem.goal)
    distance = steps * hex_step_distance(problem.grid.hex_size)
    a = problem.a_max
    u = node.velocity.magnitude
    return travel_time(u,a,distance)
    


def f_cost(g: float, h: float) -> float:
    """Total cost f = g + h."""
    return g + h



# ------------------------------
# Search engine (A*/H* loop)
# ------------------------------

# --- Heuristic type alias ---
HeuristicFn = Callable[["HStarProblem", "Node"], float]


class HStarSearch:
    """
    A* framework specialized with H* time-based costs and hex-grid actions.
    """
    
    def __init__(
        self, 
        problem: HStarProblem, 
        heuristic: Optional[HeuristicFn] = h_cost_travel_time,
    ) -> None:
        self.problem = problem
        self.h: HeuristicFn = heuristic
       
        self.open_set = PriorityQueue()
        self.closed_set: Dict[HexCoord] = {}
        self._counter = count()
        
        self.initialize()



    def initialize(self) -> Node:
        self.root = Node(
            location=self.problem.start,
            velocity=self.problem.start_v,
            parent=None,
            g_cost=0.0,
            h_cost=0.0,
            f_cost=0.0,
        )
        # Seed costs
        self.root.h_cost = self.h(self.problem, self.root)
        self.root.f_cost = self.root.g_cost + self.root.h_cost
        
        self.open_set.put((self.root.f_cost, -next(self._counter), self.root))
        self.closed_set[self.root.location] = self.root
        return self.root

            

    def f(self, node):
        return node.g_cost + self.h(self.problem, node)

    def reconstruct_path(self, node: Node) -> List[Node]:
        out: List[Node] = []
        cur = node
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out

    def search(self) -> Optional[List[Node]]:
        """
        Perform the H* search:
        - standard A* loop with open/closed sets
        - use g_cost, h_cost, and f_cost
        - ensure velocity updates & turn-handling
        """
        node = self.root
        self.open_set.put((self.h(self.problem, node), -next(self._counter), node))
        self.closed_set = {self.root.location: HexCoord}
        while not self.open_set.empty():
            # Pop from the front of the queue
            node = self.open_set.get(False)[2]
    
            if self.problem.is_goal(node.location):
                return node
            for child in self.expand(node):
                s = child.location
                if s not in self.closed_set or child.g_cost < self.closed_set[s].g_cost:
                    self.closed_set[s] = child
                    self.open_set.put((self.f(child), -next(self._counter), child))
        return None
    
    def search(self) -> Optional[List[Node]]:
        while not self.open_set.empty():
            _, _, node = self.open_set.get(False)
            if self.problem.is_goal(node.location):
                return self.reconstruct_path(node)


            for child in self.expand(node):
                s = child.location
                prev_best = self.closed_set.get(s)
                if prev_best is None or child.g_cost < prev_best.g_cost - 1e-12:
                    self.closed_set[s] = child
                    f = child.g_cost + self.h(self.problem, child)
                    self.open_set.put((f, -next(self._counter), child))
        return None

    
    def expand(self, node):
        return [self.result(node, action) for action in self.actions(node)]

    def actions(self, node):
        return self.problem.actions(node)
    def action_cost(self, node, action):
        return self.problem.action_cost(node, action)
    def result(self, node, action):
        return self.problem.result(node, action)
        


# ------------------------------
# Optional helpers for velocity statistics/smoothness analysis
# ------------------------------

def count_sharp_turns(path: List[Node], threshold_rad: float) -> int:
    """
    Count turns with |delta_theta| >= threshold.
    """
    pass


def velocity_stddev(path: List[Node]) -> Tuple[float, float]:
    """
    Return (stddev_speed, stddev_heading_change) along the path.
    """
    pass

