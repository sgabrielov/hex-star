from hexgrid import HexCoord, VelocityState, Node, HexGrid, get_neighbors_at_radius, add_coords, get_direction, get_direction_ray, get_path_direction, hex_step_distance, idx_to_step, build_obstacle_map, get_available_nodes_in_dir, hex_segment_between, angle_abc, hdist, unique_direction_vectors_up_to_n, euclid_center_distance, hexes_in_between_solid, value_at_index, circular_slice, gcd_reduce
from hexgeometry import reverse_dir
from typing import Iterable, Optional, Set, List, Tuple, Callable, Dict
from queue import PriorityQueue
from itertools import count
from collections import defaultdict

import math

from dataclasses import dataclass

@dataclass(frozen=True)
class DirectionInfo:
    s: int # distance from origin
    blockers: List[HexCoord] # HexCoords that must be open for this hex to be navigable


DEFAULT_EPSILON = 1e-9
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
        goal_v: VelocityState = VelocityState(None, None) , # Target goal velocity for BiH*
        ay_window_ms: int = 1000,  # window for lateral acceleration averaging
        collision_radius: int = 0, # How far away an obstacle must be from the agent's location to cause a collision
        jps_horizon: int = 5, # Max jumps in a row when shortcutting
        n_step: int = 2, # number of steps to take in one action | radial resolution
        search_direction: str = "forward",
        epsilon: float = DEFAULT_EPSILON # 1e-14
    ) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal
        self.a_max = a_max
        self.a_min = a_min
        self.ay_window_ms = ay_window_ms
        self.collision_radius = collision_radius
        self.jps_horizon = jps_horizon
        self.start_v = start_v
        self.goal_v = goal_v
        self.search_direction = search_direction
        self.n_step = n_step
        self.epsilon = epsilon


        # if self.search_direction == "forward":
        #     self.transition_function = next_velocity
        # elif self.search_direction == "reverse":
        #     self.transition_function = prev_velocity
        
        self.transition_function = next_velocity
        self.transition_function = next_velocity

        self.DIRECTIONS = unique_direction_vectors_up_to_n(HexCoord(0,0), self.n_step)
        self.DIRECTIONS_INFO = {}

        self.expand_counter = defaultdict(int)

        self.collision_radius_map = self.generate_collision_radius_map(self.collision_radius, self.grid.obstacles)
        
        for d in self.DIRECTIONS:
            self.DIRECTIONS_INFO[d] = DirectionInfo(
                s = euclid_center_distance(d, HexCoord(0,0)),
                blockers = hexes_in_between_solid(d, HexCoord(0,0), include_endpoints = True)
            )

    def enforce_goal_velocity(self, node):
        if self.goal_v is None:
            return node
        if self.goal_v.magnitude is None:
            return node
        return self.backtrack(node, self.goal_v.magnitude, self.a_min, transition_function = self.transition_function)
    
    def is_goal_loc(self, location: HexCoord) -> bool:
        """Return True when the agent is exactly at the goal hex."""
        return location == self.goal

    def is_goal_dir(self, direction: int) -> bool:
        """Return True when the agent is exactly at the goal hex."""
        if self.goal_v is None:
            return True
        if self.goal_v.direction is None:
            return True
        return direction == self.goal_v.direction

    def is_goal(self, node: Node) -> bool:
        """Return True when the agent is exactly at the goal hex and aligned with goal direction."""
        isGoal = self.is_goal_loc(node.location)
        isGoalDir = self.is_goal_dir(node.velocity.direction)
        return isGoal and isGoalDir
        
    def generate_collision_radius_map(self, collision_radius, obstacles):
        obs_in_r_set = set()
        for obs in obstacles:
            for r in range(1, collision_radius + 1):
                obs_in_r = get_neighbors_at_radius(obs, r) - obstacles
                obs_in_r_set = obs_in_r_set.union(obs_in_r)
        return obs_in_r_set
        
    def collision_clear(self, coord: HexCoord, blockers: Optional[Set[HexCoord]] = None)     -> bool:
        if blockers is None:
            blockers = set()
        if coord in self.grid.obstacles or coord in self.collision_radius_map:
            return False
        for b in blockers:
            if b in self.grid.obstacles or b in self.collision_radius_map:
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
        idx_range = list(range(len(self.DIRECTIONS)))
        if start_unknown_heading:
            return idx_range

        idx = node.velocity.direction
        offset = value_at_index(self.n_step)
        return circular_slice(idx_range, idx-offset, idx+offset+1)

    def actions(self, node: "Node") -> List["HexCoord"]:
        """
        Return only the **new states** (successor locations) that are reachable from `node`,
        honoring the start-unknown-heading rule and collision clearance.

        Based on JPS style shortcutting where each direction is expanded until collision
        """
        possible_successors: Set[HexCoord] = set()
        for di in self.candidate_dir_indices(node):
            # For each candidate direction, find all successors in a line up to
            # the next obstacle
            successor_line = self.get_available_hexes_in_dir(
                node.location,          # Current location
                self.DIRECTIONS[di],    # Direction
                include_start=False,    # Do not include the start location
                ray_range = self.jps_horizon
            )
            possible_successors = possible_successors.union(successor_line)

            
        
        return list(possible_successors)

    # Calculates the resulting node from applying action to node
    # Action is a hexcoord location returned by the actions function
    # Representing the destination location
    def result(self, node: "Node", action: "HexCoord") -> "Node":
        new_branch = None
        # First, check if the next location is reachable given velocity and direction
        if self.ay_window_ms is not None and self.ay_window_ms > 0:
            new_branch = self.constrain_velocity(node, action)

        if new_branch is None:
            parent = node
        else:
            parent = new_branch
            
            
        action_step, g = gcd_reduce(add_coords(action, (-parent.location.q, -parent.location.r)))
        direction = self.DIRECTIONS.index(action_step)
        s = self.DIRECTIONS_INFO[action_step].s * g
        new_v = self.transition_function(parent.velocity.magnitude, self.a_max, s)
        # Calculate the travel time from parent to child, given parent velocity,
        # child velocity, and step distance
        
        t_step = travel_time_symmetric(parent.velocity.magnitude, new_v, s)
        #t_step = travel_time_accel_limited(parent.velocity.magnitude, new_v, s, self.a_max, self.a_min)

        self.expand_counter[action] += 1
        child = Node(
            location=action,
            velocity=VelocityState(magnitude=new_v, direction=direction),
            parent=parent,
            step_distance = s
        )
        child.g_cost = parent.g_cost + t_step
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

    # def action_cost(self, prev: Node, action: HexCoord) -> float:
    #     """
    #     Travel time from `prev` to `action` (a neighbor location), using:
    #         s = √3 * hex_size
    #         t = (-u + sqrt(u^2 + 2*a*s)) / a
    #     Note: this uses the parent's speed `u` and problem's `a_max` (acceleration phase).
    #     """
        
    #     action_step, g = gcd_reduce(add_coords(action, (-node.location.q, -node.location.r)))
    #     direction = self.DIRECTIONS.index(action_step)
    #     s = self.DIRECTIONS_INFO[action_step].s * g
    #     new_v = self.transition_function(parent.velocity.magnitude, self.a_max, s)
        
    #     return travel_time_accel_limited(node.velocity.magnitude, new_v, s, self.a_max, self.a_min)
        
    def constrain_velocity(self, node, next_location):
        if node.parent is None:
            return None
        radius = self.calculate_turning_radius(node, next_location)
        if radius is None:
            return None
        v_max = math.sqrt(2*radius*self.ay_window_ms)
        if node.velocity.magnitude <= v_max + self.epsilon:
            return None # Do nothing
        return self.backtrack(node, v_max, self.a_min, transition_function = self.transition_function)
        
    def reverse(
        self,
        goal_speed_mag: Optional[float] = None,
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
            Optional discrete heading. If None, reversed search will
            treat the start heading as unknown (allowed to expand all).
        """
        # Seed reversed start with the known goal velocity (magnitude; heading optional)

        if goal_speed_mag is None:
            goal_speed_mag = self.goal_v.magnitude

        if goal_heading_idx is None:
            goal_heading_idx = reverse_dir(self.goal_v.direction)
            
        rev_start_v = VelocityState(magnitude=goal_speed_mag,
                                    direction=goal_heading_idx
                                   )
       

        # Mirror problem geometry & limits; keep a_max/a_min positive
        reversed_problem = HStarProblem(
            grid=self.grid,
            start=self.goal,     # swap endpoints
            goal=self.start,
            a_max=self.a_min,    # keep positive magnitudes 
            a_min=self.a_max,
            start_v=rev_start_v, # start speed at reversed start
            goal_v=self.start_v, # target end-state matches original start
            ay_window_ms=self.ay_window_ms,
            collision_radius=self.collision_radius,
            jps_horizon=self.jps_horizon,
            n_step=self.n_step,
            search_direction="reverse",

        )

        return reversed_problem
        
    def backtrack(
        self,
        start_node: "Node",
        v_max: float,
        a_min: float,
        transition_function
    ):
        """
        Enforce a v_max on a path by forcing a node value to v_max,
        and propagating braking constraints to ancestors. After any velocity
        updates, recompute g_costs for the affected segment so costs remain consistent.
        """
        if start_node.velocity.magnitude <= v_max + self.epsilon:
            return start_node

        # If start_node has no parent, it's the root
        # So return None because that means this branch is only possible from a different root
        if start_node.parent is None:
            return None
    
        # Make a copy of the start node
        start_node_copy = Node(
            start_node.location,
            VelocityState(magnitude = v_max, direction = start_node.velocity.direction),
            start_node.parent,
            start_node.step_distance,
            g_cost = None, # Will update after updating velocities along the path
            h_cost = None,
            f_cost = None
        )
        # Calculate the distance from the parent to the start node which is used to calculate
        # t_step for g_cost and feasible parent velocity based on braking ability
        transition_distance = start_node.step_distance

        # Calculate the v_max for the parent
        v_next = transition_function(v_max, a_min, transition_distance)

        # Apply backtracking to the parent, and get the new parent back
        # This will backtrack recursively through the linked list until no more updates are needed
        # Then, it will calculate new g_costs from the updated velocities back through the call stack 
        # Until it reaches the calling node (start_node)
        if start_node_copy.parent is None:
            return start_node_copy
        next_parent = self.backtrack(start_node_copy.parent, v_next, a_min, transition_function=transition_function)
    

        if next_parent is None:
            return None
        
        start_node_copy.parent = next_parent

        
        # Once we have the new updated parent, calcuate costs
        t_step = travel_time_symmetric(start_node_copy.velocity.magnitude, start_node_copy.parent.velocity.magnitude, transition_distance)
        #t_step = travel_time_accel_limited(start_node_copy.velocity.magnitude, start_node_copy.parent.velocity.magnitude, transition_distance, self.a_max, self.a_min)
        start_node_copy.g_cost = start_node_copy.parent.g_cost + t_step
        start_node_copy.h_cost = self.heuristic(self, start_node_copy) if hasattr(self, "heuristic") and callable(self.heuristic) else h_cost_travel_time(self, start_node_copy)
        start_node_copy.f_cost = f_cost(start_node_copy.g_cost, start_node_copy.h_cost)
    
        # Finally return this node
        return start_node_copy
            
    def calculate_turning_radius(self, node, child_location):
        if node.parent is None:
            return None
        parent = node.parent
        angle = angle_abc(parent.location, node.location, child_location)

        action_step, g = gcd_reduce(add_coords(child_location, (-node.location.q, -node.location.r)))
        direction = self.DIRECTIONS.index(action_step)
        
        edge_1 = node.step_distance
        edge_2 = self.DIRECTIONS_INFO[action_step].s * g
        return radius_from_angle(
            min([edge_1, edge_2]), 
            angle
        )
    
    def get_available_hexes_in_dir(
        self,
        start: HexCoord,
        direction: HexCoord,
        include_start: bool = True,
        ray_range: Optional[float] = None,   # max traveled distance (units of DirectionInfo.s)
        max_steps: int = 10000,              # safety cap when ray_range is None
    ) -> Set[HexCoord]:
        step_info = self.DIRECTIONS_INFO[direction]
        step_size = float(step_info.s)
    
        total_s = 0.0
        out: Set[HexCoord] = set()
        if include_start:
            out.add(start)
    
        # start at first destination if include_start=False; otherwise also start at first dest
        dest = start
    
        # helper to compute anchored blockers at the "anchor" cell (here you used the current cell)
        def anchored_blockers(anchor: HexCoord) -> Set[HexCoord]:
            return {add_coords(anchor, (step.q, step.r)) for step in step_info.blockers}
    
        steps = 0
        while True:
            # stop if we hit distance limit (only when ray_range is provided)
            if ray_range is not None and total_s >= ray_range:
                break
    
            blockers = anchored_blockers(dest)
    
            if not self.collision_clear(dest, blockers=blockers):
                break
    
    
            total_s += step_size
            dest = add_coords(dest, (direction.q, direction.r))
            out.add(dest)
    
            steps += 1
            if ray_range is None and steps >= max_steps:
                # safety cap for unbounded rays
                break
    
        return out
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

    
def radius_from_angle(edge_length, angle, inscribed=False, degrees_in: bool = False):
    """
    Compute circumradius (default) or inradius (if inscribed=True) of a regular polygon
    from its side length and interior angle.

    Parameters
    ----------
    edge_length : float
        Side length (s).
    angle : float
        Interior angle, in radians by default. If degrees_in=True, then in degrees.
    inscribed : bool, default False
        False -> circumradius (R)
        True  -> inradius / apothem (r)
    degrees_in : bool, default False
        If True, `angle` is treated as degrees; otherwise radians.

    Returns
    -------
    float
        Requested radius.
    """
    angle_rad = math.radians(angle) if degrees_in else angle

    if angle_rad == math.pi:
        return None

    # For a regular n-gon, interior angle A satisfies: A = (n-2)*pi/n
    # => n = 2*pi / (pi - A)
    n = 2 * math.pi / (math.pi - angle_rad)

    if inscribed:
        # inradius (apothem): r = s / (2 * tan(pi/n))
        return edge_length / (2 * math.tan(math.pi / n))
    else:
        # circumradius: R = s / (2 * sin(pi/n))
        return edge_length / (2 * math.sin(math.pi / n))

def construct_full_solution(solution):
    fsolution_path = []
    for node in solution:
        if node.parent:
            fsolution_path.extend(hexes_in_between_solid(node.location, node.parent.location))
    
        fsolution_path.extend([node.location])

    return fsolution_path
    
# ------------------------------
# Cost functions (time-based)
# ------------------------------

# Solving for v_next, given v_prev
def next_velocity(
    v_prev: float, a_max: float, delta_d: float
) -> float:
    return math.sqrt(v_prev*v_prev + 2*a_max*delta_d)

# Solving for v_prev, given v_next
def prev_velocity(
    v_next: float, a_max: float, delta_d: float
) -> float:
    rad = v_next*v_next - 2*a_max*delta_d
    if rad <= 0:
        return 0
    return math.sqrt(rad)



def travel_time(u: float, a: float, s: float) -> float:
    # t = (-u + sqrt(u^2 + 2 a s)) / a   (note **2 a s**, not 4 a s)
    if a == 0: raise ValueError("Acceleration cannot be zero")
    disc = u*u + 2.0*a*s
    if disc < 0: return math.inf
    return (-abs(u) + math.sqrt(disc)) / a # Take the absolute value of velocity in case its negative


def travel_time_symmetric(v1: float, v2: float, s: float) -> float:
    denom = abs(v1) + abs(v2)
    if denom == 0:
        return math.inf
    return 2.0 * s / denom
    

def travel_time_accel_limited(v1: float, v2: float, s: float, a_up: float, a_down: float) -> float:
    """
    Optimistic edge time for H*:
      - if v2 >= v1: accelerate toward v2, then cruise
      - if v2 < v1 : cruise, then brake late toward v2

    IMPORTANT:
      If the requested endpoint speed is infeasible over distance s,
      do NOT return inf. Fall back to an optimistic finite model, because
      this planner allows infeasible expansions and repairs them later
      via backtracking.
    """
    v1 = abs(v1)
    v2 = abs(v2)
    s = float(s)
    a_up = abs(float(a_up))
    a_down = abs(float(a_down))

    if s < 0:
        raise ValueError("s must be nonnegative")

    if s == 0:
        return 0.0 if v1 == v2 else math.inf

    # -------- accelerating / equal-speed case --------
    if v2 >= v1:
        if a_up == 0:
            if v1 == v2 and v1 > 0:
                return s / v1
            return math.inf

        s_change = (v2*v2 - v1*v1) / (2.0 * a_up)

        # exact target speed infeasible -> optimistic fallback:
        # accelerate for the whole segment
        if s < s_change:
            return travel_time(v1, a_up, s)

        t_change = (v2 - v1) / a_up
        s_cruise = s - s_change

        # if both speeds are zero, still allow optimistic forward motion
        if v2 == 0:
            return travel_time(v1, a_up, s)

        return t_change + s_cruise / v2

    # -------- decelerating case --------
    else:
        if a_down == 0:
            if v1 == v2 and v1 > 0:
                return s / v1
            return math.inf

        s_change = (v1*v1 - v2*v2) / (2.0 * a_down)

        # exact braking infeasible -> optimistic fallback
        # do NOT kill the branch yet
        if s < s_change:
            denom = v1 + v2
            if denom > 0:
                return 2.0 * s / denom
            # if both are zero somehow, fall back to forward accel model
            return travel_time(v1, a_up, s)

        t_change = (v1 - v2) / a_down
        s_cruise = s - s_change

        if v1 == 0:
            return travel_time(v1, a_up, s)

        return t_change + s_cruise / v1


def travel_time_one_hex(u: float, a: float, hex_size: float) -> float:
    """Travel time for exactly one axial step on a pointy-top hex grid."""
    return travel_time(u, a, hex_step_distance(hex_size))


def g_cost(problem: HStarProblem, node: Node) -> float:
    """
    Actual cost (accumulated travel time) from start to 'node'.
    Uses velocity at parent and hex step distance.
    """
    return node.g_cost

# def h_cost_travel_time(problem: HStarProblem, node: Node) -> float:
#     """
#     Heuristic: estimated time from node to goal assuming highest safe velocity
#     and straight-line (manhattan) hex distance.
#     """
#     steps = hex_manhattan_distance(node.location, problem.goal)
#     distance = steps * hex_step_distance(problem.grid.hex_size)
#     a = problem.a_max
#     a2 = problem.a_min
#     u = node.velocity.magnitude
#     goal_v = problem.goal_v.magnitude
#     return travel_time_accel_limited(u, goal_v, distance, a, a2)
    
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

import time
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

        self.benchmarks = {
            "states_reached": 0,
            "nodes_expanded": 0,
            "search_time": 0,
            "solution_cost": 0,
            "solution_length": 0,
            "open_set_sizes": [],
            "closed_set_sizes": [],
        }
        
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

        self.open_set_sizes = []
        self.closed_set_sizes = []
        
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
        start_time = time.time()
        while not self.open_set.empty():
            self.benchmarks['open_set_sizes'].append(self.open_set.qsize())
            self.benchmarks['closed_set_sizes'].append(len(self.closed_set))
            _, _, node = self.open_set.get(False)
            self.benchmarks['nodes_expanded'] += 1
            if self.problem.is_goal(node):
                self.benchmarks['search_time'] = time.time() - start_time
                node = self.problem.enforce_goal_velocity(node)
                soln_path = self.reconstruct_path(node)
                self.benchmarks['solution_cost'] = node.g_cost
                self.benchmarks['solution_length'] = len(soln_path)
                return soln_path
            for child in self.expand(node):
                s = child.location
                prev_best = self.closed_set.get(s)
                if prev_best is None or child.g_cost < prev_best.g_cost - 1e-12:
                    self.closed_set[s] = child
                    f = child.g_cost + self.h(self.problem, child)
                    self.open_set.put((f, -next(self._counter), child))
        return None  
    

    def expand(self, node):
        children = []
        for action in self.actions(node):
            child = self.result(node, action)
            self.benchmarks["states_reached"] += 1
            children.append(child)
        return children


    def actions(self, node):
        return self.problem.actions(node)
    def action_cost(self, node, action):
        return self.problem.action_cost(node, action)
    def result(self, node, action):
        return self.problem.result(node, action)
        
    def get_benchmarks(self):
        return self.benchmarks

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