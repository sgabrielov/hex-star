from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Set

from hexgrid import HexCoord, Node, HexGrid, VelocityState, get_neighbors_at_radius, add_coords, get_direction, get_path_direction, hex_step_distance, idx_to_step, hdist

from hexgeometry import hdist

class AStarProblem:
    """
    Encapsulates the pathfinding problem for A*:
    - Grid definition
    - Start & goal
    """

    def __init__(
        self,
        grid: HexGrid,
        start: HexCoord,
        goal: HexCoord,
    ) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal

    def is_goal(self, location: HexCoord) -> bool:
        """Return True when the agent is exactly at the goal hex."""
        return location == self.goal

    def actions(self, node: "Node") -> List["HexCoord"]:
        """
        Return the successor locations available from node,
        which are the immediate neighbors of node.location
        """
        return self.grid.neighbors(node.location)
    
    def result(self, location: HexCoord, nxt: HexCoord) -> HexCoord:
        """Return the successor state (here, just the next location)."""
        return nxt

    def step_cost(self, location: HexCoord, nxt: HexCoord) -> float:
        """
        Cost to move from location -> nxt.
        Option A (unit cost): return 1.0
        Option B (physical distance): return hex_step_distance(self.grid.hex_size)
        """
        return 1

def h_cost_distance(problem: AStarProblem, node: Node) -> float:
    """
    Heuristic: manhattan distance from node location to goal
    """
    return hdist(node.location, problem.goal) * hex_step_distance(problem.grid.hex_size)
    
from queue import PriorityQueue
from itertools import count
from typing import Optional, Dict, List, Callable, Iterable


# Heuristic function type for A*:
# returns estimated remaining cost from node -> goal
AStarHeuristicFn = Callable[["AStarProblem", Node], float]


class AStarSearch:
    """
    A* framework (BEST-FIRST-SEARCH) for hex grids, written in the same style as HStarSearch.

    Key properties:
    - State identity is ONLY HexCoord (node.location).
    - reached/closed lookup table: Dict[HexCoord, Node] storing the best g_cost per location.
    - PriorityQueue frontier ordered by f = g + h.
    """

    def __init__(
        self,
        problem: "AStarProblem",
        heuristic: Optional[AStarHeuristicFn] = h_cost_distance,
        weight: int = 1,
    ) -> None:
        self.problem = problem

        # If heuristic isn't provided, try problem.heuristic(location), else default to 0.
        self.h: AStarHeuristicFn = heuristic if heuristic is not None else self._default_h
        self.weight = weight

        self.open_set: PriorityQueue = PriorityQueue()
        self.reached: Dict[HexCoord, Node] = {}
        self._counter = count()
        self.root: Optional[Node] = None

        self.initialize()

    # ----------------------------
    # Initialization
    # ----------------------------
    def initialize(self) -> Node:
        # Velocity is not part of A* state here, but Node requires it.
        # Use a harmless placeholder.
        placeholder_v = VelocityState(0.0, 0)

        self.root = Node(
            location=self.problem.start,
            velocity=placeholder_v,
            parent=None,
            g_cost=0.0,
            h_cost=0.0,
            f_cost=0.0,
        )

        self.root.h_cost = self.h(self.problem, self.root)
        self.root.f_cost = self.root.g_cost + self.root.h_cost

        self.open_set.put((self.root.f_cost, -next(self._counter), self.root))
        self.reached[self.root.location] = self.root
        return self.root

    # ----------------------------
    # Heuristic / cost helpers
    # ----------------------------
    def _default_h(self, problem: "AStarProblem", node: Node) -> float:
        """
        Default heuristic: if problem.heuristic(location) exists, use it.
        Otherwise 0 (Dijkstra).
        """
        if hasattr(problem, "heuristic"):
            try:
                return float(problem.heuristic(node.location))  # type: ignore[attr-defined]
            except TypeError:
                # If someone defined heuristic(self, node) instead
                return float(problem.heuristic(node))  # type: ignore[attr-defined]
        return 0.0

    def _step_cost(self, s: HexCoord, s2: HexCoord) -> float:
        """
        Default step cost: if problem.step_cost exists, use it, else unit cost.
        """
        if hasattr(self.problem, "step_cost"):
            return float(self.problem.step_cost(s, s2))  # type: ignore[attr-defined]
        return 1.0

    def f(self, node: Node) -> float:
        return node.g_cost + weight * self.h(self.problem, node)

    # ----------------------------
    # Path reconstruction
    # ----------------------------
    def reconstruct_path(self, node: Node) -> List[Node]:
        out: List[Node] = []
        cur: Optional[Node] = node
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out

    # ----------------------------
    # Expand (matches pseudocode)
    # ----------------------------
    def expand(self, node: Node) -> Iterable[Node]:
        """
        EXPAND(problem, node) yields child nodes.

        Pseudocode:
            s <- node.STATE
            for action in ACTIONS(s):
                s' <- RESULT(s, action)
                cost <- node.PATH-COST + ACTION-COST(...)
                yield NODE(STATE=s', PARENT=node, PATH-COST=cost)
        """

        s = node.location

        # Your AStarProblem.actions currently expects a Node and returns neighbor HexCoord(s).
        for nxt in self.problem.actions(node):
            s2 = self.problem.result(s, nxt) if hasattr(self.problem, "result") else nxt
            cost = node.g_cost + self._step_cost(s, s2)

            child = Node(
                location=s2,
                velocity=node.velocity,   # placeholder; ignored in state identity
                parent=node,
                g_cost=cost,
                h_cost=0.0,
                f_cost=0.0,
            )
            child.h_cost = self.h(self.problem, child)
            child.f_cost = child.g_cost + child.h_cost
            yield child

    # ----------------------------
    # Search (BEST-FIRST-SEARCH)
    # ----------------------------
    def search(self) -> Optional[List[Node]]:
        """
        BEST-FIRST-SEARCH / A* loop, with reached table keyed ONLY by HexCoord.
        Returns reconstructed path as List[Node], or None on failure.
        """

        while not self.open_set.empty():
            _, _, node = self.open_set.get(False)

            # Stale entry check:
            # if this node is not the best-known node for its location, skip it.
            best = self.reached.get(node.location)
            if best is None:
                continue
            if node is not best and node.g_cost > best.g_cost + 1e-12:
                continue

            if self.problem.is_goal(node.location):
                return self.reconstruct_path(node)

            for child in self.expand(node):
                s = child.location
                prev_best = self.reached.get(s)

                # Reached-table update rule from pseudocode:
                if prev_best is None or child.g_cost < prev_best.g_cost - 1e-12:
                    self.reached[s] = child
                    self.open_set.put((child.f_cost, -next(self._counter), child))

        return None