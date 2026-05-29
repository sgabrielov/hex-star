from hexgrid import HexCoord, VelocityState, make_grid_with_obstacles, get_neighbors_at_radius
from hexgeometry import euclid_center_distance, unsigned_turn_angle, curvature_three_points

from typing import Iterable, Optional, Set, List, Tuple, Dict


def evaluate_path(
    path: List[HexCoord],
    *,
    hex_size: float = 1.0,
) -> Dict[str, float]:

    if len(path) < 2:
        return {
            "length": 0.0,
            "turn_sum": 0.0,
            "curvature_sum": 0.0,
            "curvature_sq": 0.0,
            "max_curvature": 0.0,
        }

    # --- length ---
    length = 0.0
    for i in range(len(path) - 1):
        length += euclid_center_distance(path[i], path[i+1], hex_size=hex_size)

    # --- curvature / turning ---
    turn_sum = 0.0
    curvature_sum = 0.0
    curvature_sq = 0.0
    max_curv = 0.0

    for i in range(1, len(path) - 1):
        a, b, c = path[i-1], path[i], path[i+1]

        # turn angle (0 if straight)
        theta = abs(unsigned_turn_angle(a, b, c, hex_size=hex_size))

        # curvature (your geometric version)
        kappa = abs(curvature_three_points(a, b, c, hex_size=hex_size))

        turn_sum += theta
        curvature_sum += kappa
        curvature_sq += kappa * kappa
        max_curv = max(max_curv, kappa)

    return {
        "length": length,
        "turn_sum": turn_sum,
        "curvature_sum": curvature_sum,
        "curvature_sq": curvature_sq,
        "max_curvature": max_curv,
    }

def normalize_metrics(metrics):
    L = metrics["length"]
    
    return {
        "length": L,
        "turn_per_length": metrics["turn_sum"] / L if L > 0 else 0.0,
        "curv_per_length": metrics["curvature_sum"] / L if L > 0 else 0.0,
        "curv_sq_per_length": metrics["curvature_sq"] / L if L > 0 else 0.0,
        "max_curvature": metrics["max_curvature"],
    }


AGENT_DEFAULTS = {
    'a_max': 5,
    'a_min': 10,
    'max_turn_gs': 9,
    'collision_radius': 0,
    'start_v': VelocityState(0, None),
    'goal_v': None,
}

SEARCH_DEFAULTS = {
    'jps_horizon': None,
    'n_step': 2,
}
import inspect
from copy import deepcopy


def _supported_kwargs(callable_obj, kwargs: dict) -> dict:
    """
    Return only the kwargs accepted by callable_obj's signature.

    Works for classes (uses __init__) and normal callables.
    """
    sig = inspect.signature(callable_obj)

    accepted = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in kwargs:
            accepted[name] = kwargs[name]

    return accepted


def _construct_with_supported_kwargs(cls, kwargs: dict):
    """
    Instantiate cls using only the kwargs supported by its constructor.
    """
    return cls(**_supported_kwargs(cls, kwargs))

def run_benchmark(
    scenarios, 
    problem_class, 
    search_class, 
    h_func, 
    agent_parms = AGENT_DEFAULTS, 
    search_parms = SEARCH_DEFAULTS, 
    seed=None,
    num_runs=20

):
    results = []

    for config in scenarios:
        for seed in range(num_runs):            
            path = run_scenario(
                config, 
                problem_class=problem_class,
                search_class=search_class,
                h_func=h_func,
                agent_parms=agent_parms,
                search_parms=search_parms,
                seed=seed
            )

            if not path:
                continue  # skip failures or track success rate separately

            raw = evaluate_path([p.location for p in path], hex_size=config['hex_size'])
            norm = normalize_metrics(raw)

            results.append(norm)

    return results
    
def run_scenario(
    map_scenario,
    problem_class,
    search_class,
    h_func,
    agent_parms=None,
    search_parms=None,
    seed=None,
):
    # Avoid mutating shared defaults
    agent_parms = deepcopy(AGENT_DEFAULTS if agent_parms is None else agent_parms)
    search_parms = deepcopy(SEARCH_DEFAULTS if search_parms is None else search_parms)

    # Fix your existing bug: you weren't assigning the fallback seed
    if seed is None:
        seed = map_scenario.get("seed", None)

    hg = make_grid_with_obstacles(
        hex_size=map_scenario["hex_size"],
        center=map_scenario["center"],
        radius=map_scenario["radius"],
        types=map_scenario["types"],
        exclude=[map_scenario["start"], map_scenario["goal"]],
        seed=seed,
    )

    # Keep your boundary ring blocked if that's intentional
    hg.obstacles = set(hg.obstacles.keys()).union(
        get_neighbors_at_radius(HexCoord(0, 0), map_scenario["radius"])
    )

    hg.obstacles.discard(map_scenario["start"])
    hg.obstacles.discard(map_scenario["goal"])

    # ----------------------------
    # Candidate kwargs for PROBLEM
    # ----------------------------
    problem_kwargs = {
        # common names
        "grid": hg,
        "hg": hg,
        "hex_grid": hg,

        "start": map_scenario["start"],
        "goal": map_scenario["goal"],

        # H* / richer problem variants
        "a_max": agent_parms["a_max"],
        "a_min": agent_parms["a_min"],
        "collision_radius": agent_parms["collision_radius"],
        "jps_horizon": search_parms["jps_horizon"],
        "ay_window_ms": agent_parms["max_turn_gs"],
        "max_turn_gs": agent_parms["max_turn_gs"],
        "n_step": search_parms["n_step"],
        "start_v": agent_parms["start_v"],
        "goal_v": agent_parms.get("goal_v", None),
    }

    problem = _construct_with_supported_kwargs(problem_class, problem_kwargs)

    # ----------------------------
    # Candidate kwargs for SEARCH
    # ----------------------------
    search_kwargs = {
        "problem": problem,
        "heuristic": h_func,
        "h_func": h_func,
    }

    searcher = _construct_with_supported_kwargs(search_class, search_kwargs)

    return searcher.search()

def run_scenario_smold(
    map_scenario, 
    problem_class, 
    search_class, 
    h_func, 
    agent_parms = AGENT_DEFAULTS, 
    search_parms = SEARCH_DEFAULTS,
    seed=None,
):
    if seed is None:
        map_scenario.get("seed", None)
    hg = make_grid_with_obstacles(
        hex_size=map_scenario["hex_size"],
        center=map_scenario["center"],
        radius=map_scenario["radius"],
        types=map_scenario["types"],
        exclude=[map_scenario['start'], map_scenario['goal']],
        seed=seed,
    )    

    hg.obstacles = set(hg.obstacles.keys()).union(get_neighbors_at_radius(HexCoord(0,0), map_scenario["radius"]))
    
    hg.obstacles.discard(map_scenario["start"])
    hg.obstacles.discard(map_scenario["goal"])

    problem = problem_class(
        hg, # hex grid
        map_scenario['start'],
        map_scenario['goal'],
        agent_parms['a_max'],  # maximum acceleration (for acceleration phase)
        agent_parms['a_min'],  # maximum deceleration (positive value; used with sign as needed)
        collision_radius = agent_parms['collision_radius'],
        jps_horizon = search_parms['jps_horizon'],
        ay_window_ms = agent_parms['max_turn_gs'],
        n_step = search_parms['n_step'],   
        start_v = agent_parms['start_v'],
        goal_v = agent_parms['start_v']
    )
    #problem.goal_v=VelocityState(0, problem.DIRECTIONS.index(HexCoord(-1,2))) # approach from the northeast
    return search_class(
        problem = problem,
        heuristic = h_func
    ).search()

def summarize(results: List[Dict[str, float]]):
    keys = results[0].keys()
    summary = {}

    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    return summary