from math import sqrt, pi
from queue import PriorityQueue
from itertools import count
from copy import copy
from collections import defaultdict
import numpy as np
import pickle as pkl

a_star_weight = 1
counter = count()

# Best first search algorithm
# using g(n) + h(n) for f will make this an A* search
def best_first_search(problem, f, h, max_solutions = 1, get_frontier = False, check_consistency = True):
    node = problem.root
    frontier = PriorityQueue(0)

    solutions = []

    # Priority queue, elements input as tuple (priority, counter, value), using f(node) for priority
    # Counter is used as the tiebreaker between nodes when they have the same f value
    # Nodes are expanded in order they were added when they have the same priority
    frontier.put((f(node, h), -next(counter), node))
    reached = {problem.root: node}

    while not frontier.empty():
        # Pop from the front of the queue
        node = frontier.get(False)[2]

        # Terminal condition
        if problem.is_goal(node.state):
            
            # We do not observe any rewards until a solution is found
            # Once the terminal state is reached, recursively apply rewards
            problem.q_update(node)
            
            solutions.append(node)
            
            if len(solutions) >= max_solutions:
                for f_cost, _, nonterminal_node in list(frontier.queue):
                    problem.q_update(nonterminal_node)
                if not get_frontier:
                    return solutions
                else:
                    return (solutions, list(frontier.queue))
            
        for child in expand(problem, node, h, check_consistency = check_consistency):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.put((f(child, h), -next(counter), child))
    return None

def expand(problem, node, h, check_consistency = True):
    s = node.state
    nodes = []
    for action in problem.actions(s):
        s_new = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action)
        new_node = problem.Node(s_new, node, action, cost, problem) # Expand a node to take an action

        
        if problem.heuristic_consistent_flag and check_consistency:
            problem.heuristic_consistent_flag = check_h_consistency(problem, new_node, h)
        nodes.append(new_node)

        # Keep track of the number of expanded states   
        problem.num_expanded_states += 1
    
        problem.num_generated_nodes += len(nodes)
    
    return nodes

def time_to_goal(node):
    
    problem = node.problem
    goal_loc = problem.goal_loc
    agent_loc = node.state[problem.state_dict['agent']]
    v, theta_a = node.state[problem.state_dict['velocity']]
    a = problem.a_max
    s = problem.hex_manhattan_distance(goal_loc, agent_loc) * sqrt(3) * problem.hex_size

    return problem.get_travel_time(v, a, s)

def q_key(node):
    problem = node.problem
    goal_loc = problem.goal_loc
    agent_loc = node.state[problem.state_dict['agent']]
    agent_v, agent_dir = node.state[problem.state_dict['velocity']]

    # Exclude agent velocity because it's a continuous value
    return (
        agent_loc,
        goal_loc,
        agent_dir
    )
    

def q_learning(node):

    # Get the learned h value for this state
    return node.problem.q_values[q_key(node)]

# Because A* needs a decent heuristic to work quickly, 
# H* will use the max of time_to_goal and q_learning 
# For A* max(h1, h2) is consistent when both h1 and h2 are consistent 
def combo_h(
    node,
    h1 = time_to_goal,
    h2 = q_learning
):
    return max(h1(node), h2(node))

    


def check_h_consistency(problem, node, h):
    heuristic_is_consistent = True
    if node.parent is not None:
        heuristic_is_consistent = h(node.parent) <= problem.action_cost(node.parent.state, node.action) + h(node)
    if not heuristic_is_consistent:
                hn = h(node)
                hp = h(node.parent)
                ac = problem.action_cost(node.parent.state, node.action)
                print("h is inconsistent")
                print(f'ac:        {ac}')
                print(f'h(parent): {hp}')
                print(f'h(node):   {hn}')
                print(f'{hp} <= {ac} + {hn}  is false')
                print(node.state)
                print(node.parent.state)
    return heuristic_is_consistent

def f(node, h=time_to_goal):
    return g(node) + a_star_weight * h(node)

def g(node):
    return node.path_cost

class PathfindingProblem:
    class Node:
        def __init__(self, state, parent, action, path_cost, problem):
            self.state = state
            self.parent = parent
            self.action = action
            self.path_cost = path_cost

            self.problem = problem

            # if the new node has a different angle than previous
            # the agent turned, apply the turning penalty and backtrack to update velocities and path costs along the path
            if self.is_turn() and not self.is_zigzag():
                r = 1
                current_node = self.parent
                while not current_node.is_turn() and current_node.is_zigzag():
                    current_node = current_node.parent
                    r += 1
                
                # calculate the max velocity for the turn
                v_max = sqrt(2*r * problem.ay_max)
                if state[1][0] > v_max:
                    # update the velocity in the current node to v_max
                    self.state = (
                        self.state[0],
                        (v_max, self.state[problem.state_dict['velocity']][1])
                    )
                self.update_velocity(problem)

        def update_velocity(self, problem):
            current_node = self
            while current_node.parent:
                new_v = problem.calculate_velocity(current_node.state[problem.state_dict['velocity']][0], problem.d_max, sqrt(3) * problem.hex_size)
                if new_v < current_node.parent.state[problem.state_dict['velocity']][0]:
                    parent_copy = copy(current_node.parent)
                    parent_copy.state = (
                        parent_copy.state[0],
                        (new_v, parent_copy.state[problem.state_dict['velocity']][1])
                    )
                    if parent_copy.parent:
                        parent_path_cost = parent_copy.parent.path_cost
                    else:
                        parent_path_cost = 0

                    parent_copy.path_cost = parent_path_cost + problem.get_travel_time(parent_copy.state[problem.state_dict['velocity']][0], problem.d_max, sqrt(3) * problem.hex_size)
                    current_node.parent = parent_copy
                    if problem.heuristic_consistent_flag:
                        problem.heuristic_consistent_flag = check_h_consistency(problem, parent_copy, time_to_goal)
                    current_node = current_node.parent
                else:
                    break
            
        def is_turn(self):
            return self.parent is not None and self.state[self.problem.state_dict['velocity']][1] != self.parent.state[self.problem.state_dict['velocity']][1]
        def is_zigzag(self):
            return self.parent is not None and self.parent.parent is not None and self.state[self.problem.state_dict['velocity']][1] == self.parent.parent.state[self.problem.state_dict['velocity']][1]

        
        def __str__(self):
            return str(self.state)

    #### HexStar problem constructor ####
    def __init__(
        self, 
        initial_state,        # start location
        hex_map,              # all locations within environment
        obstacle_map,         # all obstacles within hex_map
        goal_loc,             # goal location
        hex_radius,           # radius of hex in meters
        hex_size,             # render size of hex in pixels
        agent_size_r,         # agent cannot go closer than this to obstacles
        acceleration_max,     # agent max acceleration
        deceleration_max,     # agent max braking
        lat_acceleration_max, # agent max turning g-force
        q_learning_rate,  
        q_discount_factor,
        q_values_filename = None,
        q_values_counts = None,
        update_nonterminals = False,
        learning=False,
    ):
        self.root = self.Node(initial_state, None, None, 0, self)

        self.hex_map = hex_map
        self.obstacle_map = obstacle_map
        self.goal_loc = goal_loc
        self.hex_radius = hex_radius
        self.hex_size = hex_size
        self.a_max = acceleration_max
        self.d_max = deceleration_max
        self.ay_max = lat_acceleration_max
        self.agent_size_r = agent_size_r

        # for benchmarking
        self.heuristic_consistent_flag = True
        self.num_expanded_states = 0
        self.num_generated_nodes = 0

        # for q-learning benchmarking
        self.training_error = []
        self.episode_rewards = []
    
        self.update_nonterminals = update_nonterminals

        self.q_values_filename = q_values_filename
        self.learning = learning
        ### Q-Learning ###
        if self.q_values_filename is not None:
            self.q_values, self.q_tracker = self.load_q(self.q_values_filename)
        elif q_values_counts is not None:
            self.q_values, self.q_tracker = q_values_counts
        else:
            self.reset_q()
            

        self.q_learning_rate = q_learning_rate 
        self.q_discount_factor = q_discount_factor
        self.training_error = []

        # Defines the indices for the components of the state
        self.state_dict = {
            'agent': 0,
            'velocity': 1
        }

        # Defines the directions to the immediate neighborhood and their angle relative to horizontal
        self.neighborhood_angles = {
            (1,0): 0,
            (0,1): 1,
            (-1,1): 2,
            (-1,0): 3,
            (0,-1): 4,
            (1,-1): 5   
        }

    def update_initial_state(self, new_initial_state):
        self.root = self.Node(new_initial_state, None, None, 0, self)
    ### Q-Learning ###
    def q_update(self, node):
        if not self.learning:
            return
        if self.is_goal(node.state):
            if not self.update_nonterminals:
                return
            # self.q_update_terminal(node)
            solution_cost = node.path_cost
            self.q_values[q_key(node)] = 0
            
        else:
            solution_cost = f(node)
            self.q_values[q_key(node)] = q_learning(node)
        
        
        current_node = node
        
        while current_node.parent is not None:
            
            # Initialize the parent, this is the node which is having its 
            # Q value updated
            current_node_parent = current_node.parent
            
            # What is the Q value for the next action from the next state
            # Q(S', A')
            future_q_value = self.q_values[q_key(current_node)]  # Q value from the child

            # Calculate the reward, which is the travel time from parent to child
            reward = current_node.path_cost - current_node_parent.path_cost
            self.episode_rewards.append(reward)
            
            # What should the Q-value be? (Bellman equation)
            # R + gQ(S', A')
            target = reward + self.q_discount_factor * future_q_value

            # How wrong was our current estimate?
            # R + gQ(S', A') - Q(S, A)
            temporal_difference = target - self.q_values[q_key(current_node_parent)]
            # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)

            # Update our estimate in the direction of the error
            # Learning rate controls how big steps we take
            # Q(S,A) <- Q(S,A) + a[R + gQ(S',A') - Q(S,A)]
            self.q_values[q_key(current_node_parent)] = (
                self.q_values[q_key(current_node_parent)] + self.q_learning_rate * temporal_difference
            )

            if q_key(current_node_parent) in self.q_tracker.keys():
                self.q_tracker[q_key(current_node_parent)] += 1
            else: 
                self.q_tracker[q_key(current_node_parent)] = 1
            
            # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)
            
            # Iterate
            current_node = current_node_parent
        


    def check_collision(self, state):
        if self.agent_size_r == 0: return state in self.obstacle_map
        # initialize at r=0 and node to check if the node itself is an obstacle
        max_r = self.agent_size_r
        r = 0
        r_neighbors = {state}
        while r_neighbors and r <= max_r: 
            if r_neighbors & self.obstacle_map != set():
                return True
            r = r + 1
            r_neighbors = self.get_neighbors_at_radius(state,r)
        # if there are no neighbors at this r, then end the search because r is out of bounds for the map
        return False

    def get_neighbors_at_radius(self, center, radius):
        directions = [(-1,1),(-1,0),(0,-1), (1,-1), (1,0)]
        newloc = self.add_locations(center, tuple([a*radius for a in (1,0)]))
        locs = set()
        locs.add(newloc)
        for direction in directions:
            for i in range(radius):
                newloc = self.add_locations(newloc, direction)
                locs.add(newloc)
        return locs
        
    def hex_manhattan_distance(self, hex1, hex2):
        q1, r1 = hex1
        q2, r2 = hex2

        dq = abs(q1-q2)
        dr = abs(r1-r2)
        ds = abs((q2 + r2) - (q1 + r1))

        return (dq+dr+ds)/2

    # get the path cost of applying action to state
    def action_cost(self, state, action):
        u = state[self.state_dict['velocity']][0]
        a = self.a_max
        s = sqrt(3) * self.hex_size
        return self.get_travel_time(u, a, s)

    # calculate the travel time given initial velocity u, acceleration a, and distance s
    def get_travel_time(self, u, a, s):
        return (-u + sqrt(u*u + 4 * a * s))/a

    # check if the agent location in the state is the same as the goal location in the problem
    def is_goal(self, state):
        return state[self.state_dict['agent']] == self.goal_loc

    # Add two location tuples element-wise
    def add_locations(self, loc1, loc2):
        return tuple([sum(x) for x in zip(loc1,loc2)])

    # return a list of viable actions from state
    # an action is a direction tuple, eg (1, 0) is the hex to the right of the agent's location
    def actions(self, state):

        # Get the current direction of the agent in state
        # Cast to an int so it can be used to index neighborhood_angles.keys()
        # Coincidently, the values of the 6 directions correspond to their indices when casted this way
        current_angle = state[self.state_dict['velocity']][1]
        permutations = list(self.neighborhood_angles.keys())
        
        if current_angle is None:
            actions = permutations
        else:
            # Find the index values of the immediate turns in the clockwise (cw) and counter clockwise (ccw) directions
            ccw_turn_idx = current_angle + 1
            cw_turn_idx  = current_angle - 1
    
            # If these values are out of bounds, cycle to the front or back of the keys list
            if ccw_turn_idx > 5: ccw_turn_idx = 0 
            if cw_turn_idx  < 0: cw_turn_idx  = 5
    
            # Find the 3 actions corresponding to the straight, cw, ccw movements
            turns = [ccw_turn_idx, current_angle, cw_turn_idx]
            actions = [permutations[t] for t in turns]


        

        # Remove obstacles and out of bounds locations from the action list
        actions = [a for a in actions 
                   # if self.add_locations(a, state[self.state_dict['agent']]) not in self.obstacle_map # New hex is not an obstacle
                   if not self.check_collision(self.add_locations(a, state[self.state_dict['agent']]))
                   and self.hex_manhattan_distance(self.add_locations(a, state[self.state_dict['agent']]), (0,0)) <= self.hex_radius
                  ]
        
        return actions

    # returns the result of applying action to state
    # this is done by adding action to agent location in state 
    # agent_loc: (3,4)
    # action: (1,0)
    # result_loc (4,4)

    # also calculate the new velocity and angle after moving to the new location
    def result(self, state, action):

        # add the action tuple to the agent_loc tuple to get the new location
        new_agent_loc = self.add_locations(state[self.state_dict['agent']], action)

        params = {
            'u': state[self.state_dict['velocity']][0],
            'a': self.a_max,
            's': sqrt(3)*self.hex_size
        }
        # get the new velocity after applying action
        new_velocity = self.calculate_velocity(**params)
        new_angle = self.neighborhood_angles[action]

        return (
            new_agent_loc,
            (new_velocity, new_angle)
        )

    # calculate the velocity with initial velocity u, acceleration a, over distance s
    def calculate_velocity(self, u, a, s):
        return sqrt(u*u + 2*a*s)
    
    def get_benchmarks(self):
        return (self.heuristic_consistent_flag, self.num_expanded_states, self.num_generated_nodes)

    def save_q(self, q_values, q_tracker, filename=None):
        if filename is None:
            filename = self.q_values_filename
        with open(filename, 'wb') as file:
            pkl.dump((q_values, q_tracker), file)
    
    def load_q(self, filename=None):
        if filename is None:
            filename = self.q_values_filename
        with open(filename, 'rb') as file:
            return pkl.load(file)

    def reset_q(self):
        self.q_values = defaultdict(float) # Holds 1 value for the predicted heuristic
        self.q_tracker = defaultdict(float)
    