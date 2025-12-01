from math import sqrt, pi
from queue import PriorityQueue
from itertools import count
from copy import copy
from collections import defaultdict, deque
import numpy as np
import pickle as pkl
import random

import torch
import torch.nn as nn
import torch.optim as optim

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
            problem.deepq_update(node)
            
            solutions.append(node)
            
            if len(solutions) >= max_solutions:
                for f_cost, _, nonterminal_node in list(frontier.queue):
                    problem.deepq_update(nonterminal_node)
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

def get_state_info(node):
    state,velocity = node.state
    q,r = state
    v,a = velocity
    H = node.problem.hex_size
    gq, gr = node.problem.goal_loc
    obs_map = node.problem.obstacle_map

    return q,r,H,v,a, gq, gr, obs_map

def count_obstacles_hex_line(obstacle_map, start, end):
    """
    Count obstacles along the hex line between two axial coordinates
    using cube coordinates interpolation.

    Parameters:
        obstacle_map: set of axial coordinates that are obstacles
        start, end: tuples (q, r)

    Returns:
        int: number of obstacles along the line
    """
    def axial_to_cube(q, r):
        s = -q - r
        return (q, r, s)
    
    def cube_lerp(a, b, t):
        return (a[0] + (b[0]-a[0])*t,
                a[1] + (b[1]-a[1])*t,
                a[2] + (b[2]-a[2])*t)
    
    def cube_round(c):
        x, y, z = c
        rx = round(x)
        ry = round(y)
        rz = round(z)

        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)

        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        return (rx, ry, rz)
    
    def cube_to_axial(c):
        q, r, s = c
        return (q, r)

    # Convert to cube coordinates
    start_cube = axial_to_cube(*start)
    end_cube = axial_to_cube(*end)

    # Distance in cube coordinates
    N = int(max(abs(end_cube[0]-start_cube[0]), abs(end_cube[1]-start_cube[1]), abs(end_cube[2]-start_cube[2])))

    obstacles_count = 0
    for i in range(N+1):
        t = i / max(N, 1)
        interp = cube_lerp(start_cube, end_cube, t)
        rounded = cube_round(interp)
        axial = cube_to_axial(rounded)
        if axial in obstacle_map:
            obstacles_count += 1
    
    return obstacles_count


def encode_hex_state(q, r, H, velocity, direction, goal_q, goal_r, obs_map, Vmax=50):
    """
    Encode a hex cell + velocity + direction + goal location as an ML feature vector.

    Parameters:
        q, r       - axial coordinates of agent
        H          - map scale normalization constant
        velocity   - scalar agent velocity
        direction  - integer 0..5 (pointy-top hex direction)
        goal_q,
        goal_r     - axial coordinates of the goal
        Vmax       - normalization constant for velocity

    Returns:
        np.float32 array feature vector
    """

    s = -q - r                     # cube coordinate 3rd axis
    goal_s = -goal_q - goal_r

    # --- agent base features ---
    features = [
        q / H,
        r / H,
        abs(q),
        abs(r),
        abs(s),
        np.sqrt(q*q + r*r),
    ]

    # --- velocity magnitude normalized ---
    v_norm = velocity / Vmax
    features.append(v_norm)

    # --- direction one-hot encoding (6 dirs) ---
    direction_onehot = np.zeros(6, dtype=np.float32)
    direction_onehot[direction] = 1.0
    features.extend(direction_onehot.tolist())

    # --------------------------------------------------
    # Goal-encoding features
    # --------------------------------------------------

    # normalized goal location
    features.append(goal_q / H)
    features.append(goal_r / H)

    # relative goal direction
    dq = (goal_q - q) / H
    dr = (goal_r - r) / H
    ds = (goal_s - s) / H

    features.extend([dq, dr, ds])

    # distance to goal in axial coords
    dist_to_goal = np.sqrt(dq*dq + dr*dr)
    features.append(dist_to_goal)

    # number of obstacles from state to goal
    state = (q,r)
    goal = (goal_q, goal_r)
    num_obs = count_obstacles_hex_line(obs_map, state, goal)
    features.append(num_obs)

    return np.array(features, dtype=np.float32)

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
        
def deepq2(node):
    problem = node.problem
    q,r,H,v,a,gq,gr = get_state_info(node)
    Vmax = 50

    node_fv = encode_hex_state(q,r,H,v,a,gq,gr,Vmax)
    state_tensor = torch.tensor(node_fv, dtype=torch.float32).to(problem.device)
    return problem.DQN(state_tensor) / problem.reward_scale_factor

def deepq3(node):
    problem = node.problem
    q, r, H, v, a, gq, gr = get_state_info(node)
    Vmax = 50

    node_fv = encode_hex_state(q, r, H, v, a, gq, gr, Vmax)
    state_tensor = torch.tensor(node_fv, dtype=torch.float32).to(problem.device)

    # Inference ONLY — does not affect training graph
    with torch.no_grad():
        value = problem.DQN(state_tensor).item()

    return value

def deepq(node, use_grad=False):
    """
    Compute the predicted value V(s) for a node's state.

    Parameters:
        node      - PathfindingProblem.Node
        use_grad  - If True, keep gradient for backprop. If False, inference only.

    Returns:
        torch.Tensor scalar value
    """
    problem = node.problem

    Vmax = 50
    node_fv = encode_hex_state(*get_state_info(node), Vmax)
    state_tensor = torch.tensor(node_fv, dtype=torch.float32, device=problem.device).unsqueeze(0)  # shape [1, state_dim]

    if use_grad:
        # Keep gradients
        value = problem.DQN(state_tensor).squeeze(0)  # shape []
    else:
        # Inference only
        with torch.no_grad():
            value = problem.DQN(state_tensor).squeeze(0)

    return (value / problem.reward_scale_factor).item()




# Because A* needs a decent heuristic to work quickly, 
# H* will use the max of time_to_goal and q_learning 
# For A* max(h1, h2) is consistent when both h1 and h2 are consistent 
def combo_h(
    node,
    h1 = time_to_goal,
    h2 = deepq
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

class HexStarDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_sizes = [64, 32, 8], device='cpu'):
        super().__init__()
        layers = []

        prev_size = state_dim

        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size

        layers.append(nn.Linear(prev_size, action_dim))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.model(x)

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
        q_learning_rate = 0.001,  
        learning_rate_adam = 0.001,
        q_discount_factor = 0.95,
        q_values_filename = None,
        q_values_counts = None,
        update_nonterminals = False,
        learning=False,
        tau = 1,
        batch_size = 256,
        target_update_rate = 4,
        replay_buffer_capacity = 10000,
        network_architecture = [64, 32, 8],
        reward_scale_factor = 10,
    ):
        self.root = self.Node(initial_state, None, None, 0, self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

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
        self.reward_scale_factor = reward_scale_factor

        # Get a sample state feature vector by converting the root node
        # And use it to set the input dimension
        self.state_size = len(encode_hex_state(*get_state_info(self.root)))
        self.action_size = 1 # Predicting h values, not actions

        # Initialize DQN and target networks, move to GPU if available
        self.DQN = HexStarDQN(self.state_size, self.action_size, network_architecture, device=self.device).to(self.device)
        self.DQN_target = HexStarDQN(self.state_size, self.action_size, network_architecture, device=self.device).to(self.device)
        
        self.q_tracker = defaultdict(float)
        self.q_values = defaultdict(float)

        self.replay_buffer = deque(maxlen=replay_buffer_capacity)
        
        self.lr = q_learning_rate
        self.lr_adam = learning_rate_adam
        self.q_discount_factor = q_discount_factor  # How much we care about future rewards
        self.tau = tau                          # Soft update interpolation factor

        self.batch_size = batch_size            # How many samples to send to the networks at once
        self.target_update_rate = target_update_rate  # How often to update the target network from the Q network

        self.q_learning_rate = q_learning_rate 
        self.q_discount_factor = q_discount_factor
        self.training_error = []

        self.optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr_adam)

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
    def deepq_update(self, node):
        if not self.learning:
            return

        
        
        if self.is_goal(node.state):
            if not self.update_nonterminals:
                return
            reward = 0
            state_fv = np.array(encode_hex_state(*get_state_info(node)), dtype=np.float32)
            next_state = np.zeros_like(state_fv, dtype=np.float32)
            sample = (state_fv, 0, reward, next_state, 1)
            self.replay_buffer.append(sample)      
        
        current_node = node
        
        while current_node.parent is not None:
            
            # Initialize the parent, this is the node which is having its 
            # Q value updated
            current_node_parent = current_node.parent

            state_fv = np.array(encode_hex_state(*get_state_info(current_node_parent)), dtype=np.float32)
            next_state_fv = np.array(encode_hex_state(*get_state_info(current_node)), dtype=np.float32)
            
            # What is the Q value for the next action from the next state
            # Q(S', A')
            # future_q_value = deepq(current_node)  # Q value from the child

            # Calculate the reward, which is the travel time from parent to child
            reward = (current_node.path_cost - current_node_parent.path_cost) * self.reward_scale_factor
            self.episode_rewards.append(reward)

            sample = (
                state_fv, 
                0,
                reward,
                next_state_fv,
                0
            )

            self.replay_buffer.append(sample)



            self.q_tracker[q_key(current_node_parent)] +=1
            
                
            # Iterate
            current_node = current_node_parent

            # If we are on a target update step
                # Apply a soft update from DQN -> DQN_target linearly interpolated based on tau
            if len(self.replay_buffer) >= self.batch_size:
                samples = random.sample(self.replay_buffer, self.batch_size)
                self.update(samples)
        self.soft_update(self.DQN, self.DQN_target, self.tau)
    def update2(self, samples):
        """
        Update DQN using a batch of replay samples.
        Uses a V(s)-only network (no target network).
        """
        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)
    
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
    
        # Compute V(s) for current states
        v_pred = self.DQN(states).squeeze(1)  # shape [batch_size]
    
        # Compute TD target: r + gamma * V(s')
        v_next = self.DQN(next_states).squeeze(1)
        with torch.no_grad():
            td_target = rewards + (1 - dones) * self.q_discount_factor * v_next
    
        # Compute loss
        loss = nn.MSELoss()(v_pred, td_target)
    
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=10)  # optional: stabilize training
        self.optimizer.step()
    
        # Track loss
        self.training_error.append(loss.item())

    def update(self, samples):

        # Unpack the samples        
        states, actions, rewards, next_states, dones = zip(*samples)
        #print("Sample reward range:", np.min(rewards), np.max(rewards))
        
        # Conver to np.array -> convert to tensor -> move to cuda device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)




        v_pred = self.DQN_target(next_states).squeeze(1)
        with torch.no_grad():       
            td_target = rewards + (1 - dones) * self.q_discount_factor * v_pred

        #print("NaN in v_pred:", torch.isnan(v_pred).any().item())
        #print("NaN in td_target:", torch.isnan(td_target).any().item())

        # Calculate loss gradient with Mean Squared Error
        # Detatch from td_target to prevent updates to target network
        loss = nn.MSELoss()(v_pred, td_target.detach())

        # Backpropagate and update weights
        self.optimizer.zero_grad()

        #print("v_pred.requires_grad:", v_pred.requires_grad)
        #print("any DQN parameter requires_grad:", any(p.requires_grad for p in self.DQN.parameters()))
        #print("td_target.requires_grad:", td_target.requires_grad)

        loss.backward()
        self.optimizer.step()

        self.training_error.append(loss.item())
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)      

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

    def save_models(self, path):
        """
        Save the DQN and target DQN models to the specified path.
        """
        torch.save({
            'dqn_state_dict': self.DQN.state_dict(),
            'target_dqn_state_dict': self.DQN_target.state_dict()
        }, path)
        print(f"Models saved to {path}")

    def save_benchmarks(self, path):
        """
        Save benchmarks to the specified path
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'training_error': self.training_error,
                'training_rewards': self.training_rewards,
                'episode_durations': self.episode_durations,
                        }, f)
    def load_benchmarks(self, path):
        """
        Load benchmarks from the specified path
        """
        import pickle
        with open(path, 'rb') as f:
            benchmarks = pickle.load(f)
            
        self.training_error = benchmarks['training_error']
        self.training_rewards = benchmarks['training_rewards']
        self.episode_durations = benchmarks['episode_durations']
            
    def load_models(self, path):
        """
        Load the DQN and target DQN models from the specified path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.DQN.load_state_dict(checkpoint['dqn_state_dict'])
        self.DQN_target.load_state_dict(checkpoint['target_dqn_state_dict'])
        print(f"Models loaded from {path}")

    def save_agent(self, path, bpath):
        
        self.save_models(path)
        self.save_benchmarks(bpath)

    def load_agent(self, path, bpath):

        load_models(path)
        load_benchmarks(bpath)


    def get_q_values(self, velocity = 10):
        free_hexes =[p for p in self.hex_map if p not in self.obstacle_map and p != self.goal_loc]
        for loc in free_hexes:
            v = velocity
            for a in range(6):
                loc_node = self.Node((loc, (v,a)), None, None, 0, self)
                #qkey = (loc, self.goal_loc, a)
                self.q_values[q_key(loc_node)] = deepq(loc_node)
        return self.q_values
    
