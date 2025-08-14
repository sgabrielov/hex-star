from math import sqrt, pi, inf
from queue import PriorityQueue
from itertools import count
from copy import copy

a_star_weight = 1
counter = count()

# Best first search algorithm
# using g(n) + h(n) for f will make this an A* search
def best_first_search(problem, f, h):
    node = problem.root
    frontier = PriorityQueue(0)

    # Priority queue, elements input as tuple (priority, counter, value), using f(node) for priority
    # Counter is used as the tiebreaker between nodes when they have the same f value
    # Nodes are expanded in order they were added when they have the same priority
    frontier.put((f(node, h), -next(counter), node))
    reached = {problem.root: node}

    while not frontier.empty():
        # Pop from the front of the queue
        node = frontier.get(False)[2]

        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node, h):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.put((f(child, h), -next(counter), child))
    return None

def inf_bidirectional_search(problem, f, f2, h, h2):
    #state, frontier, and reached declaraiton
    #node_f = problem.Node((problem.root.state[0], (0,0)), None, None, 0, problem)
    node_f = problem.Node(problem.root.state, None, None, 0, problem)
    #print("Init: ", node_f.state[0])
    node_g = problem.Node(problem.goal.state, None, None, 0, problem)
    #print("Init: ", node_g.state[0])
    frontier_f = PriorityQueue(0)
    frontier_f.put((f(node_f, h), -next(counter), node_f))
    frontier_g = PriorityQueue(0)
    frontier_g.put((f2(node_g, h2), -next(counter), node_g))
    reached_f = {node_f.state: node_f}
    reached_g = {node_g.state: node_g}
    #reached_f2 = {node_f.state[0]: node_f}
    #reached_g2 = {node_g.state[0]: node_g}
    reached_f2 = {
        (
            node_f.state[0], # location
            int(node_f.state[1][1]) # direction
        ): node_f
    }
    reached_g2 = {
        (
            node_g.state[0], # location
            int(node_g.state[1][1]) # direction
        ): node_g
    }
    direction = ""  
    solution = None
    while not frontier_f.empty() and not frontier_g.empty():
        #problem could be from pulling from fontier items
        if frontier_f.queue[0][0] < frontier_g.queue[0][0]:
            direction = "f"
            node_f = frontier_f.get(False)[2]
            if ( node_f.state[0], int((node_f.state[1][1] + pi) % (2 * pi) )) in reached_g2:
                return Termination(direction, problem, node_f, reached_g, h)
            inf_bidirectional_proceed(direction, problem, node_f, frontier_f, reached_f, reached_f2, h)
        else:
            direction = "b"
            node_g = frontier_g.get(False)[2]
            if ( node_g.state[0], int((node_g.state[1][1] + pi) % (2 * pi) )) in reached_f2:
                return Termination(direction, problem, node_g, reached_f, h2)
            inf_bidirectional_proceed(direction, problem, node_g, frontier_g, reached_g, reached_g2, h2)
    return solution

def join_condition(node, reached):
    # try:
        
    #     reached_node = reached[(node.state[0], int((node.state[1][1] + pi) % (2 * pi) ))]
    #     parent = node.parent
    #     reached_parent = reached[(parent.state[0], int((parent.state[1][1] + pi) % (2 * pi) ))]
    #     # reached_parent = reached_node.parent
    #     # parent_match = reached_parent == (parent.state[0], int((parent.state[1][1] + pi) % (2 * pi) ))
    #     return reached_parent.parent == reached_node
    # except:
    #     return False

    return (node.state[0], int((node.state[1][1] + pi) % (2 * pi) )) in reached
    


def Termination(direction, problem, node, reached, h):
    print("meeting point: ", node.state[0])
    temp = {key: val for key, val in reached.items() if val.state[0] == node.state[0]}
    temp_queue = PriorityQueue(0)
    for key, val in temp.items():
        if direction == "f":
            temp_queue.put((f(temp[key], h), val))
        else:
            temp_queue.put((f2(temp[key], h), val))
    if direction == "f":
        return bi_join_nodes(node, temp_queue.get(False)[1])
    else:
        return bi_join_nodes(temp_queue.get(False)[1], node)

def inf_bidirectional_proceed(direction, problem, node, frontier, reached, reached2, h):
    if node.state[0] not in reached2:
        reached2[(node.state[0], int(node.state[1][1]))] = node
    for child in expand(problem, node, h):
        s = child.state
        if s not in reached or child.path_cost < reached[s].path_cost:
            #print(dir, s[0])
            reached[s] = child
            if direction == "f":
                frontier.put((f(child, h), -next(counter), child))
            else:
                frontier.put((f2(child, h), -next(counter), child))
            #print("placed", s[0], child.path_cost)
            #print("replaced", s[0], reached[s[0]].path_cost, child.path_cost)
    return None

def bi_join_nodes(child_f, child_b):
    #print("stuck on bijoin")
    temp = child_f.path_cost + child_b.path_cost
    current_node = child_b.parent
    next_node_b = current_node.parent
    parent_node = child_f
    current_node.parent = parent_node
    if next_node_b is None:
        return current_node
    #print("parent_node", parent_node.state[0])
    #print("current_node", current_node.state[0])
    
    while next_node_b is not None:
        #print("current_node", current_node.state[0])
        parent_node = current_node
        current_node = next_node_b
        next_node_b = next_node_b.parent
        current_node.parent = parent_node
    #print("current_node", current_node.state[0])
    return current_node

#maybe try bi-linked list for optimization
def expand(problem, node, h):
    s = node.state
    nodes = []
    for action in problem.actions(s):
        s_new = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action)
        new_node = problem.Node(s_new, node, action, cost, problem)
        if problem.heuristic_consistent_flag:
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

def time_to_start(node):
    problem = node.problem
    goal_loc = problem.root.state[0]
    agent_loc = node.state[problem.state_dict['agent']]
    v, theta_a = node.state[problem.state_dict['velocity']]
    a = problem.d_max
    s = problem.hex_manhattan_distance(goal_loc, agent_loc) * sqrt(3) * problem.hex_size
    return problem.get_travel_time(v, a, s)

def check_h_consistency(problem, node, h):
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

def f(node, h = time_to_goal):
    return g(node) + a_star_weight * h(node)

def f2(node, h = time_to_start):
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
            self.r=0

            # [1 - 12]

            # 1 -> 0 r
            # 2 -> 1/6 pi
            self.route_direction = self.get_path_direction()

            # if the new node has a different angle than previous
            # the agent turned, apply the turning penalty and backtrack to update velocities and path costs along the path
            if self.parent is not None and self.route_direction != self.parent.route_direction:
                r = self.get_turning_radius()
                self.r = r
                # calculate the max velocity for the turn
                v_max = sqrt(r * problem.ay_max)
                if state[1][0] > v_max:
                    # update the velocity in the current node to v_max
                    self.state = (
                        self.state[0],
                        (v_max, self.state[problem.state_dict['velocity']][1])
                    )
                self.update_velocity(problem)

        
        def get_node_direction(self):
            return self.state[self.problem.state_dict['velocity']][1]
        def get_path_direction(self):
            # returns an index representing the macro direction of the path that a node lies on

            # 0:  0
            # 1:  pi/6    # 30 deg
            # 2:  pi/3    # 60 deg
            # 3:  pi/2    # 90 deg
            # 4:  2pi/3   # 120 deg
            # 5:  5pi/6   # 150 deg
            # 6:  pi      # 180 deg
            # 7:  7pi/6   # 210 deg
            # 8:  4pi/3   # 240 deg
            # 9:  3pi/2   # 270 deg
            # 10: 5pi/3   # 300 deg
            # 11: 11pi/6  # 310 deg

            node_direction_idx_dict = {
                0:       0,
                pi/6:    1,
                pi/3:    2,
                pi/2:    3,
                2*pi/3:  4,
                5*pi/6:  5,
                pi:      6,
                7*pi/6:  7,
                4*pi/3:  8,
                3*pi/2:  9,
                5*pi/3:  10,
                11*pi/6: 11,
            }


            # If this node has no parent, then it is the root node and the path direction is just the starting velocity direction
            if self.parent is None:
                print(self)
                return -1
            parent = self.parent
            # If this node has no grandparent, then it is the second node in the path, which means we do not know yet if it is part of a 
            #   zig-zag path or a normal path. In order to avoid unfairly penalizing paths that happen to start in zig-zag directions, we
            #   should assume that the 2nd node has not yet resulted in a change of direction.
            # We will know for sure by the time we expand the 3rd node, and at that point we can update velocities accordingly.

            # 3 hexes in a row are identified by the case when the node vector is the same as the parent vector
            if self.get_node_direction() == parent.get_node_direction():
                return node_direction_idx_dict[self.get_node_direction()]

            # If there is a turn
            if self.get_node_direction() != parent.get_node_direction():
                if self.parent.parent is None:
                    return -1
                grandparent = self.parent.parent

                # when the parent is -1, we can't adjust based on the parent, we have to calculate
                if parent.route_direction == -1:
                    if self.get_node_direction() == grandparent.get_node_direction():
                        node_dir = node_direction_idx_dict[self.get_node_direction()]
                        print(f'node_dir: {node_dir}')
                        
                        p_dir = node_direction_idx_dict[parent.get_node_direction()]
                        print(f'p_dir: {p_dir}')
                        new_dir = (node_dir+p_dir)//2
                        parent.route_direction = new_dir
                        print(f'new_dir: {new_dir}')
                        
                        return (new_dir)//2
                        
                # Otherwise we make sure we are not simply continuing a zigzag
                if self.get_node_direction() != grandparent.get_node_direction():
                    turn_amount = 1

                    # If the grandparent has a difference of 2 hex turns from the node, it means that 
                    if abs(grandparent.get_node_direction() - self.get_node_direction()) > pi/3:
                        turn_amount = 2
                    # Then update the route direction based on which way the turn was
                    if self.is_left_turn():
                        return (parent.route_direction +turn_amount) % 12
                    else:
                        return (parent.route_direction +12 - turn_amount) % 12
                        
            return parent.route_direction

        def is_left_turn(self, mod=2*pi):

                # returns true if node makes a left turn compared to parent
                node_dir = self.state[1][1]
                parent_dir = self.parent.state[1][1]
                return (node_dir - parent_dir) % mod < mod // 2
            
        def get_turning_radius(self):
            r = 0
            current_node = self.parent
            while current_node.parent is not None and current_node.route_direction == self.parent.route_direction:
                current_node = current_node.parent
                r += 1
            if abs(self.route_direction - self.parent.route_direction) == 1: r*=2
            return r
        def update_velocity(self, problem):
            current_node = self
            while current_node.parent:
                
                new_v = problem.calculate_velocity(current_node.state[problem.state_dict['velocity']][0], problem.d_max, sqrt(3) * problem.hex_size)
                #print(new_v)
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
    def __init__(
        self, 
        initial_state_f, 
        initial_state_b, 
        hex_map, 
        obstacle_map, 
        goal_loc, 
        hex_radius, 
        hex_size, 
        agent_size_r, 
        acceleration_max, 
        deceleration_max, 
        lat_acceleration_max
    ):
        # Defines the indices for the components of the state
        self.state_dict = {
            'agent': 0,
            'velocity': 1,
        }
        
        self.hex_map = hex_map
        self.obstacle_map = obstacle_map
        self.goal_loc = goal_loc
        self.hex_radius = hex_radius
        self.hex_size = hex_size
        self.a_max = acceleration_max
        self.d_max = deceleration_max
        self.ay_max = lat_acceleration_max
        self.agent_size_r = agent_size_r
        self.root = self.Node(initial_state_f, None, None, 0, self)
        init_loc, init_v = initial_state_b
        init_s, init_d = init_v
        
        initial_state_b = (
            init_loc,
            (init_s, init_d + pi)
        )
        self.goal = self.Node(initial_state_b, None, None, 0, self)
        

        # for benchmarking
        self.heuristic_consistent_flag = True
        self.num_expanded_states = 0
        self.num_generated_nodes = 0


        # Defines the directions to the immediate neighborhood and their angle relative to horizontal
        self.neighborhood_angles = {
            (1,0): 0,
            (0,1): pi/3,
            (-1,1): 2*pi/3,
            (-1,0): pi,
            (0,-1): 4*pi/3,
            (1,-1): 5*pi/3   
        }
    def check_collision(self, state):
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
    def hex_manhattan_distance_2(self, hex1, hex2):
        print("\nhex1: ", hex1, type(hex1), "\nhex2: ", hex2, type(hex2))
        q1, r1 = hex1
        q2, r2 = hex2

        dq = abs(q1-q2)
        dr = abs(r1-r2)
        ds = abs(-q1-q2-r1-r2)

        return (dq+dr+ds)/2

    def hex_manhattan_distance(self, hex1, hex2):
        q1, r1 = hex1
        q2, r2 = hex2
        return (abs(q1 - q2) 
          + abs(q1 + r1 - q2 - r2)
          + abs(r1 - r2)) / 2

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
        current_angle = int(state[self.state_dict['velocity']][1])

        # Find the index values of the immediate turns in the clockwise (cw) and counter clockwise (ccw) directions
        ccw_turn_idx = current_angle + 1
        cw_turn_idx  = current_angle - 1

        # If these values are out of bounds, cycle to the front or back of the keys list
        if ccw_turn_idx > 5: ccw_turn_idx = 0 
        if cw_turn_idx  < 0: cw_turn_idx  = 5

        # Find the 3 actions corresponding to the straight, cw, ccw movements
        permutations = list(self.neighborhood_angles.keys())
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

        