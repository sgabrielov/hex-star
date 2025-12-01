# %%
# Date: Nov-30-2025
# Author: Simon Chan Tack
# File: qrl_hexstar_eps_vs_ucb_gt.py
# Code summary:
# ==================================================================================
# Environment: HexStar (H*) dynamics wrapped in RLHexStarEnv using class RLHexStarEnv
#
# Comparison of ε-greedy vs UCB exploration
#   - Tabular Q-learning
#   - Deep Q-Network (DQN)
#
# ==================================================================================

import os
import pickle
import random
from collections import defaultdict, deque
from math import sqrt, cos, sin, pi

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import plotly.graph_objects as go

from hex_star import PathfindingProblem, time_to_goal, best_first_search
from hex_star import f as hstar_f  # cost function for H* (g + h)


# ============================================================
# 1. Heuristic Network (used for reward shaping and H* heuristic)
# ============================================================

class HeuristicNet(nn.Module):
    # ###############################################################
    # Simple MLP that predicts time-to-goal h_theta(s) from:
    #   [q, r, v, hex_manhattan_distance_to_goal]
    # ###############################################################
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # returns scalar per sample
        return self.net(x).squeeze(-1)


def state_to_heuristic_features(state, goal):
    # ###############################################################
    # Convert an H* state into features for the heuristic net.

    # state = ( (q,r), (v, theta) )
    # goal  = (q_goal, r_goal)
    # Features:
    #   [q, r, v, dhm]
    # where dhm is a hex distance-like feature (monotonic with goal distance).
    # ###############################################################
    (q, r), (v, theta) = state

    dq = abs(goal[0] - q)
    dr = abs(goal[1] - r)
    ds = abs((-goal[0] - goal[1]) - (-q - r))
    # This is a hex distance-like feature; the NN only needs something
    # monotonic with distance, not the exact formula.
    dhm = max(dq, dr, ds)
    # Alternative exact hex distance would be: (dq + dr + ds) / 2

    return np.array([q, r, v, dhm], dtype=np.float32)


def load_heuristic_model(model_path, device=None):
    # ###############################################################
    # Load a previously trained heuristic network from disk.
    # ###############################################################
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeuristicNet(input_dim=4).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# ============================================================
# 2. RL Environment Wrapper: RLHexStarEnv
# ============================================================
# This interface H* search and creates elements required for a reinforcement learning environment
class RLHexStarEnv:
    # ###############################################################
    # RL environment built on top of PathfindingProblem (H*).
    # - State: H* state = (agent_loc, (velocity, angle))
    # - Actions: 6 global hex directions (but only a subset is valid at each state).
    # This class:
    #   * Loads the .pkl map
    #   * Wraps H* step dynamics
    #   * Computes shaped rewards using the learned heuristic (if provided)
    # ###############################################################

    def __init__(
        self,
        pkl_path,
        heuristic_model_path=None,
        shaping_gamma=0.99,
        step_penalty=0.0,
        goal_reward=100.0,
        max_episode_steps=100,
    ):
        # Load map
        with open(pkl_path, "rb") as f:
            M = pickle.load(f)

        self.hex_map      = M["hex_map"]
        self.obstacle_map = M["obstacle_map"]
        self.default_start = M["agent"]
        self.goal          = M["goal"]
        self.init_velocity = M["velocity"]  # (v, angle)
        self.hex_radius    = M["hex_radius"]
        self.hex_size      = M["hex_size"]

        # Dynamics parameters for RL (you can tune these)
        self.accel_max = 1.0
        self.decel_max = 1.0
        self.lat_accel_max = 0.5

        # Build a PathfindingProblem to reuse H* actions/transitions
        v0, theta0 = self.init_velocity[:2]
        initial_state = (self.default_start, (v0, theta0))

        self.problem = PathfindingProblem(
            initial_state,
            self.hex_map,
            self.obstacle_map,
            self.goal,
            self.hex_radius,
            self.hex_size,
            self.accel_max,
            self.decel_max,
            self.lat_accel_max,
        )

        # Define all actions
        # Fixed global action list: 6 neighbor directions on hex grid
        self.actions_all = list(self.problem.neighborhood_angles.keys())
        self.num_actions = len(self.actions_all)

        # Reward shaping parameters
        self.shaping_gamma = shaping_gamma
        self.step_penalty  = step_penalty
        self.goal_reward   = goal_reward
        self.max_episode_steps = max_episode_steps

        # Load heuristic model for shaping (optional)
        self.heuristic_model = None
        self.device = None
        if heuristic_model_path is not None:
            self.heuristic_model, self.device = load_heuristic_model(
                heuristic_model_path
            )

        self.state = None
        self.steps = 0

    # ---------- basic utilities ----------

    def _random_free_start(self):
        # ###############################################################
        # Pick a random free (non-obstacle) cell as starting location.
        # ###############################################################
        free_cells = [c for c in self.hex_map if c not in self.obstacle_map]
        return random.choice(free_cells)

    def _heuristic_value(self, state):
        # ###############################################################
        # Evaluate heuristic h_theta(s) if network exists, otherwise
        # fall back to analytic time_to_goal heuristic.
        # ###############################################################
        if self.heuristic_model is None:
            node = self.problem.Node(state, None, None, 0, self.problem)
            return time_to_goal(node)

        feat = state_to_heuristic_features(state, self.goal)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.heuristic_model(x).item())

    def _state_key(self, state):
        # ###############################################################
        # Discretize state for tabular Q-learning: (q, r, heading_index)
        # heading_index is one of 0..5 corresponding to 6 hex directions.
        # ###############################################################
        (q, r), (v, theta) = state
        angles = list(self.problem.neighborhood_angles.values())
        idx = min(range(len(angles)), key=lambda i: abs(angles[i] - theta))
        return (int(q), int(r), idx)

    def valid_action_ids(self, state):
        # ###############################################################
        # Returns list of global action IDs that are valid from this state
        # (not out of bounds, not colliding with obstacles).
        # ###############################################################
        valid_tuples = set(self.problem.actions(state))  # subset of 3 or fewer
        ids = [i for i, a in enumerate(self.actions_all) if a in valid_tuples]
        return ids

    # ---------- RL interface ----------

    def reset(self, random_start=True):
        # ###############################################################
        # Reset environment for a new episode.
        # If random_start=True, agent starts from a random free cell.
        # Else, from the default start in the .pkl map.
        # ###############################################################
        if random_start:
            loc = self._random_free_start()
        else:
            loc = self.default_start

        v0, theta0 = self.init_velocity[:2]
        init_state = (loc, (v0, theta0))

        assert isinstance(init_state, tuple)
        assert isinstance(init_state[0], tuple)
        assert isinstance(init_state[1], tuple)

        self.state = init_state
        self.steps = 0
        return self.get_observation(self.state)

    def get_observation(self, state):
        # ###############################################################
        # Convert H* state to numeric observation vector for DQN:
        #   [q, r, v, dhm, h_theta(s)]
        # ###############################################################
        if isinstance(state, (list, np.ndarray)):
            # Safety check: user passed observation instead of state
            raise RuntimeError("get_observation() expects H* state, not observation vector")

        feat4 = state_to_heuristic_features(state, self.goal)
        h_val = self._heuristic_value(state)
        return np.concatenate([feat4, np.array([h_val], dtype=np.float32)], axis=0)

    def step(self, action_id):
        # ###############################################################
        # Apply an action index in {0..5}. If invalid, penalize and stay in place.
        # Returns: (next_obs, reward, done, info)
        # ###############################################################
        self.steps += 1
        done = False
        info = {}

        state = self.state
        valid_ids = self.valid_action_ids(state)

        if action_id not in valid_ids:
            # Invalid action: strong negative reward
            reward = -5.0 + self.step_penalty
            next_state = state
        else:
            action_tuple = self.actions_all[action_id]

            # Travel time cost for this step (physics-based)
            u = state[1][0]  # current velocity
            a = self.accel_max
            s = sqrt(3) * self.hex_size
            time_cost = self.problem.get_travel_time(u, a, s)

            # Base reward: negative time cost + optional small penalty
            base_reward = -time_cost + self.step_penalty

            # Next state from H* dynamics
            next_state = self.problem.result(state, action_tuple)

            # Sanity check
            try:
                (q, r), (v, theta) = next_state
            except Exception as e:
                print("INVALID NEXT STATE:", next_state)
                raise e

            # Potential-based reward shaping using heuristic
            # This concept taken form Andrew Ng et al 1999 Reward Shaping (SUPER SMART)
            if self.heuristic_model is not None:
                phi_s = -self._heuristic_value(state)
                phi_ns = -self._heuristic_value(next_state)
                reward = base_reward + self.shaping_gamma * phi_ns - phi_s
            else:
                reward = base_reward

            # Goal check
            if self.problem.is_goal(next_state):
                reward += self.goal_reward
                done = True

        # Episode limit
        if self.steps >= self.max_episode_steps:
            done = True

        self.state = next_state
        obs_next = self.get_observation(next_state)
        return obs_next, reward, done, info

    def get_state_key(self):
        # ###############################################################
        # Get discrete state key for current state (for Tabular Q).
        # ###############################################################
        return self._state_key(self.state)


# ============================================================
# 3. Tabular Q-Learning Agent (epsilon-greedy and UCB)
# ============================================================

class TabularQAgent:
    # ###############################################################
    # Tabular Q-learning agent with support for:
    #   - epsilon-greedy exploration
    #   - UCB (Upper Confidence Bound) exploration
    # ###############################################################
    def __init__(
        self,
        num_actions,
        gamma=0.99,
        alpha=0.1,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
    ):
        self.num_actions = num_actions
        self.gamma = gamma     # discount factor
        self.alpha = alpha     # learning rate

        # ε-greedy parameters
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Q-table: Q[state_key][action] = value
        self.Q = defaultdict(lambda: np.zeros(num_actions, dtype=np.float32))

        # Visit counts for UCB: N(s) and N(s,a)
        self.Ns  = defaultdict(int)
        self.Nsa = defaultdict(lambda: np.zeros(num_actions, dtype=np.int32))

    # ------------------------------
    # ε-Greedy action selection
    # ------------------------------
    def select_action_eps_greedy(self, state_key, valid_action_ids):
        # ###############################################################
        # epsilon-greedy:
        # - With prob epsilon: random valid action
        # - Else: greedy w.r.t Q(s,a)
        # ###############################################################
        if random.random() < self.eps:
            return random.choice(valid_action_ids)
        q_vals = self.Q[state_key]
        return max(valid_action_ids, key=lambda a: q_vals[a])

    # ------------------------------
    # UCB action selection
    # ------------------------------
    def select_action_ucb(self, state_key, valid_action_ids, c=2.0):
        # ###############################################################
        # UCB exploration:
        #   a = argmax_a [ Q(s,a) + c * sqrt(log(N(s)+1)/(N(s,a)+1)) ]
        # ###############################################################
        q_vals = self.Q[state_key]
        ns  = self.Ns[state_key]
        nsa = self.Nsa[state_key]

        scores = {}
        for a in valid_action_ids:
            bonus = c * np.sqrt(np.log(ns + 1) / (nsa[a] + 1))
            scores[a] = q_vals[a] + bonus

        return max(scores.keys(), key=lambda a: scores[a])

    # Greedy selection (for evaluation)
    def select_action_greedy(self, state_key, valid_action_ids):
        q_vals = self.Q[state_key]
        return max(valid_action_ids, key=lambda a: q_vals[a])

    # ------------------------------
    # Q-learning update
    # ------------------------------
    def update(self, s_key, a, r, s2_key, valid_next_ids, done):
        # ###############################################################
        # Standard Q-learning update:
        #   Q(s,a) <- Q(s,a) + alpha [ reward + gamma * max_a' Q(s',a') - Q(s,a) ]
        # ###############################################################
        q_sa = self.Q[s_key][a]

        if done or not valid_next_ids:
            target = r
        else:
            q_next = self.Q[s2_key]
            target = r + self.gamma * max(q_next[a2] for a2 in valid_next_ids)

        self.Q[s_key][a] += self.alpha * (target - q_sa)

        # Update visit counts for UCB
        self.Ns[s_key] += 1
        self.Nsa[s_key][a] += 1

    def decay_epsilon(self):
        # ###############################################################
        # Decay epsilon over time for epsilon-greedy.
        # ###############################################################
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_q_learning(env,
                     episodes=500,
                     mode="eps",  # "eps" or "ucb"
                     eps_start=1.0,
                     eps_end=0.05,
                     eps_decay=0.995):
    # ###############################################################
    # Train a TabularQAgent on RLHexStarEnv using either:
    #   - epsilon-greedy ("eps")
    #   - UCB ("ucb")
    # Returns: (agent, episode_returns_list)
    # ###############################################################
    agent = TabularQAgent(
        num_actions=env.num_actions,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )

    rewards_history = []   # For reward tracking
    success_history = []   # For success tracking
    for ep in range(episodes):
        _ = env.reset(random_start=True)
        s_key = env.get_state_key()
        total_reward = 0.0
        done = False
        reached_goal = False   # Track reaching goal

        dead_steps = 0   # <-- Fix for infinite loops
        while not done:
            valid_ids = env.valid_action_ids(env.state)

            # # If no valid actions → treat as terminal with penalty
            # if not valid_ids:
            #     r = -10.0
            #     done = True
            #     agent.update(s_key, 0, r, s_key, [], done)                
            #     total_reward += r
            #     break

            # Handle no valid actions
            if not valid_ids:
                r = -5.0
                agent.update(s_key, 0, r, s_key, [], False)
                total_reward += r

                dead_steps += 1
                if dead_steps > 20:   # bail-out threshold
                    done = True
                continue

            # Reset dead counter (agent moved)
            dead_steps = 0
                

            # Choose action based on exploration mode
            if mode == "eps":
                a = agent.select_action_eps_greedy(s_key, valid_ids)
            elif mode == "ucb":
                a = agent.select_action_ucb(s_key, valid_ids, c=2.5)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            _, r, done, _ = env.step(a)

            if env.problem.is_goal(env.state):
                reached_goal = True    # <-- MARK SUCCESS

            s2_key = env.get_state_key()
            valid_next_ids = env.valid_action_ids(env.state)

            agent.update(s_key, a, r, s2_key, valid_next_ids, done)
            s_key = s2_key
            total_reward += r

        # Decay epsilon only in epsilon-greedy mode
        if mode == "eps":
            agent.decay_epsilon()

        rewards_history.append(total_reward)               # Update rewards
        success_history.append(1 if reached_goal else 0)   # Update success

        if (ep + 1) % 50 == 0:
            print(f"[Tabular Q-{mode}] Episode {ep+1}/{episodes}, "
                  f"Return={total_reward:.2f}, eps={agent.eps:.3f}")

    return agent, rewards_history, success_history


# ============================================================
# 4. DQN Agent (epsilon-greedy and UCB)
# ============================================================

class QNet(nn.Module):
    # ###############################################################
    # Q(s,a) approximator with fixed number of actions.
    # Input: observation vector
    # Output: Q-values for each action (num_actions)
    # ###############################################################
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    # ###############################################################
    # DQN agent that supports:
    #   - epsilon-greedy exploration
    #   - UCB-style exploration over Q(s,a)
    # Uses:
    #   * Online network q_net
    #   * Target network target_net
    #   * Replay buffer for experience replay
    #   * Soft target updates (Polyak averaging)
    # ###############################################################
    def __init__(
        self,
        obs_dim,
        num_actions,
        gamma=0.99,
        lr=1e-3,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=1,
        tau=0.005,
    ):
        self.num_actions = num_actions
        self.gamma = gamma

        # ε-greedy parameters (only used in "eps" mode)
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNet(obs_dim, num_actions).to(self.device)      # Policy network
        self.target_net = QNet(obs_dim, num_actions).to(self.device) # Target network
        self.target_net.load_state_dict(self.q_net.state_dict())  # Copy weights from Policy to Target Network
        self.target_net.eval()

        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)
        self.step_count = 0

        # UCB visit counts in observation space
        self.Ns_obs  = defaultdict(int)
        self.Nsa_obs = defaultdict(lambda: np.zeros(num_actions, dtype=np.int32))

    # Helper: observation key for visit counting (round to 2 decimals)
    def _obs_key(self, obs, decimals=2):
        return tuple(np.round(obs, decimals))

    # ------------------------------
    # epsilon-greedy action selection
    # ------------------------------
    def select_action_eps(self, obs, valid_action_ids):
        # ###############################################################
        # epsilon-greedy in continuous observation space.
        ################################################################
        if random.random() < self.eps:
            return random.choice(valid_action_ids)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)[0].cpu().numpy()

        masked = np.full_like(q_vals, -1e9, dtype=np.float32)
        for i in valid_action_ids:
            masked[i] = q_vals[i]

        return int(np.argmax(masked))

    # ------------------------------
    # UCB-based action selection
    # ------------------------------
    def select_action_ucb(self, obs, valid_action_ids, c=1.0):
        # ###############################################################
        # UCB exploration using Q(s,a) + bonus based on observation visit counts.
        # We discretize obs to an obs_key and keep visit counts Ns, Nsa.
        # ###############################################################
        obs_key = self._obs_key(obs)
        ns  = self.Ns_obs[obs_key]
        nsa = self.Nsa_obs[obs_key]

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)[0].cpu().numpy()

        scores = {}
        for a in valid_action_ids:
            bonus = c * np.sqrt(np.log(ns + 1) / (nsa[a] + 1))
            scores[a] = q_vals[a] + bonus

        # Greedy w.r.t optimistic Q + bonus
        a = max(scores.keys(), key=lambda k: scores[k])

        # Update counts for next time
        self.Ns_obs[obs_key] += 1
        self.Nsa_obs[obs_key][a] += 1
        return int(a)

    # Greedy (no exploration, for evaluation)
    def select_action_greedy(self, obs, valid_action_ids):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)[0].cpu().numpy()
        masked = np.full_like(q_vals, -1e9, dtype=np.float32)
        for i in valid_action_ids:
            masked[i] = q_vals[i]
        return int(np.argmax(masked))

    def store(self, transition):
        # ###############################################################
        # Store transition in replay buffer.
        # transition = (obs, a, r, obs2, done, valid_next_ids)
        # ###############################################################
        self.replay.append(transition)

    def soft_update(self):
        # ###############################################################
        # Soft update target_net from q_net:
        #   theta_target <- tau * theta_online + (1-tau) theta_target
        # ###############################################################
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(),
                                           self.q_net.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def update(self):
        # ###############################################################
        # Perform one gradient step from replay buffer if enough samples.
        # ###############################################################
        if len(self.replay) < self.batch_size:
            return

        # Randomly select batch of experiences to train on
        batch = random.sample(self.replay, self.batch_size)
        obs, a, r, obs2, done, valid_next_ids_list = zip(*batch) # This gives state, action, reward, next_state, done, next_ids

        obs_t  = torch.tensor(obs,  dtype=torch.float32, device=self.device)
        a_t    = torch.tensor(a,    dtype=torch.int64,   device=self.device).unsqueeze(-1)
        r_t    = torch.tensor(r,    dtype=torch.float32, device=self.device)
        obs2_t = torch.tensor(obs2, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_values = self.q_net(obs_t).gather(1, a_t).squeeze(-1)

        # Target: reward + gamma max_{a'} Q_target(s',a') computes target action value
        with torch.no_grad():
            q_next_all = self.target_net(obs2_t)
            max_next = []
            for i, valid_ids in enumerate(valid_next_ids_list):
                if not valid_ids:
                    max_next.append(0.0)
                else:
                    vals = q_next_all[i][valid_ids]
                    max_next.append(float(torch.max(vals).item()))
            max_next_t = torch.tensor(max_next, dtype=torch.float32, device=self.device)
            target = r_t + self.gamma * (1.0 - done_t) * max_next_t

        loss = nn.MSELoss()(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.soft_update()

    def decay_epsilon(self):
        # ###############################################################
        # Decay epsilon for epsilon-greedy exploration.
        # ###############################################################
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_dqn(env,
              episodes=800,
              mode="eps",  # "eps" or "ucb"
              eps_start=1.0,
              eps_end=0.05,
              eps_decay=0.995,
              gamma=0.99,
              lr=1e-3,
              buffer_size=10000,
              batch_size=64,
              target_update_freq=1,
              tau=0.005):
    # ###############################################################
    # Train a DQNAgent on RLHexStarEnv with:
    #   - epsilon-greedy exploration ("eps")
    #   - UCB exploration ("ucb")
    # ###############################################################
    # Get observation dimension
    first_obs = env.reset(random_start=True)
    obs_dim = first_obs.shape[0]

    agent = DQNAgent(
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        gamma=gamma,
        lr=lr,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        tau=tau,
    )

    rewards_history = []  # Track rewards
    success_history = []  # Track success

    for ep in range(episodes):
        obs = env.reset(random_start=True)
        total_reward = 0.0
        done = False
        reached_goal = False  # Track reaching the goal

        while not done:
            valid_ids = env.valid_action_ids(env.state)
            if not valid_ids:
                # Dead state, big penalty
                r = -10.0
                done = True
                agent.store((obs, 0, r, obs, done, []))
                total_reward += r
                break

            if mode == "eps":
                a = agent.select_action_eps(obs, valid_ids)
            elif mode == "ucb":
                a = agent.select_action_ucb(obs, valid_ids, c=0.8)
            else:
                raise ValueError(f"Unknown DQN mode: {mode}")

            obs2, r, done, _ = env.step(a)
            valid_next_ids = env.valid_action_ids(env.state)

            if env.problem.is_goal(env.state):
                reached_goal = True

            agent.store((obs, a, r, obs2, done, valid_next_ids))
            agent.update()

            obs = obs2
            total_reward += r

        if mode == "eps":
            agent.decay_epsilon()

        rewards_history.append(total_reward)              # Update rewards
        success_history.append(1 if reached_goal else 0)  # Update success
        if (ep + 1) % 50 == 0:
            print(f"[DQN-{mode}] Episode {ep+1}/{episodes}, "
                  f"Return={total_reward:.2f}, eps={agent.eps:.3f}")

    return agent, rewards_history, success_history


# ============================================================
# 5. Visualization Helpers (Hex grid + Paths)
# ============================================================

def axial_to_cartesian(q, r, size):
    # ###############################################################
    # Convert axial hex coordinates (q,r) to 2D cartesian coords (x,y)
    # for a pointy-top hex layout.
    # ###############################################################
    x = size * sqrt(3) * (q + r / 2.0)
    y = size * 1.5 * r
    return x, y


def hex_corners(x, y, size):
    # ###############################################################
    # Return the vertices of a pointy-top hex centered at (x,y).
    # ###############################################################
    corners = []
    for i in range(6):
        angle = pi / 3 * i + pi / 6  # rotate for pointy top
        cx = x + size * cos(angle)
        cy = y + size * sin(angle)
        corners.append((cx, cy))
    return corners

# Function to develop hex maps and visualize them
def plot_hex_map(hex_map, obstacle_map, start, goal, path, hex_size, title="", ax=None):
    # ###############################################################
    # Visualize hex grid:
    #   - free cells: white
    #   - walls: black
    #   - start: blue
    #   - goal: green
    #   - path: pink
    # ###############################################################
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    draw_size = hex_size / sqrt(3)  # convert from edge-to-edge to radius

    all_cells = list(hex_map)
    cell_centers = [axial_to_cartesian(q, r, draw_size) for (q, r) in all_cells]
    xs = [c[0] for c in cell_centers]
    ys = [c[1] for c in cell_centers]

    path_set = set(path) if path is not None else set()

    for (q, r) in all_cells:
        x, y = axial_to_cartesian(q, r, draw_size)

        if (q, r) in obstacle_map:
            color = "black"
        elif (q, r) == start:
            color = "blue"
        elif (q, r) == goal:
            color = "green"
        elif (q, r) in path_set:
            color = "deeppink"
        else:
            color = "white"

        poly = Polygon(
            hex_corners(x, y, draw_size),
            closed=True,
            edgecolor="black",
            facecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(poly)

    ax.set_xlim(min(xs) - draw_size * 2, max(xs) + draw_size * 2)
    ax.set_ylim(min(ys) - draw_size * 2, max(ys) + draw_size * 2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    return ax


def extract_hstar_path(env):
    # ###############################################################
    # Compute the H* path using the original time_to_goal heuristic from
    # the default start to the fixed goal.
    # ###############################################################
    v0, theta0 = env.init_velocity[:2]
    initial_state = (env.default_start, (v0, theta0))
    problem = PathfindingProblem(
        initial_state,
        env.hex_map,
        env.obstacle_map,
        env.goal,
        env.hex_radius,
        env.hex_size,
        env.accel_max,
        env.decel_max,
        env.lat_accel_max,
    )
    sol = best_first_search(problem, hstar_f, time_to_goal)
    path = []
    n = sol
    while n:
        path.append(n.state[0])
        n = n.parent
    path = list(reversed(path))
    return path, sol.path_cost


def extract_q_policy_path(env, q_agent, max_steps=200):
    # ###############################################################
    # Roll out a greedy policy from a tabular Q-learning agent
    # (no exploration) starting from default_start.
    # ###############################################################
    _ = env.reset(random_start=False)
    path = []
    total_reward = 0.0
    done = False

    while not done and len(path) < max_steps:
        state_key = env.get_state_key()
        valid_ids = env.valid_action_ids(env.state)
        if not valid_ids:
            break
        a = q_agent.select_action_greedy(state_key, valid_ids)
        path.append(env.state[0])
        _, r, done, _ = env.step(a)
        total_reward += r

    path.append(env.state[0])
    return path, total_reward


def extract_dqn_policy_path(env, dqn_agent, max_steps=200):
    # ###############################################################
    # Roll out a greedy policy from a DQN agent (no ε, no UCB)
    # starting from default_start.
    # ###############################################################
    obs = env.reset(random_start=False)
    path = []
    total_reward = 0.0
    done = False

    while not done and len(path) < max_steps:
        valid_ids = env.valid_action_ids(env.state)
        if not valid_ids:
            break
        a = dqn_agent.select_action_greedy(obs, valid_ids)
        path.append(env.state[0])
        obs, r, done, _ = env.step(a)
        total_reward += r

    path.append(env.state[0])
    return path, total_reward


def rolling_average(data, w):
    # ###############################################################
    # Compute rolling average of a list over window w.
    # ###############################################################
    data = np.array(data, dtype=np.float32)
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='valid')

# %%
#  Maps to use 
# 'r3h0.33.pkl'
# 'r15h1.00.pkl'
# 'r20h1.00.pkl'
# 'r25h1.00.pkl'
# 'r30h5.00.pkl'
# 'r40h5.00.pkl'
# 'r50h5.00.pkl'
# ============================================================
# 6. Main: Train & Compare ε-greedy vs UCB (Q-learning & DQN)
# ============================================================

if __name__ == "__main__":
    # Choose a map and heuristic model
    map_dir = "maps"
    hex_map = "r20h1.00.pkl"
    pkl_path = os.path.join(map_dir, hex_map)  # adjust as needed
    heuristic_path = "learned_heuristic.pt"          # pre-trained heuristic

    num_episodes = 1000
    print(f"Using map {hex_map}")
    # ----------------------------
    # Q-Learning with ε-greedy
    # ----------------------------
    print("\n=== Training Tabular Q-Learning (ε-greedy) ===")
    env_q_eps = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.995,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    q_eps_agent, q_eps_returns, q_eps_success = train_q_learning(
        env_q_eps,
        episodes=num_episodes,
        mode="eps",
        eps_start=0.1,
        eps_end=0.05,
        eps_decay=0.999
    )

    # ----------------------------
    # Q-Learning with UCB
    # ----------------------------
    print("\n=== Training Tabular Q-Learning (UCB) ===")
    env_q_ucb = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    q_ucb_agent, q_ucb_returns, q_ucb_success = train_q_learning(
        env_q_ucb,
        episodes=num_episodes,
        mode="ucb",
        eps_start=0.1,
        eps_end=0.05,
        eps_decay=0.999
    )

    # ----------------------------
    # DQN with ε-greedy
    # ----------------------------
    print("\n=== Training DQN (ε-greedy) ===")
    env_dqn_eps = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    dqn_eps_agent, dqn_eps_returns, dqn_eps_success = train_dqn(
        env_dqn_eps,
        episodes=num_episodes,
        mode="eps",
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=0.997,
        gamma=0.99,
        lr=1e-3,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=1,
        tau=0.005,
    )

    # ----------------------------
    # DQN with UCB
    # ----------------------------
    print("\n=== Training DQN (UCB) ===")
    env_dqn_ucb = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    dqn_ucb_agent, dqn_ucb_returns, dqn_ucb_success = train_dqn(
        env_dqn_ucb,
        episodes=num_episodes,
        mode="ucb",
        eps_start=0.0,   # eps is ignored in UCB mode but kept for API compatibility
        eps_end=0.0,
        eps_decay=1.0,
        gamma=0.99,
        lr=1e-3,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=1,
        tau=0.005,
    )

    print("\n=== Training finished for all 4 agents. ===")

    # =======================================================
    # Plot: Episode Returns (all 4)
    # =======================================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=q_eps_returns,   mode='lines', name='Q-learning ε-greedy'))
    fig.add_trace(go.Scatter(y=q_ucb_returns,   mode='lines', name='Q-learning UCB'))
    fig.add_trace(go.Scatter(y=dqn_eps_returns, mode='lines', name='DQN ε-greedy', opacity=0.8))
    fig.add_trace(go.Scatter(y=dqn_ucb_returns, mode='lines', name='DQN UCB', opacity=0.8))

    fig.update_layout(
        title=f"Episode Returns: ε-greedy vs UCB (Q-learning & DQN), {num_episodes} episodes",
        xaxis_title="Episode",
        yaxis_title="Return (sum of shaped rewards)",
        template="simple_white",
        width=900,
        height=450,
        legend=dict(x=0.01, y=0.99, bordercolor="gray", borderwidth=1),
    )
    fig.show()

    # =======================================================
    # Plot: Rolling Average Returns (all 4)
    # =======================================================
    window = 20
    q_eps_roll   = rolling_average(q_eps_returns, window)
    q_ucb_roll   = rolling_average(q_ucb_returns, window)
    dqn_eps_roll = rolling_average(dqn_eps_returns, window)
    dqn_ucb_roll = rolling_average(dqn_ucb_returns, window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=q_eps_roll,   mode='lines', name=f"Q ε-greedy (rolling {window})"))
    fig.add_trace(go.Scatter(y=q_ucb_roll,   mode='lines', name=f"Q UCB (rolling {window})"))
    fig.add_trace(go.Scatter(y=dqn_eps_roll, mode='lines', name=f"DQN ε-greedy (rolling {window})"))
    fig.add_trace(go.Scatter(y=dqn_ucb_roll, mode='lines', name=f"DQN UCB (rolling {window})"))

    fig.update_layout(
        title=f"Rolling Average Returns (window={window}), ε-greedy vs UCB",
        xaxis_title="Episode",
        yaxis_title="Rolling Average Return",
        template="plotly_white",
        width=900,
        height=450,
    )
    fig.show()

    # Compute Suboptimality vs H*
    def compute_suboptimality(agent_cost, hstar_cost):
        return {
            "cost_gap": agent_cost - hstar_cost,
            "ratio": agent_cost / hstar_cost
        }    
    
    # =======================================================
    # Path Extraction and Visualization (H*, Q, DQN)
    # =======================================================
    # Fresh environment for H* visualization
    env_vis = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )

    # H* path
    hstar_path, hstar_cost = extract_hstar_path(env_vis)
    print(f"\nH* path cost: {hstar_cost:.3f}, length: {len(hstar_path)}")    

    # Evaluate Q-learning (ε-greedy) agent greedily
    env_q_eps_eval = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    q_eps_path, q_eps_ret_eval = extract_q_policy_path(env_q_eps_eval, q_eps_agent)
    print(f"Q-learning ε-greedy eval reward: {q_eps_ret_eval:.3f}, path length: {len(q_eps_path)}")

    # Evaluate Q-learning (UCB) agent greedily
    env_q_ucb_eval = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    q_ucb_path, q_ucb_ret_eval = extract_q_policy_path(env_q_ucb_eval, q_ucb_agent)
    print(f"Q-learning UCB eval reward: {q_ucb_ret_eval:.3f}, path length: {len(q_ucb_path)}")

    # Evaluate DQN (ε-greedy) agent greedily
    env_dqn_eps_eval = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    dqn_eps_path, dqn_eps_ret_eval = extract_dqn_policy_path(env_dqn_eps_eval, dqn_eps_agent)
    print(f"DQN ε-greedy eval reward: {dqn_eps_ret_eval:.3f}, path length: {len(dqn_eps_path)}")

    # Evaluate DQN (UCB) agent greedily
    env_dqn_ucb_eval = RLHexStarEnv(
        pkl_path,
        heuristic_model_path=heuristic_path,
        shaping_gamma=0.99,
        step_penalty=-0.01,
        goal_reward=50.0,
        max_episode_steps=500,
    )
    dqn_ucb_path, dqn_ucb_ret_eval = extract_dqn_policy_path(env_dqn_ucb_eval, dqn_ucb_agent)
    print(f"DQN UCB eval reward: {dqn_ucb_ret_eval:.3f}, path length: {len(dqn_ucb_path)}")


    sub_q_eps   = compute_suboptimality(q_eps_ret_eval,   hstar_cost)
    sub_q_ucb   = compute_suboptimality(q_ucb_ret_eval,   hstar_cost)
    sub_dqn_eps = compute_suboptimality(dqn_eps_ret_eval, hstar_cost)
    sub_dqn_ucb = compute_suboptimality(dqn_ucb_ret_eval, hstar_cost)


    # Plot 5 paths: H*, Q-eps, Q-UCB, DQN-eps, DQN-UCB
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    plot_hex_map(
        env_vis.hex_map,
        env_vis.obstacle_map,
        env_vis.default_start,
        env_vis.goal,
        hstar_path,
        env_vis.hex_size,
        title="H* Path",
        ax=axes[0],
    )

    plot_hex_map(
        env_q_eps_eval.hex_map,
        env_q_eps_eval.obstacle_map,
        env_q_eps_eval.default_start,
        env_q_eps_eval.goal,
        q_eps_path,
        env_q_eps_eval.hex_size,
        title="Q-learning ε-greedy",
        ax=axes[1],
    )

    plot_hex_map(
        env_q_ucb_eval.hex_map,
        env_q_ucb_eval.obstacle_map,
        env_q_ucb_eval.default_start,
        env_q_ucb_eval.goal,
        q_ucb_path,
        env_q_ucb_eval.hex_size,
        title="Q-learning UCB",
        ax=axes[2],
    )

    plot_hex_map(
        env_dqn_eps_eval.hex_map,
        env_dqn_eps_eval.obstacle_map,
        env_dqn_eps_eval.default_start,
        env_dqn_eps_eval.goal,
        dqn_eps_path,
        env_dqn_eps_eval.hex_size,
        title="DQN ε-greedy",
        ax=axes[3],
    )

    plot_hex_map(
        env_dqn_ucb_eval.hex_map,
        env_dqn_ucb_eval.obstacle_map,
        env_dqn_ucb_eval.default_start,
        env_dqn_ucb_eval.goal,
        dqn_ucb_path,
        env_dqn_ucb_eval.hex_size,
        title="DQN UCB",
        ax=axes[4],
    )

    plt.tight_layout()
    plt.show()

import pandas as pd
summary = pd.DataFrame([
    {
        "Agent": "Tabular Q-learning",
        "Exploration": "ε-greedy",
        "Success Rate": np.mean(q_eps_success),
        "Avg Reward": np.mean(q_eps_returns),
        "Path Reward": q_eps_ret_eval,
        "Suboptimality": sub_q_eps["cost_gap"]
    },
    {
        "Agent": "Tabular Q-learning",
        "Exploration": "UCB",
        "Success Rate": np.mean(q_ucb_success),
        "Avg Reward": np.mean(q_ucb_returns),
        "Path Reward": q_ucb_ret_eval,
        "Suboptimality": sub_q_ucb["cost_gap"]
    },
    {
        "Agent": "DQN",
        "Exploration": "ε-greedy",
        "Success Rate": np.mean(dqn_eps_success),
        "Avg Reward": np.mean(dqn_eps_returns),
        "Path Reward": dqn_eps_ret_eval,
        "Suboptimality": sub_dqn_eps["cost_gap"]
    },
    {
        "Agent": "DQN",
        "Exploration": "UCB",
        "Success Rate": np.mean(dqn_ucb_success),
        "Avg Reward": np.mean(dqn_ucb_returns),
        "Path Reward": dqn_ucb_ret_eval,
        "Suboptimality": sub_dqn_ucb["cost_gap"]
    }
])

print("\n=== Experiment Summary Table ===")
print(summary.to_string(index=False))
#### THE END ####
# %%
# 'r3h0.33.pkl'
# 'r15h1.00.pkl'
# 'r20h1.00.pkl'
# 'r25h1.00.pkl'
# 'r30h5.00.pkl'
# 'r40h5.00.pkl'
# 'r50h5.00.pkl'