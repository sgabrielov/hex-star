"""Minimal deep-RL utilities for H*/HEX* heuristic learning.

The initial design has only four responsibilities:
1. Record transitions produced by search.
2. Store recent transitions for training.
3. Represent a learned cost-to-go model.
4. Train the model from Bellman/TD targets.

Keep the search itself responsible for search, and keep model training here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional
import random

import copy
import torch
import torch.nn as nn


from hexgrid import RLState, euclid_center_distance, VelocityState
from collections import defaultdict


from unet.hexunet_model import HexUNet


# Experience + Reply buffer 
# Replay buffer controls the data structure that stores experience data 
# After a node has been expanded (successor generated, transition cost and new_v calculated), 
#   That node is ready to be stored as experience because we know 
#   what the next state will be for each action 

# How to hook into H*:
#   H* allows an optional parameter, experience_collector = ReplayBuffer
#   When the reply buffer exists, store experience - 1. offline data collection 

# ValueNetwork 
#   Stub for the RL value function, whether it's a Q-table lookup, weighted parameter function, or deep net

# TDTrainer
#   The object that updates the ValueNetwork from the ReplayBuffer
#   1. Offline training - run the search, then after the search is over, process the ReplayBuffer 
#   2. Online training - pass in the TDTrainer to H* and call the trainer during the search process
    
@dataclass(frozen=True)
class Experience:
    """One transition observed during search.

    Represents the tuple (s, a, c, s'), plus whether the successor is terminal.

    Parameters
    ----------
    state:
        Search state before taking the action. Initially this may simply be a
        copy/reference containing the information needed by the model.
    action:
        Action taken from ``state``.
    cost:
        Immediate transition cost from state to next_state. For H*/HEX*, this
        is normally the incremental travel time.
    next_state:
        State reached after the action.
    terminal:
        True when next_state is a goal state.
    """

    state: Any
    goal: Any
    environment_id: int
    action: Any
    cost: float
    next_state: Any
    terminal: bool

@dataclass(frozen=True)
class SolutionTarget:
    state: RLState
    goal: RLState
    environment_id: int
    target: float

# buffer = ReplayBuffer()
# if buffer is not none:
#     process experience

class ReplayBuffer:
    """Store transition experience for minibatch training."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: Deque[Experience] = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """Add one observed transition to the replay buffer."""
        self._items.append(experience)

    def extend(self, experiences: Iterable[Experience]) -> None:
        """Add several observed transitions."""
        self._items.extend(experiences)

    def sample(self, batch_size: int):
        """Return a random minibatch of stored transitions."""
        return random.sample(
            list(self._items),
            min(batch_size, len(self._items))
        )

    def __len__(self) -> int:
        return len(self._items)

   
class ValueNetwork_DNN(nn.Module):
    """Basic neural network for estimating cost-to-go J(state, goal)."""

    def __init__(self, encoder: StateEncoder):
        super().__init__()

        self.encoder = encoder

        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        """Return the estimated cost-to-go for an input tensor."""
        return self.network(x)

    def predict(
        self,
        problem,
        state: RLState,
        goal: RLState,
    ) -> float:
        """Encode one state/goal pair and return its estimated cost-to-go."""

        features = self.encoder.encode(
            problem,
            state,
            goal,
        )

        x = torch.tensor(
            features,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            value = self.forward(x)

        return value.item()

# class ValueNetwork_HUNet(nn.Module):

#     def __init__(
#         self,
#         encoder: StateEncoder,
#         spatial_channels=16,
#         kinematic_features=5,
#     ):
#         super().__init__()

#         self.encoder = encoder

#         self.hunet = HexUNet(
#             n_channels=3,
#             n_classes=spatial_channels,
#         )

#         self.network = nn.Sequential(
#             nn.Linear(
#                 spatial_channels + kinematic_features,
#                 64,
#             ),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Softplus(),
#         )

#     def forward(
#         self,
#         spatial_features,
#         kinematic_features,
#     ):
#         x = torch.cat(
#             [
#                 spatial_features,
#                 kinematic_features,
#             ],
#             dim=1,
#         )

#         return self.network(x)

#     def value(
#         self,
#         problem,
#         state: RLState,
#         goal: RLState,
#         environment,
#         feature_map=None,
#     ):
#         if feature_map is None:
#             feature_map = self.encode_environment(
#                 environment,
#                 goal,
#             )
    
#         row, col = self.encoder.location_index(
#             state,
#             environment.center,
#             environment.radius,
#         )
    
#         spatial_features = feature_map[
#             :,
#             :,
#             row,
#             col,
#         ]
    
#         kinematic_features = self.encoder.encode_state(
#             problem,
#             state,
#             goal,
#         )
    
#         kinematic_features = torch.tensor(
#             kinematic_features,
#             dtype=torch.float32,
#         ).unsqueeze(0)
    
#         return self.forward(
#             spatial_features,
#             kinematic_features,
#         )

#     def predict(
#         self,
#         problem,
#         state: RLState,
#         goal: RLState,
#         environment,
#         feature_map=None,
#     ) -> float:
#         with torch.no_grad():
#             value = self.value(
#                 problem,
#                 state,
#                 goal,
#                 environment,
#                 feature_map=feature_map,
#             )
    
#         return value.item()
#     def encode_environment(
#         self,
#         environment,
#         goal: RLState,
#     ):
#         """Generate the HexUNet feature map for one environment and goal."""
    
#         spatial_input = self.encoder.encode_spatial(
#             environment,
#             goal,
#             environment.center,
#             environment.radius,
#         )
    
#         spatial_input = spatial_input.unsqueeze(0)
    
#         return self.hunet(spatial_input)
class ValueNetwork_HUNet(nn.Module):
    """
    Estimate cost-to-go J(state, goal).

    HexUNet produces a spatial context vector for every hex.
    The context vector at the current state's location is combined
    with relative position, velocity, and direction and passed
    through an MLP.
    """

    def __init__(
        self,
        encoder,
        spatial_channels=16,
        state_features=5,
    ):
        super().__init__()

        self.encoder = encoder
        self.spatial_channels = spatial_channels
        self.state_features = state_features

        # Environmental/spatial encoder:
        #   goal
        #   obstacles
        #   valid-hex mask
        self.hunet = HexUNet(
            n_channels=3,
            n_classes=spatial_channels,
        )

        # Context vector + state vector -> scalar cost-to-go
        self.network = nn.Sequential(
            nn.Linear(
                spatial_channels + state_features,
                64,
            ),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        context,
        state_features,
    ):
        x = torch.cat(
            [
                context,
                state_features,
            ],
            dim=-1,
        )

        return self.network(x)

    def encode_environment(
        self,
        environment,
        goal,
    ):
        """
        Produce one context vector per spatial location.

        Returns:
            [1, spatial_channels, H, W]
        """

        spatial_input = self.encoder.encode_spatial(
            environment,
            goal.location,
            environment.center,
            environment.radius,
        )

        spatial_input = spatial_input.unsqueeze(0)

        return self.hunet(spatial_input)

    def encode_state(
        self,
        state,
        goal,
    ):
        dq = state.location.q - goal.location.q
        dr = state.location.r - goal.location.r
    
        velocity = state.velocity.magnitude
    
        if state.velocity.direction is None:
            direction_q = 0.0
            direction_r = 0.0
        else:
            direction_q = state.velocity.direction.q
            direction_r = state.velocity.direction.r
    
        return torch.tensor(
            [
                dq,
                dr,
                velocity,
                direction_q,
                direction_r,
            ],
            dtype=torch.float32,
        )

    def value(
        self,
        problem,
        state,
        goal,
        environment,
        context_map=None,
    ):
        """
        Return differentiable scalar value for one state.
        """

        if context_map is None:
            context_map = self.encode_environment(
                environment,
                goal,
            )

        center = environment.center
        radius = environment.radius

        row = (
            state.location.r
            - center.r
            + radius
        )

        col = (
            state.location.q
            - center.q
            + radius
        )

        # [1, C]
        context = context_map[
            :,
            :,
            row,
            col,
        ]

        state_features = self.encode_state(
            state,
            goal,
        )

        state_features = (
            state_features
            .to(context.device)
            .unsqueeze(0)
        )

        value = self.forward(
            context,
            state_features,
        )

        return value.squeeze()

    def predict(
        self,
        problem,
        state,
        goal,
        environment,
        context_map=None,
    ):
        """
        Inference convenience function.
        """

        self.eval()

        with torch.no_grad():
            value = self.value(
                problem=problem,
                state=state,
                goal=goal,
                environment=environment,
                context_map=context_map,
            )

        return value.item()
class ExperienceCollector:
    """Convert transitions generated by H*/HEX* into replay experience.

    This class only records what happened. It does not train the network.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        environment_store: EnvironmentStore,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.environment_store = environment_store

    def observe_transition(
        self,
        problem: Any,
        parent: Any,
        action: Any,
        child: Any,
        environment_id: int,
    ) -> Experience:
        """Record one parent -> child transition."""
    
        def to_rl_velocity(velocity):
            direction = (
                None
                if velocity.direction is None
                else problem.DIRECTIONS[velocity.direction]
            )
    
            return VelocityState(
                magnitude=velocity.magnitude,
                direction=direction,
            )
    
        experience = Experience(
            state=RLState(
                location=parent.location,
                velocity=to_rl_velocity(parent.velocity),
            ),
            goal=RLState(
                location=problem.goal,
                velocity=to_rl_velocity(problem.goal_v),
            ),
            environment_id=environment_id,
            action=action,
            cost=child.g_cost - parent.g_cost,
            next_state=RLState(
                location=child.location,
                velocity=to_rl_velocity(child.velocity),
            ),
            terminal=problem.is_goal(child),
        )
    
        self.replay_buffer.add(experience)
    
        return experience

    def observe_solution(
        self,
        problem,
        solution_path,
        environment_id,
    ):
        """Create supervised value targets from a completed solution path."""
    
        solution_cost = solution_path[-1].g_cost
    
        def to_rl_velocity(velocity):
            direction = (
                None
                if velocity.direction is None
                else problem.DIRECTIONS[velocity.direction]
            )
    
            return VelocityState(
                magnitude=velocity.magnitude,
                direction=direction,
            )
    
        goal = RLState(
            location=problem.goal,
            velocity=to_rl_velocity(problem.goal_v),
        )
    
        targets = []
    
        for node in solution_path:
    
            state = RLState(
                location=node.location,
                velocity=to_rl_velocity(node.velocity),
            )
    
            target = solution_cost - node.g_cost
    
            targets.append(
                SolutionTarget(
                    state=state,
                    goal=goal,
                    environment_id=environment_id,
                    target=target,
                )
            )
    
        return targets
class StateEncoder:
    """Convert an Experience into numeric model input features."""

    def encode(self, experience: Experience):
        state = experience.state
        goal = experience.goal

        # Basic implementation:
        # state + goal only
        # environment intentionally ignored for now

        raise NotImplementedError

# normalization options:
# direction vector scaled to unit vector
# absolute position -> relative to goal (IMPLEMENTED)
#    instead of storing hexcoords of loc and goal
#    store the difference
# velocity normalization
#    H* does not have a v_max, but if we did
#    we could normalize via v/v_max

class BasicStateEncoder(StateEncoder):
    """Encode current state and goal using Euclidean distance and velocity."""

    def encode(
        self,
        problem,
        state: RLState,
        goal: RLState,
        environment=None,
    ):
        distance = euclid_center_distance(
            state.location,
            goal.location,
            problem.grid.hex_size,
        )

        if state.velocity.direction is None:
            state_dir_q = 0
            state_dir_r = 0
        else:
            direction = problem.DIRECTIONS[state.velocity.direction]
            state_dir_q = direction.q
            state_dir_r = direction.r

        if goal.velocity.direction is None:
            goal_dir_q = 0
            goal_dir_r = 0
        else:
            direction = problem.DIRECTIONS[goal.velocity.direction]
            goal_dir_q = direction.q
            goal_dir_r = direction.r

        return [
            distance,
            state.velocity.magnitude,
            state_dir_q,
            state_dir_r,
            goal.velocity.magnitude,
            goal_dir_q,
            goal_dir_r,
        ]

class HUNetStateEncoder(StateEncoder):
    """Encode an H* state for the HexUNet value network."""

    def encode_spatial(
        self,
        environment,
        goal: RLState,
        center,
        radius,
    ):
        size = 2 * radius + 1

        x = torch.zeros(
            3,
            size,
            size,
            dtype=torch.float32,
        )

        # Channel 0 = goal
        goal_row = goal.r - center.r + radius
        goal_col = goal.q - center.q + radius

        x[0, goal_row, goal_col] = 1.0

        # Channel 1 = obstacles
        for obstacle in environment.obstacles:
            row = obstacle.r - center.r + radius
            col = obstacle.q - center.q + radius

            if (
                0 <= row < size
                and 0 <= col < size
            ):
                x[1, row, col] = 1.0
        x[2] = make_valid_hex_mask(
            radius=radius,
            center=center,
        )
        return x

    def encode_state(
        self,
        problem,
        state: RLState,
        goal: RLState,
    ):
        """
        Build non-spatial state features for the value MLP.

        Initial features:
            current velocity magnitude
            current velocity direction
            goal velocity magnitude
            goal velocity direction
        """

        if state.velocity.direction is None:
            direction_q = 0
            direction_r = 0
        else:
            direction = problem.DIRECTIONS[
                state.velocity.direction
            ]
    
            direction_q = direction.q
            direction_r = direction.r
    
        return [
            state.location.q,
            state.location.r,
            state.velocity.magnitude,
            direction_q,
            direction_r,
        ]

    def location_index(
        self,
        state: RLState,
        center: HexCoord,
        radius: int,
    ):
        """Convert an axial location into HexUNet tensor indices."""

        row = state.location.r - center.r + radius
        col = state.location.q - center.q + radius

        return row, col
        
class TDTrainer:
    """Train ValueNetwork using a frozen target network for TD targets."""

    def __init__(
        self,
        model,
        replay_buffer,
        learning_rate=1e-3,
        gamma=1.0,
        target_update_every=100,
    ):
        self.model = model

        self.target_model = copy.deepcopy(
            model
        )

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_every = target_update_every
        self.training_steps = 0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()


    def train_step(
        self,
        problem,
        batch_size: int = 64,
    ) -> float:

        experiences = self.replay_buffer.sample(batch_size)

        if not experiences:
            return 0.0

        predictions = []
        targets = []

        for exp in experiences:

            # -------------------------------------------------------------
            # Current estimate: J(s, goal)
            # Uses the actively trained network.
            # -------------------------------------------------------------

            state_features = self.model.encoder.encode(
                problem,
                exp.state,
                exp.goal,
            )

            state_tensor = torch.tensor(
                state_features,
                dtype=torch.float32,
            )

            prediction = self.model(
                state_tensor
            ).squeeze()

            # -------------------------------------------------------------
            # TD target
            # -------------------------------------------------------------

            if exp.terminal:
                # J(goal) = 0
                target = exp.cost

            else:
                next_features = self.target_model.encoder.encode(
                    problem,
                    exp.next_state,
                    exp.goal,
                )

                next_tensor = torch.tensor(
                    next_features,
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    next_value = self.target_model(
                        next_tensor
                    ).squeeze()

                target = (
                    exp.cost
                    + self.gamma * next_value.item()
                )

            predictions.append(prediction)
            targets.append(target)

        # -------------------------------------------------------------
        # Train online network
        # -------------------------------------------------------------

        predictions = torch.stack(predictions)

        targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

        loss = self.loss_fn(
            predictions,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        # -------------------------------------------------------------
        # Periodically synchronize target network
        # -------------------------------------------------------------

        if self.training_steps % self.target_update_every == 0:
            self.update_target_network()

        return loss.item()


    def update_target_network(self):
        """Copy current ValueNetwork weights into the target network."""

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

class HUNetTDTrainer(TDTrainer):
    """TD trainer for the environment-aware HexUNet value network."""

    def __init__(
        self,
        model,
        replay_buffer,
        environment_store,
        learning_rate=1e-3,
        gamma=1.0,
        target_update_every=100,
    ):
        super().__init__(
            model=model,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_every=target_update_every,
        )

        self.environment_store = environment_store

    def train_step(
        self,
        problem,
        batch_size=64,
    ):
        experiences = self.replay_buffer.sample(
            batch_size
        )

        if not experiences:
            return 0.0

        predictions = []
        targets = []

        for exp in experiences:

            environment = self.environment_store.get(
                exp.environment_id
            )

            # J(s, goal, environment)
            prediction = self.model.value(
                problem,
                exp.state,
                exp.goal,
                environment,
            ).squeeze()

            if exp.terminal:
                target = exp.cost

            else:
                with torch.no_grad():
                    next_value = self.target_model.value(
                        problem,
                        exp.next_state,
                        exp.goal,
                        environment,
                    ).squeeze()

                    target = (
                        exp.cost
                        + self.gamma * next_value.item()
                    )

            predictions.append(prediction)
            targets.append(target)

        predictions = torch.stack(predictions)

        targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

        loss = self.loss_fn(
            predictions,
            targets,
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_every
            == 0
        ):
            self.update_target_network()

        return {
            "loss": loss.item(),
            "mean_prediction": predictions.mean().item(),
            "mean_target": targets.mean().item(),
        }

    def train_solution(
        self,
        problem,
        solution_targets,
    ):
        if not solution_targets:
            return 0.0
    
        losses = []
    
        for sample in solution_targets:
    
            environment = self.environment_store.get(
                sample.environment_id
            )
    
            prediction = self.model.value(
                problem=problem,
                state=sample.state,
                goal=sample.goal,
                environment=environment,
            )
    
            target = torch.tensor(
                sample.target,
                dtype=torch.float32,
                device=prediction.device,
            )
    
            losses.append(
                (prediction - target).pow(2)
            )
    
        loss = torch.stack(losses).mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.item()
class EnvironmentStore:
    """Store environments separately from Experience objects."""

    def __init__(self):
        self.environments = {}

    def add(self, environment_id, environment):
        """Add an environment to the store."""
        self.environments[environment_id] = environment

    def get(self, environment_id):
        """Return an environment by ID."""
        return self.environments[environment_id]

    def save(self, filepath):
        """Save all environments to disk."""
        raise NotImplementedError

    def load(self, filepath):
        """Load environments from disk."""
        raise NotImplementedError

    def __len__(self):
        return len(self.environments)



#### Helper Functions #### 
def get_spatial_features(
    self,
    feature_map,
    state,
    center,
    radius,
):
    row = state.location.r - center.r + radius
    col = state.location.q - center.q + radius

    return feature_map[
        :,
        :,
        row,
        col,
    ]

def get_kinematic_features(
    self,
    state,
    goal,
):
    return torch.tensor(
        [[
            state.velocity.magnitude,
            state.velocity.direction or 0,
            goal.velocity.magnitude,
            goal.velocity.direction or 0,
        ]],
        dtype=torch.float32,
    )

def make_valid_hex_mask(
    radius: int,
    center: HexCoord,
):
    size = 2 * radius + 1

    mask = torch.zeros(
        (size, size),
        dtype=torch.float32,
    )

    for row in range(size):
        for col in range(size):

            q = center.q + col - radius
            r = center.r + row - radius

            dq = q - center.q
            dr = r - center.r

            distance = max(
                abs(dq),
                abs(dr),
                abs(dq + dr),
            )

            if distance <= radius:
                mask[row, col] = 1.0

    return mask

def make_solution_target_map(
    solution_path,
    center,
    radius,
):
    size = 2 * radius + 1

    target_map = torch.zeros(
        (size, size),
        dtype=torch.float32,
    )

    target_mask = torch.zeros(
        (size, size),
        dtype=torch.bool,
    )

    solution_cost = solution_path[-1].g_cost

    for node in solution_path:
        row = node.location.r - center.r + radius
        col = node.location.q - center.q + radius

        target_map[row, col] = (
            solution_cost - node.g_cost
        )

        target_mask[row, col] = True

    return target_map, target_mask

def masked_mse_loss(
    prediction,
    target,
    mask,
):
    prediction = prediction[0, 0]

    error = (
        prediction - target
    ).pow(2)

    return error[mask].mean()


def group_experiences_by_state(
    experiences,
):
    groups = defaultdict(list)

    for exp in experiences:

        direction = exp.state.velocity.direction

        key = (
            exp.environment_id,

            exp.goal.location.q,
            exp.goal.location.r,

            exp.state.location.q,
            exp.state.location.r,

            exp.state.velocity.magnitude,

            None if direction is None else direction.q,
            None if direction is None else direction.r,
        )

        groups[key].append(exp)

    return groups