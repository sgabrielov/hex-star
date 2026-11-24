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


from collections import defaultdict


from unet.hexunet_model import HexUNet

import os
import pickle
from pathlib import Path

from hexgrid import (
    HexCoord,
    HexGrid,
    RLState,
    VelocityState,
    euclid_center_distance,
)


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

# class ReplayBuffer:
#     """Store transition experience for minibatch training."""

#     def __init__(self, capacity: int = 100_000) -> None:
#         if capacity <= 0:
#             raise ValueError("capacity must be positive")
#         self._items: Deque[Experience] = deque(maxlen=capacity)

#     def add(self, experience: Experience) -> None:
#         """Add one observed transition to the replay buffer."""
#         self._items.append(experience)

#     def extend(self, experiences: Iterable[Experience]) -> None:
#         """Add several observed transitions."""
#         self._items.extend(experiences)

#     def sample(self, batch_size: int):
#         """Return a random minibatch of stored transitions."""
#         return random.sample(
#             list(self._items),
#             min(batch_size, len(self._items))
#         )

#     def __len__(self) -> int:
#         return len(self._items)

class ReplayBuffer:
    """Store transition and grounded solution experience."""

    def __init__(
        self,
        capacity=2_000_000,
        solution_fraction=0.20,
    ):
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        if not 0.0 <= solution_fraction <= 1.0:
            raise ValueError(
                "solution_fraction must be between 0 and 1"
            )

        self.capacity = capacity
        self.solution_fraction = solution_fraction

        self.solution_capacity = round(
            capacity * solution_fraction
        )

        self.nonterminal_capacity = (
            capacity - self.solution_capacity
        )

        self._solution_items = deque(
            maxlen=self.solution_capacity
        )

        self._items = deque(
            maxlen=self.nonterminal_capacity
        )

    # -------------------------
    # Nonterminal experience
    # -------------------------

    def add(
        self,
        experience: Experience,
    ):
        self._items.append(
            experience
        )

    def extend(
        self,
        experiences,
    ):
        self._items.extend(
            experiences
        )

    def sample(
        self,
        batch_size,
    ):
        return random.sample(
            list(self._items),
            min(
                batch_size,
                len(self._items),
            ),
        )

    # -------------------------
    # Grounded solution data
    # -------------------------

    def add_solution(
        self,
        target: SolutionTarget,
    ):
        self._solution_items.append(
            target
        )

    def extend_solution(
        self,
        targets,
    ):
        self._solution_items.extend(
            targets
        )

    def sample_solution(
        self,
        batch_size,
    ):
        return random.sample(
            list(self._solution_items),
            min(
                batch_size,
                len(self._solution_items),
            ),
        )

    # -------------------------
    # Mixed sampling
    # -------------------------

    def sample_mixed(
        self,
        batch_size,
        solution_weight=1.0,
        nonterminal_weight=0.0,
    ):
        total_weight = (
            solution_weight
            + nonterminal_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one weight must be positive"
            )

        solution_count = round(
            batch_size
            * solution_weight
            / total_weight
        )

        nonterminal_count = (
            batch_size - solution_count
        )

        solution_samples = random.sample(
            list(self._solution_items),
            min(
                solution_count,
                len(self._solution_items),
            ),
        )

        nonterminal_samples = random.sample(
            list(self._items),
            min(
                nonterminal_count,
                len(self._items),
            ),
        )

        return (
            solution_samples,
            nonterminal_samples,
        )

    # -------------------------
    # Status
    # -------------------------
    def has_training_batch(
        self,
        batch_size,
        solution_weight=1.0,
        nonterminal_weight=0.0,
    ):
        total_rate = (
            solution_weight
            + nonterminal_weight
        )
    
        if total_rate <= 0:
            return False
    
        solution_count = round(
            batch_size
            * solution_weight
            / total_rate
        )
    
        td_count = (
            batch_size
            - solution_count
        )
    
        return (
            len(self._solution_items) >= solution_count
            and len(self._items) >= td_count
        )
    @property
    def solution_size(self):
        return len(self._solution_items)

    @property
    def nonterminal_size(self):
        return len(self._items)

    def __len__(self):
        return (
            len(self._solution_items)
            + len(self._items)
        )
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
            #nn.Softplus(),
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
            #nn.Softplus(),
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
    
        The encoder is device-agnostic. The resulting tensor is moved
        to the device currently used by this model.
        """
    
        spatial_input = self.encoder.encode_spatial(
            environment,
            goal.location,
            environment.center,
            environment.radius,
        )
    
        device = next(
            self.parameters()
        ).device
    
        spatial_input = (
            spatial_input
            .unsqueeze(0)
            .to(device)
        )
    
        return self.hunet(
            spatial_input
        )
    # def encode_state(
    #     self,
    #     state,
    #     goal,
    #     environment,
    # ):
    #     scale = float(environment.radius)
    
    #     dq = (
    #         state.location.q
    #         - goal.location.q
    #     ) / scale
    
    #     dr = (
    #         state.location.r
    #         - goal.location.r
    #     ) / scale
    
    #     velocity = state.velocity.magnitude
    
    #     if state.velocity.direction is None:
    #         direction_q = 0.0
    #         direction_r = 0.0
    #     else:
    #         direction_q = float(
    #             state.velocity.direction.q
    #         )
    
    #         direction_r = float(
    #             state.velocity.direction.r
    #         )
    
    #         direction_scale = max(
    #             abs(direction_q),
    #             abs(direction_r),
    #             abs(direction_q + direction_r),
    #             1.0,
    #         )
    
    #         direction_q /= direction_scale
    #         direction_r /= direction_scale
    
    #     return torch.tensor(
    #         [
    #             dq,
    #             dr,
    #             velocity,
    #             direction_q,
    #             direction_r,
    #         ],
    #         dtype=torch.float32,
    #     )

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

        row, col = self.encoder.location_index(
            state=state,
            center=environment.center,
            radius=environment.radius,
        )


        # [1, C]
        context = context_map[
            :,
            :,
            row,
            col,
        ]

        state_features = self.encoder.encode_state(
            state,
            goal,
            environment.radius,
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
    

        experience = Experience(
            state=RLState(
                location=parent.location,
                velocity=_to_rl_velocity(problem, parent.velocity),
            ),
            goal=RLState(
                location=problem.goal,
                velocity=_to_rl_velocity(problem, problem.goal_v),
            ),
            environment_id=environment_id,
            action=action,
            cost=child.g_cost - parent.g_cost,
            next_state=RLState(
                location=child.location,
                velocity=_to_rl_velocity(problem, child.velocity),
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
        """Convert a successful solution path into grounded targets and store them."""
    
        if not solution_path:
            return []
    
        solution_cost = solution_path[-1].g_cost
    
        goal = RLState(
            location=problem.goal,
            velocity=_to_rl_velocity(problem, problem.goal_v),
        )
    
        targets = []
    
        for node in solution_path:
    
            state = RLState(
                location=node.location,
                velocity=_to_rl_velocity(problem, node.velocity),
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
    
        self.replay_buffer.extend_solution(
            targets
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
        state: RLState,
        goal: RLState,
        radius: int,
    ):
        """
        Encode dynamic state relative to the goal.
    
        Features:
            relative q
            relative r
            velocity magnitude
            direction q
            direction r
        """
    
        dq = (
            state.location.q
            - goal.location.q
        ) / radius
    
        dr = (
            state.location.r
            - goal.location.r
        ) / radius
    
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
                state.velocity.magnitude,
                direction_q,
                direction_r,
            ],
            dtype=torch.float32,
        )

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

    def checkpoint_state(
        self,
    ):
        """Return the persistent trainer state."""

        return {
            "model_state_dict": (
                self.model.state_dict()
            ),

            "target_model_state_dict": (
                self.target_model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "training_steps": (
                self.training_steps
            ),

            "gamma": (
                self.gamma
            ),

            "target_update_every": (
                self.target_update_every
            ),
        }

    def load_checkpoint_state(
        self,
        checkpoint,
    ):
        """Restore persistent trainer state."""

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.target_model.load_state_dict(
            checkpoint[
                "target_model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0,
        )

        self.gamma = checkpoint.get(
            "gamma",
            self.gamma,
        )

        self.target_update_every = checkpoint.get(
            "target_update_every",
            self.target_update_every,
        )

    def save_checkpoint(
        self,
        filepath,
    ):
        """Save the complete trainer state."""

        torch.save(
            self.checkpoint_state(),
            filepath,
        )

    def load_checkpoint(
        self,
        filepath,
        map_location=None,
    ):
        """Load a previously saved trainer checkpoint."""

        checkpoint = torch.load(
            filepath,
            map_location=map_location,
            weights_only=False,
        )

        self.load_checkpoint_state(
            checkpoint
        )

        return checkpoint

    def save_weights(
        self,
        filepath,
    ):
        """Save only the online value-network weights."""

        torch.save(
            self.model.state_dict(),
            filepath,
        )

    def load_weights(
        self,
        filepath,
        map_location=None,
        sync_target=True,
    ):
        """Load online model weights."""

        state_dict = torch.load(
            filepath,
            map_location=map_location,
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        if sync_target:
            self.update_target_network()

    def update_target_network(self):
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
        solution_weight=1.0,
        nonterminal_weight=0.0,
    ):
        super().__init__(
            model=model,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_every=target_update_every,
        )
    
        self.environment_store = environment_store
    
        self.solution_weight = solution_weight
        self.nonterminal_weight = nonterminal_weight

    def train_step(
        self,
        problem,
        batch_size=64,
    ):
        solution_samples, nonterminal_samples = (
            self.replay_buffer.sample_mixed(
                batch_size=batch_size,
                solution_weight=self.solution_weight,
                nonterminal_weight=self.nonterminal_weight,
            )
        )
    
        # losses = []
        # solution_losses = []
        # td_losses = []
        losses = []
        grounded_losses = []
        nonterminal_losses = []
    
        # ---------------------------------------------------------
        # Grounded solution targets
        # ---------------------------------------------------------
    
        for sample in solution_samples:
    
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
    
            sample_loss = (
                prediction - target
            ).pow(2)
            

            losses.append(sample_loss)
            grounded_losses.append(sample_loss)

        # ---------------------------------------------------------
        # Nonterminal / TD targets
        # ---------------------------------------------------------
    
        for exp in nonterminal_samples:
    
            environment = self.environment_store.get(
                exp.environment_id
            )
    
            prediction = self.model.value(
                problem=problem,
                state=exp.state,
                goal=exp.goal,
                environment=environment,
            )
    
            if exp.terminal:
                target = torch.tensor(
                    exp.cost,
                    dtype=torch.float32,
                    device=prediction.device,
                )
    
            else:
                with torch.no_grad():
                    next_value = self.target_model.value(
                        problem=problem,
                        state=exp.next_state,
                        goal=exp.goal,
                        environment=environment,
                    )
    
                target = (
                    exp.cost
                    + self.gamma * next_value
                )
    
            sample_loss = (
                prediction - target
            ).pow(2)
            
            losses.append(sample_loss)
            nonterminal_losses.append(sample_loss)

    
        # ---------------------------------------------------------
        # No samples available
        # ---------------------------------------------------------
    
        if not losses:
            return {
                "loss": 0.0,
                "grounded_loss": 0.0,
                "nonterminal_loss": 0.0,
                "grounded_samples": 0,
                "nonterminal_samples": 0,
            }
    
        # ---------------------------------------------------------
        # ONE loss across the sampled mixture
        # ---------------------------------------------------------
    
        loss = torch.stack(
            losses
        ).mean()
    
        # ---------------------------------------------------------
        # ONE optimizer / ONE learning rate
        # ---------------------------------------------------------
    
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
    
        # ---------------------------------------------------------
        # Metrics only
        # ---------------------------------------------------------
        grounded_loss = (
            torch.stack(
                grounded_losses
            ).mean().item()
            if grounded_losses
            else 0.0
        )
        
        nonterminal_loss = (
            torch.stack(
                nonterminal_losses
            ).mean().item()
            if nonterminal_losses
            else 0.0
        )
        
        return {
            "loss": loss.item(),
            "grounded_loss": grounded_loss,
            "nonterminal_loss": nonterminal_loss,
            "grounded_samples": len(solution_samples),
            "nonterminal_samples": len(nonterminal_samples),
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

    def solution_loss(
        self,
        problem,
        batch_size=64,
    ):
        samples = self.replay_buffer.sample_solution(
            batch_size
        )
    
        if not samples:
            return None
    
        losses = []
    
        for sample in samples:
    
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
    
        return torch.stack(losses).mean()

    def checkpoint_state(
        self,
    ):
        checkpoint = super().checkpoint_state()
    
        checkpoint.update({
            "solution_weight": (
                self.solution_weight
            ),
    
            "nonterminal_weight": (
                self.nonterminal_weight
            ),
    
            "model_config": {
                "spatial_channels": (
                    self.model.spatial_channels
                ),
    
                "state_features": (
                    self.model.state_features
                ),
            },
        })
    
        return checkpoint
    
    
    def load_checkpoint_state(
        self,
        checkpoint,
    ):
        model_config = checkpoint.get(
            "model_config",
            {}
        )
    
        spatial_channels = model_config.get(
            "spatial_channels"
        )
    
        state_features = model_config.get(
            "state_features"
        )
    
        if (
            spatial_channels is not None
            and spatial_channels
            != self.model.spatial_channels
        ):
            raise ValueError(
                "Checkpoint spatial_channels "
                f"{spatial_channels} does not match "
                f"model spatial_channels "
                f"{self.model.spatial_channels}"
            )
    
        if (
            state_features is not None
            and state_features
            != self.model.state_features
        ):
            raise ValueError(
                "Checkpoint state_features "
                f"{state_features} does not match "
                f"model state_features "
                f"{self.model.state_features}"
            )
    
        super().load_checkpoint_state(
            checkpoint
        )
    
        self.solution_weight = checkpoint.get(
            "solution_weight",
            self.solution_weight,
        )
    
        self.nonterminal_weight = checkpoint.get(
            "nonterminal_weight",
            self.nonterminal_weight,
        )
        
class EnvironmentStore:
    """
    Store HexGrid environments separately from replay experiences.

    Replay-buffer records reference environments using environment_id.
    """

    FILE_VERSION = 1

    def __init__(
        self,
    ):
        self.environments = {}

    def add(
        self,
        environment_id,
        environment,
    ):
        if not isinstance(
            environment,
            HexGrid,
        ):
            raise TypeError(
                "environment must be a HexGrid"
            )

        self.environments[
            environment_id
        ] = environment

    def get(
        self,
        environment_id,
    ):
        try:
            return self.environments[
                environment_id
            ]

        except KeyError as error:
            raise KeyError(
                f"Environment ID not found: "
                f"{environment_id!r}"
            ) from error

    def save(
        self,
        filepath,
    ):
        """
        Save all environments to one binary file.

        The file contains primitive Python data rather than directly
        pickling HexGrid objects. This makes the format less dependent
        on the exact HexGrid class implementation.
        """

        filepath = Path(
            filepath
        )

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized_environments = {}

        for (
            environment_id,
            environment,
        ) in self.environments.items():

            obstacles = environment.obstacles

            if isinstance(
                obstacles,
                dict,
            ):
                obstacle_coordinates = list(
                    obstacles.keys()
                )

            else:
                obstacle_coordinates = list(
                    obstacles
                )

            allowed = (
                None
                if environment.allowed is None
                else list(
                    environment.allowed
                )
            )

            serialized_environments[
                environment_id
            ] = {
                "hex_size": (
                    environment.hex_size
                ),
                "obstacles": (
                    obstacle_coordinates
                ),
                "center": (
                    environment.center
                ),
                "radius": (
                    environment.radius
                ),
                "allowed": allowed,
            }

        payload = {
            "version": self.FILE_VERSION,
            "environments": (
                serialized_environments
            ),
        }

        temporary_path = filepath.with_suffix(
            filepath.suffix + ".tmp"
        )

        with temporary_path.open(
            "wb"
        ) as file:
            pickle.dump(
                payload,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        os.replace(
            temporary_path,
            filepath,
        )

        return filepath

    def load(
        self,
        filepath,
        clear_existing=True,
    ):
        """
        Load environments from a file created by save().

        Parameters
        ----------
        filepath:
            Saved environment-store file.

        clear_existing:
            If True, replace the current store.
            If False, merge loaded environments into the store.
            Loaded IDs replace existing environments with matching IDs.

        Returns
        -------
        EnvironmentStore
            This EnvironmentStore instance.
        """

        filepath = Path(
            filepath
        )

        if not filepath.is_file():
            raise FileNotFoundError(
                f"Environment store not found: "
                f"{filepath}"
            )

        with filepath.open(
            "rb"
        ) as file:
            payload = pickle.load(
                file
            )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Invalid environment-store file"
            )

        version = payload.get(
            "version"
        )

        if version != self.FILE_VERSION:
            raise ValueError(
                "Unsupported environment-store "
                f"version: {version!r}"
            )

        serialized_environments = (
            payload.get(
                "environments"
            )
        )

        if not isinstance(
            serialized_environments,
            dict,
        ):
            raise ValueError(
                "Environment-store file does not "
                "contain an environments dictionary"
            )

        loaded_environments = {}

        for (
            environment_id,
            environment_data,
        ) in serialized_environments.items():

            required_fields = {
                "hex_size",
                "obstacles",
                "center",
                "radius",
                "allowed",
            }

            missing_fields = (
                required_fields
                - environment_data.keys()
            )

            if missing_fields:
                raise ValueError(
                    "Environment "
                    f"{environment_id!r} is missing "
                    f"fields: {sorted(missing_fields)}"
                )

            center = environment_data[
                "center"
            ]

            if not isinstance(
                center,
                HexCoord,
            ):
                raise ValueError(
                    "Environment "
                    f"{environment_id!r} has an "
                    "invalid center"
                )

            obstacles = {
                coordinate: True
                for coordinate
                in environment_data[
                    "obstacles"
                ]
            }

            allowed_data = environment_data[
                "allowed"
            ]

            allowed = (
                None
                if allowed_data is None
                else set(
                    allowed_data
                )
            )

            loaded_environments[
                environment_id
            ] = HexGrid(
                hex_size=environment_data[
                    "hex_size"
                ],
                obstacles=obstacles,
                center=center,
                radius=environment_data[
                    "radius"
                ],
                allowed=allowed,
            )

        if clear_existing:
            self.environments.clear()

        self.environments.update(
            loaded_environments
        )

        return self

    @classmethod
    def from_file(
        cls,
        filepath,
    ):
        """
        Create a new EnvironmentStore from a saved file.
        """

        store = cls()

        store.load(
            filepath=filepath,
            clear_existing=True,
        )

        return store

    def ids(
        self,
    ):
        return list(
            self.environments.keys()
        )

    def __contains__(
        self,
        environment_id,
    ):
        return (
            environment_id
            in self.environments
        )

    def __len__(
        self,
    ):
        return len(
            self.environments
        )

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

def _to_rl_velocity(
    problem,
    velocity,
):
    direction = (
        None
        if velocity.direction is None
        else problem.DIRECTIONS[velocity.direction]
    )

    return VelocityState(
        magnitude=velocity.magnitude,
        direction=direction,
    )

class HexUNetHeuristic:
    """
    Callable H* heuristic backed by ValueNetwork_HUNet.

    The callable preserves the normal H* heuristic interface:

        heuristic(problem, node) -> estimated remaining time
    """

    def __init__(
        self,
        model,
    ):
        self.model = model

        self._cached_problem_id = None
        self._cached_goal = None
        self._cached_context_map = None

    def clear_cache(
        self,
    ):
        self._cached_problem_id = None
        self._cached_goal = None
        self._cached_context_map = None

    def _get_context_map(
        self,
        problem,
        goal,
    ):
        problem_id = id(problem)

        if (
            self._cached_context_map is None
            or self._cached_problem_id != problem_id
            or self._cached_goal != goal.location
        ):
            self._cached_context_map = (
                self.model.encode_environment(
                    environment=problem.grid,
                    goal=goal,
                )
            )

            self._cached_problem_id = problem_id
            self._cached_goal = goal.location

        return self._cached_context_map

    def __call__(
        self,
        problem,
        node,
    ):
        """
        Estimate remaining physical travel time from node to goal.
        """

        state_direction = (
            None
            if node.velocity.direction is None
            else problem.DIRECTIONS[
                node.velocity.direction
            ]
        )

        goal_direction = (
            None
            if (
                problem.goal_v is None
                or problem.goal_v.direction is None
            )
            else problem.DIRECTIONS[
                problem.goal_v.direction
            ]
        )

        state = RLState(
            location=node.location,
            velocity=VelocityState(
                magnitude=node.velocity.magnitude,
                direction=state_direction,
            ),
        )

        goal = RLState(
            location=problem.goal,
            velocity=VelocityState(
                magnitude=(
                    None
                    if problem.goal_v is None
                    else problem.goal_v.magnitude
                ),
                direction=goal_direction,
            ),
        )

        was_training = self.model.training

        self.model.eval()

        try:
            with torch.no_grad():
                context_map = self._get_context_map(
                    problem=problem,
                    goal=goal,
                )

                prediction = self.model.value(
                    problem=problem,
                    state=state,
                    goal=goal,
                    environment=problem.grid,
                    context_map=context_map,
                )

            # Clamp output, we do not ever want <0 h values
            return max(
                0.0,
                float(
                    prediction.item()
                ),
            )

        finally:
            if was_training:
                self.model.train()