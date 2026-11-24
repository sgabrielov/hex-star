from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hexgrid import HexCoord, HexGrid, RLState, VelocityState, hex_disk
from hexstar import travel_time_euclidean
from hexstarrl import HUNetStateEncoder, ValueNetwork_HUNet


@dataclass(frozen=True)
class PretrainingConfig:
    seed: int = 42
    center_q: int = 0
    center_r: int = 0
    radius: int = 80
    hex_size: float = 0.1
    acceleration: float = 1.0
    velocity_values: tuple = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    direction_values: tuple = (
        HexCoord(1, 0),
        HexCoord(1, -1),
        HexCoord(0, -1),
        HexCoord(-1, 0),
        HexCoord(-1, 1),
        HexCoord(0, 1),
    )
    validation_fraction: float = 0.20
    samples_per_goal: int = 1000
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    minimum_noise_density: float = 0.0
    maximum_noise_density: float = 1.0
    spatial_channels: int = 16
    state_features: int = 5
    num_workers: int = 0

    @property
    def center(self):
        return HexCoord(self.center_q, self.center_r)

    def to_dict(self):
        result = asdict(self)
        result["direction_values"] = [
            {"q": direction.q, "r": direction.r}
            for direction in self.direction_values
        ]
        return result


@dataclass(frozen=True)
class RelaxedSample:
    state: RLState
    goal: RLState
    target: float


class RelaxedDataset(Dataset):
    def __init__(self, samples: Iterable[RelaxedSample]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def relaxed_collate(samples):
    return list(samples)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grid_definition(config):
    valid_locations = tuple(
        sorted(
            hex_disk(config.center, config.radius),
            key=lambda location: (location.q, location.r),
        )
    )
    valid_location_set = set(valid_locations)
    environment = HexGrid(
        hex_size=config.hex_size,
        obstacles={},
        center=config.center,
        radius=config.radius,
        allowed=valid_location_set,
    )
    return valid_locations, valid_location_set, environment


def relaxed_target(state, goal, config):
    return travel_time_euclidean(
        state.location,
        goal.location,
        state.velocity.magnitude,
        config.acceleration,
        config.hex_size,
    )


def generate_relaxed_dataset(config, valid_locations):
    rng = random.Random(config.seed)
    samples = []

    for goal_location in valid_locations:
        goal = RLState(
            location=goal_location,
            velocity=VelocityState(
                magnitude=0.0,
                direction=None,
            ),
        )

        for _ in range(config.samples_per_goal):
            state = RLState(
                location=rng.choice(valid_locations),
                velocity=VelocityState(
                    magnitude=rng.choice(config.velocity_values),
                    direction=rng.choice(config.direction_values),
                ),
            )
            samples.append(
                RelaxedSample(
                    state=state,
                    goal=goal,
                    target=relaxed_target(state, goal, config),
                )
            )

    return RelaxedDataset(samples)


def split_relaxed_dataset_by_goal(dataset, validation_fraction, seed):
    goal_locations = sorted(
        {sample.goal.location for sample in dataset.samples},
        key=lambda location: (location.q, location.r),
    )

    rng = random.Random(seed)
    rng.shuffle(goal_locations)

    validation_count = max(
        1,
        round(len(goal_locations) * validation_fraction),
    )
    validation_goals = set(goal_locations[:validation_count])
    training_goals = set(goal_locations[validation_count:])

    training_samples = [
        sample
        for sample in dataset.samples
        if sample.goal.location in training_goals
    ]
    validation_samples = [
        sample
        for sample in dataset.samples
        if sample.goal.location in validation_goals
    ]

    return (
        RelaxedDataset(training_samples),
        RelaxedDataset(validation_samples),
        training_goals,
        validation_goals,
    )


def create_data_loaders(config, train_dataset, validation_dataset):
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=relaxed_collate,
        generator=loader_generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=relaxed_collate,
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader


def make_noise_environment(
    state_location,
    goal_location,
    rng,
    config,
    valid_locations,
    valid_location_set,
):
    density = rng.uniform(
        config.minimum_noise_density,
        config.maximum_noise_density,
    )
    protected_locations = {state_location, goal_location}
    candidates = [
        location
        for location in valid_locations
        if location not in protected_locations
    ]
    obstacle_count = round(density * len(candidates))
    obstacle_count = max(0, min(obstacle_count, len(candidates)))

    if obstacle_count == 0:
        obstacle_locations = set()
    else:
        obstacle_locations = set(
            rng.sample(candidates, obstacle_count)
        )

    environment = HexGrid(
        hex_size=config.hex_size,
        obstacles={location: True for location in obstacle_locations},
        center=config.center,
        radius=config.radius,
        allowed=valid_location_set,
    )
    return environment, density


def create_value_network(config, device):
    encoder = HUNetStateEncoder()
    model = ValueNetwork_HUNet(
        encoder=encoder,
        spatial_channels=config.spatial_channels,
        state_features=config.state_features,
    )
    return model.to(device)


def model_summary(model):
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "architecture": str(model),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "spatial_channels": model.spatial_channels,
        "state_features": model.state_features,
        "device": str(next(model.parameters()).device),
    }


def relaxed_batch_loss(
    model,
    samples,
    rng,
    config,
    valid_locations,
    valid_location_set,
):
    if not samples:
        return None

    predictions = []
    targets = []

    for sample in samples:
        environment, _ = make_noise_environment(
            state_location=sample.state.location,
            goal_location=sample.goal.location,
            rng=rng,
            config=config,
            valid_locations=valid_locations,
            valid_location_set=valid_location_set,
        )
        prediction = model.value(
            problem=None,
            state=sample.state,
            goal=sample.goal,
            environment=environment,
        )
        target = torch.tensor(
            sample.target,
            dtype=torch.float32,
            device=prediction.device,
        )
        predictions.append(prediction)
        targets.append(target)

    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    errors = predictions - targets

    return {
        "loss": errors.pow(2).mean(),
        "predictions": predictions,
        "targets": targets,
        "squared_errors": errors.pow(2),
        "absolute_errors": errors.abs(),
    }


def train_relaxed_epoch(
    model,
    data_loader,
    optimizer,
    epoch_seed,
    config,
    valid_locations,
    valid_location_set,
):
    model.train()
    rng = random.Random(epoch_seed)

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    batch_count = 0
    negative_prediction_count = 0
    prediction_min = math.inf
    prediction_max = -math.inf
    target_min = math.inf
    target_max = -math.inf

    progress_bar = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Training",
        unit="batch",
        leave=True,
    )

    for samples in progress_bar:
        batch_result = relaxed_batch_loss(
            model=model,
            samples=samples,
            rng=rng,
            config=config,
            valid_locations=valid_locations,
            valid_location_set=valid_location_set,
        )
        if batch_result is None:
            continue

        loss = batch_result["loss"]
        predictions = batch_result["predictions"]
        targets = batch_result["targets"]
        squared_errors = batch_result["squared_errors"]
        absolute_errors = batch_result["absolute_errors"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        detached_predictions = predictions.detach()
        detached_targets = targets.detach()
        batch_sample_count = len(samples)

        squared_error_sum += squared_errors.detach().sum().item()
        absolute_error_sum += absolute_errors.detach().sum().item()
        sample_count += batch_sample_count
        batch_count += 1
        negative_prediction_count += detached_predictions.lt(0.0).sum().item()
        prediction_min = min(prediction_min, detached_predictions.min().item())
        prediction_max = max(prediction_max, detached_predictions.max().item())
        target_min = min(target_min, detached_targets.min().item())
        target_max = max(target_max, detached_targets.max().item())

        running_mse = squared_error_sum / sample_count
        progress_bar.set_postfix(
            rmse=round(math.sqrt(running_mse), 5),
            mae=round(absolute_error_sum / sample_count, 5),
            neg=round(negative_prediction_count / sample_count, 4),
        )

    if sample_count == 0:
        raise RuntimeError("Training epoch processed zero samples.")

    mse = squared_error_sum / sample_count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error_sum / sample_count,
        "samples": sample_count,
        "batches": batch_count,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "target_min": target_min,
        "target_max": target_max,
        "negative_predictions": int(negative_prediction_count),
        "negative_fraction": negative_prediction_count / sample_count,
    }


def evaluate_relaxed_model(
    model,
    data_loader,
    validation_environment,
):
    was_training = model.training
    model.eval()

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    prediction_min = math.inf
    prediction_max = -math.inf
    target_min = math.inf
    target_max = -math.inf
    negative_prediction_count = 0
    context_cache = {}

    try:
        with torch.no_grad():
            for samples in data_loader:
                for sample in samples:
                    goal_location = sample.goal.location
                    if goal_location not in context_cache:
                        context_cache[goal_location] = model.encode_environment(
                            environment=validation_environment,
                            goal=sample.goal,
                        )

                    prediction = model.value(
                        problem=None,
                        state=sample.state,
                        goal=sample.goal,
                        environment=validation_environment,
                        context_map=context_cache[goal_location],
                    )
                    prediction_value = prediction.item()
                    target_value = float(sample.target)
                    error = prediction_value - target_value

                    squared_error_sum += error * error
                    absolute_error_sum += abs(error)
                    sample_count += 1
                    prediction_min = min(prediction_min, prediction_value)
                    prediction_max = max(prediction_max, prediction_value)
                    target_min = min(target_min, target_value)
                    target_max = max(target_max, target_value)
                    if prediction_value < 0.0:
                        negative_prediction_count += 1
    finally:
        if was_training:
            model.train()

    if sample_count == 0:
        raise RuntimeError("Validation processed zero samples.")

    mse = squared_error_sum / sample_count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error_sum / sample_count,
        "samples": sample_count,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "target_min": target_min,
        "target_max": target_max,
        "negative_predictions": negative_prediction_count,
        "negative_fraction": negative_prediction_count / sample_count,
        "context_maps": len(context_cache),
    }


def pretrain_relaxed_value_network(
    model,
    train_loader,
    validation_loader,
    validation_environment,
    config,
    valid_locations,
    valid_location_set,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = []
    best_epoch = None
    best_validation_mse = math.inf
    best_model_state = None

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        train_metrics = train_relaxed_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch_seed=config.seed + epoch_number,
            config=config,
            valid_locations=valid_locations,
            valid_location_set=valid_location_set,
        )
        validation_metrics = evaluate_relaxed_model(
            model=model,
            data_loader=validation_loader,
            validation_environment=validation_environment,
        )

        epoch_metrics = {
            "epoch": epoch_number,
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_prediction_min": train_metrics["prediction_min"],
            "train_prediction_max": train_metrics["prediction_max"],
            "train_negative_fraction": train_metrics["negative_fraction"],
            "validation_mse": validation_metrics["mse"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "validation_prediction_min": validation_metrics["prediction_min"],
            "validation_prediction_max": validation_metrics["prediction_max"],
            "validation_negative_fraction": validation_metrics["negative_fraction"],
            "validation_context_maps": validation_metrics["context_maps"],
        }
        history.append(epoch_metrics)

        print()
        print("Epoch completed:", epoch_number)
        print("Train RMSE:", epoch_metrics["train_rmse"])
        print("Validation RMSE:", epoch_metrics["validation_rmse"])
        print("Validation MAE:", epoch_metrics["validation_mae"])

        if validation_metrics["mse"] < best_validation_mse:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch_number
            best_model_state = copy.deepcopy(model.state_dict())
            print("Best-model checkpoint updated.")

    if best_model_state is None:
        raise RuntimeError("Training completed without a best model state.")

    model.load_state_dict(best_model_state)
    model.eval()

    final_validation_metrics = evaluate_relaxed_model(
        model=model,
        data_loader=validation_loader,
        validation_environment=validation_environment,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "final_validation_metrics": final_validation_metrics,
        "optimizer": optimizer,
    }
