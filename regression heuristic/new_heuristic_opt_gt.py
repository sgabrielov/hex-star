# %%
# Date: Nov-30-2025
# Author: Simon Chan Tack
# File: new_heuristic_opt_gt.py
#
# Code summary:
# H* + Learned Heuristic + (optional) Optuna Hyperparameter Tuning
# ------------------------------------------------------------------
# - IMPORTANT TO NOTe THE TRAINING TAKES VERY LONG 
# - THE PARAMETERS num_rollouts=10, and max_steps=100 can be reduced for faster training
#
# - Here we use Value Function Approximation to estimate the time-to-goal
# - RL methods learns a time-to-goal heuristic via supervised regression
# - The time-to-goal is h(n), which is the time from node to goal on a solution path 
# - Can train either:
#     (A) with your original training loop (early stopping + scheduler)
#     (B) using Optuna to tune architecture + LR + batch size + epochs
# 
# - Plugs the learned heuristic back into H* and compares:
#     * Original analytic heuristic
#     * Learned NN heuristic
#     * Hybrid heuristic (blend of both)
# - Visualizes paths on hex map and evaluates from multiple random starts
# ------------------------------------------------------------------

import os
import pickle
import random
from math import sqrt, pi, cos, sin

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, RegularPolygon

import optuna  # <-- Optuna for hyperparameter tuning

from hex_star import PathfindingProblem, best_first_search, time_to_goal
from hex_star import f as hstar_f  # cost function for H* (g + h)


# ============================================================
# 1 NEURAL NETWORK HEURISTIC MODEL (Flexible Architecture)
# ============================================================

class HeuristicNet(nn.Module):
    def __init__(self, input_dim=4, hidden_sizes=None):
        ###############################################################
        # Flexible MLP for heuristic approximation.
        #
        # input_dim : number of input features (e.g., [q, r, v, dhm])
        # hidden_sizes : list of hidden layer sizes, e.g. [128, 64].
        #                If None, defaults to [128, 64] (your original).
        ###############################################################
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def state_to_features(state, goal):
    ################################################################
    # Convert H* state -> numeric feature vector.
    # state = (agent_loc, (velocity, angle))
    #   agent_loc: (q, r) in axial coordinates
    #   velocity: float
    #   angle:    float (radians, multiple of 60 deg)

    # We use features: [q, r, velocity, hex_manhattan_distance_to_goal]
    ################################################################

    (q, r), (v, theta) = state

    dq = abs(goal[0] - q)
    dr = abs(goal[1] - r)
    ds = abs((-goal[0] - goal[1]) - (-q - r))

    # A simple "distance-like" feature that is monotonic w.r.t. true distance
    dhm = max(dq, dr, ds)
    # Could also use exact hex distance: (dq + dr + ds)/2

    return np.array([q, r, v, dhm], dtype=np.float32)


# ============================================================
# 2 TRAINING DATA GENERATION (RANDOM ROLLOUTS)
# ============================================================

def random_rollout_training_samples(problem,
                                    num_rollouts=400,
                                    max_steps=50):
    ################################################################
    # Use random rollouts in the map to collect (features, true_time_to_goal) pairs.

    # IMPORTANT CHANGE:
    # -----------------
    # Previously:
    #     ttg = time_to_goal(node_obj)
    # which measured time-to-goal in an *ideal* setting with no obstacles
    # blocking the agent. This is implmented in the file new_heuristic_opt.py

    # Now:
    #     - For each sampled next_state along a random rollout, we construct
    #       a *sub-problem* whose root is that next_state.
    #     - We run best_first_search (A*) from that state to the goal using
    #       the ORIGINAL H* cost function f = hstar_f and the analytic heuristic
    #       h = time_to_goal.
    #     - The resulting solution node's path_cost is used as the *ground-truth*
    #       remaining time-to-goal, including the effect of obstacles.

    #     If the re-rooted problem has no solution (rare but possible), that
    #     sample is skipped.

    # Note:
    #     This still learns a heuristic *specific to this map and goal*.
    ################################################################
    X_list = []
    y_list = []

    rng = random.Random(0)  # Used to randomly select actions later on

    for rollout in range(num_rollouts):
        # reset to root state at each rollout
        state = problem.root.state

        for step_num in range(max_steps):
            actions = problem.actions(state)
            if not actions:
                break

            action = rng.choice(actions)   # Random selection of actions
            next_state = problem.result(state, action)

            # ---------------------------------------------------------
            # Build a sub-problem whose root is next_state.
            # This effectively makes the current state cost = 0, so
            # the solution path_cost from this new root is exactly the
            # *remaining* time-to-goal.
            # ---------------------------------------------------------
            
            sub_problem = PathfindingProblem(
                next_state,
                problem.hex_map,
                problem.obstacle_map,
                problem.goal_loc,
                problem.hex_radius,
                problem.hex_size,               
                problem.a_max,
                problem.d_max,
                problem.ay_max,
            ) # a_max, d_max, ay_max

            # Get a solution node based on the H* best_first_search 
            sol_node = best_first_search(sub_problem, hstar_f, time_to_goal)

            # If no path from this state to the goal (should be rare), skip
            if (step_num % 50 == 0): # To help keep track of rollout process
                print(f"Step #: {step_num}  Rollout number: {rollout}")
            if sol_node is None:
                state = next_state
                if problem.is_goal(state):
                    break
                continue

            # True remaining time-to-goal including obstacles
            # Because we re-rooted at next_state with g=0,
            # ttg = solution_path_cost - prefix_cost = sol_node.path_cost - 0
            ttg = sol_node.path_cost

            features = state_to_features(next_state, problem.goal_loc)
            X_list.append(features)
            y_list.append(ttg)

            state = next_state
            if problem.is_goal(state):
                break

        if (rollout % 100 == 0):
            print(f"Rollout number {rollout}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ============================================================
# 3A ORIGINAL TRAINING (EARLY STOPPING + SCHEDULER)
# ============================================================

def train_learned_heuristic(
    X,
    y,
    epochs=40,
    lr=1e-3,
    model_path=None,
    val_split=0.15,
    patience=5,
    min_delta=1e-4,
):
    # ################################################################
    # Train heuristic regression network with:
    #   - Fixed architecture HeuristicNet(input_dim, hidden_sizes=[128, 64])
    #   - Adaptive learning rate scheduler (ReduceLROnPlateau)
    #   - Early stopping
    #   - Train/validation split

    # This is your original "baseline" training method.
    # ################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------
    # Split into train/val sets
    # ----------------------------------------------------
    N = len(X)
    idx = np.random.permutation(N)
    val_size = int(N * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    # ----------------------------------------------------
    # Model + optimizer + loss + scheduler
    # ----------------------------------------------------
    model = HeuristicNet(input_dim=X.shape[1]).to(device)  # default hidden_sizes=[128, 64]
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=min_delta
    )

    # ----------------------------------------------------
    # Early stopping setup
    # ----------------------------------------------------
    best_loss = float("inf")
    best_state = None
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------
    for ep in range(epochs):

        # ---------- Train ----------
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        train_loss = loss_fn(pred, y_train)
        train_loss.backward()
        opt.step()

        # ---------- Validation ----------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        # ---------- Adaptive LR ----------
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        # ---------- Logging ----------
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"[Baseline] Epoch {ep+1}/{epochs} | "
            f"Train Loss={train_loss.item():.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"LR={current_lr:.6f}"
        )

        # ---------- Early Stopping ----------
        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            wait = 0
            print("  ** Validation improved — saving checkpoint")
        else:
            wait += 1
            print(f"  ** No improvement ({wait}/{patience})")

            if wait >= patience:
                print("\n[Baseline] Early stopping triggered!")
                break

    # ----------------------------------------------------
    # Restore best model
    # ----------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[Baseline] Restored best model (val_loss={best_loss:.4f})")

    # ----------------------------------------------------
    # Save model
    # ----------------------------------------------------
    if model_path is not None:
        torch.save(model.state_dict(), model_path)
        print(f"[Baseline] Saved learned heuristic model to {model_path}")

    return model, history


def load_learned_heuristic(model_path, input_dim=4, hidden_sizes=None):
    # ################################################################
    # Reload a saved heuristic model.
    # If hidden_sizes is None, assume default HeuristicNet architecture.
    # ################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeuristicNet(input_dim=input_dim, hidden_sizes=hidden_sizes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded learned heuristic model from {model_path}")
    return model


# ============================================================
# 3B OPTUNA HYPERPARAMETER TUNING (ALTERNATIVE METHOD)
# ============================================================

def optuna_objective(trial, X, y, val_split=0.2, device=None):
    # ################################################################
    # Optuna objective for heuristic learning.
    # We sample:
    #   - num_layers
    #   - size of each layer
    #   - learning rate
    #   - batch size
    #   - number of epochs
    # and return the final validation MSE as the objective to minimize.
    # ################################################################
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Hyperparameters from trial
    # ---------------------------
    num_layers = trial.suggest_int("num_layers", 1, 4)
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_int(f"layer_{i}_size", 32, 256))

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 20, 60)

    # ---------------------------
    # Train-Validation Split
    # ---------------------------
    N = len(X)
    idx = np.random.permutation(N)
    val_size = int(N * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    # ---------------------------
    # Model + optimizer + loss
    # ---------------------------
    model = HeuristicNet(input_dim=X.shape[1], hidden_sizes=hidden_sizes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ---------------------------
    # Mini-batch training for this trial
    # ---------------------------
    for ep in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            opt.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        # Report intermediate objective to Optuna
        trial.report(val_loss, ep)

        # Allow pruning of bad trials
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def tune_heuristic_with_optuna(
    X,
    y,
    model_path="learned_heuristic.pt",
    n_trials=40,
    val_split=0.2,
):
    # ################################################################
    # Wrapper to run Optuna, then train a final model with the best hyperparameters.
    # Returns:
    #   model  : trained final model on full dataset
    #   history: {"train_loss": [...]} from final training
    #   study  : Optuna study object (contains best params, trial history, etc.)
    # ################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Starting Optuna Hyperparameter Search ===")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optuna_objective(trial, X, y, val_split=val_split, device=device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n[Optuna] Best Trial:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_params = study.best_trial.params

    # Reconstruct best architecture
    num_layers = best_params["num_layers"]
    hidden_sizes = [best_params[f"layer_{i}_size"] for i in range(num_layers)]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    epochs = best_params["epochs"]

    print("\n[Optuna] Training FINAL model on full dataset with best hyperparameters...")

    # Final training on full X, y (no validation split here)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model = HeuristicNet(input_dim=X.shape[1], hidden_sizes=hidden_sizes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X_t.size(0))

        epoch_loss = 0.0
        batches = 0
        for i in range(0, X_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_t[idx]
            batch_y = y_t[idx]

            opt.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        train_losses.append(avg_loss)
        print(f"[Optuna Final Train] Epoch {ep+1}/{epochs}, Loss={avg_loss:.4f}")

    # Save final tuned model
    if model_path is not None:
        torch.save(model.state_dict(), model_path)
        print(f"[Optuna] Saved tuned heuristic model to {model_path}")

    history = {"train_loss": train_losses}  # no val_loss tracked in final training
    return model, history, study


# ============================================================
# 4 HEURISTIC WRAPPERS (LEARNED + HYBRID)
# ============================================================

class LearnedHeuristic:
    def __init__(self, model, goal):
        self.model = model
        self.goal = goal
        self.device = next(model.parameters()).device

    def __call__(self, node):
        feat = state_to_features(node.state, self.goal)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            val = self.model(x).item()
        # heuristic must be non-negative
        return max(val, 0.0)


class HybridHeuristic:
    # ################################################################
    # Blend learned heuristic and analytic time_to_goal:
    #
    #    h_hybrid = alpha * h_learned + (1 - alpha) * h_analytic
    #
    # For strict admissibility, we instead take:
    #
    #    h_hybrid = min(h_learned, h_analytic)
    #
    # which guarantees we never overestimate the true remaining cost.
    # ################################################################
    def __init__(self, learned_h, alpha=0.5):
        self.learned_h = learned_h
        self.alpha = alpha

    def __call__(self, node):
        h_learn = self.learned_h(node)
        h_ana = time_to_goal(node)
        # safer variant: use min for admissibility
        return min(h_learn, h_ana)


# ============================================================
# 5 HEX GRID VISUALIZATION
# ============================================================

def axial_to_cartesian(q, r, size):
    # ################################################################
    # Pointy-top hex axial coordinates → cartesian coordinates.
    # size = hex radius (center → vertex distance).
    # ################################################################
    x = size * sqrt(3) * (q + r/2)
    y = size * 1.5 * r
    return x, y


def hex_corners(x, y, size):
    # Return vertices of a pointy-top hex centered at (x, y).
    corners = []
    for i in range(6):
        angle = pi/3 * i + pi/6   # pointy top (rotate 30 degrees)
        cx = x + size * cos(angle)
        cy = y + size * sin(angle)
        corners.append((cx, cy))
    return corners


def axial_to_pixel(q, r, size):
    # ################################################################
    # Convert axial hex coordinates (q, r) to 2D pixel (x, y) coordinates.
    # Assumes pointy-topped hexagons oriented at 30 degrees.
    # ################################################################
    x = size * np.sqrt(3) * (q + r / 2)
    y = size * 3/2 * r
    return x, y


def plot_hex_map(ax, hex_map, obstacle_map, start_loc, goal_loc, path, hex_size, title=""):
    # ################################################################
    # Visualize hex grid using RegularPolygon:
    #   - free cells: lightgray
    #   - walls: black
    #   - start: blue
    #   - goal: green
    #   - path: magenta polyline
    # ################################################################
    ax.set_aspect("equal")

    for (q, r) in hex_map:
        x, y = axial_to_pixel(q, r, hex_size)
        
        if (q, r) == start_loc:
            color = "blue"
        elif (q, r) == goal_loc:
            color = "green"
        elif (q, r) in obstacle_map:
            color = "black"
        else:
            color = "lightgray"

        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_size * 0.95,
            orientation=np.radians(30),
            facecolor=color,
            edgecolor="k"
        )
        ax.add_patch(hex_patch)

    # Plot the path if available
    if path:
        xs, ys = [], []
        for (q, r) in path:
            x, y = axial_to_pixel(q, r, hex_size)
            xs.append(x)
            ys.append(y)

        ax.plot(xs, ys, color="magenta", linewidth=3)

    ax.set_title(title)
    ax.set_axis_off()


# ============================================================
# 6 EVALUATION ON MULTIPLE RANDOM STARTS
# ============================================================

def evaluate_multiple_starts(hex_map, obstacle_map, goal_loc, hex_radius,
                             hex_size, accel_max, decel_max, lat_accel_max,
                             model, num_starts=5):
    # ################################################################
    # Evaluate original vs learned vs hybrid heuristic from multiple random starts
    # to the SAME goal.
    # ################################################################

    free_cells = [cell for cell in hex_map if cell not in obstacle_map]
    rng = random.Random(1)

    results = []

    for _ in range(num_starts):
        start = rng.choice(free_cells)
        initial_state = (start, (0.0, 0.0))  # v=0, angle=0

        # Build problems
        problem_orig = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                          hex_radius, hex_size,
                                          accel_max, decel_max, lat_accel_max)
        problem_learned = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                             hex_radius, hex_size,
                                             accel_max, decel_max, lat_accel_max)
        problem_hybrid = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                            hex_radius, hex_size,
                                            accel_max, decel_max, lat_accel_max)

        learned_h = LearnedHeuristic(model, goal_loc)
        hybrid_h = HybridHeuristic(learned_h, alpha=0.5)

        sol_orig = best_first_search(problem_orig, hstar_f, time_to_goal)
        sol_learn = best_first_search(problem_learned, hstar_f, learned_h)
        sol_hybrid = best_first_search(problem_hybrid, hstar_f, hybrid_h)

        results.append((start, sol_orig, sol_learn, sol_hybrid))

    print("\n=== Evaluation over random starts (same goal) ===")
    for (start, sol_o, sol_l, sol_h) in results:

        print(f"Start: {start}")

        if sol_o is None:
            print("  Original: NO PATH FOUND")
        else:
            print(f"  Original cost: {sol_o.path_cost:.4f}")

        if sol_l is None:
            print("  Learned:  NO PATH FOUND")
        else:
            print(f"  Learned  cost: {sol_l.path_cost:.4f}")

        if sol_h is None:
            print("  Hybrid:   NO PATH FOUND")
        else:
            print(f"  Hybrid   cost: {sol_h.path_cost:.4f}")
    print("===============================================")

# %%
# ============================================================
# 7 MAIN PIPELINE: TRAIN + EVAL + PLOTS
# ============================================================

def run_pipeline(pkl_path, model_path="learned_heuristic_gt.pt", use_optuna=True):
    # --------------------------
    # Load map
    # --------------------------
    with open(pkl_path, "rb") as f:
        M = pickle.load(f)

    hex_map = M["hex_map"]
    obstacle_map = M["obstacle_map"]
    start_loc = M["agent"]
    goal_loc = M["goal"]
    init_velocity = M["velocity"]       # (v, angle)
    hex_radius = M["hex_radius"]
    hex_size = M["hex_size"]

    accel_max = 1.0
    decel_max = 1.0
    lat_accel_max = 0.5
    map_name = pkl_path.split("\\")[-1]
    print("\nLoaded map file:", map_name)
    print("Start:", start_loc, "Goal:", goal_loc)
    print("Number of hex cells:", len(hex_map))
    print("Number of obstacles:", len(obstacle_map))

    # --------------------------
    # Build training problem
    # --------------------------
    agent_size_r = 0
    q_learning_rate = 0.0
    q_discount_factor = 1.0
    initial_state = (start_loc, init_velocity)
    train_problem = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                       hex_radius, hex_size, 
                                       accel_max, decel_max, lat_accel_max)

    # --------------------------
    # Generate training data
    # --------------------------
    print("\nCollecting training data (random rollouts)...")
    X, y = random_rollout_training_samples(train_problem, num_rollouts=10, max_steps=100)
    print("Training samples shape:", X.shape)
    print(f"Total samples collected for training model: {X.shape[0]}")
    print(f"Each sample has {X.shape[1]} features")

    # --------------------------
    # Train model (Baseline OR Optuna)
    # --------------------------
    if use_optuna:
        print("\n>>> Using Optuna-based hyperparameter tuning for heuristic <<<")
        model, history, study = tune_heuristic_with_optuna(
            X, y, model_path=model_path, n_trials=40, val_split=0.2
        )
        # history only contains "train_loss"
    else:
        print("\n>>> Using Baseline training for heuristic (no Optuna) <<<")
        model, history = train_learned_heuristic(
            X, y,
            epochs=150,
            lr=1e-3,
            model_path=model_path,
            val_split=0.2,
            patience=15,
            min_delta=1e-4
        )

    # --------------------------
    # Plot training curves (if available)
    # --------------------------
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Heuristic Network Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Run H* (original, learned, hybrid) from ORIGINAL start
    # --------------------------
    print("\nRunning H* from original start with different heuristics...")

    # Original
    problem_o = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                   hex_radius, hex_size,
                                   accel_max, decel_max, lat_accel_max)
    sol_o = best_first_search(problem_o, hstar_f, time_to_goal)

    # Learned
    problem_l = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                   hex_radius, hex_size,
                                   accel_max, decel_max, lat_accel_max)
    learned_h = LearnedHeuristic(model, goal_loc)
    sol_l = best_first_search(problem_l, hstar_f, learned_h)

    # Hybrid
    problem_h = PathfindingProblem(initial_state, hex_map, obstacle_map, goal_loc,
                                   hex_radius, hex_size,
                                   accel_max, decel_max, lat_accel_max)
    hybrid_h = HybridHeuristic(learned_h, alpha=0.5)
    sol_h = best_first_search(problem_h, hstar_f, hybrid_h)

    # Extract paths
    def extract_path(sol_node):
        path = []
        n = sol_node
        while n:
            path.append(n.state[0])
            n = n.parent
        return list(reversed(path))

    path_o = extract_path(sol_o)
    path_l = extract_path(sol_l)
    path_h = extract_path(sol_h)

    print(f"\nCosts from original start using map: {map_name}")
    print(f"  Original: {sol_o.path_cost:.4f}")
    print(f"  Learned : {sol_l.path_cost:.4f}")
    print(f"  Hybrid  : {sol_h.path_cost:.4f}")

    # --------------------------
    # Visualize map + paths
    # --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_hex_map(
        axes[0], hex_map, obstacle_map, start_loc, goal_loc, path_o, hex_size,
        title="Original Heuristic"
    )

    plot_hex_map(
        axes[1], hex_map, obstacle_map, start_loc, goal_loc, path_l, hex_size,
        title="Learned Heuristic"
    )

    plot_hex_map(
        axes[2], hex_map, obstacle_map, start_loc, goal_loc, path_h, hex_size,
        title="Hybrid Heuristic"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------
    # Evaluate multiple random starts (same goal)
    # --------------------------
    print("\nConducting evaluation of Original vs Learned Cost using random start points....")
    evaluate_multiple_starts(
        hex_map, obstacle_map, goal_loc, hex_radius,
        hex_size, accel_max, decel_max, lat_accel_max,
        model, num_starts=5
    )

# %%
# ============================================================
# 8 ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # adjust path if needed
    # e.g., "r3h0.33.pkl", "r15h1.00.pkl", "r50h5.00.pkl"
    map_dir = "maps"
    hex_map = "r20h1.00.pkl"
    pkl_path = os.path.join(map_dir, hex_map)

    # Toggle this flag:
    #   True  -> use Optuna hyperparameter tuning
    #   False -> use baseline training (original method)
    use_optuna = False
    if use_optuna:
        model_path_fname = "learned_heuristic_opt.pt"
    else:
        model_path_fname = "learned_heuristic_gt_.pt"
    
    print(f"Using map {hex_map}")
    print(f"Using Optuna HyperTuning: {use_optuna}")
    run_pipeline(pkl_path, model_path=model_path_fname, use_optuna=use_optuna)

# %%
#  Map files 
# 'r3h0.33.pkl'
# 'r15h1.00.pkl'
# 'r20h1.00.pkl'
# 'r25h1.00.pkl'
# 'r30h5.00.pkl'
# 'r40h5.00.pkl'
# 'r50h5.00.pkl'