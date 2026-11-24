from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from hexpretraining import (
    PretrainingConfig,
    build_grid_definition,
    create_data_loaders,
    create_value_network,
    generate_relaxed_dataset,
    model_summary,
    pretrain_relaxed_value_network,
    seed_everything,
    split_relaxed_dataset_by_goal,
)


OUTPUT_DIRECTORY = Path("hunet_pretraining_outputs")
MODEL_FILENAME = "hunet_relaxed_pretrained.pt"
BENCHMARK_FILENAME = "hunet_relaxed_pretraining_benchmarks.xlsx"
CONFIG_FILENAME = "hunet_relaxed_pretraining_config.json"


def save_benchmarks(history, summary_rows, output_path):
    history_frame = pd.DataFrame(history)
    summary_frame = pd.DataFrame(summary_rows)

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            history_frame.to_excel(
                writer,
                sheet_name="epoch_history",
                index=False,
            )
            summary_frame.to_excel(
                writer,
                sheet_name="run_summary",
                index=False,
            )
        return output_path
    except Exception as error:
        csv_path = output_path.with_suffix(".csv")
        history_frame.to_csv(csv_path, index=False)
        summary_frame.to_csv(
            output_path.with_name(
                output_path.stem + "_summary.csv"
            ),
            index=False,
        )
        print("XLSX export was unavailable:", error)
        print("CSV benchmark files were saved instead.")
        return csv_path


def main():
    config = PretrainingConfig(
        samples_per_goal=1000,
        epochs=10,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(config.seed)
    print("Step completed: random seeds initialized.")

    (
        valid_locations,
        valid_location_set,
        validation_environment,
    ) = build_grid_definition(config)
    print("Step completed: physical grid constructed.")
    print("Valid locations:", len(valid_locations))

    dataset = generate_relaxed_dataset(
        config=config,
        valid_locations=valid_locations,
    )
    print("Step completed: relaxed pretraining dataset generated.")
    print("Total samples:", len(dataset))

    (
        train_dataset,
        validation_dataset,
        training_goals,
        validation_goals,
    ) = split_relaxed_dataset_by_goal(
        dataset=dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    print("Step completed: goal-held-out dataset split created.")
    print("Training goals:", len(training_goals))
    print("Validation goals:", len(validation_goals))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))

    train_loader, validation_loader = create_data_loaders(
        config=config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    print("Step completed: data loaders created.")

    value_network = create_value_network(
        config=config,
        device=device,
    )
    architecture = model_summary(value_network)

    print()
    print("Model architecture")
    print(architecture["architecture"])
    print("Parameter count:", architecture["parameter_count"])
    print(
        "Trainable parameter count:",
        architecture["trainable_parameter_count"],
    )
    print("Model device:", architecture["device"])
    print("Step completed: model initialized successfully.")

    print()
    print("Starting relaxed pretraining.")
    print("Epochs:", config.epochs)
    print("Learning rate:", config.learning_rate)
    print("Batch size:", config.batch_size)

    started_at = datetime.now()
    training_result = pretrain_relaxed_value_network(
        model=value_network,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_environment=validation_environment,
        config=config,
        valid_locations=valid_locations,
        valid_location_set=valid_location_set,
    )
    completed_at = datetime.now()
    elapsed_seconds = (completed_at - started_at).total_seconds()

    print("Step completed: pretraining finished successfully.")

    model_path = OUTPUT_DIRECTORY / MODEL_FILENAME
    torch.save(
        {
            "model_state_dict": value_network.state_dict(),
            "model_config": {
                "spatial_channels": config.spatial_channels,
                "state_features": config.state_features,
            },
            "pretraining_config": config.to_dict(),
            "best_epoch": training_result["best_epoch"],
            "best_validation_mse": training_result["best_validation_mse"],
            "final_validation_metrics": training_result[
                "final_validation_metrics"
            ],
        },
        model_path,
    )
    print("Step completed: trained model saved.")
    print("Model path:", model_path.resolve())

    config_path = OUTPUT_DIRECTORY / CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            config.to_dict(),
            config_file,
            indent=2,
        )
    print("Step completed: run configuration saved.")
    print("Configuration path:", config_path.resolve())

    final_metrics = training_result["final_validation_metrics"]
    summary_rows = [
        {"metric": "started_at", "value": started_at.isoformat()},
        {"metric": "completed_at", "value": completed_at.isoformat()},
        {"metric": "elapsed_seconds", "value": elapsed_seconds},
        {"metric": "device", "value": str(device)},
        {"metric": "parameter_count", "value": architecture["parameter_count"]},
        {
            "metric": "trainable_parameter_count",
            "value": architecture["trainable_parameter_count"],
        },
        {"metric": "total_samples", "value": len(dataset)},
        {"metric": "training_samples", "value": len(train_dataset)},
        {"metric": "validation_samples", "value": len(validation_dataset)},
        {"metric": "training_goals", "value": len(training_goals)},
        {"metric": "validation_goals", "value": len(validation_goals)},
        {"metric": "best_epoch", "value": training_result["best_epoch"]},
        {
            "metric": "best_validation_mse",
            "value": training_result["best_validation_mse"],
        },
        {"metric": "final_validation_rmse", "value": final_metrics["rmse"]},
        {"metric": "final_validation_mae", "value": final_metrics["mae"]},
        {
            "metric": "final_negative_fraction",
            "value": final_metrics["negative_fraction"],
        },
        {
            "metric": "validation_context_maps",
            "value": final_metrics["context_maps"],
        },
    ]

    benchmark_path = save_benchmarks(
        history=training_result["history"],
        summary_rows=summary_rows,
        output_path=OUTPUT_DIRECTORY / BENCHMARK_FILENAME,
    )
    print("Step completed: training benchmarks saved.")
    print("Benchmark path:", benchmark_path.resolve())

    print()
    print("All HUNet relaxed pretraining steps completed successfully.")


if __name__ == "__main__":
    main()
