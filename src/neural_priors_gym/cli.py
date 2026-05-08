"""Command-line entry point for neural_priors_gym_train."""

import sys
from pathlib import Path

import numpy as np

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.data.generator import TrainingDataGenerator
from neural_priors_gym.evaluation.metrics import compute_jsd
from neural_priors_gym.logging_config import get_logger
from neural_priors_gym.plotting.plots import (
    plot_corner,
    plot_losses,
    plot_training_data,
)
from neural_priors_gym.training.trainer import FlowTrainer

logger = get_logger("neural_priors_gym.cli")


def main(config: TrainingConfig) -> None:
    """Run the full training pipeline for a neural prior.

    Steps:
    1. Generate training data from EOS samples and mass distribution.
    2. Save training data as an npz file.
    3. Train the normalizing flow.
    4. Save the flow and MinMaxScaler.
    5. Generate flow samples for evaluation.
    6. Compute JSD between training data and flow samples.
    7. Save loss plot and corner plot.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(config.masses.parameter_names) + list(
        config.lambdas.parameter_names
    )
    logger.info(f"Training neural prior for parameters: {parameter_names}")

    # Generate training data
    data_gen = TrainingDataGenerator.from_config(config)
    training_data = data_gen.generate()

    training_data_path = output_dir / "training_data.npz"
    np.savez(training_data_path, allow_pickle=True, **training_data)
    logger.info(f"Training data saved to {training_data_path}")

    # Plot training data samples so users can inspect the generated prior
    x_data = np.column_stack([training_data[name] for name in parameter_names])
    plot_training_data(x_data, parameter_names, output_dir)

    if config.generate_only:
        logger.info(
            "generate_only=True: skipping training. "
            f"Inspect training_data_corner.pdf in {output_dir}, "
            "then rerun with generate_only: false to train."
        )
        return

    # Train flow
    logger.info(f"Will report every {config.training.log_every_n_epochs} epochs")
    trainer = FlowTrainer(config)
    flow, train_losses, val_losses, scaler = trainer.train(
        training_data, parameter_names, output_dir=output_dir
    )

    # Save flow
    model_dir = output_dir / "model"
    flow.save(model_dir)

    # Copy scaler to model dir so the model directory is self-contained
    if scaler is not None:
        import shutil

        scaler_src = output_dir / "scaler.gz"
        if scaler_src.exists():
            shutil.copy(scaler_src, model_dir / "scaler.gz")

    # Plots and evaluation
    plot_losses(train_losses, val_losses, output_dir)

    n_eval = min(10_000, len(x_data))
    flow_samples = flow.sample(n_eval)

    if scaler is not None:
        flow_samples_original = scaler.inverse_transform(flow_samples)
    else:
        flow_samples_original = flow_samples

    jsd_results = compute_jsd(x_data[:n_eval], flow_samples_original)
    jsd_per_dim = [
        jsd_results[f"dim_{i}_millibits"] for i in range(len(parameter_names))
    ]
    logger.info(f"Mean JSD = {jsd_results['mean_millibits']:.2f} millibits")
    for name, jsd in zip(parameter_names, jsd_per_dim):
        logger.info(f"  {name}: {jsd:.2f} millibits")

    plot_corner(
        x_data[:n_eval], flow_samples_original, jsd_per_dim, parameter_names, output_dir
    )

    logger.info(f"Training complete. All outputs saved to {output_dir}")


def cli_entry_point() -> None:
    """Console script entry point: neural_priors_gym_train config.yaml"""
    if len(sys.argv) != 2:
        print("Usage: neural_priors_gym_train <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)


if __name__ == "__main__":
    cli_entry_point()
