"""Flow trainer: handles the training loop, early stopping, and scaling."""

import copy
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.flow import GlasflowConfig
from neural_priors_gym.flows.base import FlowBase
from neural_priors_gym.flows.glasflow import GlasflowNSF
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.training")

SCALER_FILE = "scaler.gz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowTrainer:
    """Trains a normalizing flow on tabular training data.

    Parameters
    ----------
    config:
        Top-level training configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.training_cfg = config.training
        self.output_dir = Path(config.output_dir)

    def build_flow(self, n_inputs: int) -> FlowBase:
        """Construct a flow from the config."""
        flow_config = self.config.flow
        if isinstance(flow_config, GlasflowConfig):
            return GlasflowNSF.from_config(flow_config, n_inputs)
        raise ValueError(f"Unknown flow config type: {type(flow_config)}")

    def train(
        self,
        data: dict[str, np.ndarray],
        parameter_names: list[str],
        output_dir: Optional[Path] = None,
    ) -> tuple[FlowBase, list[float], list[float], Optional[MinMaxScaler]]:
        """Train a flow on the provided data.

        Parameters
        ----------
        data:
            Dictionary mapping parameter names to arrays of shape (n_samples,).
        parameter_names:
            Ordered list of keys to extract from data (defines column order).
        output_dir:
            Directory to save the scaler. Defaults to self.output_dir.

        Returns
        -------
        flow:
            Trained flow (best checkpoint by validation loss).
        train_losses:
            Mean NLL per epoch for the training set.
        val_losses:
            Mean NLL per epoch for the validation set.
        scaler:
            Fitted MinMaxScaler if scale_input is True, else None.
        """
        cfg = self.training_cfg
        save_dir = output_dir or self.output_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Apply log transform to lambda columns in-memory before training.
        # The saved training_data.npz retains raw values; this transform is
        # only visible to the flow and scaler.
        if self.config.lambdas.log_lambda:
            data = dict(data)
            for name in self.config.lambdas.parameter_names:
                if name in data:
                    data[name] = np.log(data[name])
            logger.info(
                f"Applied log transform to lambda parameters: "
                f"{self.config.lambdas.parameter_names}"
            )

        x_full = np.column_stack([data[name] for name in parameter_names])
        n_inputs = x_full.shape[1]

        x_train, x_val = train_test_split(
            x_full,
            test_size=cfg.validation_split,
            random_state=42,
        )

        scaler: Optional[MinMaxScaler] = None
        if cfg.scale_input:
            scaler = MinMaxScaler()
            x_combined = np.vstack([x_train, x_val])
            scaler.fit(x_combined)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            joblib.dump(scaler, save_dir / SCALER_FILE)
            logger.info(f"MinMaxScaler saved to {save_dir / SCALER_FILE}")

        flow = self.build_flow(n_inputs)
        assert isinstance(flow, GlasflowNSF)
        flow.set_optimizer(cfg.learning_rate)

        train_tensor = torch.tensor(x_train, dtype=torch.float32)
        val_tensor = torch.tensor(x_val, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor), batch_size=cfg.batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict = {}
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(cfg.num_epochs):
            train_loss = flow.train_epoch(train_loader)
            val_loss = flow.eval_epoch(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(flow._flow.state_dict())  # type: ignore[attr-defined]
            else:
                patience_counter += 1

            if (epoch + 1) % cfg.log_every_n_epochs == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{cfg.num_epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}"
                )

            if patience_counter >= cfg.max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        flow._flow.load_state_dict(best_state)  # type: ignore[attr-defined]
        logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")
        return flow, train_losses, val_losses, scaler
