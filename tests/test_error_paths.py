"""Tests for ValueError branches in dispatch functions."""

import pytest
from unittest.mock import MagicMock

from neural_priors_gym.data.generator import build_mass_generator
from neural_priors_gym.training.trainer import FlowTrainer


def test_build_mass_generator_unknown_type_raises() -> None:
    """build_mass_generator raises ValueError for an unrecognised mass config type."""
    config = MagicMock()
    config.masses = object()  # not any of the known MassConfig types
    config.source_type = "bns"

    with pytest.raises(ValueError, match="Unknown mass config type"):
        build_mass_generator(config)


def test_flow_trainer_build_flow_unknown_type_raises(
    small_config_yaml, tmp_path
) -> None:
    """FlowTrainer.build_flow raises ValueError for an unrecognised flow config type."""
    from neural_priors_gym.config.parser import load_config

    config = load_config(small_config_yaml)
    trainer = FlowTrainer(config)

    # Swap the flow config for an unrecognised type
    trainer.config = MagicMock()
    trainer.config.flow = object()

    with pytest.raises(ValueError, match="Unknown flow config type"):
        trainer.build_flow(n_inputs=4)
