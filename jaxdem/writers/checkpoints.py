# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Orbax checkpoint writer and a loader.

- CheckpointWriter: saves checkpoints with preservation/decision policies
- CheckpointLoader: restores checkpoints (latest or specific step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, cast

import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import (
    preservation_policy as preservation_policy_lib,
)
from orbax.checkpoint.checkpoint_managers import (
    save_decision_policy as save_decision_policy_lib,
)

from ..state import State
from ..system import System


@dataclass(slots=True, weakref_slot=True)
class CheckpointWriter:
    """
    Thin wrapper around Orbax checkpoint saving.
    """

    directory: Path | str = Path("./checkpoints")
    """
    The base directory where checkpoints will be saved.
    """

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, everything is save.
    """

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    def __post_init__(self):
        self.directory = Path(self.directory).resolve()
        self.directory = cast(
            Path, ocp.test_utils.erase_and_create_empty(self.directory)
        )
        self.save_every = int(self.save_every)
        self.max_to_keep = (
            int(self.max_to_keep) if self.max_to_keep is not None else None
        )
        options = ocp.CheckpointManagerOptions(
            save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(
                self.save_every
            ),
            preservation_policy=preservation_policy_lib.LatestN(self.max_to_keep),
        )
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    def save(self, state: "State", system: "System") -> None:
        """
        Save a checkpoint for the provided state/system at a given step.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.
        """
        system_metadata = dict(
            dim=state.dim,
            integrator_type=system.integrator.type_name,
            collider_type=system.collider.type_name,
            domain_type=system.domain.type_name,
            force_model_type=system.force_model.type_name,
        )

        self.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                system=ocp.args.StandardSave(system),
                state_metadata=ocp.args.JsonSave(dict(shape=tuple(state.pos.shape))),
                system_metadata=ocp.args.JsonSave(system_metadata),
            ),
        )

    def block_until_ready(self):
        """
        Wait for the checkpointer to finish.
        """
        self.checkpointer.wait_until_finished()

    def close(self) -> None:
        """
        Wait for the checkpointer to finish and close it.
        """
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


@dataclass(slots=True)
class CheckpointLoader:
    """
    Thin wrapper around Orbax checkpoint restoring.

    Attributes
    ----------
    directory : Path
        Base directory where checkpoints are stored.
    checkpointer : ocp.CheckpointManager
        Underlying Orbax checkpoint manager.
    """

    directory: Path = Path("./checkpoints")
    """
    The base directory where checkpoints will be saved.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    def __post_init__(self):
        self.directory = Path(self.directory).resolve()
        options = ocp.CheckpointManagerOptions()
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    def load(
        self,
        step: Optional[int] = None,
    ) -> Tuple[State, System]:
        """
        Restore a checkpoint.

        Parameters
        ----------
        step : Optional[int]
            - If None, load the latest checkpoint.
            - Otherwise, load the specified step.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the restored `State` and `System`.
        """
        if step is None:
            step = self.checkpointer.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoints found in: {self.directory}")

        if step not in self.checkpointer.all_steps():
            raise FileNotFoundError(
                f"step={step} checkpoints not found in: {self.directory}"
            )

        metadata = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                state_metadata=ocp.args.JsonRestore(),
                system_metadata=ocp.args.JsonRestore(),
            ),
        )

        state_target = State.create(jnp.zeros(tuple(metadata.state_metadata["shape"])))
        system_target = System.create(**metadata.system_metadata)

        result = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state_target),
                system=ocp.args.StandardRestore(system_target),
            ),
        )

        return result.state, result.system

    def block_until_ready(self):
        self.checkpointer.wait_until_finished()

    def close(self) -> None:
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


__all__ = ["CheckpointWriter", "CheckpointLoader"]
