# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Orbax checkpoint writer and a loader.

- CheckpointWriter: saves checkpoints with preservation/decision policies
- CheckpointLoader: restores checkpoints (latest or specific step)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field, fields as _dc_fields, is_dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Tuple, cast, TYPE_CHECKING
from functools import partial

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import (
    preservation_policy as preservation_policy_lib,
)
from orbax.checkpoint.checkpoint_managers import (
    save_decision_policy as save_decision_policy_lib,
)

from ..state import State
from ..system import System
from ..utils import decode_callable

if TYPE_CHECKING:
    from ..rl.models import Model


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable for checkpoint metadata."
    )


def _bonded_force_manager_kw(system: System) -> Optional[dict[str, Any]]:
    bonded_model = system.bonded_force_model
    if bonded_model is None:
        return None
    if not is_dataclass(bonded_model):
        return None
    return {
        f.name: _to_jsonable(getattr(bonded_model, f.name)) for f in _dc_fields(bonded_model)
    }


@dataclass
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
    Keep the last max_to_keep checkpoints. If None, everything is saved.
    """

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    def __post_init__(self) -> None:
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

    @partial(jax.named_call, name="CheckpointWriter.save")
    def save(self, state: State, system: System) -> None:
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
            state_shape=tuple(state.pos.shape),
            linear_integrator_type=system.linear_integrator.type_name,
            rotation_integrator_type=system.rotation_integrator.type_name,
            collider_type=system.collider.type_name,
            domain_type=system.domain.type_name,
            force_model_type=system.force_model.type_name,
            bonded_force_model_type=(
                None
                if system.bonded_force_model is None
                else system.bonded_force_model.type_name
            ),
            bonded_force_manager_kw=_bonded_force_manager_kw(system),
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

    @partial(jax.named_call, name="CheckpointWriter.block_until_ready")
    def block_until_ready(self) -> None:
        """
        Wait for the checkpointer to finish.
        """
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="CheckpointWriter.close")
    def close(self) -> None:
        """
        Wait for the checkpointer to finish and close it.
        """
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


@dataclass
class CheckpointLoader:
    """
    Thin wrapper around Orbax checkpoint restoring for jaxdem.state and jaxdem.system.
    """

    directory: Path = Path("./checkpoints")
    """
    The base directory where checkpoints will be saved.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).resolve()
        options = ocp.CheckpointManagerOptions()
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    @partial(jax.named_call, name="CheckpointLoader.load")
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
                f"step={step} checkpoints not found in: {self.directory}. Available steps: {self.checkpointer.all_steps()}"
            )

        metadata = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                state_metadata=ocp.args.JsonRestore(),
                system_metadata=ocp.args.JsonRestore(),
            ),
        )

        state_shape = tuple(metadata.state_metadata["shape"])
        state_target = State.create(jnp.zeros(state_shape))
        system_metadata = dict(metadata.system_metadata)
        system_metadata["state_shape"] = tuple(
            system_metadata.get("state_shape", state_shape)
        )
        system_metadata.pop("dim", None)  # Backward compatibility with legacy metadata.
        system_metadata.setdefault("bonded_force_model_type", None)
        system_metadata.setdefault("bonded_force_manager_kw", None)

        try:
            system_target = System.create(**system_metadata)
        except Exception:
            # Legacy fallback for old checkpoints that do not carry enough
            # metadata to deterministically rebuild system targets.
            result = self.checkpointer.restore(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(),
                    system=ocp.args.StandardRestore(),
                ),
            )
            state_obj = getattr(result, "state", None)
            system_obj = getattr(result, "system", None)
            if isinstance(state_obj, State) and isinstance(system_obj, System):
                return state_obj, system_obj
            raise

        result = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state_target),
                system=ocp.args.StandardRestore(system_target),
            ),
        )

        return result.state, result.system

    @partial(jax.named_call, name="CheckpointLoader.latest_step")
    def latest_step(self) -> Optional[int]:
        return self.checkpointer.latest_step()

    @partial(jax.named_call, name="CheckpointLoader.block_until_ready")
    def block_until_ready(self) -> None:
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="CheckpointLoader.close")
    def close(self) -> None:
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


@dataclass
class CheckpointModelWriter:
    """
    Thin wrapper around Orbax checkpoint saving for jaxdem.rl.models.Model.
    """

    directory: Path | str = Path("./checkpoints")
    """
    The base directory where checkpoints will be saved.
    """

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, everything is saved.
    """

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    clean: bool = True
    """
    Whether to clean the directory.
    """

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.clean:
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

    @partial(jax.named_call, name="CheckpointModelWriter.save")
    def save(self, model: Model, step: int) -> None:
        """
        Save model at a step: stores model_state and JSON metadata.
        Assumes model.metadata includes JSON-serializable fields. We add model_type.
        """
        from flax import nnx

        model_metadata = model.metadata
        model_metadata["model_type"] = model.type_name

        graphdef, state = nnx.split(model)
        self.checkpointer.save(
            int(step),
            args=ocp.args.Composite(
                model_state=ocp.args.StandardSave(state),
                model_metadata=ocp.args.JsonSave(model_metadata),
            ),
        )

    @partial(jax.named_call, name="CheckpointModelWriter.block_until_ready")
    def block_until_ready(self) -> None:
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="CheckpointModelWriter.close")
    def close(self) -> None:
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class CheckpointModelLoader:
    """
    Thin wrapper around Orbax checkpoint restoring for jaxdem.rl.models.Model.
    """

    directory: Path = Path("./checkpoints")
    """
    The base directory where checkpoints will be saved.
    """

    checkpointer: ocp.CheckpointManager = field(init=False)
    """
    Orbax checkpoint manager for saving the checkpoints.
    """

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).resolve()
        options = ocp.CheckpointManagerOptions()
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    @partial(jax.named_call, name="CheckpointModelLoader.load")
    def load(self, step: int | None = None) -> Model:
        """
        Load a model from a given step (or the latest if None).
        """
        from flax import nnx
        from ..rl.models import Model
        from ..rl.actionSpaces import ActionSpace

        if step is None:
            step = self.checkpointer.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoints found in: {self.directory}")

        if step not in self.checkpointer.all_steps():
            raise FileNotFoundError(
                f"step={step} checkpoints not found in: {self.directory}. Available steps: {self.checkpointer.all_steps()}"
            )

        model_metadata = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                model_metadata=ocp.args.JsonRestore(),
            ),
        )
        model_metadata = model_metadata.model_metadata

        action_space = ActionSpace.create(
            model_metadata["action_space_type"], **model_metadata["action_space_kws"]
        )

        used_keys = [
            "action_space_type",
            "action_space_kws",
            "reset_shape",
            "activation",
            "model_type",
        ]

        rngs = nnx.Rngs(0)

        activation = decode_callable(model_metadata["activation"])
        model_type = model_metadata["model_type"]
        reset_shape = model_metadata.get("reset_shape", (1,))

        model_metadata = {
            key: value for key, value in model_metadata.items() if key not in used_keys
        }

        if "cell_type" in model_metadata:
            model_metadata["cell_type"] = decode_callable(model_metadata["cell_type"])

        model = Model.create(
            model_type,
            **model_metadata,
            key=rngs,
            action_space=action_space,
            activation=activation,
        )

        graphdef, state = nnx.split(model)

        result = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(state),
            ),
        )
        state = result.model_state

        return nnx.merge(graphdef, state)

    @partial(jax.named_call, name="CheckpointModelLoader.latest_step")
    def latest_step(self) -> Optional[int]:
        return self.checkpointer.latest_step()

    @partial(jax.named_call, name="CheckpointModelLoader.block_until_ready")
    def block_until_ready(self) -> None:
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="CheckpointModelLoader.close")
    def close(self) -> None:
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CheckpointWriter",
    "CheckpointLoader",
    "CheckpointModelWriter",
    "CheckpointModelLoader",
]
