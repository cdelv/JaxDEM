# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Orbax checkpoint writer and a loader.

- CheckpointWriter: saves checkpoints with preservation/decision policies
- CheckpointLoader: restores checkpoints (latest or specific step)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import logging
import warnings

from dataclasses import dataclass, field, fields as _dc_fields
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast, TYPE_CHECKING
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
from ..materials import MaterialTable
from ..material_matchmakers import MaterialMatchmaker
from ..forces import ForceModel, ForceRouter, LawCombiner
from ..forces.force_manager import default_energy_func
from ..colliders.neighbor_list import NeighborList
from ..colliders.cell_list import StaticCellList, DynamicCellList
from ..utils import encode_callable, decode_callable
import contextlib

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


def _bonded_force_manager_kw(system: System) -> dict[str, Any] | None:
    bonded_model = system.bonded_force_model
    if bonded_model is None:
        return None
    return {
        f.name: _to_jsonable(getattr(bonded_model, f.name))
        for f in _dc_fields(bonded_model)
    }


def _serialize_force_model(model: ForceModel) -> dict[str, Any]:
    """Serialize a force model to a JSON-compatible dict."""
    result: dict[str, Any] = {"type": model.type_name}
    if isinstance(model, ForceRouter):
        result["table"] = [
            [_serialize_force_model(law) for law in row] for row in model.table
        ]
    elif isinstance(model, LawCombiner) and model.laws:
        result["laws"] = [_serialize_force_model(law) for law in model.laws]
    return result


def _deserialize_force_model(data: dict[str, Any]) -> ForceModel:
    """Reconstruct a force model from serialized metadata."""
    type_name = data["type"]
    if "table" in data:
        table = tuple(
            tuple(_deserialize_force_model(law) for law in row) for row in data["table"]
        )
        return ForceRouter(table=table)
    kw: dict[str, Any] = {}
    if "laws" in data:
        kw["laws"] = tuple(_deserialize_force_model(law) for law in data["laws"])
    return ForceModel.create(type_name, **kw)


def _serialize_force_functions(system: System) -> list[dict[str, Any]] | None:
    """Serialize user-supplied custom force functions to JSON.

    Bonded-model entries (appended at the end by ``System.create``) are
    excluded because they are reconstructed from ``bonded_force_model_type``
    during load.
    """
    fm = system.force_manager
    if not fm.force_functions:
        return None

    n_total = len(fm.force_functions)
    n_bonded = 1 if system.bonded_force_model is not None else 0
    n_user = n_total - n_bonded

    if n_user <= 0:
        return None

    entries: list[dict[str, Any]] = []
    for i in range(n_user):
        force_fn = fm.force_functions[i]
        energy_fn = fm.energy_functions[i]
        is_default_energy = energy_fn is default_energy_func

        fns_to_check = (force_fn,) if is_default_energy else (force_fn, energy_fn)
        for fn in fns_to_check:
            if fn is not None:
                mod = getattr(fn, "__module__", None)
                if mod == "__main__":
                    warnings.warn(
                        f"Force function '{fn.__name__}' is defined in __main__. "
                        "It will not be restorable from a different script. "
                        "Define it in an importable module instead.",
                        stacklevel=3,
                    )

        entry: dict[str, Any] = {
            "force": encode_callable(force_fn),
            "energy": (
                None
                if is_default_energy or energy_fn is None
                else encode_callable(energy_fn)
            ),
            "is_com": bool(fm.is_com_force[i]),
        }
        entries.append(entry)
    return entries


_log = logging.getLogger(__name__)


def _deserialize_force_functions(
    data: list[dict[str, Any]],
) -> list[tuple[Any, ...]]:
    """Reconstruct custom force function entries from serialized metadata.

    Returns a list of ``(force_fn, energy_fn_or_None, is_com)`` tuples
    suitable for passing as ``force_manager_kw["force_functions"]``.

    If a callable cannot be resolved (e.g. it was defined in ``__main__``
    of a different script), the entry is skipped and a warning is logged.
    """
    entries: list[tuple[Any, ...]] = []
    for item in data:
        force_path = item["force"]
        try:
            force_fn = decode_callable(force_path)
        except (ImportError, AttributeError, ValueError) as exc:
            _log.warning(
                "Skipping force function '%s': %s. "
                "Make sure the function is defined in an importable module.",
                force_path,
                exc,
            )
            warnings.warn(
                f"Could not restore force function '{force_path}': {exc}. "
                "The loaded system will not include this custom force. "
                "Define force functions in an importable module to enable "
                "cross-script checkpoint portability.",
                stacklevel=3,
            )
            continue

        energy_path = item.get("energy")
        energy_fn = None
        if energy_path:
            try:
                energy_fn = decode_callable(energy_path)
            except (ImportError, AttributeError, ValueError) as exc:
                _log.warning(
                    "Could not restore energy function '%s': %s. "
                    "Using default (zero) energy for this force.",
                    energy_path,
                    exc,
                )

        is_com = bool(item.get("is_com", False))
        entries.append((force_fn, energy_fn, is_com))
    return entries


def _serialize_collider_kw(system: System) -> dict[str, Any] | None:
    """Extract collider construction params needed for checkpoint restore."""
    collider = system.collider
    if isinstance(collider, NeighborList):
        sub_type = collider.cell_list.type_name
        return {
            "cutoff": float(collider.cutoff),
            "skin": float(collider.skin),
            "max_neighbors": int(collider.max_neighbors),
            "secondary_collider_type": sub_type,
        }
    if isinstance(collider, (StaticCellList, DynamicCellList)):
        meta: dict[str, Any] = {}
        if isinstance(collider, StaticCellList):
            meta["max_occupancy"] = int(collider.max_occupancy)
        return meta
    return None


@dataclass
class CheckpointWriter:
    """Thin wrapper around Orbax checkpoint saving.

    Notes
    -----
    Custom force functions passed via ``force_manager_kw`` are serialized
    by their fully-qualified module path (e.g. ``mypackage.forces.trap``).
    Functions defined in the top-level script (``__main__``) **cannot** be
    restored from a different script.  A warning is emitted at save time if
    any force function lives in ``__main__``.  To ensure portability, define
    force functions in an importable module.

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
        """Save a checkpoint for the provided state/system at a given step.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.

        """
        system_metadata = {
            "state_shape": tuple(state.pos.shape),
            "linear_integrator_type": system.linear_integrator.type_name,
            "rotation_integrator_type": system.rotation_integrator.type_name,
            "collider_type": system.collider.type_name,
            "domain_type": system.domain.type_name,
            "force_model_type": system.force_model.type_name,
            "bonded_force_model_type": (
                None
                if system.bonded_force_model is None
                else system.bonded_force_model.type_name
            ),
            "bonded_force_manager_kw": _bonded_force_manager_kw(system),
            "mat_table_metadata": {
                "num_materials": len(system.mat_table),
                "prop_keys": list(system.mat_table.props.keys()),
                "pair_keys": list(system.mat_table.pair.keys()),
                "matcher_type": system.mat_table.matcher.type_name,
            },
            "force_model_metadata": _serialize_force_model(system.force_model),
            "force_function_metadata": _serialize_force_functions(system),
            "collider_kw_metadata": _serialize_collider_kw(system),
        }

        self.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                system=ocp.args.StandardSave(system),
                state_metadata=ocp.args.JsonSave({"shape": tuple(state.pos.shape)}),
                system_metadata=ocp.args.JsonSave(system_metadata),
            ),
        )

    @partial(jax.named_call, name="CheckpointWriter.block_until_ready")
    def block_until_ready(self) -> None:
        """Wait for the checkpointer to finish."""
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="CheckpointWriter.close")
    def close(self) -> None:
        """Wait for the checkpointer to finish and close it."""
        try:
            self.checkpointer.wait_until_finished()
        finally:
            self.checkpointer.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

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
    """Thin wrapper around Orbax checkpoint restoring for jaxdem.state and jaxdem.system."""

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
        step: int | None = None,
    ) -> tuple[State, System]:
        """Restore a checkpoint.

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

        mat_table_meta = system_metadata.pop("mat_table_metadata", None)
        if mat_table_meta is not None:
            M = mat_table_meta["num_materials"]
            matcher = MaterialMatchmaker.create(mat_table_meta["matcher_type"])
            props = {k: jnp.zeros(M) for k in mat_table_meta["prop_keys"]}
            pair = {k: jnp.zeros((M, M)) for k in mat_table_meta["pair_keys"]}
            system_metadata["mat_table"] = MaterialTable(
                props=props, pair=pair, matcher=matcher
            )

        force_model_meta = system_metadata.pop("force_model_metadata", None)
        if force_model_meta is not None:
            force_model = _deserialize_force_model(force_model_meta)
            if isinstance(force_model, ForceRouter):
                system_metadata["force_model_kw"] = {"table": force_model.table}
            elif isinstance(force_model, LawCombiner) and force_model.laws:
                system_metadata["force_model_kw"] = {"laws": force_model.laws}

        force_fn_meta = system_metadata.pop("force_function_metadata", None)
        if force_fn_meta is not None:
            user_fns = _deserialize_force_functions(force_fn_meta)
            fm_kw = system_metadata.get("force_manager_kw") or {}
            fm_kw["force_functions"] = user_fns
            system_metadata["force_manager_kw"] = fm_kw

        collider_kw_meta = system_metadata.pop("collider_kw_metadata", None)
        if collider_kw_meta is not None:
            collider_kw = dict(collider_kw_meta, state=state_target)
            if "secondary_collider_type" in collider_kw:
                collider_kw.setdefault("secondary_collider_kw", {"state": state_target})
                if "state" not in collider_kw.get("secondary_collider_kw", {}):
                    collider_kw["secondary_collider_kw"]["state"] = state_target
            system_metadata["collider_kw"] = collider_kw

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
    def latest_step(self) -> int | None:
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
        with contextlib.suppress(Exception):
            self.close()

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
    """Thin wrapper around Orbax checkpoint saving for jaxdem.rl.models.Model."""

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
        """Save model at a step: stores model_state and JSON metadata.
        Assumes model.metadata includes JSON-serializable fields. We add model_type.
        """
        from flax import nnx

        model_metadata = model.metadata
        model_metadata["model_type"] = model.type_name

        _graphdef, state = nnx.split(model)
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
        with contextlib.suppress(Exception):
            self.close()


@dataclass
class CheckpointModelLoader:
    """Thin wrapper around Orbax checkpoint restoring for jaxdem.rl.models.Model."""

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
        """Load a model from a given step (or the latest if None)."""
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
        model_metadata.get("reset_shape", (1,))

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
    def latest_step(self) -> int | None:
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
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "CheckpointLoader",
    "CheckpointModelLoader",
    "CheckpointModelWriter",
    "CheckpointWriter",
]
