# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Orbax checkpoint writer and a loader.

- CheckpointWriter: saves checkpoints with preservation/decision policies
- CheckpointLoader: restores checkpoints (latest or specific step)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import contextlib

import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from orbax.checkpoint.checkpoint_managers import (  # type: ignore[import-untyped]
    preservation_policy as preservation_policy_lib,
)
from orbax.checkpoint.checkpoint_managers import (
    save_decision_policy as save_decision_policy_lib,
)

from ..forces import ForceRouter, LawCombiner
from ..material_matchmakers import MaterialMatchmaker
from ..materials import MaterialTable
from ..state import State
from ..system import System
from ..utils import decode_callable

if TYPE_CHECKING:
    from ..rl.models import Model


_log = logging.getLogger(__name__)


def _deserialize_force_functions(
    data: list[dict[str, Any]],
    strict: bool = True,
) -> list[tuple[Any, ...]]:
    """Reconstruct custom force function entries from serialized metadata.

    Returns a list of ``(force_fn, energy_fn_or_None, is_com)`` tuples
    suitable for passing as ``force_manager_kw["force_functions"]``.

    If a callable cannot be resolved (e.g. it was defined in ``__main__``
    of a different script), a ``RuntimeError`` is raised when ``strict``
    is ``True`` (the default); otherwise the entry is skipped with a
    warning, which silently changes the restored physics.
    """
    entries: list[tuple[Any, ...]] = []
    for item in data:
        force_path = item["force"]
        try:
            force_fn = decode_callable(force_path)
        except (ImportError, AttributeError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"Could not restore force function '{force_path}': {exc}. "
                    "The checkpoint cannot be loaded with the physics it was "
                    "saved with. Define force functions in an importable "
                    "module, or pass `strict=False` to load without this "
                    "custom force."
                ) from exc
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


@dataclass
class BaseCheckpointManager:
    """Base class providing context management and boilerplate for Orbax checkpointers."""

    directory: Path | str = Path("./checkpoints")
    checkpointer: ocp.CheckpointManager = field(init=False)

    def _prepare_directory(self, clean: bool = False) -> None:
        """Resolve ``self.directory`` and create it if needed.

        If ``clean`` is True, the directory is erased and recreated.
        If ``clean`` is False (the default), existing contents are preserved.
        """
        self.directory = Path(self.directory).resolve()
        if clean:
            self.directory = cast(
                Path, ocp.test_utils.erase_and_create_empty(self.directory)
            )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _init_writer_manager(self) -> None:
        """Coerce ``save_every``/``max_to_keep`` and build the checkpoint manager.

        Shared by :class:`CheckpointWriter` and :class:`CheckpointModelWriter`,
        which both declare the ``save_every`` and ``max_to_keep`` dataclass
        fields this method reads.
        """
        save_every = int(getattr(self, "save_every"))
        max_to_keep_attr = getattr(self, "max_to_keep")
        max_to_keep = int(max_to_keep_attr) if max_to_keep_attr is not None else None
        setattr(self, "save_every", save_every)
        setattr(self, "max_to_keep", max_to_keep)
        options = ocp.CheckpointManagerOptions(
            save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(
                save_every
            ),
            preservation_policy=preservation_policy_lib.LatestN(max_to_keep),
        )
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    def _init_loader_manager(self) -> None:
        """Build a read-only checkpoint manager. Shared by the loaders."""
        self.directory = Path(self.directory).resolve()
        options = ocp.CheckpointManagerOptions()
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    @partial(jax.named_call, name="block_until_ready")
    def block_until_ready(self) -> None:
        self.checkpointer.wait_until_finished()

    @partial(jax.named_call, name="close")
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
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


@dataclass
class CheckpointWriter(BaseCheckpointManager):
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

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, everything is saved.
    """

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    clean: bool = False
    """
    If True, the target directory is erased and recreated on construction.
    If False (the default), existing checkpoints in the directory are kept,
    so a resumed run does not destroy earlier checkpoints.
    """

    def __post_init__(self) -> None:
        self._prepare_directory(clean=self.clean)
        self._init_writer_manager()

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
        system_metadata = system.metadata
        system_metadata["state_shape"] = tuple(state.pos.shape)
        system_metadata["bond_id_shape"] = tuple(state.bond_id.shape)

        if system.minimizer is not None:
            constructor_fn = getattr(system.minimizer, "_constructor", None)
            if constructor_fn is not None:
                mod = getattr(constructor_fn, "__module__", None)
                if mod == "__main__":
                    warnings.warn(
                        f"Minimizer constructor '{constructor_fn.__name__}' is defined in __main__. "
                        "It will not be restorable from a different script. "
                        "Define it in an importable module instead.",
                        stacklevel=2,
                    )

        if system.target_fn is not None:
            mod = getattr(system.target_fn, "__module__", None)
            if mod == "__main__":
                warnings.warn(
                    f"Target function '{system.target_fn.__name__}' is defined in __main__. "
                    "It will not be restorable from a different script. "
                    "Define it in an importable module instead.",
                    stacklevel=2,
                )

        self.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                system=ocp.args.StandardSave(system),
                state_metadata=ocp.args.JsonSave({"shape": state.shape}),
                system_metadata=ocp.args.JsonSave(system_metadata),
            ),
        )


@dataclass
class CheckpointLoader(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint restoring for jaxdem.state and jaxdem.system."""

    def __post_init__(self) -> None:
        self._init_loader_manager()

    @partial(jax.named_call, name="CheckpointLoader.load")
    def load(
        self,
        step: int | None = None,
        *,
        strict: bool = True,
    ) -> tuple[State, System]:
        """Restore a checkpoint.

        Parameters
        ----------
        step : Optional[int]
            - If None, load the latest checkpoint.
            - Otherwise, load the specified step.
        strict : bool, optional
            If ``True`` (default), fail with a ``RuntimeError`` when a custom
            force function recorded in the checkpoint cannot be re-imported,
            instead of silently loading the system without that physics. Pass
            ``False`` to skip unloadable force functions with a warning.

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

        system_metadata = dict(metadata.system_metadata)
        state_shape = tuple(metadata.state_metadata["shape"])
        system_metadata["state_shape"] = tuple(
            system_metadata.get("state_shape", state_shape)
        )
        bond_id_shape = system_metadata.pop("bond_id_shape", None)
        if bond_id_shape is not None:
            dummy_bond_id = jnp.full(tuple(bond_id_shape), -1, dtype=int)
            state_target = State.create(jnp.zeros(state_shape), bond_id=dummy_bond_id)
        else:
            state_target = State.create(jnp.zeros(state_shape))

        # Restore state first to get correct radii and properties before building the system
        try:
            res_state = self.checkpointer.restore(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state_target),
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore state from checkpoint step {step} in "
                f"{self.directory}. The checkpoint may be corrupt or "
                "incompatible with the current State schema."
            ) from exc
        state_target = res_state.state
        system_metadata.setdefault("bonded_force_model_type", None)
        system_metadata.setdefault("bonded_force_model_kw", None)
        minimizer_meta = system_metadata.pop("minimizer", None)
        if minimizer_meta is not None:
            constructor_path = minimizer_meta.get("constructor")
            kw = minimizer_meta.get("kw", {})
            if constructor_path:
                system_metadata["minimizer"] = decode_callable(constructor_path)
                system_metadata["minimizer_kw"] = kw

        target_fn_path = system_metadata.pop("target_fn", None)
        if target_fn_path is not None:
            system_metadata["target_fn"] = decode_callable(target_fn_path)

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
            from ..factory import Factory

            force_model = Factory._deserialize_component(force_model_meta)
            if isinstance(force_model, ForceRouter):
                system_metadata["force_model_kw"] = {"table": force_model.table}
            elif isinstance(force_model, LawCombiner) and force_model.laws:
                system_metadata["force_model_kw"] = {"laws": force_model.laws}

        force_fn_meta = system_metadata.pop("force_function_metadata", None)
        if force_fn_meta is not None:
            user_fns = _deserialize_force_functions(force_fn_meta, strict=strict)
            fm_kw = system_metadata.get("force_manager_kw") or {}
            fm_kw["force_functions"] = user_fns
            system_metadata["force_manager_kw"] = fm_kw

        collider_kw_meta = system_metadata.pop("collider_kw_metadata", None)
        if collider_kw_meta is not None:

            def _accepts_state(collider_type: Any) -> bool:
                # Only inject the rebuilt state into colliders whose Create
                # takes one (CellList, MultiCellList, NeighborList, ...);
                # others (naive, "") would warn about the dropped keyword.
                from inspect import signature

                from ..colliders import Collider
                from ..factory import _normalize_key

                if not isinstance(collider_type, str):
                    return False
                sub_cls = Collider._registry.get(_normalize_key(collider_type))
                create_fn = getattr(sub_cls, "Create", None)
                return (
                    create_fn is not None and "state" in signature(create_fn).parameters
                )

            collider_kw = dict(collider_kw_meta)
            if _accepts_state(system_metadata.get("collider_type")):
                collider_kw["state"] = state_target
            if "secondary_collider" in collider_kw:
                sub_val = collider_kw["secondary_collider"]
                if (
                    isinstance(sub_val, dict)
                    and "kw" in sub_val
                    and _accepts_state(sub_val.get("type"))
                ):
                    sub_val["kw"]["state"] = state_target
            elif "secondary_collider_type" in collider_kw and _accepts_state(
                collider_kw["secondary_collider_type"]
            ):
                sub_kw = dict(collider_kw.get("secondary_collider_kw") or {})
                sub_kw["state"] = state_target
                collider_kw["secondary_collider_kw"] = sub_kw
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


@dataclass
class CheckpointModelWriter(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint saving for jaxdem.rl.models.Model."""

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, everything is saved.
    """

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    clean: bool = False
    """
    If True, the target directory is erased and recreated on construction.
    If False (the default), existing checkpoints in the directory are kept,
    so a resumed run does not destroy earlier checkpoints.
    """

    def __post_init__(self) -> None:
        self._prepare_directory(clean=self.clean)
        self._init_writer_manager()

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


@dataclass
class CheckpointModelLoader(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint restoring for jaxdem.rl.models.Model."""

    def __post_init__(self) -> None:
        self._init_loader_manager()

    @partial(jax.named_call, name="CheckpointModelLoader.load")
    def load(self, step: int | None = None) -> Model:
        """Load a model from a given step (or the latest if None)."""
        from flax import nnx

        from ..rl.action_spaces import ActionSpace
        from ..rl.models import Model

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

        if "action_space_type" in model_metadata:
            action_space = ActionSpace.create(
                model_metadata["action_space_type"],
                **model_metadata["action_space_kws"],
            )
            activation = decode_callable(model_metadata["activation"])
            model_type = model_metadata["model_type"]
            used_keys = [
                "action_space_type",
                "action_space_kws",
                "reset_shape",
                "activation",
                "model_type",
            ]
            model_kw = {
                key: value
                for key, value in model_metadata.items()
                if key not in used_keys
            }
            if "cell_type" in model_kw:
                model_kw["cell_type"] = decode_callable(model_kw["cell_type"])
            model_kw["action_space"] = action_space
            model_kw["activation"] = activation
        else:
            model_kw = dict(model_metadata)
            model_type = model_kw.pop("model_type")
            model_kw.pop("reset_shape", None)

        rngs = nnx.Rngs(0)
        model = Model.create(
            model_type,
            **model_kw,
            key=rngs,
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


__all__ = [
    "CheckpointLoader",
    "CheckpointModelLoader",
    "CheckpointModelWriter",
    "CheckpointWriter",
]
