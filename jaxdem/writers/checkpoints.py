# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Orbax checkpoint writers and loaders for simulations, RL models and trainers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
import json
import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from orbax.checkpoint.checkpoint_managers import (  # type: ignore[import-untyped]
    preservation_policy as preservation_policy_lib,
)
from orbax.checkpoint.checkpoint_managers import (
    save_decision_policy as save_decision_policy_lib,
)

from ..material_matchmakers import MaterialMatchmaker
from ..materials import MaterialTable
from ..state import State
from ..system import System
from ..utils import decode_callable
from ..utils.serialization import supported_kwargs
from ..utils.serialization_schema import (
    CURRENT_SCHEMA_VERSION,
    FACET_INCIDENCE_LAYOUT_VERSION,
    LEGACY_FACET_FIELDS,
    make_serialization_manifest,
    serialization_schema_version,
)
from ._directory import prepare_output_directory

if TYPE_CHECKING:
    from ..rl.models import Model
    from ..rl.trainers import Trainer


_log = logging.getLogger(__name__)


def _checkpoint_schema_version(metadata: dict[str, Any]) -> int:
    """Consume and validate the checkpoint metadata schema version."""
    raw_version = metadata.pop("checkpoint_metadata_version", 1)
    manifest = metadata.pop("serialization_manifest", None)
    return serialization_schema_version(
        manifest, legacy_version=raw_version, expected_storage="orbax"
    )


_SCHEMA3_SYSTEM_METADATA = frozenset(
    {
        "state_shape",
        "bond_id_shape",
        "linear_integrator_type",
        "linear_integrator_kw",
        "rotation_integrator_type",
        "rotation_integrator_kw",
        "collider_type",
        "collider_kw_metadata",
        "domain_type",
        "domain_kw",
        "force_model_type",
        "force_model_metadata",
        "bonded_force_model_type",
        "bonded_force_model_kw",
        "mat_table_metadata",
        "force_function_metadata",
        "minimizer",
        "target_fn",
        "user_pre_step_actions",
        "user_post_step_actions",
    }
)


def _validate_schema3_system_metadata(metadata: dict[str, Any]) -> None:
    """Reject incomplete current metadata before defaults change the system."""
    missing = sorted(_SCHEMA3_SYSTEM_METADATA - metadata.keys())
    if missing:
        raise RuntimeError(
            f"Schema-3 checkpoint metadata is missing required fields: {missing}"
        )


def _is_key(x: Any) -> bool:
    return isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)


def _checkpoint_tree(system: Any) -> Any:
    """Omit empty array payloads and unwrap typed PRNG keys.

    Orbax can store neither. Empty shapes, dtypes and key types are recovered
    from the reconstructed target. Runtime histories remain arrays, including
    for stateless laws.
    """

    def storable(x: Any) -> Any:
        if _is_key(x):
            return jax.random.key_data(x)
        return None if isinstance(x, jax.Array) and x.size == 0 else x

    return jax.tree.map(storable, system)


def _restore_empty_arrays(target: Any, restored: Any) -> Any:
    """Invert :func:`_checkpoint_tree` with the leaves of ``target``."""

    def original(template: Any, value: Any) -> Any:
        if isinstance(template, jax.Array) and template.size == 0:
            return template
        if _is_key(template):
            return jax.random.wrap_key_data(value, impl=jax.random.key_impl(template))
        return value

    return jax.tree.map(original, target, restored, is_leaf=lambda x: x is None)


def _lane(tree: Any, ndim: int) -> Any:
    """First entry along ``ndim`` leading batch axes: one instance of a batch."""
    return jax.tree.map(lambda x: x[(0,) * ndim], tree) if ndim else tree


def _broadcast(tree: Any, batch: tuple[int, ...]) -> Any:
    """Give a single-instance restore target the batch axes of its checkpoint."""
    if not batch:
        return tree
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (*batch, *jnp.shape(x))), tree)


def _restore_layout_target(target: Any, saved: Any, *, legacy_facets: bool) -> Any:
    """Match old trees only where a known additive facet field is absent."""
    serialized = ocp.tree.serialize_tree(
        _checkpoint_tree(target), keep_empty_nodes=True
    )

    def align(value: Any, metadata: Any, original: Any) -> Any:
        if isinstance(value, dict):
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint tree structure differs from its target.")
            missing = value.keys() - metadata.keys()
            unknown = metadata.keys() - value.keys()
            class_name = f"{type(original).__module__}:{type(original).__qualname__}"
            allowed = LEGACY_FACET_FIELDS.get(class_name, frozenset())
            if unknown or (missing and not (legacy_facets and missing == allowed)):
                raise ValueError(
                    f"Checkpoint {class_name} fields differ: "
                    f"missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}"
                )
            return {
                key: align(
                    child,
                    metadata[key],
                    (
                        original[key]
                        if isinstance(original, dict)
                        else getattr(original, key)
                    ),
                )
                for key, child in value.items()
                if key in metadata
            }
        if isinstance(value, (tuple, list)):
            if len(value) != len(metadata):
                raise ValueError("Checkpoint sequence length differs from its target.")
            return type(value)(
                align(child, metadata[index], original[index])
                for index, child in enumerate(value)
            )
        return value

    return align(serialized, saved, target)


def _restore_layout_value(target: Any, restored: Any) -> Any:
    """Restore current types/static data and initialize permitted missing leaves."""
    template = ocp.tree.serialize_tree(_checkpoint_tree(target), keep_empty_nodes=True)

    def complete(expected: Any, value: Any) -> Any:
        if isinstance(expected, dict):
            return {
                key: complete(child, value[key]) if key in value else child
                for key, child in expected.items()
            }
        if isinstance(expected, (tuple, list)):
            return type(expected)(
                complete(a, b) for a, b in zip(expected, value, strict=True)
            )
        return value

    restored_tree = ocp.tree.deserialize_tree(
        complete(template, restored), _checkpoint_tree(target), keep_empty_nodes=True
    )
    return _restore_empty_arrays(target, restored_tree)


def _deserialize_force_functions(
    data: list[dict[str, Any]],
    strict: bool = True,
) -> list[tuple[Any, ...]]:
    """Reconstruct custom force function entries from serialized metadata.

    Returns a list of ``(force_fn, energy_fn_or_None, is_com)`` tuples
    suitable for passing as ``force_manager_kw["force_functions"]``.

    If the loader cannot resolve a callable (e.g. a function defined in
    ``__main__`` of a different script), it raises a ``RuntimeError`` when
    ``strict`` is ``True`` (the default). When ``strict`` is ``False``, the
    loader skips the entry with a warning, which changes the restored physics.
    """
    entries: list[tuple[Any, ...]] = []
    for item in data:
        force_path = item["force"]
        try:
            force_fn = decode_callable(force_path)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
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
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise RuntimeError(
                        f"Could not restore energy function '{energy_path}': {exc}. "
                        "The checkpoint cannot be loaded with the physics it was saved with."
                    ) from exc
                warnings.warn(
                    f"Could not restore energy function '{energy_path}': {exc}. "
                    "The loaded force will use zero custom energy.",
                    stacklevel=3,
                )

        is_com = bool(item.get("is_com", False))
        entries.append((force_fn, energy_fn, is_com))
    return entries


# Every item a JaxDEM checkpoint can hold. Loaders register them all, so that
# reading one payload of a trainer checkpoint does not warn about the others.
_ITEM_ARGS = {
    **dict.fromkeys(
        ("state", "system", "model_state", "optimizer_state", "trainer_state"),
        ocp.args.StandardRestore,
    ),
    **dict.fromkeys(
        ("state_metadata", "system_metadata", "model_metadata", "trainer_metadata"),
        ocp.args.JsonRestore,
    ),
}


def _state_system_args(state: State, system: System) -> dict[str, Any]:
    """Orbax items for a state/system pair, shared with trainer checkpoints.

    Leading batch axes are stored unchanged. The metadata describes one
    instance and records the batch shape.
    """
    batch = tuple(state.pos.shape[:-2])
    lane_state, lane_system = _lane((state, system), len(batch))
    # Building metadata also validates that every persisted callback and
    # custom force/energy function round-trips through its import path.
    system_metadata = lane_system.metadata
    system_metadata["checkpoint_metadata_version"] = CURRENT_SCHEMA_VERSION
    system_metadata["serialization_manifest"] = make_serialization_manifest(
        "orbax", "state_system"
    )
    system_metadata["state_shape"] = tuple(lane_state.pos.shape)
    system_metadata["bond_id_shape"] = tuple(lane_state.bond_id.shape)

    if system.minimizer is not None:
        constructor_fn = getattr(system.minimizer, "_constructor", None)
        if constructor_fn is not None:
            mod = getattr(constructor_fn, "__module__", None)
            if mod == "__main__":
                warnings.warn(
                    f"Minimizer constructor '{constructor_fn.__name__}' is defined in __main__. "
                    "It will not be restorable from a different script. "
                    "Define it in an importable module instead.",
                    stacklevel=3,
                )

    return {
        "state": ocp.args.StandardSave(_checkpoint_tree(state)),
        "system": ocp.args.StandardSave(_checkpoint_tree(system)),
        "state_metadata": ocp.args.JsonSave(
            {
                "shape": lane_state.shape,
                "batch_shape": batch,
                "only_spheres": state.only_spheres,
                "facet_incidence_layout": FACET_INCIDENCE_LAYOUT_VERSION,
                "facet_ids_extra_shape": lane_state.facet_ids_extra.shape,
            }
        ),
        "system_metadata": ocp.args.JsonSave(system_metadata),
    }


@dataclass(slots=True, weakref_slot=True)
class BaseCheckpointManager:
    """Base class that provides context management and shared setup for Orbax checkpointers."""

    directory: Path | str = Path("./checkpoints")
    checkpointer: ocp.CheckpointManager = field(init=False)

    def _prepare_directory(self, clean: bool = False) -> None:
        """Resolve ``self.directory`` and create it if needed.

        If ``clean`` is True, erase and recreate the directory.
        If ``clean`` is False (the default), keep the existing contents.
        """
        self.directory = prepare_output_directory(self.directory, clean=clean)

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
        registry = ocp.DefaultCheckpointHandlerRegistry()
        for item, args in _ITEM_ARGS.items():
            registry.add(item, args)
        self.checkpointer = ocp.CheckpointManager(
            self.directory,
            options=options,
            handler_registry=registry,
        )

    def _resolve_step(self, step: int | None) -> int:
        """Return ``step``, or the latest step if None, checking that it exists."""
        if step is None:
            step = self.checkpointer.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoints found in: {self.directory}")
        if step not in self.checkpointer.all_steps():
            raise FileNotFoundError(
                f"step={step} checkpoints not found in: {self.directory}. Available steps: {self.checkpointer.all_steps()}"
            )
        return int(step)

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
        try:
            self.close()
        except Exception:
            pass

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


@dataclass(slots=True)
class CheckpointWriter(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint saving.

    Notes
    -----
    The writer serializes custom force functions passed via
    ``force_manager_kw`` by their fully-qualified module path (e.g.
    ``mypackage.forces.trap``). A different script **cannot** restore
    functions defined in the top-level script (``__main__``). Nonportable
    callables are rejected at save time: force, energy, target, and callback
    functions must round-trip by import-path identity. Define these functions
    in an importable module.

    """

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, keep all checkpoints.
    """

    save_every: int = 1
    """
    Checkpoint interval measured in the step supplied to :meth:`save`,
    rather than in save calls.
    """

    clean: bool = False
    """
    If True, erase and recreate the target directory on construction.
    If False (the default), keep existing checkpoints in the directory,
    so a resumed run does not destroy earlier checkpoints.
    """

    def __post_init__(self) -> None:
        self._prepare_directory(clean=self.clean)
        self._init_writer_manager()

    @partial(jax.named_call, name="CheckpointWriter.save")
    def save(self, state: State, system: System) -> None:
        """Save a checkpoint of the given state and system at the current step.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.

        """
        self.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(**_state_system_args(state, system)),
        )


@dataclass(slots=True)
class CheckpointLoader(BaseCheckpointManager):
    """Restore current manifests and legacy JaxDEM schema versions 1 and 2.

    Checkpoints are trusted input: restoration imports Python modules named in
    callable metadata, so untrusted checkpoint directories must not be loaded.
    Unknown schema versions are rejected rather than guessed or partially
    migrated.
    """

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
            If ``True`` (default), raise a ``RuntimeError`` when the loader
            cannot re-import a custom force function recorded in the
            checkpoint. If ``False``, skip force functions that do not load,
            with a warning.

        Returns
        -------
        Tuple[State, System]
            The restored `State` and `System`.

        """
        step = self._resolve_step(step)

        metadata = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                state_metadata=ocp.args.JsonRestore(),
                system_metadata=ocp.args.JsonRestore(),
            ),
        )

        system_metadata = dict(metadata.system_metadata)
        metadata_version = _checkpoint_schema_version(system_metadata)
        if metadata_version == CURRENT_SCHEMA_VERSION:
            _validate_schema3_system_metadata(system_metadata)
        state_shape = tuple(metadata.state_metadata["shape"])
        # Batched pairs, as in trainer checkpoints, are rebuilt from one instance.
        batch = tuple(metadata.state_metadata.get("batch_shape", ()))
        system_metadata["state_shape"] = tuple(
            system_metadata.get("state_shape", state_shape)
        )
        bond_id_shape = system_metadata.pop("bond_id_shape", None)
        state_kw: dict[str, Any] = {
            "only_spheres": metadata.state_metadata.get("only_spheres", False)
        }
        if bond_id_shape is not None:
            state_kw["bond_id"] = jnp.full(tuple(bond_id_shape), -1, dtype=int)
        extra_shape = metadata.state_metadata.get("facet_ids_extra_shape")
        has_layout = "facet_incidence_layout" in metadata.state_metadata
        if has_layout != (extra_shape is not None):
            raise RuntimeError(
                "Checkpoint facet incidence layout and shape must be saved together."
            )
        if extra_shape is not None:
            if (
                not isinstance(extra_shape, (list, tuple))
                or len(extra_shape) != len(state_shape)
                or tuple(extra_shape[:-1]) != state_shape[:-1]
                or type(extra_shape[-1]) is not int
                or extra_shape[-1] < 0
            ):
                raise RuntimeError("Invalid checkpoint facet incidence shape.")
            state_kw["facet_ids_extra"] = jnp.full(tuple(extra_shape), -1, dtype=int)
        state_target = _broadcast(
            State.create(jnp.zeros(state_shape), **state_kw), batch
        )
        legacy_facets = not has_layout
        if not legacy_facets:
            layout = metadata.state_metadata["facet_incidence_layout"]
            if type(layout) is not int or layout != FACET_INCIDENCE_LAYOUT_VERSION:
                raise RuntimeError("Unsupported checkpoint facet incidence layout.")
        item_metadata = self.checkpointer.metadata(step).item_metadata
        if legacy_facets and (
            LEGACY_FACET_FIELDS["jaxdem.state:State"]
            & item_metadata["state"].tree.keys()
        ):
            raise RuntimeError("Stored facet incidence fields require layout metadata.")
        if extra_shape is not None:
            saved_ids = item_metadata["state"].tree.get("facet_ids_extra")
            saved_vertices = item_metadata["state"].tree.get("facet_vertices_extra")
            expected_ids = (*batch, *extra_shape)
            expected_vertices = (*expected_ids, state_shape[-1])
            if expected_ids[-1] == 0:
                if saved_ids is not None or saved_vertices is not None:
                    raise RuntimeError(
                        "Checkpoint facet incidence shape does not match its payload."
                    )
            elif (
                getattr(saved_ids, "shape", None) != expected_ids
                or getattr(saved_vertices, "shape", None) != expected_vertices
            ):
                raise RuntimeError(
                    "Checkpoint facet incidence shape does not match its payload."
                )

        # Restore state first to get correct radii and properties before building the system
        try:
            state_restore_target = _restore_layout_target(
                state_target, item_metadata["state"].tree, legacy_facets=legacy_facets
            )
            res_state = self.checkpointer.restore(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state_restore_target),
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore state from checkpoint step {step} in "
                f"{self.directory}. The checkpoint may be corrupt or "
                "incompatible with the current State schema."
            ) from exc
        state_target = _restore_layout_value(state_target, res_state.state)
        lane_state = _lane(state_target, len(batch))
        system_metadata.setdefault("bonded_force_model_type", None)
        system_metadata.setdefault("bonded_force_model_kw", None)
        minimizer_meta = system_metadata.pop("minimizer", None)
        if minimizer_meta is not None:
            constructor_path = minimizer_meta.get("constructor")
            kw = minimizer_meta.get("kw", {})
            if constructor_path:
                constructor = decode_callable(constructor_path)
                kw, dropped = supported_kwargs(constructor, kw)
                if dropped:
                    warnings.warn(
                        "Ignoring saved minimizer arguments no longer accepted "
                        f"by {constructor_path}: {', '.join(dropped)}",
                        stacklevel=2,
                    )
                system_metadata["minimizer"] = constructor
                system_metadata["minimizer_kw"] = kw

        target_fn_path = system_metadata.pop("target_fn", None)
        if target_fn_path is not None:
            system_metadata["target_fn"] = decode_callable(target_fn_path)

        for callback_name in ("user_pre_step_actions", "user_post_step_actions"):
            callback_path = system_metadata.get(callback_name)
            if isinstance(callback_path, str):
                try:
                    system_metadata[callback_name] = decode_callable(callback_path)
                except (ImportError, AttributeError, TypeError, ValueError) as exc:
                    if strict or metadata_version >= 2:
                        raise RuntimeError(
                            f"Could not restore required callback '{callback_name}' "
                            f"from '{callback_path}': {exc}."
                        ) from exc
                    warnings.warn(
                        f"Could not restore legacy callback '{callback_name}' from "
                        f"'{callback_path}'; using the default callback.",
                        stacklevel=2,
                    )
                    system_metadata.pop(callback_name, None)

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
            system_metadata["force_model"] = force_model

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
                collider_kw["state"] = lane_state
            if "secondary_collider" in collider_kw:
                sub_val = collider_kw["secondary_collider"]
                if (
                    isinstance(sub_val, dict)
                    and "kw" in sub_val
                    and _accepts_state(sub_val.get("type"))
                ):
                    sub_val["kw"]["state"] = lane_state
            elif "secondary_collider_type" in collider_kw and _accepts_state(
                collider_kw["secondary_collider_type"]
            ):
                sub_kw = dict(collider_kw.get("secondary_collider_kw") or {})
                sub_kw["state"] = lane_state
                collider_kw["secondary_collider_kw"] = sub_kw
            system_metadata["collider_kw"] = collider_kw

        try:
            system_target = _broadcast(System.create(**system_metadata), batch)
        except Exception:
            if metadata_version >= 2:
                raise
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

        system_restore_target = _restore_layout_target(
            system_target, item_metadata["system"].tree, legacy_facets=legacy_facets
        )
        result = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                system=ocp.args.StandardRestore(system_restore_target),
            ),
        )

        return state_target, _restore_layout_value(system_target, result.system)

    @partial(jax.named_call, name="CheckpointLoader.latest_step")
    def latest_step(self) -> int | None:
        return self.checkpointer.latest_step()


@dataclass(slots=True)
class CheckpointModelWriter(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint saving for jaxdem.rl.models.Model."""

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, keep all checkpoints.
    """

    save_every: int = 1
    """
    Checkpoint interval measured in the step supplied to :meth:`save`,
    rather than in save calls.
    """

    clean: bool = False
    """
    If True, erase and recreate the target directory on construction.
    If False (the default), keep existing checkpoints in the directory,
    so a resumed run does not destroy earlier checkpoints.
    """

    def __post_init__(self) -> None:
        self._prepare_directory(clean=self.clean)
        self._init_writer_manager()

    @partial(jax.named_call, name="CheckpointModelWriter.save")
    def save(self, model: Model, step: int) -> None:
        """Save the model at the given step.

        Stores ``model_state`` and JSON metadata. ``model.metadata`` must
        contain JSON-serializable fields. The writer adds ``model_type``
        to the metadata.
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


@dataclass(slots=True)
class CheckpointModelLoader(BaseCheckpointManager):
    """Thin wrapper around Orbax checkpoint restoring for jaxdem.rl.models.Model."""

    def __post_init__(self) -> None:
        self._init_loader_manager()

    @partial(jax.named_call, name="CheckpointModelLoader.load")
    def load(self, step: int | None = None) -> Model:
        """Load the model at the given step. If ``step`` is None, load the latest checkpoint."""
        from flax import nnx

        from ..rl.action_spaces import ActionSpace
        from ..rl.models import Model

        step = self._resolve_step(step)

        model_metadata = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(
                model_metadata=ocp.args.JsonRestore(),
            ),
        )
        model_metadata = dict(model_metadata.model_metadata)
        # Stacked models, as in batched trainer checkpoints, share one definition.
        batch = tuple(model_metadata.pop("batch_shape", ()))

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
                model_state=ocp.args.StandardRestore(_broadcast(state, batch)),
            ),
        )
        state = result.model_state

        return nnx.merge(graphdef, state)

    @partial(jax.named_call, name="CheckpointModelLoader.latest_step")
    def latest_step(self) -> int | None:
        return self.checkpointer.latest_step()


_TRAINER_CHECKPOINT_VERSION = 1
# Trainer fields with their own checkpoint items. The rest are plain values.
_TRAINER_GRAPH_FIELDS = ("env", "graphdef", "graphstate", "key")


def _dynamic_fields(obj: Any, skip: tuple[str, ...]) -> dict[str, Any]:
    """Non-static dataclass fields of ``obj`` not named in ``skip``."""
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if f.name not in skip and not f.metadata.get("static")
    }


def _trainer_state(trainer: Any, scalars: Any) -> dict[str, Any]:
    """Trainer leaves outside the environment state/system and the model graph.

    ``scalars`` names the fields kept as Python numbers in the JSON metadata.
    The other fields are arrays, also in a restore target that still holds
    Python numbers where the saved trainer held arrays.
    """
    fields = _dynamic_fields(trainer, _TRAINER_GRAPH_FIELDS)
    return {
        "key": trainer.key,
        "fields": {k: jnp.asarray(v) for k, v in fields.items() if k not in scalars},
        "env": _dynamic_fields(trainer.env, ("state", "system")),
    }


def _static_metadata(obj: Any, lost: list[str]) -> dict[str, Any]:
    """JSON form of the static dataclass fields of ``obj``.

    A field without one, such as an optimizer that is not importable, is
    stored as None and named in ``lost``.
    """
    meta: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        if f.metadata.get("static"):
            value = obj._serialize_value(getattr(obj, f.name))
            try:
                json.dumps(value)
            except TypeError:
                lost.append(f.name)
                value = None
            meta[f.name] = value
    return meta


def _static_values(meta: dict[str, Any]) -> dict[str, Any]:
    """Invert :func:`_static_metadata`. Lists return as hashable tuples."""
    from ..factory import Factory

    def hashable(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(hashable(v) for v in value)
        return value

    return {k: hashable(Factory._deserialize_kws(v)) for k, v in meta.items()}


def _registered(root: Any, name: Any) -> Any:
    """Class that a checkpoint names by its registry key."""
    from ..factory import _normalize_key

    cls = root._registry.get(_normalize_key(name)) if isinstance(name, str) else None
    if cls is None:
        raise RuntimeError(
            f"{root.__name__} {name!r} is not registered. Import the module "
            "that defines it before loading, or pass `target`."
        )
    return cls


@dataclass(slots=True)
class CheckpointTrainerWriter(BaseCheckpointManager):
    """Save complete :class:`jaxdem.rl.trainers.Trainer` objects with Orbax.

    One checkpoint holds the environment, the model, the optimizer state, the
    PRNG key and the hyperparameters: everything an exact continuation needs.
    Stacked trainers (one leading axis per ``vmap`` or ``stack``) are saved as
    they are.

    :class:`CheckpointModelLoader` reads the model from the same directory
    without the rest, and :class:`CheckpointLoader` reads the environment
    state and system.

    Notes
    -----
    The trainer graph must be ``(model, optimizer)``, as in
    :class:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer`.
    """

    max_to_keep: int | None = None
    """
    Keep the last max_to_keep checkpoints. If None, keep all checkpoints.
    """

    save_every: int = 1
    """
    Checkpoint interval measured in the step supplied to :meth:`save`,
    rather than in save calls.
    """

    clean: bool = False
    """
    If True, erase and recreate the target directory on construction.
    If False (the default), keep existing checkpoints in the directory,
    so a resumed run does not destroy earlier checkpoints.
    """

    def __post_init__(self) -> None:
        self._prepare_directory(clean=self.clean)
        self._init_writer_manager()

    @partial(jax.named_call, name="CheckpointTrainerWriter.save")
    def save(self, trainer: Trainer, step: int, *, force: bool = False) -> None:
        """Save the trainer at the given step.

        Parameters
        ----------
        trainer : Trainer
            The trainer to save.
        step : int
            Number of completed epochs. Pass the same number as
            ``start_epoch`` to continue from this checkpoint. The writer
            skips a step that is not a multiple of ``save_every`` or that
            does not exceed the latest saved step. A skipped step moves no
            data off the device.
        force : bool
            Save even when ``save_every`` excludes this step. An existing
            step is never replaced.

        """
        from ..rl.env_wrappers import _wrapper_chain

        step = int(step)
        if not force:
            if not self.checkpointer.should_save(step):
                return
        elif step in self.checkpointer.all_steps():
            # Orbax raises here, which would lose the trainer at the end of a run.
            warnings.warn(
                f"A checkpoint for step {step} already exists in "
                f"{self.directory}. It was not replaced.",
                stacklevel=2,
            )
            return

        batch = tuple(jnp.shape(trainer.advantage_gamma))
        model = _lane(trainer, len(batch)).model
        model_metadata = {
            **model.metadata,
            "model_type": model.type_name,
            "batch_shape": batch,
        }

        fields = _dynamic_fields(trainer, _TRAINER_GRAPH_FIELDS)
        scalars = {k: v for k, v in fields.items() if isinstance(v, (bool, int, float))}
        state = _trainer_state(trainer, scalars)
        # These parts cannot be rebuilt from metadata: loading needs `target`.
        needs_target: list[str] = []
        if not getattr(trainer, "optimizer_config", ()):
            needs_target.append("optimizer_config")
        for name, tree in {**state["fields"], **state["env"]}.items():
            if not all(
                jnp.size(leaf) > 0
                and not _is_key(leaf)
                and all(isinstance(k, jax.tree_util.DictKey) for k in path)
                for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
            ):
                needs_target.append(name)

        env_cls = type(trainer.env)
        env_cls = getattr(env_cls, "_base_env_cls", env_cls)
        metadata = {
            "trainer_checkpoint_version": _TRAINER_CHECKPOINT_VERSION,
            "trainer_type": getattr(type(trainer), "__registry_name__", None),
            "batch_shape": batch,
            "static": _static_metadata(trainer, needs_target),
            "scalars": scalars,
            "key_impl": (
                str(jax.random.key_impl(trainer.key)) if _is_key(trainer.key) else None
            ),
            "env_type": getattr(env_cls, "__registry_name__", None),
            "env_wrappers": _wrapper_chain(trainer.env),
            "env_static": _static_metadata(trainer.env, needs_target),
            "needs_target": needs_target,
        }

        self.checkpointer.save(
            step,
            args=ocp.args.Composite(
                **_state_system_args(trainer.env.state, trainer.env.system),
                model_state=ocp.args.StandardSave(trainer.graphstate[0]),
                model_metadata=ocp.args.JsonSave(model_metadata),
                optimizer_state=ocp.args.StandardSave(trainer.graphstate[1]),
                trainer_state=ocp.args.StandardSave(_checkpoint_tree(state)),
                trainer_metadata=ocp.args.JsonSave(metadata),
            ),
            force=True,
        )


@dataclass(slots=True)
class CheckpointTrainerLoader(BaseCheckpointManager):
    """Restore trainers saved by :class:`CheckpointTrainerWriter`.

    Checkpoints are trusted input: restoration imports Python modules named in
    their metadata, so untrusted checkpoint directories must not be loaded.
    """

    def __post_init__(self) -> None:
        self._init_loader_manager()

    @partial(jax.named_call, name="CheckpointTrainerLoader.load")
    def load(
        self,
        step: int | None = None,
        *,
        target: Trainer | None = None,
        strict: bool = True,
    ) -> Trainer:
        """Load the trainer at the given step.

        The checkpoint describes itself: the environment, model, optimizer and
        trainer are rebuilt from its metadata. Custom environment, model and
        trainer classes must be registered, so import their modules first.

        Parameters
        ----------
        step : Optional[int]
            Number of completed epochs. If None, load the latest checkpoint.
        target : Trainer, optional
            A trainer with the structure of the saved one, built by the same
            code. The saved arrays replace its arrays. Its classes, static
            configuration and optimizer stay. A checkpoint needs a target
            when the metadata cannot describe the trainer: an optimizer that
            is not an importable function (a lambda or ``partial``), or
            ``env_params`` that hold empty arrays, PRNG keys or containers
            other than dicts.
        strict : bool
            Passed to :meth:`CheckpointLoader.load` for the environment system.

        Returns
        -------
        Trainer
            The restored trainer. Continue it with
            ``trainer.train(trainer, start_epoch=step)``.

        """
        from flax import nnx

        from ..rl.env_wrappers import _rewrap
        from ..rl.environments import Environment
        from ..rl.trainers import Trainer

        step = self._resolve_step(step)

        def restore(name: str, tree: Any = None) -> Any:
            args = ocp.args.StandardRestore(tree)
            return self.checkpointer.restore(
                step, args=ocp.args.Composite(**{name: args})
            )[name]

        def restore_into(name: str, template: Any) -> Any:
            return _restore_empty_arrays(
                template, restore(name, _checkpoint_tree(template))
            )

        meta = self.checkpointer.restore(
            step,
            args=ocp.args.Composite(trainer_metadata=ocp.args.JsonRestore()),
        ).trainer_metadata
        if meta.get("trainer_checkpoint_version") != _TRAINER_CHECKPOINT_VERSION:
            raise RuntimeError(
                "Unsupported trainer checkpoint version "
                f"{meta.get('trainer_checkpoint_version')!r}"
            )
        batch = tuple(meta["batch_shape"])

        build: Any
        if target is not None:
            rest = restore_into(
                "trainer_state", _trainer_state(target, meta["scalars"])
            )
            env = dataclasses.replace(
                target.env,
                state=restore_into("state", target.env.state),
                system=restore_into("system", target.env.system),
                **rest["env"],
            )
            graphdef, graph_target = target.graphdef, target.graphstate
            model_state = restore("model_state", graph_target[0])
            optimizer_target = graph_target[1]
            key = rest["key"]
            build = partial(dataclasses.replace, target)
        else:
            if meta["needs_target"]:
                raise RuntimeError(
                    "This checkpoint cannot rebuild the trainer fields "
                    f"{meta['needs_target']} from its metadata. Build the "
                    "trainer with the code that made the saved one and pass "
                    "it as `target`."
                )
            trainer_cls = _registered(Trainer, meta["trainer_type"])
            env_cls = _registered(Environment, meta["env_type"])
            with CheckpointLoader(self.directory) as loader:
                state, system = loader.load(step, strict=strict)
            with CheckpointModelLoader(self.directory) as model_loader:
                model_def, model_state = nnx.split(model_loader.load(step))
            rest = jax.tree.map(jnp.asarray, restore("trainer_state"))

            saved_env = rest.get("env", {})
            env = _rewrap(
                env_cls(
                    state=state,
                    system=system,
                    # Orbax drops empty containers, such as env_params={}.
                    **{
                        f.name: saved_env.get(f.name, {})
                        for f in dataclasses.fields(env_cls)
                        if f.name not in ("state", "system")
                        and not f.metadata.get("static")
                    },
                    **_static_values(meta["env_static"]),
                ),
                meta["env_wrappers"],
            )

            # The optimizer target comes from one model, then gains the batch axes.
            static = _static_values(meta["static"])
            model = nnx.merge(model_def, _lane(model_state, len(batch)))
            tx = trainer_cls.optimizer_tx(**dict(static["optimizer_config"]))
            graphdef, graph_target = nnx.split(
                (model, nnx.Optimizer(model, tx, wrt=nnx.Param))
            )
            optimizer_target = _broadcast(graph_target[1], batch)
            key = rest["key"]
            if meta["key_impl"] is not None:
                key = jax.random.wrap_key_data(key, impl=meta["key_impl"])
            build = partial(trainer_cls, graphdef=graphdef, **static)

        model, optimizer = nnx.merge(graphdef, graph_target, copy=True)
        nnx.update(model, model_state)
        nnx.update(optimizer, restore("optimizer_state", optimizer_target))
        return build(
            env=env,
            graphstate=nnx.state((model, optimizer)),
            key=key,
            **meta["scalars"],
            **rest.get("fields", {}),
        )

    @partial(jax.named_call, name="CheckpointTrainerLoader.latest_step")
    def latest_step(self) -> int | None:
        return self.checkpointer.latest_step()


__all__ = [
    "CheckpointLoader",
    "CheckpointModelLoader",
    "CheckpointModelWriter",
    "CheckpointTrainerLoader",
    "CheckpointTrainerWriter",
    "CheckpointWriter",
]
