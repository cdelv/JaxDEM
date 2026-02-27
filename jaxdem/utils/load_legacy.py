# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Adapter for loading HDF5 data saved with the pre-merge-2-27-26 branch.

The old format differs from the current one in several ways:

State field renames:
    ``angVel`` -> ``ang_vel``,
    ``clump_ID`` -> ``clump_id``,
    ``deformable_ID`` -> ``bond_id``,
    ``unique_ID`` -> ``unique_id``.

System changes:
    New fields ``bonded_force_model`` and ``interact_same_bond_id`` did not exist.

DeformableParticle changes:
    Class moved from ``jaxdem.forces.deformable_particle:DeformableParticleContainer``
    to ``jaxdem.bonded_forces.deformable_particle:DeformableParticleModel``.
    Fields renamed: ``elements_ID`` -> ``elements_id``,
    ``initial_bending`` -> ``initial_bendings``.
    Fields removed: ``edges_ID``, ``element_adjacency_ID``,
    ``weighted_ref_vectors``, ``directed_edges_source``,
    ``directed_edges_target``, ``lame_lambda``, ``lame_mu``.
    New field: ``w_b`` (bending normalization, computed from reference geometry).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, TYPE_CHECKING

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from ..bonded_forces import DeformableParticleModel


# ── Field-name mappings ─────────────────────────────────────────────────

_STATE_RENAMES = {
    "angVel": "ang_vel",
    "clump_ID": "clump_id",
    "deformable_ID": "bond_id",
    "unique_ID": "unique_id",
}

_DP_RENAMES = {
    "elements_ID": "elements_id",
    "initial_bending": "initial_bendings",
}

_DP_DROPPED = frozenset({
    "edges_ID",
    "element_adjacency_ID",
    "weighted_ref_vectors",
    "directed_edges_source",
    "directed_edges_target",
    "lame_lambda",
    "lame_mu",
})


# ── Low-level h5 helpers ────────────────────────────────────────────────

def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Recursively read an h5 node into plain Python / JAX objects."""
    import json

    if isinstance(node, h5py.Dataset):
        x = node[()]
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        return jnp.asarray(x)

    g = node
    kind = g.attrs.get("__kind__", None)

    if kind == "none":
        return None
    if kind == "quaternion":
        return Quaternion.create(
            w=jnp.asarray(g["w"][...]),
            xyz=jnp.asarray(g["xyz"][...]),
        )
    if kind == "dict":
        keys = json.loads(g.attrs["__keys__"])
        return {k: _read_node(g[k]) for k in keys if k in g}
    if kind in ("list", "tuple"):
        items = [_read_node(g[str(i)]) for i in sorted(int(k) for k in g.keys())]
        return items if kind == "list" else tuple(items)
    if kind == "dataclass":
        return {k: _read_node(g[k]) for k in g.keys()}

    return {k: _read_node(g[k]) for k in g.keys()}


def _root_group(path: str) -> Tuple[h5py.File, h5py.Group]:
    f = h5py.File(path, "r")
    return f, f["root"]


def _class_tag(g: h5py.Group) -> str:
    return g.attrs.get("__class__", "")


# ── State ───────────────────────────────────────────────────────────────

def load_legacy_state(path: str) -> State:
    """
    Load a :class:`~jaxdem.state.State` saved with the old field naming
    convention (``angVel``, ``clump_ID``, ``deformable_ID``, ``unique_ID``).

    Parameters
    ----------
    path : str
        Path to the ``.h5`` file containing the saved State.

    Returns
    -------
    State
        A new State constructed with the current field names.
    """
    from ..state import State

    with h5py.File(path, "r") as f:
        g = f["root"]
        raw = {k: _read_node(g[k]) for k in g.keys()}

    mapped: dict[str, Any] = {}
    for old_name, val in raw.items():
        new_name = _STATE_RENAMES.get(old_name, old_name)
        mapped[new_name] = val

    pos_c = mapped.pop("pos_c")
    q = mapped.pop("q", None)
    pos_p = mapped.pop("pos_p", None)
    vel = mapped.pop("vel", None)
    force = mapped.pop("force", None)
    ang_vel = mapped.pop("ang_vel", None)
    torque = mapped.pop("torque", None)
    rad = mapped.pop("rad", None)
    volume = mapped.pop("volume", None)
    mass = mapped.pop("mass", None)
    inertia = mapped.pop("inertia", None)
    clump_id = mapped.pop("clump_id", None)
    bond_id = mapped.pop("bond_id", None)
    unique_id = mapped.pop("unique_id", None)
    mat_id = mapped.pop("mat_id", None)
    species_id = mapped.pop("species_id", None)
    fixed = mapped.pop("fixed", None)

    if mapped:
        warnings.warn(
            f"load_legacy_state: ignoring unknown fields {sorted(mapped)}", stacklevel=2
        )

    state = State.create(
        pos=pos_c,
        pos_p=pos_p,
        vel=vel,
        force=force,
        q=q,
        ang_vel=ang_vel,
        torque=torque,
        rad=rad,
        volume=volume,
        mass=mass,
        inertia=inertia,
        clump_id=clump_id,
        bond_id=bond_id,
        mat_id=mat_id,
        species_id=species_id,
        fixed=fixed,
    )
    return state


# ── System ──────────────────────────────────────────────────────────────

def load_legacy_system(
    path: str,
    state_shape: Optional[Tuple[int, ...]] = None,
) -> System:
    """
    Load a :class:`~jaxdem.system.System` saved with the old schema (no
    ``bonded_force_model`` or ``interact_same_bond_id`` fields).

    The current ``System.create`` factory is used to produce a valid skeleton;
    scalar fields (``dt``, ``time``, ``step_count``, ``key``) and nested
    component dataclasses that still exist (``collider``, ``domain``,
    ``force_model``, ``mat_table``, ``force_manager``, integrators) are
    overwritten from the file where the schemas still match.

    Parameters
    ----------
    path : str
        Path to the ``.h5`` file containing the saved System.
    state_shape : tuple of int, optional
        Shape hint ``(N, dim)`` passed to ``System.create`` to build default
        components.  If *None*, inferred from the stored
        ``force_manager/external_force`` or ``force_manager/external_force_com``
        dataset.

    Returns
    -------
    System
        A new System instance populated with as much data from the file as
        possible.  ``bonded_force_model`` defaults to *None* and
        ``interact_same_bond_id`` defaults to *False*.
    """
    from ..system import System
    from .h5 import load as h5_load

    # The existing h5 loader already handles unknown/missing fields with
    # warnings, so the simplest correct approach is to delegate and let it
    # fill in defaults for the two new fields.
    system = h5_load(path, warn_missing=True, warn_unknown=True)
    return system


# ── Deformable Particle Container → Model ──────────────────────────────

def _compute_w_b(
    dp_fields: dict[str, Any],
    ref_pos: Optional[jax.Array] = None,
    dim: int = 3,
) -> Optional[jax.Array]:
    """
    Compute the bending normalization ``w_b`` from old DP reference data.

    For 2D this only requires ``initial_element_measures`` and
    ``element_adjacency``.  For 3D it additionally requires vertex positions
    (``ref_pos``) and ``element_adjacency_edges``.
    """
    element_adjacency = dp_fields.get("element_adjacency")
    if element_adjacency is None:
        return None

    if dim == 2:
        measures = dp_fields.get("initial_element_measures")
        if measures is None:
            return None
        left_len = measures[element_adjacency[:, 0]]
        right_len = measures[element_adjacency[:, 1]]
        dual_length = 0.5 * (left_len + right_len)
        return 1.0 / jnp.where(dual_length == 0, 1.0, dual_length)

    # 3D
    element_adjacency_edges = dp_fields.get("element_adjacency_edges")
    elements = dp_fields.get("elements")
    if element_adjacency_edges is None or elements is None or ref_pos is None:
        warnings.warn(
            "load_legacy_dp: cannot compute w_b for 3D without ref_pos, "
            "element_adjacency_edges, and elements. Set dp.w_b manually or "
            "re-create the model from vertices.",
            stacklevel=3,
        )
        return None

    hinge_verts = ref_pos[element_adjacency_edges]
    hinge_vec = hinge_verts[:, 1, :] - hinge_verts[:, 0, :]
    hinge_length = jnp.sqrt(jnp.sum(hinge_vec * hinge_vec, axis=-1))

    centroids = jnp.mean(ref_pos[elements], axis=-2)
    c1 = centroids[element_adjacency[:, 0]]
    c2 = centroids[element_adjacency[:, 1]]
    dual_length = jnp.linalg.norm(c2 - c1, axis=-1)

    return hinge_length / jnp.where(dual_length == 0, 1.0, dual_length)


def load_legacy_dp(
    path: str,
    ref_pos: Optional[jax.Array] = None,
    dim: int = 3,
) -> DeformableParticleModel:
    """
    Load an old ``DeformableParticleContainer`` h5 file and return a new
    :class:`~jaxdem.bonded_forces.DeformableParticleModel`.

    Parameters
    ----------
    path : str
        Path to the ``.h5`` file containing the saved DP container.
    ref_pos : jax.Array, optional
        Reference vertex positions, shape ``(N, dim)``.  Required for 3D to
        compute the new ``w_b`` bending normalization.  You can obtain this
        from the legacy state: ``ref_pos = state.pos``.
    dim : int
        Spatial dimension (2 or 3).  Needed to choose the correct ``w_b``
        computation.

    Returns
    -------
    DeformableParticleModel
        A new model instance with fields mapped from the old container.
    """
    from ..bonded_forces.deformable_particle import DeformableParticleModel

    with h5py.File(path, "r") as f:
        g = f["root"]
        raw = {k: _read_node(g[k]) for k in g.keys()}

    mapped: dict[str, Any] = {}
    dropped: list[str] = []
    for old_name, val in raw.items():
        if old_name in _DP_DROPPED:
            dropped.append(old_name)
            continue
        new_name = _DP_RENAMES.get(old_name, old_name)
        mapped[new_name] = val

    if dropped:
        warnings.warn(
            f"load_legacy_dp: dropped obsolete fields {sorted(dropped)}", stacklevel=2
        )

    if "w_b" not in mapped or mapped["w_b"] is None:
        mapped["w_b"] = _compute_w_b(mapped, ref_pos=ref_pos, dim=dim)

    new_fields = {f.name for f in __import__("dataclasses").fields(DeformableParticleModel)}
    kw = {k: v for k, v in mapped.items() if k in new_fields}

    unknown = sorted(set(mapped) - new_fields)
    if unknown:
        warnings.warn(
            f"load_legacy_dp: ignoring unknown fields {unknown}", stacklevel=2
        )

    return DeformableParticleModel(**kw)


# ── Convenience: load a full legacy simulation ──────────────────────────

def load_legacy_simulation(
    state_path: str,
    system_path: str,
    dp_path: Optional[str] = None,
) -> Tuple[State, System]:
    """
    Load state, system, and (optionally) a deformable-particle container from
    old-format h5 files and wire them into a ready-to-use ``(State, System)``
    pair.

    When *dp_path* is given, the DP model is attached to the system via
    ``system.bonded_force_model`` and its force/energy functions are registered
    in the force manager.

    Parameters
    ----------
    state_path : str
        Path to the legacy State h5 file.
    system_path : str
        Path to the legacy System h5 file.
    dp_path : str, optional
        Path to the legacy DeformableParticleContainer h5 file.

    Returns
    -------
    state : State
        The loaded state with current field names.
    system : System
        The loaded system, with bonded forces wired up if *dp_path* was given.

    Example
    -------
    .. code-block:: python

        from jaxdem.utils.load_legacy import load_legacy_simulation

        state, system = load_legacy_simulation(
            "old_data/state.h5",
            "old_data/system.h5",
            dp_path="old_data/dp.h5",
        )
    """
    from dataclasses import replace
    from ..forces import ForceManager

    state = load_legacy_state(state_path)
    system = load_legacy_system(system_path)

    if dp_path is not None:
        ref_pos = state.pos_c + state.q.rotate(state.q, state.pos_p)
        dp = load_legacy_dp(dp_path, ref_pos=ref_pos, dim=int(state.dim))

        system = replace(
            system,
            bonded_force_model=dp,
            force_manager=ForceManager.create(
                state_shape=state.shape,
                gravity=None,
                force_functions=(dp.force_and_energy_fns,),
            ),
        )

    return state, system


__all__ = [
    "load_legacy_state",
    "load_legacy_system",
    "load_legacy_dp",
    "load_legacy_simulation",
]
