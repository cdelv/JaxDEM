# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Protocols that interleave integration with periodic state/system rescaling.

Temperature control is handled by the thermostat integrators
(``linear_integrator_type="verlet_rescaling"`` or ``"langevin"``) at
``System.create`` time; nothing in this module duplicates that. What
is left here is the one control job no integrator does on its own:
rescaling the periodic box on a user-specified schedule while
integration runs.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .packingUtils import scale_to_packing_fraction

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, static_argnames=("unroll",))
def run_packing_fraction_protocol(
    state: State,
    system: System,
    *,
    strides: jax.Array,
    phi_at_frames: jax.Array,
    unroll: int = 2,
) -> tuple[State, System, tuple[State, System]]:
    """Integrate + scheduled box-rescale protocol, saving one frame per event.

    For each frame ``i`` in ``range(K)``:

    1. Advance ``strides[i]`` integration steps via :meth:`System.step`.
    2. Rescale the periodic box to ``phi_at_frames[i]`` via
       :func:`scale_to_packing_fraction`.
    3. Record ``(state, system)`` as the frame.

    All dynamics — pairwise forces, bonded forces, thermostat integrators,
    neighbor-list rebuilds, etc. — are delegated to ``system.step``. That
    means temperature control, if desired, is set up at ``System.create``
    time by picking ``linear_integrator_type="verlet_rescaling"`` (deterministic
    velocity rescaling) or ``"langevin"`` (stochastic). This function then
    runs whatever integrator is on the System and adds the box-rescale
    schedule on top.

    Parameters
    ----------
    state, system
        Initial state / system; the system's integrators + collider +
        force model determine what happens between rescales.
    strides
        1D integer array of per-frame integration step counts. Length
        ``K`` sets the number of frames.
    phi_at_frames
        1D float array of target packing fractions, one per frame, applied
        after the frame's integration strides. Must have the same length
        as ``strides``.
    unroll
        Unroll factor for the outer :func:`jax.lax.scan` (same semantics
        as :meth:`System.trajectory_rollout`).

    Returns
    -------
    (state, system, (traj_state, traj_system))
        Final state/system and the per-frame trajectory, stacked along
        leading axis ``K`` — same layout as ``trajectory_rollout``'s
        default ``save_fn``.

    Notes
    -----
    - For a non-rescaling rollout, call :meth:`System.trajectory_rollout`
      directly.
    - To drive from a ``save_steps`` array produced by
      :func:`make_save_steps_pseudolog` or :func:`make_save_steps_linear`,
      pass ``strides=np.diff(save_steps)`` and a matching
      ``phi_at_frames`` array.
    """
    strides = jnp.asarray(strides, dtype=int)
    phi_arr = jnp.asarray(phi_at_frames, dtype=float)
    if strides.ndim != 1 or phi_arr.ndim != 1:
        raise ValueError(
            "strides and phi_at_frames must be 1D arrays; got shapes "
            f"{strides.shape} and {phi_arr.shape}"
        )
    if strides.shape != phi_arr.shape:
        raise ValueError(
            "strides and phi_at_frames must have the same length; got "
            f"{strides.shape[0]} and {phi_arr.shape[0]}"
        )

    def body(
        carry: tuple[State, System], xs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[State, System], tuple[State, System]]:
        st, sys = carry
        stride, phi = xs
        st, sys = sys.step(st, sys, n=stride)
        st, sys = scale_to_packing_fraction(st, sys, phi)
        return (st, sys), (st, sys)

    (state, system), traj = jax.lax.scan(
        body, (state, system), xs=(strides, phi_arr), unroll=unroll
    )
    return state, system, traj
