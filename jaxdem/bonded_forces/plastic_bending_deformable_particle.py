# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle model with plastic bending angles."""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import jax
from jax.typing import ArrayLike

from . import BondedForceModel
from .deformable_particle import (
    DeformableParticleModel,
    _angle_difference,
    _broadcast_tau_s,
    _relax_reference,
    _merge_metric_field,
    current_bending_angles,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@BondedForceModel.register("PlasticBendingDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticBendingDeformableParticleModel(DeformableParticleModel):
    r"""Deformable particle model with plastic bending angles.

    Elastic forces match :class:`DeformableParticleModel <jaxdem.bonded_forces.deformable_particle.DeformableParticleModel>`. The model keeps
    the edge-spring, measure, content, bending, and surface-tension terms.
    See the :class:`DeformableParticleModel <jaxdem.bonded_forces.deformable_particle.DeformableParticleModel>` docstring for the full energy
    definition and shape conventions. The only difference is the plastic
    update rule for the reference bending angles.

    Instead of relaxing each edge rest length (as in
    :class:`PlasticDeformableParticleModel <jaxdem.bonded_forces.plastic_deformable_particle.PlasticDeformableParticleModel>`), the **reference bending angle**
    of each adjacency relaxes toward the current bending angle:

    The update runs once per physical time step, after the integrator drift
    and before force evaluation.

    .. math::

        \theta_{a,0}(t+dt) = \theta_{a,0}(t)
            + (1-e^{-dt/\tau_{s,a}})\operatorname{wrap}(\theta_a(t)-\theta_{a,0}(t))

    where :math:`\theta_{a,0}` is the reference (rest) bending angle
    (``initial_bendings``), :math:`\theta_a` is the current bending angle,
    :math:`\tau_{s,a}` is the relaxation time (``tau_s``), and :math:`dt` is the
    simulation time step. The wrapped difference lies in [-pi, pi]. Each
    update integrates relaxation exactly with the current angle held fixed;
    infinite tau_s leaves the reference unchanged.

    Shapes (see :class:`DeformableParticleModel <jaxdem.bonded_forces.deformable_particle.DeformableParticleModel>` for definitions of K, M, E, A):

    - ``tau_s``: ``(A,)`` — one relaxation time per adjacency (bending hinge).
    """

    tau_s: jax.Array | None = None
    """
    Plastic relaxation time for each adjacency (bending hinge). Shape: (A,).
    """

    @classmethod
    def Create(
        cls,
        *,
        vertices: ArrayLike | None = None,
        # topology
        elements: ArrayLike | None = None,
        edges: ArrayLike | None = None,
        element_adjacency: ArrayLike | None = None,
        element_adjacency_edges: ArrayLike | None = None,
        # ID mappings
        elements_id: ArrayLike | None = None,
        # reference configuration
        initial_body_contents: ArrayLike | None = None,
        initial_element_measures: ArrayLike | None = None,
        initial_edge_lengths: ArrayLike | None = None,
        initial_bendings: ArrayLike | None = None,
        # coefficients
        em: ArrayLike | None = None,
        ec: ArrayLike | None = None,
        eb: ArrayLike | None = None,
        el: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
        tau_s: ArrayLike | None = None,
        w_b: ArrayLike | None = None,  # compatibility with checkpointer.
        vertex_count: int | None = None,
    ) -> Self:
        # Explicit two-arg super: dataclass(slots=True) recreates the class, so
        # zero-arg super()'s __class__ cell points at the discarded original
        # and raises TypeError on Python < 3.14.
        base = super(PlasticBendingDeformableParticleModel, cls).Create(
            vertices=vertices,
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=element_adjacency_edges,
            elements_id=elements_id,
            initial_body_contents=initial_body_contents,
            initial_element_measures=initial_element_measures,
            initial_edge_lengths=initial_edge_lengths,
            initial_bendings=initial_bendings,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
            w_b=w_b,
            vertex_count=vertex_count,
        )

        tau_s_arr = None
        if base.eb is not None and tau_s is not None:
            assert base.element_adjacency is not None
            tau_s_arr = _broadcast_tau_s(tau_s, base.element_adjacency.shape[:-1])

        return dataclasses.replace(base, tau_s=tau_s_arr)

    @classmethod
    def _merge_extra_fields(
        cls,
        current: DeformableParticleModel,
        nxt: DeformableParticleModel,
    ) -> dict[str, Any]:
        current = cast(PlasticBendingDeformableParticleModel, current)
        nxt = cast(PlasticBendingDeformableParticleModel, nxt)
        n_adj_cur = (
            0
            if current.element_adjacency is None
            else int(current.element_adjacency.shape[0])
        )
        n_adj_new = (
            0 if nxt.element_adjacency is None else int(nxt.element_adjacency.shape[0])
        )
        # Use inf as fill so that 1/tau_s = 0 (no plasticity) for padded hinges.
        return {
            "tau_s": _merge_metric_field(
                current.tau_s,
                nxt.tau_s,
                n_adj_cur,
                n_adj_new,
                float("inf"),
            )
        }

    def update_reference_state(
        self,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> DeformableParticleModel:
        """Relax each reference bending angle toward the current bending angle."""
        if (
            self.eb is None
            or self.initial_bendings is None
            or self.w_b is None
            or self.tau_s is None
        ):
            return self

        theta = current_bending_angles(self, pos, state)
        if theta is None:
            return self

        theta_0 = self.initial_bendings
        new_initial_bendings = _relax_reference(
            theta_0, theta_0 + _angle_difference(theta, theta_0), system.dt, self.tau_s
        )
        return dataclasses.replace(self, initial_bendings=new_initial_bendings)
