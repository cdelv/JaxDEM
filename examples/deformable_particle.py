"""
Deformable Particle Model (Bonded Forces)
-----------------------------------------

This example demonstrates the use of deformable particle model through the bonded_force_model API:
- Creating deformable models with ``jdem.BonndedForceModel.create(...)``
- Adding a new body with ``add``
- Merging existing models with ``merge``
- Stacking / unstacking models for batched workflows
- Running parallel simulations with ``jax.vmap``
- Two ways to pass a bonded model into ``System.create``:
  1) pass the model object directly
  2) pass the registered type + kwargs

Notes on constructor behavior:
- Scalar coefficients are broadcast to the correct target shapes.
- ``ec`` is special: it is per-body (shape ``(K,)``), not per-element.
  Body mapping is controlled by ``elements_ID``.
- ``create`` stores only data needed by active terms.

Notes on merge behavior:
- ``merge`` concatenates topology/reference arrays.
- When one side has a term and the other does not, missing coefficients are padded with ``0``
  and missing reference values are padded with ``1``.
- For content terms, body IDs are shifted so merged ``elements_ID`` remains consistent.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

import jaxdem as jdem
from jaxdem.bonded_forces.deformable_particle import DeformableParticleModel

# A small closed 2D boundary mesh (square perimeter as segments)
VERTS = jnp.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=float,
)
ELEMENTS = jnp.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ],
    dtype=int,
)
# For a ring in 2D, each segment is adjacent to the next segment.
ADJ = jnp.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ],
    dtype=int,
)
EDGES = ELEMENTS


def build_state() -> jdem.State:
    return jdem.State.create(pos=VERTS)


def demo_create_add_merge_stack() -> None:
    # Body 1: edge + measure + surface terms
    dp1 = jdem.BonndedForceModel.create(
        "deformableparticlemodel",
        vertices=VERTS,
        elements=ELEMENTS,
        edges=EDGES,
        em=2.0,
        el=0.3,
        gamma=0.2,
    )
    dp1 = cast(DeformableParticleModel, dp1)

    # Body 2: bending + content terms (ec is per-body, tied to elements_ID)
    dp2 = jdem.BonndedForceModel.create(
        "deformableparticlemodel",
        vertices=VERTS,
        elements=ELEMENTS,
        element_adjacency=ADJ,
        elements_ID=jnp.zeros((ELEMENTS.shape[0],), dtype=int),
        eb=1.0,
        ec=0.05,
    )
    dp2 = cast(DeformableParticleModel, dp2)

    # merge pads missing terms: coefficients with 0, references with 1
    dp_cls = type(dp1)
    merged = dp_cls.merge(dp1, dp2)

    # add == create(new) + merge(existing, new)
    added = dp_cls.add(
        dp1,
        vertices=VERTS,
        elements=ELEMENTS,
        elements_ID=jnp.zeros((ELEMENTS.shape[0],), dtype=int),
        ec=0.1,
    )

    # stack/unstack in State style
    stacked = dp_cls.stack([dp1, dp1])
    unstacked = dp_cls.unstack(stacked)

    print("Create/Add/Merge/Stack demo")
    print(
        " merged elements:", None if merged.elements is None else merged.elements.shape
    )
    print(" added ec shape:", None if added.ec is None else added.ec.shape)
    print(" unstacked length:", len(unstacked))


def demo_system_two_bonded_options() -> tuple[jdem.State, jdem.System, jdem.System]:
    state = build_state()

    # Option 1: pass bonded model object directly
    dp_obj = jdem.BonndedForceModel.create(
        "deformableparticlemodel",
        vertices=state.pos,
        elements=ELEMENTS,
        element_adjacency=ADJ,
        edges=EDGES,
        em=1.0,
        eb=1.0,
        ec=0.1,
        el=0.2,
        gamma=0.05,
    )
    system_obj = jdem.System.create(
        state.shape,
        bonded_force_model=dp_obj,
    )

    # Option 2: pass registered type + kwargs
    system_type = jdem.System.create(
        state.shape,
        bonded_force_model_type="deformableparticlemodel",
        bonded_force_manager_kw=dict(
            vertices=state.pos,
            elements=ELEMENTS,
            element_adjacency=ADJ,
            edges=EDGES,
            em=1.0,
            eb=1.0,
            ec=0.1,
            el=0.2,
            gamma=0.05,
        ),
    )

    return state, system_obj, system_type


def demo_parallel_simulation() -> None:
    def create_one(_i: jax.Array) -> tuple[jdem.State, jdem.System]:
        state = build_state()
        dp_model = jdem.BonndedForceModel.create(
            "deformableparticlemodel",
            vertices=state.pos,
            elements=ELEMENTS,
            element_adjacency=ADJ,
            edges=EDGES,
            em=[1.0],
            eb=[1.0],
            ec=0.1,
            el=0.2,
            gamma=0.05,
        )
        system = jdem.System.create(
            state.shape,
            bonded_force_model=dp_model,
        )
        return state, system

    # Build a batch of independent state/system pairs
    states, systems = jax.vmap(create_one)(jnp.arange(4))

    # Step all of them in parallel
    states, systems = systems.step(states, systems)

    print("Parallel simulation demo")
    print(" states shape:", states.shape)
    print(" bonded model type:", type(systems.bonded_force_model).__name__)


if __name__ == "__main__":
    demo_create_add_merge_stack()
    state, sys_obj, sys_type = demo_system_two_bonded_options()
    print("Two System.create options demo")
    print(" object path model:", type(sys_obj.bonded_force_model).__name__)
    print(" type+kwargs path model:", type(sys_type.bonded_force_model).__name__)
    demo_parallel_simulation()
