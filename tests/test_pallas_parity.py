# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Drift-prevention, activation, and numerical parity tests for Pallas Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.colliders._pallas import (
    can_use_pallas_csr_forces,
    compile_displacement_spec,
    compile_force_spec,
    extract_grid_topology,
    is_pallas_enabled,
    pallas_enabled,
    set_pallas_enabled,
)
from jaxdem.colliders._pallas.refs import ArgPack
from jaxdem.domains import Domain, SearchGeometry
from jaxdem.forces.cundall_strack import (
    CundallStrackForce,
    arithmetic_mean_friction,
    geometric_mean_friction,
    maximum_friction,
    minimum_friction,
)
from jaxdem.forces.hertz import HertzianForce
from jaxdem.forces.law_combiner import LawCombiner
from jaxdem.forces.lennardjones import LennardJones
from jaxdem.forces.router import ForceRouter
from jaxdem.forces.spring import DampedSpringForce, SpringForce
from jaxdem.forces.wca import WCA
from jaxdem.forces.wca_shifted import WCAShifted


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _AllPropsMaterial(jdem.Material):
    young: float
    poisson: float
    e: float
    mu: float
    mu_r: float
    k_n: float
    k_t: float
    b_n: float
    b_t: float
    epsilon: float


def _make_materials(n_mat: int = 2) -> jdem.MaterialTable:
    mats = [
        _AllPropsMaterial(
            density=1.0,
            young=1.0 + 0.3 * m,
            poisson=0.25 + 0.05 * m,
            e=0.6 + 0.1 * m,
            mu=0.3 + 0.1 * m,
            mu_r=0.05 + 0.02 * m,
            k_n=10.0 + 2.0 * m,
            k_t=6.0 + 1.5 * m,
            b_n=0.4 + 0.1 * m,
            b_t=0.2 + 0.05 * m,
            epsilon=0.5 + 0.2 * m,
        )
        for m in range(n_mat)
    ]
    return jdem.MaterialTable.from_materials(
        mats,
        matcher=jdem.MaterialMatchmaker.create("harmonic"),
    )


def _make_state(
    dim: int,
    n: int,
    box_l: float,
    *,
    n_mat: int = 2,
    n_species: int = 1,
    clumps: bool = False,
    seed: int = 0,
) -> jdem.State:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1.0 / dim)))
    spacing = box_l / side
    axes = [np.linspace(0.5 * spacing, box_l - 0.5 * spacing, side) for _ in range(dim)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    pos = grid[:n] + rng.uniform(-0.18 * spacing, 0.18 * spacing, size=(n, dim))
    pos = np.clip(pos, 0.08 * box_l, 0.92 * box_l)
    r_base = 0.56 * spacing
    rad = rng.uniform(0.92 * r_base, 1.08 * r_base, size=(n,))
    vel = rng.normal(0.0, 0.25, size=(n, dim))
    ang_dim = 3 if dim == 3 else 1
    ang_vel = rng.normal(0.0, 0.4, size=(n, ang_dim))
    mat_id = rng.integers(0, n_mat, size=(n,))
    species_id = rng.integers(0, n_species, size=(n,))
    kw: dict[str, jax.Array] = dict(
        pos=jnp.asarray(pos),
        rad=jnp.asarray(rad),
        vel=jnp.asarray(vel),
        ang_vel=jnp.asarray(ang_vel),
        mat_id=jnp.asarray(mat_id, dtype=int),
        species_id=jnp.asarray(species_id, dtype=int),
    )
    if clumps:
        kw["clump_id"] = jnp.asarray(np.arange(n) // 2, dtype=int)
    return jdem.State.create(**kw)


def _make_domain(kind: str, dim: int, box_l: float) -> jdem.Domain:
    box = jnp.full((dim,), box_l)
    if kind == "free":
        return jdem.Domain.create("free", dim=dim, box_size=box)
    if kind == "reflect":
        return jdem.Domain.create("reflect", dim=dim, box_size=box)
    if kind == "reflect_sphere":
        return jdem.Domain.create("reflect_sphere", dim=dim, box_size=box)
    if kind == "periodic":
        return jdem.Domain.create(
            "periodic", dim=dim, box_size=box, box_rate=jnp.full((dim,), 0.02)
        )
    if kind == "lees_edwards":
        return jdem.Domain.create(
            "leesedwards",
            dim=dim,
            box_size=box,
            box_rate=jnp.full((dim,), 0.01),
            gamma=0.27,
            gamma_dot=0.15,
        )
    raise ValueError(kind)


def _builtin_force_models() -> list[tuple[str, jdem.ForceModel, int]]:
    cs_elastic = CundallStrackForce(
        parameterization="elastic",
        friction_mixing=minimum_friction,
        rolling_friction_mixing=maximum_friction,
    )
    cs_coeff = CundallStrackForce(
        parameterization="coefficients",
        friction_mixing=arithmetic_mean_friction,
        rolling_friction_mixing=geometric_mean_friction,
    )
    combiner_stateless = LawCombiner(laws=(SpringForce(), WCA()))
    combiner_history = LawCombiner(laws=(cs_elastic, WCA()))
    router_stateless = ForceRouter(
        table=(
            (SpringForce(), HertzianForce()),
            (HertzianForce(), WCAShifted()),
        )
    )
    router_history = ForceRouter(
        table=(
            (cs_elastic, combiner_history),
            (combiner_history, cs_coeff),
        )
    )
    return [
        ("spring", SpringForce(), 1),
        ("damped_spring", DampedSpringForce(), 1),
        ("hertz", HertzianForce(), 1),
        ("wca", WCA(), 1),
        ("wca_shifted", WCAShifted(), 1),
        ("lj", LennardJones.Create(cutoff_ratio=1.6), 1),
        ("cundall_elastic", cs_elastic, 1),
        ("cundall_coeff", cs_coeff, 1),
        ("combiner_stateless", combiner_stateless, 1),
        ("combiner_history", combiner_history, 1),
        ("router_stateless", router_stateless, 2),
        ("router_history", router_history, 2),
    ]


def _assert_close_pair(a: jax.Array, b: jax.Array, name: str) -> None:
    is_f64 = a.dtype == jnp.float64
    rtol = 1e-11 if is_f64 else 1e-3
    atol = 1e-11 if is_f64 else 2e-4
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=name
    )


def test_schema_and_extractors_activate_for_all_builtins() -> None:
    """Every built-in domain & force law must compile into Pallas specs (no silent fallback)."""
    mat_table = _make_materials(2)
    for dim in (2, 3):
        st_dom = _make_state(dim, 16, 6.0, n_mat=1)
        for dom_kind in (
            "free",
            "reflect",
            "reflect_sphere",
            "periodic",
            "lees_edwards",
        ):
            dom = _make_domain(dom_kind, dim, 6.0)
            sy_dom = jdem.System.create(
                state_shape=st_dom.shape,
                dt=1e-3,
                domain=dom,
                force_model=SpringForce(),
                mat_table=mat_table,
            )
            assert extract_grid_topology(dom, dim) is not None
            assert (
                compile_displacement_spec(sy_dom, dim, st_dom.pos.dtype, pack=ArgPack())
                is not None
            )

        for label, law, n_sp in _builtin_force_models():
            # Use N=100 (> _MAX_REG_ELEMS) with clumps=True (only_spheres=False)
            # so any accidental O(N) rotation refresh or whole-array op fails loudly.
            st = _make_state(dim, 100, 6.0, n_mat=2, n_species=n_sp, clumps=True)
            sy = jdem.System.create(
                state_shape=st.shape,
                dt=1e-3,
                domain_type="periodic",
                domain_kw={"box_size": jnp.full((dim,), 6.0)},
                force_model=law,
                mat_table=mat_table,
                collider_type="NeighborList",
                collider_kw={"state": st, "cutoff": 1.8, "skin": 0.3},
            )
            assert (
                compile_force_spec(
                    law,
                    st,
                    sy,
                    st.pos,
                    advance_history=True,
                    quasistatic=False,
                    pack=ArgPack(),
                )
                is not None
            ), f"Force spec failed for {label} {dim}D"
            if is_pallas_enabled():
                assert can_use_pallas_csr_forces(
                    st, sy, advance_history=True, quasistatic=False
                ), f"CSR activation failed for {label} {dim}D"


def test_runtime_toggle_api() -> None:
    """``set_pallas_enabled`` and ``pallas_enabled`` cleanly toggle on supported GPUs."""
    initial = is_pallas_enabled()
    with pallas_enabled(False, clear_caches=False):
        assert is_pallas_enabled() is False
    assert is_pallas_enabled() is initial
    set_pallas_enabled(None, clear_caches=False)
    assert is_pallas_enabled() is initial


def _eval_force_once_or_twice(
    st: jdem.State, sy: jdem.System
) -> tuple[jdem.State, jdem.System]:
    """Evaluate collider force (twice for stateful laws so non-zero history is exercised)."""
    st1, sy1 = sy.collider.compute_force(st, sy)
    if getattr(sy1.collider, "history", jnp.empty((0,))).size > 0:
        st1 = replace(st1, pos_c=st1.pos_c + st1.vel * sy1.dt)
        st1, sy1 = sy1.collider.compute_force(st1, sy1)
    return st1, sy1


def test_pallas_vs_pure_jax_all_forces_neighbor_list() -> None:
    """Parity across distinct built-in force law families (direct force + multi-step remap)."""
    if not is_pallas_enabled():
        pytest.skip("Pallas Triton GPU backend not active on this host")

    mat = _make_materials(2)
    # test_schema_and_extractors_activate_for_all_builtins already compiles all
    # 12 laws in 2D & 3D. Execute one representative per Triton codegen family:
    # central (spring), velocity-dependent (damped_spring), non-trivial cutoff
    # (lj), stateful+cond+torque (cundall_elastic), and router+combiner+both
    # Cundall parameterizations+WCA (router_history).
    laws_3d = {
        "spring",
        "damped_spring",
        "lj",
        "cundall_elastic",
        "router_history",
    }
    laws_2d = {"spring", "router_history"}
    force_cases: list[tuple[str, jdem.State, jdem.System]] = []
    step_cases: list[tuple[str, jdem.State, jdem.System]] = []
    for dim in (3, 2):
        box_l = 4.2 if dim == 3 else 5.2
        wanted = laws_3d if dim == 3 else laws_2d
        for label, law, n_sp in _builtin_force_models():
            if label not in wanted:
                continue
            st = _make_state(dim, 43, box_l, n_mat=2, n_species=n_sp, seed=10 + dim)
            sy = jdem.System.create(
                state_shape=st.shape,
                dt=2e-3,
                domain_type="periodic",
                domain_kw={"box_size": jnp.full((dim,), box_l)},
                force_model=law,
                mat_table=mat,
                collider_type="NeighborList",
                collider_kw={"state": st, "cutoff": 1.6, "skin": 0.15},
            )
            name = f"{label}_{dim}D"
            if name in ("spring_3D", "cundall_elastic_3D"):
                step_cases.append((name, st, sy))
            else:
                force_cases.append((name, st, sy))

    # Run all Pallas cases together, then clear cache once and run all Pure-JAX cases.
    with pallas_enabled(True):
        pallas_out = {
            name: _eval_force_once_or_twice(st, sy) for name, st, sy in force_cases
        }
        pallas_step = {
            name: jdem.System.step(st, sy, n=6) for name, st, sy in step_cases
        }
    with pallas_enabled(False):
        pure_out = {
            name: _eval_force_once_or_twice(st, sy) for name, st, sy in force_cases
        }
        pure_step = {name: jdem.System.step(st, sy, n=6) for name, st, sy in step_cases}

    for name, _, _ in force_cases:
        st_p, sy_p = pallas_out[name]
        st_j, sy_j = pure_out[name]
        _assert_close_pair(st_p.force, st_j.force, f"{name} force")
        _assert_close_pair(st_p.torque, st_j.torque, f"{name} torque")
        assert int(sy_p.collider.n_build_times) == int(sy_j.collider.n_build_times)

    for name, _, _ in step_cases:
        st_p, sy_p = pallas_step[name]
        st_j, sy_j = pure_step[name]
        _assert_close_pair(st_p.pos_c, st_j.pos_c, f"{name} step pos_c")
        _assert_close_pair(st_p.vel, st_j.vel, f"{name} step vel")
        _assert_close_pair(st_p.force, st_j.force, f"{name} step force")
        _assert_close_pair(st_p.torque, st_j.torque, f"{name} step torque")
        assert int(sy_p.collider.n_build_times) == int(sy_j.collider.n_build_times)


def test_pallas_vs_pure_jax_all_domains_cell_and_nl() -> None:
    """Parity across all 5 built-in domains (2D & 3D, clumps) for CellList, StaticCellList, NL."""
    if not is_pallas_enabled():
        pytest.skip("Pallas Triton GPU backend not active on this host")

    mat = _make_materials(2)
    force_cases: list[tuple[str, jdem.State, jdem.System]] = []
    step_cases: list[tuple[str, jdem.State, jdem.System]] = []
    # Cover every domain topology (non-periodic, periodic, shear-periodic) and
    # collider family (CellList, StaticCellList, NeighborList) with clumps.
    matrix: list[tuple[int, str, list[tuple[str, str, dict[str, object]]]]] = [
        (3, "free", [("cell", "CellList", {"cell_size": 1.3})]),
        (
            3,
            "reflect",
            [
                ("static_cell", "CellList", {"cell_size": 1.3, "static": True}),
                ("nl", "NeighborList", {"cutoff": 1.3, "skin": 0.2}),
            ],
        ),
        (3, "reflect_sphere", [("cell", "CellList", {"cell_size": 1.3})]),
        (
            3,
            "periodic",
            [
                ("cell", "CellList", {"cell_size": 1.3}),
                ("static_cell", "CellList", {"cell_size": 1.3, "static": True}),
                ("nl", "NeighborList", {"cutoff": 1.3, "skin": 0.2}),
            ],
        ),
        (
            3,
            "lees_edwards",
            [
                ("cell", "CellList", {"cell_size": 1.3}),
                ("nl", "NeighborList", {"cutoff": 1.3, "skin": 0.2}),
            ],
        ),
        (
            2,
            "periodic",
            [
                ("cell", "CellList", {"cell_size": 1.3}),
                ("nl", "NeighborList", {"cutoff": 1.3, "skin": 0.2}),
            ],
        ),
        (
            2,
            "lees_edwards",
            [
                ("cell", "CellList", {"cell_size": 1.3}),
                ("nl", "NeighborList", {"cutoff": 1.3, "skin": 0.2}),
            ],
        ),
    ]
    states = {
        dim: _make_state(
            dim, 37, 4.5 if dim == 3 else 5.5, n_mat=2, clumps=True, seed=21 + dim
        )
        for dim in (3, 2)
    }
    for dim, dom_kind, col_specs in matrix:
        box_l = 4.5 if dim == 3 else 5.5
        st = states[dim]
        dom = _make_domain(dom_kind, dim, box_l)
        for tag, col_type, extra_kw in col_specs:
            sy = jdem.System.create(
                state_shape=st.shape,
                dt=2e-3,
                domain=dom,
                force_model=DampedSpringForce(),
                mat_table=mat,
                collider_type=col_type,
                collider_kw={"state": st, **extra_kw},
            )
            name = f"{dom_kind}_{tag}_{dim}D"
            if dim == 3 and dom_kind == "lees_edwards":
                step_cases.append((name, st, sy))
            else:
                force_cases.append((name, st, sy))

    with pallas_enabled(True):
        pallas_out = {
            name: sy.collider.compute_force(st, sy) for name, st, sy in force_cases
        }
        pallas_step = {
            name: jdem.System.step(st, sy, n=5) for name, st, sy in step_cases
        }
    with pallas_enabled(False):
        pure_out = {
            name: sy.collider.compute_force(st, sy) for name, st, sy in force_cases
        }
        pure_step = {name: jdem.System.step(st, sy, n=5) for name, st, sy in step_cases}

    for name, _, _ in force_cases:
        st_p, _ = pallas_out[name]
        st_j, _ = pure_out[name]
        _assert_close_pair(st_p.force, st_j.force, f"{name} force")
        _assert_close_pair(st_p.torque, st_j.torque, f"{name} torque")

    for name, _, _ in step_cases:
        st_p, _ = pallas_step[name]
        st_j, _ = pure_step[name]
        _assert_close_pair(st_p.pos_c, st_j.pos_c, f"{name} step pos_c")
        _assert_close_pair(st_p.force, st_j.force, f"{name} step force")
        _assert_close_pair(st_p.torque, st_j.torque, f"{name} step torque")


def test_pallas_vs_pure_jax_batched_vmap_and_ad() -> None:
    """Verify 2D batched vmap (non-pow2 B & N) and reverse/forward AD parity."""
    if not is_pallas_enabled():
        pytest.skip("Pallas Triton GPU backend not active on this host")

    B, N, dim = 7, 29, 3
    box_l = 4.0
    mat2 = _make_materials(2)
    states = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[_make_state(dim, N, box_l, n_mat=2, seed=100 + b) for b in range(B)],
    )
    sy0 = jdem.System.create(
        state_shape=(N, dim),
        dt=2e-3,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full((dim,), box_l)},
        force_model=CundallStrackForce(),
        mat_table=mat2,
        collider_type="NeighborList",
        collider_kw={
            "state": jax.tree.map(lambda x: x[0], states),
            "cutoff": 1.3,
            "skin": 0.15,
        },
    )
    systems = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None, ...], (B,) + x.shape), sy0
    )

    st1 = _make_state(3, 25, 3.8, n_mat=1, seed=55)
    mat1 = _make_materials(1)
    ad_systems = {
        "static_cell": jdem.System.create(
            state_shape=st1.shape,
            dt=1e-3,
            domain_type="periodic",
            domain_kw={"box_size": jnp.full((3,), 3.8)},
            force_model=SpringForce(),
            mat_table=mat1,
            collider_type="CellList",
            collider_kw={"state": st1, "cell_size": 1.3, "static": True},
        ),
        "nl": jdem.System.create(
            state_shape=st1.shape,
            dt=1e-3,
            domain_type="periodic",
            domain_kw={"box_size": jnp.full((3,), 3.8)},
            force_model=SpringForce(),
            mat_table=mat1,
            collider_type="NeighborList",
            collider_kw={"state": st1, "cutoff": 1.3, "skin": 0.2},
        ),
    }
    sy_dyn = jdem.System.create(
        state_shape=st1.shape,
        dt=1e-3,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full((3,), 3.8)},
        force_model=SpringForce(),
        mat_table=mat1,
        collider_type="CellList",
        collider_kw={"state": st1, "cell_size": 1.3},
    )
    tangent = jnp.ones_like(st1.pos_c) * 1e-2

    def _run_suite() -> tuple[jdem.State, dict[str, jax.Array], jax.Array]:
        step_b = jax.jit(jax.vmap(lambda s, sy: jdem.System.step(s, sy, n=5)))
        st_b, _ = step_b(states, systems)
        grads = {}
        for k, sy1 in ad_systems.items():
            grads[k] = jax.grad(
                lambda pc: jnp.sum(
                    sy1.collider.compute_force(replace(st1, pos_c=pc), sy1)[0].force
                    ** 2
                )
            )(st1.pos_c)
        _, jvp_out = jax.jvp(
            lambda pc: (
                sy_dyn.collider.compute_force(replace(st1, pos_c=pc), sy_dyn)[0].force
            ),
            (st1.pos_c,),
            (tangent,),
        )
        return st_b, grads, jvp_out

    with pallas_enabled(True):
        st_p, g_p, jvp_p = _run_suite()
    with pallas_enabled(False):
        st_j, g_j, jvp_j = _run_suite()

    _assert_close_pair(st_p.pos_c, st_j.pos_c, "batched NL+CS pos_c")
    _assert_close_pair(st_p.force, st_j.force, "batched NL+CS force")
    _assert_close_pair(st_p.torque, st_j.torque, "batched NL+CS torque")
    for k in ad_systems:
        _assert_close_pair(g_p[k], g_j[k], f"grad {k}")
    _assert_close_pair(jvp_p, jvp_j, "jvp DynamicCellList")


@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class _CustomMorseForce(jdem.ForceModel):
    """User-defined central force law unknown to Pallas."""

    depth: jax.Array
    alpha: jax.Array

    @property
    def has_pair_torque(self) -> bool:
        return False

    @property
    def supports_sorted_order(self) -> bool:
        return True

    def search_radii(self, state: jdem.State, system: jdem.System) -> jax.Array:
        return state.rad

    def force(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        from jaxdem.utils import linalg

        rij = system.domain.displacement(pos[i], pos[j], system)
        n, r = linalg.unit_and_norm(rij)
        R = state.rad[i] + state.rad[j]
        delta = jnp.maximum(0.0, R - r)
        e = jnp.exp(-self.alpha * (r - R))
        mag = 2.0 * self.depth * self.alpha * (e * e - e)
        f = jnp.where((r > 0.0) & (delta > 0.0) & (i != j), mag, 0.0)[..., None] * n
        torque = jnp.zeros(
            (*f.shape[:-1], state.torque.shape[-1]), dtype=state.torque.dtype
        )
        return f, torque, history

    def energy(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
    ) -> jax.Array:
        return jnp.asarray(0.0, dtype=pos.dtype)


@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class _CustomHistoryTorqueForce(jdem.ForceModel):
    """User-defined stateful force law with contact history and pair torque."""

    k_n: jax.Array
    k_t: jax.Array

    @property
    def has_pair_torque(self) -> bool:
        return True

    @property
    def supports_sorted_order(self) -> bool:
        return True

    def search_radii(self, state: jdem.State, system: jdem.System) -> jax.Array:
        return state.rad

    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (dim,)

    def force(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        from jaxdem.utils import linalg

        rij = system.domain.displacement(pos[i], pos[j], system)
        n, r = linalg.unit_and_norm(rij)
        R = state.rad[i] + state.rad[j]
        delta = jnp.maximum(0.0, R - r)
        contact = ((r > 0.0) & (delta > 0.0) & (i != j))[..., None]

        dv = system.domain.relative_velocity(
            pos[i], pos[j], state.vel[i], state.vel[j], system
        )
        vt = dv - linalg.dot(dv, n)[..., None] * n
        h_proj = history - linalg.dot(history, n)[..., None] * n
        h_next = jnp.where(
            contact,
            h_proj + vt * system.dt if advance_history else h_proj,
            jnp.zeros_like(history),
        )
        f_n = (self.k_n * delta)[..., None] * n
        f_t = -self.k_t * h_next
        f = jnp.where(contact, f_n + f_t, jnp.zeros_like(rij))
        zero_tau = jnp.zeros(
            (*f.shape[:-1], state.torque.shape[-1]), dtype=state.torque.dtype
        )
        tau = jnp.where(
            contact,
            linalg.cross(-state.rad[i][..., None] * n, f_t),
            zero_tau,
        )
        return f, tau, h_next

    def energy(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
    ) -> jax.Array:
        return jnp.asarray(0.0, dtype=pos.dtype)


@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class _UnsupportedSortForce(jdem.ForceModel):
    """Custom force using jnp.sort to verify graceful Pure-JAX fallback."""

    @property
    def has_pair_torque(self) -> bool:
        return False

    @property
    def supports_sorted_order(self) -> bool:
        return True

    def search_radii(self, state: jdem.State, system: jdem.System) -> jax.Array:
        return state.rad

    def force(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rij = system.domain.displacement(pos[i], pos[j], system)
        s = jnp.sort(jnp.abs(rij), axis=-1)
        f = 0.01 * s * (i != j)[..., None]
        torque = jnp.zeros(
            (*f.shape[:-1], state.torque.shape[-1]), dtype=state.torque.dtype
        )
        return f, torque, history

    def energy(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
    ) -> jax.Array:
        return jnp.asarray(0.0, dtype=pos.dtype)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _CustomPeriodicDomain(Domain):
    """User-defined orthogonal periodic domain unknown to Pallas."""

    search_geometry = SearchGeometry.ORTHOGONAL

    @property
    def periodic(self) -> bool:
        return True

    @staticmethod
    @jax.jit(inline=True)
    def _displacement(ri: jax.Array, rj: jax.Array, system: jdem.System) -> jax.Array:
        dr = ri - rj
        box = system.domain.box_size
        return dr - box * jnp.rint(dr * system.domain.inv_box_size)

    @staticmethod
    @jax.jit(inline=True)
    def displacement(ri: jax.Array, rj: jax.Array, system: jdem.System) -> jax.Array:
        return _CustomPeriodicDomain._displacement(ri, rj, system)

    @staticmethod
    @jax.jit(inline=True)
    def _shift(pos: jax.Array, system: jdem.System) -> jax.Array:
        box = system.domain.box_size
        anchor = system.domain.anchor
        return anchor + jnp.mod(pos - anchor, box)

    @staticmethod
    @jax.jit(inline=True)
    def apply(state: jdem.State, system: jdem.System) -> tuple[jdem.State, jdem.System]:
        box = system.domain.box_size
        anchor = system.domain.anchor
        return replace(state, pos_c=anchor + jnp.mod(state.pos_c - anchor, box)), system


def test_pallas_custom_force_laws_and_fallback() -> None:
    """Arbitrary user ForceModels and Domains compile to Pallas & match Pure-JAX."""
    mat = _make_materials(2)
    morse = _CustomMorseForce(depth=jnp.asarray(2.0), alpha=jnp.asarray(1.5))
    hist_law = _CustomHistoryTorqueForce(k_n=jnp.asarray(15.0), k_t=jnp.asarray(5.0))
    mixed_comb = LawCombiner(laws=(SpringForce(), morse))
    mixed_router = ForceRouter(
        table=(
            (SpringForce(), morse),
            (morse, hist_law),
        )
    )
    unsupported = _UnsupportedSortForce()

    st_check = _make_state(3, 19, 4.0, n_mat=2, n_species=2, seed=7)
    for law in (morse, hist_law, mixed_comb, mixed_router):
        sy = jdem.System.create(
            state_shape=st_check.shape,
            dt=1e-3,
            domain_type="periodic",
            domain_kw={"box_size": jnp.full((3,), 4.0)},
            force_model=law,
            mat_table=mat,
            collider_type="NeighborList",
            collider_kw={"state": st_check, "cutoff": 1.6, "skin": 0.2},
        )
        assert (
            compile_force_spec(
                law,
                st_check,
                sy,
                st_check.pos,
                advance_history=True,
                quasistatic=False,
                pack=ArgPack(),
            )
            is not None
        ), f"Custom law failed to compile: {type(law).__name__}"

    sy_unsup = jdem.System.create(
        state_shape=st_check.shape,
        dt=1e-3,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full((3,), 4.0)},
        force_model=unsupported,
        mat_table=mat,
        collider_type="NeighborList",
        collider_kw={"state": st_check, "cutoff": 1.6, "skin": 0.2},
    )
    assert (
        compile_force_spec(
            unsupported,
            st_check,
            sy_unsup,
            st_check.pos,
            advance_history=True,
            quasistatic=False,
            pack=ArgPack(),
        )
        is None
    )

    if not is_pallas_enabled():
        pytest.skip("Pallas Triton GPU backend not active on this host")

    assert not can_use_pallas_csr_forces(
        st_check, sy_unsup, advance_history=True, quasistatic=False
    )

    cases: list[tuple[str, jdem.State, jdem.System]] = []
    st_3d = _make_state(3, 35, 4.2, n_mat=2, n_species=2, seed=43)
    st_2d = _make_state(2, 35, 5.2, n_mat=2, n_species=2, seed=42)
    cases.append(
        (
            "morse_cell_3D",
            st_3d,
            jdem.System.create(
                state_shape=st_3d.shape,
                dt=2e-3,
                domain_type="periodic",
                domain_kw={"box_size": jnp.full((3,), 4.2)},
                force_model=morse,
                mat_table=mat,
                collider_type="CellList",
                collider_kw={"state": st_3d, "cell_size": 1.5},
            ),
        )
    )
    cases.append(
        (
            "custom_hist_le_2D",
            st_2d,
            jdem.System.create(
                state_shape=st_2d.shape,
                dt=2e-3,
                domain=_make_domain("lees_edwards", 2, 5.2),
                force_model=hist_law,
                mat_table=mat,
                collider_type="NeighborList",
                collider_kw={"state": st_2d, "cutoff": 1.6, "skin": 0.15},
            ),
        )
    )
    cases.append(
        (
            "mixed_router_3D",
            st_3d,
            jdem.System.create(
                state_shape=st_3d.shape,
                dt=2e-3,
                domain_type="periodic",
                domain_kw={"box_size": jnp.full((3,), 4.2)},
                force_model=mixed_router,
                mat_table=mat,
                collider_type="NeighborList",
                collider_kw={"state": st_3d, "cutoff": 1.6, "skin": 0.15},
            ),
        )
    )
    step_custom = (
        st_3d,
        jdem.System.create(
            state_shape=st_3d.shape,
            dt=2e-3,
            domain=_make_domain("lees_edwards", 3, 4.2),
            force_model=hist_law,
            mat_table=mat,
            collider_type="NeighborList",
            collider_kw={"state": st_3d, "cutoff": 1.6, "skin": 0.15},
        ),
    )
    # Custom Domain subclass (unknown to Pallas) must compile for both CellList & NL.
    custom_dom_3d = _CustomPeriodicDomain.Create(dim=3, box_size=jnp.full((3,), 4.2))
    st_cd = _make_state(3, 31, 4.2, n_mat=2, seed=77)
    sy_cd_nl = jdem.System.create(
        state_shape=st_cd.shape,
        dt=2e-3,
        domain=custom_dom_3d,
        force_model=morse,
        mat_table=mat,
        collider_type="NeighborList",
        collider_kw={"state": st_cd, "cutoff": 1.6, "skin": 0.15},
    )
    sy_cd_cell = jdem.System.create(
        state_shape=st_cd.shape,
        dt=2e-3,
        domain=custom_dom_3d,
        force_model=morse,
        mat_table=mat,
        collider_type="CellList",
        collider_kw={"state": st_cd, "cell_size": 1.5},
    )
    assert extract_grid_topology(custom_dom_3d, 3) is not None
    assert (
        compile_displacement_spec(sy_cd_nl, 3, st_cd.pos.dtype, pack=ArgPack())
        is not None
    )
    cases.append(("custom_domain_nl_3D", st_cd, sy_cd_nl))
    cases.append(("custom_domain_cell_3D", st_cd, sy_cd_cell))

    # Fallback law must also match Pure-JAX seamlessly.
    cases.append(("unsupported_fallback_3D", st_check, sy_unsup))

    # Batched vmap with per-replica varying custom force parameters + AD.
    B, N = 5, 23
    states_b = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[_make_state(3, N, 4.0, n_mat=2, seed=200 + b) for b in range(B)],
    )
    sy_b_list = [
        jdem.System.create(
            state_shape=(N, 3),
            dt=2e-3,
            domain_type="periodic",
            domain_kw={"box_size": jnp.full((3,), 4.0)},
            force_model=_CustomMorseForce(
                depth=jnp.asarray(1.0 + 0.5 * b), alpha=jnp.asarray(1.2 + 0.1 * b)
            ),
            mat_table=mat,
            collider_type="NeighborList",
            collider_kw={
                "state": jax.tree.map(lambda x: x[b], states_b),
                "cutoff": 1.5,
                "skin": 0.15,
            },
        )
        for b in range(B)
    ]
    systems_b = jax.tree.map(lambda *xs: jnp.stack(xs), *sy_b_list)
    st_ad = _make_state(3, 21, 3.8, n_mat=1, seed=99)
    sy_ad = jdem.System.create(
        state_shape=st_ad.shape,
        dt=1e-3,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full((3,), 3.8)},
        force_model=morse,
        mat_table=_make_materials(1),
        collider_type="NeighborList",
        collider_kw={"state": st_ad, "cutoff": 1.5, "skin": 0.2},
    )

    def _run_all() -> tuple[
        dict[str, tuple[jdem.State, jdem.System]],
        tuple[jdem.State, jdem.System],
        jdem.State,
        jax.Array,
    ]:
        out = {name: _eval_force_once_or_twice(st, sy) for name, st, sy in cases}
        st_step = jdem.System.step(step_custom[0], step_custom[1], n=5)
        st_v, _ = jax.jit(jax.vmap(lambda s, sy: sy.collider.compute_force(s, sy)))(
            states_b, systems_b
        )
        g = jax.grad(
            lambda d: jnp.sum(
                sy_ad.collider.compute_force(
                    st_ad,
                    replace(
                        sy_ad, force_model=_CustomMorseForce(depth=d, alpha=morse.alpha)
                    ),
                )[0].force
                ** 2
            )
        )(jnp.asarray(2.0))
        return out, st_step, st_v, g

    with pallas_enabled(True):
        out_p, step_p, st_vp, g_p = _run_all()
    with pallas_enabled(False):
        out_j, step_j, st_vj, g_j = _run_all()

    for name, _, _ in cases:
        st_p, _ = out_p[name]
        st_j, _ = out_j[name]
        _assert_close_pair(st_p.force, st_j.force, f"{name} force")
        _assert_close_pair(st_p.torque, st_j.torque, f"{name} torque")
    _assert_close_pair(step_p[0].pos_c, step_j[0].pos_c, "custom_hist_le_3D step pos_c")
    _assert_close_pair(step_p[0].force, step_j[0].force, "custom_hist_le_3D step force")
    _assert_close_pair(
        step_p[0].torque, step_j[0].torque, "custom_hist_le_3D step torque"
    )
    _assert_close_pair(st_vp.force, st_vj.force, "vmap custom morse force")
    _assert_close_pair(g_p, g_j, "grad custom morse depth")
