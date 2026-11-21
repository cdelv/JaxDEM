# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unified spheres-in-a-box benchmark across dims, domains, colliders, forces, modes, and regimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jdem

# (step_fn, total_particles_B_times_N, steps_per_call)
BuiltCase = tuple[Callable[[], Any], int, int]

# Physical defaults (matching the spheres-in-a-box benchmark).
SPHERE_RAD = 1.0
SPHERE_MASS = 1.0
SPRING_K = 1000.0
DT = 1e-2
BATCH_SIZE = 8
NL_SKIN = 0.4
NL_MAX_NEIGHBORS = 64

DIMS: tuple[str, ...] = ("3d", "2d")

DOMAINS: tuple[str, ...] = (
    "periodic",
    "leesedwards",
    "reflect",
    "reflectsphere",
)

COLLIDERS: tuple[str, ...] = (
    "cell_list",
    "cell_list_static",
    "multi_cell_list",
    "multi_cell_list_static",
    "naive",
    "nl_cell_list",
    "nl_cell_list_static",
    "nl_multi_cell_list",
    "nl_multi_cell_list_static",
    "nl_naive",
)

HISTORY_COLLIDERS: frozenset[str] = frozenset(
    {
        "naive",
        "nl_cell_list",
        "nl_cell_list_static",
        "nl_multi_cell_list",
        "nl_multi_cell_list_static",
        "nl_naive",
    }
)

NAIVE_SIZED_COLLIDERS: frozenset[str] = frozenset({"naive", "nl_naive"})

FORCES: tuple[str, ...] = ("spring", "cundallstrack")

MODES: tuple[str, ...] = ("single", "batch")

# Fixed 4 report regimes: high/low velocity x high/low packing fraction.
# High phi uses 0.60 in 3D (FCC) and 0.72 in 2D (hexagonal); low phi uses 0.25 / 0.30.
REGIMES: dict[str, dict[str, Any]] = {
    "high_vel_high_phi": {
        "label": "High v, High φ",
        "short": "Hi v / Hi φ",
        "vel": 0.5,
        "phi": {"3d": 0.60, "2d": 0.72},
    },
    "high_vel_low_phi": {
        "label": "High v, Low φ",
        "short": "Hi v / Lo φ",
        "vel": 0.5,
        "phi": {"3d": 0.25, "2d": 0.30},
    },
    "low_vel_high_phi": {
        "label": "Low v, High φ",
        "short": "Lo v / Hi φ",
        "vel": 0.05,
        "phi": {"3d": 0.60, "2d": 0.72},
    },
    "low_vel_low_phi": {
        "label": "Low v, Low φ",
        "short": "Lo v / Lo φ",
        "vel": 0.05,
        "phi": {"3d": 0.25, "2d": 0.30},
    },
}

REGIME_ORDER: tuple[str, ...] = tuple(REGIMES.keys())

PROFILES: dict[str, dict[str, int]] = {
    "gpu": {
        "n": 65_536,
        "n_naive": 5_000,
        "batch": BATCH_SIZE,
        "steps": 2_000,
        "repeats": 3,
    },
    "cpu": {
        "n": 2_048,
        "n_naive": 512,
        "batch": 4,
        "steps": 200,
        "repeats": 2,
    },
    "quick": {
        "n": 4_096,
        "n_naive": 1_024,
        "batch": 4,
        "steps": 200,
        "repeats": 2,
    },
}


def validate_phi_vel(dim: str, phi: float, vel: float) -> None:
    """Validate user-supplied ``phi`` ([0, 0.7] in 3D, [0, 0.8] in 2D) and ``vel`` ([0, 1])."""
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}.")
    max_phi = 0.7 if dim == "3d" else 0.8
    if not (0.0 <= phi <= max_phi):
        raise ValueError(
            f"Packing fraction phi={phi} out of bounds for {dim} (expected [0, {max_phi}])."
        )
    if not (0.0 <= vel <= 1.0):
        raise ValueError(
            f"Initial speed vel={vel} out of bounds (expected [0, 1])."
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One (dim, domain, force, mode, collider, regime) benchmark configuration."""

    dim: str
    domain: str
    force: str
    mode: str
    collider: str
    regime: str

    @property
    def name(self) -> str:
        return (
            f"{self.dim}/{self.domain}/{self.force}/"
            f"{self.mode}/{self.collider}/{self.regime}"
        )

    @property
    def phi(self) -> float:
        return float(REGIMES[self.regime]["phi"][self.dim])

    @property
    def vel(self) -> float:
        return float(REGIMES[self.regime]["vel"])


def all_cases() -> list[BenchmarkCase]:
    """Enumerate all valid (dim, domain, force, mode, collider, regime) cases."""
    cases: list[BenchmarkCase] = []
    for dim in DIMS:
        for domain in DOMAINS:
            for force in FORCES:
                valid_colliders = (
                    COLLIDERS
                    if force == "spring"
                    else tuple(c for c in COLLIDERS if c in HISTORY_COLLIDERS)
                )
                for mode in MODES:
                    for collider in valid_colliders:
                        for regime in REGIME_ORDER:
                            cases.append(
                                BenchmarkCase(
                                    dim=dim,
                                    domain=domain,
                                    force=force,
                                    mode=mode,
                                    collider=collider,
                                    regime=regime,
                                )
                            )
    return cases


def select_cases(
    *,
    pattern: str | None = None,
    dim: str | None = None,
    domain: str | None = None,
    collider: str | None = None,
    force: str | None = None,
    mode: str | None = None,
    regime: str | None = None,
) -> list[BenchmarkCase]:
    """Filter cases by explicit axes and/or comma-separated substring patterns."""
    selected = all_cases()

    def _split(val: str | None) -> set[str] | None:
        if not val:
            return None
        return {tok.strip().lower() for tok in val.split(",") if tok.strip()}

    for attr, raw in (
        ("dim", dim),
        ("domain", domain),
        ("collider", collider),
        ("force", force),
        ("mode", mode),
        ("regime", regime),
    ):
        allowed = _split(raw)
        if allowed:
            selected = [c for c in selected if getattr(c, attr) in allowed]

    if pattern:
        tokens = [tok.strip().lower() for tok in pattern.split(",") if tok.strip()]
        selected = [
            c for c in selected if any(tok in c.name.lower() for tok in tokens)
        ]

    return selected


# ---------------------------------------------------------------------------
# Zero-overlap 2D (square / hex) & 3D (SC / FCC) sphere-in-box generators
# ---------------------------------------------------------------------------


def _lattice_2d(n: int, rad: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Place ``n`` disks of radius ``rad`` in a 2D box at packing ``phi`` (0..0.80) with zero overlap."""
    phi_eff = max(phi, 1e-4)
    total_area = n * math.pi * (rad**2)
    target_area = total_area / phi_eff

    if phi_eff <= 0.70:
        # Square lattice (max packing pi/4 ~= 0.7854).
        m = math.ceil(math.sqrt(n))
        box_len = math.sqrt(target_area)
        spacing = box_len / m
        idx = np.arange(n, dtype=np.int64)
        pos = (np.stack([idx % m, idx // m], axis=-1) + 0.5) * spacing
        return pos, np.array([box_len, box_len], dtype=np.float64)

    # Hexagonal / triangular lattice (2 sites per rectangular cell, max packing pi/(2*sqrt(3)) ~= 0.9069).
    n_cells = math.ceil(n / 2.0)
    mx = max(1, round(math.sqrt(n_cells * math.sqrt(3.0))))
    my = math.ceil(n_cells / mx)
    # Rectangular box with aspect ratio mx : my*sqrt(3) matching exact target_area.
    a = math.sqrt(target_area / (mx * my * math.sqrt(3.0)))
    bx = mx * a
    by = my * a * math.sqrt(3.0)
    gx, gy = np.meshgrid(np.arange(mx), np.arange(my), indexing="ij")
    base = np.stack([gx.ravel() * a, gy.ravel() * a * math.sqrt(3.0)], axis=-1)
    offset = np.array([0.5 * a, 0.5 * a * math.sqrt(3.0)], dtype=np.float64)
    all_pos = np.concatenate([base + 0.25 * offset, base + 1.25 * offset], axis=0)
    pos = np.mod(all_pos[:n], np.array([bx, by]))
    return pos, np.array([bx, by], dtype=np.float64)


def _lattice_3d(n: int, rad: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Place ``n`` spheres of radius ``rad`` in a 3D cube at packing ``phi`` (0..0.70) with zero overlap."""
    phi_eff = max(phi, 1e-4)
    vol_sphere = (4.0 / 3.0) * math.pi * (rad**3)
    box_len = (n * vol_sphere / phi_eff) ** (1.0 / 3.0)

    if phi_eff <= 0.50:
        # Simple cubic lattice (max packing pi/6 ~= 0.5236).
        m = math.ceil(n ** (1.0 / 3.0))
        spacing = box_len / m
        idx = np.arange(n, dtype=np.int64)
        ix = idx % m
        iy = (idx // m) % m
        iz = idx // (m * m)
        pos = (np.stack([ix, iy, iz], axis=-1) + 0.5) * spacing
    else:
        # Face-centered cubic lattice (4 sites/cell, max packing pi/(3*sqrt(2)) ~= 0.7405).
        m = math.ceil((n / 4.0) ** (1.0 / 3.0))
        spacing = box_len / m
        basis = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
            dtype=np.float64,
        )
        grid = np.mgrid[0:m, 0:m, 0:m].reshape(3, -1).T
        all_pos = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3) * spacing
        pos = all_pos[:n]

    return pos, np.full((3,), box_len, dtype=np.float64)


def _lattice_positions(
    n: int, dim: str, rad: float, phi: float, domain: str
) -> tuple[np.ndarray, np.ndarray]:
    pos, box = _lattice_3d(n, rad, phi) if dim == "3d" else _lattice_2d(n, rad, phi)
    if domain in ("reflect", "reflectsphere"):
        # Keep every sphere center at least `rad` away from reflective box walls.
        scale = (box - 2.0 * rad) / box
        pos = rad + pos * scale
    return pos, box


def _random_velocities(n: int, dim_int: int, speed: float, seed: int) -> np.ndarray:
    if speed == 0.0:
        return np.zeros((n, dim_int), dtype=np.float64)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, dim_int))
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (dirs / norms) * speed


def _material_table(force: str) -> jdem.MaterialTable:
    if force == "spring":
        mats = [
            jdem.Material.create("elastic", young=SPRING_K, poisson=0.3, density=1.0)
        ]
    elif force == "cundallstrack":
        mats = [
            jdem.Material.create(
                "elasticfrict",
                young=SPRING_K,
                poisson=0.3,
                density=1.0,
                mu=0.5,
                e=0.8,
                mu_r=0.05,
            )
        ]
    else:
        raise ValueError(f"Unsupported force law: {force!r}")
    return jdem.MaterialTable.from_materials(
        mats, matcher=jdem.MaterialMatchmaker.create("harmonic")
    )


def _domain_kw(domain: str, box: jax.Array) -> dict[str, Any]:
    if domain == "leesedwards":
        return {"box_size": box, "gamma": 0.0, "gamma_dot": 0.2, "alpha": 0, "beta": 1}
    return {"box_size": box}


def _parse_collider(collider: str) -> tuple[str, str | None, bool]:
    """Return ``(primary_type, secondary_type_or_None, is_static)``."""
    if collider == "naive":
        return "naive", None, False
    if collider == "cell_list":
        return "cell_list", None, False
    if collider == "cell_list_static":
        return "cell_list", None, True
    if collider == "multi_cell_list":
        return "multi_cell_list", None, False
    if collider == "multi_cell_list_static":
        return "multi_cell_list", None, True
    if collider.startswith("nl_"):
        sub = collider[3:]
        sub_type, _, sub_static = _parse_collider(sub)
        return "neighbor_list", sub_type, sub_static
    raise ValueError(f"Unsupported collider: {collider!r}")


def _collider_spec(
    collider: str,
    *,
    batched: bool,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    static_tile_cap: int | None = None,
) -> tuple[str, dict[str, Any]]:
    cutoff = 2.0 * SPHERE_RAD
    primary, secondary, is_static = _parse_collider(collider)

    if primary == "naive":
        return "naive", {}

    if primary in ("cell_list", "multi_cell_list"):
        kw: dict[str, Any] = {}
        if is_static:
            kw["static"] = True
        if batched:
            kw.update({"cell_size": cutoff, "search_range": 1})
        if static_tile_cap is not None:
            kw["tile_capacity"] = static_tile_cap
        return primary, kw

    # NeighborList + explicit secondary collider
    sec_kw: dict[str, Any] = {}
    if secondary in ("cell_list", "multi_cell_list"):
        if is_static:
            sec_kw["static"] = True
        if batched:
            sec_kw.update({"cell_size": cutoff + nl_skin, "search_range": 1})
        if static_tile_cap is not None:
            sec_kw["tile_capacity"] = static_tile_cap

    nl_kw: dict[str, Any] = {
        "cutoff": cutoff,
        "skin": nl_skin,
        "max_neighbors": nl_max_neighbors,
        "secondary_collider_type": secondary,
        "secondary_collider_kw": sec_kw,
    }
    return "neighbor_list", nl_kw


def build_spheres_in_box(
    *,
    collider: str,
    force: str = "spring",
    domain: str = "periodic",
    dim: str = "3d",
    mode: str = "single",
    phi: float = 0.60,
    vel: float = 0.5,
    n: int = 65_536,
    n_naive: int = 5_000,
    batch_size: int = BATCH_SIZE,
    steps: int = 2_000,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    only_spheres: bool = False,
) -> BuiltCase:
    """Build any spheres-in-a-box benchmark with arbitrary valid ``phi`` and ``vel``."""
    validate_phi_vel(dim, phi, vel)
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}.")
    if force not in FORCES:
        raise ValueError(f"force must be one of {FORCES}, got {force!r}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    if force == "cundallstrack" and collider not in HISTORY_COLLIDERS:
        raise ValueError(f"Collider {collider!r} does not support pair history ({force!r}).")

    dim_int = 3 if dim == "3d" else 2
    base_n = n_naive if collider in NAIVE_SIZED_COLLIDERS else n
    batched = mode == "batch"
    n_per_sim = max(16, base_n // 2) if batched else base_n

    pos_np, box_np = _lattice_positions(n_per_sim, dim, SPHERE_RAD, phi, domain)
    dtype = jnp.asarray(0.0).dtype
    box = jnp.asarray(box_np, dtype=dtype)
    mat_table = _material_table(force)
    dom_kw = _domain_kw(domain, box)
    rot_integrator = "verletspiral" if force == "cundallstrack" else ""

    # Resolve static tile_capacity on host before vmap when batching static cell colliders.
    primary, secondary, is_static = _parse_collider(collider)
    static_tile_cap: int | None = None
    if batched and is_static:
        cell_type = secondary if primary == "neighbor_list" else primary
        probe_state = jdem.State.create(
            pos=jnp.asarray(pos_np, dtype=dtype),
            rad=jnp.full((n_per_sim,), SPHERE_RAD, dtype=dtype),
            mass=jnp.full((n_per_sim,), SPHERE_MASS, dtype=dtype),
            only_spheres=only_spheres,
        )
        probe_kw: dict[str, Any] = {"static": True}
        if primary == "neighbor_list":
            probe_kw["cell_size"] = 2.0 * SPHERE_RAD + nl_skin
        probe_sys = jdem.System.create(
            state=probe_state,
            domain_type=domain,
            domain_kw=dom_kw,
            collider_type=cell_type,
            collider_kw=probe_kw,
            force_model_type=force if primary != "neighbor_list" else "spring",
            mat_table=mat_table,
        )
        static_tile_cap = int(probe_sys.collider.tile_capacity)

    col_type, col_kw = _collider_spec(
        collider,
        batched=batched,
        nl_skin=nl_skin,
        nl_max_neighbors=nl_max_neighbors,
        static_tile_cap=static_tile_cap,
    )

    if not batched:
        vel_np = _random_velocities(n_per_sim, dim_int, vel, seed=0)
        state = jdem.State.create(
            pos=jnp.asarray(pos_np, dtype=dtype),
            vel=jnp.asarray(vel_np, dtype=dtype),
            rad=jnp.full((n_per_sim,), SPHERE_RAD, dtype=dtype),
            mass=jnp.full((n_per_sim,), SPHERE_MASS, dtype=dtype),
            only_spheres=only_spheres,
        )
        system = jdem.System.create(
            state=state,
            dt=DT,
            mat_table=mat_table,
            domain_type=domain,
            domain_kw=dom_kw,
            collider_type=col_type,
            collider_kw=col_kw,
            force_model_type=force,
            linear_integrator_type="verlet",
            rotation_integrator_type=rot_integrator,
        )
        state, system = jdem.System.initialize(state, system)
        total_particles = n_per_sim
    else:
        pos_j = jnp.asarray(pos_np, dtype=dtype)
        vel_all = jnp.asarray(
            np.stack(
                [_random_velocities(n_per_sim, dim_int, vel, seed=b) for b in range(batch_size)],
                axis=0,
            ),
            dtype=dtype,
        )

        def make_one(vel_i: jax.Array) -> tuple[jdem.State, jdem.System]:
            st = jdem.State.create(
                pos=pos_j,
                vel=vel_i,
                rad=jnp.full((n_per_sim,), SPHERE_RAD, dtype=dtype),
                mass=jnp.full((n_per_sim,), SPHERE_MASS, dtype=dtype),
                only_spheres=only_spheres,
            )
            sys = jdem.System.create(
                state=st,
                dt=DT,
                mat_table=mat_table,
                domain_type=domain,
                domain_kw=dom_kw,
                collider_type=col_type,
                collider_kw=col_kw,
                force_model_type=force,
                linear_integrator_type="verlet",
                rotation_integrator_type=rot_integrator,
            )
            return st, sys

        state, system = jax.vmap(make_one)(vel_all)
        state, system = jdem.System.initialize(state, system)
        total_particles = batch_size * n_per_sim

    def step_fn() -> tuple[jdem.State, jdem.System]:
        return jdem.System.step(state, system, n=steps)

    return step_fn, int(total_particles), int(steps)


def build_case(
    case: BenchmarkCase,
    *,
    n: int,
    n_naive: int,
    batch_size: int,
    steps: int,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    only_spheres: bool = False,
) -> BuiltCase:
    """Build one fixed-regime ``BenchmarkCase``."""
    return build_spheres_in_box(
        collider=case.collider,
        force=case.force,
        domain=case.domain,
        dim=case.dim,
        mode=case.mode,
        phi=case.phi,
        vel=case.vel,
        n=n,
        n_naive=n_naive,
        batch_size=batch_size,
        steps=steps,
        nl_skin=nl_skin,
        nl_max_neighbors=nl_max_neighbors,
        only_spheres=only_spheres,
    )
