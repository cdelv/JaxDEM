import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem


def set_up_spheres(
    n: int,
    dim: int,
    polydispersity: float = 1.5,
    collider_type: str = "naive",
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
) -> tuple[jdem.State, jdem.System]:
    spacing = 1.2
    radius = 0.5

    _n_per_axis = int(n ** (1 / dim))
    n_per_axis = (_n_per_axis,) * dim

    state = jdem.utils.grid_state(
        n_per_axis=n_per_axis,
        spacing=spacing,
        radius_range=(radius / polydispersity, radius),
        vel_range=[-1.0, 1.0],
        seed=seed,
    )

    collider_kw = dict()
    if collider_type in [
        "celllist",
        "CellList",
        "sap",
        "sap_pca",
        "sap_shifted",
        "SweepAndPrune",
        "MultiCellList",
        "tsg",
        "tsg_base",
    ]:
        collider_kw["state"] = state

    if neighbor_list:
        max_rad = jnp.max(state.rad)
        cutoff = float(2.0 * max_rad)
        collider_kw_actual = {
            "state": state,
            "cutoff": cutoff,
            "skin": skin,
            "secondary_collider_type": collider_type,
            "secondary_collider_kw": collider_kw,
        }
        collider_type_actual = "NeighborList"
    else:
        collider_kw_actual = collider_kw
        collider_type_actual = collider_type

    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e3, poisson=0.3)],
    )
    system = jdem.System.create(
        state.shape,
        domain_type="periodic",
        domain_kw={
            "box_size": (spacing * _n_per_axis,) * dim,
            "anchor": (-radius,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=0.001,
        mat_table=mat_table,
    )

    return state, system


def run_benchmark(
    n_particles: int,
    dim: int,
    collider_type: str,
    n_steps: int = 1000,
    neighbor_list=False,
):
    state, system = set_up_spheres(
        n=n_particles, dim=dim, collider_type=collider_type, neighbor_list=neighbor_list
    )

    # Warm up JIT
    state, system = system.step(state, system, n=n_steps)
    jax.block_until_ready(state.pos)

    # Benchmark
    start = time.perf_counter()
    state, system = system.step(state, system, n=n_steps)
    jax.block_until_ready(state.pos)
    end = time.perf_counter()

    duration = end - start
    throughput = state.N * n_steps / duration
    return throughput


def set_up_facets_and_spheres(
    dim: int,
    collider_type: str = "naive",
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
) -> tuple[jdem.State, jdem.System]:
    np.random.seed(seed)

    # 1. Create empty state
    state = jdem.State.create()

    # 2. Generate a single large grid to avoid overlaps
    n_facets = 50
    n_spheres = 100
    n_total = n_facets + n_spheres

    grid = jdem.utils.grid_state(
        n_per_axis=(6, 6, 6) if dim == 3 else (13, 13),
        spacing=1.5,
        radius_range=(1.0, 1.0),
        vel_range=[-1.0, 1.0],
        seed=seed,
    )

    L = 0.5
    for i in range(n_facets):
        com = np.array(grid.pos[i])
        vel = jnp.array([grid.vel[i]])

        if dim == 3:
            v1 = com + np.array([L, -L / 2, 0.0])
            v2 = com + np.array([-L, -L / 2, 0.0])
            v3 = com + np.array([0.0, L, 0.0])
            vertices = jnp.stack([v1, v2, v3])
            ang_vel = vel / 2.0
        else:
            v1 = com + np.array([L, -L / 2])
            v2 = com + np.array([-L, -L / 2])
            vertices = jnp.stack([v1, v2])
            ang_vel = jnp.array([vel[0, 0] / 2.0])

        state = jdem.State.add_facet(
            state,
            vertices,
            vel=vel,
            ang_vel=ang_vel,
            mass=jnp.array([1.0]),
            species_id=jnp.array([1]),
            thickness=0.1,
        )

    # 3. Add spheres safely from the remainder of the grid
    pos_s = np.array(grid.pos[n_facets:n_total])
    rad_s = jnp.ones(n_spheres) * 0.2
    mass_s = jnp.ones(n_spheres) * 0.5
    vel_s = np.array(grid.vel[n_facets:n_total])
    species_ids_s = jnp.zeros(n_spheres, dtype=int)

    sphere_state = jdem.State.create(
        pos=pos_s, rad=rad_s, mass=mass_s, vel=vel_s, species_id=species_ids_s
    )
    state = jdem.State.merge(state, sphere_state)

    collider_kw = dict()
    if collider_type in [
        "celllist",
        "CellList",
        "sap",
        "sap_pca",
        "sap_shifted",
        "SweepAndPrune",
        "MultiCellList",
    ]:
        collider_kw["state"] = state

    if collider_type == "CellList":
        collider_kw["max_neighbors"] = 512
    elif collider_type == "MultiCellList":
        collider_kw["max_hashes"] = 125

    if neighbor_list:
        max_rad = jnp.max(state.rad)
        cutoff = float(2.0 * max_rad)
        collider_kw_actual = {
            "state": state,
            "cutoff": cutoff,
            "skin": skin,
            "secondary_collider_type": collider_type,
            "secondary_collider_kw": collider_kw,
        }
        collider_type_actual = "NeighborList"
    else:
        collider_kw_actual = collider_kw
        collider_type_actual = collider_type

    mat = jdem.Material.create(
        "elasticfrict", young=1e3, poisson=0.3, density=2000.0, mu=0.5, e=0.5, mu_r=0.1
    )
    mat_table = jdem.MaterialTable.from_materials([mat])

    # Force router to handle sphere-sphere, sphere-facet, facet-facet
    router = jdem.ForceRouter.from_dict(
        S=2,
        mapping={
            (0, 0): jdem.ForceModel.create("cundallstrack"),
            (1, 0): jdem.ForceModel.create("sphere_facet_spring", thickness=0.1),
            (1, 1): jdem.ForceModel.create("facet_facet_spring", thickness=0.1),
        },
    )

    system = jdem.System.create(
        state.shape,
        domain_type="periodic",
        domain_kw={
            "box_size": (8.5,) * dim,
            "anchor": (-1.0,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    return state, system


if __name__ == "__main__":
    n_values = [2**i for i in range(4, 19)]  # 4k, 8k, 16k, 32k
    colliders = ["SweepAndPrune", "CellList"]
    dim = 3

    results = {c: [] for c in colliders}

    print("Starting benchmark...")
    for n in n_values:
        print(f"\nTesting N={n}")
        for c in colliders:
            throughput = run_benchmark(n, dim, c, n_steps=1000, neighbor_list=False)
            results[c].append(throughput)
            print(f"  {c}:\t {throughput:.2e} steps/sec")

    # Plotting
    plt.figure(figsize=(10, 6))
    for c in colliders:
        plt.plot(n_values, results[c], marker="o", label=c)

    plt.title(f"Collider Throughput Benchmark ({dim}D Spheres)")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Throughput (steps/sec)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"benchmark_results_{dim}D.png")
    print(f"\\nBenchmark plot saved to benchmark_results_{dim}D.png")

    # state, system = set_up_facets_and_spheres(
    #     dim=3, collider_type="naive", seed=42, neighbor_list=False, skin=0.5
    # )
    # writer = jdem.VTKWriter()
    # writer.save(state, system)

    # for i in range(500):
    #     state, system = system.step(state, system, n=20)
    #     writer.save(state, system)
