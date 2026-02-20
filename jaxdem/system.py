# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Defines the simulation configuration and the tooling for driving the simulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, final, Tuple, Optional, Dict, Any, Sequence, Callable

from .integrators import LinearIntegrator, RotationIntegrator
from .colliders import Collider
from .domains import Domain
from .bonded_forces import BondedForceModel
from .forces import ForceModel, ForceManager
from .materials import MaterialTable, Material

if TYPE_CHECKING:
    from .state import State
    from .minimizers import LinearMinimizer, RotationMinimizer


def _check_material_table(table: MaterialTable, required: Sequence[str]) -> None:
    """
    Checks if the provided MaterialTable contains all required properties for a given force model.

    This helper function ensures that all material properties specified by a
    :class:`ForceModel` (via :attr:`ForceModel.required_material_properties`)
    are present as attributes in the given :class:`MaterialTable`.

    Parameters
    ----------
    table : MaterialTable
        The material table instance to check.
    required : Sequence[str]
        A sequence of strings representing the names of material properties that
        are required by a specific force model.

    Raises
    ------
    KeyError
        If the `MaterialTable` instance is missing any of the `required` material properties.
    """
    missing = [k for k in required if not hasattr(table, k)]
    if missing:
        raise KeyError(
            f"MaterialTable lacks fields {missing}, required by the selected force model."
        )


def _save_state_system(state: State, system: System) -> Tuple[State, System]:
    return state, system


@jax.jit(donate_argnames=("state", "system"))
def _step_once(state: State, system: System) -> Tuple[State, System]:
    system = dataclasses.replace(
        system,
        time=system.time + system.dt,
        step_count=system.step_count + 1,
    )
    state, system = system.domain.apply(state, system)
    state, system = system.linear_integrator.step_before_force(state, system)
    state, system = system.rotation_integrator.step_before_force(state, system)
    state, system = system.collider.compute_force(state, system)
    state, system = system.force_manager.apply(state, system)
    state, system = system.linear_integrator.step_after_force(state, system)
    state, system = system.rotation_integrator.step_after_force(state, system)
    return state, system


@jax.jit(donate_argnames=("state", "system"))
def _steps_fori_loop(
    state: State, system: System, n: int | jax.Array
) -> Tuple[State, System]:
    return jax.lax.fori_loop(0, n, lambda i, carry: _step_once(*carry), (state, system))


@final
@jax.tree_util.register_dataclass
@dataclass
class System:
    """
    Encapsulates the entire simulation configuration.

    Notes
    -----
    - The `System` object is designed to be JIT-compiled for efficient execution.
    - The `System` dataclass is compatible with :func:`jax.jit`, so every field should remain JAX arrays for best performance.

    Example
    -------
    Creating a basic 2D simulation system:

    >>> import jaxdem as jdem
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a System instance
    >>> sim_system = jdem.System.create(
    >>>     state_shape=state.shape,
    >>>     dt=0.001,
    >>>     linear_integrator_type="euler",
    >>>     rotation_integrator_type="spiral",
    >>>     collider_type="naive",
    >>>     domain_type="free",
    >>>     force_model_type="spring",
    >>>     # You can pass keyword arguments to component constructors via '_kw' dicts
    >>>     domain_kw=dict(box_size=jnp.array([5.0, 5.0]), anchor=jnp.array([0.0, 0.0]))
    >>> )
    >>>
    >>> print(f"System integrator: {sim_system.linear_integrator.__class__.__name__}")
    >>> print(f"System force model: {sim_system.force_model.__class__.__name__}")
    >>> print(f"Domain box size: {sim_system.domain.box_size}")
    """

    linear_integrator: LinearIntegrator | LinearMinimizer
    """Instance of :class:`jaxdem.LinearIntegrator` that advances the simulation linear state in time."""

    rotation_integrator: RotationIntegrator | RotationMinimizer
    """Instance of :class:`jaxdem.RotationIntegrator` that advances the simulation angular state in time."""

    collider: Collider
    """Instance of :class:`jaxdem.Collider` that performs contact detection and computes inter-particle forces and potential energies."""

    domain: Domain
    """Instance of :class:`jaxdem.Domain` that defines the simulation boundaries, displacement rules, and boundary conditions."""

    force_manager: ForceManager
    """Instance of :class:`jaxdem.ForceManager` that handles per particle forces like external forces and resets forces."""

    bonded_force_model: Optional[BondedForceModel]
    """
    Optional instance of :class:`jaxdem.ForceModel` that defines bonded interactions
    by passing a force and energy function to the `ForceManager`.
    """

    force_model: ForceModel
    """Instance of :class:`jaxdem.ForceModel` that defines the physical laws for inter-particle interactions."""

    mat_table: MaterialTable
    """Instance of :class:`jaxdem.MaterialTable` holding material properties and pairwise interaction parameters."""

    dt: jax.Array
    r"""The global simulation time step :math:`\Delta t`."""

    time: jax.Array
    """Elapsed simulation time."""

    dim: jax.Array
    """Spatial dimension of the system."""

    step_count: jax.Array
    """Number of integration steps that have been performed."""

    key: jax.Array
    """PRNG key supporting stochastic functionality.  Always update using split to ensure new numbers are generated."""

    interact_same_deformable_id: jax.Array
    """
    Boolean scalar controlling interactions between particles with the same ``bond_id``.

    If ``False`` (default), such pairs are masked out in colliders.
    If ``True``, these pairs are allowed to interact.
    """

    @staticmethod
    @partial(jax.named_call, name="System.create")
    def create(
        state_shape: Tuple[int, ...],
        *,
        dt: float = 0.005,
        time: float = 0.0,
        linear_integrator_type: str = "verlet",
        rotation_integrator_type: str = "verletspiral",
        collider_type: str = "naive",
        domain_type: str = "free",
        bonded_force_model_type: Optional[str] = None,
        bonded_force_manager_kw: Optional[Dict[str, Any]] = None,
        bonded_force_model: Optional[BondedForceModel] = None,
        force_model_type: str = "spring",
        force_manager_kw: Optional[Dict[str, Any]] = None,
        mat_table: Optional[MaterialTable] = None,
        linear_integrator_kw: Optional[Dict[str, Any]] = None,
        rotation_integrator_kw: Optional[Dict[str, Any]] = None,
        collider_kw: Optional[Dict[str, Any]] = None,
        domain_kw: Optional[Dict[str, Any]] = None,
        force_model_kw: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        key: Optional[jax.Array] = None,
        interact_same_deformable_id: bool = False,
    ) -> System:
        """
        Factory method to create a :class:`System` instance with specified components.

        Parameters
        ----------
        state_shape : Tuple
            Shape of the state tensors handled by the simulation. The penultimate
            dimension corresponds to the number of particles ``N`` and the last
            dimension corresponds to the spatial dimension ``dim``.
        dt : float, optional
            The global simulation time step.
        linear_integrator_type : str, optional
            The registered type string for the :class:`jaxdem.integrators.LinearIntegrator`
            used to evolve translational degrees of freedom.
        rotation_integrator_type : str, optional
            The registered type string for the :class:`jaxdem.integrators.RotationIntegrator`
            used to evolve angular degrees of freedom.
        collider_type : str, optional
            The registered type string for the :class:`jaxdem.Collider` to use.
        domain_type : str, optional
            The registered type string for the :class:`jaxdem.Domain` to use.
        force_model_type : str, optional
            The registered type string for the :class:`jaxdem.ForceModel` to use.
        force_manager_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of `ForceManager`.
        mat_table : MaterialTable or None, optional
            An optional pre-configured :class:`jaxdem.MaterialTable`. If `None`, a
            default `jaxdem.MaterialTable` will be created with one generic elastic material and "harmonic" `jaxdem.MaterialMatchmaker`.
        linear_integrator_kw : Dict[str, Any] or None, optional
            Keyword arguments forwarded to the constructor of the selected
            `LinearIntegrator` type.
        rotation_integrator_kw : Dict[str, Any] or None, optional
            Keyword arguments forwarded to the constructor of the selected
            `RotationIntegrator` type.
        collider_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `Collider` type.
        domain_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `Domain` type.
        force_model_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `ForceModel` type.
        seed : int, optional
            Integer seed used for random number generation.  Defaults to 0.
        key : jax.Array, optional
            Key used for the jax random number generation.  Defaults to None, shadowed by seed.

        Returns
        -------
        System
            A fully configured `System` instance ready for simulation.

        Raises
        ------
        KeyError
            If a specified `*_type` is not registered in its respective factory,
            or if the `mat_table` is missing properties required by the `force_model`.
        TypeError
            If constructor keyword arguments are invalid for any component.
        AssertionError
            If the `domain_kw` 'box_size' or 'anchor' shapes do not match the `dim`.

        Example
        -------
        Creating a 3D system with reflective boundaries and a custom `dt`:

        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> system_reflect = jdem.System.create(
        >>>     state_shape=(N, 3),
        >>>     dt=0.0005,
        >>>     domain_type="reflect",
        >>>     domain_kw=dict(box_size=jnp.array([20.0, 20.0, 20.0]), anchor=jnp.array([-10.0, -10.0, -10.0])),
        >>>     force_model_type="spring",
        >>> )
        >>> print(f"System dt: {system_reflect.dt}")
        >>> print(f"Domain type: {system_reflect.domain.__class__.__name__}")

        Creating a system with a pre-defined MaterialTable:

        >>> custom_mat_kw = dict(young=2.0e5, poisson=0.25)
        >>> custom_material = jdem.Material.create("custom_mat", **custom_mat_kw)
        >>> custom_mat_table = jdem.MaterialTable.from_materials(
        ...     [custom_material], matcher=jdem.MaterialMatchmaker.create("linear")
        ... )
        >>>
        >>> system_custom_mat = jdem.System.create(
        ...     state_shape=(N, 2),
        ...     mat_table=custom_mat_table,
        ...     force_model_type="spring"
        ... )
        """
        dim = state_shape[-1]
        linear_integrator_kw = (
            {} if linear_integrator_kw is None else dict(linear_integrator_kw)
        )
        rotation_integrator_kw = (
            {} if rotation_integrator_kw is None else dict(rotation_integrator_kw)
        )
        collider_kw = {} if collider_kw is None else dict(collider_kw)
        force_model_kw = {} if force_model_kw is None else dict(force_model_kw)
        domain_kw = {} if domain_kw is None else dict(domain_kw)

        force_manager_kw = (
            dict(
                gravity=None,
                force_functions=(),
            )
            if force_manager_kw is None
            else dict(force_manager_kw)
        )

        if mat_table is None:
            mat_table = MaterialTable.from_materials(
                [Material.create("elastic", density=0.27, young=1.0e4, poisson=0.3)],
            )

        force_model = ForceModel.create(force_model_type, **force_model_kw)
        _check_material_table(mat_table, force_model.required_material_properties)

        if bonded_force_model is None:
            if bonded_force_model_type is not None:
                bonded_force_manager_kw = (
                    {}
                    if bonded_force_manager_kw is None
                    else dict(bonded_force_manager_kw)
                )
                bonded_force_model = BondedForceModel.create(
                    bonded_force_model_type, **bonded_force_manager_kw
                )

        if bonded_force_model is not None:
            force_manager_kw["force_functions"] = tuple(
                force_manager_kw["force_functions"]
            ) + (bonded_force_model.force_and_energy_fns,)

        force_manager = ForceManager.create(state_shape, **force_manager_kw)

        if key is None:
            key = jax.random.PRNGKey(seed)

        return System(
            linear_integrator=LinearIntegrator.create(
                linear_integrator_type, **linear_integrator_kw
            ),
            rotation_integrator=RotationIntegrator.create(
                rotation_integrator_type, **rotation_integrator_kw
            ),
            collider=Collider.create(collider_type, **collider_kw),
            domain=Domain.create(domain_type, dim=dim, **domain_kw),
            force_manager=force_manager,
            bonded_force_model=bonded_force_model,
            force_model=force_model,
            mat_table=mat_table,
            dim=jnp.asarray(dim, dtype=int),
            dt=jnp.asarray(dt, dtype=float),
            time=jnp.asarray(time, dtype=float),
            step_count=jnp.asarray(0, dtype=int),
            key=key,
            interact_same_deformable_id=jnp.asarray(
                interact_same_deformable_id, dtype=bool
            ),
        )

    @staticmethod
    def trajectory_rollout(
        state: State,
        system: System,
        *,
        n: Optional[int] = None,
        stride: int = 1,
        strides: Optional[jax.Array] = None,
        save_fn: Callable[[State, System], Any] = _save_state_system,
        unroll: int = 2,
    ) -> Tuple[State, System, Any]:
        """
        Roll the system forward while collecting saved outputs at each frame.

        The rollout always stores one output per frame via `save_fn(state, system)`.
        The output of save_fn must be a pytree. Frame spacing can be either:
        - constant (`stride`), or
        - variable (`strides` jax.Array).

        Parameters
        ----------
        state : State
            Initial state.
        system : System
            Initial system configuration.
        n : int, optional
            Number of saved frames.
            Required when `strides` is `None`.
            Ignored when `strides` is provided.
        stride : int, optional
            Constant number of integration steps between consecutive saves.
            Used only when `strides` is `None`. Defaults to 1.
        strides : jax.Array, optional
            Integer 1D array of per-frame integration strides. When provided,
            this overrides `stride`, and `n` is inferred from `len(strides)`.
        save_fn : Callable[[State, System], Any], optional
            Function called after each saved frame. Its return pytree is stacked
            along axis 0 across frames. Defaults to returning `(state, system)`.
        unroll : int, optional
            Unroll factor passed to the outer `jax.lax.scan`. Defaults to 2.

        Returns
        -------
        Tuple[State, System, Any]
            `(final_state, final_system, trajectory_like)` where
            `trajectory_like` is the stacked output of `save_fn`.

        Raises
        ------
        ValueError
            If `n` is missing while `strides is None`, or if `strides` is not 1D.

        Example
        -------
        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> state = jdem.utils.grid_state(n_per_axis=(1, 1), spacing=1.0, radius=0.1)
        >>> system = jdem.System.create(state_shape=state.shape, dt=0.01)
        >>>
        >>> # Constant stride: n is required
        >>> final_state, final_system, traj = jdem.System.trajectory_rollout(
        ...     state, system, n=10, stride=5
        ... )
        >>>
        >>> # Variable strides: n inferred from len(strides)
        >>> deltas = jnp.array([1, 2, 4, 8])
        >>> final_state, final_system, traj = jdem.System.trajectory_rollout(
        ...     state, system, strides=deltas
        ... )
        """
        stride = int(stride)
        if strides is not None:
            strides = jnp.asarray(strides, dtype=int)
            n = None
        else:
            if n is None:
                raise ValueError("`n` must be provided when `strides` is None.")

        def scan_fn(
            carry: Tuple[State, System], xs: Optional[jax.Array]
        ) -> Tuple[Tuple[State, System], Any]:
            n = xs if xs is not None else stride
            state, system = carry
            state, system = system.step(state, system, n=n)
            return (state, system), save_fn(state, system)

        (state, system), traj = jax.lax.scan(
            scan_fn, (state, system), length=n, xs=strides, unroll=unroll
        )
        return state, system, traj

    @staticmethod
    @partial(jax.named_call, name="System.step")
    def step(
        state: State,
        system: System,
        *,
        n: int | jax.Array = 1,
    ) -> Tuple[State, System]:
        """
        Advance the simulation by `n` integration steps.

        Parameters
        ----------
        state : State
            Current state.
        system : System
            Current system configuration.
        n : int or jax.Array, optional
            Number of integration steps. May be a Python `int` or a scalar
            JAX array. Defaults to 1.

        Returns
        -------
        Tuple[State, System]
            `(final_state, final_system)` after `n` steps.

        Example
        -------
        >>> # Advance by 10 steps
        >>> state_after_10_steps, system_after_10_steps = jdem.System.step(state, system, n=10)
        """
        body = _steps_fori_loop

        if state.batch_size > 1:
            body = jax.vmap(body, in_axes=(0, 0, None))

        return body(state, system, n)

    @staticmethod
    @partial(jax.named_call, name="System.stack")
    def stack(systems: Sequence[System]) -> System:
        """
        Concatenates a sequence of :class:`System` snapshots into a trajectory or batch along axis 0.

        This method is useful for collecting simulation snapshots over time into a
        single `System` object where the leading dimension represents time or when
        preparing a batched system.

        Parameters
        ----------
        systems : Sequence[System]
            A sequence (e.g., list, tuple) of :class:`System` instances to be stacked.

        Returns
        -------
        System
            A new :class:`System` instance where each attribute is a JAX array with an
            additional leading dimension representing the stacked trajectory.
            For example, if input `pos` was `(N, dim)`, output `pos` will be `(T, N, dim)`.
        """
        systems = list(systems)
        if not systems:
            raise ValueError("System.stack() received an empty list")

        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *systems)

    @staticmethod
    @partial(jax.named_call, name="System.unstack")
    def unstack(system: System) -> list[System]:
        """
        Split a stacked/batched :class:`System` along the leading axis into a Python list.

        This is the convenient inverse of :meth:`System.stack`:

        - If `stacked = System.stack([sys0, sys1, ...])`, then `System.unstack(stacked)` returns
          `[sys0, sys1, ...]`.

        Notes
        -----
        - The split is performed along axis 0 (the leading axis).
        - A single snapshot `System` cannot be unstacked with this method.
        """
        if system.dt.ndim < 1:
            raise ValueError(
                "System.unstack() expects a stacked/batched System with a leading axis "
                f"(dt.ndim >= 1). Got dt.shape={system.dt.shape}."
            )

        n = int(system.dt.shape[0])
        return [jax.tree_util.tree_map(lambda x, i=i: x[i], system) for i in range(n)]
