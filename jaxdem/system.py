# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Defines the simulation configuration and the tooling for driving the simulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, final, Tuple, Optional, Dict, Any, Sequence

from .integrators import Integrator
from .colliders import Collider
from .domains import Domain
from .forces import ForceModel, ForceManager
from .materials import MaterialTable, Material
from .material_matchmakers import MaterialMatchmaker

if TYPE_CHECKING:
    from .state import State


def _check_material_table(table, required: Sequence[str]):
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
        A sequence of strings representing the names of material properties that are required by a specific force model.

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


@final
@jax.tree_util.register_dataclass
@dataclass
class System:
    """
    Encapsulates the entire simulation configuration.

    Notes
    -----
    - The `System` object is designed to be JIT-compiled for efficient execution.
    - The `System` class is a frozen dataclass, meaning its attributes cannot be changed after instantiation.

    Example
    -------
    Creating a basic 2D simulation system:

    >>> import jaxdem as jdem
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a System instance
    >>> sim_system = jdem.System.create(
    >>>     dim=state.dim,
    >>>     dt=0.001,
    >>>     integrator_type="euler",
    >>>     collider_type="naive",
    >>>     domain_type="free",
    >>>     force_model_type="spring",
    >>>     # You can pass keyword arguments to component constructors via '_kw' dicts
    >>>     domain_kw=dict(box_size=jnp.array([5.0, 5.0]), anchor=jnp.array([0.0, 0.0]))
    >>> )
    >>>
    >>> print(f"System integrator: {sim_system.integrator.__class__.__name__}")
    >>> print(f"System force model: {sim_system.force_model.__class__.__name__}")
    >>> print(f"Domain box size: {sim_system.domain.box_size}")
    """

    integrator: "Integrator"
    """Instance of :class:`jaxdem.Integrator` that advances the simulation state in time."""

    collider: "Collider"
    """Instance of :class:`jaxdem.Collider` that performs contact detection and computes inter-particle forces and potential energies."""

    domain: "Domain"
    """Instance of :class:`jaxdem.Domain` that defines the simulation boundaries, displacement rules, and boundary conditions."""

    force_manager: "ForceManager"
    """Instance of :class:`jaxdem.ForceManager` that handles per particle forces like external forces and resets forces."""

    force_model: "ForceModel"
    """Instance of :class:`jaxdem.ForceModel` that defines the physical laws for inter-particle interactions."""

    mat_table: "MaterialTable"
    """Instance of :class:`jaxdem.MaterialTable` holding material properties and pairwise interaction parameters."""

    dt: jax.Array
    r"""The global simulation time step :math:`\Delta t`."""

    time: jax.Array
    """Elapsed simulation time."""

    dim: jax.Array
    """Spatial dimension of the system."""

    step_count: jax.Array
    """Number of integration steps that have been performed."""

    @staticmethod
    @partial(jax.named_call, name="System.create")
    def create(
        state_shape: Tuple,
        *,
        dt: float = 0.005,
        time: float = 0.0,
        integrator_type: str = "euler",
        collider_type: str = "naive",
        domain_type: str = "free",
        force_model_type: str = "spring",
        force_manager_kw: Optional[Dict[str, Any]] = None,
        mat_table: Optional["MaterialTable"] = None,
        integrator_kw: Optional[Dict[str, Any]] = None,
        collider_kw: Optional[Dict[str, Any]] = None,
        domain_kw: Optional[Dict[str, Any]] = None,
        force_model_kw: Optional[Dict[str, Any]] = None,
    ) -> "System":
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
        integrator_type : str, optional
            The registered type string for the :class:`jaxdem.Integrator` to use.
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
        integrator_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `Integrator` type.
        collider_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `Collider` type.
        domain_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `Domain` type.
        force_model_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of the selected `ForceModel` type.

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
        integrator_kw = {} if integrator_kw is None else dict(integrator_kw)
        collider_kw = {} if collider_kw is None else dict(collider_kw)
        force_model_kw = {} if force_model_kw is None else dict(force_model_kw)

        force_manager_kw = (
            dict(
                gravity=None,
                force_functions=(),
            )
            if force_manager_kw is None
            else dict(force_manager_kw)
        )

        if domain_kw is None:
            domain_kw = {
                "box_size": jnp.ones(dim, dtype=float),
                "anchor": jnp.zeros(dim, dtype=float),
            }
        domain_kw = dict(domain_kw)

        if mat_table is None:
            matcher = MaterialMatchmaker.create("harmonic")
            mat_table = MaterialTable.from_materials(
                [Material.create("elastic", young=1.0e4, poisson=0.3)], matcher=matcher
            )

        force_model = ForceModel.create(force_model_type, **force_model_kw)
        force_manager = ForceManager.create(dim, state_shape, **force_manager_kw)

        _check_material_table(mat_table, force_model.required_material_properties)

        return System(
            integrator=Integrator.create(integrator_type, **integrator_kw),
            collider=Collider.create(collider_type, **collider_kw),
            domain=Domain.create(domain_type, dim=dim, **domain_kw),
            force_manager=force_manager,
            force_model=force_model,
            mat_table=mat_table,
            dim=jnp.asarray(dim, dtype=int),
            dt=jnp.asarray(dt, dtype=float),
            time=jnp.asarray(time, dtype=float),
            step_count=jnp.asarray(0, dtype=int),
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("n"), donate_argnames=("state", "system"))
    def _steps(state: "State", system: "System", n: int) -> Tuple["State", "System"]:
        """
        Internal method to advance the simulation state by multiple steps using `jax.lax.scan`.

        This function is an optimized JIT-compiled loop for performing `n` integration
        steps without re-entering Python between steps.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.
        n : int
            The number of integration steps to perform. This argument must be static.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the final `State` and `System` after `n` integration steps.
        """

        @partial(jax.named_call, name="System._steps")
        def body(carry, _):
            st, sys = carry
            return sys.integrator.step(st, sys), None

        (state, system), _ = jax.lax.scan(
            body, (state, system), xs=None, length=n, unroll=2
        )
        return state, system

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("n", "stride", "batched"),
    )
    def trajectory_rollout(
        state: "State",
        system: "System",
        *,
        n: int,
        stride: int = 1,
        batched: bool = False,
    ) -> Tuple["State", "System", Tuple["State", "System"]]:
        """
        Rolls the system forward for a specified number of frames, collecting a trajectory.

        This method performs `n * stride` total simulation steps, but it saves
        the `State` and `System` every `stride` steps, returning a trajectory
        of `n` snapshots. This is highly efficient for data collection within JAX
        as it leverages `jax.lax.scan`.

        Parameters
        ----------
        state : State
            The initial state of the simulation.
        system : System
            The initial system configuration.
        n : int
            The number of frames (snapshots) to collect in the trajectory. This argument must be static.
        stride : int, optional
            The number of integration steps to advance between each collected frame. Defaults to 1, meaning every step is collected. This argument must be static.

        Returns
        -------
        Tuple[State, System, Tuple[State, System]]
            A tuple containing:

            - `final_state`:
                The `State` object at the end of the rollout.

            - `final_system`:
                The `System` object at the end of the rollout.

            - `trajectory`:
                A tuple of `State` and `System` objects, where each leaf array has an additional leading axis of size `n` representing the trajectory. The `State` and `System` objects within `trajectory` are structured as if created by :meth:`jaxdem.State.stack` and a similar :class:`jaxdem.System` stack.

        Raises
        ------
        ValueError
            If `n` or `stride` are non-positive. (Implicit by `jax.lax.scan` length)

        Example
        -------
        >>> import jaxdem as jdem
        >>>
        >>> state = jdem.utils.grid_state(n_per_axis=(1, 1), spacing=1.0, radius=0.1)
        >>> system = jdem.System.create(state_shape=state.shape, dt=0.01)
        >>>
        >>> # Roll out for 10 frames, saving every 5 steps
        >>> final_state, final_system, traj = jdem.System.trajectory_rollout(
        ...     state, system, n=10, stride=5
        ... )
        >>>
        >>> print(f"Total simulation steps performed: {10 * 5}")
        >>> print(f"Trajectory length (number of frames): {traj[0].pos.shape[0]}")  # traj[0] is the state part of the trajectory
        >>> print(f"First frame position:\n{traj[0].pos[0]}")
        >>> print(f"Last frame position:\n{traj[0].pos[-1]}")
        >>> print(f"Final state position (should match last frame):\n{final_state.pos}")
        """

        @partial(jax.named_call, name="System.trajectory_rollout")
        def body(carry, _):
            st, sys = carry
            carry = sys._steps(st, sys, stride)
            return carry, carry

        if batched:
            body = jax.vmap(body, in_axes=(0, None))

        (state, system), traj = jax.lax.scan(body, (state, system), xs=None, length=n)
        return state, system, traj

    @staticmethod
    @partial(jax.named_call, name="System.step")
    def step(
        state: "State",
        system: "System",
        *,
        n: int = 1,
        batched: bool = False,
    ) -> Tuple["State", "System"]:
        """
        Advances the simulation state by `n` time steps.

        This method provides a convenient way to run multiple integration steps.
        For a single step (`n=1`), it directly calls the integrator's step method.
        For multiple steps (`n > 1`), it uses an optimized internal loop based on
        `jax.lax.scan` to maintain JIT-compilation efficiency.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.
        n : int, optional
            The number of integration steps to perform. Defaults to 1. This argument must be static.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the final `State` and `System` after `n` integration steps.

        Raises
        ------
        ValueError
            If `n` is non-positive. (Implicit by `jax.lax.scan` length check).

        Example
        -------
        >>> import jaxdem as jdem
        >>>
        >>> state = jdem.utils.grid_state(n_per_axis=(1, 1), spacing=1.0, radius=0.1)
        >>> system = jdem.System.create(state.accel.shape, dt=0.01)
        >>>
        >>> # Advance by 1 step
        >>> state_after_1_step, system_after_1_step = jdem.System.step(state, system)
        >>> print("Position after 1 step:", state_after_1_step.pos[0])
        >>>
        >>> # Advance by 10 steps
        >>> state_after_10_steps, system_after_10_steps = jdem.System.step(state, system, n=10)
        >>> print("Position after 10 steps:", state_after_10_steps.pos[0])
        """

        body = system._steps

        if batched:
            body = jax.vmap(body, in_axes=(0, 0, None))

        return body(state, system, n)
