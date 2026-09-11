# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""The simulation configuration and the tools that drive the simulation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast, final

import jax
import jax.numpy as jnp

from .bonded_forces import BondedForceModel
from .colliders import Collider
from .domains import Domain
from .forces import ForceManager, ForceModel
from .integrators import LinearIntegrator, RotationIntegrator
from .materials import Material, MaterialTable

if TYPE_CHECKING:
    from .minimizers import MinimizationResult
    from .state import State
    from .simulation_status import StepResult


def _check_material_table(table: MaterialTable, required: Sequence[str]) -> None:
    """Check that a MaterialTable contains all properties that a force model requires.

    The :class:`ForceModel` lists the required properties in
    :attr:`ForceModel.required_material_properties`. Each one must be an
    attribute of the given :class:`MaterialTable`.

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


@jax.jit(inline=True)
def _save_state_system(state: State, system: System) -> tuple[State, System]:
    return state, system


@jax.jit(inline=True)
def _step_once(state: State, system: System) -> tuple[State, System]:
    system = dataclasses.replace(
        system,
        time=system.time + system.dt,
        step_count=system.step_count + 1,
    )
    state, system = system.user_pre_step_actions(state, system)
    state, system = system.domain.apply(state, system)
    system = dataclasses.replace(
        system,
        domain=dataclasses.replace(
            system.domain, inv_box_size=1.0 / system.domain.box_size
        ),
    )
    state, system = system.linear_integrator.step_before_force(state, system)
    state, system = system.rotation_integrator.step_before_force(state, system)
    if system.bonded_force_model is not None:
        bonded_force_model = system.bonded_force_model.update_reference_state(
            state.pos, state, system
        )
        system = dataclasses.replace(system, bonded_force_model=bonded_force_model)
    state, system = system.collider.compute_force(state, system)
    system = dataclasses.replace(
        system,
        search_overflow=system.search_overflow | system.collider.overflow,
    )
    state, system = system.force_manager.apply(state, system)
    state, system = system.linear_integrator.step_after_force(state, system)
    state, system = system.rotation_integrator.step_after_force(state, system)
    state, system = system.linear_integrator.finalize_step(state, system)
    state, system = system.rotation_integrator.finalize_step(state, system)
    state, system = system.user_post_step_actions(state, system)
    return state, system


@jax.jit(inline=True)
def _steps_fori_loop(
    state: State, system: System, n: int | jax.Array
) -> tuple[State, System]:
    return jax.lax.fori_loop(0, n, lambda i, carry: _step_once(*carry), (state, system))


@jax.jit(inline=True, static_argnames=("n",))
def _steps_fori_loop_unrolled(
    state: State, system: System, n: int
) -> tuple[State, System]:
    return jax.lax.fori_loop(
        0, n, lambda i, carry: _step_once(*carry), (state, system), unroll=2
    )


@jax.jit(static_argnames=("n", "stride", "save_fn", "unroll"))
def _trajectory_rollout(
    state: State,
    system: System,
    strides: jax.Array | None,
    *,
    n: int | None,
    stride: int,
    save_fn: Callable[[State, System], Any],
    unroll: int,
) -> tuple[tuple[State, System], Any]:
    def scan_fn(
        carry: tuple[State, System], xs: jax.Array | None
    ) -> tuple[tuple[State, System], Any]:
        state, system = carry
        if xs is None:
            state, system = System.step(state, system, n=stride)
        else:
            state, system = System.step_dynamic(state, system, n=xs)
        return (state, system), save_fn(state, system)

    return jax.lax.scan(scan_fn, (state, system), length=n, xs=strides, unroll=unroll)


@final
@jax.tree_util.register_dataclass
@dataclass
class System:
    """The full simulation configuration.

    Notes:
    ------
    - The `System` object supports JIT compilation for efficient execution.
    - The `System` dataclass is compatible with :func:`jax.jit`, so every field should remain JAX arrays for best performance.

    Example:
    --------
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

    linear_integrator: LinearIntegrator
    """Instance of :class:`jaxdem.LinearIntegrator` that advances the simulation linear state in time."""

    rotation_integrator: RotationIntegrator
    """Instance of :class:`jaxdem.RotationIntegrator` that advances the simulation angular state in time."""

    collider: Collider
    """Instance of :class:`jaxdem.Collider` that performs contact detection and computes inter-particle forces and potential energies."""

    domain: Domain
    """Instance of :class:`jaxdem.Domain` that defines the simulation boundaries, displacement rules, and boundary conditions."""

    force_manager: ForceManager
    """Instance of :class:`jaxdem.ForceManager` that handles per particle forces like external forces and resets forces."""

    bonded_force_model: BondedForceModel | None
    """
    Optional instance of :class:`jaxdem.BondedForceModel` that defines bonded interactions
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
    """PRNG key for stochastic operations. Always update it with split so each use gets new random numbers."""

    interact_same_bond_id: jax.Array
    """
    Boolean scalar controlling interactions between particles connected by ``bond_id`` adjacency.

    If ``False`` (default), colliders mask out directly bonded pairs.
    If ``True``, these pairs interact.
    """

    search_overflow: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(False)
    )
    """Sticky flag: a force evaluation used an incomplete spatial search.

    Check with :meth:`check_overflow` at host chunk boundaries. Retry from the
    last valid state after resizing; clearing this flag cannot repair a trajectory.
    """

    user_pre_step_actions: Callable[[State, System], tuple[State, System]] = (
        jax.tree.static(default=_save_state_system)
    )
    """Function called before every step to perform user-defined actions."""

    user_post_step_actions: Callable[[State, System], tuple[State, System]] = (
        jax.tree.static(default=_save_state_system)
    )
    """Function called after every step to perform user-defined actions."""

    minimizer: Any = jax.tree.static(default=None)
    """An optax GradientTransformation wrapped in CustomGradientTransformation, used for target_fn minimization."""

    target_fn: Callable[[State, System], jax.Array] | None = jax.tree.static(
        default=None
    )
    """Optional custom target evaluation function for minimization."""

    @staticmethod
    @partial(jax.named_call, name="System.create")
    def create(
        state_shape: tuple[int, ...] | None = None,
        *,
        state: State | None = None,
        dt: float = 0.005,
        time: float = 0.0,
        linear_integrator_type: str | None = "verlet",
        rotation_integrator_type: str | None = "verletspiral",
        collider_type: str = "naive",
        domain_type: str = "free",
        bonded_force_model_type: str | None = None,
        bonded_force_model_kw: dict[str, Any] | None = None,
        bonded_force_manager_kw: dict[str, Any] | None = None,
        bonded_force_model: BondedForceModel | None = None,
        force_model_type: str = "spring",
        force_manager_kw: dict[str, Any] | None = None,
        mat_table: MaterialTable | None = None,
        linear_integrator: LinearIntegrator | None = None,
        rotation_integrator: RotationIntegrator | None = None,
        collider: Collider | None = None,
        domain: Domain | None = None,
        force_model: ForceModel | None = None,
        force_manager: ForceManager | None = None,
        linear_integrator_kw: dict[str, Any] | None = None,
        rotation_integrator_kw: dict[str, Any] | None = None,
        collider_kw: dict[str, Any] | None = None,
        domain_kw: dict[str, Any] | None = None,
        force_model_kw: dict[str, Any] | None = None,
        seed: int = 0,
        key: jax.Array | None = None,
        interact_same_bond_id: bool = False,
        user_pre_step_actions: (
            Callable[[State, System], tuple[State, System]] | None
        ) = None,
        user_post_step_actions: (
            Callable[[State, System], tuple[State, System]] | None
        ) = None,
        minimizer: Any = None,
        minimizer_kw: dict[str, Any] | None = None,
        target_fn: Callable[[State, System], jax.Array] | None = None,
    ) -> System:
        """Factory method to create a :class:`System` instance with specified components.

        Every component slot accepts either a pre-built instance
        (``linear_integrator``, ``rotation_integrator``, ``collider``,
        ``domain``, ``force_model``, ``force_manager``, ``bonded_force_model``,
        ``mat_table``) or a registered type string plus keyword dict
        (``<component>_type`` / ``<component>_kw``). When you provide an
        instance, the method uses it as-is and ignores the corresponding
        ``*_type`` / ``*_kw`` arguments.

        Parameters
        ----------
        state_shape : Tuple, optional
            Shape of the state tensors handled by the simulation. The penultimate
            dimension corresponds to the number of particles ``N`` and the last
            dimension corresponds to the spatial dimension ``dim``. You can omit
            it when you provide ``state``.
        state : State, optional
            The initial simulation state. When provided, the method infers
            ``state_shape`` from it and forwards the state to colliders whose
            ``Create`` method requires one (e.g. ``"CellList"``,
            ``"NeighborList"``), so ``collider_kw={"state": state}`` is not
            needed.
        dt : float, optional
            The global simulation time step.
        linear_integrator_type : str or None, optional
            The registered type string for the :class:`jaxdem.integrators.LinearIntegrator`
            used to evolve translational degrees of freedom. ``None`` (or the
            empty string) disables linear integration (no-op integrator).
        rotation_integrator_type : str or None, optional
            The registered type string for the :class:`jaxdem.integrators.RotationIntegrator`
            used to evolve angular degrees of freedom. ``None`` (or the empty
            string) disables rotational integration (no-op integrator).
        collider_type : str, optional
            The registered type string for the :class:`jaxdem.Collider` to use.
        domain_type : str, optional
            The registered type string for the :class:`jaxdem.Domain` to use.
        bonded_force_model_type : str or None, optional
            The registered type string for the :class:`jaxdem.BondedForceModel` to use.
        bonded_force_model_kw : Dict[str, Any] or None, optional
            Keyword arguments forwarded to ``BondedForceModel.create``.
        bonded_force_manager_kw : Dict[str, Any] or None, optional
            Deprecated alias of ``bonded_force_model_kw`` (the dict has always
            been forwarded to the bonded force *model*, not the manager).
        force_model_type : str, optional
            The registered type string for the :class:`jaxdem.ForceModel` to use.
        force_manager_kw : Dict[str, Any] or None, optional
            Keyword arguments to pass to the constructor of `ForceManager`.
        mat_table : MaterialTable or None, optional
            An optional pre-configured :class:`jaxdem.MaterialTable`. If `None`, the
            method creates a default `jaxdem.MaterialTable` with one generic elastic material and the "harmonic" `jaxdem.MaterialMatchmaker`.
        linear_integrator : LinearIntegrator, optional
            Pre-built linear integrator instance. Overrides
            ``linear_integrator_type`` / ``linear_integrator_kw``.
        rotation_integrator : RotationIntegrator, optional
            Pre-built rotation integrator instance. Overrides
            ``rotation_integrator_type`` / ``rotation_integrator_kw``.
        collider : Collider, optional
            Pre-built collider instance. Overrides ``collider_type`` /
            ``collider_kw``.
        domain : Domain, optional
            Pre-built domain instance. Overrides ``domain_type`` / ``domain_kw``.
        force_model : ForceModel, optional
            Pre-built force model instance. Overrides ``force_model_type`` /
            ``force_model_kw``.
        force_manager : ForceManager, optional
            Pre-built force manager instance. Overrides ``force_manager_kw``.
            Cannot be combined with a bonded force model (the bonded force
            functions must already be part of the provided manager).
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
            Used only when ``key`` is not provided.
        key : jax.Array, optional
            Key for JAX random number generation. When you provide ``key``,
            the method ignores ``seed``.
        interact_same_bond_id : bool, optional
            Whether particles with the same bond_id interact. Defaults to False.
        user_pre_step_actions : Callable, optional
            A function called before every time step to perform user-defined actions.
        user_post_step_actions : Callable, optional
            A function called after every time step to perform user-defined actions.
        minimizer : Callable, optional
            Optimizer factory used by :meth:`System.minimize`. Called as
            ``minimizer(**minimizer_kw)`` and must return an optax-style
            ``GradientTransformation``. Defaults to FIRE
            (:func:`jaxdem.minimizers.fire`).
        minimizer_kw : Dict[str, Any] or None, optional
            Keyword arguments passed to ``minimizer``. If the minimizer's
            signature accepts a ``dt`` parameter and none is given here, the
            method passes the system ``dt`` automatically.
        target_fn : Callable, optional
            Custom objective ``(state, system) -> scalar`` minimized by
            :meth:`System.minimize`. When ``None``, :meth:`System.minimize`
            uses the total potential energy.

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
        ValueError
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
        if state is not None:
            if state_shape is not None and tuple(state_shape) != tuple(state.shape):
                raise ValueError(
                    f"state_shape {tuple(state_shape)} does not match the shape "
                    f"of the provided state {tuple(state.shape)}."
                )
            state_shape = tuple(state.shape)
        if state_shape is None:
            raise TypeError("System.create requires either `state_shape` or `state`.")

        dim = state_shape[-1]
        # `None` disables an integrator (registered as the no-op "" integrator).
        if linear_integrator_type is None:
            linear_integrator_type = ""
        if rotation_integrator_type is None:
            rotation_integrator_type = ""

        if bonded_force_manager_kw is not None:
            import warnings

            warnings.warn(
                "`bonded_force_manager_kw` is deprecated (the dict is forwarded "
                "to BondedForceModel.create, not to the manager); use "
                "`bonded_force_model_kw` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if bonded_force_model_kw is None:
                bonded_force_model_kw = bonded_force_manager_kw

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
            {
                "gravity": None,
                "force_functions": (),
            }
            if force_manager_kw is None
            else dict(force_manager_kw)
        )

        if mat_table is None:
            mat_table = MaterialTable.from_materials(
                [Material.create("elastic", density=0.27, young=1.0e4, poisson=0.3)],
            )

        if force_model is None:
            force_model = ForceModel.create(force_model_type, **force_model_kw)
        _check_material_table(mat_table, force_model.required_material_properties)

        if bonded_force_model is None and bonded_force_model_type is not None:
            bonded_force_model_kw = (
                {} if bonded_force_model_kw is None else dict(bonded_force_model_kw)
            )
            bonded_force_model = BondedForceModel.create(
                bonded_force_model_type, **bonded_force_model_kw
            )

        if force_manager is None:
            if bonded_force_model is not None:
                force_manager_kw.setdefault("force_functions", ())
                force_manager_kw["force_functions"] = (
                    *tuple(force_manager_kw["force_functions"]),
                    bonded_force_model.force_and_energy_fns,
                )
            force_manager = ForceManager.create(state_shape, **force_manager_kw)
        elif bonded_force_model is not None:
            raise ValueError(
                "Cannot combine a pre-built `force_manager` with a "
                "`bonded_force_model`: the bonded force functions must already "
                "be registered on the provided manager. Pass `force_manager_kw` "
                "instead, or build the manager with the bonded force functions "
                "included."
            )

        if key is None:
            key = jax.random.PRNGKey(seed)

        if user_pre_step_actions is None:
            user_pre_step_actions = _save_state_system

        if user_post_step_actions is None:
            user_post_step_actions = _save_state_system

        if minimizer is None:
            from .minimizers import fire

            minimizer = fire

        minimizer_kw = {} if minimizer_kw is None else dict(minimizer_kw)
        import inspect

        try:
            sig = inspect.signature(minimizer)
            if "dt" in sig.parameters:
                minimizer_kw.setdefault("dt", dt)
        except (ValueError, TypeError):
            pass
        opt_obj = minimizer(**minimizer_kw)

        from .minimizers.optimizers import CustomGradientTransformation

        minimizer_wrapped = CustomGradientTransformation(
            opt_obj.init,
            opt_obj.update,
            minimizer,
            minimizer_kw,
            type_name=getattr(minimizer, "__name__", ""),
        )

        if collider is None:
            if state is not None and "state" not in collider_kw:
                # Forward the state to colliders whose Create method needs one
                # (CellList, MultiCellList, NeighborList, ...).
                from inspect import signature

                from .factory import _normalize_key

                sub_cls = Collider._registry.get(_normalize_key(collider_type))
                create_fn = getattr(sub_cls, "Create", None)
                if create_fn is not None and "state" in signature(create_fn).parameters:
                    collider_kw["state"] = state
            collider = Collider.create(collider_type, **collider_kw)

        history_shape = force_model.history_shape(state_shape[-1])
        if history_shape != (0,) and not collider.supports_history:
            force_name = type(force_model).__name__
            collider_name = type(collider).__name__
            raise ValueError(
                f"Force model '{force_name}' has nonempty pair history "
                f"{history_shape}, but collider '{collider_name}' cannot persist it."
            )
        if collider.supports_history:
            from dataclasses import replace

            pair_shape = tuple(state_shape[:-1]) + (cast(Any, collider).max_neighbors,)
            expected_shape = pair_shape + history_shape
            history = cast(Any, collider).history
            if history.shape == pair_shape + (0,) and history_shape != (0,):
                history = force_model.init_history(pair_shape, state_shape[-1])
                collider = replace(cast(Any, collider), history=history)
            elif history.shape != expected_shape:
                raise ValueError(
                    f"NeighborList history has shape {history.shape}, expected "
                    f"{expected_shape} for force model '{type(force_model).__name__}'."
                )

        if domain is None:
            domain = Domain.create(domain_type, dim=dim, **domain_kw)
        collider.validate_domain(domain)

        return System(
            linear_integrator=(
                LinearIntegrator.create(linear_integrator_type, **linear_integrator_kw)
                if linear_integrator is None
                else linear_integrator
            ),
            rotation_integrator=(
                RotationIntegrator.create(
                    rotation_integrator_type, **rotation_integrator_kw
                )
                if rotation_integrator is None
                else rotation_integrator
            ),
            collider=collider,
            domain=domain,
            force_manager=force_manager,
            bonded_force_model=bonded_force_model,
            force_model=force_model,
            mat_table=mat_table,
            dim=jnp.asarray(dim, dtype=int),
            dt=jnp.asarray(dt, dtype=float),
            time=jnp.asarray(time, dtype=float),
            step_count=jnp.asarray(0, dtype=int),
            key=key,
            interact_same_bond_id=jnp.asarray(interact_same_bond_id, dtype=bool),
            user_pre_step_actions=user_pre_step_actions,
            user_post_step_actions=user_post_step_actions,
            minimizer=minimizer_wrapped,
            target_fn=target_fn,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="System.initialize")
    def initialize(state: State, system: System) -> tuple[State, System]:
        """Compute initial forces and initialize both configured integrators.

        Call explicitly after constructing the state and system, before the
        first integration step. This method supports one snapshot or matching
        stacked states and systems, like :meth:`step`.

        Bounds and collider caches are updated, pair and managed forces are
        evaluated, then the linear and rotational ``initialize`` hooks run.
        An integrator can use its hook to stagger velocities, for example by
        a backward half-kick. The default integrator hook leaves state unchanged.

        Initialization does not advance time, step count, contact/plastic
        history, or timestep callbacks. Managed force buffers follow their
        ordinary apply-and-clear semantics. No initialized flag is stored:
        callers must initialize new trajectories themselves and must not repeat
        integrator setup when continuing an initialized checkpoint.

        Example
        -------
        >>> state, system = System.initialize(state, system)
        >>> state, system = System.step(state, system, n=100)
        """
        state_rank = state.pos_c.ndim
        if state_rank not in (2, 3) or system.dt.ndim != state_rank - 2:
            raise ValueError(
                "System.initialize() requires matching state and system layouts: "
                "(N, dim) with a scalar system, or (B, N, dim) with a stacked system. "
                f"Got state.pos_c.shape={state.pos_c.shape} and "
                f"system.dt.shape={system.dt.shape}."
            )

        def initialize_snapshot(st: State, sys: System) -> tuple[State, System]:
            sys = sys.domain.update_bounds(st.pos, sys, padding=st.rad)
            st, sys = sys.collider.evaluate_force(st, sys)
            st, sys = sys.force_manager.apply(st, sys)
            sys = dataclasses.replace(
                sys, search_overflow=sys.search_overflow | sys.collider.overflow
            )
            st, sys = sys.linear_integrator.initialize(st, sys)
            return sys.rotation_integrator.initialize(st, sys)

        if state_rank == 3:
            return jax.vmap(initialize_snapshot)(state, system)
        return initialize_snapshot(state, system)

    @staticmethod
    @jax.jit
    def evaluate_forces(state: State, system: System) -> tuple[State, System]:
        """Inspect instantaneous forces without advancing physical state.

        Return updated forces and search caches while preserving contact and
        plastic history, queued loads, time, and integrator state. Queued loads
        contribute to the returned forces but remain available for the next
        physical step. This is an observation, not integrator initialization.
        Custom force callbacks must themselves be pure evaluations.
        """
        rank = state.pos_c.ndim
        if rank not in (2, 3) or system.dt.ndim != rank - 2:
            raise ValueError(
                "System.evaluate_forces() requires matching state and system layouts."
            )

        def evaluate_snapshot(st: State, sys: System) -> tuple[State, System]:
            sys = sys.domain.update_bounds(st.pos, sys, padding=st.rad)
            manager = sys.force_manager
            sys = dataclasses.replace(sys, force_manager=dataclasses.replace(manager))
            st, sys = sys.collider.evaluate_force(st, sys)
            st, sys = sys.force_manager.apply(st, sys)
            return st, dataclasses.replace(
                sys,
                force_manager=manager,
                search_overflow=sys.search_overflow | sys.collider.overflow,
            )

        if rank == 3:
            return jax.vmap(evaluate_snapshot)(state, system)
        return evaluate_snapshot(state, system)

    def check_overflow(self) -> None:
        """Raise at a host boundary if this trajectory used truncated searches.

        This synchronizes the status flag, so call once per completed chunk,
        outside compiled loops. A successful later query does not clear it.
        """
        if bool(jnp.any(self.search_overflow | self.collider.overflow)):
            raise RuntimeError(
                "Spatial search overflow: resize the collider and retry from "
                "a valid pre-overflow state."
            )

    def validate(self, state: State, *, strict_clumps: bool = True) -> None:
        """Run opt-in host validation for a state against this system."""
        self.collider.validate_domain(self.domain)
        if not bool(jnp.all(jnp.isfinite(self.dt))) or bool(jnp.any(self.dt <= 0)):
            raise ValueError("System.dt must be finite and positive.")
        if not bool(jnp.all(jnp.isfinite(self.time))):
            raise ValueError("System.time must be finite.")
        if not bool(jnp.all(jnp.isfinite(self.domain.box_size))) or bool(
            jnp.any(self.domain.box_size <= 0)
        ):
            raise ValueError("Domain box_size must be finite and positive.")
        if not bool(jnp.all(jnp.isfinite(self.domain.anchor))):
            raise ValueError("Domain anchor must be finite.")
        if not bool(jnp.allclose(self.domain.inv_box_size, 1.0 / self.domain.box_size)):
            raise ValueError("Domain inv_box_size is inconsistent with box_size.")
        state.validate(
            num_materials=len(self.mat_table),
            num_species=self.force_model.species_capacity,
            strict_clumps=strict_clumps,
        )

    @staticmethod
    def trajectory_rollout(
        state: State,
        system: System,
        *,
        n: int | None = None,
        stride: int = 1,
        strides: jax.Array | None = None,
        save_fn: Callable[[State, System], Any] = _save_state_system,
        unroll: int = 2,
    ) -> tuple[State, System, Any]:
        """Roll the system forward while collecting saved outputs at each frame.

        The rollout always stores one output per frame via `save_fn(state, system)`.
        The output of save_fn must be a pytree. Frame spacing can be either:
        - constant (`stride`), or
        - variable (`strides` jax.Array).

        The rollout saves each frame *after* its integration steps, so it does
        not store the initial (step-0) state. To record it, save it before
        the rollout, or pass a leading ``0`` entry in `strides`.

        Parameters
        ----------
        state : State
            Initial state.
        system : System
            Initial system configuration.
        n : int, optional
            Number of saved frames.
            Required when `strides` is `None`.
            The method ignores `n` when you provide `strides`.
        stride : int, optional
            Constant number of integration steps between consecutive saves.
            Used only when `strides` is `None`. Defaults to 1.
        strides : jax.Array, optional
            Integer 1D array of per-frame integration strides. When provided,
            this overrides `stride`, and the method infers `n` from `len(strides)`.
        save_fn : Callable[[State, System], Any], optional
            Function called after each saved frame. The rollout stacks its
            return pytree along axis 0 across frames. Defaults to returning `(state, system)`.
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
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 0:
            raise ValueError("`stride` must be a nonnegative Python integer.")
        if isinstance(unroll, bool) or not isinstance(unroll, int) or unroll < 1:
            raise ValueError("`unroll` must be a positive Python integer.")
        if strides is not None:
            strides = jnp.asarray(strides)
            if strides.ndim != 1 or not jnp.issubdtype(strides.dtype, jnp.integer):
                raise ValueError("`strides` must be a 1D integer array.")
            if bool(jnp.any(strides < 0)):
                raise ValueError("`strides` entries must be nonnegative.")
            n = None
        else:
            if n is None:
                raise ValueError("`n` must be provided when `strides` is None.")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("`n` must be a nonnegative Python integer.")

        (state, system), traj = _trajectory_rollout(
            state,
            system,
            strides,
            n=n,
            stride=stride,
            save_fn=save_fn,
            unroll=unroll,
        )

        return state, system, traj

    @staticmethod
    @partial(jax.named_call, name="System.step")
    def step(
        state: State,
        system: System,
        *,
        n: int = 1,
    ) -> tuple[State, System]:
        """Advance the simulation by `n` integration steps.

        Parameters
        ----------
        state : State
            Current state.
        system : System
            Current system configuration.
        n : int, optional
            Static number of integration steps. Defaults to 1. Use
            :meth:`step_dynamic` for a traced scalar count.

        Returns
        -------
        Tuple[State, System]
            `(final_state, final_system)` after `n` steps.

        Example
        -------
        >>> # Advance by 10 steps
        >>> state_after_10_steps, system_after_10_steps = jdem.System.step(state, system, n=10)

        Notes
        -----
        Call :meth:`initialize` explicitly before stepping a new trajectory.
        This method performs no automatic force or integrator initialization.
        Checkpoint continuation uses its saved forces and integrator state
        without repeating initialization.

        This method does not check collider overflow, to avoid a host
        synchronization per step.

        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("System.step() requires a nonnegative Python integer `n`.")
        body = _steps_fori_loop_unrolled

        state_rank = state.pos_c.ndim
        system_rank = system.dt.ndim
        if state_rank not in (2, 3):
            raise ValueError(
                "System.step() expects state.pos_c with shape (N, dim) or "
                f"(B, N, dim). Got shape={state.pos_c.shape}."
            )
        if system_rank != state_rank - 2:
            raise ValueError(
                "System.step() requires matching state and system layouts: "
                "an unbatched state needs a scalar system, and a batched state "
                f"needs a stacked system. Got state.pos_c.shape={state.pos_c.shape} "
                f"and system.dt.shape={system.dt.shape}."
            )

        if state_rank == 3:
            body = jax.vmap(body, in_axes=(0, 0, None))
        state, system = body(state, system, n)

        return state, system

    @staticmethod
    def step_checked(state: State, system: System, *, n: int = 1) -> StepResult:
        """Advance at most ``n`` steps, stopping and rolling back on failure.

        Unlike :meth:`step`, this opt-in device loop checks finite particle
        state and spatial-search overflow after every step. ``result.steps``
        counts accepted steps; ``result.status`` preserves failure even though
        the returned state/system precede the failed step. An invalid input is
        returned unchanged with zero accepted steps. Check ``result.check()``
        on the host before continuing or writing successful output.

        Initialization and physical setup validation remain explicit. A batch
        returns one status/count per snapshot. This dynamic stopping loop does
        not support reverse-mode differentiation. No automatic retry changes
        the timestep, neighbor capacity, or physical history.
        """
        from .simulation_status import checked_steps

        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(
                "System.step_checked() requires a nonnegative Python integer `n`."
            )
        rank = state.pos_c.ndim
        if rank not in (2, 3) or system.dt.ndim != rank - 2:
            raise ValueError(
                "System.step_checked() requires matching state and system layouts."
            )
        if rank == 3:
            return jax.vmap(checked_steps, in_axes=(0, 0, None))(state, system, n)
        return checked_steps(state, system, n)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="System.step_dynamic")
    def step_dynamic(
        state: State, system: System, *, n: int | jax.Array
    ) -> tuple[State, System]:
        """Advance by a scalar integer count that may be traced at run time.

        This path uses ``lax.fori_loop`` with dynamic bounds. It is intended
        for compiled control flow such as variable trajectory strides. Reverse
        mode differentiation through dynamic loop bounds is not supported by
        JAX; use :meth:`step` with a Python integer for differentiable rollouts.
        """
        n_array = jnp.asarray(n)
        if n_array.ndim != 0 or not jnp.issubdtype(n_array.dtype, jnp.integer):
            raise ValueError("System.step_dynamic() requires a scalar integer `n`.")
        state_rank = state.pos_c.ndim
        if state_rank not in (2, 3) or system.dt.ndim != state_rank - 2:
            raise ValueError(
                "System.step_dynamic() requires (N, dim) with a scalar system "
                "or (B, N, dim) with a stacked system."
            )
        body = _steps_fori_loop
        if state_rank == 3:
            body = jax.vmap(body, in_axes=(0, 0, None))
        return body(state, system, n_array)

    @staticmethod
    @partial(jax.named_call, name="System.stack")
    def stack(systems: Sequence[System]) -> System:
        """Concatenate a sequence of :class:`System` snapshots into a trajectory or batch along axis 0.

        Use this method to collect simulation snapshots over time into a single
        `System` object where the leading dimension represents time, or to
        prepare a batched system.

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

        return jax.tree.map(lambda *xs: jnp.stack(xs), *systems)

    @staticmethod
    @partial(jax.named_call, name="System.unstack")
    def unstack(system: System) -> list[System]:
        """Split a stacked/batched :class:`System` along the leading axis into a Python list.

        This method is the inverse of :meth:`System.stack`:

        - If `stacked = System.stack([sys0, sys1, ...])`, then `System.unstack(stacked)` returns
          `[sys0, sys1, ...]`.

        Notes
        -----
        - The method splits along axis 0 (the leading axis).
        - This method cannot split a single snapshot `System`.

        """
        if system.dt.ndim < 1:
            raise ValueError(
                "System.unstack() expects a stacked/batched System with a leading axis "
                f"(dt.ndim >= 1). Got dt.shape={system.dt.shape}."
            )

        n = int(system.dt.shape[0])
        return [jax.tree.map(lambda x, i=i: x[i], system) for i in range(n)]

    @staticmethod
    @partial(jax.named_call, name="System.minimize")
    def minimize(
        state: State,
        system: System,
        *,
        max_steps: int = 10000,
        pe_tol: float = 1e-16,
        pe_diff_tol: float = 1e-16,
        force_tol: float = 0.0,
    ) -> MinimizationResult:
        """Minimize the energy of the system using the configured minimizer.

        Parameters
        ----------
        state : State
            The state of the simulation.
        system : System
            The system configuration.
        max_steps : int, optional
            The maximum number of steps to take. Defaults to 10000.
        pe_tol : float, optional
            The tolerance for the potential energy. Defaults to 1e-16.
        pe_diff_tol : float, optional
            The tolerance for the difference in potential energy. Defaults to 1e-16.

        force_tol : float, optional
            Maximum free-degree gradient tolerance. Defaults to 0.0.

        Returns
        -------
        MinimizationResult
            The final state, system, number of steps, and potential energy
            (per particle when no custom ``target_fn`` is set), plus a
            ``reason`` code. The result remains iterable as the historical
            four return values.

        Notes
        -----
        The loop stops as soon as **any** convergence criterion is met (energy
        tolerance, relative energy change, or force tolerance) — see
        :func:`jaxdem.minimizers.minimize` for the full list.
        """
        from .minimizers import minimize

        return minimize(
            state,
            system,
            max_steps=max_steps,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
            force_tol=force_tol,
        )

    def _serialize_force_functions(self) -> list[dict[str, Any]] | None:
        """Serialize user-supplied custom force functions to JSON."""
        from .forces.force_manager import default_energy_func
        from .utils import encode_callable

        fm = self.force_manager
        if not fm.force_functions:
            return None

        n_total = len(fm.force_functions)
        n_bonded = 1 if self.bonded_force_model is not None else 0
        n_user = n_total - n_bonded

        if n_user <= 0:
            return None

        entries: list[dict[str, Any]] = []
        for i in range(n_user):
            force_fn = fm.force_functions[i]
            energy_fn = fm.energy_functions[i]
            is_default_energy = energy_fn is default_energy_func

            entry: dict[str, Any] = {
                "force": encode_callable(force_fn),
                "energy": (
                    None
                    if is_default_energy or energy_fn is None
                    else encode_callable(energy_fn)
                ),
                "is_com": bool(fm.is_com_force[i]),
            }
            entries.append(entry)
        return entries

    @property
    def metadata(self) -> dict[str, Any]:
        """System configuration parameters for serialization and restoration."""
        from .utils import encode_callable

        return {
            "checkpoint_metadata_version": 2,
            "linear_integrator_type": self.linear_integrator.type_name,
            "linear_integrator_kw": self.linear_integrator.metadata or None,
            "rotation_integrator_type": self.rotation_integrator.type_name,
            "rotation_integrator_kw": self.rotation_integrator.metadata or None,
            "collider_type": self.collider.type_name,
            "domain_type": self.domain.type_name,
            "domain_kw": self.domain.metadata or None,
            "force_model_type": self.force_model.type_name,
            "bonded_force_model_type": (
                None
                if self.bonded_force_model is None
                else self.bonded_force_model.type_name
            ),
            "bonded_force_model_kw": (
                self.bonded_force_model.metadata
                if self.bonded_force_model is not None
                else None
            ),
            "mat_table_metadata": self.mat_table.metadata,
            "force_model_metadata": {
                "type": self.force_model.type_name,
                "kw": self.force_model.metadata,
            },
            "force_function_metadata": self._serialize_force_functions(),
            "collider_kw_metadata": self.collider.metadata or None,
            "minimizer": (
                self.minimizer.metadata
                if self.minimizer is not None and hasattr(self.minimizer, "metadata")
                else None
            ),
            "target_fn": (
                encode_callable(self.target_fn) if self.target_fn is not None else None
            ),
            "user_pre_step_actions": encode_callable(self.user_pre_step_actions),
            "user_post_step_actions": encode_callable(self.user_post_step_actions),
        }
