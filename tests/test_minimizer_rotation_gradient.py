"""An independent rigid-body energy constrains the quaternion optimizer chart."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

import jaxdem as jd
from jaxdem.utils import Quaternion
from jaxdem.minimizers.routines import (
    _objective_energy,
    _state_to_delta_params,
    _rotation_residual,
)


def test_nonidentity_quaternion_gradient_matches_independent_energy():
    body = np.array([[1.0, 0.0, 0.0], [-0.5, 0.6, 0.2], [-0.5, -0.6, -0.2]])
    target = np.array([[0.3, 0.5, -0.2], [-0.7, 0.1, 0.4], [0.2, -0.3, 0.6]])
    reference_rotation = Rotation.from_rotvec(np.deg2rad([20.0, -10.0, 35.0]))
    q = reference_rotation.as_quat(scalar_first=True)
    state = jd.State.add_clump(
        jd.State.create(dim=3), pos=jnp.asarray(body), rad=jnp.full(3, 0.1)
    )
    state = replace(
        state,
        q=Quaternion(
            w=jnp.full((3, 1), q[0]), xyz=jnp.broadcast_to(jnp.asarray(q[1:]), (3, 3))
        ),
    )

    def force(pos, st, sy):
        return -(pos - jnp.asarray(target)), jnp.zeros_like(st.torque)

    def energy(pos, st, sy):
        return 0.5 * jnp.sum((pos - jnp.asarray(target)) ** 2, axis=-1)

    system = jd.System.create(
        state=state,
        collider_type="",
        force_manager_kw={"force_functions": [(force, energy)]},
    )
    params = _state_to_delta_params(state)
    center = np.array([0.3, -0.2, 0.1])
    params["pos_c"] = params["pos_c"].at[0].set(jnp.asarray(center))
    body = np.asarray(state.pos_p) @ reference_rotation.as_matrix().T

    def independent(raw, extra=np.eye(3)):
        w, x, y, z = raw / np.linalg.norm(raw)
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + 2 * w * skew + 2 * skew @ skew
        return 0.5 * np.sum((body @ rotation.T @ extra.T + center - target) ** 2)

    raw0 = np.array([0.8, 0.3, -0.4, 0.1])
    for scale in (0.1, 1.0, 10.0):
        raw = raw0 * scale
        trial = {**params, "rotation": params["rotation"].at[0].set(jnp.asarray(raw))}
        value, gradient = jax.value_and_grad(
            lambda p: _objective_energy(p, state, system)[0]
        )(trial)
        actual = np.asarray(gradient["rotation"][0])
        step = 1e-5 * scale
        predicted = np.array(
            [
                (
                    independent(raw + np.eye(4)[i] * step)
                    - independent(raw - np.eye(4)[i] * step)
                )
                / (2 * step)
                for i in range(4)
            ]
        )
        np.testing.assert_allclose(value, independent(raw), rtol=2e-6)
        np.testing.assert_allclose(
            actual * scale, predicted * scale, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(actual @ raw, 0, atol=1e-6)
        angular = []
        for axis in np.eye(3):
            plus = Rotation.from_rotvec(axis * 1e-5).as_matrix()
            minus = Rotation.from_rotvec(-axis * 1e-5).as_matrix()
            angular.append((independent(raw, plus) - independent(raw, minus)) / (2e-5))
        np.testing.assert_allclose(
            _rotation_residual(jnp.asarray(raw[None]), jnp.asarray(actual[None]))[0],
            angular,
            rtol=2e-5,
            atol=2e-6,
        )
