# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining bijectors that constraint the action spaceof reinforcement learning policy distributions.
"""

import jax
import jax.numpy as jnp

from typing import Tuple, Optional

import distrax
from distrax._src.bijectors.bijector import Array

from ..factory import Factory


class ActionSpace(Factory):
    """
    Registry/namespace for action-space **constraints** implemented as
    `distrax.Bijector`s.

    These bijectors are intended to be wrapped around a base policy
    distribution (e.g., `MultivariateNormalDiag`) via
    `distrax.Transformed`, so that sampling and log-probabilities are
    correctly adjusted using the bijector’s `forward_and_log_det` /
    `inverse_and_log_det` methods. See Distrax/TFP bijector interface
    for details on shape semantics and `event_ndims_in/out`.
    """

    __slots__ = ()


@ActionSpace.register("Free")
class FreeSpace(distrax.Bijector, ActionSpace):
    r"""
    Identity constraint (no transform).

    **Mapping**

    .. math::
        y = f(x) = x, \qquad x = f^{-1}(y) = y.

    **Jacobian**

    .. math::
        J_f(x) = I,\qquad \log\lvert\det J_f(x)\rvert = 0, \qquad \log\lvert\det J_{f^{-1}}(y)\rvert = 0.

    Parameters
    ----------
    -event_ndims_in : int
        dimensionality of a *single event* seen by the bijector (defaults to 0 for a scalar transform).

    -event_ndims_out : Optional[int]
        standard Distrax/TFP bijector flags.

    -is_constant_jacobian : bool
        standard Distrax/TFP bijector flags.

    -is_constant_log_det : bool
        standard Distrax/TFP bijector flags.

    Note
    ----------
    This bijector is **scalar** (``event_ndims_in = 0``). For vector actions,
    needs to be wrap it with ``distrax.Block(bijector, ndims=1)`. Let the model do that for you!
    """

    __slots__ = ()

    def __init__(
        self,
        event_ndims_in: int = 0,
        event_ndims_out: Optional[int] = None,
        is_constant_jacobian: bool = True,
        is_constant_log_det: Optional[bool] = True,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )

    def forward_and_log_det(self, x: Array) -> Tuple[Array, jax.Array]:
        # log|det J| = 0 for identity; shape matches x for a scalar bijector
        return x, jnp.zeros_like(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, jax.Array]:
        # inverse is identity; log|det J_inv| = 0
        return y, jnp.zeros_like(y)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is FreeSpace  # pylint: disable=unidiomatic-typecheck


@ActionSpace.register("Box")
class BoxSpace(distrax.Bijector, ActionSpace):
    r"""
    Elementwise **box** constraint implemented with a scaled `tanh`.

    **Mapping (componentwise)**

    .. math::
        y_i \;=\; c_i + h_i\,\tanh\!\left(\frac{x_i}{w}\right),
        \qquad c_i=\tfrac{1}{2}(x_{\min,i}+x_{\max,i}),
        \quad h_i=\tfrac{1-\varepsilon}{2}(x_{\max,i}-x_{\min,i}),

    with width parameter (:math:`w>0`) and small (:math:`\epsilon>0`) for numerical safety.

    **Jacobian (componentwise)**
    For each component,

    .. math::
        \frac{\partial y_i}{\partial x_i} = \frac{h_i}{w} sech^2 \left(\frac{x_i}{w}\right),
        \qquad
        \log\left| \frac{\partial y_i}{\partial x_i} \right| = \log h_i - \log w + \log\!\big(sech^2(\frac{x_i}{w})\big).

    Using the stable identity :math:`\log(sech^2 z)=2 [\log 2 - z - softplus(-2z)]`,
    which we apply for good numerical behavior.

    Parameters
    ----------
    -x_min : jax.Array
        Elementwise lower bounds of the distribution.

    -x_max : jax.Array
        Elementwise upper bounds of the distribution. Must satisfy x_max > x_min elementwise.

    -width : float
        slope control.

    -eps : float
        Small offset to avoid arctanh divergence close to bounds.

    -event_ndims_in : int
        dimensionality of a *single event* seen by the bijector (defaults to 0 for a scalar transform).

    -event_ndims_out : Optional[int]
        standard Distrax/TFP bijector flags.

    -is_constant_jacobian : bool
        standard Distrax/TFP bijector flags.

    -is_constant_log_det : bool
        standard Distrax/TFP bijector flags.

    Note
    ----------
    This bijector is **scalar** (``event_ndims_in = 0``). For vector actions,
    needs to be wrap it with ``distrax.Block(bijector, ndims=1)`. Let the model do that for you!
    """

    __slots__ = ()

    def __init__(
        self,
        x_min: Array,
        x_max: Array,
        width: float = 1.0,
        eps: float = 1e-6,
        event_ndims_in: int = 0,
        event_ndims_out: Optional[int] = None,
        is_constant_jacobian: bool = False,
        is_constant_log_det: Optional[bool] = None,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )
        x_min = jnp.asarray(x_min, dtype=float)
        x_max = jnp.asarray(x_max, dtype=float)
        if not jnp.all(x_max > x_min):
            raise ValueError("Box: require x_max > x_min elementwise.")

        self.center = (x_min + x_max) / 2.0
        self.half = (1.0 - eps) * (x_max - x_min) / 2.0
        self.width = width
        self.eps = float(eps)

    @staticmethod
    def sec2_log(x):
        return 2 * (jnp.log(2) - x - jax.nn.softplus(-2.0 * x))

    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        """
        Computes log|det J(f)(x)|.
        log|dy/dx| = log|half| + log(sech^2 x)
        Stable log(sech^2 x) = 2*(log(2) - x - softplus(-2x))
        """
        return jnp.log(self.half) + self.sec2_log(x / self.width) - jnp.log(self.width)

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.center + self.half * jnp.tanh(x / self.width)
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        u = (y - self.center) / (self.half + self.eps)
        u = u.clip(-1.0 + self.eps, 1.0 - self.eps)
        x = self.width * jnp.arctanh(u)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is BoxSpace  # pylint: disable=unidiomatic-typecheck


@ActionSpace.register("MaxNorm")
class MaxNormSpace(distrax.Bijector, ActionSpace):
    r"""
    **Radial max-norm** constraint for vector actions:
    scales the radius with a `tanh` squashing while preserving direction.

    **Mapping (vector case,** :math:`x \in \mathbb{R}^d`

    .. math::
        r = \lVert \vec{x} \rVert_2,\qquad
        \hat{u} = \begin{cases}
            \frac{\vec{x}}{r}, & r>0,\\[4pt]
            0, & r=0,
        \end{cases}
        \qquad
        y = s \tanh(r) \hat{u},
        \quad s = (1-\epsilon) \texttt{max_norm}.

    Equivalently, :math:`y = b(r)\,x` with :math:`b(r)= s\,\tanh(r)/r` for :math:`r>0`.


    **Jacobian determinant**

    For an isotropic radial map :math:`f(x)=b(r)` with :math:`x \in \mathbb{R}^d`, the Jacobian
    eigenvalues are :math:`b` (multiplicity d-1) on the tangent subspace and :math:`b + r\,b'(r)` on the radial direction, hence

    .. math::
        \bigl|\det J_f(x)\bigr| = b(r)^{\,d-1}\,\bigl(b(r)+r\,b'(r)\bigr)
        = s^d \left(\frac{\tanh r}{r}\right)^{\!d-1} sech^2 r.

    Therefore

    .. math::
        \log\lvert\det J_f(x)\rvert
        = d\log s + (d-1)\bigl(\log\tanh r - \log r\bigr) + \log( sech^2 r),

    We use the stable identity :math:`\log(sech^2 z)=2 [\log 2 - z - softplus(-2z)]`,
    which we apply for good numerical behavior.

    Near :math:`r\approx 0`, we use the second-order expansion

    .. math::
        \log\lvert\det J_f(x)\rvert \approx d\log s - \tfrac{2}{3} r^2

    to avoid division by :math:`r`.

    Parameters
    ----------
    -max_norm : float
        target radius \(s\) after squashing (default 1.0). We actually use \(s=(1-\varepsilon)\,\texttt{max\_norm}\) to avoid the exact boundary.

    -eps : float
        numerical safety margin used near \(r=0\) and \(r\to\infty\).

    -event_ndims_in : int
        dimensionality of a *single event* seen by the bijector (defaults to 0 for a scalar transform).

    -event_ndims_out : Optional[int]
        standard Distrax/TFP bijector flags.

    -is_constant_jacobian : bool
        standard Distrax/TFP bijector flags.

    -is_constant_log_det : bool
        standard Distrax/TFP bijector flags.

    Note
    ----------
    This bijector is **vector-valued** with ``event_ndims_in = 1`` (i.e., it operates on length-\(d\) action vectors as a
    single event). Do **not** wrap it in `Block` unless you intend to apply it independently to multiple last-axis blocks.
    """

    __slots__ = ()

    def __init__(
        self,
        max_norm: float = 1.0,
        eps: float = 1e-6,
        event_ndims_in: int = 1,
        event_ndims_out: Optional[int] = None,
        is_constant_jacobian: bool = False,
        is_constant_log_det: Optional[bool] = None,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )
        self.eps = float(eps)
        self.max_norm = float(max_norm)
        self.max_norm = (1.0 - self.eps) * self.max_norm

    @staticmethod
    def sec2_log(r):
        # r is scalar radius
        return 2 * (jnp.log(2.0) - r - jax.nn.softplus(-2.0 * r))

    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        r = jnp.linalg.norm(x, axis=-1)  # shape (...,)
        x = jnp.atleast_1d(x)  # ensures x.ndim >= 1
        d = jnp.asarray(x.shape[-1], x.dtype)  # scalar, works under jit

        # Stable pieces
        log_s = jnp.log(self.max_norm + self.eps)
        log_tanh_r = jnp.log(jnp.tanh(r) + self.eps)
        log_r = jnp.log(r + self.eps)
        log_sech2_r = MaxNormSpace.sec2_log(r)

        main = d * log_s + (d - 1.0) * (log_tanh_r - log_r) + log_sech2_r
        small = d * log_s - (2.0 / 3.0) * (r * r)
        return jnp.where(r < self.eps, small, main)

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        unit = jnp.where(r > 0.0, x / r, jnp.zeros_like(x))
        y = self.max_norm * jnp.tanh(r) * unit
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(y, axis=-1, keepdims=True)
        u = (r / self.max_norm).clip(-1.0 + self.eps, 1.0 - self.eps)
        unit = jnp.where(r > 0.0, y / r, jnp.zeros_like(y))
        x = jnp.arctanh(u) * unit
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is MaxNormSpace
