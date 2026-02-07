# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Bessel functions for JAX.

Originally adapted from:
`https://github.com/benjaminpope/sibylla/blob/main/notebooks/bessel_test.ipynb`

Note:
This module does *not* change global JAX configuration (e.g. x64 enablement).
If you want 64-bit execution, set it in your application before importing JaxDEM:

    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
"""

from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

JaxArray: TypeAlias = Array

_RP1 = jnp.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)

_RQ1 = jnp.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

_PP1 = jnp.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)

_PQ1 = jnp.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

_QP1 = jnp.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)

_QQ1 = jnp.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

_YP1 = jnp.array(
    [
        1.26320474790178026440e9,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    ]
)
_YQ1 = jnp.array(
    [
        5.94301592346128195359e2,
        2.35564092943068577943e5,
        7.34811944459721705660e7,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    ]
)

_Z1 = 1.46819706421238932572e1
_Z2 = 4.92184563216946036703e1
_PIO4 = 0.78539816339744830962  # pi/4
_THPIO4 = 2.35619449019234492885  # 3*pi/4
_SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)

_PP0 = jnp.array(
    [
        7.96936729297347051624e-4,
        8.28352392107440799803e-2,
        1.23953371646414299388e0,
        5.44725003058768775090e0,
        8.74716500199817011941e0,
        5.30324038235394892183e0,
        9.99999999999999997821e-1,
    ]
)

_PQ0 = jnp.array(
    [
        9.24408810558863637013e-4,
        8.56288474354474431428e-2,
        1.25352743901058953537e0,
        5.47097740330417105182e0,
        8.76190883237069594232e0,
        5.30605288235394617618e0,
        1.00000000000000000218e0,
    ]
)

_QP0 = jnp.array(
    [
        -1.13663838898469149931e-2,
        -1.28252718670509318512e0,
        -1.95539544257735972385e1,
        -9.32060152123768231369e1,
        -1.77681167980488050595e2,
        -1.47077505154951170175e2,
        -5.14105326766599330220e1,
        -6.05014350600728481186e0,
    ]
)

_QQ0 = jnp.array(
    [
        1.0,
        6.43178256118178023184e1,
        8.56430025976980587198e2,
        3.88240183605401609683e3,
        7.24046774195652478189e3,
        5.93072701187316984827e3,
        2.06209331660327847417e3,
        2.42005740240291393179e2,
    ]
)

_YP0 = jnp.array(
    [
        1.55924367855235737965e4,
        -1.46639295903971606143e7,
        5.43526477051876500413e9,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    ]
)

_YQ0 = jnp.array(
    [
        1.04128353664259848412e3,
        6.26107330137134956842e5,
        2.68919633393814121987e8,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    ]
)

_DR10 = 5.78318596294678452118e0
_DR20 = 3.04712623436620863991e1

_RP0 = jnp.array(
    [
        -4.79443220978201773821e9,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    ]
)

_RQ0 = jnp.array(
    [
        1.0,
        4.99563147152651017219e2,
        1.73785401676374683123e5,
        4.84409658339962045305e7,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    ]
)


def _j1_small(x: ArrayLike) -> JaxArray:
    z = x * x
    w = jnp.polyval(_RP1, z) / jnp.polyval(_RQ1, z)
    w = w * x * (z - _Z1) * (z - _Z2)
    return w


def _j1_large_c(x: ArrayLike) -> JaxArray:
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(_PP1, z) / jnp.polyval(_PQ1, z)
    q = jnp.polyval(_QP1, z) / jnp.polyval(_QQ1, z)
    xn = x - _THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * _SQ2OPI / jnp.sqrt(x)


def j1(x: ArrayLike) -> JaxArray:
    """
    Bessel function of order one - using the implementation from CEPHES, translated to Jax.
    """
    return jnp.sign(x) * jnp.where(
        jnp.abs(x) < 5.0, _j1_small(jnp.abs(x)), _j1_large_c(jnp.abs(x))
    )


def _j0_small(x: ArrayLike) -> JaxArray:
    """
    Implementation of J0 for x < 5
    """
    z = x * x
    # if x < 1.0e-5:
    #     return 1.0 - z/4.0

    p = (z - _DR10) * (z - _DR20)
    p = p * jnp.polyval(_RP0, z) / jnp.polyval(_RQ0, z)
    return jnp.where(x < 1e-5, 1 - z / 4.0, p)


def _j0_large(x: ArrayLike) -> JaxArray:
    """
    Implementation of J0 for x >= 5
    """

    w = 5.0 / x
    q = 25.0 / (x * x)
    p = jnp.polyval(_PP0, q) / jnp.polyval(_PQ0, q)
    q = jnp.polyval(_QP0, q) / jnp.polyval(_QQ0, q)
    xn = x - _PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * _SQ2OPI / jnp.sqrt(x)


def j0(x: ArrayLike) -> JaxArray:
    """
    Implementation of J0 for all x in Jax
    """

    return jnp.where(jnp.abs(x) < 5.0, _j0_small(jnp.abs(x)), _j0_large(jnp.abs(x)))


__all__ = ["j0", "j1"]
