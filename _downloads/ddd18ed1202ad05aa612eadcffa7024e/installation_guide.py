# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Installation and Setup
--------------------------

Install JaxDEM from GitHub or a local checkout. Python 3.12 or newer and
JAX 0.8.1 or newer are required. The core install includes dynamics and
minimization. Feature extras are ``[io]`` for VTK/HDF5/checkpoints, ``[rl]``
for reinforcement learning, ``[docs]`` for documentation, and ``[test]`` for
testing. ``[all]`` installs all of these features together.

Select hardware support through JAX itself. JaxDEM and its JAX-based
dependencies use the same installed JAX backend.
"""

# %%
# Choose the features you need
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The dependency map follows the extras defined in ``pyproject.toml``. Both
# documentation and testing install the I/O and reinforcement-learning extras;
# the complete extra includes all four groups. The JAX backend is a separate
# choice, so selecting CUDA or TPU does not select a JaxDEM feature extra.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.set_axis_off()
feature_nodes = [
    (0.5, 0.90, "Core\nJAX · NumPy · Optax", "#e1e9ef"),
    (0.25, 0.69, "[io]\nVTK · h5py · Orbax", "#e1f0ec"),
    (
        0.75,
        0.69,
        "[rl]\nFlax · Distrax · SciPy\ntqdm · TensorBoard · tensorboardX",
        "#e1f0ec",
    ),
    (
        0.25,
        0.32,
        "[docs] includes io + rl\nMatplotlib · Sphinx · MyST\nPyData theme · Sphinx-Gallery",
        "#f4eadb",
    ),
    (0.75, 0.32, "[test] includes io + rl\npytest · mypy · build", "#f4eadb"),
    (0.5, 0.08, "[all] includes io + rl + docs + test", "#e1e9ef"),
]
for x, y, label, color in feature_nodes:
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#8795a1"},
    )
for start, end in [
    ((0.43, 0.84), (0.27, 0.77)),
    ((0.57, 0.84), (0.73, 0.79)),
    ((0.25, 0.62), (0.5, 0.50)),
    ((0.75, 0.59), (0.5, 0.50)),
    ((0.5, 0.50), (0.27, 0.42)),
    ((0.5, 0.50), (0.73, 0.40)),
    ((0.25, 0.22), (0.43, 0.12)),
    ((0.75, 0.24), (0.57, 0.12)),
]:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#52657c", "lw": 1.3},
    )
ax.text(0.5, 0.53, "both included", ha="center", fontsize=9, color="#52657c")
ax.set_title("Feature extras share one installed JAX backend", pad=12)
plt.show()

# %%
# Create a Virtual Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    python -m venv .venv
#    source .venv/bin/activate
#    python -m pip install --upgrade pip
#
# On Windows PowerShell, activate with ``.\.venv\Scripts\Activate.ps1``.

# %%
# Install from GitHub
# ~~~~~~~~~~~~~~~~~~~
# Core simulation tools:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM @ git+https://github.com/cdelv/JaxDEM.git"
#
# The introduction runs with the core install. To follow its optional export
# section and the VTK/checkpoint tutorials, install the I/O extra:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[io] @ git+https://github.com/cdelv/JaxDEM.git"
#
# A CPU installation is sufficient for the small tutorial configurations. The
# first run compiles JAX functions; later calls with the same array shapes reuse
# those compiled functions within that Python process.
#
# All optional features, with JAX's CUDA 13 backend:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[all] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]"
#
# The two quoted requirements are separate arguments to pip; there is no ``+``
# between them. JaxDEM extras select features, while JAX extras select its backend.
# Pip resolves one compatible JAX installation for JaxDEM, Optax, Flax, and other
# JAX-based dependencies. Their version requirements may raise the effective
# minimum above JaxDEM's core minimum of 0.8.1.
#
# To pin a particular JAX version, add an exact ``==`` constraint:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[rl] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]==0.11.0"
#
# Choose a version compatible with your Python version and selected features.
# For CUDA 12, use ``jax[cuda12]``; on a TPU VM use
# ``jax[tpu]``. See the `JAX installation guide
# <https://docs.jax.dev/en/latest/installation.html>`_ for platform and driver
# requirements. TensorFlow GPU support is not required for JaxDEM dynamics or
# JAX-based training; metrics logging is independent of the JAX backend.

# %%
# Install a Local Checkout
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    git clone https://github.com/cdelv/JaxDEM.git
#    cd JaxDEM
#    python -m pip install .
#
# Pick individual features or install everything, together with the desired JAX:
#
# .. code-block:: bash
#
#    python -m pip install ".[io,rl]" "jax[cuda13]"
#    python -m pip install ".[all]" "jax[cuda13]"
#
# Once the package is published on PyPI, the equivalent command is
# ``python -m pip install 'jaxdem[all]' 'jax[cuda13]'``.
# For local development, use ``python -m pip install -e ".[io]"`` so edits to
# the checkout are reflected in imports without reinstalling.
# Verify the selected backend with:
#
# .. code-block:: bash
#
#    python -c "import jax; print(jax.__version__, jax.devices())"

# %%
# Next Steps
# ~~~~~~~~~~
#
# - :doc:`Introduction <../auto_examples/introduction>`
# - :doc:`System Guide <../auto_examples/system_guide>`
# - :doc:`Materials Guide <../auto_examples/materials_guide>`
