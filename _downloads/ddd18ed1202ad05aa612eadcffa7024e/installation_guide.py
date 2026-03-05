r"""
Installation and Setup
----------------------

JaxDEM is not yet published on PyPI, so install it from the GitHub repository
(or by cloning the repository and installing locally). Optional extras are
defined in ``pyproject.toml``.
"""

# %%
# Create a Virtual Environment (Recommended)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    python -m venv .venv
#    source .venv/bin/activate
#    python -m pip install --upgrade pip
#
# On Windows (PowerShell), activation is:
#
# .. code-block:: powershell
#
#    .\.venv\Scripts\Activate.ps1


# %%
# Install JaxDEM from GitHub
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic install:
#
# .. code-block:: bash
#
#    python -m pip install "git+https://github.com/cdelv/JaxDEM.git"
#
# With CUDA 13 support:
#
# .. code-block:: bash
#
#    python -m pip install "git+https://github.com/cdelv/JaxDEM.git#egg=JaxDEM[cuda13]"
#
# With RL capabilities:
#
# .. code-block:: bash
#
#    python -m pip install "git+https://github.com/cdelv/JaxDEM.git#egg=JaxDEM[rl]"
#
# You can combine extras as needed:
#
# .. code-block:: bash
#
#    python -m pip install "git+https://github.com/cdelv/JaxDEM.git#egg=JaxDEM[cuda13,rl]"

# %%
# Install by Cloning (Alternative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can also clone first, then install locally:
#
# .. code-block:: bash
#
#    git clone https://github.com/cdelv/JaxDEM.git
#    cd JaxDEM
#    python -m pip install .
#
# With extras:
#
# .. code-block:: bash
#
#    python -m pip install ".[rl]"
#    python -m pip install ".[cuda13,rl]"

# %%
# Next Steps
# ~~~~~~~~~~
# After installation succeeds, continue with:
#
# - :doc:`Introduction <../auto_examples/introduction>`
# - :doc:`System Guide <../auto_examples/system_guide>`
# - :doc:`Materials Guide <../auto_examples/materials_guide>`
