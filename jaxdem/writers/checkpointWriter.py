# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining data writers.

This module provides a high-level VTKWriter frontend, a VTKBaseWriter
plugin interface, and concrete writers (e.g., SpheresWriter, DomainWriter)
for exporting JAX-based simulation snapshots to VTK files. It also contains
utilities for safe directory cleanup and an (experimental) checkpointing
wrapper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import math
import os
import tempfile
import threading
from pathlib import Path
import shutil
import concurrent.futures as cf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Set, Optional

import numpy as np
import xml.etree.ElementTree as ET
import orbax.checkpoint as ocp

from ..factory import Factory

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass(slots=True)
class CheckpointWriter:
    """
    Thin wrapper around Orbax checkpointing.

    Note: At present, `checkpointer` is not instantiated and `save`/`load`
    are placeholders. This class documents the intended API only.

    Attributes
    ----------
    directory : Path
        Base directory where checkpoints are stored. Defaults to "./checkpoints".
    clean : bool
        If True, the directory is cleared on initialization (when safe to do so).
    max_to_keep : int | None
        Maximum number of checkpoints to keep. If None, keep all.
    save_interval_steps : int
        Intended interval (in steps) between successive auto-saves.
    checkpointer : ocp.CheckpointManager
        Underlying Orbax checkpoint manager (currently None).
    """

    directory: Path = Path("./checkpoints")
    clean: bool = True
    max_to_keep: int | None = 1
    save_interval_steps: int = 2
    checkpointer: ocp.CheckpointManager = None

    def __del__(self):
        """
        Destructor ensuring pending checkpoint operations finish and resources
        are released.

        Notes
        -----
        This assumes `self.checkpointer` is a valid CheckpointManager. As the
        current implementation leaves it as None, this method is a best-effort
        placeholder for the finalized API.
        """
        self.checkpointer.wait_until_finished()
        self.checkpointer.close()

    def __post_init__(self):
        """
        Post-initialization: resolve the directory, optionally clean it,
        and prepare checkpointing options and handlers.

        Notes
        -----
        The actual `CheckpointManager` construction is commented out. When
        enabled, it should use `options` and `item_handlers`.
        """
        self.directory = self.directory.resolve()
        if self.clean and _is_safe_to_clean(self.directory):
            ocp.test_utils.erase_and_create_empty(self.directory)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep, save_interval_steps=self.save_interval_steps
        )
        item_handlers = {
            "state": ocp.StandardCheckpointHandler(),
            "system": ocp.StandardCheckpointHandler(),
            "system_metadata": ocp.JsonCheckpointHandler(),
        }
        # self.checkpointer = ocp.CheckpointManager(
        #     self.directory,
        #     options=options,
        #     handler_registry=item_handlers,
        #     logger=
        # )

    def save(self, state, system):
        """
        Save a checkpoint for the provided state/system.

        Note
        ----
        Not implemented yet.
        """
        pass

    def load(self):
        """
        Load the latest (or a specific) checkpoint.

        Note
        ----
        Not implemented yet.
        """
        pass
