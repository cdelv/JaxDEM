"""Shared output-directory preparation for writers."""

from pathlib import Path
import shutil


def prepare_output_directory(directory: Path | str, *, clean: bool = False) -> Path:
    """Create an output directory, optionally removing only a safe child tree."""
    path = Path(directory).resolve()
    cwd = Path.cwd().resolve()
    if clean:
        if path == cwd or path in cwd.parents or path == Path(path.anchor):
            raise ValueError(
                f"Refusing to clean protected directory {path}; choose a dedicated "
                "output directory below the working tree."
            )
        if path.exists():
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["prepare_output_directory"]
