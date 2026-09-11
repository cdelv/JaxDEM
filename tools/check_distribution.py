"""Reject unrelated files in a built wheel and source distribution."""

import argparse
from pathlib import Path
import tarfile
import zipfile


def check_distribution(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        if "jaxdem/py.typed" not in names:
            raise ValueError("Wheel is missing jaxdem/py.typed")
        for name in names:
            if not (name.startswith("jaxdem/") or ".dist-info/" in name):
                raise ValueError(f"Unexpected wheel entry: {name}")
    with tarfile.open(sources[0]) as archive:
        for name in archive.getnames():
            parts = Path(name).parts[1:]
            if parts and parts[0] in {
                "docs",
                "tests",
                "examples",
                "benchmarks",
                "tools",
                "audit",
            }:
                raise ValueError(f"Unexpected sdist entry: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check_distribution(parser.parse_args().directory)
