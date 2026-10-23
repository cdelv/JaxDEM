# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Install the pinned official ParaView binary into a disposable/cache directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request

VERSION = "6.2.0"
DIRECTORY = f"ParaView-{VERSION}-MPI-Linux-Python3.12-x86_64"
ARCHIVE = DIRECTORY + ".tar.gz"
URL = f"https://www.paraview.org/files/v6.2/{ARCHIVE}"
# Published at https://www.paraview.org/files/v6.2/md5sum.txt.
MD5 = "f05c9c7a1c893216d8c4f18460747019"
SHA256 = "242052db6569073cb77c9e683c8a81a4cd31c06b0901818f72c9c23546420e46"


def install(directory: Path) -> Path:
    directory = directory.expanduser().resolve()
    executable = directory / DIRECTORY / "bin" / "pvpython"
    if executable.is_file():
        return executable
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        print(f"Downloading official ParaView {VERSION}...", flush=True)
        with (
            urllib.request.urlopen(URL, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != MD5:
        raise RuntimeError(f"ParaView archive checksum mismatch: {checksum}")
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != SHA256:
        raise RuntimeError(f"ParaView archive SHA256 mismatch: {checksum}")
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    if not executable.is_file():
        raise RuntimeError("The ParaView archive did not contain pvpython")
    archive.unlink()
    return executable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(install(args.directory))
