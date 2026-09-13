"""Execute the curated short guides in isolated output directories."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("introduction", "materials_guide", "custom_modules_guide"):
        with tempfile.TemporaryDirectory(prefix="jaxdem-example-") as directory:
            env = dict(os.environ)
            env["TMPDIR"] = directory
            env["MPLCONFIGDIR"] = directory
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(root), env.get("PYTHONPATH", "")))
            )
            subprocess.run(
                [sys.executable, str(root / "examples" / f"{name}.py")],
                cwd=directory,
                env=env,
                check=True,
                timeout=180,
            )
            print(f"Passed: {name}", flush=True)


if __name__ == "__main__":
    main()
