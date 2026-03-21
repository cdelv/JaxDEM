import subprocess
import jax
import json
import os
import datetime
from typing import Any


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        return "unknown"


def get_commit_date(commit: str = "HEAD") -> str:
    try:
        # Get ISO 8601 date of the commit
        return (
            subprocess.check_output(["git", "show", "-s", "--format=%cI", commit])
            .decode("ascii")
            .strip()
        )
    except:
        return datetime.datetime.now().isoformat()


def get_hardware_info() -> str:
    devices = jax.devices()
    if not devices:
        return "Unknown"
    device = devices[0]
    return f"{device.platform} - {device.device_kind}"


def update_results(results_file: str, new_entry: dict[str, Any]) -> None:
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    commit = new_entry["commit"]
    commit_short = commit[:7]
    config_key = (
        new_entry["function"],
        new_entry["module_type"],
        new_entry["system"],
        new_entry["hardware"],
    )

    # Helper to find entries for this config
    def get_config_entries(d: list[dict[str, Any]]) -> list[int]:
        return [
            i
            for i, e in enumerate(d)
            if (e["function"], e["module_type"], e["system"], e["hardware"])
            == config_key
        ]

    config_indices = get_config_entries(data)

    if not config_indices:
        # First time seeing this config
        entry_first = new_entry.copy()
        entry_first["label"] = commit_short
        entry_current = new_entry.copy()
        entry_current["label"] = "current"
        data.append(entry_first)
        data.append(entry_current)
    else:
        # Config exists. Check the commit of the baseline (the one that isn't "current")
        baseline_idx = -1
        current_idx = -1
        for idx in config_indices:
            if data[idx]["label"] == "current":
                current_idx = idx
            else:
                baseline_idx = idx

        if baseline_idx != -1:
            if data[baseline_idx]["commit"] == commit:
                # Same commit, just update the "current" entry
                if current_idx != -1:
                    new_current = new_entry.copy()
                    new_current["label"] = "current"
                    data[current_idx] = new_current
                else:
                    # Should not happen if logic is consistent, but for safety:
                    new_current = new_entry.copy()
                    new_current["label"] = "current"
                    data.append(new_current)
            else:
                # New commit!
                # 1. Update the OLD baseline with the data from the OLD "current"
                if current_idx != -1:
                    old_current_data = data[current_idx].copy()
                    old_current_data["label"] = data[baseline_idx][
                        "label"
                    ]  # Keep the old commit hash label
                    data[baseline_idx] = old_current_data

                    # 2. Remove the old "current"
                    data.pop(current_idx)

                # 3. Add the NEW baseline and NEW "current"
                entry_first = new_entry.copy()
                entry_first["label"] = commit_short
                entry_current = new_entry.copy()
                entry_current["label"] = "current"
                data.append(entry_first)
                data.append(entry_current)
        else:
            # Only "current" existed? Add baseline.
            entry_first = new_entry.copy()
            entry_first["label"] = commit_short
            data.append(entry_first)

    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Commit: {get_git_commit()}")
    print(f"Hardware: {get_hardware_info()}")
