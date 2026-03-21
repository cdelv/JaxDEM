"""
Performance Report Generator
============================

This script processes the benchmark results and generates a Markdown report with plots
for the JaxDEM documentation.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import shutil


def get_category(row):
    func = row["function"]
    m_type = row["module_type"]

    if m_type == "ForceManager":
        return "ForceManager"
    if func in ["apply", "displacement", "shift"]:
        return "Domain"
    if func in ["compute_force", "create_neighbor_list"]:
        return "Collider"
    if func in ["step_before_force", "step_after_force"]:
        return "Integrator"
    if func in ["force", "energy", "compute_potential_energy"]:
        return "Force Model"
    return "Other"


def generate_report(
    results_file="benchmarks/results.json", output_dir="docs/source/benchmarks", k=10
):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Filter out failed experiments
    df = df[~df["module_type"].str.lower().isin(["aabb", "beast"])]

    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)
    # Sort by date to keep historical order
    df = df.sort_values("date")

    # Assign categories
    df["category"] = df.apply(get_category, axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # Clean output directory
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    md_report = "# Benchmarks\n\n"

    # Grouping order: system (H1), hardware (H2), category (H3)
    systems = sorted(df["system"].unique())
    for sys_name in systems:
        sys_group = df[df["system"] == sys_name]
        md_report += f"# System: {sys_name}\n\n"

        hardwares = sorted(sys_group["hardware"].unique())
        for hw in hardwares:
            hw_group = sys_group[sys_group["hardware"] == hw]
            md_report += f"## Hardware: {hw}\n\n"

            category_order = [
                "Domain",
                "Collider",
                "Integrator",
                "ForceManager",
                "Force Model",
            ]
            for cat in category_order:
                cat_group = hw_group[hw_group["category"] == cat]
                if cat_group.empty:
                    continue

                md_report += f"### {cat}\n\n"

                for func_name, func_group in cat_group.groupby("function"):
                    md_report += f"**Method: {func_name}**\n\n"

                    plt.figure(figsize=(10, 6))
                    plot_data_found = False
                    module_types = sorted(func_group["module_type"].unique())

                    for m_type in module_types:
                        type_group = func_group[
                            func_group["module_type"] == m_type
                        ].sort_values("date")

                        # Only take last k entries (labels) for display
                        # But wait, labels are unique per commit/current.
                        # We want to keep the historical order.
                        type_group_k = type_group.tail(k)

                        if type_group_k.empty:
                            continue

                        md_report += f"**Type: {m_type}**\n\n"

                        # Prepare table
                        table = type_group_k.copy()
                        table["mean (ms)"] = table["mean"] * 1000
                        table["std (ms)"] = table["std"] * 1000
                        table["formatted_date"] = table["date"].dt.strftime(
                            "%Y-%m-%d %H:%M"
                        )

                        headers = ["label", "date", "mean (ms)", "std (ms)"]
                        md_report += "| " + " | ".join(headers) + " |\n"
                        md_report += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for _, row in table.iterrows():
                            md_report += f"| {row['label']} | {row['formatted_date']} | {row['mean (ms)']:.4f} | {row['std (ms)']:.4f} |\n"
                        md_report += "\n\n"

                        # Plotting - Use 'label' for x-axis but we need to ensure they are unique if we use them as tick labels
                        # If we have multiple "current" in the history (shouldn't happen with new pruning logic),
                        # it might look weird. But new logic keeps only ONE "current" per config.
                        plt.errorbar(
                            table["label"],
                            table["mean (ms)"],
                            yerr=table["std (ms)"],
                            label=m_type,
                            fmt="-o",
                            capsize=5,
                        )
                        plot_data_found = True

                    if plot_data_found:
                        plt.title(f"Method: {func_name} in {cat} ({sys_name} on {hw})")
                        plt.ylabel("Time (ms)")
                        plt.xlabel("Run Label")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                        plot_name = (
                            f"{sys_name}_{hw}_{cat}_{func_name}".replace(
                                " ", "_"
                            ).replace(".", "_")
                            + ".png"
                        )
                        plot_path = os.path.join(output_dir, "plots", plot_name)
                        plt.savefig(plot_path)
                        plt.close()

                        md_report += f"![Performance plot](plots/{plot_name})\n\n"
                    else:
                        plt.close()

    with open(os.path.join(output_dir, "index.md"), "w") as f:
        f.write(md_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance report.")
    parser.add_argument(
        "--results", default="benchmarks/results.json", help="Path to results file."
    )
    parser.add_argument(
        "--output", default="docs/source/benchmarks", help="Output directory."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of last entries to show."
    )
    args = parser.parse_args()

    generate_report(args.results, args.output, args.k)
