"""
Performance Report Generator
============================

This script processes the benchmark results and generates a Markdown report with plots
for the JaxDEM documentation, using only the Python standard library.
"""

import json
import os
import argparse
import shutil
from datetime import datetime
from collections import defaultdict


def get_category(row: dict) -> str:
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


def generate_svg_plot(
    data_by_type: dict, title: str, output_path: str, width: int = 800, height: int = 400
) -> None:
    """Generates a simple SVG plot with error bars."""
    padding_left = 60
    padding_right = 150
    padding_top = 40
    padding_bottom = 60
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    # Find global min/max for scaling
    all_values = []
    all_labels = set()
    for points in data_by_type.values():
        for pt in points:
            all_values.append(pt["mean"] * 1000 + pt["std"] * 1000)
            all_values.append(max(0, pt["mean"] * 1000 - pt["std"] * 1000))
            all_labels.add(pt["label"])

    if not all_values:
        return

    y_max = max(all_values) * 1.1
    y_min = 0

    def get_x(i: int, n: int) -> float:
        if n <= 1:
            return padding_left + plot_width / 2
        return padding_left + (i / (n - 1)) * plot_width

    def get_y(val: float) -> float:
        if y_max == y_min:
            return padding_top + plot_height / 2
        return padding_top + plot_height - ((val - y_min) / (y_max - y_min)) * plot_height

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="{padding_top/2}" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
    ]

    # Axes
    svg.append(
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height-padding_bottom}" stroke="black" stroke-width="1"/>'
    )
    svg.append(
        f'<line x1="{padding_left}" y1="{height-padding_bottom}" x2="{width-padding_right}" y2="{height-padding_bottom}" stroke="black" stroke-width="1"/>'
    )

    # Y-axis labels
    for i in range(5):
        val = y_min + (i / 4) * (y_max - y_min)
        y = get_y(val)
        svg.append(
            f'<text x="{padding_left-5}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="10">{val:.2f}</text>'
        )
        svg.append(
            f'<line x1="{padding_left}" y1="{y}" x2="{width-padding_right}" y2="{y}" stroke="#ddd" stroke-width="0.5"/>'
        )

    # Plots
    type_names = sorted(data_by_type.keys())
    # Labels should be consistent across types for same x-axis
    # We use the labels from the first type as reference
    reference_labels = []
    if type_names:
        reference_labels = [pt["label"] for pt in data_by_type[type_names[0]]]

    # X-axis labels
    for i, label in enumerate(reference_labels):
        x = get_x(i, len(reference_labels))
        svg.append(
            f'<g transform="translate({x},{height-padding_bottom+15}) rotate(45)">'
            f'<text x="0" y="0" font-family="sans-serif" font-size="10">{label}</text></g>'
        )

    for idx, t_name in enumerate(type_names):
        color = colors[idx % len(colors)]
        points = data_by_type[t_name]
        n = len(points)
        
        path_d = []
        for i, pt in enumerate(points):
            x = get_x(i, n)
            y = get_y(pt["mean"] * 1000)
            y_err_up = get_y(pt["mean"] * 1000 + pt["std"] * 1000)
            y_err_down = get_y(max(0, pt["mean"] * 1000 - pt["std"] * 1000))
            
            # Error bar
            svg.append(f'<line x1="{x}" y1="{y_err_down}" x2="{x}" y2="{y_err_up}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<line x1="{x-3}" y1="{y_err_down}" x2="{x+3}" y2="{y_err_down}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<line x1="{x-3}" y1="{y_err_up}" x2="{x+3}" y2="{y_err_up}" stroke="{color}" stroke-width="1"/>')
            
            if i == 0: path_d.append(f"M {x} {y}")
            else: path_d.append(f"L {x} {y}")
            
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')

        svg.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="{color}" stroke-width="2"/>')
        
        # Legend
        ly = padding_top + idx * 20
        svg.append(f'<rect x="{width-padding_right+10}" y="{ly}" width="15" height="10" fill="{color}"/>')
        svg.append(f'<text x="{width-padding_right+30}" y="{ly+10}" font-family="sans-serif" font-size="12">{t_name}</text>')

    svg.append("</svg>")
    with open(output_path, "w") as f:
        f.write("\n".join(svg))


def generate_report(
    results_file: str = "benchmarks/results.json",
    output_dir: str = "docs/source/benchmarks",
    k: int = 10,
) -> None:
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    # Filter out failed experiments
    data = [e for e in data if e["module_type"].lower() not in ["aabb", "beast"]]

    # Sort by date
    data.sort(key=lambda x: x["date"])

    # Grouping: system -> hardware -> category -> function -> module_type
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for entry in data:
        sys = entry["system"]
        hw = entry["hardware"]
        cat = get_category(entry)
        func = entry["function"]
        structured_data[sys][hw][cat][func].append(entry)

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

    md_report = "# benchmarks\n\n"

    systems = sorted(structured_data.keys())
    for sys_name in systems:
        md_report += f"# System: {sys_name}\n\n"
        
        hardwares = sorted(structured_data[sys_name].keys())
        for hw in hardwares:
            md_report += f"## Hardware: {hw}\n\n"
            
            category_order = ["Domain", "Collider", "Integrator", "ForceManager", "Force Model"]
            for cat in category_order:
                if cat not in structured_data[sys_name][hw]:
                    continue
                    
                md_report += f"### {cat}\n\n"
                
                funcs = sorted(structured_data[sys_name][hw][cat].keys())
                for func_name in funcs:
                    md_report += f"**Method: {func_name}**\n\n"
                    
                    entries = structured_data[sys_name][hw][cat][func_name]
                    
                    # Group by module_type
                    by_type = defaultdict(list)
                    for e in entries:
                        by_type[e["module_type"]].append(e)
                    
                    plot_data = {}
                    for m_type in sorted(by_type.keys()):
                        type_entries = by_type[m_type][-k:]
                        if not type_entries: continue
                        
                        md_report += f"**Type: {m_type}**\n\n"
                        headers = ["label", "date", "mean (ms)", "std (ms)"]
                        md_report += "| " + " | ".join(headers) + " |\n"
                        md_report += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        
                        points = []
                        for e in type_entries:
                            dt = datetime.fromisoformat(e["date"]).strftime("%Y-%m-%d %H:%M")
                            m_ms = e["mean"] * 1000
                            s_ms = e["std"] * 1000
                            label = e.get("label", e["commit"][:7])
                            md_report += f"| {label} | {dt} | {m_ms:.4f} | {s_ms:.4f} |\n"
                            points.append({"label": label, "mean": e["mean"], "std": e["std"]})
                        
                        md_report += "\n"
                        plot_data[m_type] = points

                    if plot_data:
                        plot_filename = f"{sys_name}_{hw}_{cat}_{func_name}".replace(" ", "_").replace(".", "_") + ".svg"
                        plot_path = os.path.join(output_dir, "plots", plot_filename)
                        title = f"{func_name} in {cat} ({sys_name} on {hw})"
                        generate_svg_plot(plot_data, title, plot_path)
                        md_report += f"![Performance plot](plots/{plot_filename})\n\n"

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
