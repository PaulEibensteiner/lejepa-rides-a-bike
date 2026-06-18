from __future__ import annotations

import argparse
import json
from pathlib import Path


def gather_outputs(results_dir: Path) -> dict[str, list[Path]]:
    categories = {
        "checkpoints": sorted(results_dir.glob("model_*/checkpoint.pt")),
        "data": sorted((results_dir / "data").glob("*.npz")),
        "metrics": (
            sorted((results_dir / "metrics").glob("*.json"))
            + sorted((results_dir / "metrics").glob("*.csv"))
        ),
        "figures": sorted((results_dir / "figures").glob("*.png")),
        "videos": sorted((results_dir / "videos").glob("*.mp4")),
    }
    return categories


def print_inventory(categories: dict[str, list[Path]]) -> None:
    for section, paths in categories.items():
        print(f"[{section}]")
        if not paths:
            print("  (none)")
            continue
        for p in paths:
            print(f"  {p.resolve()}")


def write_results_readme(
    results_dir: Path, categories: dict[str, list[Path]], top3: list[Path]
) -> Path:
    out = results_dir / "README.txt"
    lines: list[str] = []
    lines.append("Bike RL Output Inspection Guide")
    lines.append("")
    lines.append("Best viewers:")
    lines.append("- MP4 videos: VS Code Explorer preview or host browser")
    lines.append("- PNG figures: VS Code image preview")
    lines.append("- JSON/CSV metrics: VS Code text editor")
    lines.append("")
    lines.append("Remote usage:")
    lines.append('- Open with host browser: "$BROWSER" <absolute-path-to-file>')
    lines.append("- Optional copy to host: scp <remote>:<absolute-path> .")
    lines.append("")
    lines.append("Verification checklist:")
    lines.append(
        "- Check model_metrics.png for success_rate, mean_distance, mean_rms_lean"
    )
    lines.append("- Check trajectory_comparison.png for stable long trajectories")
    lines.append("- Watch one representative video per model in results/videos/")
    lines.append("")
    lines.append("Top 3 files to inspect first:")
    for p in top3:
        lines.append(f"- {p.resolve()}")
    lines.append("")
    lines.append("Full inventory:")
    for section, paths in categories.items():
        lines.append(f"[{section}]")
        if not paths:
            lines.append("- (none)")
        for p in paths:
            lines.append(f"- {p.resolve()}")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def choose_top3(categories: dict[str, list[Path]]) -> list[Path]:
    top: list[Path] = []
    if categories["videos"]:
        top.append(categories["videos"][0])
    if categories["figures"]:
        pref = None
        for p in categories["figures"]:
            if p.name == "trajectory_comparison.png":
                pref = p
                break
        top.append(pref or categories["figures"][0])
    if categories["metrics"]:
        top.append(categories["metrics"][0])
    return top[:3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List generated bike experiment outputs"
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    categories = gather_outputs(results_dir)
    print_inventory(categories)

    top3 = choose_top3(categories)
    print("\n[top-3]")
    for p in top3:
        print(f"  {p.resolve()}")

    if args.write_readme:
        readme = write_results_readme(results_dir, categories, top3)
        print(f"\nWrote: {readme.resolve()}")


if __name__ == "__main__":
    main()
