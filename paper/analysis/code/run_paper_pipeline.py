#!/usr/bin/env python3
"""Full pipeline: run folders -> data and plots.

All pipeline code lives in one place — paper/analysis/code/: per-task
swarm analysis, per-run analyses, metrics, combine, and plots.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve()
REPO = next(p for p in [_here, *_here.parents] if (p / "paper" / "analysis" / "code" / "run_paper_pipeline.py").exists())
CODE = REPO / "paper" / "analysis" / "code"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def spec(path: Path) -> str:
    return f"{path.name}={path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("paper/analysis"))
    args = ap.parse_args()
    runs = []
    for raw in args.runs:
        if "=" in raw:
            name, path = raw.split("=", 1)
        else:
            path = raw
            n = Path(raw).name
            name = n.split("-", 1)[0] if n.split("-", 1)[0] in {"1", "2", "3"} else n
        runs.append((name, Path(path)))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    for name, root in runs:
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "blackboard").exists():
                sw = d / "analysis" / "swarm"
                if not (sw / "summary.json").exists():
                    run([sys.executable, CODE / "swarm_analysis.py", "--board", d / "blackboard",
                         "--out", sw, "--label", d.name])
        run([sys.executable, CODE / "board_reading.py", "--root", root, "--out", root / "analysis"])
        run([sys.executable, CODE / "context_usage.py", "--root", root, "--out", root / "analysis"])
        run([sys.executable, CODE / "internet_usage.py", "--root", root, "--out", root / "analysis"])
        run([sys.executable, CODE / "log_composition.py", "--root", root, "--out", root / "analysis"])
        run([sys.executable, CODE / "forgetting_analysis.py", "--root", root,
             "--out", root / "analysis", "--read-depth-only"])

    specs = [f"{name}={path}" for name, path in runs]
    run([sys.executable, CODE / "paper_metrics.py", "--runs", *specs,
         "--out", out])
    run([sys.executable, CODE / "combine_runs.py", "--runs", *specs,
         "--data", out / "data"])
    run([sys.executable, CODE / "paper_figures_combined.py",
         "--data", out / "data" / "combined", "--out", out / "figures"])
    run([sys.executable, CODE / "paper_figures_tex.py",
         "--data", out / "data" / "combined", "--out", REPO / "paper" / "tex"])
    run([sys.executable, CODE / "fig_tags_compact.py",
         "--data", out / "data" / "combined" / "tag_citations.csv",
         "--out", REPO / "paper" / "tex" / "fig_swarm_tags.png"])


if __name__ == "__main__":
    main()
