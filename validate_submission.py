#!/usr/bin/env python3
"""
Pre-submission checks: OpenEnv layout, unit tests, oracle inference, grader scores in [0,1].

Run from repository root (after ``pip install -e data_cleaning_env`` or ``pip install -e ./data_cleaning_env``):

  python validate_submission.py

Optional:

  python validate_submission.py --docker    # also ``docker build`` (requires Docker)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / "data_cleaning_env"


def _openenv_cli() -> str | None:
    """Resolve ``openenv`` executable (same venv as this script, then PATH)."""
    sibling = Path(sys.executable).resolve().parent / "openenv"
    if sibling.is_file():
        return str(sibling)
    return shutil.which("openenv")


def run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=cwd or ROOT)


def check_graders_three_tasks() -> int:
    """Run bundled easy/medium/hard graders; scores must be in [0, 1]."""
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from data_cleaning_env.graders import grade_task

    for name in ("easy", "medium", "hard"):
        tdir = ENV_DIR / "tasks" / name
        gt = pd.read_csv(tdir / "ground_truth.csv")
        with open(tdir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        # Build a synthetic perfect plot action matching the first expected plot (if any)
        plot_action = None
        expected = meta.get("expected_plots")
        if expected:
            ep = expected[0]
            plot_action = {
                "plot_type": ep.get("type", "scatter"),
                "x": ep.get("x", "OrderDate"),
                "y": ep.get("y", "Revenue"),
            }
        s = grade_task(gt, gt, meta, plot_action, None)
        if s is None:
            print(f"FAIL: grader returned None for task={name}", file=sys.stderr)
            return 1
        if not (0.0 <= float(s) <= 1.0):
            print(f"FAIL: task={name} score {s} not in [0,1]", file=sys.stderr)
            return 1
        print(f"ok: task={name} identity grader score={s:.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-submission validation")
    ap.add_argument(
        "--docker",
        action="store_true",
        help="Run docker build from repo root (requires Docker daemon)",
    )
    args = ap.parse_args()
    code = 0

    if not ENV_DIR.is_dir():
        print(f"Missing {ENV_DIR}", file=sys.stderr)
        return 1

    oe = _openenv_cli()
    if not oe:
        print(
            "openenv CLI not found. Install with: pip install 'openenv-core[core]'",
            file=sys.stderr,
        )
        return 1

    r = run([oe, "validate", "--verbose", str(ENV_DIR)])
    code |= r
    if r != 0:
        return code

    r = run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ENV_DIR)
    code |= r
    if r != 0:
        return code

    r = check_graders_three_tasks()
    code |= r
    if r != 0:
        return code

    r = run([sys.executable, str(ROOT / "inference.py"), "--oracle", "--no-report"])
    code |= r
    if r != 0:
        return code

    if args.docker:
        r = run(
            [
                "docker",
                "build",
                "-t",
                "autodatalab-openenv",
                "-f",
                str(ROOT / "Dockerfile"),
                str(ROOT),
            ]
        )
        code |= r

    if code == 0:
        print("validate_submission: all checks passed.", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
