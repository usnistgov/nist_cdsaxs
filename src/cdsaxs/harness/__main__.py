"""Command-line entry point for configuration, discovery, and staged runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .config import DEFAULTS, STAGES, load_settings, write_json
from .diagnostics import NeedsReview
from .script_generator import generate


def main(argv=None, default_config=None, default_resume=None):
    parser = argparse.ArgumentParser(description="Staged CD-SAXS reduction harness")
    parser.add_argument("--config", type=Path, default=default_config,
                        help="User JSON settings; relative paths use its directory")
    parser.add_argument("--init", type=Path, metavar="DIRECTORY",
                        help="Generate a starter settings.json and runnable script")
    parser.add_argument(
        "--stage", default="all",
        choices=("discover", "calibration", "phi", "chi", "omega",
                 "preview", "reduce", "export", "all"),
    )
    parser.add_argument("--resume", type=Path,
                        help="Existing sample run directory or batch JSON manifest")
    parser.add_argument("--accept", action="append", choices=STAGES, default=[],
                        help="Accept an already inspected, unchanged proposal; repeat as needed")
    parser.add_argument("--sample", help="Process only this named batch sample")
    parser.add_argument("--output-root", type=Path, help="Consolidated output directory")
    args = parser.parse_args(argv)
    try:
        if args.init:
            settings = load_settings(args.config) if args.config else deepcopy(DEFAULTS)
            check_path = args.init / "settings.json"
            if check_path.exists():
                parser.error(f"Settings already exist: {check_path}")
            script = generate(args.init, settings)
            print(f"Generated {script}; edit {script.with_name('settings.json')}")
            return 0
        if args.config is None:
            parser.error("Provide --config settings.json or --init DIRECTORY")
        settings = load_settings(args.config)
        if args.output_root:
            settings["output"]["root"] = str(args.output_root.resolve())
        from .runner import inventory, run_batch
        if args.stage == "discover":
            result = inventory(settings)
            target = args.config.resolve().parent / "inventory.json"
            write_json(target, result)
            print(f"Inventory: {target}")
            print(f"Reference candidates: {result['reference_candidates']}")
            return 0
        resume = args.resume
        if resume is None and args.output_root is None:
            resume = default_resume
        results = run_batch(settings, args.stage, resume, args.accept, args.sample)
        for result in results:
            print(f"{result['sample']}: {result['status']} — {result['directory']}")
        return 2 if any(result["status"] == "needs_review" for result in results) else 0
    except (ValueError, OSError, NeedsReview) as error:
        print(f"Harness configuration/reduction needs attention: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
