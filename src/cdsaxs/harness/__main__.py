"""Command-line entry point for configuration, discovery, and staged runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from .config import DEFAULTS, STAGES, load_settings, write_json
from .script_generator import generate
from .recording import RunRecorder, session_log_path


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
    parser.add_argument("--output-profile", choices=("compact", "full"),
                        help="Compact deliverables (default) or separate diagnostic JSON files")
    parser.add_argument("--codex-session-log", type=Path, metavar="PATH",
                        help="Optional Codex JSONL log for token snapshots; otherwise detect this thread")
    parser.add_argument("--agent-wait", choices=("foreground", "suspend"), default="foreground",
                        help="Suspend: detach the whole reduction and notify the originating Codex thread")
    parser.add_argument("--agent-thread-id", help="Originating conversation UUID; defaults to CODEX_THREAD_ID")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--agent-job-status", type=Path, metavar="PATH", help="Read a job.json record")
    recovery.add_argument("--agent-job-cancel", type=Path, metavar="PATH", help="Cancel a job and suppress continuation")
    recovery.add_argument("--agent-job-notify", type=Path, metavar="PATH", help="Explicitly retry completion notification without rerunning Python")
    args = parser.parse_args(argv)
    try:
        for action in ("status", "cancel", "notify"):
            target = getattr(args, f"agent_job_{action}")
            if target:
                from . import jobs
                result = getattr(jobs, action)(target)
                print(json.dumps(result, indent=2))
                return 2 if action == "notify" and result["notification_status"] == "pending" else 0
        if args.agent_wait == "suspend" and (args.init or args.stage == "discover"):
            parser.error("Use foreground mode for initialization/discovery; suspend a reduction stage")
        if args.init:
            settings = load_settings(args.config) if args.config else deepcopy(DEFAULTS)
            if args.output_profile:
                settings["output"]["profile"] = args.output_profile
            check_path = args.init / "settings.json"
            if check_path.exists():
                parser.error(f"Settings already exist: {check_path}")
            script = generate(args.init, settings)
            print(f"Generated {script}; edit {script.with_name('settings.json')}")
            return 0
        if args.config is None:
            parser.error("Provide --config settings.json or --init DIRECTORY")
        settings = load_settings(args.config)
        if args.output_profile:
            settings["output"]["profile"] = args.output_profile
        if args.output_root:
            settings["output"]["root"] = str(args.output_root.resolve())
        if args.stage == "discover":
            from .runner import inventory
            recording_state = {}
            recorder = RunRecorder(recording_state, session_log_path(args.codex_session_log))
            recorder.record["requested_stage"] = "discover"
            with recorder.stage("discover"):
                result = inventory(settings)
            recorder.finish("complete")
            result["recordings"] = recording_state["recordings"]
            if settings["output"]["profile"] == "full":
                target = args.config.resolve().parent / "inventory.json"
                write_json(target, result)
                print(f"Inventory: {target}")
            else:
                print(json.dumps(result, indent=2))
            print(f"Reference candidates: {result['reference_candidates']}")
            return 0
        resume = args.resume
        if resume is None and args.output_root is None:
            resume = default_resume
        if args.agent_wait == "suspend":
            from .jobs import launch
            result = launch(settings, args.stage, resume, args.accept, args.sample,
                            args.codex_session_log, args.agent_thread_id,
                            command_args=list(argv) if argv is not None else sys.argv[1:])
            print(json.dumps(result, indent=2))
            return 0
        from .runner import run_batch
        results = run_batch(settings, args.stage, resume, args.accept, args.sample,
                            codex_session_log=args.codex_session_log)
        for result in results:
            print(f"{result['sample']}: {result['status']} — {result['directory']}")
        return 2 if any(result["status"] == "needs_review" for result in results) else 0
    except (ValueError, OSError) as error:
        print(f"Harness configuration/reduction needs attention: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
