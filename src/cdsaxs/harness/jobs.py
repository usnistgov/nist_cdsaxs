"""Detached reduction supervision. No model calls occur while the worker runs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from uuid import UUID, uuid4

from .config import read_json, sample_settings, slug, write_json
from .recording import session_log_path, utc_now
from .reporting import recording_summary


def plan_batch(settings, resume=None, sample=None):
    """Reserve stable destinations without loading detector data."""
    samples = list(sample_settings(settings))
    if sample is not None:
        samples = [cfg for cfg in samples if cfg["sample_name"] == sample]
        if not samples:
            raise ValueError(f"Sample {sample} is not in settings")
    mapping = {}
    if resume and Path(resume).is_file():
        mapping = {item["sample"]: item["directory"] for item in read_json(resume)["results"]}
    elif resume:
        if len(samples) != 1:
            raise ValueError("Resume a batch using its batch JSON manifest")
        mapping[samples[0]["sample_name"]] = str(Path(resume).resolve())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    entries = []
    for cfg in samples:
        root = Path(cfg["output"]["root"] or Path(cfg["input"]["data_directory"] or ".") / "Results")
        directory = Path(mapping.get(cfg["sample_name"],
                         root / slug(cfg["experiment"]) / slug(cfg["sample_name"]) / run_id)).resolve()
        entries.append({"settings": cfg, "directory": str(directory)})
    if len({entry["directory"] for entry in entries}) != len(entries):
        raise ValueError("Batch samples cannot share a run directory")
    return {"run_id": run_id, "entries": entries}


def reserve(plan):
    """Hold advisory locks across launcher, supervisor and worker lifetimes."""
    if os.name != "posix":
        raise ValueError("Run-directory locking requires POSIX; use a supported Linux/macOS host")
    import fcntl
    handles = []
    try:
        for entry in sorted(plan["entries"], key=lambda item: item["directory"]):
            directory = Path(entry["directory"]) / ".harness"
            directory.mkdir(parents=True, exist_ok=True)
            handle = os.open(directory / "run.lock", os.O_CREAT | os.O_RDWR, 0o600)
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Run directory is already in use: {entry['directory']}") from None
            # Fail permission checks before detaching, including existing outputs.
            run_directory = Path(entry["directory"])
            for target in (run_directory, *run_directory.rglob("*")):
                if not os.access(target, os.W_OK):
                    raise PermissionError(f"Output is not writable: {target}")
        return handles
    except BaseException:
        close_locks(handles)
        raise


def close_locks(handles):
    # Do not LOCK_UN: inherited descriptors share the same open-file description.
    for handle in handles:
        os.close(handle)


@contextmanager
def control_lock(path):
    import fcntl
    with Path(path).with_name("control.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def notifier_preflight(thread_id=None):
    try:
        thread = str(UUID(thread_id or os.environ.get("CODEX_THREAD_ID", "")))
    except (ValueError, AttributeError):
        raise ValueError("Suspend mode needs --agent-thread-id UUID or CODEX_THREAD_ID; "
                         "otherwise use --agent-wait foreground") from None
    executable = shutil.which("codex")
    if not executable:
        raise ValueError("Suspend mode requires Codex with queue --resume on PATH")
    try:
        help_result = subprocess.run([executable, "queue", "--help"], capture_output=True,
                                     text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Cannot check Codex notifier: {error}") from error
    if help_result.returncode or "--resume" not in help_result.stdout:
        raise ValueError("Installed Codex does not support queue --resume; use foreground mode")
    return thread, executable


def launch(settings, stage, resume=None, accept=(), sample=None,
           codex_session_log=None, thread_id=None, command_args=None):
    if os.name != "posix":
        raise ValueError("Suspend mode requires POSIX process sessions; use foreground mode")
    thread, notifier = notifier_preflight(thread_id)
    plan = plan_batch(settings, resume, sample)
    handles = reserve(plan)
    job_id = uuid4().hex
    path = Path.cwd() / ".harness" / "jobs" / job_id / "job.json"
    try:
        path.parent.mkdir(parents=True, mode=0o700)
        log_path = session_log_path(codex_session_log, thread_id=thread)
        snapshot = {"settings": settings, "stage": stage, "accept": list(accept),
                    "sample": sample, "resume": str(resume) if resume else None,
                    "plan": plan, "command_args": command_args,
                    "codex_session_log": str(log_path) if log_path else None}
        write_json(path.with_name("snapshot.json"), snapshot)
        record = {"schema_version": 1, "job_id": job_id, "thread_id": thread,
                  "execution_mode": "suspend", "execution_status": "launching",
                  "notification_status": "not_attempted", "notifier": notifier,
                  "created_at": utc_now(), "record_path": str(path),
                  "run_directories": [entry["directory"] for entry in plan["entries"]],
                  "worker_log": str(path.with_name("worker.log")), "results": []}
        write_json(path, record)
        with path.with_name("worker.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "cdsaxs.harness.jobs", "supervise", str(path),
                 *map(str, handles)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, pass_fds=handles,
                env={**os.environ, "CODEX_THREAD_ID": thread})
        # The supervisor alone updates job.json after launch; avoid fast-finish races.
        return {"job_id": job_id, "record_path": str(path), "supervisor_pid": process.pid}
    except BaseException as error:
        if path.exists():
            record.update(execution_status="failed", error=str(error), finished_at=utc_now())
            write_json(path, record)
        raise
    finally:
        close_locks(handles)


def _update_summaries(record):
    """Attach lifecycle accounting without changing scientific settings or estimates."""
    for item in record.get("results", []):
        directory = Path(item["directory"])
        candidates = [target for target in (directory / "state.json", directory / ".harness/state.json")
                      if target.exists()]
        state = None
        if candidates:
            state_path = max(candidates, key=lambda target: target.stat().st_mtime_ns)
            state = read_json(state_path)
            for invocation in state.get("recordings", []):
                if invocation.get("agent_job_id") == record["job_id"]:
                    invocation.update(background_elapsed_seconds=record.get("elapsed_seconds"),
                                      notification_status=record["notification_status"])
            write_json(state_path, state)
        summary = directory / item.get("summary_name", "summary.md")
        if summary.exists() and state:
            body = summary.read_text(encoding="utf-8")
            runtime = "\n".join(recording_summary(state)).strip() + "\n\n"
            body = re.sub(r"(?ms)^## Runtime and token usage\n.*?(?=^## (?:Outputs|Artifacts)\n|\Z)",
                          lambda match: runtime, body)
            temporary = summary.with_suffix(".md.tmp")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(summary)


def _notify(path, record):
    if record["notification_status"] == "delivered":
        _safe_summaries(path, record)
        return record
    if path.with_name("cancel.json").exists() or record["execution_status"] == "cancelled":
        record["notification_status"] = "suppressed"
        write_json(path, record)
        _safe_summaries(path, record)
        return record
    message = (f"Reduction job {record['job_id']}: {record['execution_status']}. "
               f"Read {path} and its sample summaries, then continue the original task. "
               "Existing review acceptance and permissions still apply.")
    # Persist an uncertain state before delivery. Never automatically retry it.
    record.update(notification_status="pending", notification_attempted_at=utc_now())
    write_json(path, record)
    _safe_summaries(path, record)
    try:
        result = subprocess.run([record["notifier"], "queue", "--thread", record["thread_id"],
                                 "--message", message, "--resume"], capture_output=True,
                                text=True, timeout=30)
        output = result.stdout + result.stderr
        # queue succeeds even when resumption warns. Treat any warning conservatively.
        status = "delivered" if result.returncode == 0 and not any(
            word in output.lower() for word in ("warning", "failed", "error")) else "pending"
        record.update(notification_status=status, notification_returncode=result.returncode,
                      notification_output=output[-4000:])
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(notification_status="pending", notification_error=str(error))
    write_json(path, record)
    _safe_summaries(path, record)
    return record


def notify(path):
    path = Path(path).resolve()
    with control_lock(path):
        record = read_json(path)
        if record["execution_status"] not in ("complete", "needs_review", "failed", "cancelled"):
            raise ValueError("Job is still active; notification recovery requires a completed job")
        return _notify(path, record)


def cancel(path):
    path = Path(path).resolve()
    with control_lock(path):
        record = read_json(path)
        write_json(path.with_name("cancel.json"), {"requested_at": utc_now()})
        return {"job_id": record["job_id"], "cancellation": "requested",
                "execution_status": record["execution_status"]}


def status(path):
    path = Path(path).resolve()
    record = read_json(path)
    record["cancel_requested"] = path.with_name("cancel.json").exists()
    if record["execution_status"] == "running":
        try:
            os.kill(record["supervisor_pid"], 0)
        except ProcessLookupError:
            record["supervisor_unavailable"] = True
            record["recovery_note"] = "Supervisor exited unexpectedly; inspect worker.log and worker_pid before rerunning"
    return record


def worker(path):
    from .runner import run_batch
    snapshot = read_json(path.with_name("snapshot.json"))
    record = read_json(path)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Background reduction cancelled")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_batch(snapshot["settings"], snapshot["stage"],
                            accept=snapshot["accept"],
                            codex_session_log=snapshot["codex_session_log"],
                            _plan=snapshot["plan"], _locks_held=True,
                            lifecycle={"agent_job_id": record["job_id"],
                                       "execution_mode": "suspend"})
        write_json(path.with_name("result.json"), {"results": results})
        return 2 if any(item["status"] == "needs_review" for item in results) else 0
    except BaseException as error:
        write_json(path.with_name("result.json"), {"results": [],
                   "error": f"{type(error).__name__}: {error}"})
        raise


def _collect_results(path, record):
    """Recover partial sample outcomes even if an exception prevented batch return."""
    snapshot = read_json(path.with_name("snapshot.json"))
    existing = {item["directory"]: item for item in record.get("results", [])}
    results = []
    for entry in snapshot["plan"]["entries"]:
        directory = Path(entry["directory"])
        item = existing.get(str(directory))
        if item is None:
            states = [target for target in (directory / "state.json", directory / ".harness/state.json")
                      if target.exists()]
            state = read_json(max(states, key=lambda target: target.stat().st_mtime_ns)) if states else {}
            item = {"sample": entry["settings"]["sample_name"], "directory": str(directory),
                    "status": state.get("status", "not_started")}
        item["summary_name"] = "summary.md" if entry["settings"]["output"]["profile"] == "compact" else "report.md"
        results.append(item)
    record["results"] = results


def _safe_summaries(path, record, locks_held=False):
    handles = []
    try:
        if not locks_held:
            handles = reserve(read_json(path.with_name("snapshot.json"))["plan"])
        _update_summaries(record)
    except (OSError, ValueError) as error:
        record["summary_update_error"] = str(error)
        write_json(path, record)
    finally:
        close_locks(handles)


def supervise(path, handles):
    started = time.monotonic()
    process = None
    record = read_json(path)
    try:
        record.update(execution_status="running", started_at=utc_now(), supervisor_pid=os.getpid())
        write_json(path, record)
        process = subprocess.Popen([sys.executable, "-m", "cdsaxs.harness.jobs", "worker", str(path)],
                                   stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=handles)
        record["worker_pid"] = process.pid
        write_json(path, record)
        cancelled = False
        while process.poll() is None:
            if path.with_name("cancel.json").exists():
                cancelled = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        result_path = path.with_name("result.json")
        outcome = read_json(result_path) if result_path.exists() else {"results": []}
        record.update(outcome)
        record["worker_exit_code"] = process.returncode
        record["execution_status"] = ("cancelled" if cancelled else
                                      "failed" if outcome.get("error") else
                                      "needs_review" if process.returncode == 2 else
                                      "complete" if process.returncode == 0 else "failed")
    except BaseException as error:
        record.update(execution_status="failed", error=f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        record.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        try:
            _collect_results(path, record)
        except (OSError, ValueError) as error:
            record["result_collection_error"] = str(error)
        summaries = [Path(item["directory"]) / item.get("summary_name", "summary.md")
                     for item in record.get("results", [])]
        record["summaries"] = [str(summary) for summary in summaries if summary.exists()]
        with control_lock(path):
            if path.with_name("cancel.json").exists():
                record.update(execution_status="cancelled", notification_status="suppressed")
            write_json(path, record)
            _safe_summaries(path, record, locks_held=True)
            # Resumption may start immediately. Release scientific destinations
            # before notifying; subsequent summary updates take fresh locks.
            close_locks(handles)
            handles = []
            _notify(path, record)
        close_locks(handles)


if __name__ == "__main__":
    action, target, *descriptors = sys.argv[1:]
    path = Path(target).resolve()
    if action == "supervise":
        supervise(path, [int(value) for value in descriptors])
    elif action == "worker":
        raise SystemExit(worker(path))
    else:
        raise SystemExit(f"Unknown job action: {action}")
