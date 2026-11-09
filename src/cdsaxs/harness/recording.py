"""Wall-clock timings and published Codex usage; no model or network calls."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def session_log_path(explicit=None):
    """Find this thread's log only; never select an unrelated latest session."""
    explicit = explicit or os.environ.get("CDSAXS_CODEX_SESSION_LOG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    thread = os.environ.get("CODEX_THREAD_ID")
    if not thread:
        return None
    # Validate before using an environment value in a glob pattern.
    if any(character not in "0123456789abcdefABCDEF-" for character in thread):
        return None
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    try:
        matches = [path for directory in (root / "sessions", root / "archived_sessions")
                   for path in directory.rglob(f"*{thread}*.jsonl")]
        return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None
    except OSError:
        return None


class CodexUsageLog:
    """Read token events incrementally, retaining neither prompts nor tool output."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.generation = 0
        self.latest = None
        self.error = None

    def snapshot(self):
        try:
            with self.path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset, self.latest = 0, None
                    self.generation += 1
                self.identity = identity
                handle.seek(self.offset)
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        # Retry a partially appended JSON event on the next snapshot.
                        break
                    self.offset = handle.tell()
                    if b'"token_count"' not in line:
                        continue
                    try:
                        event = json.loads(line)
                        payload = event["payload"]
                        if event["type"] != "event_msg" or payload["type"] != "token_count":
                            continue
                        tokens = payload["info"]["total_token_usage"]
                        if not all(type(tokens[key]) is int and tokens[key] >= 0
                                   for key in TOKEN_FIELDS):
                            continue
                        self.latest = {"timestamp": event.get("timestamp"),
                                       "log_identity": [*identity, self.generation],
                                       "tokens": {key: tokens[key] for key in TOKEN_FIELDS}}
                    except (ValueError, KeyError, TypeError):
                        continue
            self.error = None if self.latest else "No published token-count event"
            return deepcopy(self.latest)
        except OSError as error:
            self.error = f"Cannot read Codex session log: {error.strerror or type(error).__name__}"
            return None


def token_delta(start, end):
    """Unknown/reset counters are unavailable, rather than zero or negative usage."""
    if start is None or end is None:
        return None
    if start.get("log_identity") != end.get("log_identity"):
        return None
    delta = {key: end["tokens"][key] - start["tokens"][key] for key in TOKEN_FIELDS}
    return delta if all(value >= 0 for value in delta.values()) else None


def measured(recorder, name):
    """Keep the public standalone numerical helpers usable without a recorder."""
    return recorder.stage(name) if recorder is not None else nullcontext()


class RunRecorder:
    """Append invocation/step records to existing state, preserving resume history."""

    def __init__(self, state, session_log=None, persist=None):
        self.log = CodexUsageLog(session_log) if session_log is not None else None
        self.persist = persist
        self.stack = []
        self.started = perf_counter()
        history = state.setdefault("recordings", [])
        self.record = {
            "invocation_id": uuid4().hex, "started_at": utc_now(),
            "requested_stage": None, "status": "running", "elapsed_seconds": None,
            "codex_session_log": str(session_log) if session_log is not None else None,
            "token_start": self.snapshot(), "token_end": None, "token_delta": None,
            "between_invocations_token_delta": None, "steps": [],
        }
        if history and self.record["codex_session_log"] == history[-1].get("codex_session_log"):
            self.record["between_invocations_token_delta"] = token_delta(
                history[-1].get("token_end"), self.record["token_start"])
        history.append(self.record)

    def snapshot(self):
        return self.log.snapshot() if self.log is not None else None

    @contextmanager
    def stage(self, name, cached=False):
        entry = {
            "stage": name, "parent_index": self.stack[-1] if self.stack else None,
            "started_at": utc_now(), "finished_at": None, "elapsed_seconds": None,
            "status": "running", "cached": cached,
            "token_start": self.snapshot(), "token_end": None, "token_delta": None,
        }
        index = len(self.record["steps"])
        self.record["steps"].append(entry)
        outermost = not self.stack
        self.stack.append(index)
        if outermost and self.persist is not None:
            self.persist()
        print(f"Recording {name}" + (" (cached)" if cached else ""), flush=True)
        started = perf_counter()
        try:
            yield entry
            entry["status"] = "completed"
        except BaseException as error:
            entry["status"] = "failed"
            entry["error_type"] = type(error).__name__
            raise
        finally:
            entry["elapsed_seconds"] = perf_counter() - started
            entry["finished_at"] = utc_now()
            entry["token_end"] = self.snapshot()
            entry["token_delta"] = token_delta(entry["token_start"], entry["token_end"])
            self.stack.pop()
            delta = entry["token_delta"]
            usage = f"{delta['total_tokens']} tokens" if delta is not None else "tokens unavailable"
            print(f"{name}: {entry['elapsed_seconds']:.3f} s; {usage}; {entry['status']}",
                  flush=True)
            if outermost and self.persist is not None:
                self.persist()

    def finish(self, status):
        self.record.update(status=status, finished_at=utc_now(),
                           elapsed_seconds=perf_counter() - self.started,
                           token_end=self.snapshot())
        self.record["token_delta"] = token_delta(
            self.record["token_start"], self.record["token_end"])
        self.record["token_note"] = (self.log.error if self.log is not None
                                     else "No Codex session log selected")
        if self.record["token_start"] is None and self.record["token_end"] is not None:
            self.record["token_note"] = "No published baseline at invocation start"
        elif (self.record["token_delta"] is None and self.record["token_start"] is not None
              and self.record["token_end"] is not None):
            self.record["token_note"] = "Codex counters reset or session log replaced"
