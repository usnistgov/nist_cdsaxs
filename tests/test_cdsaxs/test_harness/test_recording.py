"""Usage accounting from asynchronous JSONL logs and nested runtime records."""
import json

import pytest

from cdsaxs.harness import recording
from cdsaxs.harness.recording import CodexUsageLog, RunRecorder, session_log_path, token_delta


def event(input_tokens, cached_input_tokens, output_tokens):
    return json.dumps({
        "timestamp": "2026-11-09T12:00:00Z", "type": "event_msg",
        "payload": {"type": "token_count", "info": {"total_token_usage": {
            "input_tokens": input_tokens, "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
            "reasoning_output_tokens": 2,
        }}},
    })


def test_usage_waits_for_complete_events_and_uses_cumulative_differences(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('invalid JSON\n' + event(100, 60, 10) + '\n')
    reader = CodexUsageLog(path)
    start = reader.snapshot()
    second = event(150, 90, 20)
    with path.open("a") as handle:
        handle.write(second)
    assert reader.snapshot() == start  # An event still being written is not a boundary.
    with path.open("a") as handle:
        handle.write('\n' + second + '\n')  # Repeated cumulative events cannot double count.
    end = reader.snapshot()
    assert reader.snapshot() == end
    assert token_delta(start, end) == {
        "input_tokens": 50, "cached_input_tokens": 30,
        "output_tokens": 10, "total_tokens": 60,
    }
    assert token_delta(end, end)["total_tokens"] == 0
    # Truncation/reset must not become negative or fabricated zero usage.
    path.write_text(event(2, 1, 0) + '\n')
    reset = reader.snapshot()
    assert token_delta(end, reset) is None
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(event(5000, 3000, 50) + '\n')
    replacement.replace(path)
    assert token_delta(reset, reader.snapshot()) is None


def test_missing_or_invalid_counters_are_unavailable(tmp_path):
    path = tmp_path / "session.jsonl"
    reader = CodexUsageLog(path)
    assert reader.snapshot() is None
    assert "Cannot read" in reader.error
    invalid_info = '{"type":"event_msg","payload":{"type":"token_count","info":null}}'
    path.write_text('\n'.join((event(-1, 0, 0), event(True, 0, 0), invalid_info)) + '\n')
    assert reader.snapshot() is None
    assert token_delta(None, None) is None
    assert reader.error == "No published token-count event"


def test_thread_discovery_does_not_choose_another_session(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAXS_CODEX_SESSION_LOG", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_THREAD_ID", "abcd-1234")
    directory = tmp_path / "sessions/2026/11/09"
    directory.mkdir(parents=True)
    other = directory / "rollout-other.jsonl"
    other.touch()
    assert session_log_path() is None
    expected = directory / "rollout-abcd-1234.jsonl"
    expected.touch()
    assert session_log_path() == expected
    monkeypatch.setenv("CDSAXS_CODEX_SESSION_LOG", str(other))
    assert session_log_path() == other
    assert session_log_path(expected) == expected
    monkeypatch.delenv("CDSAXS_CODEX_SESSION_LOG")
    monkeypatch.setenv("CODEX_THREAD_ID", "*")
    assert session_log_path() is None


def test_nested_timings_and_failure_recording(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    path.write_text(event(100, 60, 10) + '\n')
    times = iter((1, 2, 3, 7, 9, 10))
    monkeypatch.setattr(recording, "perf_counter", lambda: next(times))
    state = {}
    recorder = RunRecorder(state, path)
    with pytest.raises(RuntimeError, match="failed computation"):
        with recorder.stage("load"):
            with recorder.stage("metadata"):
                with path.open("a") as handle:
                    handle.write(event(150, 90, 20) + '\n')
                raise RuntimeError("failed computation")
    recorder.finish("needs_review")
    record = state["recordings"][0]
    parent, child = record["steps"]
    assert parent["elapsed_seconds"] == 7
    assert child["elapsed_seconds"] == 4
    assert record["elapsed_seconds"] == 9
    assert parent["parent_index"] is None and child["parent_index"] == 0
    assert parent["status"] == child["status"] == "failed"
    assert parent["error_type"] == child["error_type"] == "RuntimeError"
    assert record["token_delta"]["total_tokens"] == 60
    assert parent["token_delta"] == child["token_delta"] == record["token_delta"]


def test_resume_captures_intervening_activity_only_in_same_session(tmp_path):
    state = {}
    path = tmp_path / "session.jsonl"
    path.write_text(event(100, 60, 10) + '\n')
    first = RunRecorder(state, path)
    first.finish("stage_complete")
    with path.open("a") as handle:
        handle.write(event(150, 90, 20) + '\n')
    second = RunRecorder(state, path)
    assert second.record["between_invocations_token_delta"]["total_tokens"] == 60
    second.finish("complete")
    third = RunRecorder(state, tmp_path / "another-session.jsonl")
    assert third.record["between_invocations_token_delta"] is None
    third.finish("complete")
    assert len(state["recordings"]) == 3
    assert third.record["token_delta"] is None
