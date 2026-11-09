"""Background lifecycle tests use local fake notifiers, never a live conversation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest
import tifffile

from cdsaxs.harness import jobs
from cdsaxs.harness.__main__ import main
from cdsaxs.harness.config import DEFAULTS, write_json
from cdsaxs.harness.runner import run_batch

PYTHON = sys.executable


@pytest.fixture
def cfg(tmp_path):
    settings = deepcopy(DEFAULTS)
    yy, xx = np.mgrid[:81, :121]
    image = np.full((81, 121), 30.0)
    for column in (25, 40, 80, 95):
        image += 2000 * np.exp(-((yy - 40) ** 2 / 2 + (xx - column) ** 2 / 3))
    filenames = []
    for angle in (-30, 0, 30):
        name = f"sample_{angle}.tif"
        tifffile.imwrite(tmp_path / name, image)
        filenames.append(name)
    settings["input"].update(data_directory=str(tmp_path), filenames=filenames,
        filetype="tif", metadata_pattern="sample_{sample_phi_deg}.tif",
        metadata={"energy_ev": 16100, "pixel_size_um": 172, "exposure_time_s": 2},
        user_params={"bpm": 4})
    settings["sample_kind"] = "circular_contact_holes"
    settings["calibration"].update(center_px=[40, 60], sdd_cm=300)
    settings["offsets"] = {key: 0.0 for key in settings["offsets"]}
    settings["offsets"]["sample_phi_offset_deg"] = 0.3
    settings["corrections"]["substrate_absorption"] = False
    settings["integration"].update(height_px=7, width_px=100, background_offsets_px=[-10, 10])
    settings["quality"]["max_alignment_residual_px"] = 3
    settings["output"].update(root=str(tmp_path / "outputs"), dpi=60)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["choose_slice_locations_and_widths", "skip_gap_slices"]
    return settings


@pytest.fixture
def notifier(tmp_path, monkeypatch):
    """Mock at the executable boundary so detached processes inherit the mock."""
    script = tmp_path / "codex"
    script.write_text(f"#!{PYTHON}\n" + '''import json, os, sys
from pathlib import Path
if '--help' in sys.argv:
    print('queue --thread UUID --message TEXT --resume')
else:
    if os.environ.get('TEST_NOTIFY_CHECK_LOCK'):
        import fcntl
        with open(os.environ['TEST_NOTIFY_CHECK_LOCK'], 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with Path(os.environ['TEST_NOTIFY_LOG']).open('a') as log:
        log.write(json.dumps(sys.argv[1:]) + '\\n')
    print(os.environ.get('TEST_NOTIFY_OUTPUT', 'Queued and resumed'))
    sys.exit(int(os.environ.get('TEST_NOTIFY_EXIT', '0')))
''')
    script.chmod(0o755)
    log = tmp_path / "notifications.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_NOTIFY_LOG", str(log))
    monkeypatch.chdir(tmp_path)
    return log


def completed(path, timeout=30):
    """Bounded test-only polling; production agent workflows never poll."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.status(path)
        if (record["execution_status"] in ("complete", "needs_review", "failed", "cancelled")
                and record["notification_status"] in ("delivered", "pending", "suppressed")):
            # Synchronize with notifier and summary updates before asserting.
            with jobs.control_lock(path):
                return jobs.status(path)
        time.sleep(0.05)
    raise AssertionError(f"Job timed out: {jobs.status(path)}\n{Path(path).with_name('worker.log').read_text()}")


@pytest.mark.parametrize("mode", ["automatic", "review"])
def test_background_matches_foreground_science_and_stage_order(cfg, notifier, mode):
    cfg["mode"] = mode
    foreground = run_batch(cfg)
    thread = str(uuid4())
    launched = jobs.launch(cfg, "all", thread_id=thread, command_args=["--stage", "all"])
    record = completed(launched["record_path"])
    assert record["execution_status"] == ("complete" if mode == "automatic" else "needs_review")
    assert record["worker_exit_code"] == (0 if mode == "automatic" else 2)
    calls = [json.loads(line) for line in notifier.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][:3] == ["queue", "--thread", thread]
    assert calls[0][-1] == "--resume"
    assert str(launched["record_path"]) in calls[0][4]
    assert jobs.notify(launched["record_path"])["notification_status"] == "delivered"
    assert len(notifier.read_text().splitlines()) == 1
    left = Path(foreground[0]["directory"])
    right = Path(record["results"][0]["directory"])
    a = json.loads((left / ".harness/state.json").read_text())
    b = json.loads((right / ".harness/state.json").read_text())
    assert a["status"] == b["status"]
    assert a["stages"] == b["stages"]
    for relative in ("settings.json",):
        assert (left / relative).read_bytes() == (right / relative).read_bytes()
    files = sorted(path.name for path in (left / "csv").glob("*.csv"))
    assert files == sorted(path.name for path in (right / "csv").glob("*.csv"))
    for filename in files:
        assert (left / "csv" / filename).read_bytes() == (right / "csv" / filename).read_bytes()
    steps_a = [step["stage"] for step in a["recordings"][-1]["steps"]]
    steps_b = [step["stage"] for step in b["recordings"][-1]["steps"]]
    assert steps_a == steps_b
    assert steps_b.index("phi") < steps_b.index("reload_accepted_phi") < steps_b.index("chi")
    invocation = b["recordings"][-1]
    assert invocation["execution_mode"] == "suspend"
    assert invocation["agent_job_id"] == record["job_id"]
    assert invocation["notification_status"] == "delivered"
    assert "notification delivered" in (right / "summary.md").read_text()


def test_preflight_rejects_missing_thread_before_reserving(cfg, notifier, monkeypatch):
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    with pytest.raises(ValueError, match="agent-thread-id"):
        jobs.launch(cfg, "all")
    with pytest.raises(ValueError, match="agent-thread-id"):
        jobs.launch(cfg, "all", thread_id="unsafe/thread")
    assert not Path(cfg["output"]["root"]).exists()
    assert not notifier.exists()


@pytest.mark.parametrize("exit_code,output", [(0, "Warning: queued but resume failed"),
                                             (1, "Server unavailable")])
def test_notifications_pending_recovery_and_cancellation(cfg, notifier, monkeypatch, exit_code, output):
    monkeypatch.setenv("TEST_NOTIFY_OUTPUT", output)
    monkeypatch.setenv("TEST_NOTIFY_EXIT", str(exit_code))
    launched = jobs.launch(cfg, "calibration", thread_id=str(uuid4()))
    path = Path(launched["record_path"])
    record = completed(path)
    assert record["execution_status"] == "complete"
    assert record["notification_status"] == "pending"
    assert len(notifier.read_text().splitlines()) == 1
    monkeypatch.delenv("TEST_NOTIFY_OUTPUT")
    monkeypatch.delenv("TEST_NOTIFY_EXIT")
    assert jobs.notify(path)["notification_status"] == "delivered"
    assert len(notifier.read_text().splitlines()) == 2
    # Cancellation also suppresses an explicitly requested notification recovery.
    jobs.cancel(path)
    record["notification_status"] = "pending"
    write_json(path, record)
    assert jobs.notify(path)["notification_status"] == "suppressed"
    assert len(notifier.read_text().splitlines()) == 2


def test_background_batch_preserves_independent_outcomes_and_profiles(cfg, notifier):
    cfg["samples"] = [{"name": "good"}, {"name": "held",
                      "input": {"filenames": ["missing.tif"]}, "output": {"profile": "full"}}]
    launched = jobs.launch(cfg, "all", thread_id=str(uuid4()))
    record = completed(launched["record_path"])
    assert record["execution_status"] == "needs_review"
    good, held = record["results"]
    assert good["sample"] == "good" and good["status"] == "complete"
    assert held["sample"] == "held" and held["status"] == "needs_review"
    assert good["summary_name"] == "summary.md"
    assert held["summary_name"] == "report.md"
    assert (Path(good["directory"]) / "csv").is_dir()
    assert (Path(held["directory"]) / "failure.json").exists()
    assert len(notifier.read_text().splitlines()) == 1


@pytest.fixture
def delayed_worker(tmp_path, monkeypatch):
    proxy = tmp_path / "delayed-python"
    proxy.write_text(f"#!{PYTHON}\n" + '''import os, sys, time
from pathlib import Path
from cdsaxs.harness import jobs
action, target, *fds = sys.argv[3:]
path = Path(target)
if action == 'supervise':
    jobs.sys.executable = __file__
    jobs.supervise(path, [int(fd) for fd in fds])
else:
    time.sleep(2)
    if os.environ.get('TEST_WORKER_CRASH'):
        raise RuntimeError('deliberate worker exception')
    sys.exit(jobs.worker(path))
''')
    proxy.chmod(0o755)
    monkeypatch.setattr(jobs.sys, "executable", str(proxy))
    return proxy


def test_prompt_return_duplicate_launch_and_cancel(cfg, notifier, delayed_worker):
    resume = Path(cfg["input"]["data_directory"]) / "reserved"
    started = time.monotonic()
    launched = jobs.launch(cfg, "all", resume=resume, thread_id=str(uuid4()))
    assert time.monotonic() - started < 1.5
    with pytest.raises(ValueError, match="already in use"):
        jobs.launch(cfg, "all", resume=resume, thread_id=str(uuid4()))
    with pytest.raises(ValueError, match="already in use"):
        run_batch(cfg, resume=resume)
    jobs.cancel(launched["record_path"])
    record = completed(launched["record_path"])
    assert record["execution_status"] == "cancelled"
    assert record["notification_status"] == "suppressed"
    assert not notifier.exists()
    handles = jobs.reserve(jobs.plan_batch(cfg, resume))
    jobs.close_locks(handles)


def test_detached_survives_launcher_exit(cfg, notifier, delayed_worker, tmp_path):
    config = tmp_path / "input.json"
    write_json(config, cfg)
    thread = str(uuid4())
    # The launcher is a separate short-lived process, equivalent to ending a tool call.
    command = [PYTHON, "-c", """import json, sys
from cdsaxs.harness import jobs
from cdsaxs.harness.config import load_settings
jobs.sys.executable = sys.argv[2]
print(json.dumps(jobs.launch(load_settings(sys.argv[1]), 'all', thread_id=sys.argv[3])))
""", str(config), str(delayed_worker), thread]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    path = json.loads(result.stdout)["record_path"]
    assert jobs.status(path)["execution_status"] in ("launching", "running")
    assert time.monotonic() - started < 4
    record = completed(path)
    assert record["execution_status"] == "complete"
    assert record["elapsed_seconds"] >= 2
    assert len(notifier.read_text().splitlines()) == 1


def test_worker_crash_notifies_once_and_preserves_record(cfg, notifier, delayed_worker, monkeypatch):
    monkeypatch.setenv("TEST_WORKER_CRASH", "1")
    launched = jobs.launch(cfg, "all", thread_id=str(uuid4()))
    record = completed(launched["record_path"])
    assert record["execution_status"] == "failed"
    assert record["worker_exit_code"] == 1
    assert record["notification_status"] == "delivered"
    assert record["results"][0]["status"] == "not_started"
    assert "deliberate worker exception" in Path(record["worker_log"]).read_text()
    assert len(notifier.read_text().splitlines()) == 1


def test_cli_recovery_without_config(cfg, notifier, capsys):
    launched = jobs.launch(cfg, "calibration", thread_id=str(uuid4()))
    record = completed(launched["record_path"])
    assert main(["--agent-job-status", launched["record_path"]]) == 0
    assert record["job_id"] in capsys.readouterr().out
    assert main(["--agent-job-notify", launched["record_path"]]) == 0
    assert len(notifier.read_text().splitlines()) == 1


def test_fast_completion_does_not_get_overwritten(cfg, notifier, monkeypatch):
    original = jobs.subprocess.Popen
    def finish_before_return(*args, **kwargs):
        process = original(*args, **kwargs)
        process.wait(timeout=30)
        return process
    monkeypatch.setattr(jobs.subprocess, "Popen", finish_before_return)
    launched = jobs.launch(cfg, "calibration", thread_id=str(uuid4()))
    assert jobs.status(launched["record_path"])["execution_status"] == "complete"
    assert len(notifier.read_text().splitlines()) == 1


def test_notification_releases_run_directory_for_immediate_continuation(cfg, notifier, monkeypatch):
    directory = Path(cfg["input"]["data_directory"]) / "immediate-resume"
    monkeypatch.setenv("TEST_NOTIFY_CHECK_LOCK", str(directory / ".harness/run.lock"))
    launched = jobs.launch(cfg, "calibration", resume=directory, thread_id=str(uuid4()))
    assert completed(launched["record_path"])["notification_status"] == "delivered"


def test_launcher_is_lightweight_and_public_imports_remain_available():
    result = subprocess.run([PYTHON, "-c", """import sys
import cdsaxs.harness.__main__
assert 'numpy' not in sys.modules
from cdsaxs import LoadData, LoadDataset, filter_filenames, slice_reduced_dataset
from cdsaxs.loaders.load_data import LoadData as implementation
assert LoadData is implementation
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
