"""Scientific and orchestration checks for the standalone reduction harness."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from cdsaxs.harness.config import DEFAULTS, load_settings, validate, write_json
from cdsaxs.harness.diagnostics import (
    NeedsReview, box_ranges, choose_backgrounds, fit_phi, row_center, slice_coverage,
)
from cdsaxs.harness.runner import (
    ReductionRun, apply_intensity_corrections, integrate_row, load_dataset,
    phi_groups, run_batch,
)
from cdsaxs.harness.script_generator import generate


@pytest.fixture
def settings(tmp_path):
    """Resolved settings for synthetic circular-hole peak rows."""
    cfg = deepcopy(DEFAULTS)
    cfg["input"]["data_directory"] = str(tmp_path)
    cfg["input"]["filenames"] = []
    cfg["input"]["filetype"] = "tif"
    cfg["input"]["metadata"] = {
        "energy_ev": 16100, "pixel_size_um": 172, "exposure_time_s": 2,
    }
    cfg["input"]["user_params"] = {"bpm": 4}
    cfg["input"]["metadata_pattern"] = "sample_{sample_phi_deg}.tif"
    cfg["sample_kind"] = "circular_contact_holes"
    cfg["calibration"].update(center_px=[40, 60], sdd_cm=300)
    cfg["offsets"] = {key: 0.0 for key in cfg["offsets"]}
    cfg["corrections"]["substrate_absorption"] = False
    cfg["integration"].update(height_px=7, width_px=100, background_offsets_px=[-10, 10])
    cfg["quality"]["max_alignment_residual_px"] = 3
    cfg["output"].update(root=str(tmp_path / "outputs"), figure_formats=["png"], dpi=60)
    return cfg


def write_images(cfg, angles=(-30, 0, 30), extra_rows=False):
    yy, xx = np.mgrid[:81, :121]
    base = np.full((81, 121), 30.0)
    for row in ([25, 40, 55] if extra_rows else [40]):
        for column in (25, 40, 80, 95):
            base += 2000 * np.exp(-((yy - row) ** 2 / 2 + (xx - column) ** 2 / 3))
    files = []
    for angle in angles:
        name = f"sample_{angle}.tif"
        tifffile.imwrite(Path(cfg["input"]["data_directory"]) / name, base)
        files.append(name)
    cfg["input"]["filenames"] = files
    return base


def test_json_profiles_paths_null_and_zero(tmp_path):
    write_json(tmp_path / "profile.json", {
        "offsets": {"sample_phi_offset_deg": 0.0}, "integration": {"height_px": 8},
    })
    write_json(tmp_path / "config.json", {
        "sample_profile": "profile.json", "input": {"data_directory": "raw"},
        "integration": {"height_px": None},
    })
    cfg = load_settings(tmp_path / "config.json")
    assert cfg["offsets"]["sample_phi_offset_deg"] == 0.0
    assert cfg["integration"]["height_px"] is None
    assert cfg["input"]["data_directory"] == str(tmp_path / "raw")
    (tmp_path / "bad.json").write_text('{"calibration":{"sdd_cm":NaN}}')
    with pytest.raises(ValueError, match="Nonfinite"):
        load_settings(tmp_path / "bad.json")


@pytest.mark.parametrize("change", [
    {"mode": "autmatic"}, {"calibration": {"sdd_cm": -1}},
    {"additional_qy_rows": {"enabled": True}},
    {"slices": {"centers_qsx_Ainv": [0.01, 0.02], "widths_Ainv": [0.001]}},
    {"unknown_option": True},
])
def test_invalid_settings_are_rejected(change):
    from cdsaxs.harness.config import merge
    with pytest.raises(ValueError):
        validate(merge(DEFAULTS, change))


def test_phi_sign_and_model_checks():
    angles = np.linspace(-5, 5, 11)
    distances = 300 + 0.2 * (angles + 0.7) ** 2
    result = fit_phi(angles, distances, DEFAULTS["quality"])
    assert result["sample_phi_offset_deg"] == pytest.approx(0.7, abs=1e-10)
    assert not result["failures"]
    with pytest.raises(NeedsReview, match="curvature"):
        fit_phi(angles, 300 - angles ** 2, DEFAULTS["quality"])
    outside = fit_phi(angles, 300 + (angles + 9) ** 2, DEFAULTS["quality"])
    assert any("outside" in failure for failure in outside["failures"])


def test_reload_reapplies_corrections_without_accumulation(settings):
    base = write_images(settings)
    calibration = settings["calibration"]
    offsets = settings["offsets"]
    first = load_dataset(settings, calibration, offsets)
    apply_intensity_corrections(first, settings)
    # Simulate the accepted phi offset and mandatory fresh load.
    offsets = {**offsets, "sample_phi_offset_deg": 0.5}
    second = load_dataset(settings, calibration, offsets)
    apply_intensity_corrections(second, settings)
    for data in second.datas.values():
        expected = base / 8 * np.cos(np.deg2rad(data.sample_phi_deg))
        np.testing.assert_allclose(data.image, expected)
    third = load_dataset(settings, calibration, offsets)
    apply_intensity_corrections(third, settings)
    for left, right in zip(second.datas.values(), third.datas.values()):
        np.testing.assert_array_equal(left.image, right.image)


def test_physical_qy_rows_and_pixel_offsets(settings):
    write_images(settings, angles=(0,))
    data = next(iter(load_dataset(settings, settings["calibration"], settings["offsets"]).datas.values()))
    target = float(data.qsy[25, 60])
    assert row_center(data, {"target_qsy_Ainv": target}) == pytest.approx(25)
    assert row_center(data, {"offset_qdy_px": 15}) == 25


def test_gap_coverage_and_background_boundaries(settings):
    write_images(settings, angles=(0,))
    dataset = load_dataset(settings, settings["calibration"], settings["offsets"])
    data = next(iter(dataset.datas.values()))
    data.mask[:, 80] = True
    row = {"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": None}
    geometry = {"height_px": 7, "width_px": 100,
                "background_offsets_px": [-10, 10], "mode": "sum"}
    reduced = integrate_row(dataset, row, geometry)
    center = float(reduced.datas[0].qsx[80 - reduced.datas[0].limits_axis1[0]])
    report = slice_coverage(reduced, center, 0.001)
    assert report[0]["valid"] is False
    assert report[0]["reason"] == "masked_or_nonfinite_selection"
    with pytest.raises(NeedsReview, match="leaves detector"):
        integrate_row(dataset, row, {**geometry, "background_offsets_px": [50]})


def test_box_bounds_use_native_fractional_center_and_reject_clipping(settings):
    write_images(settings, angles=(0,))
    settings["calibration"]["center_px"] = [40.2, 60.2]
    data = next(iter(load_dataset(
        settings, settings["calibration"], settings["offsets"]).datas.values()))
    row = {"row_id": "qy0", "offset_qdy_px": 0}
    geometry = {"height_px": 6, "width_px": 100}
    expected = data.get_box_dims_size(6, 100, center=(40.2, 60.2))
    assert box_ranges(data, row, geometry) == expected
    assert expected == ((38, 44), (11, 111))
    with pytest.raises(NeedsReview, match="exceeds"):
        box_ranges(data, row, {"height_px": 1000, "width_px": 100})


@pytest.mark.parametrize("height", [7, 14])
@pytest.mark.parametrize("strategy,sample_kind", [
    ("adjacent", None), ("auto", "contact_holes"), ("auto", "circular_contact_holes"),
])
def test_adjacent_background_gap_is_less_than_box_height(
        settings, height, strategy, sample_kind, monkeypatch):
    write_images(settings, angles=(0,))
    settings["calibration"]["center_px"] = [40.2, 60.2]
    settings["sample_kind"] = sample_kind
    settings["integration"].update(background_offsets_px=None, background_strategy=strategy)
    data = next(iter(load_dataset(
        settings, settings["calibration"], settings["offsets"]).datas.values()))
    # Background contrast decreases with distance: cleaner far boxes must not
    # escape the adjacent gap limit, including at a fractional beam center.
    amplitude = np.maximum(40 - abs(np.arange(81) - 40.2), 0)
    pattern = np.zeros(121)
    pattern[80] = 100
    data.image[:] = 30 + amplitude[:, None] * pattern
    row = {"row_id": "qy0", "offset_qdy_px": 0}
    geometry = {"height_px": height, "width_px": 100}
    yr, xr = box_ranges(data, row, geometry)
    visited_gaps = []
    native_mean = data.mean_box

    def record_mean(limits_axis0, limits_axis1, axis):
        if limits_axis0 != yr:
            lo, hi = limits_axis0
            assert hi - lo == height
            assert limits_axis1 == xr
            gap = yr[0] - hi if hi <= yr[0] else lo - yr[1]
            assert 0 <= gap < height
            visited_gaps.append(gap)
        return native_mean(limits_axis0, limits_axis1, axis)

    monkeypatch.setattr(data, "mean_box", record_mean)
    offsets, score = choose_backgrounds([(data, [])], row, geometry, settings)
    assert offsets == [-(2 * height - 1), 2 * height - 1]
    assert min(visited_gaps) == 0
    assert max(visited_gaps) == height - 1
    assert score > 0  # Cleaner far boxes exist but were excluded.


def test_adjacent_background_search_holds_when_nearby_regions_are_masked(settings):
    write_images(settings, angles=(0,))
    settings["integration"].update(background_offsets_px=None, background_strategy="adjacent")
    data = next(iter(load_dataset(
        settings, settings["calibration"], settings["offsets"]).datas.values()))
    row = {"row_id": "qy0", "offset_qdy_px": 0}
    geometry = {"height_px": 7, "width_px": 100}
    yr, xr = box_ranges(data, row, geometry)
    data.mask[yr[0] - 13:yr[0], xr[0]:xr[1]] = True
    data.mask[yr[1]:yr[1] + 13, xr[0]:xr[1]] = True
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        with pytest.raises(NeedsReview, match="No usable background regions"):
            choose_backgrounds([(data, [])], row, geometry, settings)


def test_explicit_background_offsets_override_adjacent_search(settings):
    write_images(settings, angles=(0,))
    settings["integration"].update(background_offsets_px=[-21, 21], background_strategy="adjacent")
    data = next(iter(load_dataset(
        settings, settings["calibration"], settings["offsets"]).datas.values()))
    offsets, _ = choose_backgrounds(
        [(data, [])], {"row_id": "qy0", "offset_qdy_px": 0},
        {"height_px": 7, "width_px": 100}, settings)
    assert offsets == [-21, 21]


def test_native_waterfall_does_not_reset_after_eight_slices(settings, tmp_path, monkeypatch):
    from cdsaxs.plotting import plotting
    write_images(settings, angles=(0,))
    dataset = load_dataset(settings, settings["calibration"], settings["offsets"])
    row = {"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": None}
    geometry = {"height_px": 7, "width_px": 100,
                "background_offsets_px": None, "mode": "sum"}
    integrated = integrate_row(dataset, row, geometry)
    centers = integrated.datas[0].qsx[np.linspace(0, 99, 12, dtype=int)]
    items = [{"center_qsx_Ainv": float(center), "width_Ainv": 1e-7} for center in centers]
    settings["export"]["split_phi"] = False
    observed = []
    native = plotting.plot_reduced_slices

    def capture(slices, **kwargs):
        before = [data.Iq.copy() for data in slices.data]
        figure, axes = native(slices, **kwargs)
        ordered = sorted(slices.data, key=lambda data: float(data.qsx))
        np.testing.assert_allclose(axes.lines[8].get_ydata(), ordered[8].Iq * 1e-16)
        assert axes.get_legend() is not None
        for old, data in zip(before, slices.data):
            np.testing.assert_array_equal(old, data.Iq)
        observed.append(axes)
        return figure, axes

    monkeypatch.setattr(plotting, "plot_reduced_slices", capture)
    run = ReductionRun(settings, tmp_path / "native_waterfall")
    run.export(dataset, [(row, geometry, integrated, items, {"mode": "sum"})])
    assert len(observed) == 1
    assert len(observed[0].lines) == 12
    assert (run.directory / "csv/sample_all.csv").is_file()


@pytest.mark.parametrize("policy", ["positive", "negative", "separate", "exclude"])
def test_phi_partition_disjoint_and_corrected(settings, policy):
    write_images(settings, angles=(-1, 0, 1))
    dataset = load_dataset(settings, settings["calibration"], settings["offsets"])
    settings["export"]["zero_phi"] = policy
    groups = phi_groups(dataset, settings)
    assert not groups["posPhi"] & groups["negPhi"]
    assert len(groups["posPhi"]) == (2 if policy == "positive" else 1)
    assert len(groups["negPhi"]) == (2 if policy == "negative" else 1)
    dataset.update_all_metadata({"sample_phi_offset_deg": 2}, verbose=False)
    corrected = phi_groups(dataset, settings)
    assert len(corrected["posPhi"]) == 3
    settings["export"]["phi_basis"] = "nominal"
    nominal = phi_groups(dataset, settings)
    assert len(nominal["posPhi"]) == len(groups["posPhi"])


def test_review_acceptance_and_invalidation(settings, tmp_path):
    write_images(settings)
    run = ReductionRun(settings, tmp_path / "run")
    with pytest.raises(NeedsReview, match="Review phi"):
        run.checkpoint("phi", {"setting": 1}, lambda: {"offset": 0.2}, ["accept_angle_corrections"])
    resumed = ReductionRun(settings, tmp_path / "run", accept=["phi"])
    assert resumed.checkpoint("phi", {"setting": 1}, lambda: pytest.fail("cached"))["offset"] == 0.2
    with pytest.raises(NeedsReview, match="Inspect"):
        resumed.checkpoint("phi", {"setting": 2}, lambda: {"offset": 0.4})


def test_automatic_permissions_and_quality(settings, tmp_path):
    write_images(settings)
    settings["mode"] = "automatic"
    run = ReductionRun(settings, tmp_path / "run")
    with pytest.raises(NeedsReview, match="Review"):
        run.checkpoint("integration:qy0", {}, lambda: {}, ["choose_integration_box"])
    settings["allowed_decisions"] = ["choose_integration_box"]
    authorized = ReductionRun(settings, tmp_path / "run")
    authorized.checkpoint("integration:qy0", {}, lambda: {}, ["choose_integration_box"])
    with pytest.raises(NeedsReview, match="bad fit"):
        authorized.checkpoint("slices:qy0", {}, lambda: {"failures": ["bad fit"]}, supplied=True)


@pytest.mark.parametrize(
    "per_row,combine_qy_rows", [(True, True), (True, False), (False, True)],
)
def test_end_to_end_automatic_rows_and_native_csv(
        settings, tmp_path, per_row, combine_qy_rows):
    write_images(settings, extra_rows=True)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = [
        "choose_slice_locations_and_widths", "skip_gap_slices", "choose_qy_offsets",
    ]
    settings["additional_qy_rows"].update(enabled=True, offsets_qdy_px=[-15, 15])
    settings["integration"]["search_height_px"] = 12
    settings["export"]["split_phi"] = True
    settings["export"].update(per_row=per_row, combine_qy_rows=combine_qy_rows)
    run = ReductionRun(settings, tmp_path / "run")
    result = run.run()
    assert result["status"] == "complete", run.state["events"]
    rows = json.loads((tmp_path / "run" / "row_reductions.json").read_text())
    assert len(rows) == 3
    row_ids = [row["row"]["row_id"] for row in rows] if per_row else []
    if combine_qy_rows:
        row_ids.append("qyall")
    expected = {f"sample_{row_id}_{group}.csv"
                for row_id in row_ids for group in ("all", "posPhi", "negPhi")}
    csv_directory = tmp_path / "run" / "csv"
    assert {path.name for path in csv_directory.glob("*.csv")} == expected
    exported = csv_directory / f"sample_{row_ids[-1]}_all.csv"
    with exported.open(newline="") as handle:
        headers = next(csv.reader(handle))
    assert len(headers) > 5
    records = np.atleast_2d(np.genfromtxt(exported, delimiter=",", skip_header=1))
    radial = records[:, 3::5]
    assert np.isfinite(radial).any()
    assert np.all(radial[np.isfinite(radial)] > 0)
    # A resumed full reduction reproduces CSV values without accumulating corrections.
    original = exported.read_bytes()
    resumed = ReductionRun(settings, tmp_path / "run")
    assert resumed.run()["status"] == "complete"
    assert exported.read_bytes() == original
    assert {path.name for path in csv_directory.glob("*.csv")} == expected


def test_detect_additional_qy_rows(settings):
    from cdsaxs.harness.diagnostics import detect_qy_rows
    write_images(settings, angles=(0,), extra_rows=True)
    dataset = load_dataset(settings, settings["calibration"], settings["offsets"])
    data = next(iter(dataset.datas.values()))
    result = detect_qy_rows(data, settings)
    expected = sorted([float(data.qsy[25, 60]), float(data.qsy[55, 60])])
    np.testing.assert_allclose(result["offsets_qsy_Ainv"], expected)


@pytest.mark.parametrize(
    "per_row,combine_qy_rows", [(True, True), (True, False), (False, True)],
)
@pytest.mark.parametrize("split_phi", [False, True])
def test_central_only_exports_once_without_qy_labels(
        settings, tmp_path, per_row, combine_qy_rows, split_phi):
    # Visible extra rows and permission to choose offsets do not enable them.
    write_images(settings, extra_rows=True)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = [
        "choose_slice_locations_and_widths", "skip_gap_slices", "choose_qy_offsets",
    ]
    settings["export"].update(
        per_row=per_row, combine_qy_rows=combine_qy_rows, split_phi=split_phi,
    )
    run = ReductionRun(settings, tmp_path / "central_only")
    assert run.run()["status"] == "complete", run.state["events"]
    rows = json.loads((run.directory / "row_reductions.json").read_text())
    assert [row["row"]["row_id"] for row in rows] == ["qy0"]
    assert run.resolved["additional_qy_rows"]["enabled"] is False
    assert run.state["export_settings"]["per_row"] is False
    assert run.state["export_settings"]["combine_qy_rows"] is True
    groups = ("all", "posPhi", "negPhi") if split_phi else ("all",)
    expected = {f"sample_{group}.csv" for group in groups}
    assert {p.name for p in (run.directory / "csv").glob("*.csv")} == expected
    original = {name: (run.directory / "csv" / name).read_bytes() for name in expected}
    independent = ReductionRun(run.resolved, tmp_path / "independent_central")
    assert independent.run()["status"] == "complete", independent.state["events"]
    assert {p.name for p in (independent.directory / "csv").glob("*.csv")} == expected
    assert {name: (independent.directory / "csv" / name).read_bytes()
            for name in expected} == original


def test_batch_continues_after_missing_sample(settings):
    write_images(settings)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["choose_slice_locations_and_widths", "skip_gap_slices"]
    settings["samples"] = [
        {"name": "missing", "input": {"filenames": ["absent.tif"]}}, {"name": "good"},
    ]
    result = run_batch(settings)
    assert [item["status"] for item in result] == ["needs_review", "complete"]


def test_generated_script_runs_from_another_directory(settings, tmp_path):
    write_images(settings)
    script = generate(tmp_path / "generated", settings)
    command = subprocess.run([sys.executable, str(script), "--stage", "calibration"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert command.returncode == 0, command.stdout + command.stderr
    assert (script.parent / "calibration.json").is_file()


def test_calibration_file_and_stale_exports_preserved(settings, tmp_path):
    write_images(settings)
    source = tmp_path / "calibration_source.json"
    write_json(source, {"sdd_cm": 300, "center_px": [40, 60]})
    settings["calibration"].update(source_file=str(source), sdd_cm=None, center_px=None)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["choose_slice_locations_and_widths", "skip_gap_slices"]
    run = ReductionRun(settings, tmp_path / "run")
    assert run.run()["status"] == "complete"
    original = (run.directory / "csv" / "sample_all.csv").read_bytes()
    settings["calibration"]["sdd_cm"] = 301
    settings["slices"]["centers_qsx_Ainv"] = [99]
    settings["slices"]["widths_Ainv"] = 0.001
    failed = ReductionRun(settings, tmp_path / "run")
    assert failed.run()["status"] == "needs_review"
    assert not (run.directory / "csv").exists()
    archived = list((run.directory / "archives").glob("*/csv/sample_all.csv"))
    assert archived and archived[0].read_bytes() == original


def test_row_specific_resolved_settings_reproduce(settings, tmp_path):
    write_images(settings, extra_rows=True)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["choose_slice_locations_and_widths", "skip_gap_slices"]
    settings["additional_qy_rows"].update(enabled=True, offsets_qdy_px=[-15, 15])
    settings["integration"]["search_height_px"] = 12
    run = ReductionRun(settings, tmp_path / "run")
    assert run.run()["status"] == "complete"
    resolved = load_settings(run.directory / "resolved_settings.json")
    assert resolved["additional_qy_rows"]["row_settings"]["qy_001"]["slices"]["centers_qsx_Ainv"]
    original = (run.directory / "csv" / "sample_qyall_all.csv").read_bytes()
    independent = ReductionRun(resolved, tmp_path / "independent")
    assert independent.run()["status"] == "complete", independent.state["events"]
    assert (independent.directory / "csv" / "sample_qyall_all.csv").read_bytes() == original


def test_actual_hdf5_loader_preserves_duplicate_angles(settings):
    import h5py
    source = Path(__file__).resolve().parents[2] / "data" / "test_loaders" / "mock_smi_h5_multiple_scans.h5"
    target = Path(settings["input"]["data_directory"]) / "scan.h5"
    target.write_bytes(source.read_bytes())
    with h5py.File(target, "r+") as handle:
        handle["primary"]["stage_phi"][:] = 0
    settings["input"].update(filenames=["scan.h5"], filetype="smi-h5", metadata_pattern=None)
    dataset = load_dataset(settings)
    assert len(dataset.datas) == 61
    assert len(set(dataset.datas)) == 61
    assert {data.sample_phi_deg for data in dataset.datas.values()} == {0}
    assert all("bpm" in data.user_params for data in dataset.datas.values())


def test_actual_tiff_fixture_normalization_and_export(settings, tmp_path):
    source = Path(__file__).resolve().parents[2] / "data" / "test_load_dataset"
    settings["input"].update(
        data_directory=str(source), filenames=None,
        metadata_csv=str(source / "metadata_load.csv"),
        metadata_pattern=None, metadata={"exposure_time_s": 2}, user_params={},
    )
    settings["calibration"].update(center_px=[738, 492], sdd_cm=504.982)
    settings["input"]["metadata"]["pixel_size_um"] = 172
    settings["corrections"].update(normalize_by=["exposure_time_s"], footprint=False)
    settings["integration"].update(
        height_px=5, width_px=884, search_height_px=5, subtract_background=False,
    )
    settings["slices"].update(centers_qsx_Ainv=[-0.01275, 0.01249], widths_Ainv=0.001)
    settings["quality"].update(max_alignment_residual_px=5, min_peak_inside_fraction=0.8)
    run = ReductionRun(settings, tmp_path / "fixture_run")
    assert run.run()["status"] == "complete", run.state["events"]
    assert (run.directory / "csv" / "sample_all.csv").is_file()


def test_native_phi_estimation_and_restart(settings, tmp_path, monkeypatch):
    from cdsaxs.calculators import energy_to_wavelength
    from cdsaxs.harness import runner
    settings["calibration"].update(center_px=[100, 300], sdd_cm=300)
    settings["offsets"]["sample_phi_offset_deg"] = None
    settings["phi"].update(pitch_nm=35, peak_orders=None)
    settings["phi"]["search"].update(width_qdy_px=20, width_qdx_px=550)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["accept_angle_corrections"]
    yy, xx = np.mgrid[:201, :601]
    files = []
    wavelength = energy_to_wavelength(16100)
    for angle in np.linspace(-5, 5, 11):
        image = np.full((201, 601), 30.0)
        for order in (-4, -3, -2, -1, 1, 2, 3, 4):
            theta = 2 * np.arcsin(abs(order) * wavelength / (2 * 35))
            radius = 300 * np.tan(theta) * 10000 / 172
            position = 300 + np.sign(order) * radius / np.cos(np.deg2rad(angle + 0.7))
            image += 3000 * np.exp(-((yy - 100) ** 2 / 2 + (xx - position) ** 2 / 3))
        filename = f"sample_{angle}.tif"
        tifffile.imwrite(tmp_path / filename, image)
        files.append(filename)
    settings["input"]["filenames"] = files
    applied = []
    original_apply = runner.apply_intensity_corrections

    def capture(dataset, cfg):
        applied.append([data.sample_phi_deg for data in dataset.datas.values()])
        original_apply(dataset, cfg)

    monkeypatch.setattr(runner, "apply_intensity_corrections", capture)
    run = ReductionRun(settings, tmp_path / "native_phi")
    assert run.run("chi")["status"] == "stage_complete", run.state["events"]
    estimate = run.resolved["offsets"]["sample_phi_offset_deg"]
    assert estimate == pytest.approx(0.7, abs=0.025)
    assert len(applied) == 2
    np.testing.assert_allclose(np.array(applied[1]) - np.array(applied[0]), estimate)
    assert run.resolved["calibration"]["sdd_cm"] == 300


def test_native_chi_omega_correction_signs(settings, tmp_path):
    settings["calibration"].update(center_px=[100, 300], sdd_cm=300)
    settings["offsets"].update(sample_chi_offset_deg=None, sample_omega_offset_deg=None)
    settings["chi"]["search"].update(width_qdy_px=30, width_qdx_px=550)
    settings["omega"]["search"].update(width_qdy_px=30, width_qdx_px=550)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["accept_angle_corrections"]
    chi, omega = np.deg2rad([0.8, 0.5])
    yy, xx = np.mgrid[:201, :601]
    files = []
    for angle in (-40, -30, 0, 30, 40):
        phi = np.deg2rad(angle)
        slope = np.tan(chi) * np.cos(phi) - np.sin(phi) * np.tan(omega) / np.cos(chi)
        image = np.full((201, 601), 30.0)
        for column in (50, 100, 180, 420, 500, 550):
            row = 100 + slope * (column - 300)
            image += 3000 * np.exp(-((yy - row) ** 2 / 2 + (xx - column) ** 2 / 3))
        filename = f"sample_{angle}.tif"
        tifffile.imwrite(tmp_path / filename, image)
        files.append(filename)
    settings["input"]["filenames"] = files
    run = ReductionRun(settings, tmp_path / "native_angles")
    assert run.run("omega")["status"] == "stage_complete", run.state["events"]
    assert run.resolved["offsets"]["sample_chi_offset_deg"] == pytest.approx(0.8, abs=0.15)
    assert run.resolved["offsets"]["sample_omega_offset_deg"] == pytest.approx(0.5, abs=0.2)


def test_native_reference_calibration_from_unknown_geometry(settings, tmp_path):
    from cdsaxs.calculators import energy_to_wavelength
    write_images(settings)
    yy, xx = np.mgrid[:81, :121]
    image = np.full((81, 121), 30.0)
    pitch = 100
    wavelength = energy_to_wavelength(16100)
    for order in (-3, -2, -1, 1, 2, 3):
        theta = 2 * np.arcsin(abs(order) * wavelength / (2 * pitch))
        radius = 300 * np.tan(theta) * 10000 / 172
        image += 3000 * np.exp(-((yy - 40) ** 2 / 2 + (xx - 60 - np.sign(order) * radius) ** 2 / 2))
    tifffile.imwrite(tmp_path / "srm_reference.tif", image)
    settings["calibration"].update(
        center_px=None, sdd_cm=None, sdd_guess_cm=290,
        beam_center_guess_px=[40, 60], reference_pitch_nm=pitch,
    )
    settings["calibration"]["search"].update(width_qdy_px=12, width_qdx_px=100, min_distance=8)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = ["accept_calibration"]
    run = ReductionRun(settings, tmp_path / "calibration_run")
    assert run.run("calibration")["status"] == "stage_complete", run.state["events"]
    assert run.resolved["calibration"]["sdd_cm"] == pytest.approx(300, abs=0.1)
    np.testing.assert_allclose(run.resolved["calibration"]["center_px"], [40, 60], atol=0.05)


def test_filename_monitor_strings_and_csv_selection(settings, tmp_path):
    base = write_images(settings, angles=(0,))
    settings["input"]["user_params"]["bpm"] = "4.0"
    dataset = load_dataset(settings, settings["calibration"], settings["offsets"])
    apply_intensity_corrections(dataset, settings)
    np.testing.assert_allclose(next(iter(dataset.datas.values())).image, base / 8)
    source = Path(__file__).resolve().parents[2] / "data" / "test_load_dataset"
    files = sorted(source.glob("*.tif"))
    settings["input"].update(
        data_directory=str(source), filenames=[files[0].name],
        metadata_csv=str(source / "metadata_load.csv"),
        metadata_pattern=None, metadata={"exposure_time_s": 2}, user_params={},
    )
    selected = load_dataset(settings)
    assert len(selected.datas) == 1
    assert next(iter(selected.datas.values())).metadata["filename"] == files[0].name


def test_automatic_boxes_backgrounds_and_detected_rows(settings, tmp_path):
    write_images(settings, extra_rows=True)
    settings["mode"] = "automatic"
    settings["allowed_decisions"] = [
        "choose_integration_box", "choose_backgrounds",
        "choose_slice_locations_and_widths", "choose_qy_offsets", "skip_gap_slices",
    ]
    settings["integration"].update(height_px=None, width_px=None, background_offsets_px=None)
    settings["additional_qy_rows"]["enabled"] = True
    run = ReductionRun(settings, tmp_path / "automated")
    assert run.run()["status"] == "complete", run.state["events"]
    rows = json.loads((run.directory / "row_reductions.json").read_text())
    assert len(rows) == 3
    assert all(3 <= row["integration"]["height_px"] <= 40 for row in rows)
    assert all(row["integration"]["background_offsets_px"] for row in rows)


def test_nonpositive_reduction_is_held(settings, tmp_path):
    write_images(settings)
    for filename in settings["input"]["filenames"]:
        path = tmp_path / filename
        image = tifffile.imread(path)
        image[27:34] = 5000
        image[47:54] = 5000
        tifffile.imwrite(path, image)
    settings["mode"] = "automatic"
    settings["integration"]["search_height_px"] = 12
    settings["allowed_decisions"] = ["choose_slice_locations_and_widths", "skip_gap_slices"]
    run = ReductionRun(settings, tmp_path / "nonpositive")
    assert run.run()["status"] == "needs_review"
    assert any("No positive intensity" in event for event in run.state["events"])
    assert not (run.directory / "csv").exists()


def test_generated_script_can_choose_consolidated_root(settings, tmp_path):
    write_images(settings)
    script = generate(tmp_path / "generated", settings)
    root = tmp_path / "consolidated"
    command = subprocess.run([sys.executable, str(script), "--stage", "calibration",
                              "--output-root", str(root)], cwd=tmp_path,
                             capture_output=True, text=True)
    assert command.returncode == 0, command.stdout + command.stderr
    assert list(root.glob("experiment/sample/*/calibration.json"))
    assert not (script.parent / "calibration.json").exists()
