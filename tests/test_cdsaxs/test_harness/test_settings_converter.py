"""Conversion of explicit standalone rows into shared qy0 constants."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def converter():
    path = Path(__file__).resolve().parents[3] / "harness/convert_settings_qy0.py"
    spec = importlib.util.spec_from_file_location("convert_settings_qy0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def explicit_settings():
    return {
        "sample_name": "sample",
        "input": {"data_directory": "../images"},
        "integration": {"height_px": 6, "width_px": 1400, "mode": "sum",
                        "subtract_background": True, "background_offsets_px": [-9, 9]},
        "rows": [
            {"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": None,
             "slices": {"centers_qsx_Ainv": [-0.026, 0.026], "widths_Ainv": [0.003, 0.003],
                        "mode": "sum"},
             "background_choices_by_slice_frame_px": [{"frame0": [-9, 9]}, {"frame0": None}]},
            {"row_id": "offset", "target_qsy_Ainv": None, "offset_qdy_px": 15,
             "slices": {"centers_qsx_Ainv": [0.013], "widths_Ainv": [0.001], "mode": "sum"},
             "background_choices_by_slice_frame_px": [{"frame0": [-9]}]},
        ],
        "output": {"root": "."},
    }


def test_all_rows_use_qy0_constants_and_conversion_is_idempotent(converter, explicit_settings):
    original = deepcopy(explicit_settings)
    converted = converter.convert_settings(explicit_settings)
    assert converted["slices"] == {"centers_qsx_Ainv": [-0.026, 0.026],
                                   "widths_Ainv": 0.003, "mode": "sum"}
    assert converted["integration"]["background_choices"] == [-9, 9]
    assert "background_offsets_px" not in converted["integration"]
    assert all("slices" not in row and "background_choices_by_slice_frame_px" not in row
               for row in converted["rows"])
    assert converted["rows"][1]["offset_qdy_px"] == 15
    assert converted["input"] == original["input"]
    assert explicit_settings == original
    assert converter.convert_settings(converted) == converted


@pytest.mark.parametrize("widths", [0.003, [0.001, 0.002]])
def test_supplied_qy0_scalar_and_unequal_widths_are_preserved(converter, explicit_settings, widths):
    explicit_settings["rows"][0]["slices"]["widths_Ainv"] = widths
    explicit_settings["integration"]["background_offsets_px"] = [-9]
    converted = converter.convert_settings(explicit_settings)
    assert converted["slices"]["widths_Ainv"] == widths
    assert converted["integration"]["background_choices"] == -9


def test_compact_qy0_override_takes_precedence(converter, explicit_settings):
    compact = converter.convert_settings(explicit_settings)
    compact["rows"][0]["slices"] = {"widths_Ainv": 0.002}
    compact["rows"][1]["slices"] = {"centers_qsx_Ainv": [0.013]}
    compact["integration"]["background_choices"] = {
        "default": [-9, 9], "by_frame_px": {"frame0": -9}}
    converted = converter.convert_settings(compact)
    assert converted["slices"]["widths_Ainv"] == 0.002
    assert converted["slices"]["centers_qsx_Ainv"] == [-0.026, 0.026]
    assert converted["integration"]["background_choices"] == [-9, 9]
    assert all("slices" not in row for row in converted["rows"])


@pytest.mark.parametrize("in_place", [False, True])
def test_cli_writes_compatible_script_and_preserves_originals(
        converter, explicit_settings, tmp_path, in_place):
    source = tmp_path / "settings.json"
    original = json.dumps(explicit_settings)
    source.write_text(original)
    old_script = tmp_path / "reduce_sample.py"
    old_script.write_text("# existing custom reduction\n")
    args = [str(source), "--update-script"] + (["--in-place"] if in_place else [])
    assert converter.main(args) == 0
    destination = source if in_place else tmp_path / "settings.qy0.json"
    script = old_script if in_place else tmp_path / "reduce_sample_qy0.py"
    text = script.read_text()
    compile(text, str(script), "exec")
    assert "cdsaxs.harness" not in text
    assert f"with_name({destination.name!r})" in text
    assert json.loads(destination.read_text())["slices"]["widths_Ainv"] == 0.003
    if in_place:
        assert next(tmp_path.glob("settings.json.*.bak")).read_text() == original
        assert next(tmp_path.glob("reduce_sample.py.*.bak")).read_text() == "# existing custom reduction\n"
    else:
        assert source.read_text() == original
        assert old_script.read_text() == "# existing custom reduction\n"


def test_invalid_conversion_does_not_replace_originals(converter, explicit_settings, tmp_path):
    explicit_settings["rows"][0]["slices"]["widths_Ainv"] = [0.003]
    source = tmp_path / "setting.json"
    original = json.dumps(explicit_settings)
    source.write_text(original)
    assert converter.main([str(source), "--in-place", "--update-script"]) == 1
    assert source.read_text() == original
    assert not (tmp_path / "reduce_sample.py").exists()
    assert not list(tmp_path.glob("*.bak"))
