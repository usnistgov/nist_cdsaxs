"""JSON settings, validation, and paths for the reduction harness."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


SEARCH = {
    "width_qdy_px": 40, "width_qdx_px": 400,
    "log_scale": True, "refinement_size": 7,
    "refinement_method": "gaussian",
    "min_distance": 10, "threshold_rel": 0.1, "threshold_abs": None,
}

DEFAULTS = {
    "schema_version": 1,
    "experiment": "experiment", "sample_name": "sample",
    "sample_kind": None, "samples": [], "sample_profile": None,
    "mode": "review", "allowed_decisions": [],
    "refit_known_constants": False, "decision_notes": [],
    "input": {
        "data_directory": None, "filenames": None,
        "filter_substrings": None, "file_extension": None,
        "filetype": None, "detector_type": None,
        "metadata_csv": None, "metadata_pattern": None,
        "metadata_scales": {}, "metadata": {}, "user_params": {},
    },
    "calibration": {
        "source_file": None, "reference_sample": None,
        "reference_filenames": None, "reference_pitch_nm": None,
        "beam_center_guess_px": None, "center_px": None, "sdd_cm": None,
        "sdd_guess_cm": None, "reference_metadata_pattern": None,
        "peak_orders": None, "search": deepcopy(SEARCH),
    },
    "offsets": {
        "sample_phi_offset_deg": None, "sample_chi_offset_deg": None,
        "sample_omega_offset_deg": None,
    },
    "corrections": {
        "normalize_by": ["bpm", "exposure_time_s"],
        "substrate_absorption": True, "footprint": True,
        "substrate_thickness_um": None,
        "substrate_attenuation_coeff_um-1": None,
    },
    "phi": {
        "pitch_nm": None, "angles_deg": None,
        "angle_range_deg": [-10.0, 10.0], "peak_orders": None,
        "search": deepcopy(SEARCH),
    },
    "chi": {"search": deepcopy(SEARCH)},
    "omega": {
        "angles_deg": None, "min_abs_phi_deg": 20.0,
        "max_frames": 8, "search": deepcopy(SEARCH),
    },
    "integration": {
        "height_px": None, "width_px": None,
        "background_offsets_px": None, "subtract_background": True,
        "background_strategy": "auto", "mode": "sum",
        "search_height_px": 40, "height_bounds_px": [3, 40],
        "peak_padding_px": 3,
    },
    "slices": {
        "centers_qsx_Ainv": None, "widths_Ainv": None,
        "mode": "sum",
        "width_bounds_Ainv": [0.0002, 0.003],
        "prominence_relative": 0.1, "min_distance_px": 5,
        "exclude_center_px": 8,
    },
    "additional_qy_rows": {
        "enabled": False, "offsets_qsy_Ainv": None,
        "offsets_qdy_px": None, "max_rows": 12,
        "min_separation_px": 10, "prominence_relative": 0.1,
        "row_settings": {},
    },
    "quality": {
        "min_valid_phi_points": 5,
        "max_sdd_relative_scatter": 0.02,
        "max_phi_fit_relative_rmse": 0.03,
        "max_phi_offset_abs_deg": 5.0,
        "max_phi_fit_uncertainty_deg": 1.0,
        "max_chi_near_zero_deg": 2.0, "max_chi_abs_deg": 5.0,
        "min_omega_frames": 2, "require_omega_both_signs": True,
        "max_omega_abs_deg": 5.0, "max_omega_scatter_deg": 1.0,
        "max_alignment_residual_px": 2.0,
        "max_background_peak_fraction": 0.3,
        "min_roi_valid_fraction": 0.2,
        "min_slice_angle_coverage": 0.6,
        "min_peak_inside_fraction": 0.95,
    },
    "export": {
        "per_row": True, "combine_qy_rows": True,
        "split_phi": False, "phi_basis": "corrected",
        "zero_phi": "positive", "zero_tolerance_deg": 1e-9,
        "export_qr": True, "decimals": 7,
    },
    "output": {
        "root": None, "figure_formats": ["png"], "dpi": 300,
    },
}

DECISIONS = {
    "accept_calibration", "accept_angle_corrections",
    "choose_integration_box", "choose_backgrounds",
    "choose_slice_locations_and_widths", "skip_gap_slices",
    "choose_qy_offsets",
}
STAGES = ("calibration", "phi", "chi", "omega", "qy_rows",
          "integration", "slices")


def merge(base, overrides):
    """Recursively merge settings; explicit null requests estimation."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}")

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path, value):
    """Atomically replace a small JSON artifact, rejecting NaN/Inf."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "sample"


def _paths(settings, directory):
    settings = deepcopy(settings)
    for section, keys in (("input", ("data_directory", "metadata_csv")),
                          ("calibration", ("source_file",)),
                          ("output", ("root",))):
        for key in keys:
            value = settings.get(section, {}).get(key)
            if value is not None:
                path = Path(value).expanduser()
                settings[section][key] = str((directory / path).resolve())
    for sample in settings.get("samples", []):
        if isinstance(sample, dict):
            converted = _paths(sample, directory)
            sample.clear()
            sample.update(converted)
    return settings


def load_settings(path):
    path = Path(path).resolve()
    supplied = read_json(path)
    profile = supplied.get("sample_profile")
    inherited = {}
    if profile:
        profile_path = (path.parent / profile).resolve()
        inherited = _paths(read_json(profile_path), profile_path.parent)
        if "samples" in inherited or "sample_profile" in inherited:
            raise ValueError("A sample profile cannot contain samples or profiles")
        supplied["sample_profile"] = str(profile_path)
    settings = merge(DEFAULTS, merge(inherited, _paths(supplied, path.parent)))
    validate(settings)
    return settings


def sample_settings(settings):
    """Expand batch overrides; names imply filename filters unless supplied."""
    samples = settings["samples"] or [{"name": settings["sample_name"]}]
    names = set()
    for sample in samples:
        if isinstance(sample, str):
            sample = {"name": sample}
        if not isinstance(sample, dict) or not sample.get("name"):
            raise ValueError("Each samples entry needs a name")
        name = sample["name"]
        if slug(name) in names:
            raise ValueError("Sample names must have distinct filesystem names")
        names.add(slug(name))
        effective = merge(settings, {k: v for k, v in sample.items()
                                     if k != "name"})
        effective["sample_name"] = name
        effective["samples"] = []
        if settings["samples"] and not any(
            effective["input"][k] is not None
            for k in ("filenames", "metadata_csv", "filter_substrings")
        ):
            effective["input"]["filter_substrings"] = [name]
        validate(effective)
        yield effective


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))


def _known_keys(value, template, prefix=""):
    for key, item in value.items():
        if key not in template:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        if isinstance(template[key], dict) and not isinstance(item, dict):
            raise ValueError(f"{prefix}{key} must be a JSON object")
        # Search arguments and loader metadata are deliberately extensible.
        if (isinstance(item, dict) and isinstance(template[key], dict)
                and key not in {"search", "metadata", "metadata_scales", "user_params", "row_settings"}):
            _known_keys(item, template[key], f"{prefix}{key}.")


def validate(settings):
    _known_keys(settings, DEFAULTS)
    if settings["schema_version"] != 1:
        raise ValueError("Unsupported harness schema_version")
    if settings["mode"] not in {"review", "automatic"}:
        raise ValueError("mode must be review or automatic")
    if not isinstance(settings["allowed_decisions"], list):
        raise ValueError("allowed_decisions must be a list")
    if set(settings["allowed_decisions"]) - DECISIONS:
        raise ValueError("Unrecognized allowed_decisions")
    if not isinstance(settings["decision_notes"], list) or not all(
        isinstance(note, str) for note in settings["decision_notes"]
    ):
        raise ValueError("decision_notes must be a list of strings")
    if not isinstance(settings["samples"], list):
        raise ValueError("samples must be a list")
    for section, keys in (
        (None, ("refit_known_constants",)),
        ("corrections", ("footprint", "substrate_absorption")),
        ("integration", ("subtract_background",)),
        ("additional_qy_rows", ("enabled",)),
        ("export", ("per_row", "combine_qy_rows", "split_phi", "export_qr")),
    ):
        for key in keys:
            value = settings[key] if section is None else settings[section][key]
            if not isinstance(value, bool):
                raise ValueError(f"{section or ''}.{key} must be boolean")
    for key in ("filenames",):
        value = settings["input"][key]
        if value is not None and (not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        )):
            raise ValueError(f"input.{key} must be a list of filenames or null")
    normalize = settings["corrections"]["normalize_by"]
    if (not isinstance(normalize, list) or not all(isinstance(key, str) for key in normalize)
            or len(set(normalize)) != len(normalize)):
        raise ValueError("normalize_by must be a list of distinct metadata keys")
    if settings["sample_kind"] not in {
        None, "contact_holes", "circular_contact_holes", "line_grating"
    }:
        raise ValueError("Unknown sample_kind")
    for section in ("calibration", "corrections", "offsets"):
        for key, value in settings[section].items():
            if value is not None and (key.endswith("_deg") or key in {
                "sdd_cm", "sdd_guess_cm", "reference_pitch_nm", "substrate_thickness_um",
                "substrate_attenuation_coeff_um-1",
            }):
                _number(value, f"{section}.{key}", not key.endswith("_deg"))
    for key in ("center_px", "beam_center_guess_px"):
        value = settings["calibration"][key]
        if value is not None:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"calibration.{key} must be [row, column]")
            for component in value:
                _number(component, key)
                if component < 0:
                    raise ValueError("Beam center pixels must be nonnegative")
    rows = settings["additional_qy_rows"]
    for row_id, overrides in rows["row_settings"].items():
        if not isinstance(row_id, str) or not isinstance(overrides, dict):
            raise ValueError("row_settings must map row IDs to settings objects")
        _known_keys(overrides, {"integration": DEFAULTS["integration"], "slices": DEFAULTS["slices"]})
        child = merge(settings, overrides)
        child["additional_qy_rows"]["row_settings"] = {}
        validate(child)
    if rows["enabled"] and settings["sample_kind"] != "circular_contact_holes":
        raise ValueError("Additional qy rows require sample_kind=circular_contact_holes")
    if rows["offsets_qsy_Ainv"] is not None and rows["offsets_qdy_px"] is not None:
        raise ValueError("Supply qsy offsets or detector-pixel offsets, not both")
    for key in ("offsets_qsy_Ainv", "offsets_qdy_px"):
        if rows[key] is not None:
            if not isinstance(rows[key], list):
                raise ValueError(f"{key} must be a list or null")
            for value in rows[key]:
                _number(value, key)
                if key.endswith("_px") and int(value) != value:
                    raise ValueError("Detector-pixel offsets must be integers")
    for key in ("height_px", "width_px"):
        value = settings["integration"][key]
        if value is not None:
            _number(value, key, True)
            if int(value) != value:
                raise ValueError(f"{key} must be an integer")
    backgrounds = settings["integration"]["background_offsets_px"]
    if backgrounds is not None and (not isinstance(backgrounds, list) or not backgrounds):
        raise ValueError("background_offsets_px must be a nonempty list or null")
    if backgrounds:
        for value in backgrounds:
            _number(value, "background_offsets_px")
            if int(value) != value:
                raise ValueError("Background offsets must be integers")
    for section in ("integration", "slices"):
        if settings[section]["mode"] not in {"sum", "mean"}:
            raise ValueError(f"{section}.mode must be sum or mean")
    if settings["integration"]["background_strategy"] not in {"auto", "adjacent", "distant"}:
        raise ValueError("background_strategy must be auto, adjacent, or distant")
    for section, key in (("integration", "height_bounds_px"),
                         ("slices", "width_bounds_Ainv"), ("phi", "angle_range_deg")):
        bounds = settings[section][key]
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"{section}.{key} must be an increasing pair")
        for value in bounds:
            _number(value, key, section != "phi")
    for section, key, minimum in (
        ("omega", "max_frames", 2), ("integration", "search_height_px", 3),
        ("additional_qy_rows", "max_rows", 1),
        ("additional_qy_rows", "min_separation_px", 1),
        ("slices", "min_distance_px", 1), ("slices", "exclude_center_px", 0),
        ("integration", "peak_padding_px", 0), ("output", "dpi", 1),
        ("quality", "min_valid_phi_points", 4),
        ("quality", "min_omega_frames", 2), ("export", "decimals", 0),
    ):
        value = settings[section][key]
        _number(value, f"{section}.{key}")
        if int(value) != value or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
        settings[section][key] = int(value)
    for section in ("calibration", "phi"):
        orders = settings[section]["peak_orders"]
        if orders is not None:
            if not isinstance(orders, list) or not orders:
                raise ValueError(f"{section}.peak_orders must be a nonempty list or null")
            for order in orders:
                _number(order, "peak_orders", True)
                if int(order) != order:
                    raise ValueError("Peak orders must be positive integer magnitudes")
    for section in ("phi", "omega"):
        angles = settings[section]["angles_deg"]
        if angles is not None:
            if not isinstance(angles, list) or not angles or len(set(angles)) != len(angles):
                raise ValueError(f"{section}.angles_deg must be a nonempty distinct list")
            for angle in angles:
                _number(angle, "angles_deg")
    if settings["phi"]["pitch_nm"] is not None:
        _number(settings["phi"]["pitch_nm"], "phi.pitch_nm", True)
    for section in ("slices", "additional_qy_rows"):
        _number(settings[section]["prominence_relative"], "prominence_relative", True)
        if settings[section]["prominence_relative"] > 1:
            raise ValueError("prominence_relative must be <= 1")
    centers = settings["slices"]["centers_qsx_Ainv"]
    widths = settings["slices"]["widths_Ainv"]
    if centers is not None:
        if not isinstance(centers, list) or not centers:
            raise ValueError("centers_qsx_Ainv must be a nonempty list or null")
        for value in centers:
            _number(value, "centers_qsx_Ainv")
        if len(set(centers)) != len(centers):
            raise ValueError("Slice centers must be distinct")
    if widths is not None:
        values = widths if isinstance(widths, list) else [widths]
        if centers is not None and isinstance(widths, list) and len(widths) != len(centers):
            raise ValueError("Slice widths must match slice centers")
        for value in values:
            _number(value, "widths_Ainv", True)
    for key, value in settings["quality"].items():
        if key == "require_omega_both_signs":
            if not isinstance(value, bool):
                raise ValueError(f"quality.{key} must be boolean")
        else:
            _number(value, f"quality.{key}", True)
            if ("fraction" in key or "coverage" in key) and value > 1:
                raise ValueError(f"quality.{key} must be <= 1")
    exp = settings["export"]
    if exp["phi_basis"] not in {"nominal", "corrected"}:
        raise ValueError("export.phi_basis must be nominal or corrected")
    if exp["zero_phi"] not in {"positive", "negative", "separate", "exclude"}:
        raise ValueError("Unknown export.zero_phi policy")
    _number(exp["zero_tolerance_deg"], "zero_tolerance_deg")
    if exp["zero_tolerance_deg"] < 0:
        raise ValueError("zero_tolerance_deg must be nonnegative")
    if not settings["output"]["figure_formats"] or set(
        settings["output"]["figure_formats"]
    ) - {"png", "pdf", "svg"}:
        raise ValueError("figure_formats must select png, pdf, or svg")
