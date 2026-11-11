"""Generate a runnable Python entry point with external JSON settings."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import shutil

from .config import fingerprint, merge, write_json


SCRIPT = '''#!/usr/bin/env python3
"""Generated CD-SAXS reduction script.

Edit settings.json to tune constants, integration, slices, and exports.
Examples:
    python reduce_sample.py --stage phi
    python reduce_sample.py --accept phi --stage chi
    python reduce_sample.py --stage reduce
    python reduce_sample.py --output-profile full

Requires the nist_cdsaxs package (pip install -e /path/to/nist_cdsaxs).
Compact runs save effective settings, figures, CSVs, and summary.md.
"""
from pathlib import Path
from cdsaxs.harness.__main__ import main

SETTINGS_FILE = Path(__file__).resolve().with_name("settings.json")


def reduce(argv=None):
    """Run staged calibration, corrections, integration, slicing, and export.

    Every rebuild reloads raw detector data and reapplies intensity corrections.
    Stage functions are implemented in cdsaxs.harness.runner and can also be
    imported for custom workflows. All scientific parameters live in JSON.
    """
    return main(argv, default_config=SETTINGS_FILE,
                default_resume=Path(__file__).resolve().parent)


if __name__ == "__main__":
    raise SystemExit(reduce())
'''


def generate(directory, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings_path = directory / "settings.json"
    script_path = directory / "reduce_sample.py"
    if not settings_path.exists():
        write_json(settings_path, settings)
    if not script_path.exists():
        script_path.write_text(SCRIPT, encoding="utf-8")
    return script_path


STANDALONE_SCRIPT = '''#!/usr/bin/env python3
"""Fixed CD-SAXS reduction using native nist_cdsaxs functions.

Edit settings.json and copy both files to reuse the accepted reduction.
Run: python reduce_sample.py
Relative paths use the settings directory. Requires installed nist_cdsaxs.
No harness fitting, peak searches, background checks, or slice review is run.
Native masks and CSV filtering of nonpositive intensities still apply.
"""
import json
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from cdsaxs.loaders import load_data as loaders
from cdsaxs.data.dataset import Dataset, ReducedDataset, ReducedSlices
from cdsaxs.reduction import slice_reduced_dataset
from cdsaxs.plotting import plotting
from cdsaxs.tools import default_mask

SETTINGS_FILE = Path(__file__).resolve().with_name("settings.json")


def reduce():
    with SETTINGS_FILE.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    legacy = ("background_offsets_by_frame_px", "background_choices_by_slice_frame_px")
    if (any(key in settings["integration"] for key in legacy)
            or any(key in row for row in settings["rows"] for key in (*legacy, "background_choices"))):
        raise TypeError("Convert background overrides to one global int or list of ints before reduction")
    base = SETTINGS_FILE.parent
    cfg = settings["input"]
    data_directory = (base / Path(cfg["data_directory"]).expanduser()).resolve()
    output = (base / Path(settings["output"]["root"]).expanduser()).resolve()
    csv_directory = output / "csv"
    figure_directory = output / "figures"
    csv_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})

    def save_figure(figure, stem):
        if isinstance(figure, (list, tuple)):
            for index, item in enumerate(figure):
                save_figure(item, f"{stem}_{index + 1}")
            return
        for extension in settings["output"]["figure_formats"]:
            figure.savefig(figure_directory / f"{stem}.{extension}",
                           dpi=settings["output"]["dpi"], bbox_inches="tight")
        plt.close(figure)

    # Load fresh detector images and apply the accepted metadata and offsets.
    metadata = dict(cfg["metadata"])
    metadata.update(settings["calibration"])
    metadata.update(settings["offsets"])
    files = [(data_directory / filename).resolve() for filename in cfg["filenames"]]
    if cfg["metadata_csv"]:
        loaded = loaders.LoadDataset_MetadataCSV(
            settings["sample_name"], (base / cfg["metadata_csv"]).resolve(),
            filetype=cfg["filetype"], detector_type=cfg["detector_type"], verbose=False)
        requested = set(files)
        frames = [data for data in loaded.datas.values() if (
            Path(data.metadata["data_directory"]) / data.metadata["filename"]
        ).resolve() in requested]
    else:
        frames = []
        for filepath in files:
            if cfg["filetype"] == "smi-h5":
                loaded = loaders.LoadData(str(filepath), metadata=metadata,
                                          user_params=cfg["user_params"], filetype=cfg["filetype"])
                frames.extend(loaded if isinstance(loaded, list) else [loaded])
            else:
                loaded = loaders.LoadDataset(
                    settings["sample_name"], str(filepath.parent), filenames=[filepath.name],
                    metadata=metadata, user_params=cfg["user_params"],
                    metadata_pattern=cfg["metadata_pattern"], metadata_scales=cfg["metadata_scales"],
                    filetype=cfg["filetype"], detector_type=cfg["detector_type"], verbose=False)
                frames.extend(loaded.datas.values())
    dataset = Dataset(name=settings["sample_name"])
    for index, data in enumerate(frames):
        data.update_metadata(metadata, overwrite=True)
        data.update_user_params(cfg["user_params"])
        if any(abs(data.metadata["sample_phi_deg"] - angle) < 0.05
               for angle in cfg["exclude_nominal_phi_deg"]):
            continue
        data.name = f"{data.metadata.get('filename', 'frame')}:{index:06d}"
        data.update_metadata({"name": data.name}, overwrite=True)
        data.mask |= ~np.isfinite(data.image)
        dataset.add_data(data)
    # Explicit frame exclusions are fixed settings, not data-quality decisions.
    raw = next(iter(dataset.datas.values()))
    save_figure(raw.plot_data(log_scale=False), "00_raw_data")

    # Normalize once, using the final accepted phi for intensity corrections.
    corrections = settings["corrections"]
    additions = {key: corrections[key] for key in (
        "substrate_thickness_um", "substrate_attenuation_coeff_um-1")
        if key in corrections}
    dataset.update_all_metadata(additions, verbose=False)
    dataset.normalize_all_data_by_metadata(corrections["normalize_by"])
    if corrections["substrate_absorption"]:
        dataset.apply_substrate_absorption_correction()
    if corrections["footprint"]:
        dataset.apply_footprint_correction()
    for data in dataset.datas.values():
        data.mask |= ~np.isfinite(data.image)
    dataset.apply_rotation_correction_all_data(verbose=False)
    for data in dataset.datas.values():
        data.mask |= ~np.isfinite(data.image)

    # Integrate every fixed row with the same box and backgrounds.
    geometry = settings["integration"]
    backgrounds = geometry.get("background_choices", geometry.get("background_offsets_px"))
    central = next(row for row in settings["rows"] if row["row_id"] == "qy0")
    shared_slices = dict(settings.get("slices", {}))
    shared_slices.update(central.get("slices", {}))
    reductions = []
    for configured_row in settings["rows"]:
        row = dict(configured_row)
        row["slices"] = dict(shared_slices)
        row["slices"].update(configured_row.get("slices", {}))
        widths = row["slices"]["widths_Ainv"]
        if not isinstance(widths, list):
            widths = [widths] * len(row["slices"]["centers_qsx_Ainv"])
        row["slices"]["widths_Ainv"] = widths
        integrated = ReducedDataset()
        for key, data in dataset.datas.items():
            beam_y, beam_x = data.metadata["center_px_detector"]
            if row["offset_qdy_px"] is not None:
                center_y = beam_y - row["offset_qdy_px"]
            elif "center_qdy_px" in row:
                centers = row["center_qdy_px"]
                center_y = centers[key] if isinstance(centers, dict) else centers
            elif row["target_qsy_Ainv"] == 0:
                center_y = beam_y
            else:
                raise ValueError(f"Row {row['row_id']} requires accepted center_qdy_px settings")
            yrange, xrange = data.get_box_dims_size(
                geometry["height_px"], geometry["width_px"], center=(center_y, beam_x))
            profile, _ = data.integrate_box(
                mode=geometry["mode"], axis=0, show_plot=False,
                range_qdy_px=yrange, range_qdx_px=xrange,
                subtract_background_offset=(backgrounds if geometry["subtract_background"] else None))
            profile.row_id = row["row_id"]
            for component in ("Iq", "qsx", "qsy", "qsz", "qsr", "qs"):
                profile.mask |= default_mask(getattr(profile, component))
            integrated.add_data(profile)
        # Native integration outlines show exactly the boxes being used.
        save_figure(plotting.plot_data2d_integrate_box(integrated.datas[0], log_scale=False),
                    f"05_{row['row_id']}_integration")
        bins = [(center - width / 2, center, center + width / 2)
                for center, width in zip(row["slices"]["centers_qsx_Ainv"],
                                         row["slices"]["widths_Ainv"])]
        save_figure(plotting.plot_slice_reduced_dataset(
            integrated, q_bins=bins, log_scale=False, interpolated_data=False,
            slice_color="#D55E00", slice_lw=0.8, s=2), f"07_{row['row_id']}_slices")
        reductions.append((row, integrated))

    # Fixed phi grouping and native slicing/export; no slice is quality-rejected.
    export = settings["export"]
    groups = {"all": set(dataset.datas)}
    if export["split_phi"]:
        groups.update({"posPhi": set(), "negPhi": set()})
        if export["zero_phi"] == "separate":
            groups["zeroPhi"] = set()
        for key, data in dataset.datas.items():
            angle = (data.sample_phi_deg if export["phi_basis"] == "corrected"
                     else data.metadata["sample_phi_deg"])
            tolerance = export["zero_tolerance_deg"]
            if angle > tolerance:
                groups["posPhi"].add(key)
            elif angle < -tolerance:
                groups["negPhi"].add(key)
            elif export["zero_phi"] != "exclude":
                group = {"positive": "posPhi", "negative": "negPhi", "separate": "zeroPhi"}
                groups[group[export["zero_phi"]]].add(key)
    sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", settings["sample_name"]).strip("._") or "sample"
    central_only = len(reductions) == 1 and reductions[0][0]["row_id"] == "qy0"
    for group, keys in groups.items():
        if not keys:
            continue
        combined = ReducedSlices()
        for row, integrated in reductions:
            subset = ReducedDataset([profile for profile in integrated.datas
                                     if profile.data2d.name in keys])
            slices, _ = slice_reduced_dataset(
                subset, q_values=row["slices"]["centers_qsx_Ainv"],
                q_widths=row["slices"]["widths_Ainv"], mode=row["slices"]["mode"], show_plot=False)
            combined.add_slices(slices)
            suffix = "" if central_only else f"_{row['row_id']}"
            stem = f"{sample}{suffix}_{group}"
            if export["per_row"]:
                slices.export_reduced_slices(csv_directory / f"{stem}.csv",
                                             export_qr=export["export_qr"], decimals=export["decimals"])
            figure, axis = plotting.plot_reduced_slices(slices, log_scale=False)
            axis.set_title(f"{row['row_id']} / {group}")
            save_figure(figure, f"08_{stem}_reduced")
        if export["combine_qy_rows"]:
            suffix = "" if central_only else "_qyall"
            combined.export_reduced_slices(csv_directory / f"{sample}{suffix}_{group}.csv",
                                          export_qr=export["export_qr"], decimals=export["decimals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(reduce())
'''


def compact_widths(widths):
    """Represent equal widths once while retaining unequal explicit widths."""
    if isinstance(widths, list) and widths and all(value == widths[0] for value in widths):
        return deepcopy(widths[0])
    return deepcopy(widths)


def compact_backgrounds(offsets):
    """Store one signed offset or one shared list of integer offsets."""
    if offsets is None or type(offsets) is int:
        return offsets
    if (not isinstance(offsets, list) or not offsets
            or any(type(offset) is not int for offset in offsets)):
        raise TypeError("Background choices must be an int, a nonempty list of ints, or None")
    return offsets[0] if len(offsets) == 1 else deepcopy(offsets)


def standalone_settings(settings, rows):
    """Project accepted settings into constants needed by native reduction."""
    cfg = settings["input"]
    required = [settings["calibration"][key] for key in ("center_px", "sdd_cm")]
    required += list(settings["offsets"].values())
    required += [settings["integration"][key] for key in ("height_px", "width_px")]
    if any(value is None for value in required):
        raise ValueError("Standalone export requires resolved calibration, offsets, and dimensions")
    if settings["integration"]["subtract_background"] and settings["integration"]["background_offsets_px"] is None:
        raise ValueError("Standalone export requires fixed background offsets")
    filenames = cfg["filenames"]
    if filenames is None:
        from .runner import selected_files
        import os
        filenames = [os.path.relpath(path, cfg["data_directory"])
                     for path in selected_files(settings)]
    filenames = sorted(set(filenames), key=lambda name: str(
        (Path(cfg["data_directory"]) / name).resolve()))
    input_settings = {key: deepcopy(cfg[key]) for key in (
        "data_directory", "metadata_csv", "filetype", "detector_type",
        "metadata", "user_params", "metadata_pattern", "metadata_scales",
        "exclude_nominal_phi_deg")}
    input_settings["filenames"] = deepcopy(filenames)
    if input_settings["filetype"] is None and filenames and all(
            Path(name).suffix.lower() in {".h5", ".hdf5"} for name in filenames):
        input_settings["filetype"] = "smi-h5"
    corrections = {key: deepcopy(settings["corrections"][key]) for key in (
        "normalize_by", "substrate_absorption", "footprint")}
    if corrections["substrate_absorption"]:
        for key in ("substrate_thickness_um", "substrate_attenuation_coeff_um-1"):
            value = settings["corrections"][key]
            if value is not None:
                corrections[key] = value
    result = {"sample_name": settings["sample_name"], "input": input_settings,
              "calibration": {key: deepcopy(settings["calibration"][key])
                              for key in ("center_px", "sdd_cm")},
              "offsets": deepcopy(settings["offsets"]), "corrections": corrections,
              "integration": {key: deepcopy(settings["integration"][key]) for key in (
                  "height_px", "width_px", "subtract_background", "mode")},
              "slices": {}, "rows": [], "export": deepcopy(settings["export"]),
              "output": {"root": ".", "figure_formats": deepcopy(settings["output"]["figure_formats"]),
                         "dpi": settings["output"]["dpi"]}}
    result["integration"]["background_choices"] = compact_backgrounds(
        settings["integration"]["background_offsets_px"])
    row_slices = {}
    for row in rows:
        row_cfg = merge(settings["slices"], settings["additional_qy_rows"]["row_settings"].get(
            row["row_id"], {}).get("slices", {}))
        centers, widths = row_cfg["centers_qsx_Ainv"], row_cfg["widths_Ainv"]
        if centers is None or widths is None:
            raise ValueError(f"Standalone export requires resolved slices for {row['row_id']}")
        item = deepcopy(row)
        if (item.get("offset_qdy_px") is None and item["target_qsy_Ainv"] != 0
                and "center_qdy_px" not in item):
            raise ValueError(f"Standalone export requires accepted pixel centers for {row['row_id']}")
        row_slices[row["row_id"]] = {"centers_qsx_Ainv": deepcopy(centers),
                                     "widths_Ainv": compact_widths(widths), "mode": row_cfg["mode"]}
        if any(key in item for key in (
                "background_choices", "background_offsets_by_frame_px", "background_choices_by_slice_frame_px")):
            raise ValueError("Convert row/frame/slice background overrides to shared constants before export")
        result["rows"].append(item)
    result["slices"] = row_slices["qy0"]
    for item in result["rows"]:
        differences = {key: value for key, value in row_slices[item["row_id"]].items()
                       if value != result["slices"][key]}
        item.pop("slices", None)
        if differences:
            item["slices"] = differences
    return result


def publish_standalone(directory, settings, state):
    """Publish once per accepted result, preserving edits on later saves."""
    directory = Path(directory)
    portable = standalone_settings(settings, state["rows"])
    for record in state.get("stages", {}).values():
        if "background_offsets_by_frame_px" in record.get("result", {}):
            raise ValueError("Rerun the harness to replace obsolete frame-specific background checkpoints")
    signature = fingerprint({"settings": portable, "script": STANDALONE_SCRIPT})
    if signature == state.get("standalone_signature"):
        return
    old_hashes = state.get("standalone_hashes", {})
    archive = directory / "archives" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    for name in ("reduce_sample.py", "settings.json"):
        path = directory / name
        if not path.exists():
            continue
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        edited = (name in old_hashes and current != old_hashes[name])
        if name == "reduce_sample.py" and not old_hashes:
            edited = path.read_text(encoding="utf-8") != SCRIPT
        if edited:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / name)
    write_json(directory / "settings.json", portable)
    script = directory / "reduce_sample.py"
    temporary = script.with_suffix(".py.tmp")
    temporary.write_text(STANDALONE_SCRIPT, encoding="utf-8")
    temporary.replace(script)
    state["standalone_signature"] = signature
    state["standalone_hashes"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                  for name in ("reduce_sample.py", "settings.json")}
    state["standalone_published"] = True


def restore_harness(directory, settings, state):
    """Return completed deliverables to harness mode, retaining portable files."""
    directory = Path(directory)
    archive = directory / "archives" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("reduce_sample.py", "settings.json"):
        path = directory / name
        if path.exists():
            shutil.copy2(path, archive / name)
    write_json(directory / "settings.json", settings)
    script = directory / "reduce_sample.py"
    temporary = script.with_suffix(".py.tmp")
    temporary.write_text(SCRIPT, encoding="utf-8")
    temporary.replace(script)
    for key in ("standalone_signature", "standalone_hashes", "standalone_published"):
        state.pop(key, None)
