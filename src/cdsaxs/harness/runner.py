"""Staged reductions with immutable input settings and review checkpoints."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

from ..data.dataset import Dataset, ReducedDataset, ReducedSlices
from ..loaders import load_data as loaders
from ..reduction import slice_reduced_dataset
from .config import fingerprint, merge, read_json, sample_settings, slug, write_json
from .diagnostics import (
    NeedsReview, box_ranges, check, detect_qy_rows, fit_phi, geometry_proposal,
    phi_figure, representative_frames, row_figure, slice_proposal,
)
from .script_generator import generate


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def selected_files(settings):
    cfg = settings["input"]
    check(cfg["data_directory"], "Specify input.data_directory")
    directory = Path(cfg["data_directory"])
    check(directory.is_dir(), f"Data directory does not exist: {directory}")
    if cfg["filenames"] is not None:
        files = [str((directory / filename).resolve()) for filename in cfg["filenames"]]
    elif cfg["metadata_csv"]:
        with Path(cfg["metadata_csv"]).open(newline="", encoding="utf-8-sig") as handle:
            files = [str((Path(cfg["metadata_csv"]).parent / row["filename"]).resolve())
                     for row in csv.DictReader(handle)]
    else:
        names = loaders.filter_filenames(str(directory), cfg["filter_substrings"],
                                         cfg["file_extension"])
        supported = {".h5", ".hdf5", ".tif", ".tiff", ".edf", ".bin"}
        files = [str((directory / name).resolve()) for name in names
                 if Path(name).suffix.lower() in supported]
    check(files, "No data files matched input selection")
    check(all(Path(path).is_file() for path in files), "One or more input files are missing")
    return sorted(set(files))


def inventory(settings):
    cfg = settings["input"]
    check(cfg["data_directory"], "Specify input.data_directory")
    directory = Path(cfg["data_directory"])
    check(directory.is_dir(), f"Data directory does not exist: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file()
                   and path.suffix.lower() in {".h5", ".hdf5", ".tif", ".tiff", ".edf", ".bin"})
    candidates = [path.name for path in files
                  if any(term in path.name.casefold() for term in ("srm", "agbh"))]
    return {"data_directory": str(directory), "files": [file_identity(p) for p in files],
            "reference_candidates": candidates}


def load_dataset(settings, calibration=None, offsets=None, reference=False):
    """Always construct fresh Data2D objects; frame IDs survive repeated angles."""
    cfg = settings["input"]
    metadata = deepcopy(cfg["metadata"])
    if calibration:
        metadata.update({key: calibration[key] for key in ("sdd_cm", "center_px")})
    if offsets:
        metadata.update({key: value for key, value in offsets.items() if value is not None})
    files = selected_files(settings)
    dataset = Dataset(name=settings["sample_name"])
    if cfg["metadata_csv"]:
        loaded = loaders.LoadDataset_MetadataCSV(
            settings["sample_name"], cfg["metadata_csv"], verbose=False,
            filetype=cfg["filetype"], detector_type=cfg["detector_type"])
        frames = list(loaded.datas.values())
        requested = set(files)
        frames = [data for data in frames if str((
            Path(data.metadata["data_directory"]) / data.metadata["filename"]
        ).resolve()) in requested]
    else:
        frames = []
        for filepath in files:
            kind = cfg["filetype"]
            if kind is None and Path(filepath).suffix.lower() in {".h5", ".hdf5"}:
                kind = "smi-h5"
            if kind == "smi-h5":
                loaded = loaders.LoadData(filepath, metadata=metadata,
                                          user_params=cfg["user_params"], filetype=kind)
                frames.extend(loaded if isinstance(loaded, list) else [loaded])
            else:
                loaded = loaders.LoadDataset(
                    settings["sample_name"], str(Path(filepath).parent),
                    filenames=[Path(filepath).name], metadata=metadata,
                    user_params=cfg["user_params"], metadata_pattern=cfg["metadata_pattern"],
                    metadata_scales=cfg["metadata_scales"],
                    filetype=kind, detector_type=cfg["detector_type"], verbose=False)
                frames.extend(loaded.datas.values())
    for index, data in enumerate(frames):
        if reference and "sample_phi_deg" not in data.metadata:
            data.update_metadata({"sample_phi_deg": 0.0})
            data.update_user_params({"harness_reference_phi_assumed": True})
        data.update_metadata(cfg["metadata"], overwrite=True)
        data.update_user_params(cfg["user_params"])
        frame_id = f"{data.metadata.get('filename', 'frame')}:{index:06d}"
        data.update_metadata({"name": frame_id}, overwrite=True)
        data.name = frame_id
        data.mask |= ~np.isfinite(data.image)
        dataset.add_data(data)
    if calibration:
        metadata = {key: calibration[key] for key in ("sdd_cm", "center_px")}
        dataset.update_all_metadata(metadata, verbose=False)
    if offsets:
        dataset.update_all_metadata({k: v for k, v in offsets.items() if v is not None},
                                    verbose=False)
    check(dataset.datas, "No detector frames loaded")
    for data in dataset.datas.values():
        check(np.isfinite(data.sample_phi_deg), f"Invalid phi angle in {data.name}")
        if calibration:
            center = data.metadata["center_px_detector"]
            check(all(0 <= c < size for c, size in zip(center, data.image.shape)),
                  f"Calibrated beam center is outside detector for {data.name}")
            check(data.metadata["sdd_cm"] > 0 and data.metadata["wavelength_nm"] > 0
                  and data.metadata["pixel_size_um"] > 0,
                  "SDD, wavelength, and pixel size must be positive")
    return dataset


def apply_intensity_corrections(dataset, settings):
    """Apply corrections once to fresh images using current effective phi."""
    cfg = settings["corrections"]
    for data in dataset.datas.values():
        for key in cfg["normalize_by"]:
            value = data._get_metadata(key)
            check(np.isscalar(value) and not isinstance(value, (bool, np.bool_)),
                  f"A positive scalar {key} is required for {data.name}")
            try:
                value = float(value)
            except (ValueError, TypeError) as error:
                raise NeedsReview(f"A positive numeric {key} is required for {data.name}") from error
            check(np.isfinite(value) and value > 0,
                  f"A positive scalar {key} is required for {data.name}")
        if cfg["substrate_absorption"]:
            required = ("substrate_thickness_um", "substrate_attenuation_coeff_um-1")
            additions = {key: cfg[key] for key in required if cfg[key] is not None}
            data.update_metadata(additions)
            check(all(key in data.metadata and np.isfinite(data.metadata[key])
                      and data.metadata[key] > 0 for key in required),
                  "Supply substrate thickness and attenuation coefficient, or disable absorption")
        if cfg["footprint"] or cfg["substrate_absorption"]:
            check(abs(data.sample_phi_deg) < 89.9,
                  "Footprint/absorption model requires transmission away from grazing incidence")
    dataset.normalize_all_data_by_metadata(cfg["normalize_by"])
    if cfg["substrate_absorption"]:
        dataset.apply_substrate_absorption_correction()
    if cfg["footprint"]:
        dataset.apply_footprint_correction()
    for data in dataset.datas.values():
        data.mask |= ~np.isfinite(data.image)
        check(np.isfinite(data.image[~data.mask]).any(), "Corrections left no finite intensity data")


def search_arguments(data, settings):
    """Set a contrast-based threshold when no absolute threshold is supplied.

    A relative threshold applied to log(counts) can identify background plateaus
    as peaks. The automatic absolute threshold sits above the measured baseline.
    """
    arguments = {key: value for key, value in settings.items() if value is not None}
    reserved = {"show_plot", "update", "beam_center_guess", "pitch_nm", "peak_orders"}
    check(not reserved.intersection(arguments), "Peak search contains harness-controlled arguments")
    box_keys = {"width_qdy_px", "width_qdx_px", "range_qdy_px", "range_qdx_px",
                "center_qdy", "center_qdx", "shift_box_qdy_px", "shift_box_qdx_px"}
    yr, xr = data.get_box_dims(**{key: value for key, value in arguments.items() if key in box_keys})
    image = np.where(data.mask[yr[0]:yr[1], xr[0]:xr[1]], np.nan,
                     data.image[yr[0]:yr[1], xr[0]:xr[1]])
    axis = arguments.get("peak_axis", 1 if xr[1] - xr[0] >= yr[1] - yr[0] else 0)
    axis = {"qdx": 1, "qdy": 0}.get(axis, axis)
    check(axis in (0, 1), "Peak axis must be qdx, qdy, 0, or 1")
    profile = (np.nanmean(image, axis=1 - axis) if arguments.get("integration_mode") == "mean"
               else np.nansum(image, axis=1 - axis))
    profile[np.isnan(image).any(axis=1 - axis)] = np.nan
    values = profile[np.isfinite(profile)]
    if arguments.get("log_scale", True):
        values = np.log10(values[values > 0])
    check(len(values), "No finite profile values for peak finding")
    baseline = float(np.median(values))
    contrast = float(np.max(values) - baseline)
    if arguments.get("algorithm", "scikit") == "scipy":
        distance = arguments.pop("min_distance", None)
        arguments.pop("threshold_rel", None)
        arguments.pop("threshold_abs", None)
        if distance is not None:
            arguments.setdefault("distance", distance)
        arguments.setdefault("prominence", max(contrast * 0.2, np.finfo(float).eps))
    elif settings.get("threshold_abs") is None:
        arguments["threshold_abs"] = baseline + 0.2 * contrast
    return arguments


def integrate_row(dataset, row, geometry):
    ranges_y, ranges_x = {}, {}
    backgrounds = geometry["background_offsets_px"]
    for key, data in dataset.datas.items():
        yr, xr = box_ranges(data, row, geometry)
        for offset in backgrounds or []:
            check(abs(offset) >= geometry["height_px"], "Background overlaps signal box")
            check(0 <= yr[0] + offset < yr[1] + offset <= data.image.shape[0],
                  f"Background region leaves detector for {data.name}")
        ranges_y[key], ranges_x[key] = yr, xr
    reduced = dataset.integrate_dataset(
        mode=geometry["mode"], axis=0, range_qdy_px=ranges_y, range_qdx_px=ranges_x,
        subtract_background_offset=backgrounds)
    for profile in reduced.datas:
        profile.row_id = row["row_id"]
        for component in ("Iq", "qsx", "qsy", "qsz", "qsr", "qs"):
            profile.mask |= ~np.isfinite(getattr(profile, component))
    return reduced


def phi_groups(dataset, settings):
    """Use native metadata selection with an explicit, disjoint zero policy."""
    cfg = settings["export"]
    tolerance = cfg["zero_tolerance_deg"]
    for data in dataset.datas.values():
        angle = (data.sample_phi_deg if cfg["phi_basis"] == "corrected"
                 else data.metadata["sample_phi_deg"])
        data.update_user_params({"harness_export_phi_deg": float(angle)})
    positive = dataset.filter_data_by_metadata(
        harness_export_phi_deg=(float(np.nextafter(tolerance, np.inf)), float("inf")))
    negative = dataset.filter_data_by_metadata(
        harness_export_phi_deg=(float("-inf"), float(np.nextafter(-tolerance, -np.inf))))
    zero = dataset.filter_data_by_metadata(harness_export_phi_deg=(-tolerance, tolerance))
    if cfg["zero_phi"] == "positive":
        positive += zero
    elif cfg["zero_phi"] == "negative":
        negative += zero
    result = {"posPhi": set(positive), "negPhi": set(negative)}
    if cfg["zero_phi"] == "separate":
        result["zeroPhi"] = set(zero)
    return result


class ReductionRun:
    def __init__(self, settings, directory, accept=()):
        self.settings = deepcopy(settings)
        self.resolved = deepcopy(settings)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.accept = set(accept)
        self.state_path = self.directory / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema_version": 1, "stages": {}, "events": [], "artifacts": [],
        }
        check(self.state.get("sample", settings["sample_name"]) == settings["sample_name"],
              "This resume directory belongs to a different sample")
        self.state["status"] = "running"
        self.state["sample"] = settings["sample_name"]
        generate(self.directory, settings)
        write_json(self.directory / "invocation_settings.json", settings)
        try:
            self.files = selected_files(settings)
            self.inputs = [file_identity(path) for path in self.files]
            for filepath in self.files:
                if Path(filepath).suffix == ".bin" and Path(filepath).with_suffix(".info").is_file():
                    self.inputs.append(file_identity(Path(filepath).with_suffix(".info")))
            if settings["input"]["metadata_csv"]:
                self.inputs.append(file_identity(settings["input"]["metadata_csv"]))
        except (NeedsReview, OSError) as error:
            self.state["status"] = "needs_review"
            self.archive_exports("Input selection is no longer valid")
            self.event(str(error))
            self.save()
            raise
        self.state["inputs"] = self.inputs
        self.state["versions"] = self.versions()
        execution_settings = {key: value for key, value in settings.items()
                              if key not in {"output", "mode", "allowed_decisions", "decision_notes"}}
        execution_signature = fingerprint({"settings": execution_settings, "inputs": self.inputs})
        previous_signature = self.state.get("execution_signature")
        if previous_signature is not None and previous_signature != execution_signature:
            self.archive_exports("Settings or input files changed")
        self.state["execution_signature"] = execution_signature
        self.dependencies = {}

    @staticmethod
    def versions():
        packages = {}
        for name in ("nist_cdsaxs", "numpy", "scipy", "matplotlib", "h5py"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "source checkout"
        repository = Path(__file__).resolve().parents[3]
        try:
            packages["repository_commit"] = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            packages["repository_commit"] = None
        digest = hashlib.sha256()
        for module in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
            digest.update(str(module.relative_to(Path(__file__).resolve().parents[1])).encode())
            digest.update(module.read_bytes())
        packages["source_digest"] = digest.hexdigest()
        return packages

    def event(self, message):
        print(message)
        self.state["events"].append(message)

    def archive_exports(self, reason):
        """Preserve previous exports while a revised reduction is pending."""
        csv_directory = self.directory / "csv"
        if not csv_directory.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        destination = self.directory / "archives" / stamp / "csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        csv_directory.rename(destination)
        prefix = str(destination.relative_to(self.directory))
        self.state["artifacts"] = [prefix + name[3:] if name.startswith("csv/") else name
                                   for name in self.state["artifacts"]]
        self.event(f"{reason}; previous CSVs preserved in {prefix}")

    def save(self):
        for name in ("reduce_sample.py", "settings.json", "invocation_settings.json",
                     "resolved_settings.json", "calibration.json", "coverage.json",
                     "row_reductions.json"):
            if (self.directory / name).exists():
                self.state["artifacts"].append(name)
        self.state["artifacts"] = sorted(set(self.state["artifacts"]))
        write_json(self.state_path, self.state)
        write_json(self.directory / "resolved_settings.json", self.resolved)
        self.report()

    def report(self):
        lines = [f"# Reduction: {self.settings['sample_name']}", "",
                 f"Status: **{self.state['status']}**", "",
                 "Q units: Å⁻¹; SDD: cm; beam center: [row, column] pixels; angles: degrees.",
                 "Intensity is normalized in arbitrary units. Fit scatter and conditional fit",
                 "uncertainties exclude calibration systematics; full intensity uncertainty",
                 "propagation is not implemented by the underlying reduction.", ""]
        lines += [f"- Decision note: {note}" for note in self.settings["decision_notes"]]
        lines += [f"- {message}" for message in self.state["events"]]
        lines += ["", "## Stages", ""]
        for name, record in self.state["stages"].items():
            lines.append(f"- {name}: {record['status']}")
        lines += ["", "## Artifacts", ""]
        lines += [f"- [{name}]({name})" for name in sorted(set(self.state["artifacts"]))]
        (self.directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def figures(self, figure, stem):
        import matplotlib.pyplot as plt
        if figure is None:
            return
        if isinstance(figure, (list, tuple)):
            for index, item in enumerate(figure):
                self.figures(item, f"{stem}_{index + 1}")
            return
        for extension in self.settings["output"]["figure_formats"]:
            path = self.directory / "figures" / f"{slug(stem)}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=self.settings["output"]["dpi"], bbox_inches="tight")
            self.state["artifacts"].append(str(path.relative_to(self.directory)))
        plt.close(figure)

    def checkpoint(self, name, relevant, compute, decisions=(), supplied=False):
        signature = fingerprint({"inputs": self.inputs, "dependencies": self.dependencies,
                                 "settings": relevant, "quality": self.settings["quality"],
                                 "implementation": self.state["versions"]})
        previous = self.state["stages"].get(name)
        cached = previous is not None and previous["fingerprint"] == signature
        if cached:
            record = previous
        else:
            # A changed stage invalidates every later result and approval.
            if name in self.state["stages"]:
                stages = list(self.state["stages"])
                for later in stages[stages.index(name):]:
                    self.state["stages"].pop(later)
                self.archive_exports(f"Stage {name} changed")
                self.event(f"Settings/input change invalidated {name} and subsequent stages")
            result = compute()
            record = {"fingerprint": signature, "status": "proposed", "result": result}
            self.state["stages"][name] = record
        result = record["result"]
        failures = result.get("failures", [])
        if failures:
            record["status"] = "failed_quality"
            self.save()
            raise NeedsReview("; ".join(failures))
        if supplied:
            record["status"] = "accepted_supplied"
        elif name.split(":")[0] in self.accept:
            check(cached, f"Inspect {name} diagnostics before accepting this new result")
            record["status"] = "accepted_review"
        elif self.settings["mode"] == "automatic" and set(decisions) <= set(self.settings["allowed_decisions"]):
            record["status"] = "accepted_automatic"
        if not record["status"].startswith("accepted"):
            self.save()
            raise NeedsReview(f"Review {name} figures and state.json, then rerun with "
                              f"--accept {name.split(':')[0]}, or supply accepted JSON values")
        self.dependencies[name] = signature
        self.save()
        return deepcopy(result)

    def calibrate(self):
        cfg = deepcopy(self.settings["calibration"])
        source = None
        if cfg["source_file"]:
            source = read_json(cfg["source_file"])
            values = source.get("calibration", source)
            for key in ("sdd_cm", "center_px"):
                if cfg[key] is None:
                    cfg[key] = values.get(key)
        for key in ("sdd_cm", "center_px"):
            if cfg[key] is None:
                cfg[key] = self.settings["input"]["metadata"].get(key)
        known = cfg["center_px"] is not None and cfg["sdd_cm"] is not None
        reference_paths = []
        if not known or self.settings["refit_known_constants"]:
            found = inventory(self.settings)
            references = cfg["reference_filenames"]
            if references is None and cfg["reference_sample"]:
                references = [item["path"] for item in found["files"]
                              if cfg["reference_sample"].casefold() in Path(item["path"]).name.casefold()]
            if references is None:
                candidates = found["reference_candidates"]
                check(len(candidates) == 1, "Identify calibration.reference_sample/reference_filenames "
                      "or supply calibrated SDD and beam center. Candidates: " + repr(candidates))
                references = candidates
            reference_paths = [str((Path(self.settings["input"]["data_directory"]) / p).resolve())
                               for p in references]
            check(reference_paths, "No reference files found")
            check(cfg["reference_pitch_nm"] is not None,
                  "Supply calibration.reference_pitch_nm from the reference certificate")
        relevant = {"calibration": cfg, "input": self.settings["input"],
                    "reference_inputs": [file_identity(p) for p in reference_paths],
                    "source_input": file_identity(cfg["source_file"]) if source else None,
                    "refit": self.settings["refit_known_constants"]}

        def compute():
            if known and not self.settings["refit_known_constants"]:
                return {"sdd_cm": cfg["sdd_cm"], "center_px": cfg["center_px"],
                        "source": cfg["source_file"] or "supplied_settings", "failures": []}
            reference_settings = deepcopy(self.settings)
            reference_settings["input"].update(
                filenames=reference_paths, metadata_csv=None, filter_substrings=None,
                metadata_pattern=cfg["reference_metadata_pattern"],
            )
            dataset = load_dataset(reference_settings, reference=True)
            data = min(dataset.datas.values(), key=lambda d: abs(d.sample_phi_deg))
            if cfg["sdd_guess_cm"] is not None:
                data.update_metadata({"sdd_cm": cfg["sdd_guess_cm"]})
            if cfg["peak_orders"] is None:
                check(data.metadata.get("sdd_cm", 0) > 0,
                      "Supply calibration.sdd_guess_cm or explicit peak_orders to index reference peaks")
            search = deepcopy(cfg["search"])
            center = cfg["center_px"]
            if center is None or self.settings["refit_known_constants"]:
                guess = (cfg["beam_center_guess_px"] or cfg["center_px"]
                         or [size / 2 for size in data.image.shape])
                data.update_metadata({"center_px": guess})
                center, figure = data.find_beam_center_from_peaks(
                    beam_center_guess=guess, show_plot=True, update=False,
                    **search_arguments(data, search))
                self.figures(figure, "00_reference_beam_center")
            data.update_metadata({"center_px": center})
            if cfg["sdd_cm"] is not None:
                data.update_metadata({"sdd_cm": cfg["sdd_cm"]})
            sdd, scatter, figure = data.find_sdd_from_reference_peaks(
                cfg["reference_pitch_nm"], peak_orders=cfg["peak_orders"], show_plot=True,
                update=False, **search_arguments(data, search))
            self.figures(figure, "00_reference_sdd")
            if cfg["sdd_cm"] is not None and not self.settings["refit_known_constants"]:
                sdd = cfg["sdd_cm"]
            check(np.isfinite(sdd) and sdd > 0, "Invalid reference SDD estimate")
            relative_scatter = float(scatter / sdd)
            failures = []
            if (not np.isfinite(relative_scatter) or relative_scatter
                    > self.settings["quality"]["max_sdd_relative_scatter"]):
                failures.append("Reference SDD peak-order scatter exceeds its configured bound")
            return {"sdd_cm": float(sdd), "center_px": list(center),
                    "sdd_peak_order_scatter_cm": float(scatter),
                    "reference_inputs": relevant["reference_inputs"],
                    "reference_pitch_nm": cfg["reference_pitch_nm"],
                    "reference_phi_assumed_zero": bool(data.user_params.get("harness_reference_phi_assumed")),
                    "failures": failures}

        result = self.checkpoint("calibration", relevant, compute, ["accept_calibration"],
                                 supplied=known and not self.settings["refit_known_constants"])
        self.resolved["calibration"].update({k: result[k] for k in ("sdd_cm", "center_px")})
        write_json(self.directory / "calibration.json", result)
        return result

    def estimate_phi(self, dataset):
        known = self.settings["offsets"]["sample_phi_offset_deg"]
        cfg = self.settings["phi"]

        def compute():
            if known is not None and not self.settings["refit_known_constants"]:
                return {"sample_phi_offset_deg": known, "failures": []}
            check(cfg["pitch_nm"] is not None, "Supply phi.pitch_nm for the apparent-SDD diagnostic")
            frames = list(dataset.datas.values())
            if cfg["angles_deg"] is None:
                lo, hi = cfg["angle_range_deg"]
                frames = [d for d in frames if lo <= d.metadata["sample_phi_deg"] <= hi]
            else:
                frames = [min(frames, key=lambda d: abs(d.metadata["sample_phi_deg"] - angle))
                          for angle in cfg["angles_deg"]]
                check(all(abs(d.metadata["sample_phi_deg"] - angle) < 0.05
                          for d, angle in zip(frames, cfg["angles_deg"])),
                      "Requested phi diagnostic angles are not available")
            angles, sdds, scatters, failures, inferred_orders = [], [], [], [], []
            for data in frames:
                try:
                    search = search_arguments(data, cfg["search"])
                    orders = cfg["peak_orders"]
                    if orders is None:
                        check(search.get("peak_axis", 1) in (1, "qdx"),
                              "Phi order inference requires horizontal peaks")
                        peaks, _, _, _ = data.find_peaks2D_one_axis(show_plot=False, **search)
                        peaks = peaks[np.argsort(peaks[:, 1])]
                        peak_q = np.interp(peaks[:, 1], np.arange(len(data.qbx_1d)), data.qbx_1d)
                        fundamental_q = 2 * np.pi / (cfg["pitch_nm"] * 10)
                        orders = np.abs(np.rint(peak_q / fundamental_q)).astype(int).tolist()
                        check(orders and all(order > 0 for order in orders),
                              "Inferred zero peak order; exclude direct beam or supply phi.peak_orders")
                        if inferred_orders:
                            check(orders == inferred_orders[0]["orders"],
                                  "Detected peak orders changed across phi; tune search or supply fixed orders")
                    sdd, scatter, figure = data.find_sdd_from_reference_peaks(
                        cfg["pitch_nm"], peak_orders=orders,
                        show_plot=True, update=False, **search)
                    self.figures(figure, f"01_phi_peaks_{slug(data.name)}")
                    if not np.isfinite(sdd) or not np.isfinite(scatter):
                        raise ValueError("nonfinite peak estimate")
                    angles.append(float(data.metadata["sample_phi_deg"]))
                    sdds.append(float(sdd))
                    scatters.append(float(scatter))
                    inferred_orders.append({"frame": data.name, "orders": orders})
                except (ValueError, IndexError, RuntimeError) as error:
                    failures.append({"frame": data.name, "reason": str(error)})
            write_json(self.directory / "phi_measurements.json", {
                "nominal_phi_deg": angles, "apparent_sdd_cm": sdds,
                "sdd_peak_order_scatter_cm": scatters, "failed_frames": failures,
                "peak_orders_by_frame": inferred_orders})
            self.state["artifacts"].append("phi_measurements.json")
            for failure in failures:
                self.event(f"Phi diagnostic omitted {failure['frame']}: {failure['reason']}")
            result = fit_phi(angles, sdds, self.settings["quality"])
            result["sdd_peak_order_scatter_cm"] = scatters
            result["failed_frames"] = failures
            result["peak_orders_by_frame"] = inferred_orders
            self.figures(phi_figure(result), "01_phi_offset")
            self.event("Proposed JSON: " + json.dumps({"offsets": {
                "sample_phi_offset_deg": result["sample_phi_offset_deg"]}}))
            return result

        result = self.checkpoint("phi", {"phi": cfg, "known": known,
                                         "corrections": self.settings["corrections"],
                                         "refit": self.settings["refit_known_constants"]}, compute,
                                 ["accept_angle_corrections"],
                                 supplied=known is not None and not self.settings["refit_known_constants"])
        self.resolved["offsets"]["sample_phi_offset_deg"] = result["sample_phi_offset_deg"]
        return result["sample_phi_offset_deg"]

    def estimate_chi(self, dataset):
        known = self.settings["offsets"]["sample_chi_offset_deg"]
        data = min(dataset.datas.values(), key=lambda d: abs(d.sample_phi_deg))

        def compute():
            if known is not None and not self.settings["refit_known_constants"]:
                return {"sample_chi_offset_deg": known, "failures": []}
            angle, figure = data.find_chi_from_peaks(
                show_plot=True, **search_arguments(data, self.settings["chi"]["search"]))
            self.figures(figure, "02_chi_correction")
            check(np.isfinite(angle), "No finite chi estimate; tune the peak search")
            offset = float(angle - data.metadata["sample_chi_deg"])
            failures = []
            if abs(data.sample_phi_deg) > self.settings["quality"]["max_chi_near_zero_deg"]:
                failures.append("Nearest corrected-normal-incidence image is too far from zero for chi")
            if abs(offset) > self.settings["quality"]["max_chi_abs_deg"]:
                failures.append("Chi offset exceeds its configured bound")
            return {"sample_chi_offset_deg": offset, "total_chi_deg": float(angle),
                    "frame": data.name, "effective_phi_deg": float(data.sample_phi_deg),
                    "failures": failures}

        result = self.checkpoint("chi", {"chi": self.settings["chi"], "known": known,
                                         "refit": self.settings["refit_known_constants"]}, compute,
                                 ["accept_angle_corrections"],
                                 supplied=known is not None and not self.settings["refit_known_constants"])
        self.resolved["offsets"]["sample_chi_offset_deg"] = result["sample_chi_offset_deg"]
        dataset.update_all_metadata({"sample_chi_offset_deg": result["sample_chi_offset_deg"]}, verbose=False)

    def estimate_omega(self, dataset):
        known = self.settings["offsets"]["sample_omega_offset_deg"]
        cfg = self.settings["omega"]

        def compute():
            if known is not None and not self.settings["refit_known_constants"]:
                return {"sample_omega_offset_deg": known, "failures": []}
            frames = list(dataset.datas.values())
            if cfg["angles_deg"] is not None:
                frames = [min(frames, key=lambda d: abs(d.sample_phi_deg - angle))
                          for angle in cfg["angles_deg"]]
                check(all(abs(d.sample_phi_deg - angle) < 0.05
                          for d, angle in zip(frames, cfg["angles_deg"])),
                      "Requested corrected omega angles are not available")
            else:
                frames = sorted([d for d in frames if abs(d.sample_phi_deg) >= cfg["min_abs_phi_deg"]],
                                key=lambda d: d.sample_phi_deg)
                if len(frames) > cfg["max_frames"]:
                    indexes = np.linspace(0, len(frames) - 1, cfg["max_frames"], dtype=int)
                    frames = [frames[i] for i in indexes]
            check(len({data.name for data in frames}) == len(frames),
                  "Requested omega angles select repeated frames")
            estimates, failed = [], []
            for data in frames:
                try:
                    check(abs(data.sample_phi_deg) >= cfg["min_abs_phi_deg"],
                          "Omega angle too close to normal incidence")
                    angle, figure = data.find_omega_from_peaks(
                        show_plot=True, **search_arguments(data, cfg["search"]))
                    self.figures(figure, f"03_omega_{slug(data.name)}")
                    check(np.isfinite(angle), "No finite omega estimate")
                    estimates.append({"frame": data.name, "phi_deg": float(data.sample_phi_deg),
                                      "offset_deg": float(angle - data.metadata["sample_omega_deg"])})
                except (ValueError, IndexError, RuntimeError) as error:
                    failed.append({"frame": data.name, "reason": str(error)})
            check(len(estimates) >= self.settings["quality"]["min_omega_frames"],
                  "Insufficient valid omega estimates; tune searches or supply a known offset")
            values = [item["offset_deg"] for item in estimates]
            average, scatter = float(np.mean(values)), float(np.std(values, ddof=1))
            failures = []
            q = self.settings["quality"]
            if q["require_omega_both_signs"] and not (
                min(item["phi_deg"] for item in estimates) < 0 < max(item["phi_deg"] for item in estimates)
            ):
                failures.append("Omega estimates do not cover both signs of phi")
            if abs(average) > q["max_omega_abs_deg"] or scatter > q["max_omega_scatter_deg"]:
                failures.append("Omega magnitude or scatter exceeds its configured bound")
            return {"sample_omega_offset_deg": average, "scatter_deg": scatter,
                    "estimates": estimates, "failed_frames": failed, "failures": failures}

        result = self.checkpoint("omega", {"omega": cfg, "known": known,
                                           "refit": self.settings["refit_known_constants"]}, compute,
                                 ["accept_angle_corrections"],
                                 supplied=known is not None and not self.settings["refit_known_constants"])
        self.resolved["offsets"]["sample_omega_offset_deg"] = result["sample_omega_offset_deg"]
        dataset.update_all_metadata({"sample_omega_offset_deg": result["sample_omega_offset_deg"]}, verbose=False)

    def rows(self, dataset):
        cfg = self.settings["additional_qy_rows"]
        rows = [{"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": None}]
        if not cfg["enabled"]:
            return rows
        reference = min(dataset.datas.values(), key=lambda d: abs(d.sample_phi_deg))

        def compute():
            if cfg["offsets_qsy_Ainv"] is not None:
                result = {"offsets_qsy_Ainv": cfg["offsets_qsy_Ainv"]}
            elif cfg["offsets_qdy_px"] is not None:
                result = {"offsets_qdy_px": cfg["offsets_qdy_px"]}
            else:
                result = detect_qy_rows(reference, self.settings)
            result["rows"] = self.make_rows(result)
            self.figures(row_figure(reference, result["rows"]), "04_qy_rows")
            return result

        supplied = cfg["offsets_qsy_Ainv"] is not None or cfg["offsets_qdy_px"] is not None
        result = self.checkpoint("qy_rows", cfg, compute, ["choose_qy_offsets"], supplied=supplied)
        for key in ("offsets_qsy_Ainv", "offsets_qdy_px"):
            if key in result:
                self.resolved["additional_qy_rows"][key] = result[key]
        if result.get("omitted_detector_rows_px"):
            self.event("Additional rows omitted by max_rows: " + repr(result["omitted_detector_rows_px"]))
        return result["rows"]

    @staticmethod
    def make_rows(result):
        rows = [{"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": None}]
        pixel_mode = "offsets_qdy_px" in result
        key = "offsets_qdy_px" if pixel_mode else "offsets_qsy_Ainv"
        for index, value in enumerate(sorted(set(result[key]))):
            if value == 0:
                continue
            rows.append({"row_id": f"qy_{index + 1:03d}",
                         "target_qsy_Ainv": None if pixel_mode else value,
                         "offset_qdy_px": value if pixel_mode else None})
        return rows

    def row_settings(self, row):
        overrides = self.settings["additional_qy_rows"]["row_settings"].get(row["row_id"], {})
        return merge(self.settings, overrides)

    def preview(self, dataset, row):
        settings = self.row_settings(row)
        cfg = settings["integration"]
        decisions = []
        if cfg["height_px"] is None or cfg["width_px"] is None:
            decisions.append("choose_integration_box")
        if cfg["subtract_background"] and cfg["background_offsets_px"] is None:
            decisions.append("choose_backgrounds")

        def compute():
            geometry = geometry_proposal(dataset, row, settings)
            has_positive_measurements = False
            for index, data in enumerate(representative_frames(dataset, 3)):
                yrange, xrange = box_ranges(data, row, geometry)
                profile, _ = data.integrate_box(
                    mode=geometry["mode"], axis=0,
                    range_qdy_px=yrange, range_qdx_px=xrange,
                    subtract_background_offset=geometry["background_offsets_px"], show_plot=False)
                positive = bool(np.any(np.isfinite(profile.Iq) & (profile.Iq > 0)))
                has_positive_measurements |= positive
                from ..plotting.plotting import plot_data2d_integrate_box
                figures = plot_data2d_integrate_box(profile, log_scale=positive)
                self.figures(figures, f"05_{row['row_id']}_integration_{index}")
            check(has_positive_measurements,
                  f"No positive intensity measurements remain in {row['row_id']}; inspect backgrounds")
            return geometry

        geometry = self.checkpoint(f"integration:{row['row_id']}",
                                   {"integration": cfg, "slices": settings["slices"], "row": row},
                                   compute, decisions, supplied=not decisions)
        resolved_geometry = {key: geometry[key] for key in ("height_px", "width_px", "background_offsets_px")}
        if row["row_id"] == "qy0":
            self.resolved["integration"].update(resolved_geometry)
        else:
            self.resolved["additional_qy_rows"]["row_settings"].setdefault(row["row_id"], {})[
                "integration"] = merge(cfg, resolved_geometry)
        integrated = integrate_row(dataset, row, geometry)
        self.figures(integrated.plot_data(interpolated_data=False, s=2),
                     f"06_{row['row_id']}_bowtie")
        return geometry, integrated

    def choose_slices(self, integrated, row):
        settings = self.row_settings(row)
        cfg = settings["slices"]
        decisions = []
        if cfg["centers_qsx_Ainv"] is None or cfg["widths_Ainv"] is None:
            decisions.append("choose_slice_locations_and_widths")

        def compute():
            result = slice_proposal(integrated, settings)
            self.slice_overlay(integrated, result, row)
            return result

        result = self.checkpoint(f"slices:{row['row_id']}", {"slices": cfg, "row": row},
                                 compute, decisions, supplied=not decisions)
        accepted = []
        skipped = []
        for item in result["slices"]:
            if item["skip"]:
                permitted = (self.settings["mode"] == "automatic"
                             and "skip_gap_slices" in self.settings["allowed_decisions"])
                check(permitted, f"Slice {item['center_qsx_Ainv']:g} A^-1 has inadequate coverage; "
                      "remove it from settings or authorize skip_gap_slices in automatic mode")
                skipped.append(item)
                self.event(f"Skipped {row['row_id']} slice {item['center_qsx_Ainv']:g} A^-1: "
                           f"coverage {item['angle_coverage']:.1%}")
            else:
                accepted.append(item)
                missing = sum(not point["valid"] for point in item["coverage"])
                if missing:
                    self.event(f"{row['row_id']} slice {item['center_qsx_Ainv']:g} A^-1 "
                               f"omits {missing} angle frames; see coverage.json")
        check(accepted, f"No usable slices remain in {row['row_id']}")
        self.state.setdefault("skipped_slices", {})[row["row_id"]] = skipped
        resolved_slices = {"centers_qsx_Ainv": [item["center_qsx_Ainv"] for item in accepted],
                           "widths_Ainv": [item["width_Ainv"] for item in accepted]}
        if row["row_id"] == "qy0":
            self.resolved["slices"].update(resolved_slices)
        else:
            self.resolved["additional_qy_rows"]["row_settings"].setdefault(row["row_id"], {})[
                "slices"] = merge(cfg, resolved_slices)
        return accepted, result

    def slice_overlay(self, integrated, result, row):
        import matplotlib.pyplot as plt
        figure = integrated.plot_data(interpolated_data=False, s=2)
        axis = figure.axes[0]
        for item in result["slices"]:
            center, width = item["center_qsx_Ainv"], item["width_Ainv"]
            color = "0.5" if item["skip"] else "#D55E00"
            axis.axvspan(center - width / 2, center + width / 2, alpha=0.2, color=color)
            axis.axvline(center, color=color, lw=0.8)
        axis.set_title(f"{row['row_id']}: proposed slices; grey = insufficient coverage")
        self.figures(figure, f"07_{row['row_id']}_slices")
        # Measured profiles retain gaps; waterfall is a separate diagnostic.
        fig, ax = plt.subplots(figsize=(7, 5))
        for item in result["slices"]:
            ax.plot([point["phi_deg"] for point in item["coverage"]],
                    [int(point["valid"]) for point in item["coverage"]],
                    ".", label=f"{item['center_qsx_Ainv']:.4g}")
        ax.set(xlabel="Corrected phi (degrees)", ylabel="Slice available (0/1)",
               title="Measured slice coverage")
        self.figures(fig, f"07_{row['row_id']}_coverage")

    def export(self, dataset, reductions):
        import matplotlib.pyplot as plt
        cfg = self.settings["export"]
        groups = {"all": set(dataset.datas)}
        if cfg["split_phi"]:
            groups.update(phi_groups(dataset, self.settings))
        coverage = {}
        exports = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        csv_directory = self.directory / "pending_exports" / stamp
        csv_directory.mkdir(parents=True)
        for group, keys in groups.items():
            combined = ReducedSlices()
            for row, geometry, integrated, items, proposal in reductions:
                subset = ReducedDataset([p for p in integrated.datas if p.data2d.name in keys])
                if not subset.datas:
                    continue
                slices, _ = slice_reduced_dataset(
                    subset, q_values=[item["center_qsx_Ainv"] for item in items],
                    q_widths=[item["width_Ainv"] for item in items], mode=proposal["mode"], show_plot=False)
                slices.data = [data for data in slices.data if len(data.Iq)]
                if not slices.data:
                    continue
                check(any(np.any(np.isfinite(data.Iq) & (data.Iq > 0)) for data in slices.data),
                      f"No positive intensity measurements remain in {row['row_id']} / {group}")
                combined.add_slices(slices)
                stem = f"{slug(self.settings['sample_name'])}_{row['row_id']}_{group}"
                if cfg["per_row"]:
                    slices.export_reduced_slices(csv_directory / f"{stem}.csv",
                                                 export_qr=cfg["export_qr"], decimals=cfg["decimals"])
                    exports.append(f"csv/{stem}.csv")
                # Long table retains nonpositive values and frame/row provenance.
                long_path = csv_directory / f"{stem}_measurements.csv"
                with long_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["row", "frame", "nominal_phi_deg", "corrected_phi_deg",
                                     "slice_center_qsx_Ainv", "qsx_Ainv", "qsy_Ainv", "qsz_Ainv",
                                     "qsr_Ainv", "intensity_au"])
                    for item in items:
                        for profile in subset.datas:
                            selection = np.flatnonzero(abs(profile.qsx - item["center_qsx_Ainv"]) <= item["width_Ainv"] / 2)
                            if not len(selection) or profile.mask[selection].any():
                                continue
                            intensity = (np.sum(profile.Iq[selection]) if proposal["mode"] == "sum"
                                         else np.mean(profile.Iq[selection]))
                            coordinates = [float(np.mean(getattr(profile, q)[selection]))
                                           for q in ("qsx", "qsy", "qsz", "qsr")]
                            writer.writerow([row["row_id"], profile.data2d.name,
                                             profile.data2d.metadata["sample_phi_deg"], profile.sample_phi_deg,
                                             item["center_qsx_Ainv"], *coordinates, float(intensity)])
                exports.append(f"csv/{long_path.name}")
                fig, ax = plt.subplots(figsize=(7, 5))
                removed = 0
                for index, data in enumerate(slices.data):
                    valid = np.isfinite(data.Iq) & (data.Iq > 0)
                    removed += int((~valid).sum())
                    order = np.argsort(data.qsz)
                    ax.plot(data.qsz[order], data.Iq[order] * 10.0 ** (index % 8),
                            ".-", ms=2, lw=0.6, label=f"{data.qsx:g}")
                ax.set(xlabel=r"$q_{s,z}$ (Å$^{-1}$)", ylabel="Offset normalized intensity (a.u.)",
                       yscale="log", title=f"{row['row_id']} / {group}")
                self.figures(fig, f"08_{stem}_reduced")
                if removed:
                    self.event(f"{stem}: native CSV excludes {removed} nonpositive/nonfinite points; "
                               "measurements CSV preserves finite nonpositive intensities")
                coverage[row["row_id"]] = proposal
            if cfg["combine_qy_rows"] and combined.data:
                stem = f"{slug(self.settings['sample_name'])}_qyall_{group}"
                combined.export_reduced_slices(csv_directory / f"{stem}.csv",
                                               export_qr=cfg["export_qr"], decimals=cfg["decimals"])
                exports.append(f"csv/{stem}.csv")
        check(exports, "No exportable data remain")
        self.archive_exports("Completed a replacement reduction")
        csv_directory.rename(self.directory / "csv")
        self.state["artifacts"].extend(exports)
        self.state["export_settings"] = cfg
        write_json(self.directory / "coverage.json", coverage)
        write_json(self.directory / "row_reductions.json", [
            {"row": row, "integration": geometry, "slices": proposal}
            for row, geometry, _, _, proposal in reductions])
        self.event(f"Exported {len(exports)} CSV files")

    def run(self, stage="all"):
        import matplotlib.pyplot as plt
        plot_settings = {"font.size": 11, "figure.dpi": 100, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"}
        with plt.style.context("default"), plt.rc_context(plot_settings):
            try:
                calibration = self.calibrate()
                if stage == "calibration":
                    return self.finish_stage()
                provisional = {key: value if value is not None else 0.0
                               for key, value in self.settings["offsets"].items()}
                dataset = load_dataset(self.settings, calibration, provisional)
                raw = min(dataset.datas.values(), key=lambda d: abs(d.metadata["sample_phi_deg"]))
                self.figures(raw.plot_data(), "00_raw_data")
                self.state["raw_image_nominal_phi_deg"] = float(raw.metadata["sample_phi_deg"])
                apply_intensity_corrections(dataset, self.settings)
                phi = self.estimate_phi(dataset)
                del dataset
                if stage == "phi":
                    return self.finish_stage()
                # This fresh load is mandatory after phi estimation/acceptance.
                provisional["sample_phi_offset_deg"] = phi
                dataset = load_dataset(self.settings, calibration, provisional)
                apply_intensity_corrections(dataset, self.settings)
                self.event("Reloaded raw data and reapplied intensity corrections with accepted phi offset")
                self.estimate_chi(dataset)
                if stage == "chi":
                    return self.finish_stage()
                self.estimate_omega(dataset)
                if stage == "omega":
                    return self.finish_stage()
                dataset.apply_rotation_correction_all_data(verbose=False)
                for data in dataset.datas.values():
                    data.mask |= ~np.isfinite(data.image)
                rows = self.rows(dataset)
                reductions = []
                for row in rows:
                    geometry, integrated = self.preview(dataset, row)
                    items, proposal = self.choose_slices(integrated, row)
                    reductions.append((row, geometry, integrated, items, proposal))
                if stage == "preview":
                    write_json(self.directory / "row_reductions.json", [
                        {"row": row, "integration": geometry, "slices": proposal}
                        for row, geometry, _, _, proposal in reductions])
                    return self.finish_stage()
                self.export(dataset, reductions)
                self.state["status"] = "complete"
                self.save()
            except NeedsReview as error:
                self.state["status"] = "needs_review"
                self.archive_exports("Current reduction requires review")
                self.event(str(error))
                self.save()
            except (ValueError, KeyError, IndexError, RuntimeError, OSError) as error:
                self.state["status"] = "needs_review"
                self.archive_exports("Current reduction requires review")
                self.event(f"Reduction error ({type(error).__name__}): {error}")
                self.save()
        return {"sample": self.settings["sample_name"], "status": self.state["status"],
                "directory": str(self.directory)}

    def finish_stage(self):
        self.state["status"] = "stage_complete"
        self.save()
        return {"sample": self.settings["sample_name"], "status": "stage_complete",
                "directory": str(self.directory)}


def run_batch(settings, stage="all", resume=None, accept=(), sample=None):
    samples = list(sample_settings(settings))
    if sample is not None:
        samples = [cfg for cfg in samples if cfg["sample_name"] == sample]
        check(samples, f"Sample {sample} is not in settings")
    mapping = {}
    if resume and Path(resume).is_file():
        previous = read_json(resume)
        mapping = {item["sample"]: item["directory"] for item in previous["results"]}
    elif resume:
        check(len(samples) == 1, "Resume a batch using its batch JSON manifest")
        mapping[samples[0]["sample_name"]] = str(Path(resume).resolve())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    results = []
    for cfg in samples:
        root = Path(cfg["output"]["root"] or Path(cfg["input"]["data_directory"] or ".") / "Results")
        default_directory = root / slug(cfg["experiment"]) / slug(cfg["sample_name"]) / run_id
        directory = Path(mapping.get(cfg["sample_name"], default_directory))
        try:
            run = ReductionRun(cfg, directory, accept)
            results.append(run.run(stage))
        except (NeedsReview, OSError, ValueError) as error:
            directory.mkdir(parents=True, exist_ok=True)
            result = {"sample": cfg["sample_name"], "status": "needs_review",
                      "directory": str(directory), "reason": str(error)}
            write_json(directory / "failure.json", result)
            print(f"{cfg['sample_name']}: {error}")
            results.append(result)
    root = Path(settings["output"]["root"] or Path(settings["input"]["data_directory"] or ".") / "Results")
    manifest = root / slug(settings["experiment"]) / f"batch_{run_id}.json"
    write_json(manifest, {"results": results, "stage": stage})
    print(f"Batch manifest: {manifest}")
    return results
