from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np

from .SiGe_model import SiGeModelArray


class SiGeModelArray_vectorized(SiGeModelArray):
    """
    Initial CPU-vectorized SiGe variant.

    The first implementation milestone fixes optimizer-facing batched objective
    semantics while preserving the incumbent scalar SiGe physics as the oracle.
    Multi-candidate evaluation normalizes the incoming batch shape and then
    evaluates the incumbent scalar objective candidate-by-candidate.
    """

    _TRANSIENT_STATE_ATTRS = (
        "Coord_curved",
        "curved_sidewall_trapezoids",
        "curved_sidewall_trapezoid_rows",
    )
    _DIRECT_LAYER = "direct"
    _ELLIPSE_LAYER = "ellipse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized_cache = {}

    def process_imported_data(self):
        super().process_imported_data()
        self._refresh_vectorized_dataset_cache()

    def _refresh_vectorized_dataset_cache(self):
        """Refresh dataset invariants used by the batched objective path."""
        if not hasattr(self, "Intensity") or self.Intensity is None:
            return

        intensity = np.asarray(self.Intensity, dtype=float)
        qx = np.asarray(self.Qx, dtype=float)
        qz = np.asarray(self.Qz, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_intensity = np.log(intensity)
        log_intensity = np.nan_to_num(log_intensity, copy=False)

        self._vectorized_cache["intensity_shape"] = intensity.shape
        self._vectorized_cache["log_intensity"] = log_intensity
        self._vectorized_cache["qsq"] = np.square(qx) + np.square(qz)

    def _get_vectorized_qsq(self, Qx=None, Qz=None):
        if (
            Qx is None
            or Qz is None
            or (Qx is self.Qx and Qz is self.Qz)
            or (
                hasattr(self, "Qx")
                and hasattr(self, "Qz")
                and np.may_share_memory(np.asarray(Qx), np.asarray(self.Qx))
                and np.may_share_memory(np.asarray(Qz), np.asarray(self.Qz))
            )
        ):
            cached_qsq = self._vectorized_cache.get("qsq")
            if cached_qsq is None:
                self._refresh_vectorized_dataset_cache()
                cached_qsq = self._vectorized_cache.get("qsq")
            if cached_qsq is not None:
                return cached_qsq
        return np.square(np.asarray(Qx, dtype=float)) + np.square(np.asarray(Qz, dtype=float))

    def _parse_batched_target(self, name):
        """Parse one supported optimization/constraint target into a batched-array descriptor."""
        if name.startswith("trap_"):
            parts = name.split("_")
            if len(parts) < 3:
                raise ValueError(f"Invalid trapezoid parameter name: {name}")
            return ("trap", int(parts[1]), "_".join(parts[2:]))
        if name.startswith("sld_"):
            return ("sld", int(name.split("_", 1)[1]), None)
        if name.startswith("Bk_"):
            return ("bk_column", int(name.split("_", 1)[1]), None)
        if name == "Bk":
            return ("bk_scalar", None, None)
        if name == "DW":
            return ("dw", None, None)
        if name == "I0":
            return ("i0", None, None)
        raise ValueError(f"Unsupported batched target: {name}")

    def _compile_batched_path_spec(self, param_names):
        """Compile the fixed-topology batched execution spec for the current realistic workflow."""
        key = tuple(param_names)
        cached = self._vectorized_cache.get(("batched_spec", key))
        if cached is not None:
            return cached

        model_params = copy.deepcopy(self.model_params)
        design_traps = model_params.get("design_trapezoids")
        if not isinstance(design_traps, list):
            raise ValueError("Batched SiGe path requires design_trapezoids in model_params")

        design_slds = model_params.get("design_slds", model_params.get("slds"))
        if design_slds is None:
            design_slds = [1.0] * len(design_traps)
        design_slds = np.asarray(design_slds, dtype=float)

        if any(str(trap.get("Layer_Type", "")).strip().lower() == "curved_sides" for trap in design_traps):
            raise ValueError("Batched SiGe path does not yet support curved_sides layers")

        layer_types = []
        segment_starts = []
        segment_counts = []
        ellipse_profiles = {}
        expanded_count = 0

        for design_idx, trap in enumerate(design_traps):
            layer_type = str(trap.get("Layer_Type", "")).strip().lower()
            if layer_type == self._ELLIPSE_LAYER:
                count = int(trap.get("num_layers", 1))
                if count < 1:
                    count = 1
                u = np.linspace(0.0, 1.0, count + 1, dtype=float)
                ellipse_profiles[design_idx] = {
                    "count": count,
                    "u": u,
                    "indent_profile": np.sqrt(np.maximum(0.0, 1.0 - np.square(2.0 * u - 1.0))),
                }
                layer_types.append(self._ELLIPSE_LAYER)
            else:
                count = 1
                layer_types.append(self._DIRECT_LAYER)

            segment_starts.append(expanded_count)
            segment_counts.append(count)
            expanded_count += count

        base_widths = np.array([float(trap.get("width", 0.0)) for trap in design_traps], dtype=float)
        base_heights = np.array([float(trap.get("height", 0.0)) for trap in design_traps], dtype=float)
        base_twidths = np.array(
            [
                np.nan if trap.get("twidth", None) is None else float(trap.get("twidth"))
                for trap in design_traps
            ],
            dtype=float,
        )
        base_depths = np.array([float(trap.get("depth", 0.0) or 0.0) for trap in design_traps], dtype=float)

        param_plan = [self._parse_batched_target(name) for name in key]

        constraint_specs = []
        for rule in model_params.get("constraints", []):
            if not isinstance(rule, dict):
                continue
            lhs = rule.get("lhs")
            rhs = rule.get("rhs")
            op = str(rule.get("op", "")).strip()
            if lhs is None or rhs is None or not op:
                continue
            constraint_specs.append(
                {
                    "lhs": self._parse_batched_target(lhs),
                    "rhs": self._parse_batched_target(rhs),
                    "op": op,
                    "offset": float(rule.get("offset", 0.0) or 0.0),
                }
            )

        spec = {
            "param_names": key,
            "param_plan": tuple(param_plan),
            "constraint_specs": tuple(constraint_specs),
            "layer_types": tuple(layer_types),
            "segment_starts": np.asarray(segment_starts, dtype=int),
            "segment_counts": np.asarray(segment_counts, dtype=int),
            "ellipse_profiles": ellipse_profiles,
            "expanded_count": int(expanded_count),
            "base_widths": base_widths,
            "base_heights": base_heights,
            "base_twidths": base_twidths,
            "base_depths": base_depths,
            "base_slds": np.asarray(design_slds, dtype=float),
            "base_dw": float(model_params.get("DW", self.DW)),
            "base_i0": float(model_params.get("I0", self.I0)),
            "base_bk": np.asarray(self.Bk, dtype=float).copy()
            if isinstance(self.Bk, np.ndarray)
            else np.array([float(self.Bk)], dtype=float),
            "design_count": int(len(design_traps)),
        }
        self._vectorized_cache[("batched_spec", key)] = spec
        return spec

    def _batched_state_get(self, state, target):
        kind, index, field = target
        if kind == "trap":
            return state[field][:, index]
        if kind == "sld":
            return state["slds"][:, index]
        if kind == "dw":
            return state["dw"]
        if kind == "i0":
            return state["i0"]
        if kind == "bk_scalar":
            if state["bk"].shape[1] == 1:
                return state["bk"][:, 0]
            raise ValueError("Batched scalar background target is ambiguous when bk is column-resolved")
        if kind == "bk_column":
            if state["bk"].shape[1] <= index:
                raise IndexError(f"Background column index {index} out of range")
            return state["bk"][:, index]
        raise ValueError(f"Unsupported batched state target: {target}")

    def _batched_state_set(self, state, target, values):
        kind, index, field = target
        values = np.asarray(values, dtype=float)
        if kind == "trap":
            state[field][:, index] = values
            return
        if kind == "sld":
            state["slds"][:, index] = values
            return
        if kind == "dw":
            state["dw"][:] = values
            return
        if kind == "i0":
            state["i0"][:] = values
            return
        if kind == "bk_scalar":
            if state["bk"].shape[1] == 1:
                state["bk"][:, 0] = values
            else:
                state["bk"][:] = values[:, None]
            return
        if kind == "bk_column":
            if state["bk"].shape[1] == 1:
                state["bk"] = np.broadcast_to(state["bk"], (state["bk"].shape[0], self.Intensity.shape[1])).copy()
            state["bk"][:, index] = values
            return
        raise ValueError(f"Unsupported batched state target: {target}")

    def _initialize_batched_state(self, candidates, spec):
        """Build dense batched design/global arrays for one candidate batch."""
        batch_size = int(candidates.shape[0])
        design_count = int(spec["design_count"])
        width = np.broadcast_to(spec["base_widths"], (batch_size, design_count)).copy()
        height = np.broadcast_to(spec["base_heights"], (batch_size, design_count)).copy()
        twidth = np.broadcast_to(spec["base_twidths"], (batch_size, design_count)).copy()
        depth = np.broadcast_to(spec["base_depths"], (batch_size, design_count)).copy()
        slds = np.broadcast_to(spec["base_slds"], (batch_size, design_count)).copy()
        dw = np.full(batch_size, spec["base_dw"], dtype=float)
        i0 = np.full(batch_size, spec["base_i0"], dtype=float)
        bk = np.broadcast_to(spec["base_bk"], (batch_size, spec["base_bk"].shape[0])).copy()

        state = {
            "width": width,
            "height": height,
            "twidth": twidth,
            "depth": depth,
            "slds": slds,
            "dw": dw,
            "i0": i0,
            "bk": bk,
        }

        for column_index, target in enumerate(spec["param_plan"]):
            self._batched_state_set(state, target, candidates[:, column_index])

        return state

    def _apply_batched_constraints(self, state, spec):
        """Apply compiled equality/inequality constraints over the dense batched design arrays."""
        eq_rules = [rule for rule in spec["constraint_specs"] if rule["op"] == "=="]
        le_rules = [rule for rule in spec["constraint_specs"] if rule["op"] in ("<=", "<")]

        for _ in range(10):
            for rule in eq_rules:
                rhs_val = self._batched_state_get(state, rule["rhs"]) + rule["offset"]
                self._batched_state_set(state, rule["lhs"], rhs_val)

        for rule in le_rules:
            rhs_val = self._batched_state_get(state, rule["rhs"]) + rule["offset"]
            lhs_val = self._batched_state_get(state, rule["lhs"])
            self._batched_state_set(state, rule["lhs"], np.minimum(lhs_val, rhs_val))

        return state

    def _expand_design_batch(self, state, spec):
        """Expand the fixed-topology design batch into dense batched simulation arrays."""
        batch_size = state["width"].shape[0]
        expanded_count = int(spec["expanded_count"])
        expanded_width = np.empty((batch_size, expanded_count), dtype=float)
        expanded_height = np.empty((batch_size, expanded_count), dtype=float)
        expanded_twidth = np.empty((batch_size, expanded_count), dtype=float)
        expanded_slds = np.empty((batch_size, expanded_count), dtype=float)

        for design_index, layer_type in enumerate(spec["layer_types"]):
            start = int(spec["segment_starts"][design_index])
            count = int(spec["segment_counts"][design_index])

            if layer_type == self._DIRECT_LAYER:
                expanded_width[:, start] = state["width"][:, design_index]
                expanded_height[:, start] = state["height"][:, design_index]
                expanded_twidth[:, start] = state["twidth"][:, design_index]
                expanded_slds[:, start] = state["slds"][:, design_index]
                continue

            ellipse_profile = spec["ellipse_profiles"][design_index]
            u = ellipse_profile["u"][None, :]
            indent_profile = ellipse_profile["indent_profile"][None, :]

            width0 = state["width"][:, design_index : design_index + 1]
            width1 = np.where(
                np.isnan(state["twidth"][:, design_index : design_index + 1]),
                width0,
                state["twidth"][:, design_index : design_index + 1],
            )
            height_val = state["height"][:, design_index : design_index + 1]
            depth_val = state["depth"][:, design_index : design_index + 1]

            base_widths = width0 + (width1 - width0) * u
            ellipse_widths = np.maximum(base_widths - 2.0 * depth_val * indent_profile, 1e-12)

            expanded_width[:, start : start + count] = ellipse_widths[:, :-1]
            expanded_height[:, start : start + count] = height_val / float(count)
            expanded_twidth[:, start : start + count] = ellipse_widths[:, 1:]
            expanded_slds[:, start : start + count] = state["slds"][:, design_index : design_index + 1]

        return expanded_width, expanded_height, expanded_twidth, expanded_slds

    def _batched_form_factor(self, expanded_width, expanded_height, expanded_twidth, expanded_slds, Qx, Qz):
        """Evaluate the center-stack SiGe form factor over a full candidate batch."""
        batch_size, expanded_count = expanded_width.shape
        qx = np.asarray(Qx, dtype=float)
        qz = np.asarray(Qz, dtype=float)

        center = 0.5 * expanded_width[:, [0]]
        next_width = np.concatenate([expanded_width[:, 1:], expanded_width[:, -1:]], axis=1)
        top_width = np.where(np.isnan(expanded_twidth), next_width, expanded_twidth)

        x1 = center - 0.5 * expanded_width
        x4 = center + 0.5 * expanded_width
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = np.where(np.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = np.where(np.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = np.cumsum(expanded_height, axis=1)
        h1 = h2 - expanded_height

        qx_b = qx[None, :, :]
        qz_b = qz[None, :, :]
        form = np.zeros((batch_size, qx.shape[0], qx.shape[1]), dtype=np.complex128)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_qx = 1j / qx_b
            for layer_index in range(expanded_count):
                h1_b = h1[:, layer_index][:, None, None]
                h2_b = h2[:, layer_index][:, None, None]
                x1_b = x1[:, layer_index][:, None, None]
                x2_b = x2[:, layer_index][:, None, None]
                x3_b = x3[:, layer_index][:, None, None]
                x4_b = x4[:, layer_index][:, None, None]
                sl_b = (expanded_height[:, layer_index] / (x2[:, layer_index] - x1[:, layer_index]))[:, None, None]
                sr_b = (-expanded_height[:, layer_index] / (x4[:, layer_index] - x3[:, layer_index]))[:, None, None]
                sld_b = expanded_slds[:, layer_index][:, None, None]

                a1 = (
                    np.exp(1j * qx_b * ((h1_b - sr_b * x4_b) / sr_b)) / (qx_b / sr_b + qz_b)
                ) * (
                    np.exp(-1j * h2_b * (qx_b / sr_b + qz_b))
                    - np.exp(-1j * h1_b * (qx_b / sr_b + qz_b))
                )
                a2 = (
                    np.exp(1j * qx_b * ((h1_b - sl_b * x1_b) / sl_b)) / (qx_b / sl_b + qz_b)
                ) * (
                    np.exp(-1j * h2_b * (qx_b / sl_b + qz_b))
                    - np.exp(-1j * h1_b * (qx_b / sl_b + qz_b))
                )
                form = form + inv_qx * (a1 - a2) * sld_b

        return form

    def _batched_gf_from_simint(self, sim_intensity, Intensity=None):
        """Compute GF for a batched simulated intensity stack using cached log(Intensity) when possible."""
        if Intensity is None:
            Intensity = self.Intensity

        use_cached = (
            Intensity is self.Intensity
            or (
                hasattr(self, "Intensity")
                and np.asarray(Intensity).shape == np.asarray(self.Intensity).shape
                and np.may_share_memory(np.asarray(Intensity), np.asarray(self.Intensity))
            )
        )

        if use_cached:
            log_intensity = self._vectorized_cache.get("log_intensity")
            if log_intensity is None:
                self._refresh_vectorized_dataset_cache()
                log_intensity = self._vectorized_cache.get("log_intensity")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_intensity = np.log(np.asarray(Intensity, dtype=float))
            log_intensity = np.nan_to_num(log_intensity, copy=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_sim = np.log(np.asarray(sim_intensity, dtype=float))
        log_sim = np.nan_to_num(log_sim, copy=False)

        base = log_intensity
        while base.ndim < log_sim.ndim:
            base = base[np.newaxis, ...]
        base = np.broadcast_to(base, log_sim.shape)

        gf_matrix = np.abs(base - log_sim)
        gf_matrix = np.nan_to_num(gf_matrix, copy=False)
        return np.sum(gf_matrix, axis=tuple(range(-2, 0)))

    def _supports_batched_path(self, param_names) -> bool:
        """Return True when the current model state matches the realistic fully batched workflow."""
        try:
            self._compile_batched_path_spec(param_names)
            return True
        except Exception:
            return False

    def _normalize_candidate_matrix(self, optimization_values, param_names) -> np.ndarray:
        """Normalize batched candidate input to shape ``(S, D)``."""
        values = np.asarray(optimization_values, dtype=float)
        if values.ndim != 2:
            raise ValueError("optimization_values must be a 2D array for batched normalization")

        n_params = len(param_names)
        if values.shape[1] == n_params:
            return np.ascontiguousarray(values)
        if values.shape[0] == n_params:
            return np.ascontiguousarray(values.T)

        raise ValueError(
            f"Parameter count mismatch: got {values.shape}, expected (*,{n_params}) or ({n_params},*)"
        )

    @contextmanager
    def _preserve_transient_state(self):
        """Keep optimizer-only objective evaluation from mutating inspection state."""
        snapshot = {}
        missing = []

        for attr_name in self._TRANSIENT_STATE_ATTRS:
            if hasattr(self, attr_name):
                snapshot[attr_name] = copy.deepcopy(getattr(self, attr_name))
            else:
                missing.append(attr_name)

        try:
            yield
        finally:
            for attr_name, value in snapshot.items():
                setattr(self, attr_name, value)
            for attr_name in missing:
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

    def _evaluate_scalar_oracle(self, vector, param_names, Intensity, Qx, Qz) -> float:
        """Evaluate one candidate with the incumbent scalar SiGe objective."""
        candidate = np.asarray(vector, dtype=float)
        with self._preserve_transient_state():
            value = SiGeModelArray.SimTrap_GF(self, candidate, param_names, Intensity, Qx, Qz)
        return float(value)

    def _evaluate_batched_realistic_path(self, candidates, param_names, Intensity, Qx, Qz) -> np.ndarray:
        """Evaluate the fully batched realistic ellipse-stack SiGe objective path."""
        spec = self._compile_batched_path_spec(param_names)
        state = self._initialize_batched_state(candidates, spec)
        state = self._apply_batched_constraints(state, spec)
        expanded_width, expanded_height, expanded_twidth, expanded_slds = self._expand_design_batch(state, spec)

        form = self._batched_form_factor(expanded_width, expanded_height, expanded_twidth, expanded_slds, Qx, Qz)
        qsq = self._get_vectorized_qsq(Qx, Qz)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            debye = np.exp(-0.5 * qsq[None, :, :] * np.square(state["dw"])[:, None, None])
            formfactor = np.abs(form * debye)
            intensity_base = np.square(formfactor) * state["i0"][:, None, None]

        if state["bk"].shape[1] == intensity_base.shape[2]:
            sim_intensity = intensity_base + state["bk"][:, None, :]
        elif state["bk"].shape[1] == 1:
            sim_intensity = intensity_base + state["bk"][:, 0][:, None, None]
        else:
            raise ValueError("Background array length mismatch in batched SiGe path")

        return np.asarray(self._batched_gf_from_simint(sim_intensity, Intensity), dtype=float)

    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None):
        """
        Evaluate the SiGe objective with correct 1D and 2D optimizer semantics.

        1D input returns one scalar objective value.
        2D input accepts either ``(S, D)`` or ``(D, S)`` and returns an ``(S,)``
        vector of objective values.
        """
        try:
            values = np.asarray(optimization_values, dtype=float)

            if param_names is None:
                param_names = self._resolve_active_parameter_names(values)
            else:
                param_names = list(param_names)

            if Intensity is None:
                Intensity = self.Intensity
            if Qx is None:
                Qx = self.Qx
            if Qz is None:
                Qz = self.Qz

            if values.ndim == 1:
                if values.shape[0] != len(param_names):
                    raise ValueError(
                        f"Number of optimization values ({values.shape[0]}) must match "
                        f"number of parameter names ({len(param_names)})"
                    )
                if self._supports_batched_path(param_names):
                    batched = self._evaluate_batched_realistic_path(values[None, :], param_names, Intensity, Qx, Qz)
                    return float(batched[0])
                return self._evaluate_scalar_oracle(values, param_names, Intensity, Qx, Qz)

            if values.ndim == 2:
                candidates = self._normalize_candidate_matrix(values, param_names)
                if self._supports_batched_path(param_names):
                    return self._evaluate_batched_realistic_path(candidates, param_names, Intensity, Qx, Qz)

                results = np.empty(candidates.shape[0], dtype=float)
                for idx, candidate in enumerate(candidates):
                    results[idx] = self._evaluate_scalar_oracle(candidate, param_names, Intensity, Qx, Qz)
                return results

            raise ValueError("optimization_values must be a 1D or 2D array")

        except Exception:
            if np.asarray(optimization_values).ndim == 2:
                try:
                    if param_names is None:
                        resolved = self._resolve_active_parameter_names(np.asarray(optimization_values))
                    else:
                        resolved = list(param_names)
                    candidate_count = self._normalize_candidate_matrix(
                        np.asarray(optimization_values, dtype=float),
                        resolved,
                    ).shape[0]
                except Exception:
                    candidate_count = int(np.asarray(optimization_values).shape[-1])
                return np.full(candidate_count, float("inf"), dtype=float)
            return float("inf")

    def _trapezoid_optimization_wrapper(self, optimization_values):
        """Optimizer-facing wrapper that preserves current public SiGe workflow semantics."""
        try:
            values = np.asarray(optimization_values, dtype=float)
            param_names = self._resolve_active_parameter_names(values)
            return self.SimTrap_GF(values, param_names, self.Intensity, self.Qx, self.Qz)
        except Exception:
            values = np.asarray(optimization_values)
            if values.ndim == 2:
                try:
                    param_names = self._resolve_active_parameter_names(values)
                    candidate_count = self._normalize_candidate_matrix(values, param_names).shape[0]
                except Exception:
                    candidate_count = int(values.shape[-1])
                return np.full(candidate_count, float("inf"), dtype=float)
            return float("inf")


SiGeModel_vectorized = SiGeModelArray_vectorized
