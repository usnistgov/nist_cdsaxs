from __future__ import annotations

import numpy as np

from .Trapezoid_model import TrapezoidModelArray


class TrapezoidModelArrayDean(TrapezoidModelArray):
    """
    Dean optimization variant of the trapezoid model.

    The public constructor and fitting workflow stay aligned with
    `TrapezoidModelArray`. The first optimization stage is CPU-only and focuses on:

    - caching dataset invariants
    - reducing per-call parameter parsing overhead
    - reducing temporary-object churn in scalar and batched objective paths
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dean_cache = {}

    def process_imported_data(self):
        super().process_imported_data()
        self._refresh_dean_dataset_cache()

    def _refresh_dean_dataset_cache(self):
        """Refresh cached dataset invariants for the current loaded data."""
        if not hasattr(self, "Intensity") or self.Intensity is None:
            return

        intensity = np.asarray(self.Intensity, dtype=float)
        qx = np.asarray(self.Qx, dtype=float)
        qz = np.asarray(self.Qz, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_intensity = np.log(intensity)
        log_intensity = np.nan_to_num(log_intensity, copy=False)

        self._dean_cache["intensity_shape"] = intensity.shape
        self._dean_cache["log_intensity"] = log_intensity
        self._dean_cache["qsq"] = np.square(qx) + np.square(qz)

    def _get_dean_qsq(self, Qx=None, Qz=None):
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
            cached_qsq = self._dean_cache.get("qsq")
            if cached_qsq is None:
                self._refresh_dean_dataset_cache()
                cached_qsq = self._dean_cache.get("qsq")
            if cached_qsq is not None:
                return cached_qsq

        return np.square(np.asarray(Qx, dtype=float)) + np.square(np.asarray(Qz, dtype=float))

    def _get_dean_param_plan(self, param_names):
        key = tuple(param_names)
        plan = self._dean_cache.get(("param_plan", key))
        if plan is not None:
            return plan

        plan = []
        for name in key:
            if name.startswith("trap_"):
                _, trap_idx_str, param_type = name.split("_", 2)
                trap_idx = int(trap_idx_str)
                plan.append(("trap", trap_idx, 0 if param_type == "width" else 1))
            elif name.startswith("sld_"):
                plan.append(("sld", int(name.split("_", 1)[1]), None))
            elif name.startswith("Bk_"):
                plan.append(("bk_column", int(name.split("_", 1)[1]), None))
            elif name == "Bk":
                plan.append(("bk_scalar", None, None))
            elif name == "DW":
                plan.append(("dw", None, None))
            elif name == "I0":
                plan.append(("i0", None, None))
            else:
                plan.append(("passthrough", name, None))

        self._dean_cache[("param_plan", key)] = plan
        return plan

    def _normalize_candidate_matrix(self, values, n_params):
        """Normalize batched candidate input to shape ``(S, D)``."""
        if values.shape[1] == n_params:
            return values
        if values.shape[0] == n_params:
            return np.ascontiguousarray(values.T)
        raise ValueError(f"Parameter count mismatch: got {values.shape}, expected (*,{n_params}) or ({n_params},*)")

    def _map_scalar_candidate(self, values, plan, intensity):
        """Map one parameter vector onto trapezoid, SLD, DW, I0, and background state."""
        base_par = np.asarray(self.PAR, dtype=float)
        base_slds = (
            np.asarray(self.sld_values, dtype=float)
            if hasattr(self, "sld_values")
            else np.ones(self.layers, dtype=float)
        )

        temp_par = np.array(base_par, copy=True)
        temp_slds = np.array(base_slds, copy=True)
        temp_dw = float(self.DW)
        temp_i0 = float(self.I0)
        temp_bk = np.array(self.Bk, copy=True) if isinstance(self.Bk, np.ndarray) else float(self.Bk)

        for idx, spec in enumerate(plan):
            kind, arg1, arg2 = spec
            value = values[idx]
            if kind == "trap":
                temp_par[arg1, arg2] = value
            elif kind == "sld":
                temp_slds[arg1] = value
            elif kind == "bk_column":
                if not isinstance(temp_bk, np.ndarray):
                    temp_bk = np.full(intensity.shape[1], temp_bk, dtype=float)
                temp_bk[arg1] = value
            elif kind == "bk_scalar":
                temp_bk = value
            elif kind == "dw":
                temp_dw = value
            elif kind == "i0":
                temp_i0 = value

        return temp_par, temp_slds, temp_dw, temp_i0, temp_bk

    def _map_batched_candidates(self, candidates, plan, intensity):
        """Map batched candidates onto arrays needed for vectorized trapezoid evaluation."""
        base_par = np.asarray(self.PAR, dtype=float)
        base_slds = (
            np.asarray(self.sld_values, dtype=float)
            if hasattr(self, "sld_values")
            else np.ones(self.layers, dtype=float)
        )

        n_candidates = candidates.shape[0]
        batched_par = np.broadcast_to(base_par, (n_candidates,) + base_par.shape).copy()
        batched_slds = np.broadcast_to(base_slds, (n_candidates, base_slds.shape[0])).copy()
        dw_arr = np.full(n_candidates, float(self.DW), dtype=float)
        i0_arr = np.full(n_candidates, float(self.I0), dtype=float)

        if isinstance(self.Bk, np.ndarray):
            bk_arr = np.broadcast_to(np.asarray(self.Bk, dtype=float), (n_candidates, self.Bk.shape[0])).copy()
        else:
            bk_arr = np.full((n_candidates, 1), float(self.Bk), dtype=float)

        for idx, spec in enumerate(plan):
            kind, arg1, arg2 = spec
            column = candidates[:, idx]
            if kind == "trap":
                batched_par[:, arg1, arg2] = column
            elif kind == "sld":
                batched_slds[:, arg1] = column
            elif kind == "bk_column":
                if bk_arr.shape[1] == 1:
                    bk_arr = np.broadcast_to(bk_arr, (n_candidates, intensity.shape[1])).copy()
                bk_arr[:, arg1] = column
            elif kind == "bk_scalar":
                if bk_arr.shape[1] == 1:
                    bk_arr[:, 0] = column
                else:
                    bk_arr[:] = column[:, None]
            elif kind == "dw":
                dw_arr = column
            elif kind == "i0":
                i0_arr = column

        return batched_par, batched_slds, dw_arr, i0_arr, bk_arr

    def GF_calc(self, SimInt, Intensity=None):
        """
        Cached GF calculation for self-loaded datasets, with fallback behavior preserved.
        """
        try:
            if Intensity is None:
                Intensity = self.Intensity

            if SimInt is None:
                raise ValueError("SimInt must not be None")

            sim = np.asarray(SimInt, dtype=float)

            use_cached_intensity = (
                Intensity is self.Intensity
                or (
                    hasattr(self, "Intensity")
                    and np.asarray(Intensity).shape == np.asarray(self.Intensity).shape
                    and np.may_share_memory(np.asarray(Intensity), np.asarray(self.Intensity))
                )
            )

            if use_cached_intensity:
                log_intensity = self._dean_cache.get("log_intensity")
                if log_intensity is None:
                    self._refresh_dean_dataset_cache()
                    log_intensity = self._dean_cache.get("log_intensity")
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_intensity = np.log(np.asarray(Intensity, dtype=float))
                log_intensity = np.nan_to_num(log_intensity, copy=False)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_sim = np.log(sim)
            log_sim = np.nan_to_num(log_sim, copy=False)

            if sim.shape != np.asarray(Intensity).shape:
                base = log_intensity
                while base.ndim < log_sim.ndim:
                    base = base[np.newaxis, ...]
                log_intensity = np.broadcast_to(base, log_sim.shape)

            gf_matrix = np.abs(log_intensity - log_sim)
            gf_matrix = np.nan_to_num(gf_matrix, copy=False)

            if gf_matrix.ndim >= 2:
                return np.sum(gf_matrix, axis=tuple(range(-2, 0)))
            return np.sum(gf_matrix)

        except Exception:
            return super().GF_calc(SimInt, Intensity)

    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None, use_cupy=False):
        """
        Dean CPU optimization path with reduced parameter parsing and cached invariants.
        """
        try:
            if optimization_values is None:
                raise TypeError("optimization_values must not be None")

            if param_names is None:
                if not hasattr(self, "param_names"):
                    raise AttributeError("Missing required attribute: param_names")
                param_names = self.param_names

            if Intensity is None:
                Intensity = self.Intensity
            if Qx is None:
                Qx = self.Qx
            if Qz is None:
                Qz = self.Qz

            values = np.asarray(optimization_values, dtype=float)
            plan = self._get_dean_param_plan(param_names)
            qsq = self._get_dean_qsq(Qx, Qz)

            if values.ndim == 1:
                temp_par, temp_slds, temp_dw, temp_i0, temp_bk = self._map_scalar_candidate(values, plan, Intensity)

                coord = self.SymCoordAssign(temp_par, self.layers, sld_values=temp_slds)
                if coord is None:
                    return float("inf")

                form = self.FreeFormTrapezoid(coord, self.layers, Qx, Qz)
                if form is None:
                    return float("inf")

                debye = np.exp(-0.5 * qsq * temp_dw * temp_dw)
                form_mag = np.abs(form) * debye
                intensity_base = np.square(form_mag) * temp_i0
                sim_int = intensity_base + (temp_bk[np.newaxis, :] if isinstance(temp_bk, np.ndarray) else temp_bk)
                return self.GF_calc(sim_int, Intensity)

            if values.ndim == 2:
                n_params = len(param_names)
                candidates = self._normalize_candidate_matrix(values, n_params)
                batched_par, batched_slds, dw_arr, i0_arr, bk_arr = self._map_batched_candidates(
                    candidates,
                    plan,
                    Intensity,
                )
                n_candidates = candidates.shape[0]

                coord_b = self.SymCoordAssign(batched_par, self.layers, sld_values=batched_slds)
                if coord_b is None:
                    return np.full(n_candidates, float("inf"))

                form_b = self.FreeFormTrapezoid(coord_b, self.layers, Qx, Qz)
                if form_b is None:
                    return np.full(n_candidates, float("inf"))

                debye_b = np.exp(-0.5 * qsq[None, :, :] * np.square(dw_arr)[:, None, None])
                form_mag_b = np.abs(form_b) * debye_b
                intensity_base_b = np.square(form_mag_b) * i0_arr[:, None, None]

                if bk_arr.shape[1] == intensity_base_b.shape[2]:
                    sim_int_b = intensity_base_b + bk_arr[:, None, :]
                elif bk_arr.shape[1] == 1:
                    sim_int_b = intensity_base_b + bk_arr[:, 0][:, None, None]
                else:
                    raise ValueError("Background array length mismatch in Dean batched path")

                return self.GF_calc(sim_int_b, Intensity)

            raise ValueError("optimization_values must be a 1D or 2D array")

        except Exception:
            return super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)

    def _trapezoid_optimization_wrapper(self, optimization_values):
        """
        Wrapper that preserves the current user-facing optimization API while routing to Dean objective code.
        """
        try:
            values = np.asarray(optimization_values, dtype=float)
            if hasattr(self, "param_names"):
                param_names = self.param_names
            elif hasattr(self, "mcmc_param_names"):
                param_names = self.mcmc_param_names
            else:
                param_names = list(self.model_params.get("optimization", {}).keys())

            if values.ndim == 1:
                return self.SimTrap_GF(values, param_names, self.Intensity, self.Qx, self.Qz)
            if values.ndim == 2:
                return self.SimTrap_GF(values, param_names, self.Intensity, self.Qx, self.Qz)
            raise ValueError("optimization_values must be a 1D or 2D numpy array")
        except Exception:
            return super()._trapezoid_optimization_wrapper(optimization_values)


TrapezoidModelDean = TrapezoidModelArrayDean
