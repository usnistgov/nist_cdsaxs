from __future__ import annotations

import contextlib
import numpy as np

from .Trapezoid_model_dean import TrapezoidModelArrayDean

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

_CUPY_ERRSTATE_SUPPORTED = False
if cp is not None:  # pragma: no branch - one-time capability probe
    _cupy_errstate = getattr(cp, "errstate", None)
    if _cupy_errstate is not None:
        try:
            with _cupy_errstate(divide="ignore", invalid="ignore"):
                pass
        except Exception:
            _CUPY_ERRSTATE_SUPPORTED = False
        else:
            _CUPY_ERRSTATE_SUPPORTED = True


@contextlib.contextmanager
def _gpu_errstate():
    """
    CuPy 14 no longer exposes a working ``cp.errstate`` in this environment.

    The resident Dean GPU path is already guarded with explicit NaN cleanup, so
    if a usable CuPy errstate context is unavailable we continue without it
    rather than silently forcing a CPU fallback.
    """
    if cp is None or not _CUPY_ERRSTATE_SUPPORTED:
        yield
        return

    with cp.errstate(divide="ignore", invalid="ignore"):
        yield


if cp is not None:  # pragma: no branch - defined only when CuPy is available
    @cp.fuse()
    def _fused_batched_log_residual(form_abs_sq, qsq_b, dw_sq, i0_b, bk_b, log_intensity_b):
        sim_int = form_abs_sq * cp.exp(-qsq_b * dw_sq) * i0_b + bk_b
        return cp.abs(log_intensity_b - cp.log(sim_int))

    @cp.fuse()
    def _fused_trapezoid_layer_contribution(qx_b, qz_b, h1b, h2b, slb, srb, x1b, x4b, sld):
        a1 = (
            cp.exp(1j * qx_b * ((h1b - srb * x4b) / srb)) / (qx_b / srb + qz_b)
        ) * (
            cp.exp(-1j * h2b * (qx_b / srb + qz_b))
            - cp.exp(-1j * h1b * (qx_b / srb + qz_b))
        )
        a2 = (
            cp.exp(1j * qx_b * ((h1b - slb * x1b) / slb)) / (qx_b / slb + qz_b)
        ) * (
            cp.exp(-1j * h2b * (qx_b / slb + qz_b))
            - cp.exp(-1j * h1b * (qx_b / slb + qz_b))
        )
        return (1j / qx_b) * (a1 - a2) * sld


class TrapezoidModelArrayDeanGPU(TrapezoidModelArrayDean):
    """
    GPU-resident Dean trapezoid variant.

    This class keeps the public fitting workflow aligned with the existing trapezoid
    model while moving the dominant batched form-factor and GF work onto the GPU.
    The optimizer still lives on the CPU, so each objective call transfers only the
    candidate matrix in and the final GF vector out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dean_gpu_cache = {}
        self._dean_gpu_min_batch = 8
        self._dean_last_execution_path = None
        self._dean_last_gpu_exception = None

    def _mark_execution_path(self, path, exc=None):
        self._dean_last_execution_path = str(path)
        self._dean_last_gpu_exception = None if exc is None else f"{type(exc).__name__}: {exc}"

    def process_imported_data(self):
        super().process_imported_data()
        self._refresh_dean_gpu_dataset_cache()

    def _dean_gpu_enabled(self):
        return bool(getattr(self, "_freeform_use_cupy", False) and cp is not None)

    def _refresh_dean_gpu_dataset_cache(self):
        if cp is None or not hasattr(self, "Intensity") or self.Intensity is None:
            return

        if self._dean_cache.get("log_intensity") is None or self._dean_cache.get("qsq") is None:
            self._refresh_dean_dataset_cache()

        intensity_gpu = cp.asarray(np.asarray(self.Intensity, dtype=float))
        qx_gpu = cp.asarray(np.asarray(self.Qx, dtype=float))
        qz_gpu = cp.asarray(np.asarray(self.Qz, dtype=float))
        log_intensity_gpu = cp.asarray(np.asarray(self._dean_cache["log_intensity"], dtype=float))
        qsq_gpu = cp.asarray(np.asarray(self._dean_cache["qsq"], dtype=float))

        self._dean_gpu_cache["intensity_gpu"] = intensity_gpu
        self._dean_gpu_cache["qx_gpu"] = qx_gpu
        self._dean_gpu_cache["qz_gpu"] = qz_gpu
        self._dean_gpu_cache["qx_b_gpu"] = qx_gpu[None, :, :]
        self._dean_gpu_cache["qz_b_gpu"] = qz_gpu[None, :, :]
        self._dean_gpu_cache["log_intensity_gpu"] = log_intensity_gpu
        self._dean_gpu_cache["log_intensity_b_gpu"] = log_intensity_gpu[None, :, :]
        self._dean_gpu_cache["qsq_gpu"] = qsq_gpu
        self._dean_gpu_cache["qsq_b_gpu"] = qsq_gpu[None, :, :]

    def _get_dean_gpu_dataset_cache(self):
        if not self._dean_gpu_enabled():
            return None
        if "qx_gpu" not in self._dean_gpu_cache:
            self._refresh_dean_gpu_dataset_cache()
        return self._dean_gpu_cache

    def _free_form_trapezoid_gpu(self, coord_gpu, layers=None):
        cache = self._get_dean_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        layers = int(self.layers if layers is None else layers)
        qx_gpu = cache["qx_gpu"]
        qz_gpu = cache["qz_gpu"]

        with _gpu_errstate():
            if coord_gpu.ndim == 4:
                qx_b = cache["qx_b_gpu"]
                qz_b = cache["qz_b_gpu"]
                n_candidates = coord_gpu.shape[0]
                form = cp.zeros((n_candidates,) + qx_gpu.shape, dtype=cp.complex128)
                h1 = coord_gpu[:, 0, 3, 0]
                h2 = h1.copy()

                for i in range(layers):
                    h2 = h2 + coord_gpu[:, i, 2, 0]
                    if i > 0:
                        h1 = h1 + coord_gpu[:, i - 1, 2, 0]

                    x1 = coord_gpu[:, i, 0, 0]
                    x4 = coord_gpu[:, i, 1, 0]
                    x2 = coord_gpu[:, i + 1, 0, 0]
                    x3 = coord_gpu[:, i + 1, 1, 0]

                    x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
                    x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

                    height = coord_gpu[:, i, 2, 0]
                    sl = height / (x2 - x1)
                    sr = -height / (x4 - x3)

                    h1b = h1[:, None, None]
                    h2b = h2[:, None, None]
                    slb = sl[:, None, None]
                    srb = sr[:, None, None]
                    x1b = x1[:, None, None]
                    x4b = x4[:, None, None]
                    sld = coord_gpu[:, i, 4, 0][:, None, None]

                    form = form + _fused_trapezoid_layer_contribution(
                        qx_b,
                        qz_b,
                        h1b,
                        h2b,
                        slb,
                        srb,
                        x1b,
                        x4b,
                        sld,
                    )

                return cp.nan_to_num(form, copy=False)

            form = cp.zeros(qx_gpu.shape, dtype=cp.complex128)
            h1 = coord_gpu[0, 3, 0]
            h2 = h1

            for i in range(layers):
                h2 = h2 + coord_gpu[i, 2, 0]
                if i > 0:
                    h1 = h1 + coord_gpu[i - 1, 2, 0]

                x1 = coord_gpu[i, 0, 0]
                x4 = coord_gpu[i, 1, 0]
                x2 = coord_gpu[i + 1, 0, 0]
                x3 = coord_gpu[i + 1, 1, 0]

                x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
                x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

                height = coord_gpu[i, 2, 0]
                sl = height / (x2 - x1)
                sr = -height / (x4 - x3)

                form = form + _fused_trapezoid_layer_contribution(
                    qx_gpu,
                    qz_gpu,
                    h1,
                    h2,
                    sl,
                    sr,
                    x1,
                    x4,
                    coord_gpu[i, 4, 0],
                )

            return cp.nan_to_num(form, copy=False)

    def _gf_calc_gpu(self, sim_int_gpu):
        cache = self._get_dean_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        with _gpu_errstate():
            log_sim_gpu = cp.log(sim_int_gpu)
        log_sim_gpu = cp.nan_to_num(log_sim_gpu, copy=False)

        if log_sim_gpu.ndim == 2:
            diff = cp.abs(cache["log_intensity_gpu"] - log_sim_gpu)
            return cp.sum(cp.nan_to_num(diff, copy=False))

        diff = cp.abs(cache["log_intensity_b_gpu"] - log_sim_gpu)
        diff = cp.nan_to_num(diff, copy=False)
        return cp.sum(diff, axis=tuple(range(-2, 0)))

    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None, use_cupy=False):
        if not self._dean_gpu_enabled():
            result = super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)
            self._mark_execution_path("gpu_disabled_cpu_fallback")
            return result

        try:
            cache = self._get_dean_gpu_dataset_cache()
            if cache is None:
                raise RuntimeError("GPU dataset cache unavailable")

            if param_names is None:
                if not hasattr(self, "param_names"):
                    raise AttributeError("Missing required attribute: param_names")
                param_names = self.param_names

            if Intensity is not None and Intensity is not self.Intensity:
                raise ValueError("Dean GPU path currently requires the model's loaded Intensity array")
            if Qx is not None and Qx is not self.Qx:
                raise ValueError("Dean GPU path currently requires the model's loaded Qx array")
            if Qz is not None and Qz is not self.Qz:
                raise ValueError("Dean GPU path currently requires the model's loaded Qz array")

            values = np.asarray(optimization_values, dtype=float)
            plan = self._get_dean_param_plan(param_names)

            if values.ndim == 1:
                result = super().SimTrap_GF(values, param_names, self.Intensity, self.Qx, self.Qz, use_cupy=False)
                self._mark_execution_path("gpu_scalar_cpu_fallback")
                return result

            if values.ndim == 2:
                candidates = self._normalize_candidate_matrix(values, len(param_names))
                if candidates.shape[0] < int(self._dean_gpu_min_batch):
                    result = super().SimTrap_GF(
                        candidates,
                        param_names,
                        self.Intensity,
                        self.Qx,
                        self.Qz,
                        use_cupy=False,
                    )
                    self._mark_execution_path("gpu_small_batch_cpu_fallback")
                    return result
                batched_par, batched_slds, dw_arr, i0_arr, bk_arr = self._map_batched_candidates(
                    candidates,
                    plan,
                    self.Intensity,
                )

                coord_b = self.SymCoordAssign(batched_par, self.layers, sld_values=batched_slds)
                if coord_b is None:
                    return np.full(candidates.shape[0], float("inf"))

                form_b_gpu = self._free_form_trapezoid_gpu(cp.asarray(coord_b))
                dw_gpu = cp.asarray(dw_arr, dtype=cp.float64)
                i0_gpu = cp.asarray(i0_arr, dtype=cp.float64)

                intensity_base_b_gpu = (
                    cp.square(cp.abs(form_b_gpu))
                    * cp.exp(-cache["qsq_b_gpu"] * cp.square(dw_gpu)[:, None, None])
                    * i0_gpu[:, None, None]
                )

                if bk_arr.shape[1] == intensity_base_b_gpu.shape[2]:
                    sim_int_b_gpu = intensity_base_b_gpu + cp.asarray(bk_arr, dtype=cp.float64)[:, None, :]
                elif bk_arr.shape[1] == 1:
                    sim_int_b_gpu = intensity_base_b_gpu + cp.asarray(bk_arr[:, 0], dtype=cp.float64)[:, None, None]
                else:
                    raise ValueError("Background array length mismatch in Dean GPU path")

                result = cp.asnumpy(self._gf_calc_gpu(sim_int_b_gpu))
                self._mark_execution_path("gpu_resident")
                return result

            raise ValueError("optimization_values must be a 1D or 2D array")

        except Exception as exc:
            result = super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)
            self._mark_execution_path("gpu_exception_cpu_fallback", exc)
            return result


TrapezoidModelDeanGPU = TrapezoidModelArrayDeanGPU


class TrapezoidModelArrayDeanGPUFused(TrapezoidModelArrayDeanGPU):
    """
    GPU Dean variant with a fused post-form-factor residual path.
    """

    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None, use_cupy=False):
        if not self._dean_gpu_enabled() or cp is None:
            return super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)

        try:
            cache = self._get_dean_gpu_dataset_cache()
            if cache is None:
                raise RuntimeError("GPU dataset cache unavailable")

            if param_names is None:
                if not hasattr(self, "param_names"):
                    raise AttributeError("Missing required attribute: param_names")
                param_names = self.param_names

            values = np.asarray(optimization_values, dtype=float)
            if values.ndim != 2:
                return super().SimTrap_GF(values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)

            candidates = self._normalize_candidate_matrix(values, len(param_names))
            if candidates.shape[0] < int(self._dean_gpu_min_batch):
                return super().SimTrap_GF(candidates, param_names, Intensity, Qx, Qz, use_cupy=False)

            if Intensity is not None and Intensity is not self.Intensity:
                raise ValueError("Dean GPU fused path currently requires the model's loaded Intensity array")
            if Qx is not None and Qx is not self.Qx:
                raise ValueError("Dean GPU fused path currently requires the model's loaded Qx array")
            if Qz is not None and Qz is not self.Qz:
                raise ValueError("Dean GPU fused path currently requires the model's loaded Qz array")

            plan = self._get_dean_param_plan(param_names)
            batched_par, batched_slds, dw_arr, i0_arr, bk_arr = self._map_batched_candidates(
                candidates,
                plan,
                self.Intensity,
            )

            coord_b = self.SymCoordAssign(batched_par, self.layers, sld_values=batched_slds)
            if coord_b is None:
                return np.full(candidates.shape[0], float("inf"))

            form_b_gpu = self._free_form_trapezoid_gpu(cp.asarray(coord_b))
            form_abs_sq = cp.square(cp.abs(form_b_gpu))
            dw_sq_gpu = cp.asarray(np.square(dw_arr), dtype=cp.float64)[:, None, None]
            i0_gpu = cp.asarray(i0_arr, dtype=cp.float64)[:, None, None]

            if bk_arr.shape[1] == form_abs_sq.shape[2]:
                bk_gpu = cp.asarray(bk_arr, dtype=cp.float64)[:, None, :]
            elif bk_arr.shape[1] == 1:
                bk_gpu = cp.asarray(bk_arr[:, 0], dtype=cp.float64)[:, None, None]
            else:
                raise ValueError("Background array length mismatch in Dean GPU fused path")

            residual_gpu = _fused_batched_log_residual(
                form_abs_sq,
                cache["qsq_b_gpu"],
                dw_sq_gpu,
                i0_gpu,
                bk_gpu,
                cache["log_intensity_b_gpu"],
            )
            residual_gpu = cp.nan_to_num(residual_gpu, copy=False)
            result = cp.asnumpy(cp.sum(residual_gpu, axis=tuple(range(-2, 0))))
            self._mark_execution_path("gpu_resident_fused")
            return result

        except Exception as exc:
            result = super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz, use_cupy=use_cupy)
            self._mark_execution_path("gpu_exception_cpu_fallback_fused", exc)
            return result


TrapezoidModelDeanGPUFused = TrapezoidModelArrayDeanGPUFused
