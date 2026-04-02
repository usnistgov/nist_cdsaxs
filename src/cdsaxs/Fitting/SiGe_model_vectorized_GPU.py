from __future__ import annotations

import contextlib

import numpy as np

from .SiGe_model_vectorized import SiGeModelArray_vectorized

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
    if cp is None or not _CUPY_ERRSTATE_SUPPORTED:
        yield
        return

    with cp.errstate(divide="ignore", invalid="ignore", over="ignore"):
        yield


if cp is not None:  # pragma: no branch - defined only when CuPy is available
    @cp.fuse()
    def _fused_sige_layer_contribution(qx_b, qz_b, h1_b, h2_b, sl_b, sr_b, x1_b, x4_b, sld_b):
        a1 = (
            cp.exp(1j * qx_b * ((h1_b - sr_b * x4_b) / sr_b)) / (qx_b / sr_b + qz_b)
        ) * (
            cp.exp(-1j * h2_b * (qx_b / sr_b + qz_b))
            - cp.exp(-1j * h1_b * (qx_b / sr_b + qz_b))
        )
        a2 = (
            cp.exp(1j * qx_b * ((h1_b - sl_b * x1_b) / sl_b)) / (qx_b / sl_b + qz_b)
        ) * (
            cp.exp(-1j * h2_b * (qx_b / sl_b + qz_b))
            - cp.exp(-1j * h1_b * (qx_b / sl_b + qz_b))
        )
        return (1j / qx_b) * (a1 - a2) * sld_b

    @cp.fuse()
    def _fused_sige_log_residual(form_abs_sq, qsq_b, dw_sq_b, i0_b, bk_b, log_intensity_b):
        sim_intensity = form_abs_sq * cp.exp(-qsq_b * dw_sq_b) * i0_b + bk_b
        return cp.abs(log_intensity_b - cp.log(sim_intensity))

    _sige_form_factor_reduce_rawkernel = cp.RawKernel(
        r'''
        #include <cupy/complex.cuh>

        extern "C" __global__
        void sige_form_factor_reduce_rawkernel(
            const double* qx,
            const double* qz,
            const double* h1,
            const double* h2,
            const double* sl,
            const double* sr,
            const double* x1,
            const double* x4,
            const double* sld,
            const int expanded_count,
            const int q_size,
            const int total_size,
            complex<double>* out
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= total_size) {
                return;
            }

            const complex<double> imag_unit(0.0, 1.0);
            int candidate = idx / q_size;
            int q_index = idx - candidate * q_size;
            double qx_value = qx[q_index];
            double qz_value = qz[q_index];
            complex<double> acc(0.0, 0.0);

            for (int layer_index = 0; layer_index < expanded_count; ++layer_index) {
                int layer_offset = candidate * expanded_count + layer_index;
                double h1_value = h1[layer_offset];
                double h2_value = h2[layer_offset];
                double sl_value = sl[layer_offset];
                double sr_value = sr[layer_offset];
                double x1_value = x1[layer_offset];
                double x4_value = x4[layer_offset];
                double sld_value = sld[layer_offset];
                double right_term = qx_value / sr_value + qz_value;
                double left_term = qx_value / sl_value + qz_value;

                complex<double> a1 = (
                    exp(imag_unit * (qx_value * ((h1_value - sr_value * x4_value) / sr_value))) / right_term
                ) * (
                    exp(complex<double>(0.0, -h2_value * right_term))
                    - exp(complex<double>(0.0, -h1_value * right_term))
                );
                complex<double> a2 = (
                    exp(imag_unit * (qx_value * ((h1_value - sl_value * x1_value) / sl_value))) / left_term
                ) * (
                    exp(complex<double>(0.0, -h2_value * left_term))
                    - exp(complex<double>(0.0, -h1_value * left_term))
                );

                acc += (imag_unit / qx_value) * (a1 - a2) * sld_value;
            }

            out[idx] = acc;
        }
        ''',
        "sige_form_factor_reduce_rawkernel",
    )


class SiGeModelArray_vectorized_GPU(SiGeModelArray_vectorized):
    """
    GPU-resident SiGe variant.

    The design-level preprocessing remains on the CPU, while the batched
    expanded-stack form-factor evaluation and GF reduction move onto the GPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized_gpu_cache = {}
        self._vectorized_gpu_min_batch = 8
        self._vectorized_gpu_layer_algorithm = "4d"
        self._vectorized_gpu_4d_layer_tile = 13
        self._vectorized_gpu_4d_candidate_tile = 32
        self._vectorized_gpu_rawkernel_threads = 128
        self._vectorized_last_execution_path = None
        self._vectorized_last_gpu_exception = None

    def _mark_execution_path(self, path, exc=None):
        self._vectorized_last_execution_path = str(path)
        self._vectorized_last_gpu_exception = None if exc is None else f"{type(exc).__name__}: {exc}"

    def process_imported_data(self):
        super().process_imported_data()
        self._refresh_vectorized_gpu_dataset_cache()

    def _vectorized_gpu_enabled(self):
        return bool(getattr(self, "_freeform_use_cupy", False) and cp is not None)

    def _refresh_vectorized_gpu_dataset_cache(self):
        if cp is None or not hasattr(self, "Intensity") or self.Intensity is None:
            return

        log_intensity = self._vectorized_cache.get("log_intensity")
        qsq = self._vectorized_cache.get("qsq")
        if log_intensity is None or qsq is None:
            self._refresh_vectorized_dataset_cache()
            log_intensity = self._vectorized_cache.get("log_intensity")
            qsq = self._vectorized_cache.get("qsq")

        qx_gpu = cp.asarray(np.asarray(self.Qx, dtype=float))
        qz_gpu = cp.asarray(np.asarray(self.Qz, dtype=float))
        log_intensity_gpu = cp.asarray(np.asarray(log_intensity, dtype=float))
        qsq_gpu = cp.asarray(np.asarray(qsq, dtype=float))

        self._vectorized_gpu_cache["qx_gpu"] = qx_gpu
        self._vectorized_gpu_cache["qz_gpu"] = qz_gpu
        self._vectorized_gpu_cache["qx_b_gpu"] = qx_gpu[None, :, :]
        self._vectorized_gpu_cache["qz_b_gpu"] = qz_gpu[None, :, :]
        self._vectorized_gpu_cache["qx_flat_gpu"] = cp.ascontiguousarray(qx_gpu.ravel())
        self._vectorized_gpu_cache["qz_flat_gpu"] = cp.ascontiguousarray(qz_gpu.ravel())
        self._vectorized_gpu_cache["q_grid_shape"] = tuple(int(v) for v in qx_gpu.shape)
        self._vectorized_gpu_cache["log_intensity_gpu"] = log_intensity_gpu
        self._vectorized_gpu_cache["log_intensity_b_gpu"] = log_intensity_gpu[None, :, :]
        self._vectorized_gpu_cache["qsq_gpu"] = qsq_gpu
        self._vectorized_gpu_cache["qsq_b_gpu"] = qsq_gpu[None, :, :]

    def _get_vectorized_gpu_dataset_cache(self):
        if not self._vectorized_gpu_enabled():
            return None
        if "qx_gpu" not in self._vectorized_gpu_cache:
            self._refresh_vectorized_gpu_dataset_cache()
        return self._vectorized_gpu_cache

    def _batched_form_factor_gpu_loop(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        expanded_width_gpu = cp.asarray(np.asarray(expanded_width, dtype=float))
        expanded_height_gpu = cp.asarray(np.asarray(expanded_height, dtype=float))
        expanded_twidth_gpu = cp.asarray(np.asarray(expanded_twidth, dtype=float))
        expanded_slds_gpu = cp.asarray(np.asarray(expanded_slds, dtype=float))

        batch_size, expanded_count = expanded_width_gpu.shape
        qx_b = cache["qx_b_gpu"]
        qz_b = cache["qz_b_gpu"]

        center = 0.5 * expanded_width_gpu[:, [0]]
        next_width = cp.concatenate([expanded_width_gpu[:, 1:], expanded_width_gpu[:, -1:]], axis=1)
        top_width = cp.where(cp.isnan(expanded_twidth_gpu), next_width, expanded_twidth_gpu)

        x1 = center - 0.5 * expanded_width_gpu
        x4 = center + 0.5 * expanded_width_gpu
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = cp.cumsum(expanded_height_gpu, axis=1)
        h1 = h2 - expanded_height_gpu

        form = cp.zeros((batch_size,) + tuple(cache["qx_gpu"].shape), dtype=cp.complex128)

        with _gpu_errstate():
            for layer_index in range(expanded_count):
                h1_b = h1[:, layer_index][:, None, None]
                h2_b = h2[:, layer_index][:, None, None]
                x1_b = x1[:, layer_index][:, None, None]
                x4_b = x4[:, layer_index][:, None, None]
                sl_b = (
                    expanded_height_gpu[:, layer_index]
                    / (x2[:, layer_index] - x1[:, layer_index])
                )[:, None, None]
                sr_b = (
                    -expanded_height_gpu[:, layer_index]
                    / (x4[:, layer_index] - x3[:, layer_index])
                )[:, None, None]
                sld_b = expanded_slds_gpu[:, layer_index][:, None, None]

                form = form + _fused_sige_layer_contribution(
                    qx_b,
                    qz_b,
                    h1_b,
                    h2_b,
                    sl_b,
                    sr_b,
                    x1_b,
                    x4_b,
                    sld_b,
                )

        return cp.nan_to_num(form, copy=False)

    def _batched_form_factor_gpu_4d(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        expanded_width_gpu = cp.asarray(np.asarray(expanded_width, dtype=float))
        expanded_height_gpu = cp.asarray(np.asarray(expanded_height, dtype=float))
        expanded_twidth_gpu = cp.asarray(np.asarray(expanded_twidth, dtype=float))
        expanded_slds_gpu = cp.asarray(np.asarray(expanded_slds, dtype=float))

        center = 0.5 * expanded_width_gpu[:, [0]]
        next_width = cp.concatenate([expanded_width_gpu[:, 1:], expanded_width_gpu[:, -1:]], axis=1)
        top_width = cp.where(cp.isnan(expanded_twidth_gpu), next_width, expanded_twidth_gpu)

        x1 = center - 0.5 * expanded_width_gpu
        x4 = center + 0.5 * expanded_width_gpu
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = cp.cumsum(expanded_height_gpu, axis=1)
        h1 = h2 - expanded_height_gpu

        h1_b = h1[:, :, None, None]
        h2_b = h2[:, :, None, None]
        x1_b = x1[:, :, None, None]
        x4_b = x4[:, :, None, None]
        x2_b = x2[:, :, None, None]
        x3_b = x3[:, :, None, None]
        qx_b = cache["qx_b_gpu"][:, None, :, :]
        qz_b = cache["qz_b_gpu"][:, None, :, :]

        sl_b = (expanded_height_gpu / (x2 - x1))[:, :, None, None]
        sr_b = (-expanded_height_gpu / (x4 - x3))[:, :, None, None]
        sld_b = expanded_slds_gpu[:, :, None, None]

        with _gpu_errstate():
            a1 = (
                cp.exp(1j * qx_b * ((h1_b - sr_b * x4_b) / sr_b)) / (qx_b / sr_b + qz_b)
            ) * (
                cp.exp(-1j * h2_b * (qx_b / sr_b + qz_b))
                - cp.exp(-1j * h1_b * (qx_b / sr_b + qz_b))
            )
            a2 = (
                cp.exp(1j * qx_b * ((h1_b - sl_b * x1_b) / sl_b)) / (qx_b / sl_b + qz_b)
            ) * (
                cp.exp(-1j * h2_b * (qx_b / sl_b + qz_b))
                - cp.exp(-1j * h1_b * (qx_b / sl_b + qz_b))
            )
            form = cp.sum((1j / qx_b) * (a1 - a2) * sld_b, axis=1)

        return cp.nan_to_num(form, copy=False)

    def _batched_form_factor_gpu_4d_tiled_layers(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        expanded_width_gpu = cp.asarray(np.asarray(expanded_width, dtype=float))
        expanded_height_gpu = cp.asarray(np.asarray(expanded_height, dtype=float))
        expanded_twidth_gpu = cp.asarray(np.asarray(expanded_twidth, dtype=float))
        expanded_slds_gpu = cp.asarray(np.asarray(expanded_slds, dtype=float))

        center = 0.5 * expanded_width_gpu[:, [0]]
        next_width = cp.concatenate([expanded_width_gpu[:, 1:], expanded_width_gpu[:, -1:]], axis=1)
        top_width = cp.where(cp.isnan(expanded_twidth_gpu), next_width, expanded_twidth_gpu)

        x1 = center - 0.5 * expanded_width_gpu
        x4 = center + 0.5 * expanded_width_gpu
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = cp.cumsum(expanded_height_gpu, axis=1)
        h1 = h2 - expanded_height_gpu

        qx_b = cache["qx_b_gpu"][:, None, :, :]
        qz_b = cache["qz_b_gpu"][:, None, :, :]
        form = cp.zeros((expanded_width_gpu.shape[0],) + tuple(cache["qx_gpu"].shape), dtype=cp.complex128)
        tile_size = max(1, int(getattr(self, "_vectorized_gpu_4d_layer_tile", 4)))
        expanded_count = int(expanded_width_gpu.shape[1])

        with _gpu_errstate():
            for start in range(0, expanded_count, tile_size):
                stop = min(start + tile_size, expanded_count)
                layer_slice = slice(start, stop)

                h1_b = h1[:, layer_slice][:, :, None, None]
                h2_b = h2[:, layer_slice][:, :, None, None]
                x1_b = x1[:, layer_slice][:, :, None, None]
                x4_b = x4[:, layer_slice][:, :, None, None]
                sl_b = (
                    expanded_height_gpu[:, layer_slice] / (x2[:, layer_slice] - x1[:, layer_slice])
                )[:, :, None, None]
                sr_b = (
                    -expanded_height_gpu[:, layer_slice] / (x4[:, layer_slice] - x3[:, layer_slice])
                )[:, :, None, None]
                sld_b = expanded_slds_gpu[:, layer_slice][:, :, None, None]

                a1 = (
                    cp.exp(1j * qx_b * ((h1_b - sr_b * x4_b) / sr_b)) / (qx_b / sr_b + qz_b)
                ) * (
                    cp.exp(-1j * h2_b * (qx_b / sr_b + qz_b))
                    - cp.exp(-1j * h1_b * (qx_b / sr_b + qz_b))
                )
                a2 = (
                    cp.exp(1j * qx_b * ((h1_b - sl_b * x1_b) / sl_b)) / (qx_b / sl_b + qz_b)
                ) * (
                    cp.exp(-1j * h2_b * (qx_b / sl_b + qz_b))
                    - cp.exp(-1j * h1_b * (qx_b / sl_b + qz_b))
                )
                form = form + cp.sum((1j / qx_b) * (a1 - a2) * sld_b, axis=1)

        return cp.nan_to_num(form, copy=False)

    def _batched_form_factor_gpu_4d_tiled_layers_candidates(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        expanded_width_gpu = cp.asarray(np.asarray(expanded_width, dtype=float))
        expanded_height_gpu = cp.asarray(np.asarray(expanded_height, dtype=float))
        expanded_twidth_gpu = cp.asarray(np.asarray(expanded_twidth, dtype=float))
        expanded_slds_gpu = cp.asarray(np.asarray(expanded_slds, dtype=float))

        batch_size, expanded_count = expanded_width_gpu.shape
        center = 0.5 * expanded_width_gpu[:, [0]]
        next_width = cp.concatenate([expanded_width_gpu[:, 1:], expanded_width_gpu[:, -1:]], axis=1)
        top_width = cp.where(cp.isnan(expanded_twidth_gpu), next_width, expanded_twidth_gpu)

        x1 = center - 0.5 * expanded_width_gpu
        x4 = center + 0.5 * expanded_width_gpu
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = cp.cumsum(expanded_height_gpu, axis=1)
        h1 = h2 - expanded_height_gpu

        form = cp.zeros((batch_size,) + tuple(cache["qx_gpu"].shape), dtype=cp.complex128)
        layer_tile_size = max(1, int(getattr(self, "_vectorized_gpu_4d_layer_tile", 13)))
        candidate_tile_size = max(1, int(getattr(self, "_vectorized_gpu_4d_candidate_tile", 32)))

        with _gpu_errstate():
            for candidate_start in range(0, batch_size, candidate_tile_size):
                candidate_stop = min(candidate_start + candidate_tile_size, batch_size)
                candidate_slice = slice(candidate_start, candidate_stop)
                qx_b = cache["qx_b_gpu"][:, None, :, :]
                qz_b = cache["qz_b_gpu"][:, None, :, :]
                form_chunk = cp.zeros((candidate_stop - candidate_start,) + tuple(cache["qx_gpu"].shape), dtype=cp.complex128)

                for layer_start in range(0, expanded_count, layer_tile_size):
                    layer_stop = min(layer_start + layer_tile_size, expanded_count)
                    layer_slice = slice(layer_start, layer_stop)

                    h1_b = h1[candidate_slice, layer_slice][:, :, None, None]
                    h2_b = h2[candidate_slice, layer_slice][:, :, None, None]
                    x1_b = x1[candidate_slice, layer_slice][:, :, None, None]
                    x4_b = x4[candidate_slice, layer_slice][:, :, None, None]
                    sl_b = (
                        expanded_height_gpu[candidate_slice, layer_slice]
                        / (x2[candidate_slice, layer_slice] - x1[candidate_slice, layer_slice])
                    )[:, :, None, None]
                    sr_b = (
                        -expanded_height_gpu[candidate_slice, layer_slice]
                        / (x4[candidate_slice, layer_slice] - x3[candidate_slice, layer_slice])
                    )[:, :, None, None]
                    sld_b = expanded_slds_gpu[candidate_slice, layer_slice][:, :, None, None]

                    a1 = (
                        cp.exp(1j * qx_b * ((h1_b - sr_b * x4_b) / sr_b)) / (qx_b / sr_b + qz_b)
                    ) * (
                        cp.exp(-1j * h2_b * (qx_b / sr_b + qz_b))
                        - cp.exp(-1j * h1_b * (qx_b / sr_b + qz_b))
                    )
                    a2 = (
                        cp.exp(1j * qx_b * ((h1_b - sl_b * x1_b) / sl_b)) / (qx_b / sl_b + qz_b)
                    ) * (
                        cp.exp(-1j * h2_b * (qx_b / sl_b + qz_b))
                        - cp.exp(-1j * h1_b * (qx_b / sl_b + qz_b))
                    )
                    form_chunk = form_chunk + cp.sum((1j / qx_b) * (a1 - a2) * sld_b, axis=1)

                form[candidate_slice] = form_chunk

        return cp.nan_to_num(form, copy=False)

    def _batched_form_factor_gpu_4d_rawkernel(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        expanded_width_gpu = cp.asarray(np.asarray(expanded_width, dtype=float))
        expanded_height_gpu = cp.asarray(np.asarray(expanded_height, dtype=float))
        expanded_twidth_gpu = cp.asarray(np.asarray(expanded_twidth, dtype=float))
        expanded_slds_gpu = cp.asarray(np.asarray(expanded_slds, dtype=float))

        batch_size, expanded_count = expanded_width_gpu.shape
        center = 0.5 * expanded_width_gpu[:, [0]]
        next_width = cp.concatenate([expanded_width_gpu[:, 1:], expanded_width_gpu[:, -1:]], axis=1)
        top_width = cp.where(cp.isnan(expanded_twidth_gpu), next_width, expanded_twidth_gpu)

        x1 = center - 0.5 * expanded_width_gpu
        x4 = center + 0.5 * expanded_width_gpu
        x2 = center - 0.5 * top_width
        x3 = center + 0.5 * top_width

        x2 = cp.where(cp.isclose(x2, x1), x1 - 1e-6, x2)
        x4 = cp.where(cp.isclose(x4, x3), x3 - 1e-6, x4)

        h2 = cp.cumsum(expanded_height_gpu, axis=1)
        h1 = h2 - expanded_height_gpu
        sl = expanded_height_gpu / (x2 - x1)
        sr = -expanded_height_gpu / (x4 - x3)

        q_size = int(cache["qx_flat_gpu"].size)
        total_size = int(batch_size * q_size)
        threads = max(1, int(getattr(self, "_vectorized_gpu_rawkernel_threads", 128)))
        blocks = max(1, (total_size + threads - 1) // threads)

        form_flat = cp.empty(total_size, dtype=cp.complex128)
        _sige_form_factor_reduce_rawkernel(
            (blocks,),
            (threads,),
            (
                cache["qx_flat_gpu"],
                cache["qz_flat_gpu"],
                cp.ascontiguousarray(h1.ravel()),
                cp.ascontiguousarray(h2.ravel()),
                cp.ascontiguousarray(sl.ravel()),
                cp.ascontiguousarray(sr.ravel()),
                cp.ascontiguousarray(x1.ravel()),
                cp.ascontiguousarray(x4.ravel()),
                cp.ascontiguousarray(expanded_slds_gpu.ravel()),
                np.int32(expanded_count),
                np.int32(q_size),
                np.int32(total_size),
                form_flat,
            ),
        )

        form = form_flat.reshape((batch_size,) + tuple(cache["q_grid_shape"]))
        return cp.nan_to_num(form, copy=False)

    def _batched_form_factor_gpu(
        self,
        expanded_width,
        expanded_height,
        expanded_twidth,
        expanded_slds,
    ):
        algorithm = str(getattr(self, "_vectorized_gpu_layer_algorithm", "4d")).strip().lower()
        if algorithm in {"4d_naive", "4d_full"}:
            return self._batched_form_factor_gpu_4d(
                expanded_width,
                expanded_height,
                expanded_twidth,
                expanded_slds,
            )
        if algorithm == "4d_tiled_layers":
            return self._batched_form_factor_gpu_4d_tiled_layers(
                expanded_width,
                expanded_height,
                expanded_twidth,
                expanded_slds,
            )
        if algorithm == "4d_tiled_layers_candidates":
            return self._batched_form_factor_gpu_4d_tiled_layers_candidates(
                expanded_width,
                expanded_height,
                expanded_twidth,
                expanded_slds,
            )
        if algorithm in {"4d", "4d_rawkernel"}:
            return self._batched_form_factor_gpu_4d_rawkernel(
                expanded_width,
                expanded_height,
                expanded_twidth,
                expanded_slds,
            )
        return self._batched_form_factor_gpu_loop(
            expanded_width,
            expanded_height,
            expanded_twidth,
            expanded_slds,
        )

    def _evaluate_batched_realistic_path_gpu(self, candidates, param_names, Intensity, Qx, Qz):
        spec = self._compile_batched_path_spec(param_names)
        state = self._initialize_batched_state(candidates, spec)
        state = self._apply_batched_constraints(state, spec)
        expanded_width, expanded_height, expanded_twidth, expanded_slds = self._expand_design_batch(state, spec)

        cache = self._get_vectorized_gpu_dataset_cache()
        if cache is None:
            raise RuntimeError("GPU dataset cache unavailable")

        form_gpu = self._batched_form_factor_gpu(
            expanded_width,
            expanded_height,
            expanded_twidth,
            expanded_slds,
        )
        form_abs_sq_gpu = cp.square(cp.abs(form_gpu))
        dw_sq_gpu = cp.asarray(np.square(state["dw"]), dtype=cp.float64)[:, None, None]
        i0_gpu = cp.asarray(np.asarray(state["i0"], dtype=float), dtype=cp.float64)[:, None, None]

        intensity_width = int(np.asarray(Intensity).shape[1])
        if state["bk"].shape[1] == intensity_width:
            bk_gpu = cp.asarray(np.asarray(state["bk"], dtype=float), dtype=cp.float64)[:, None, :]
        elif state["bk"].shape[1] == 1:
            bk_gpu = cp.asarray(np.asarray(state["bk"][:, 0], dtype=float), dtype=cp.float64)[:, None, None]
        else:
            raise ValueError("Background array length mismatch in batched SiGe GPU path")

        with _gpu_errstate():
            residual_gpu = _fused_sige_log_residual(
                form_abs_sq_gpu,
                cache["qsq_b_gpu"],
                dw_sq_gpu,
                i0_gpu,
                bk_gpu,
                cache["log_intensity_b_gpu"],
            )

        residual_gpu = cp.nan_to_num(residual_gpu, copy=False)
        return cp.asnumpy(cp.sum(residual_gpu, axis=tuple(range(-2, 0))))

    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None):
        values = np.asarray(optimization_values, dtype=float)

        try:
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
                result = super().SimTrap_GF(values, param_names, Intensity, Qx, Qz)
                self._mark_execution_path("gpu_scalar_cpu_fallback")
                return result

            if values.ndim != 2:
                raise ValueError("optimization_values must be a 1D or 2D array")

            candidates = self._normalize_candidate_matrix(values, param_names)
            if not self._supports_batched_path(param_names):
                result = super().SimTrap_GF(candidates, param_names, Intensity, Qx, Qz)
                self._mark_execution_path("gpu_unsupported_cpu_fallback")
                return result

            if not self._vectorized_gpu_enabled():
                result = super().SimTrap_GF(candidates, param_names, Intensity, Qx, Qz)
                self._mark_execution_path("gpu_disabled_cpu_fallback")
                return result

            if Intensity is not self.Intensity or Qx is not self.Qx or Qz is not self.Qz:
                result = super().SimTrap_GF(candidates, param_names, Intensity, Qx, Qz)
                self._mark_execution_path("gpu_dataset_mismatch_cpu_fallback")
                return result

            if candidates.shape[0] < int(self._vectorized_gpu_min_batch):
                result = super().SimTrap_GF(candidates, param_names, Intensity, Qx, Qz)
                self._mark_execution_path("gpu_small_batch_cpu_fallback")
                return result

            result = self._evaluate_batched_realistic_path_gpu(candidates, param_names, Intensity, Qx, Qz)
            execution_suffix = "_4d" if str(getattr(self, "_vectorized_gpu_layer_algorithm", "4d")).strip().lower() == "4d" else ""
            self._mark_execution_path(f"gpu_resident{execution_suffix}")
            return np.asarray(result, dtype=float)

        except Exception as exc:
            result = super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz)
            self._mark_execution_path("gpu_exception_cpu_fallback", exc)
            return result


SiGeModel_vectorized_GPU = SiGeModelArray_vectorized_GPU
