"""JAX simulation/objective core for SiGe typed-layer CDSAXS models."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


TARGET_TRAP_WIDTH = 0
TARGET_TRAP_HEIGHT = 1
TARGET_TRAP_TWIDTH = 2
TARGET_TRAP_DEPTH = 3
TARGET_DW = 4
TARGET_I0 = 5
TARGET_BK = 6
TARGET_BK_INDEX = 7

LAYER_TRAPEZOID = 0
LAYER_ELLIPSE = 1


@dataclass(frozen=True)
class SiGeJaxProblem:
    """Prepared fixed-shape arrays for the JAX SiGe objective."""

    param_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    default: np.ndarray
    qx: np.ndarray
    qz: np.ndarray
    intensity: np.ndarray
    design_width: np.ndarray
    design_height: np.ndarray
    design_twidth: np.ndarray
    design_depth: np.ndarray
    design_type: np.ndarray
    design_num_segments: np.ndarray
    design_slds: np.ndarray
    expanded_design_index: np.ndarray
    expanded_segment_index: np.ndarray
    param_target_type: np.ndarray
    param_target_index: np.ndarray
    eq_lhs_type: np.ndarray
    eq_lhs_index: np.ndarray
    eq_rhs_type: np.ndarray
    eq_rhs_index: np.ndarray
    eq_offset: np.ndarray
    le_lhs_type: np.ndarray
    le_lhs_index: np.ndarray
    le_rhs_type: np.ndarray
    le_rhs_index: np.ndarray
    le_offset: np.ndarray
    dw: np.float32
    i0: np.float32
    bk: np.ndarray
    base_model_params: dict


def drop_equality_derived(params_to_optimize, constraints):
    """Drop optimization keys that are derived as the lhs of equality constraints."""
    derived = set()
    for rule in constraints or []:
        if isinstance(rule, dict) and str(rule.get("op", "")).strip() == "==":
            lhs = rule.get("lhs")
            if isinstance(lhs, str):
                derived.add(lhs)
    return {k: v for k, v in params_to_optimize.items() if k not in derived}


def unit_to_physical_np(x_unit, lower, upper):
    x_unit = np.clip(np.asarray(x_unit, dtype=np.float32), 0.0, 1.0)
    return np.asarray(lower, dtype=np.float32) + x_unit * (
        np.asarray(upper, dtype=np.float32) - np.asarray(lower, dtype=np.float32)
    )


def unit_to_physical_jax(x_unit, lower, upper):
    x_unit = jnp.clip(x_unit, 0.0, 1.0)
    return lower + x_unit * (upper - lower)


def prepare_sige_jax_problem(model) -> SiGeJaxProblem:
    """Prepare a SiGe model for the JAX D24/ellipse objective path."""
    params = model.initialize_optimization_params()
    params = model._ensure_defaults_in_params(params)
    params = drop_equality_derived(params, model.model_params.get("constraints", None))

    param_names = tuple(params)
    lower = np.asarray([params[p]["min"] for p in param_names], dtype=np.float32)
    upper = np.asarray([params[p]["max"] for p in param_names], dtype=np.float32)
    default = np.asarray([params[p]["default"] for p in param_names], dtype=np.float32)

    model_params = model.model_params
    design_traps = model_params.get("design_trapezoids", model_params.get("trapezoids", []))
    if not isinstance(design_traps, list) or not design_traps:
        raise ValueError("SiGe JAX path requires a non-empty trapezoid design stack")

    design_slds = _normalise_design_slds(
        model_params.get("design_slds", model_params.get("slds", None)),
        len(design_traps),
    )

    width = []
    height = []
    twidth = []
    depth = []
    layer_type = []
    num_segments = []
    expanded_design_index = []
    expanded_segment_index = []

    for idx, trap in enumerate(design_traps):
        if not isinstance(trap, dict):
            raise TypeError(f"trapezoids[{idx}] must be a dict")
        layer_type_name = trap.get("Layer_Type", None)
        layer_type_norm = (
            str(layer_type_name).strip().lower()
            if layer_type_name is not None
            else "trapezoid"
        )
        if layer_type_norm == "curved_sides":
            raise NotImplementedError("SiGe JAX v1 does not support Layer_Type='curved_sides'")
        if layer_type_norm not in ("trapezoid", "ellipse"):
            raise NotImplementedError(f"SiGe JAX v1 does not support Layer_Type={layer_type_name!r}")

        n = int(trap.get("num_layers", 1)) if layer_type_norm == "ellipse" else 1
        n = max(1, n)

        width.append(float(trap["width"]))
        height.append(float(trap["height"]))
        tw = trap.get("twidth", None)
        twidth.append(np.nan if tw is None else float(tw))
        depth.append(float(trap.get("depth", 0.0)))
        layer_type.append(LAYER_ELLIPSE if layer_type_norm == "ellipse" else LAYER_TRAPEZOID)
        num_segments.append(n)

        for seg_idx in range(n):
            expanded_design_index.append(idx)
            expanded_segment_index.append(seg_idx)

    if len(expanded_design_index) < 1:
        raise ValueError("SiGe JAX path produced no expanded trapezoid rows")

    param_target_type = []
    param_target_index = []
    for name in param_names:
        target_type, target_index = _parse_target_name(name)
        param_target_type.append(target_type)
        param_target_index.append(target_index)

    (
        eq_lhs_type,
        eq_lhs_index,
        eq_rhs_type,
        eq_rhs_index,
        eq_offset,
        le_lhs_type,
        le_lhs_index,
        le_rhs_type,
        le_rhs_index,
        le_offset,
    ) = _prepare_constraints(model_params.get("constraints", None))

    qx = np.asarray(model.Qx, dtype=np.float32)
    qz = np.asarray(model.Qz, dtype=np.float32)
    intensity = np.asarray(model.Intensity, dtype=np.float32)
    if qx.shape != qz.shape or qx.shape != intensity.shape:
        raise ValueError(
            f"Qx, Qz, and Intensity must share shape; got {qx.shape}, {qz.shape}, {intensity.shape}"
        )

    bk = _background_vector(model.Bk, qx.shape[1]).astype(np.float32)

    return SiGeJaxProblem(
        param_names=param_names,
        lower=lower,
        upper=upper,
        default=default,
        qx=qx,
        qz=qz,
        intensity=intensity,
        design_width=np.asarray(width, dtype=np.float32),
        design_height=np.asarray(height, dtype=np.float32),
        design_twidth=np.asarray(twidth, dtype=np.float32),
        design_depth=np.asarray(depth, dtype=np.float32),
        design_type=np.asarray(layer_type, dtype=np.int32),
        design_num_segments=np.asarray(num_segments, dtype=np.int32),
        design_slds=np.asarray(design_slds, dtype=np.float32),
        expanded_design_index=np.asarray(expanded_design_index, dtype=np.int32),
        expanded_segment_index=np.asarray(expanded_segment_index, dtype=np.int32),
        param_target_type=np.asarray(param_target_type, dtype=np.int32),
        param_target_index=np.asarray(param_target_index, dtype=np.int32),
        eq_lhs_type=np.asarray(eq_lhs_type, dtype=np.int32),
        eq_lhs_index=np.asarray(eq_lhs_index, dtype=np.int32),
        eq_rhs_type=np.asarray(eq_rhs_type, dtype=np.int32),
        eq_rhs_index=np.asarray(eq_rhs_index, dtype=np.int32),
        eq_offset=np.asarray(eq_offset, dtype=np.float32),
        le_lhs_type=np.asarray(le_lhs_type, dtype=np.int32),
        le_lhs_index=np.asarray(le_lhs_index, dtype=np.int32),
        le_rhs_type=np.asarray(le_rhs_type, dtype=np.int32),
        le_rhs_index=np.asarray(le_rhs_index, dtype=np.int32),
        le_offset=np.asarray(le_offset, dtype=np.float32),
        dw=np.float32(model.DW),
        i0=np.float32(model.I0),
        bk=bk,
        base_model_params=copy.deepcopy(model.model_params),
    )


def make_objective_batch(problem: SiGeJaxProblem):
    """Build a JIT-compiled batch objective for normalized unit-cube candidates."""
    lower = jnp.asarray(problem.lower, dtype=jnp.float32)
    upper = jnp.asarray(problem.upper, dtype=jnp.float32)
    arrays = _problem_to_jax_arrays(problem)

    def objective_one(x_unit):
        x_phys = unit_to_physical_jax(x_unit, lower, upper)
        return sige_objective_jax(x_phys, arrays)

    return jax.jit(jax.vmap(objective_one))


def _problem_to_jax_arrays(problem: SiGeJaxProblem):
    return {
        "qx": jnp.asarray(problem.qx, dtype=jnp.float32),
        "qz": jnp.asarray(problem.qz, dtype=jnp.float32),
        "intensity": jnp.asarray(problem.intensity, dtype=jnp.float32),
        "design_width": jnp.asarray(problem.design_width, dtype=jnp.float32),
        "design_height": jnp.asarray(problem.design_height, dtype=jnp.float32),
        "design_twidth": jnp.asarray(problem.design_twidth, dtype=jnp.float32),
        "design_depth": jnp.asarray(problem.design_depth, dtype=jnp.float32),
        "design_type": jnp.asarray(problem.design_type, dtype=jnp.int32),
        "design_num_segments": jnp.asarray(problem.design_num_segments, dtype=jnp.int32),
        "design_slds": jnp.asarray(problem.design_slds, dtype=jnp.float32),
        "expanded_design_index": jnp.asarray(problem.expanded_design_index, dtype=jnp.int32),
        "expanded_segment_index": jnp.asarray(problem.expanded_segment_index, dtype=jnp.int32),
        "param_target_type": jnp.asarray(problem.param_target_type, dtype=jnp.int32),
        "param_target_index": jnp.asarray(problem.param_target_index, dtype=jnp.int32),
        "eq_lhs_type": jnp.asarray(problem.eq_lhs_type, dtype=jnp.int32),
        "eq_lhs_index": jnp.asarray(problem.eq_lhs_index, dtype=jnp.int32),
        "eq_rhs_type": jnp.asarray(problem.eq_rhs_type, dtype=jnp.int32),
        "eq_rhs_index": jnp.asarray(problem.eq_rhs_index, dtype=jnp.int32),
        "eq_offset": jnp.asarray(problem.eq_offset, dtype=jnp.float32),
        "le_lhs_type": jnp.asarray(problem.le_lhs_type, dtype=jnp.int32),
        "le_lhs_index": jnp.asarray(problem.le_lhs_index, dtype=jnp.int32),
        "le_rhs_type": jnp.asarray(problem.le_rhs_type, dtype=jnp.int32),
        "le_rhs_index": jnp.asarray(problem.le_rhs_index, dtype=jnp.int32),
        "le_offset": jnp.asarray(problem.le_offset, dtype=jnp.float32),
        "dw": jnp.asarray(problem.dw, dtype=jnp.float32),
        "i0": jnp.asarray(problem.i0, dtype=jnp.float32),
        "bk": jnp.asarray(problem.bk, dtype=jnp.float32),
    }


def sige_objective_jax(x_phys, arrays):
    par, slds, dw, i0, bk = build_expanded_arrays_jax(x_phys, arrays)
    sim_int = simulate_sige_jax(par, slds, arrays["qx"], arrays["qz"], dw, i0, bk)
    return gf_jax(sim_int, arrays["intensity"])


def build_expanded_arrays_jax(x_phys, arrays):
    width = arrays["design_width"]
    height = arrays["design_height"]
    twidth = arrays["design_twidth"]
    depth = arrays["design_depth"]
    dw = arrays["dw"]
    i0 = arrays["i0"]
    bk = arrays["bk"]

    def apply_param(i, carry):
        width_i, height_i, twidth_i, depth_i, dw_i, i0_i, bk_i = carry
        return _set_target_value(
            width_i,
            height_i,
            twidth_i,
            depth_i,
            dw_i,
            i0_i,
            bk_i,
            arrays["param_target_type"][i],
            arrays["param_target_index"][i],
            x_phys[i],
        )

    width, height, twidth, depth, dw, i0, bk = jax.lax.fori_loop(
        0, x_phys.shape[0], apply_param, (width, height, twidth, depth, dw, i0, bk)
    )

    def apply_eq_once(carry):
        width_i, height_i, twidth_i, depth_i, dw_i, i0_i, bk_i = carry

        def body(rule_i, rule_carry):
            width_r, height_r, twidth_r, depth_r, dw_r, i0_r, bk_r = rule_carry
            rhs = _get_target_value(
                width_r,
                height_r,
                twidth_r,
                depth_r,
                dw_r,
                i0_r,
                bk_r,
                arrays["eq_rhs_type"][rule_i],
                arrays["eq_rhs_index"][rule_i],
            )
            value = rhs + arrays["eq_offset"][rule_i]
            return _set_target_value(
                width_r,
                height_r,
                twidth_r,
                depth_r,
                dw_r,
                i0_r,
                bk_r,
                arrays["eq_lhs_type"][rule_i],
                arrays["eq_lhs_index"][rule_i],
                value,
            )

        return jax.lax.fori_loop(
            0, arrays["eq_offset"].shape[0], body, (width_i, height_i, twidth_i, depth_i, dw_i, i0_i, bk_i)
        )

    def eq_iter(_, carry):
        return apply_eq_once(carry)

    width, height, twidth, depth, dw, i0, bk = jax.lax.fori_loop(
        0, 10, eq_iter, (width, height, twidth, depth, dw, i0, bk)
    )

    def apply_le(rule_i, carry):
        width_r, height_r, twidth_r, depth_r, dw_r, i0_r, bk_r = carry
        rhs = _get_target_value(
            width_r,
            height_r,
            twidth_r,
            depth_r,
            dw_r,
            i0_r,
            bk_r,
            arrays["le_rhs_type"][rule_i],
            arrays["le_rhs_index"][rule_i],
        )
        lhs = _get_target_value(
            width_r,
            height_r,
            twidth_r,
            depth_r,
            dw_r,
            i0_r,
            bk_r,
            arrays["le_lhs_type"][rule_i],
            arrays["le_lhs_index"][rule_i],
        )
        clipped = jnp.minimum(lhs, rhs + arrays["le_offset"][rule_i])
        return _set_target_value(
            width_r,
            height_r,
            twidth_r,
            depth_r,
            dw_r,
            i0_r,
            bk_r,
            arrays["le_lhs_type"][rule_i],
            arrays["le_lhs_index"][rule_i],
            clipped,
        )

    width, height, twidth, depth, dw, i0, bk = jax.lax.fori_loop(
        0, arrays["le_offset"].shape[0], apply_le, (width, height, twidth, depth, dw, i0, bk)
    )

    par, slds = expand_design_jax(width, height, twidth, depth, arrays)
    return par, slds, dw, i0, bk


def expand_design_jax(width, height, twidth, depth, arrays):
    d_idx = arrays["expanded_design_index"]
    seg_idx = arrays["expanded_segment_index"]
    layer_type = arrays["design_type"][d_idx]
    nseg = arrays["design_num_segments"][d_idx].astype(jnp.float32)

    w0 = width[d_idx]
    h = height[d_idx]
    tw = twidth[d_idx]
    d = depth[d_idx]
    w1_design = jnp.where(jnp.isnan(tw), w0, tw)

    y0 = seg_idx.astype(jnp.float32) * h / nseg
    y1 = (seg_idx.astype(jnp.float32) + 1.0) * h / nseg

    def ellipse_width(y):
        base_w = jnp.where(h == 0.0, w0, w0 + (w1_design - w0) * (y / h))
        semi_major = h / 2.0
        center_y = h / 2.0
        norm = jnp.where(semi_major == 0.0, 0.0, (y - center_y) / semi_major)
        indent = d * jnp.sqrt(jnp.maximum(0.0, 1.0 - norm**2))
        return jnp.maximum(base_w - 2.0 * indent, 1e-12)

    ellipse_w0 = ellipse_width(y0)
    ellipse_w1 = ellipse_width(y1)
    is_ellipse = layer_type == LAYER_ELLIPSE

    expanded_width = jnp.where(is_ellipse, ellipse_w0, w0)
    expanded_height = jnp.where(is_ellipse, h / nseg, h)
    expanded_twidth = jnp.where(is_ellipse, ellipse_w1, tw)
    par = jnp.stack([expanded_width, expanded_height, expanded_twidth], axis=1)
    slds = arrays["design_slds"][d_idx]
    return par, slds


def coord_assign_jax(par, slds):
    center = 0.5 * par[0, 0]
    widths = par[:, 0]
    heights = par[:, 1]
    twidths = par[:, 2]
    next_widths = jnp.concatenate([widths[1:], widths[-1:]])
    top_widths = jnp.where(jnp.isnan(twidths), next_widths, twidths)

    coord = jnp.zeros((par.shape[0], 7), dtype=par.dtype)
    coord = coord.at[:, 0].set(center - 0.5 * widths)
    coord = coord.at[:, 1].set(center + 0.5 * widths)
    coord = coord.at[:, 2].set(heights)
    coord = coord.at[:, 3].set(0.0)
    coord = coord.at[:, 4].set(slds)
    coord = coord.at[:, 5].set(center - 0.5 * top_widths)
    coord = coord.at[:, 6].set(center + 0.5 * top_widths)
    return coord


def free_form_trapezoid_jax(coord, qx, qz):
    qx_c = qx.astype(jnp.complex64)
    qz_c = qz.astype(jnp.complex64)

    def body(carry, row):
        h1, form = carry
        height = row[2]
        h2 = h1 + height
        x1, x4, sld, x2, x3 = row[0], row[1], row[4], row[5], row[6]

        # The legacy NumPy guard uses 1e-6 in float64. In the default JAX
        # float32 path that offset rounds away for coordinates around 1e3.
        x2 = jnp.where(jnp.isclose(x2, x1), x1 - 1e-3, x2)
        x4 = jnp.where(jnp.isclose(x4, x3), x3 - 1e-3, x4)

        sl = height / (x2 - x1)
        sr = -height / (x4 - x3)

        a1_arg = qx_c / sr + qz_c
        a2_arg = qx_c / sl + qz_c
        a1 = (
            jnp.exp(1j * qx_c * ((h1 - sr * x4) / sr))
            / a1_arg
            * (jnp.exp(-1j * h2 * a1_arg) - jnp.exp(-1j * h1 * a1_arg))
        )
        a2 = (
            jnp.exp(1j * qx_c * ((h1 - sl * x1) / sl))
            / a2_arg
            * (jnp.exp(-1j * h2 * a2_arg) - jnp.exp(-1j * h1 * a2_arg))
        )
        form = form + (1j / qx_c) * (a1 - a2) * sld
        return (h2, form), None

    init_form = jnp.zeros_like(qx, dtype=jnp.complex64)
    (_, form), _ = jax.lax.scan(body, (coord[0, 3], init_form), coord)
    return form


def simulate_sige_jax(par, slds, qx, qz, dw, i0, bk):
    coord = coord_assign_jax(par, slds)
    form = free_form_trapezoid_jax(coord, qx, qz)
    m = jnp.exp(-0.5 * (qx**2 + qz**2) * dw**2)
    formfactor = jnp.abs(form * m)
    intensity_base = formfactor**2 * i0
    return intensity_base + bk[jnp.newaxis, :]


def gf_jax(sim_int, intensity):
    valid_intensity = jnp.isfinite(intensity) & (intensity > 0.0)
    valid_sim = jnp.isfinite(sim_int) & (sim_int > 0.0)
    valid = valid_intensity & valid_sim

    safe_intensity = jnp.where(valid_intensity, intensity, 1.0)
    safe_sim = jnp.where(valid_sim, sim_int, 1.0)
    gf_m = jnp.abs(jnp.log(safe_intensity) - jnp.log(safe_sim))
    gf_m = jnp.where(valid, gf_m, 0.0)

    invalid_sim_points = jnp.sum(valid_intensity & ~valid_sim)
    return jnp.sum(gf_m) + invalid_sim_points * jnp.asarray(1.0e6, dtype=gf_m.dtype)


def _get_target_value(width, height, twidth, depth, dw, i0, bk, target_type, target_index):
    design_idx = jnp.clip(target_index, 0, width.shape[0] - 1)
    bk_idx = jnp.clip(target_index, 0, bk.shape[0] - 1)
    trap_value = jnp.select(
        [
            target_type == TARGET_TRAP_WIDTH,
            target_type == TARGET_TRAP_HEIGHT,
            target_type == TARGET_TRAP_TWIDTH,
            target_type == TARGET_TRAP_DEPTH,
        ],
        [width[design_idx], height[design_idx], twidth[design_idx], depth[design_idx]],
        default=0.0,
    )
    return jnp.select(
        [
            target_type <= TARGET_TRAP_DEPTH,
            target_type == TARGET_DW,
            target_type == TARGET_I0,
            target_type == TARGET_BK,
            target_type == TARGET_BK_INDEX,
        ],
        [trap_value, dw, i0, bk[0], bk[bk_idx]],
        default=0.0,
    )


def _set_target_value(width, height, twidth, depth, dw, i0, bk, target_type, target_index, value):
    design_idx = jnp.clip(target_index, 0, width.shape[0] - 1)
    bk_idx = jnp.clip(target_index, 0, bk.shape[0] - 1)
    width = width.at[design_idx].set(
        jnp.where(target_type == TARGET_TRAP_WIDTH, value, width[design_idx])
    )
    height = height.at[design_idx].set(
        jnp.where(target_type == TARGET_TRAP_HEIGHT, value, height[design_idx])
    )
    twidth = twidth.at[design_idx].set(
        jnp.where(target_type == TARGET_TRAP_TWIDTH, value, twidth[design_idx])
    )
    depth = depth.at[design_idx].set(
        jnp.where(target_type == TARGET_TRAP_DEPTH, value, depth[design_idx])
    )
    dw = jnp.where(target_type == TARGET_DW, value, dw)
    i0 = jnp.where(target_type == TARGET_I0, value, i0)
    bk = jnp.where(target_type == TARGET_BK, jnp.full_like(bk, value), bk)
    bk = bk.at[bk_idx].set(jnp.where(target_type == TARGET_BK_INDEX, value, bk[bk_idx]))
    return width, height, twidth, depth, dw, i0, bk


def _normalise_design_slds(slds, n_design):
    if slds is None:
        return np.ones(n_design, dtype=np.float32)
    if isinstance(slds, (list, tuple, np.ndarray)):
        sld_list = list(slds)
    else:
        sld_list = [float(slds)]
    if len(sld_list) == n_design:
        return np.asarray(sld_list, dtype=np.float32)
    if len(sld_list) == 1:
        return np.full(n_design, float(sld_list[0]), dtype=np.float32)
    return np.resize(np.asarray(sld_list, dtype=np.float32), n_design).astype(np.float32)


def _background_vector(bk, n_columns):
    if isinstance(bk, np.ndarray):
        if bk.ndim == 1 and bk.shape[0] == n_columns:
            return bk
        if bk.size == 1:
            return np.full(n_columns, float(bk.ravel()[0]), dtype=np.float32)
        raise ValueError(f"Background array length ({bk.size}) must match data columns ({n_columns})")
    if isinstance(bk, (list, tuple)):
        if len(bk) == n_columns:
            return np.asarray(bk, dtype=np.float32)
        if len(bk) == 1:
            return np.full(n_columns, float(bk[0]), dtype=np.float32)
        raise ValueError(f"Background list length ({len(bk)}) must match data columns ({n_columns})")
    return np.full(n_columns, float(bk), dtype=np.float32)


def _parse_target_name(name):
    if name.startswith("trap_"):
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid trap parameter name: {name}")
        idx = int(parts[1])
        field = "_".join(parts[2:])
        field_map = {
            "width": TARGET_TRAP_WIDTH,
            "height": TARGET_TRAP_HEIGHT,
            "twidth": TARGET_TRAP_TWIDTH,
            "depth": TARGET_TRAP_DEPTH,
        }
        if field not in field_map:
            raise NotImplementedError(f"SiGe JAX v1 does not support optimized parameter {name!r}")
        return field_map[field], idx
    if name == "DW":
        return TARGET_DW, 0
    if name == "I0":
        return TARGET_I0, 0
    if name == "Bk":
        return TARGET_BK, 0
    if name.startswith("Bk_"):
        return TARGET_BK_INDEX, int(name.split("_")[1])
    raise NotImplementedError(f"SiGe JAX v1 does not support optimized parameter {name!r}")


def _prepare_constraints(constraints):
    eq_lhs_type = []
    eq_lhs_index = []
    eq_rhs_type = []
    eq_rhs_index = []
    eq_offset = []
    le_lhs_type = []
    le_lhs_index = []
    le_rhs_type = []
    le_rhs_index = []
    le_offset = []

    for rule in constraints or []:
        if not isinstance(rule, dict):
            continue
        lhs = rule.get("lhs")
        rhs = rule.get("rhs")
        op = str(rule.get("op", "")).strip()
        if lhs is None or rhs is None or not op:
            continue
        lhs_type, lhs_index = _parse_target_name(lhs)
        rhs_type, rhs_index = _parse_target_name(rhs)
        offset = float(rule.get("offset", 0.0) or 0.0)
        if op == "==":
            eq_lhs_type.append(lhs_type)
            eq_lhs_index.append(lhs_index)
            eq_rhs_type.append(rhs_type)
            eq_rhs_index.append(rhs_index)
            eq_offset.append(offset)
        elif op in ("<=", "<"):
            le_lhs_type.append(lhs_type)
            le_lhs_index.append(lhs_index)
            le_rhs_type.append(rhs_type)
            le_rhs_index.append(rhs_index)
            le_offset.append(offset)
        else:
            raise ValueError(f"Unsupported constraint op: {op}")

    return (
        eq_lhs_type,
        eq_lhs_index,
        eq_rhs_type,
        eq_rhs_index,
        eq_offset,
        le_lhs_type,
        le_lhs_index,
        le_rhs_type,
        le_rhs_index,
        le_offset,
    )
