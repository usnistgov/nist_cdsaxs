# Updated Trapezoid_model.py with common functions moved to base class

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Patch
import copy
from types import SimpleNamespace
from scipy.optimize import differential_evolution

from .CDSAXS_base_model import CDSAXS_Model

class SiGeModelArray(CDSAXS_Model):
    """
    CDSAXS model for SiGe trapezoid structures with array-based background support.
    Each column can have its own background value.
    Includes twidth parameter support for trapezoid top width specification.
    """
    
    def __init__(self, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Initialize the trapezoid model with array background support.
        
        Parameters:
        -----------
        model : str
            Model type
        layers : int
            Number of layers
        PAR : numpy.ndarray, optional
            Traditional parameter array
        SLD : numpy.ndarray, optional
            Scattering length density array
        DW : float, optional
            Debye-Waller factor
        I0 : float, optional
            Intensity scaling factor
        Bk : float or numpy.ndarray, optional
            Background intensity (scalar or array with one value per column)
        Pitch : float, optional
            Pitch parameter
        model_params : dict, optional
            Dictionary-based parameters
        """
        # Handle array background before calling parent constructor
        original_model_params = model_params
        modified_model_params = None
        
        if model_params is not None and 'Bk' in model_params:
            bk_param = model_params['Bk']
            if isinstance(bk_param, (list, np.ndarray)) and len(bk_param) > 1:
                # Store the full array for later use
                self._full_background_array = np.array(bk_param)
                # Create modified model_params with scalar background for parent class
                modified_model_params = model_params.copy()
                modified_model_params['Bk'] = bk_param[0]  # Use first value
                # Also update direct Bk parameter if provided
                if Bk is None:
                    Bk = bk_param[0]
            else:
                self._full_background_array = None
        elif isinstance(Bk, (list, np.ndarray)) and len(Bk) > 1:
            # Direct array background parameter
            self._full_background_array = np.array(Bk)
            Bk = Bk[0]  # Use first value for parent class
        else:
            self._full_background_array = None
        
        # Call parent constructor with scalar background
        super().__init__('sige', model, layers, PAR, SLD, DW, I0, Bk, Pitch, 
                         modified_model_params if modified_model_params is not None else model_params)
        
        # Restore full background array after parent initialization
        if self._full_background_array is not None:
            self.Bk = self._full_background_array
            self.Bk_Initial = self.Bk.copy()
            # Update model_params to ensure consistency with original
            if hasattr(self, 'model_params') and original_model_params is not None:
                self.model_params = original_model_params.copy()
        
        # Initialize array background support
        self._initialize_array_background()
        
        # Initialize SLD values
        self._initialize_sld_values()
    
    def _initialize_array_background(self):
        """
        Initialize array background support based on current Bk value.
        """
        # Background has already been handled in __init__, so this method
        # just ensures consistency and prepares for data import
        if hasattr(self, 'Bk') and self.Bk is not None:
            if not np.isscalar(self.Bk):
                # Already an array - ensure it's numpy array
                if not isinstance(self.Bk, np.ndarray):
                    self.Bk = np.array(self.Bk)
                if not hasattr(self, 'Bk_Initial') or self.Bk_Initial is None:
                    self.Bk_Initial = self.Bk.copy()
        
        # Clean up temporary attribute
        if hasattr(self, '_full_background_array'):
            delattr(self, '_full_background_array')

    def _extract_inline_trapezoid_bounds(self, trapezoids):
        """
        Extract inline optimization bounds from a list of (design-level) trapezoid dicts.

        Expected inline schema per layer dict (all optional):
        - width_bounds:  (min, max)
        - height_bounds: (min, max)
        - twidth_bounds: (min, max)
        - depth_bounds:  (min, max)   (typically for Layer_Type='Ellipse')

        Returns a dict compatible with model_params['optimization'], e.g.
        {'trap_0_width': {'min':..., 'max':...}, ...}
        """
        if trapezoids is None:
            return {}
        if not isinstance(trapezoids, list):
            raise TypeError("model_params['trapezoids'] must be a list of dicts")

        def _validate_bounds(val, key):
            if val is None:
                return None
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"{key} must be a 2-tuple/list (min,max); got {val!r}")
            lo, hi = val[0], val[1]
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ValueError(f"{key} bounds must be numeric; got {val!r}")
            lo, hi = float(lo), float(hi)
            if not lo < hi:
                raise ValueError(f"{key} must satisfy min < max; got {val!r}")
            return lo, hi

        bounds_map = {
            'width_bounds': 'width',
            'height_bounds': 'height',
            'twidth_bounds': 'twidth',
            'depth_bounds': 'depth',
            'side_length_bounds': 'side_length',
            'tip_deflection_bounds': 'tip_deflection',
        }

        opt = {}
        for i, trap in enumerate(trapezoids):
            if not isinstance(trap, dict):
                continue
            for bounds_key, field in bounds_map.items():
                if bounds_key not in trap:
                    continue
                lo_hi = _validate_bounds(trap.get(bounds_key), f"trapezoids[{i}].{bounds_key}")
                if lo_hi is None:
                    continue
                lo, hi = lo_hi
                opt[f"trap_{i}_{field}"] = {'min': lo, 'max': hi}
        return opt

    def _extract_inline_global_bounds(self, model_params):
        """
        Extract inline optimization bounds for global parameters (DW, I0, Bk) from model_params.

        Expected inline schema (all optional):
        - DW_bounds:  (min, max)
        - I0_bounds:  (min, max)
        - Bk_bounds:  (min, max)   (applies to scalar Bk or all elements of array Bk)

        Returns a dict compatible with model_params['optimization'], e.g.
        {'DW': {'min':..., 'max':...}, 'I0': {'min':..., 'max':...}, ...}
        """
        if model_params is None:
            return {}

        def _validate_bounds(val, key):
            if val is None:
                return None
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"{key} must be a 2-tuple/list (min,max); got {val!r}")
            lo, hi = val[0], val[1]
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ValueError(f"{key} bounds must be numeric; got {val!r}")
            lo, hi = float(lo), float(hi)
            if not lo < hi:
                raise ValueError(f"{key} must satisfy min < max; got {val!r}")
            return lo, hi

        opt = {}

        # Extract DW_bounds
        if 'DW_bounds' in model_params:
            lo_hi = _validate_bounds(model_params.get('DW_bounds'), 'DW_bounds')
            if lo_hi is not None:
                lo, hi = lo_hi
                opt['DW'] = {'min': lo, 'max': hi}

        # Extract I0_bounds
        if 'I0_bounds' in model_params:
            lo_hi = _validate_bounds(model_params.get('I0_bounds'), 'I0_bounds')
            if lo_hi is not None:
                lo, hi = lo_hi
                opt['I0'] = {'min': lo, 'max': hi}

        # Extract Bk_bounds
        if 'Bk_bounds' in model_params:
            lo_hi = _validate_bounds(model_params.get('Bk_bounds'), 'Bk_bounds')
            if lo_hi is not None:
                lo, hi = lo_hi
                bk = model_params.get('Bk', None)
                if isinstance(bk, (list, tuple, np.ndarray)):
                    # Array background: apply bounds to each element (Bk_0, Bk_1, ...)
                    bk_list = list(bk)
                    for i in range(len(bk_list)):
                        opt[f'Bk_{i}'] = {'min': lo, 'max': hi}
                else:
                    # Scalar background: apply to 'Bk'
                    opt['Bk'] = {'min': lo, 'max': hi}

        return opt

    def _merge_inline_bounds_into_optimization(self, param_limits=None):
        """
        Merge inline bounds (trapezoid and global parameters) into an optimization dict.

        Supports inline bounds for:
        - Trapezoid parameters: width_bounds, height_bounds, twidth_bounds, depth_bounds (in trapezoid dicts)
        - Global parameters: DW_bounds, I0_bounds, Bk_bounds (in model_params top-level)

        Precedence: explicit entries in `param_limits` (or model_params['optimization'])
        override inline bounds.

        Returns merged dict (does not guarantee defaults are present).
        """
        if not hasattr(self, 'model_params') or self.model_params is None:
            return param_limits if param_limits is not None else {}

        # Prefer design-level trapezoids when present (typed-layer models)
        design_traps = self.model_params.get('design_trapezoids', None)
        traps = design_traps if isinstance(design_traps, list) else self.model_params.get('trapezoids', [])

        # Extract inline bounds from trapezoids
        inline_trap_opt = self._extract_inline_trapezoid_bounds(traps)

        # Extract inline bounds from global parameters (DW, I0, Bk)
        inline_global_opt = self._extract_inline_global_bounds(self.model_params)

        # Merge all inline bounds
        inline_opt = {**inline_trap_opt, **inline_global_opt}

        base = {}
        if isinstance(param_limits, dict):
            base = {k: v.copy() if isinstance(v, dict) else v for k, v in param_limits.items()}
        else:
            existing = self.model_params.get('optimization', {})
            if isinstance(existing, dict):
                base = {k: v.copy() if isinstance(v, dict) else v for k, v in existing.items()}

        # Add inline bounds only if not already in base (explicit config takes precedence)
        for k, v in inline_opt.items():
            if k not in base:
                base[k] = v

        return base

    def _get_param_by_name(self, params_dict, name):
        """
        Get a parameter value from a model_params-like dict by a string name.

        Supported:
        - trap_{i}_{field}   (design-level; prefers design_trapezoids when present)
        - sld_{i}
        - DW, I0
        - Bk, Bk_{i}
        """
        if params_dict is None:
            raise ValueError("params_dict cannot be None")
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if name.startswith('trap_'):
            parts = name.split('_')
            if len(parts) < 3:
                raise ValueError(f"Invalid trap parameter name: {name}")
            idx = int(parts[1])
            field = '_'.join(parts[2:])
            traps = params_dict.get('design_trapezoids', None)
            if not isinstance(traps, list):
                traps = params_dict.get('trapezoids', None)
            if not isinstance(traps, list) or idx >= len(traps):
                raise IndexError(f"Trapezoid index {idx} out of range for {name}")
            return traps[idx][field]

        if name.startswith('sld_'):
            idx = int(name.split('_')[1])
            slds = params_dict.get('design_slds', None)
            if slds is None:
                slds = params_dict.get('slds', None)
            if slds is None:
                raise KeyError("No SLD array found in params_dict")
            slds_list = list(slds) if isinstance(slds, (list, tuple, np.ndarray)) else [float(slds)]
            if idx >= len(slds_list):
                raise IndexError(f"SLD index {idx} out of range for {name}")
            return slds_list[idx]

        if name in ('DW', 'I0'):
            return params_dict[name]

        if name == 'Bk':
            return params_dict.get('Bk', None)

        if name.startswith('Bk_'):
            idx = int(name.split('_')[1])
            bk = params_dict.get('Bk', None)
            if isinstance(bk, (list, tuple, np.ndarray)):
                bk_list = list(bk)
                if idx >= len(bk_list):
                    raise IndexError(f"Bk index {idx} out of range for {name}")
                return bk_list[idx]
            return bk

        raise ValueError(f"Unknown parameter name for constraints: {name}")

    def _set_param_by_name(self, params_dict, name, value):
        """
        Set a parameter value in a model_params-like dict by a string name.
        Mirrors `_get_param_by_name` supported names.
        """
        if params_dict is None:
            raise ValueError("params_dict cannot be None")
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if name.startswith('trap_'):
            parts = name.split('_')
            if len(parts) < 3:
                raise ValueError(f"Invalid trap parameter name: {name}")
            idx = int(parts[1])
            field = '_'.join(parts[2:])
            traps_key = 'design_trapezoids' if isinstance(params_dict.get('design_trapezoids', None), list) else 'trapezoids'
            traps = params_dict.get(traps_key, None)
            if not isinstance(traps, list) or idx >= len(traps):
                raise IndexError(f"Trapezoid index {idx} out of range for {name}")
            traps[idx][field] = value
            return

        if name.startswith('sld_'):
            idx = int(name.split('_')[1])
            key = 'design_slds' if params_dict.get('design_slds', None) is not None else 'slds'
            slds = params_dict.get(key, None)
            if slds is None:
                raise KeyError("No SLD array found in params_dict")
            slds_list = list(slds) if isinstance(slds, (list, tuple, np.ndarray)) else [float(slds)]
            if idx >= len(slds_list):
                raise IndexError(f"SLD index {idx} out of range for {name}")
            slds_list[idx] = float(value)
            params_dict[key] = slds_list
            return

        if name in ('DW', 'I0'):
            params_dict[name] = float(value)
            return

        if name == 'Bk':
            params_dict['Bk'] = value
            return

        if name.startswith('Bk_'):
            idx = int(name.split('_')[1])
            bk = params_dict.get('Bk', None)
            if isinstance(bk, (list, tuple, np.ndarray)):
                bk_list = list(bk)
            else:
                # Create a 1-element list if scalar
                bk_list = [bk]
            if idx >= len(bk_list):
                # Extend with last value
                last = bk_list[-1] if bk_list else 0.0
                while len(bk_list) <= idx:
                    bk_list.append(last)
            bk_list[idx] = float(value)
            params_dict['Bk'] = bk_list
            return

        raise ValueError(f"Unknown parameter name for constraints: {name}")

    def _apply_constraints(self, params_dict, constraints):
        """
        Apply constraints to a model_params-like dict in-place.

        Constraints format:
          {'lhs': 'trap_0_width', 'op': '==', 'rhs': 'trap_1_width', 'offset': 0.0}
          {'lhs': 'trap_2_depth', 'op': '<=', 'rhs': 'trap_2_width', 'offset': 5.0}

        For '==': lhs is derived from rhs (+ offset).
        For '<=': lhs is clipped to rhs (+ offset) if it violates.
        """
        if not constraints:
            return params_dict
        if not isinstance(constraints, list):
            raise TypeError("model_params['constraints'] must be a list of dict rules")

        # Normalize rules
        eq_rules = []
        le_rules = []
        for rule in constraints:
            if not isinstance(rule, dict):
                continue
            lhs = rule.get('lhs', None)
            rhs = rule.get('rhs', None)
            op = rule.get('op', None)
            offset = float(rule.get('offset', 0.0) or 0.0)
            if lhs is None or rhs is None or op is None:
                continue
            op = str(op).strip()
            if op == '==':
                eq_rules.append((lhs, rhs, offset))
            elif op in ('<=', '<'):
                le_rules.append((lhs, rhs, offset))
            else:
                raise ValueError(f"Unsupported constraint op: {op}")

        # Resolve equalities (iterate to allow chained equalities)
        for _ in range(10):
            changed = False
            for lhs, rhs, offset in eq_rules:
                rhs_val = self._get_param_by_name(params_dict, rhs)
                new_val = float(rhs_val) + float(offset)
                old_val = self._get_param_by_name(params_dict, lhs)
                if old_val != new_val:
                    self._set_param_by_name(params_dict, lhs, new_val)
                    changed = True
            if not changed:
                break

        # Enforce inequalities by clipping lhs
        for lhs, rhs, offset in le_rules:
            rhs_val = float(self._get_param_by_name(params_dict, rhs)) + float(offset)
            lhs_val = float(self._get_param_by_name(params_dict, lhs))
            if lhs_val > rhs_val:
                self._set_param_by_name(params_dict, lhs, rhs_val)

        return params_dict
    
    
    
    
    
    def build_model_params_from_traditional(self):
        """
        Build model_params dictionary from traditional parameters including SLD support.
        """
        if not hasattr(self, 'PAR') or self.PAR is None:
            return
            
        # Create trapezoids list from PAR
        trapezoids = []
        for i in range(self.layers + 1):
            tw = self.PAR[i, 2] if (self.PAR.shape[1] >= 3) else np.nan
            trapezoid = {
                'width': self.PAR[i, 0],
                'height': self.PAR[i, 1],
                'twidth': None if (isinstance(tw, float) and np.isnan(tw)) else tw
            }
            trapezoids.append(trapezoid)
        
        # Create the model_params dictionary
        self.model_params = {
            'layers': self.layers,
            'trapezoids': trapezoids,
            'DW': self.DW,
            'I0': self.I0,
            'Bk': self.Bk.tolist() if isinstance(self.Bk, np.ndarray) else self.Bk
        }
        
        # Add SLD values to model_params
        if hasattr(self, 'sld_values'):
            self.model_params['slds'] = self.sld_values.tolist()
        
        # Add optional parameters if they exist
        if hasattr(self, 'SLD') and self.SLD is not None:
            # For backward compatibility, but slds takes precedence
            if 'slds' not in self.model_params:
                if np.isscalar(self.SLD):
                    self.model_params['slds'] = [self.SLD] * (self.layers + 1)
                else:
                    self.model_params['slds'] = self.SLD.tolist()
            
        if hasattr(self, 'Pitch') and self.Pitch is not None:
            self.model_params['Pitch'] = self.Pitch
            
        return self.model_params

    def _has_typed_layers(self, trapezoids):
        """
        Return True if any trapezoid dict indicates a non-standard (typed) layer.
        Currently supports typed layers via `Layer_Type` (case-insensitive), including:
        - 'ellipse'
        - 'curved_sides' (center trapezoid + curved sidewalls; expanded at simulation-time)
        """
        if trapezoids is None:
            return False
        for t in trapezoids:
            if isinstance(t, dict) and t.get('Layer_Type', None) is not None:
                return True
        return False

    def _expand_ellipse_layer(self, layer_dict, next_layer_dict=None):
        """
        Expand a single ellipse layer dict into a list of standard trapezoid segments.

        Expected schema:
        - width: bottom width
        - height: total height
        - twidth: top width (may be None; falls back to next layer width or bottom width)
        - depth: semi-minor axis (indentation depth)
        - num_layers: number of discretization segments
        """
        width0 = float(layer_dict.get('width', 0.0))
        height = float(layer_dict.get('height', 0.0))
        depth = float(layer_dict.get('depth', 0.0))
        n = int(layer_dict.get('num_layers', 1))
        if n < 1:
            n = 1

        # Top width can be independent (mismatch allowed)
        # Default behavior: if `twidth` is not provided/None, set top width == bottom width.
        twidth_val = layer_dict.get('twidth', None)
        if twidth_val is None:
            width1 = width0
        else:
            width1 = float(twidth_val)

        if height < 0:
            raise ValueError("Ellipse layer height must be non-negative")
        if width0 <= 0 or width1 <= 0:
            raise ValueError("Ellipse layer widths must be positive")
        if depth < 0:
            raise ValueError("Ellipse layer depth must be non-negative")
        if depth >= min(width0, width1) / 2.0 and height > 0:
            raise ValueError("Ellipse layer depth must be < min(width0,width1)/2 to keep positive widths")

        # Zero-height layer: treat as a single degenerate segment
        if height == 0:
            return [{'width': width0, 'height': 0.0, 'twidth': width1}]

        dy = height / n
        y = np.linspace(0.0, height, n + 1)

        # Base width transitions linearly between bottom and top widths
        base_w = width0 + (width1 - width0) * (y / height)

        # Ellipse indentation profile: horizontal ellipse, semi-major (vertical) = height/2, semi-minor (horizontal) = depth
        if depth == 0:
            w = base_w
        else:
            semi_major = height / 2.0
            center_y = height / 2.0
            norm = (y - center_y) / semi_major
            indent = depth * np.sqrt(np.maximum(0.0, 1.0 - norm**2))
            w = base_w - 2.0 * indent

        # Safety clamp against tiny negatives from numeric noise
        w = np.maximum(w, 1e-12)

        segments = []
        for i in range(n):
            segments.append({'width': float(w[i]), 'height': float(dy), 'twidth': float(w[i + 1])})
        return segments

    def _expand_typed_layers(self, model_params):
        """
        Expand user-provided (design) layers into a pure trapezoid stack suitable for simulation.

        Returns:
        - expanded_trapezoids: list[dict] length = expanded_layers + 1
        - expanded_slds: list[float] length = expanded_layers + 1
        - expanded_layers: int (the `layers` parameter used everywhere else)
        - expanded_design_index: list[int] mapping each expanded segment to its design entry index
        """
        if model_params is None:
            raise ValueError("model_params cannot be None")

        design_traps = model_params.get('trapezoids', [])
        if not isinstance(design_traps, list):
            raise TypeError("model_params['trapezoids'] must be a list of dicts")

        # Design layer count is informational; use provided value if present, else infer
        design_layers = int(model_params.get('layers', max(0, len(design_traps))))

        # Require explicit trapezoid count convention: len(trapezoids) == layers + 1
        # We avoid injecting any height=0 boundary layers (h=0 should not occur in normal use).
        if design_layers > 0 and len(design_traps) != design_layers + 1:
            raise ValueError(
                f"Typed-layer schema requires len(model_params['trapezoids']) == layers + 1. "
                f"Got layers={design_layers} and trapezoids={len(design_traps)}. "
                f"Please add the top (layers+1) trapezoid entry explicitly (with non-zero height)."
            )

        # SLDs: if missing, default to 1 per design entry; if provided, resize permissively
        design_slds = model_params.get('slds', None)
        if design_slds is None:
            design_slds_list = [1.0] * max(1, len(design_traps))
        else:
            design_slds_list = list(design_slds) if isinstance(design_slds, (list, tuple, np.ndarray)) else [float(design_slds)]
            if len(design_slds_list) != len(design_traps):
                if len(design_slds_list) == 1:
                    design_slds_list = [float(design_slds_list[0])] * max(1, len(design_traps))
                else:
                    design_slds_list = list(np.resize(np.array(design_slds_list, dtype=float), max(1, len(design_traps))))
            # No implicit boundary insertion; keep SLD list aligned to user-provided trapezoids

        expanded_traps = []
        expanded_slds = []
        expanded_design_index = []

        for idx, trap in enumerate(design_traps):
            if not isinstance(trap, dict):
                raise TypeError(f"Each trapezoid entry must be a dict; got {type(trap)} at index {idx}")

            layer_type = trap.get('Layer_Type', None)
            layer_type_norm = str(layer_type).strip().lower() if layer_type is not None else 'trapezoid'
            sld_val = float(design_slds_list[idx]) if idx < len(design_slds_list) else 1.0

            # Ensure required keys exist for standard usage
            trap_width = trap.get('width', None)
            trap_height = trap.get('height', None)
            trap_twidth = trap.get('twidth', None)

            if layer_type_norm == 'ellipse':
                next_trap = design_traps[idx + 1] if idx + 1 < len(design_traps) else None
                ellipse_segments = self._expand_ellipse_layer(trap, next_layer_dict=next_trap)
                for seg in ellipse_segments:
                    expanded_traps.append(seg)
                    expanded_slds.append(sld_val)
                    expanded_design_index.append(int(idx))
            else:
                if trap_width is None:
                    raise ValueError(f"Trapezoid {idx} missing 'width'")
                if trap_height is None:
                    raise ValueError(f"Trapezoid {idx} missing 'height'")
                expanded_traps.append({'width': trap_width, 'height': trap_height, 'twidth': trap_twidth})
                expanded_slds.append(sld_val)
                expanded_design_index.append(int(idx))

        # Ensure the last segment has an explicit top width to avoid PAR[T+1] out-of-range in legacy logic
        if expanded_traps:
            if expanded_traps[-1].get('twidth', None) is None:
                expanded_traps[-1]['twidth'] = expanded_traps[-1]['width']

        expanded_layers = max(0, len(expanded_traps) - 1)

        return expanded_traps, expanded_slds, expanded_layers, design_layers, design_traps, design_slds_list, expanded_design_index

    def _ensure_expanded_model_params(self):
        """
        Ensure `self.model_params` is in expanded (simulation) form.
        If typed layers are present, this mutates `self.model_params` in-place and preserves
        the original user-provided design structure under `design_*` keys.
        """
        if not hasattr(self, 'model_params') or self.model_params is None:
            return

        # Priority 1: If design_trapezoids exists, always use it as source (even if trapezoids is already expanded)
        if 'design_trapezoids' in self.model_params:
            design_traps = self.model_params['design_trapezoids']
            design_layers = self.model_params.get('design_layers', len(design_traps) - 1)
            design_slds_list = self.model_params.get('design_slds', self.model_params.get('slds', None))
            
            # Check if design_trapezoids has typed layers that need expansion
            if self._has_typed_layers(design_traps):
                # Build temporary model_params for expansion
                tmp_model_params = {
                    'trapezoids': [t.copy() for t in design_traps],
                    'layers': design_layers,
                    'slds': copy.deepcopy(design_slds_list) if design_slds_list is not None else None,
                }
                
                # Apply constraints if they exist
                constraints = self.model_params.get('constraints', None)
                if constraints:
                    self._apply_constraints(tmp_model_params, constraints)
                    # IMPORTANT: Update design_trapezoids with constrained values
                    # so that design_trapezoids always reflects the current constrained state
                    self.model_params['design_trapezoids'] = [t.copy() for t in tmp_model_params['trapezoids']]
                
                # Expand from design_trapezoids (or constrained copy if constraints were applied)
                (
                    expanded_traps,
                    expanded_slds,
                    expanded_layers,
                    _,
                    _,
                    _,
                    _,
                ) = self._expand_typed_layers(tmp_model_params)
                
                # Update expanded structure
                self.model_params['layers'] = int(expanded_layers)
                self.model_params['trapezoids'] = expanded_traps
                self.model_params['slds'] = expanded_slds
                self.model_params['_expanded_from_typed_layers'] = True
            else:
                # Design has no typed layers, so trapezoids = design_trapezoids
                self.model_params['trapezoids'] = [t.copy() for t in design_traps]
                self.model_params['layers'] = int(design_layers)
                if design_slds_list is not None:
                    self.model_params['slds'] = copy.deepcopy(design_slds_list)
                self.model_params['_expanded_from_typed_layers'] = False
            return

        # Priority 2: If no design_trapezoids, check if current trapezoids has typed layers
        trapezoids = self.model_params.get('trapezoids', None)
        if not self._has_typed_layers(trapezoids):
            # No typed layers, nothing to expand
            return

        # Avoid re-expanding an already-expanded structure unless the current list is still typed
        if self.model_params.get('_expanded_from_typed_layers', False) and not self._has_typed_layers(trapezoids):
            return

        # Expand from current trapezoids (no design_trapezoids exists)
        (
            expanded_traps,
            expanded_slds,
            expanded_layers,
            design_layers,
            design_traps,
            design_slds_list,
            _,
        ) = self._expand_typed_layers(self.model_params)

        # Preserve original (design) structure
        if 'design_trapezoids' not in self.model_params:
            self.model_params['design_layers'] = design_layers
            self.model_params['design_trapezoids'] = copy.deepcopy(design_traps)
            self.model_params['design_slds'] = copy.deepcopy(design_slds_list)

        # Replace with expanded (simulation) structure
        self.model_params['layers'] = int(expanded_layers)
        self.model_params['trapezoids'] = expanded_traps
        self.model_params['slds'] = expanded_slds
        self.model_params['_expanded_from_typed_layers'] = True
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary including SLD support.
        """
        if not hasattr(self, 'model_params'):
            return

        # Expand any typed layers (e.g., ellipse) into standard trapezoid segments for simulation
        self._ensure_expanded_model_params()
            
        # Update PAR from trapezoids
        trapezoids = self.model_params['trapezoids']
        if not hasattr(self, 'PAR') or self.PAR is None or self.PAR.shape[0] != len(trapezoids):
            self.PAR = np.zeros((len(trapezoids), 3))
            
        for i, trap in enumerate(trapezoids):
            self.PAR[i, 0] = trap.get('width')
            self.PAR[i, 1] = trap.get('height')
            tw = trap.get('twidth', None)
            self.PAR[i, 2] = np.nan if tw is None else tw

        # Ensure `self.layers` matches the simulation trapezoid count convention (len(trapezoids) == layers+1)
        self.layers = int(self.model_params.get('layers', max(0, len(trapezoids) - 1)))
        
        # Update global parameters
        self.DW = self.model_params['DW']
        self.I0 = self.model_params['I0']
        
        # Handle background - convert to array if needed
        bk_param = self.model_params['Bk']
        if isinstance(bk_param, list):
            self.Bk = np.array(bk_param)
        elif np.isscalar(bk_param):
            self.Bk = bk_param
        else:
            self.Bk = bk_param
        
        # Update SLD parameters
        if 'slds' in self.model_params:
            sld_values = self.model_params['slds']
            if isinstance(sld_values, list):
                self.sld_values = np.array(sld_values)
            else:
                self.sld_values = np.array([sld_values])
        
        # Update optional parameters
        if 'SLD' in self.model_params:
            self.SLD = self.model_params['SLD']
            
        if 'Pitch' in self.model_params:
            self.Pitch = self.model_params['Pitch']
            
        # Update SimPar - for array background, use first value for compatibility
        bk_scalar = self.Bk[0] if isinstance(self.Bk, np.ndarray) else self.Bk
        self.SimPar = np.append(self.PAR.ravel(), [self.I0, self.DW, bk_scalar])
        
        return True
    
    def process_imported_data(self):
        """
        Process imported data for trapezoid model and initialize array background.
        """
        try:
            # Initialize array background based on data dimensions
            if hasattr(self, 'Intensity') and self.Intensity is not None:
                n_columns = self.Intensity.shape[1]
                
                # Convert scalar background to array if needed
                if np.isscalar(self.Bk):
                    self.Bk = np.full(n_columns, self.Bk)
                    self.Bk_Initial = self.Bk.copy()
                elif isinstance(self.Bk, np.ndarray) and len(self.Bk) != n_columns:
                    # Resize background array if dimensions don't match
                    if len(self.Bk) == 1:
                        self.Bk = np.full(n_columns, self.Bk[0])
                    else:
                        # Interpolate or extend existing array
                        self.Bk = np.resize(self.Bk, n_columns)
                    self.Bk_Initial = self.Bk.copy()
            
            # For trapezoid model, run initial simulation
            self.SymCoordAssign()
            self.SimTrap_SM()
            self.SimInt_Initial = self.SimInt.copy() if hasattr(self, 'SimInt') else None
            self.GF = self.GF_calc(self.SimInt)
            self.GF_Initial = self.GF
            self.BIC = self.BIC_calc(self.GF)
            self.BIC_Initial = self.BIC
        except Exception as e:
            print(f"Warning: Error in trapezoid initialization: {str(e)}")
            print("Data import successful, but trapezoid initialization failed.")
    
    def initialize_optimization_params(self, param_limits=None):
        """
        Initialize optimization parameters with bounds including SLD support.
        """
        if not hasattr(self, 'model_params'):
            self.build_model_params_from_traditional()

        # Ensure typed layers (if any) are expanded before generating optimization params
        self._ensure_expanded_model_params()

        # Merge inline bounds (if present) into optimization config (explicit config wins)
        param_limits = self._merge_inline_bounds_into_optimization(param_limits)
            
        # Create default limits if not provided
        auto_generated = False
        if not param_limits:
            auto_generated = True
            param_limits = {}
            
            # Add trapezoid parameters (design-level when available; otherwise current trapezoids)
            traps_for_defaults = self.model_params.get('design_trapezoids', self.model_params['trapezoids'])
            for i, trap in enumerate(traps_for_defaults):
                param_limits[f'trap_{i}_width'] = {
                    'min': trap['width'] * 0.9,
                    'max': trap['width'] * 1.1,
                    'default': trap['width']
                }
                
                param_limits[f'trap_{i}_height'] = {
                    'min': trap['height'] * 0.9,
                    'max': trap['height'] * 1.1,
                    'default': trap['height']
                }
                
                # Add twidth parameter if present (used by SiGe model)
                if 'twidth' in trap and trap['twidth'] is not None:
                    param_limits[f'trap_{i}_twidth'] = {
                        'min': trap['twidth'] * 0.9,
                        'max': trap['twidth'] * 1.1,
                        'default': trap['twidth']
                    }

                # Curved-sidewall typed-layer optimizable params (user-facing & optimizable)
                # Non-optimizable slice controls are intentionally NOT added here.
                layer_type = trap.get('Layer_Type', None)
                if layer_type is not None and str(layer_type).strip().lower() == 'curved_sides':
                    if 'side_length' in trap:
                        v = float(trap.get('side_length', 0.0))
                        if v <= 0:
                            lo, hi = 0.0, max(1.0, abs(v) * 2.0)
                        else:
                            lo, hi = v * 0.9, v * 1.1
                        param_limits[f'trap_{i}_side_length'] = {
                            'min': float(lo),
                            'max': float(hi),
                            'default': float(v)
                        }
                    if 'tip_deflection' in trap:
                        v = float(trap.get('tip_deflection', 0.0))
                        dv = max(abs(v) * 0.1, 1e-6)
                        param_limits[f'trap_{i}_tip_deflection'] = {
                            'min': float(v - dv),
                            'max': float(v + dv),
                            'default': float(v)
                        }
            
            # Add global parameters
            param_limits['DW'] = {
                'min': self.DW * 0.9,
                'max': self.DW * 1.1,
                'default': self.DW
            }
            
            param_limits['I0'] = {
                'min': self.I0 * 0.9,
                'max': self.I0 * 1.1,
                'default': self.I0
            }
            
            # Add background parameters (one per column if array)
            if isinstance(self.Bk, np.ndarray):
                for i, bk_val in enumerate(self.Bk):
                    param_limits[f'Bk_{i}'] = {
                        'min': bk_val * 0.9,
                        'max': bk_val * 1.1,
                        'default': bk_val
                    }
            else:
                param_limits['Bk'] = {
                    'min': self.Bk * 0.9,
                    'max': self.Bk * 1.1,
                    'default': self.Bk
                }
        else:
            # Ensure default values are set for all provided parameters
            for param, limits in param_limits.items():
                if 'default' not in limits:
                    try:
                        # Get default value from current model state
                        default_value = self._get_current_parameter_value(param)
                        limits['default'] = default_value
                    except Exception as e:
                        # Fallback: use middle of min/max range
                        if 'min' in limits and 'max' in limits:
                            default_value = (limits['min'] + limits['max']) / 2
                            limits['default'] = default_value
                            print(f"WARNING: Could not get current value for {param}, using range midpoint: {default_value}")
                        else:
                            raise ValueError(f"Cannot determine default value for parameter {param}: {str(e)}")
        
        # Add SLD parameters only when auto-generating a full default optimization set.
        # If the user provided explicit bounds (either via param_limits or inline *_bounds),
        # we do NOT implicitly add extra optimizable parameters.
        if auto_generated and hasattr(self, 'sld_values'):
            for i, sld_val in enumerate(self.sld_values):
                param_name = f'sld_{i}'
                if param_name not in param_limits:
                    param_limits[param_name] = {
                        'min': max(0.1, sld_val * 0.5),
                        'max': sld_val * 2.0,
                        'default': sld_val
                    }
        
        # Update stored optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits
    
    def _ensure_defaults_in_params(self, params_to_optimize):
        """
        Ensure all optimization parameters have default values set.
        
        Parameters:
        -----------
        params_to_optimize : dict
            Dictionary of optimization parameters
            
        Returns:
        --------
        dict
            Updated parameters with defaults ensured
        """
        updated_params = {}
        
        for param_name, param_config in params_to_optimize.items():
            # Copy the existing configuration
            updated_config = param_config.copy()
            
            # Add default if missing
            if 'default' not in updated_config:
                try:
                    default_value = self._get_current_parameter_value(param_name)
                    updated_config['default'] = default_value
                    #print(f"INFO: Added missing default for {param_name}: {default_value}")
                except Exception as e:
                    # Fallback: use middle of min/max range
                    if 'min' in updated_config and 'max' in updated_config:
                        default_value = (updated_config['min'] + updated_config['max']) / 2
                        updated_config['default'] = default_value
                        print(f"WARNING: Could not get current value for {param_name}, using range midpoint: {default_value}")
                    else:
                        raise ValueError(f"Cannot determine default value for parameter {param_name}: {str(e)}")
            
            updated_params[param_name] = updated_config
        
        return updated_params
    
    def _extract_PAR_from_model_params(self):
        """
        Helper method to extract PAR array from model_params.
        Used by SymCoordAssign_SingleMaterial when PAR is not directly available.
        
        Returns:
        --------
        numpy.ndarray
            PAR array extracted from model_params
        """
        if not hasattr(self, 'model_params'):
            raise AttributeError("Missing required attribute: model_params")
            
        # Ensure any typed layers have been expanded before extracting PAR
        self._ensure_expanded_model_params()

        trapezoids = self.model_params['trapezoids']
        layers = int(self.model_params['layers'])
        
        # Create PAR array
        PAR = np.zeros([layers + 1, 3])
        for i, trap in enumerate(trapezoids):
            if i <= layers:
                PAR[i, 0] = trap.get('width')
                PAR[i, 1] = trap.get('height')
                tw = trap.get('twidth', None)
                PAR[i, 2] = np.nan if tw is None else tw
                
        return PAR
    
    def SymCoordAssign_SingleMaterial(self, PAR=None, layers=None):
        """
        Assigns trapezoid coordinates for a symmetric trapezoid with a single material.
        
        Parameters:
        -----------
        PAR : numpy.ndarray, optional
            Array with parameters for each layer
            If None, uses self.PAR
        layers : int, optional
            Number of layers in the trapezoid structure
            If None, uses self.layers
            
        Returns:
        --------
        numpy.ndarray
            Coordinate array for the trapezoid structure
            If called with self attributes, also sets self.Coord
        """
        try:
            # Determine whether to use passed parameters or class attributes
            using_self = False
            
            if PAR is None:
                if not hasattr(self, 'PAR'):
                    # Try to use model_params if available
                    if hasattr(self, 'model_params'):
                        PAR = self._extract_PAR_from_model_params()
                    else:
                        raise AttributeError("Missing required attribute: PAR")
                PAR = self.PAR
                using_self = True
                
            if layers is None:
                if not hasattr(self, 'layers'):
                    raise AttributeError("Missing required attribute: layers")
                layers = self.layers
                
            # Validate PAR dimensions
            if not isinstance(PAR, np.ndarray):
                raise TypeError("PAR must be a numpy array")
                
            if len(PAR) < layers + 1:
                raise ValueError(f"PAR array must have at least {layers + 1} rows, but has {len(PAR)}")
                
            # Check PAR shape
            if len(PAR.shape) < 2 or PAR.shape[1] < 2:
                raise ValueError(f"PAR must have at least 2 columns, but has shape {PAR.shape}")
            
            # Main calculation code
            # Center each segment independently about the global centerline, allowing width mismatches
            center = 0.5 * float(PAR[0, 0])
            Coord = np.zeros([layers + 1, 7, 1])
            for T in range(layers + 1):
                w_bottom = float(PAR[T, 0])
                h = float(PAR[T, 1])
                Coord[T, 2, 0] = h
                Coord[T, 3, 0] = 0
                Coord[T, 4, 0] = 1  # single material

                # Bottom edges (x1,x4)
                x_left = center - 0.5 * w_bottom
                x_right = center + 0.5 * w_bottom
                Coord[T, 0, 0] = x_left
                Coord[T, 1, 0] = x_right

                # Top width selection (twidth overrides; otherwise uses next width where available)
                if not np.isnan(PAR[T, 2]):
                    w_top = float(PAR[T, 2])
                else:
                    if T < layers:
                        w_top = float(PAR[T + 1, 0])
                    else:
                        w_top = w_bottom

                Coord[T, 5, 0] = center - 0.5 * w_top
                Coord[T, 6, 0] = center + 0.5 * w_top
            
            # If using self attributes, update self.Coord
            if using_self:
                self.Coord = Coord
                return True
                
            return Coord
        
        except Exception as e:
            print(f"Error in SymCoordAssign_SingleMaterial: {str(e)}")
            if using_self:
                return False
            return None
    
    def SymCoordAssign(self, PAR=None, layers=None, sld_values=None):
        """
        Alternative implementation with even clearer SLD assignment logic.
        Each coordinate index directly corresponds to its layer index.
        """
        try:
            # Parameter validation (same as above)
            using_self = False
            
            if PAR is None:
                if not hasattr(self, 'PAR'):
                    if hasattr(self, 'model_params'):
                        PAR = self._extract_PAR_from_model_params()
                    else:
                        raise AttributeError("Missing required attribute: PAR")
                PAR = self.PAR
                using_self = True
                
            if layers is None:
                if not hasattr(self, 'layers'):
                    raise AttributeError("Missing required attribute: layers")
                layers = self.layers
                
            if sld_values is not None:
                sld_array = np.array(sld_values, dtype=float)
            elif hasattr(self, 'sld_values'):
                sld_array = self.sld_values.copy()
            else:
                sld_array = np.ones(layers+1, dtype=float)
            
            # STRICT VALIDATION
            if len(sld_array)-1 != layers:
                raise ValueError(
                    f"SLD array length ({len(sld_array)-1}) must exactly match number of layers ({layers}). "
                    f"Each layer requires its own SLD value."
                )
            
            # Validate PAR
            if not isinstance(PAR, np.ndarray) or len(PAR) < layers + 1 or PAR.shape[1] < 2:
                raise ValueError("Invalid PAR array dimensions")
            
            # Initialize coordinate array
            Coord = np.zeros([layers + 1, 7, 1])
            
            # Center each segment independently about the global centerline, allowing width mismatches
            center = 0.5 * float(PAR[0, 0])

            for T in range(layers + 1):
                w_bottom = float(PAR[T, 0])
                h = float(PAR[T, 1])

                # Bottom edges
                Coord[T, 0, 0] = center - 0.5 * w_bottom
                Coord[T, 1, 0] = center + 0.5 * w_bottom
                Coord[T, 2, 0] = h
                Coord[T, 3, 0] = 0

                # Top width selection (twidth overrides; otherwise uses next width where available)
                if not np.isnan(PAR[T, 2]):
                    w_top = float(PAR[T, 2])
                else:
                    if T < layers:
                        w_top = float(PAR[T + 1, 0])
                    else:
                        w_top = w_bottom
                Coord[T, 5, 0] = center - 0.5 * w_top
                Coord[T, 6, 0] = center + 0.5 * w_top

                # SLD assignment: one value per trapezoid entry
                Coord[T, 4, 0] = sld_array[T]
            
            if using_self:
                self.Coord = Coord
                return True
                
            return Coord
            
        except Exception as e:
            print(f"Error in SymCoordAssign_Alternative: {str(e)}")
            if using_self:
                return False
            return None

    # ----------------------------
    # Curved sidewall typed layer
    # ----------------------------
    @staticmethod
    def _curved_sides_incompressible_diving_board_local(length, thickness, tip_deflection, n_points=500):
        """
        Local cantilever with constant thickness (top/bottom from centerline + normal offset).
        Ported from the final notebook implementation.
        """
        length = float(length)
        thickness = float(thickness)
        tip_deflection = float(tip_deflection)
        n_points = int(n_points)
        if n_points < 3:
            n_points = 3
        if length <= 0:
            s = np.zeros(n_points, dtype=float)
            y = np.linspace(0.5 * thickness, -0.5 * thickness, n_points)
            return s, y, s, y

        s = np.linspace(0.0, length, n_points)
        y_center = -(tip_deflection / length**3) * s**2 * (3.0 * length - s)
        dy_ds = -(tip_deflection / length**3) * s * (6.0 * length - 3.0 * s)

        theta = np.arctan(dy_ds)
        normal_x = -np.sin(theta)
        normal_y = np.cos(theta)

        x_top = s + 0.5 * thickness * normal_x
        y_top = y_center + 0.5 * thickness * normal_y
        x_bottom = s - 0.5 * thickness * normal_x
        y_bottom = y_center - 0.5 * thickness * normal_y
        return x_top, y_top, x_bottom, y_bottom

    @staticmethod
    def _curved_sides_build_layer_center_trapezoid_incompressible_sides(
        center_x,
        center_y_bottom,
        center_height,
        center_bottom_width,
        center_top_width,
        side_length,
        side_tip_deflection,
        n_side_pts=500,
    ):
        """
        Build a closed polygon for the full layer boundary (center trapezoid + left/right incompressible sides),
        plus metadata used by the smart side-only slicing.
        """
        yb = float(center_y_bottom)
        yt = float(center_y_bottom + center_height)

        xbL = float(center_x - 0.5 * center_bottom_width)
        xbR = float(center_x + 0.5 * center_bottom_width)
        xtL = float(center_x - 0.5 * center_top_width)
        xtR = float(center_x + 0.5 * center_top_width)

        # Side cross-section thickness at attachment is length of center side edge
        t_left = float(np.hypot(xtL - xbL, yt - yb))
        t_right = float(np.hypot(xtR - xbR, yt - yb))

        cxL, cyL = 0.5 * (xtL + xbL), 0.5 * (yt + yb)
        cxR, cyR = 0.5 * (xtR + xbR), 0.5 * (yt + yb)

        xl_t, yl_t, xl_b, yl_b = SiGeModel._curved_sides_incompressible_diving_board_local(
            side_length, t_left, side_tip_deflection, n_points=n_side_pts
        )
        xr_t, yr_t, xr_b, yr_b = SiGeModel._curved_sides_incompressible_diving_board_local(
            side_length, t_right, side_tip_deflection, n_points=n_side_pts
        )

        # Map local to global
        x_left_top = cxL - xl_t
        y_left_top = cyL + yl_t
        x_left_bottom = cxL - xl_b
        y_left_bottom = cyL + yl_b

        x_right_top = cxR + xr_t
        y_right_top = cyR + yr_t
        x_right_bottom = cxR + xr_b
        y_right_bottom = cyR + yr_b

        # Clockwise polygon
        px, py = [], []
        px.extend(x_left_bottom[::-1])
        py.extend(y_left_bottom[::-1])
        px.extend([xbL, xbR])
        py.extend([yb, yb])
        px.extend(x_right_bottom)
        py.extend(y_right_bottom)
        px.extend([x_right_bottom[-1], x_right_top[-1]])
        py.extend([y_right_bottom[-1], y_right_top[-1]])
        px.extend(x_right_top[::-1])
        py.extend(y_right_top[::-1])
        px.extend([xtR, xtL])
        py.extend([yt, yt])
        px.extend(x_left_top)
        py.extend(y_left_top)
        px.extend([x_left_top[-1], x_left_bottom[-1]])
        py.extend([y_left_top[-1], y_left_bottom[-1]])

        return {
            "poly_x": np.array(px, dtype=float),
            "poly_y": np.array(py, dtype=float),
            "center": {"xbL": xbL, "xbR": xbR, "xtL": xtL, "xtR": xtR, "yb": yb, "yt": yt},
            "top_side_min": float(min(np.min(y_left_top), np.min(y_right_top))),
            "bottom_side_min": float(min(np.min(y_left_bottom), np.min(y_right_bottom))),
        }

    @staticmethod
    def _curved_sides_polygon_x_hits(poly_x, poly_y, y_target):
        xs = []
        for i in range(len(poly_x) - 1):
            x1, y1 = poly_x[i], poly_y[i]
            x2, y2 = poly_x[i + 1], poly_y[i + 1]
            if (y1 <= y_target <= y2) or (y2 <= y_target <= y1):
                if abs(y2 - y1) < 1e-12:
                    continue
                t = (y_target - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))
        return xs

    @staticmethod
    def _curved_sides_x_on_segment_at_y(x1, y1, x2, y2, y):
        if (y < min(y1, y2)) or (y > max(y1, y2)) or abs(y2 - y1) < 1e-12:
            return None
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    @staticmethod
    def _curved_sides_center_edge_x_at_y(center, y, side="left"):
        if side == "left":
            return SiGeModel._curved_sides_x_on_segment_at_y(center["xbL"], center["yb"], center["xtL"], center["yt"], y)
        return SiGeModel._curved_sides_x_on_segment_at_y(center["xbR"], center["yb"], center["xtR"], center["yt"], y)

    @staticmethod
    def _curved_sides_side_outer_and_inner_x_at_y(layer, y, side="left"):
        xs = sorted(SiGeModel._curved_sides_polygon_x_hits(layer["poly_x"], layer["poly_y"], y))
        if len(xs) < 2:
            return None
        c = layer["center"]
        x_center = 0.5 * (c["xbL"] + c["xbR"])
        if side == "left":
            x_outer = xs[0]
            x_center_edge = SiGeModel._curved_sides_center_edge_x_at_y(c, y, side="left")
            left_candidates = [x for x in xs if x <= x_center + 1e-9 and x > x_outer + 1e-9]
            x_inner = x_center_edge if x_center_edge is not None else (max(left_candidates) if left_candidates else None)
            if x_inner is None or not (x_outer < x_inner - 1e-9):
                return None
            return x_outer, x_inner  # outer, inner
        x_outer = xs[-1]
        x_center_edge = SiGeModel._curved_sides_center_edge_x_at_y(c, y, side="right")
        right_candidates = [x for x in xs if x >= x_center - 1e-9 and x < x_outer - 1e-9]
        x_inner = x_center_edge if x_center_edge is not None else (min(right_candidates) if right_candidates else None)
        if x_inner is None or not (x_inner < x_outer - 1e-9):
            return None
        return x_inner, x_outer  # inner, outer

    @staticmethod
    def _curved_sides_nonuniform_levels(y0, y1, n, power=1.6):
        if n <= 0 or y1 <= y0:
            return np.array([])
        u = np.linspace(0.0, 1.0, n + 1)
        u2 = 0.5 * (u**power + (1.0 - (1.0 - u) ** power))
        return y0 + (y1 - y0) * u2

    @staticmethod
    def _curved_sides_slice_band(layer, y_levels, side="left"):
        traps = []
        for i in range(len(y_levels) - 1):
            y0 = float(y_levels[i])
            y1 = float(y_levels[i + 1])
            s0 = SiGeModel._curved_sides_side_outer_and_inner_x_at_y(layer, y0, side=side)
            s1 = SiGeModel._curved_sides_side_outer_and_inner_x_at_y(layer, y1, side=side)
            if s0 is None or s1 is None:
                continue
            if side == "left":
                xL0, xR0 = s0  # outer, inner
                xL1, xR1 = s1
            else:
                xL0, xR0 = s0  # inner, outer
                xL1, xR1 = s1
            traps.append(
                {
                    "y_bottom": y0,
                    "y_top": y1,
                    "x_left_bottom": float(xL0),
                    "x_right_bottom": float(xR0),
                    "x_left_top": float(xL1),
                    "x_right_top": float(xR1),
                    "height": float(y1 - y0),
                }
            )
        return traps

    @staticmethod
    def _curved_sides_smart_slice_sides_by_regions(layer, target_total_per_side=8, middle_fixed=1, nonuniform=True, eps=1e-6):
        c = layer["center"]
        yb = float(c["yb"])
        yt = float(c["yt"])

        y_top_min = float(layer.get("top_side_min", yt))
        y_bot_min = float(layer.get("bottom_side_min", np.min(layer["poly_y"])))

        top_lo, top_hi = min(y_top_min, yt), max(y_top_min, yt)
        mid_lo, mid_hi = yb, min(yt, y_top_min)
        bot_lo, bot_hi = y_bot_min, yb

        top_exists = (top_hi - top_lo) > eps
        mid_exists = (mid_hi - mid_lo) > eps
        bot_exists = (bot_hi - bot_lo) > eps

        n_mid = int(middle_fixed) if mid_exists else 0
        curved_budget = max(int(target_total_per_side) - n_mid, 1)

        top_span = (top_hi - top_lo) if top_exists else 0.0
        bot_span = (bot_hi - bot_lo) if bot_exists else 0.0

        if (not top_exists) and bot_exists:
            n_top, n_bot = 0, curved_budget
        elif top_exists and (not bot_exists):
            n_top, n_bot = curved_budget, 0
        elif top_exists and bot_exists:
            frac_top = top_span / max(top_span + bot_span, eps)
            n_top = int(round(curved_budget * frac_top))
            if curved_budget > 1:
                n_top = max(1, min(curved_budget - 1, n_top))
            else:
                n_top = 1
            n_bot = curved_budget - n_top
        else:
            n_top, n_bot = 0, 0

        def levels(y0, y1, n):
            if n <= 0 or y1 <= y0 + eps:
                return np.array([])
            return SiGeModel._curved_sides_nonuniform_levels(y0, y1, n) if nonuniform else np.linspace(y0, y1, n + 1)

        lv_top = levels(top_lo, top_hi, n_top)
        lv_mid = levels(mid_lo, mid_hi, n_mid)
        lv_bot = levels(bot_lo, bot_hi, n_bot)

        return {
            "anchors": {"top_lo": top_lo, "top_hi": top_hi, "mid_lo": mid_lo, "mid_hi": mid_hi, "bot_lo": bot_lo, "bot_hi": bot_hi},
            "alloc": {"n_top": n_top, "n_middle": n_mid, "n_bottom": n_bot},
            "left": {
                "top": SiGeModel._curved_sides_slice_band(layer, lv_top, side="left") if len(lv_top) else [],
                "middle": SiGeModel._curved_sides_slice_band(layer, lv_mid, side="left") if len(lv_mid) else [],
                "bottom": SiGeModel._curved_sides_slice_band(layer, lv_bot, side="left") if len(lv_bot) else [],
            },
            "right": {
                "top": SiGeModel._curved_sides_slice_band(layer, lv_top, side="right") if len(lv_top) else [],
                "middle": SiGeModel._curved_sides_slice_band(layer, lv_mid, side="right") if len(lv_mid) else [],
                "bottom": SiGeModel._curved_sides_slice_band(layer, lv_bot, side="right") if len(lv_bot) else [],
            },
        }

    @staticmethod
    def _curved_sides_flatten_smart_trapezoids(smart_dict):
        rows = []
        for side in ("left", "right"):
            for region in ("top", "middle", "bottom"):
                for i, t in enumerate(smart_dict.get(side, {}).get(region, [])):
                    row = {"side": side, "region": region, "slice_idx": int(i)}
                    row.update({k: float(t[k]) for k in ("y_bottom", "y_top", "x_left_bottom", "x_right_bottom", "x_left_top", "x_right_top", "height")})
                    rows.append(row)
        return rows

    @staticmethod
    def _curved_sides_traps_to_coord(traps, z_offset, sld):
        if not traps:
            return None
        n = len(traps)
        Coord = np.zeros([n, 7, 1], dtype=float)
        for i, t in enumerate(traps):
            Coord[i, 0, 0] = float(t["x_left_bottom"])
            Coord[i, 1, 0] = float(t["x_right_bottom"])
            Coord[i, 2, 0] = float(t["height"])
            Coord[i, 3, 0] = 0.0
            Coord[i, 4, 0] = float(sld)
            Coord[i, 5, 0] = float(t["x_left_top"])
            Coord[i, 6, 0] = float(t["x_right_top"])
        Coord[0, 3, 0] = float(z_offset)
        return Coord

    def _curved_sides_compute_for_expanded_stack(self, design_traps, expanded_design_index, PAR, layers, expanded_slds=None):
        coord_by_design = {"left": {}, "right": {}}
        traps_by_design = {"left": {}, "right": {}}
        flat_rows = []
        if not design_traps or not expanded_design_index or PAR is None:
            return coord_by_design, traps_by_design, flat_rows

        center_x = 0.5 * float(PAR[0, 0])
        heights = PAR[: int(layers) + 1, 1].astype(float)
        cum_heights = np.concatenate(([0.0], np.cumsum(heights)))

        design_to_expanded = {}
        for exp_idx, d_idx in enumerate(expanded_design_index):
            design_to_expanded.setdefault(int(d_idx), []).append(int(exp_idx))

        for design_idx, d in enumerate(design_traps):
            layer_type = d.get("Layer_Type", None)
            if layer_type is None or str(layer_type).strip().lower() != "curved_sides":
                continue

            side_length = float(d.get("side_length", 0.0))
            tip_deflection = float(d.get("tip_deflection", 0.0))
            slice_budget = int(d.get("slice_budget", 8))
            middle_fixed = int(d.get("middle_fixed", 1))
            nonuniform = bool(d.get("nonuniform", True))
            n_side_pts = int(d.get("n_side_pts", 500))

            exp_indices = design_to_expanded.get(int(design_idx), [])
            if not exp_indices:
                continue

            traps_left_all = []
            traps_right_all = []

            for exp_idx in exp_indices:
                if exp_idx > layers:
                    continue
                yb = float(cum_heights[exp_idx])
                h = float(PAR[exp_idx, 1])
                if h <= 0:
                    continue

                w0 = float(PAR[exp_idx, 0])
                if not np.isnan(PAR[exp_idx, 2]):
                    w1 = float(PAR[exp_idx, 2])
                else:
                    w1 = float(PAR[exp_idx + 1, 0]) if exp_idx < layers else w0

                layer = self._curved_sides_build_layer_center_trapezoid_incompressible_sides(
                    center_x=center_x,
                    center_y_bottom=yb,
                    center_height=h,
                    center_bottom_width=w0,
                    center_top_width=w1,
                    side_length=side_length,
                    side_tip_deflection=tip_deflection,
                    n_side_pts=n_side_pts,
                )
                smart = self._curved_sides_smart_slice_sides_by_regions(
                    layer,
                    target_total_per_side=slice_budget,
                    middle_fixed=middle_fixed,
                    nonuniform=nonuniform,
                )

                rows = self._curved_sides_flatten_smart_trapezoids(smart)
                for r in rows:
                    r2 = dict(r)
                    r2["design_idx"] = int(design_idx)
                    r2["expanded_idx"] = int(exp_idx)
                    flat_rows.append(r2)

                for region in ("top", "middle", "bottom"):
                    traps_left_all.extend(smart["left"][region])
                    traps_right_all.extend(smart["right"][region])

            # SLD: same as the corresponding center layer
            sld_val = None
            if expanded_slds is not None:
                try:
                    sld_val = float(expanded_slds[exp_indices[0]])
                except Exception:
                    sld_val = None
            if sld_val is None:
                sld_val = float(d.get("sld", 1.0))

            z0 = min([t["y_bottom"] for t in traps_left_all], default=None)
            if z0 is None:
                z0 = min([t["y_bottom"] for t in traps_right_all], default=0.0)

            Coord_left = self._curved_sides_traps_to_coord(traps_left_all, z_offset=z0, sld=sld_val)
            Coord_right = self._curved_sides_traps_to_coord(traps_right_all, z_offset=z0, sld=sld_val)

            coord_by_design["left"][int(design_idx)] = Coord_left
            coord_by_design["right"][int(design_idx)] = Coord_right
            traps_by_design["left"][int(design_idx)] = traps_left_all
            traps_by_design["right"][int(design_idx)] = traps_right_all

            if abs(tip_deflection) > 0 and (Coord_left is None or Coord_right is None):
                print(
                    f"WARNING: curved_sides design entry {design_idx} produced no sidewall slices "
                    f"(side_length={side_length}, tip_deflection={tip_deflection})."
                )

        return coord_by_design, traps_by_design, flat_rows

    
    def _get_current_parameter_value(self, param_name):
        """
        Get the current value of a parameter from the model including SLD support.
        """
        if param_name.startswith('sld_'):
            sld_idx = int(param_name.split('_')[1])
            if hasattr(self, 'sld_values') and sld_idx < len(self.sld_values):
                return float(self.sld_values[sld_idx])
            else:
                raise ValueError(f"SLD index {sld_idx} out of range")
        
        elif param_name.startswith('trap_'):
            # Trapezoid (design-layer) parameter: width, height, twidth, depth, etc.
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = '_'.join(parts[2:])

            if not hasattr(self, 'model_params'):
                raise AttributeError("Missing required attribute: model_params")

            # Prefer design_trapezoids if present (typed-layer aware, including ellipse depth)
            if 'design_trapezoids' in self.model_params:
                design_traps = self.model_params['design_trapezoids']
                if trap_idx >= len(design_traps):
                    raise IndexError(f"design_trapezoids index {trap_idx} out of range")
                return design_traps[trap_idx][param_type]

            # Fallback: use current trapezoids list (pure trapezoid models)
            traps = self.model_params.get('trapezoids', None)
            if traps is None or trap_idx >= len(traps):
                raise IndexError(f"trapezoids index {trap_idx} out of range")
            return traps[trap_idx][param_type]
        
        elif param_name.startswith('Bk_'):
            bk_idx = int(param_name.split('_')[1])
            if isinstance(self.Bk, np.ndarray):
                return self.Bk[bk_idx]
            else:
                return self.Bk
        
        elif param_name == 'Bk':
            if isinstance(self.Bk, np.ndarray):
                return self.Bk[0]  # Return first element for scalar case
            else:
                return self.Bk
        
        elif param_name in ['DW', 'I0']:
            return getattr(self, param_name)
        
        else:
            # Try to get from model_params
            if hasattr(self, 'model_params') and param_name in self.model_params:
                return self.model_params[param_name]
            else:
                raise ValueError(f"Unknown parameter: {param_name}")

    def _set_parameter_value(self, param_name, value):
        """
        Set a parameter value in the model including SLD support.
        """
        if param_name.startswith('sld_'):
            sld_idx = int(param_name.split('_')[1])
            if hasattr(self, 'sld_values') and sld_idx < len(self.sld_values):
                self.sld_values[sld_idx] = value
                # Update model_params if it exists
                if hasattr(self, 'model_params') and 'slds' in self.model_params:
                    self.model_params['slds'][sld_idx] = value
            else:
                raise ValueError(f"SLD index {sld_idx} out of range")
        
        elif param_name.startswith('trap_'):
            # Trapezoid (design-layer) parameter
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = '_'.join(parts[2:])

            if not hasattr(self, 'model_params'):
                raise AttributeError("Missing required attribute: model_params")

            # Prefer updating design_trapezoids if present so that typed layers (ellipse)
            # remain under design control and are re-expanded consistently.
            if 'design_trapezoids' in self.model_params:
                design_traps = self.model_params['design_trapezoids']
                if trap_idx >= len(design_traps):
                    raise IndexError(f"design_trapezoids index {trap_idx} out of range")
                design_traps[trap_idx][param_type] = value
            else:
                traps = self.model_params.get('trapezoids', None)
                if traps is None or trap_idx >= len(traps):
                    raise IndexError(f"trapezoids index {trap_idx} out of range")
                traps[trap_idx][param_type] = value
            
        elif param_name.startswith('Bk_'):
            # Background parameter for specific column
            bk_idx = int(param_name.split('_')[1])
            if isinstance(self.Bk, np.ndarray):
                self.Bk[bk_idx] = value
            else:
                # Convert scalar to array if needed
                n_columns = getattr(self, 'Intensity', np.array([[0]])).shape[1]
                self.Bk = np.full(n_columns, self.Bk)
                self.Bk[bk_idx] = value
                
        elif param_name == 'Bk':
            # Scalar background parameter
            self.Bk = value
            
        else:
            # Global parameter (DW, I0) or other model parameter
            if hasattr(self, param_name):
                setattr(self, param_name, value)
            if hasattr(self, 'model_params'):
                self.model_params[param_name] = value
        
        # Update traditional/expanded parameters from model_params after any change
        self.update_traditional_from_model_params()
            
    def _initialize_sld_values(self):
        """
        Initialize SLD values from various sources, with sensible defaults.
        FIXED: Ensures SLD values are always float dtype for mathematical operations.
        """
        # For trapezoids, we need SLD values for each LAYER (trapezoid), not each vertex
        n_sld_values = self.layers + 1 # Number of actual trapezoids/layers
        
        # Priority order: model_params['slds'] > SLD parameter > default values
        if hasattr(self, 'model_params') and 'slds' in self.model_params:
            # Use SLD values from model_params (main approach)
            sld_values = self.model_params['slds']
            if isinstance(sld_values, list):
                # FIXED: Explicitly convert to float dtype
                self.sld_values = np.array(sld_values, dtype=float)
            else:
                # FIXED: Ensure single values are also float
                self.sld_values = np.array([float(sld_values)])
                
        elif hasattr(self, 'SLD') and self.SLD is not None:
            # Use legacy SLD parameter for backward compatibility
            if np.isscalar(self.SLD):
                # FIXED: Use float dtype
                self.sld_values = np.full(n_sld_values, float(self.SLD))
            else:
                # FIXED: Convert array to float dtype
                self.sld_values = np.array(self.SLD, dtype=float)
                
        else:
            # Default: all SLDs = 1.0 (single material behavior)
            # FIXED: Use float dtype for defaults
            self.sld_values = np.ones(n_sld_values, dtype=float)
        
        # Ensure correct array size
        if len(self.sld_values) != n_sld_values:
            if len(self.sld_values) == 1:
                # Extend single value to all layers
                # FIXED: Maintain float dtype
                self.sld_values = np.full(n_sld_values, float(self.sld_values[0]))
            else:
                # Resize array to correct length
                # FIXED: Ensure float dtype after resize
                self.sld_values = np.resize(self.sld_values, n_sld_values).astype(float)
                print(f"Warning: Resized SLD array to {n_sld_values} elements for {self.layers} layers")
    
    def FreeFormTrapezoid(self, Coord=None, layers=None, Qx=None, Qz=None):
        """
        Calculates the form factor for a free-form trapezoid structure.
        
        Parameters:
        -----------
        Coord : numpy.ndarray, optional
            Coordinate array with trapezoid coordinates and parameters
            If None, uses self.Coord
        layers : int, optional
            Number of layers in the trapezoid structure
            If None, uses self.layers
        Qx : numpy.ndarray, optional
            X-component of scattering vector, 2D array
            If None, uses self.Qx
        Qz : numpy.ndarray, optional
            Z-component of scattering vector, 2D array
            If None, uses self.Qz
        
        Returns:
        --------
        numpy.ndarray
            The calculated form factor
            If called with self attributes, also sets self.form
        """
        try:
            # Determine whether to use passed parameters or class attributes
            using_self = False
            
            if Coord is None:
                # Check if required attributes exist
                if not hasattr(self, 'Coord'):
                    raise AttributeError("Missing required attribute: Coord")
                Coord = self.Coord
                using_self = True
                
            if layers is None:
                if not hasattr(self, 'layers'):
                    raise AttributeError("Missing required attribute: layers")
                layers = self.layers
                
            if Qx is None:
                if not hasattr(self, 'Qx'):
                    raise AttributeError("Missing required scattering vector attribute: Qx")
                Qx = self.Qx
                
            if Qz is None:
                if not hasattr(self, 'Qz'):
                    raise AttributeError("Missing required scattering vector attribute: Qz")
                Qz = self.Qz
                
            # Check if arrays have proper dimensions
            if len(Qx.shape) != 2:
                raise ValueError(f"Qx must be a 2D array, got shape {Qx.shape}")
                
            # Validate Coord shape for indexing
            if int(layers) + 1 > len(Coord):
                raise IndexError(f"Not enough rows in Coord ({len(Coord)}) for {int(layers)} layers")
            
            # Initialize height values and form factor array
            H1 = Coord[0, 3, 0]
            H2 = H1
            form = np.zeros([len(Qx[:,1]), len(Qx[1,:])])
            
            # Calculate form factor
            for i in range(int(layers)+1):
                H2 = H2 + Coord[i, 2, 0]
                if i > 0:
                    H1 = H1 + Coord[i-1, 2, 0]
                    
                x1 = Coord[i, 0, 0]
                x4 = Coord[i, 1, 0]
                x2 = Coord[i, 5, 0]
                x3 = Coord[i, 6, 0]
                
                # Avoid division by zero
                x2 = x1 - 1e-6 if np.isclose(x2, x1) else x2
                x4 = x3 - 1e-6 if np.isclose(x4, x3) else x4
                
                SL = Coord[i, 2, 0] / (x2 - x1)
                SR = -Coord[i, 2, 0] / (x4 - x3)
                
                A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
                A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
                form = form + (1j/Qx)*(A1-A2)*Coord[i, 4, 0]
            
            # If using self attributes, update self.form
            if using_self:
                self.form = form
                
            return form
            
        except Exception as e:
            print(f"Error in FreeFormTrapezoid: {str(e)}")
            if using_self:
                self.form = None
            return None
    
    def SimTrap_SM(self, PAR=None, layers=None, Qx=None, Qz=None, DW=None, I0=None, Bk=None):
        """
        Enhanced simulation that uses the new coordinate assignment function.
        """
        try:
            # Determine whether to use passed parameters or class attributes
            using_self = False
            
            if PAR is None:
                if not hasattr(self, 'PAR'):
                    if hasattr(self, 'model_params'):
                        PAR = self._extract_PAR_from_model_params()
                    else:
                        if not hasattr(self, 'Coord'):
                            raise AttributeError("Missing required attributes: PAR and Coord")
                else:
                    PAR = self.PAR
                using_self = True
                
            if layers is None:
                if not hasattr(self, 'layers'):
                    raise AttributeError("Missing required attribute: layers")
                layers = self.layers
                
            if Qx is None:
                if not hasattr(self, 'Qx'):
                    raise AttributeError("Missing required attribute: Qx")
                Qx = self.Qx
                
            if Qz is None:
                if not hasattr(self, 'Qz'):
                    raise AttributeError("Missing required attribute: Qz")
                Qz = self.Qz
                
            if DW is None:
                if not hasattr(self, 'DW'):
                    raise AttributeError("Missing required attribute: DW")
                DW = self.DW
                
            if I0 is None:
                if not hasattr(self, 'I0'):
                    raise AttributeError("Missing required attribute: I0")
                I0 = self.I0
                
            if Bk is None:
                if not hasattr(self, 'Bk'):
                    raise AttributeError("Missing required attribute: Bk")
                Bk = self.Bk

            # Always prefer design_trapezoids if it exists (regardless of constraints).
            # Apply constraints if they exist, then expand typed layers.
            constraints = None
            if hasattr(self, 'model_params') and isinstance(self.model_params, dict):
                constraints = self.model_params.get('constraints', None)

            # Check if we should use design_trapezoids (design_slds is optional)
            use_design = 'design_trapezoids' in self.model_params
            
            if use_design:
                # Get design_slds if available, otherwise fall back to slds
                design_slds = self.model_params.get('design_slds', None)
                if design_slds is None:
                    design_slds = self.model_params.get('slds', None)
                
                # Build design-level params dict
                tmp_design = {
                    'design_trapezoids': [t.copy() for t in self.model_params['design_trapezoids']],
                    'design_slds': copy.deepcopy(design_slds) if design_slds is not None else None,
                    'DW': self.model_params.get('DW', DW),
                    'I0': self.model_params.get('I0', I0),
                    'Bk': self.model_params.get('Bk', Bk),
                    'layers': self.model_params.get('design_layers', len(self.model_params['design_trapezoids']) - 1),
                }
                # Apply constraints if they exist
                if constraints:
                    self._apply_constraints(tmp_design, constraints)
                tmp_model_params = {
                    'trapezoids': tmp_design['design_trapezoids'],
                    'layers': tmp_design['layers'],
                    'slds': tmp_design.get('design_slds', None),
                }
                expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(tmp_model_params)
                # Override PAR/layers for this simulation call
                temp_PAR = np.zeros((expanded_layers + 1, 3))
                for ii, trap in enumerate(expanded_traps):
                    if ii <= expanded_layers:
                        temp_PAR[ii, 0] = trap.get('width')
                        temp_PAR[ii, 1] = trap.get('height')
                        tw = trap.get('twidth', None)
                        temp_PAR[ii, 2] = np.nan if tw is None else tw
                PAR = temp_PAR
                layers = expanded_layers
                # Also override SLDs by passing through SymCoordAssign via sld_values
                # (SimTrap_SM calls SymCoordAssign(PAR,layers) which will use self.sld_values;
                # keep self.sld_values consistent for this call.)
                try:
                    self.sld_values = np.array(expanded_slds, dtype=float)
                except Exception:
                    pass
            elif constraints:
                # No design_trapezoids but constraints exist - apply to trapezoids
                tmp_model_params = {
                    'trapezoids': [t.copy() for t in self.model_params.get('trapezoids', [])],
                    'layers': int(self.model_params.get('layers', max(0, len(self.model_params.get('trapezoids', [])) - 1))),
                    'slds': copy.deepcopy(self.model_params.get('slds', None)),
                }
                self._apply_constraints(tmp_model_params, constraints)
                expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(tmp_model_params)
                # Override PAR/layers for this simulation call
                temp_PAR = np.zeros((expanded_layers + 1, 3))
                for ii, trap in enumerate(expanded_traps):
                    if ii <= expanded_layers:
                        temp_PAR[ii, 0] = trap.get('width')
                        temp_PAR[ii, 1] = trap.get('height')
                        tw = trap.get('twidth', None)
                        temp_PAR[ii, 2] = np.nan if tw is None else tw
                PAR = temp_PAR
                layers = expanded_layers
                # Also override SLDs
                try:
                    self.sld_values = np.array(expanded_slds, dtype=float)
                except Exception:
                    pass
            
            # Generate coordinates if PAR is provided - uses current SLD values
            if PAR is not None:
                Coord = self.SymCoordAssign(PAR, layers)
                if Coord is None or (using_self and Coord is False):
                    raise RuntimeError("Failed to assign coordinates in SymCoordAssign")
            else:
                # Use existing Coord
                if not hasattr(self, 'Coord'):
                    raise AttributeError("Missing required attribute: Coord")
                Coord = self.Coord
            
            # Calculate form factor for the center stack
            form_center = self.FreeFormTrapezoid(Coord, layers, Qx, Qz)
            if form_center is None:
                raise RuntimeError("Failed to calculate form factor in FreeFormTrapezoid")

            # Curved sidewalls (typed design layer): coherent complex sum
            form_total = form_center
            self.Coord_curved = {"left": {}, "right": {}}
            self.curved_sidewall_trapezoids = {"left": {}, "right": {}}
            self.curved_sidewall_trapezoid_rows = []

            try:
                if use_design:
                    # tmp_design['design_trapezoids'] reflects constrained design values
                    coord_by_design, traps_by_design, flat_rows = self._curved_sides_compute_for_expanded_stack(
                        design_traps=tmp_design["design_trapezoids"],
                        expanded_design_index=expanded_design_index,
                        PAR=PAR,
                        layers=layers,
                        expanded_slds=expanded_slds,
                    )
                    self.Coord_curved = coord_by_design
                    self.curved_sidewall_trapezoids = traps_by_design
                    self.curved_sidewall_trapezoid_rows = flat_rows

                    for d_idx, Cleft in coord_by_design.get("left", {}).items():
                        if Cleft is None:
                            continue
                        form_total = form_total + self.FreeFormTrapezoid(Cleft, len(Cleft) - 1, Qx, Qz)
                    for d_idx, Cright in coord_by_design.get("right", {}).items():
                        if Cright is None:
                            continue
                        form_total = form_total + self.FreeFormTrapezoid(Cright, len(Cright) - 1, Qx, Qz)
            except Exception as e:
                print(f"WARNING: curved sidewall contribution failed: {str(e)}")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qx, 2) + np.power(Qz, 2)) * np.power(DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to coherent form factor
            Formfactor = form_total * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity with array background support
            intensity_base = np.power(Formfactor, 2) * I0
            
            if isinstance(Bk, np.ndarray):
                # Array background - broadcast across columns
                if len(Bk) != intensity_base.shape[1]:
                    raise ValueError(f"Background array length ({len(Bk)}) must match number of columns ({intensity_base.shape[1]})")
                
                # Add background to each column
                SimInt = intensity_base + Bk[np.newaxis, :]
            else:
                # Scalar background
                SimInt = intensity_base + Bk
            
            # If using self attributes, update self.SimInt
            if using_self:
                self.SimInt = SimInt
                
            return SimInt
       
        except Exception as e:
            print(f"Error in SimTrap_SM: {str(e)}")
            if using_self:
                self.SimInt = None
            return None
    
    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None):
        """
        Enhanced goodness of fit calculation with SLD support.
        """
        try:
            # Validate input parameters
            if optimization_values is None or not isinstance(optimization_values, np.ndarray):
                raise TypeError("optimization_values must be a numpy array")
                
            # Determine whether to use passed parameters or class attributes
            if param_names is None:
                if not hasattr(self, 'param_names'):
                    raise AttributeError("Missing required attribute: param_names")
                param_names = self.param_names
                
            if Intensity is None:
                if not hasattr(self, 'Intensity'):
                    raise AttributeError("Missing required attribute: Intensity")
                Intensity = self.Intensity
                
            if Qx is None:
                if not hasattr(self, 'Qx'):
                    raise AttributeError("Missing required attribute: Qx")
                Qx = self.Qx
                
            if Qz is None:
                if not hasattr(self, 'Qz'):
                    raise AttributeError("Missing required attribute: Qz")
                Qz = self.Qz
                
            # Check if we have enough values for parameters
            if len(optimization_values) != len(param_names):
                raise ValueError(f"Number of optimization values ({len(optimization_values)}) must match number of parameter names ({len(param_names)})")
            
            # Create a copy of the current model parameters
            params = self.model_params.copy()
            
            # Initialize background array
            if isinstance(self.Bk, np.ndarray):
                temp_Bk = self.Bk.copy()
            else:
                temp_Bk = self.Bk
            
            # Initialize design-level SLD array placeholder (filled below)
            temp_sld_values = None
            
            # Update parameters with optimization values
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid (design-layer) parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = '_'.join(parts[2:])

                    # We do not mutate self.model_params here; changes are applied to
                    # a local design-level trapezoid list constructed below.
                    if 'design_trapezoids' in params:
                        # Ensure local copy exists
                        if 'design_trapezoids' not in params or params['design_trapezoids'] is self.model_params.get('design_trapezoids'):
                            params['design_trapezoids'] = [t.copy() for t in self.model_params['design_trapezoids']]
                        params['design_trapezoids'][trap_idx][param_type] = optimization_values[i]
                    else:
                        if 'trapezoids' not in params or params['trapezoids'] is self.model_params['trapezoids']:
                            params['trapezoids'] = [trap.copy() for trap in self.model_params['trapezoids']]
                        params['trapezoids'][trap_idx][param_type] = optimization_values[i]

                elif param_name.startswith('sld_'):
                    # SLD parameter at design-layer index
                    sld_idx = int(param_name.split('_')[1])
                    if temp_sld_values is None:
                        # Will be sized properly once design_slds_list is created; just record index/value pair.
                        temp_sld_values = {}
                    temp_sld_values[sld_idx] = optimization_values[i]
                    
                elif param_name.startswith('Bk_'):
                    # Background parameter for specific column
                    bk_idx = int(param_name.split('_')[1])
                    if isinstance(temp_Bk, np.ndarray):
                        temp_Bk[bk_idx] = optimization_values[i]
                    else:
                        # Convert scalar to array if needed
                        n_columns = Intensity.shape[1]
                        temp_Bk = np.full(n_columns, temp_Bk)
                        temp_Bk[bk_idx] = optimization_values[i]
                elif param_name == 'Bk':
                    # Scalar background parameter
                    temp_Bk = optimization_values[i]
                else:
                    # Global parameter (DW, I0)
                    params[param_name] = optimization_values[i]
            
            # Build a local design-level trapezoid/SRD list for this evaluation
            if 'design_trapezoids' in params:
                design_traps = [t.copy() for t in params['design_trapezoids']]
                design_layers = params.get('design_layers', len(design_traps) - 1)
                design_slds_list = list(params.get('design_slds', []))
            else:
                # Fall back to treating current trapezoids as design-level
                design_traps = [t.copy() for t in params['trapezoids']]
                design_layers = len(design_traps) - 1
                if 'slds' in params:
                    slds_val = params['slds']
                    if isinstance(slds_val, (list, tuple, np.ndarray)):
                        design_slds_list = list(slds_val)
                    else:
                        design_slds_list = [float(slds_val)] * len(design_traps)
                elif hasattr(self, 'sld_values'):
                    design_slds_list = list(self.sld_values)
                else:
                    design_slds_list = [1.0] * len(design_traps)

            # Apply any optimized SLDs at design index level
            if isinstance(temp_sld_values, dict):
                for idx, val in temp_sld_values.items():
                    if 0 <= idx < len(design_slds_list):
                        design_slds_list[idx] = float(val)

            # Use existing typed-layer expansion logic on a temporary model_params-like dict
            tmp_model_params = {
                'trapezoids': design_traps,
                'layers': design_layers,
                'slds': design_slds_list,
            }

            # Apply constraints on the design parameters before expansion/coordinates
            constraints = self.model_params.get('constraints', None) if hasattr(self, 'model_params') else None
            if constraints:
                self._apply_constraints(tmp_model_params, constraints)

            expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(tmp_model_params)

            # Create temporary PAR array and effective SLD array for compatibility
            temp_PAR = np.zeros((expanded_layers + 1, 3))
            sld_eff = np.ones(expanded_layers + 1, dtype=float)

            for i, trap in enumerate(expanded_traps):
                if i <= expanded_layers:
                    temp_PAR[i, 0] = trap['width']
                    temp_PAR[i, 1] = trap['height']
                    temp_PAR[i, 2] = trap.get('twidth', trap['width'])
                    sld_eff[i] = float(expanded_slds[i]) if i < len(expanded_slds) else 1.0
            
            # Extract global parameters
            temp_DW = params['DW']
            temp_I0 = params['I0']
            
            # Use SymCoordAssign with expanded SLD values
            Coord = self.SymCoordAssign(temp_PAR, expanded_layers, sld_values=sld_eff)
            if Coord is None:
                raise RuntimeError("Failed to assign coordinates with SLD values")
            
            # Calculate form factor (center stack)
            form_center = self.FreeFormTrapezoid(Coord, expanded_layers, Qx, Qz)
            if form_center is None:
                raise RuntimeError("Failed to calculate form factor")

            # Curved sidewalls (typed design layer): coherent complex sum
            form_total = form_center
            try:
                coord_by_design, traps_by_design, flat_rows = self._curved_sides_compute_for_expanded_stack(
                    design_traps=tmp_model_params.get('trapezoids', []),
                    expanded_design_index=expanded_design_index,
                    PAR=temp_PAR,
                    layers=expanded_layers,
                    expanded_slds=expanded_slds,
                )
                # Store for optional downstream inspection (even during optimization)
                self.Coord_curved = coord_by_design
                self.curved_sidewall_trapezoids = traps_by_design
                self.curved_sidewall_trapezoid_rows = flat_rows

                for _, Cleft in coord_by_design.get("left", {}).items():
                    if Cleft is None:
                        continue
                    form_total = form_total + self.FreeFormTrapezoid(Cleft, len(Cleft) - 1, Qx, Qz)
                for _, Cright in coord_by_design.get("right", {}).items():
                    if Cright is None:
                        continue
                    form_total = form_total + self.FreeFormTrapezoid(Cright, len(Cright) - 1, Qx, Qz)
            except Exception as e:
                print(f"WARNING: curved sidewall contribution failed (GF): {str(e)}")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qx, 2) + np.power(Qz, 2)) * np.power(temp_DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = form_total * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity with array background support
            intensity_base = np.power(Formfactor, 2) * temp_I0
            
            if isinstance(temp_Bk, np.ndarray):
                # Array background - broadcast across columns
                if len(temp_Bk) != intensity_base.shape[1]:
                    raise ValueError(f"Background array length ({len(temp_Bk)}) must match number of columns ({intensity_base.shape[1]})")
                
                # Add background to each column
                SimInt = intensity_base + temp_Bk[np.newaxis, :]
            else:
                # Scalar background
                SimInt = intensity_base + temp_Bk
            
            # Calculate goodness of fit
            Chi2 = self.GF_calc(SimInt, Intensity)
            
            return Chi2
            
        except Exception as e:
            print(f"Error in SimTrap_GF: {str(e)}")
            return float('inf')  # Return infinity as worst-case fit
        
        
    def CDSAXS_DiffEvolution(self, params_to_optimize=None, plot_results=True, 
                    plot_structure=True, plot_grid=True, plot_combined=True,
                    verbose=False, structure_curved_show_slices=False, **kwargs):
        """
        Performs differential evolution optimization for CDSAXS trapezoid model fitting
        with array background support and shows before/after comparison plots.
        
        Fixed to respect verbose parameter properly.

        structure_curved_show_slices : bool, optional
            If True, structure comparison includes individual curved-sidewall slice outlines;
            default False shows envelope only (cleaner overlay with initial vs optimized).
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Intensity'):
                raise AttributeError("Missing required attribute: Intensity")
                
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                raise AttributeError("Missing required scattering vector attributes: Qx and/or Qz")
            
            # Initialize optimization parameters if needed
            if not hasattr(self, 'model_params') or 'optimization' not in self.model_params:
                self.initialize_optimization_params()
            
            # Determine parameters to optimize
            if params_to_optimize is None:
                params_to_optimize = self.model_params['optimization']
            
            # FIXED: Ensure all parameters have default values
            params_to_optimize = self._ensure_defaults_in_params(params_to_optimize)
            
            # Create parameter names list and bounds list
            param_names = []
            bounds = []
            initial_values = []
            
            for param_name, param_config in params_to_optimize.items():
                param_names.append(param_name)
                bounds.append((param_config['min'], param_config['max']))
                initial_values.append(param_config['default'])  # This should now always exist
            
            # Store for use in SimTrap_GF
            self.param_names = param_names
            
            # Set default optimization parameters if not provided in kwargs
            default_params = {
                'polish': True,
                'x0': np.array(initial_values)
            }
            
            # Allow structure_curved_show_slices via **kwargs for backward compatibility
            structure_curved_show_slices = kwargs.pop(
                'structure_curved_show_slices', structure_curved_show_slices
            )
            # Update default parameters with any provided kwargs
            optimization_params = {**default_params, **kwargs}
            
            # Store current parameters and simulation results for before/after comparison
            initial_model_params = copy.deepcopy(self.model_params)
            
            # Calculate initial simulated intensity if not already done
            if not hasattr(self, 'SimInt') or self.SimInt is None:
                self.SimInt = self.SimTrap_SM()
                
            # Store initial simulation results
            initial_simInt = copy.deepcopy(self.SimInt)
            
            # Baseline χ² for this run: same objective and seed vector as differential_evolution
            # (`SimTrap_GF`), not `GF_calc(SimInt)` — those differ if `x0` / optimization defaults
            # disagree with the geometry used to build the current `SimInt`.
            x0_seed = np.asarray(optimization_params.get('x0', np.array(initial_values, dtype=float)), dtype=float).ravel()
            if x0_seed.shape[0] == len(param_names):
                _gf0 = self.SimTrap_GF(x0_seed, param_names, self.Intensity, self.Qx, self.Qz)
                self.GF_Initial = float(_gf0) if np.isfinite(_gf0) else float(self.GF_calc(self.SimInt))
            else:
                self.GF_Initial = float(self.GF_calc(self.SimInt))
            self.BIC_Initial = self.BIC_calc(self.GF_Initial)
            
            # Run differential evolution optimization
            if verbose:  # Only print if verbose=True
                print(f"Starting optimization with {len(param_names)} parameters...")
            
            result = differential_evolution(
                self.SimTrap_GF,
                bounds, 
                args=(param_names, self.Intensity, self.Qx, self.Qz),
                **optimization_params
            )
            
            # Store the optimization result
            self.optimization_result = result
            
            # Update model parameters with optimized values (design-level where available)
            optimized_params = copy.deepcopy(self.model_params)
            
            # Initialize background array for updates
            if isinstance(self.Bk, np.ndarray):
                optimized_bk = self.Bk.copy()
            else:
                optimized_bk = self.Bk
            
            # First pass: Write all optimized values to design_trapezoids (if exists) or trapezoids
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid (design-layer) parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = '_'.join(parts[2:])

                    if 'design_trapezoids' in optimized_params:
                        # Store best-fit values on the design geometry
                        if trap_idx < len(optimized_params['design_trapezoids']):
                            optimized_params['design_trapezoids'][trap_idx][param_type] = result.x[i]
                    else:
                        # Pure trapezoid case: write directly to trapezoids
                        if trap_idx < len(optimized_params['trapezoids']):
                            optimized_params['trapezoids'][trap_idx][param_type] = result.x[i]
                elif param_name.startswith('Bk_'):
                    # Background parameter for specific column
                    bk_idx = int(param_name.split('_')[1])
                    if isinstance(optimized_bk, np.ndarray):
                        optimized_bk[bk_idx] = result.x[i]
                    else:
                        # Convert scalar to array if needed
                        n_columns = self.Intensity.shape[1]
                        optimized_bk = np.full(n_columns, optimized_bk)
                        optimized_bk[bk_idx] = result.x[i]
                elif param_name == 'Bk':
                    # Scalar background parameter
                    optimized_bk = result.x[i]
                else:
                    # Global parameter (DW, I0)
                    optimized_params[param_name] = result.x[i]
            
            # If design_trapezoids was updated, immediately re-expand trapezoids from it
            if 'design_trapezoids' in optimized_params:
                # Build a temporary model_params dict for expansion
                tmp_model_params = {
                    'trapezoids': [t.copy() for t in optimized_params['design_trapezoids']],
                    'layers': optimized_params.get('design_layers', len(optimized_params['design_trapezoids']) - 1),
                    'slds': optimized_params.get('design_slds', optimized_params.get('slds', None)),
                }
                
                # Apply constraints if they exist (before expansion)
                constraints = optimized_params.get('constraints', None)
                if constraints:
                    self._apply_constraints(tmp_model_params, constraints)
                    # IMPORTANT: Copy constrained values back to optimized_params['design_trapezoids']
                    # so that design_trapezoids reflects the constrained (final) values
                    optimized_params['design_trapezoids'] = [t.copy() for t in tmp_model_params['trapezoids']]
                
                # Expand typed layers (ellipse -> many segments)
                expanded_traps, expanded_slds, expanded_layers, _, _, _, _ = self._expand_typed_layers(tmp_model_params)
                
                # Update optimized_params with expanded structure
                optimized_params['trapezoids'] = expanded_traps
                optimized_params['layers'] = int(expanded_layers)
                optimized_params['slds'] = expanded_slds
            
            # Update background in optimized parameters
            optimized_params['Bk'] = optimized_bk.tolist() if isinstance(optimized_bk, np.ndarray) else optimized_bk
            
            # Update class attributes with optimized values
            self.model_params = optimized_params
            self.update_traditional_from_model_params()
            
            # Simulate with optimized parameters
            self.SimInt = self.SimTrap_SM()
            
            # Calculate goodness of fit and BIC
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            # Print optimization results only if verbose
            if verbose:
                print(f"Optimization complete after {result.nfev} function evaluations")
                print(f"Initial goodness of fit: {self.GF_Initial:.4f}")
                print(f"Final goodness of fit: {self.GF:.4f}")
                print(f"Improvement: {self.GF_Initial - self.GF:.4f} ({(1 - self.GF/self.GF_Initial)*100:.2f}%)")
            
            # Generate before/after comparison plots if requested
            if plot_results:
                self._plot_optimization_results(
                    initial_model_params,
                    initial_simInt,
                    plot_structure,
                    plot_grid,
                    plot_combined,
                    structure_curved_show_slices=structure_curved_show_slices,
                )
            
            # Print parameter changes only if verbose
            if verbose:
                self.print_parameter_changes(initial_model_params)
            
            return self.model_params
                
        except Exception as e:
            if verbose:  # Only print errors if verbose
                print(f"Error in CDSAXS_DiffEvolution: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

    
    def _plot_optimization_results(self, initial_model_params, initial_simInt, 
                                  plot_structure=True, plot_grid=True, plot_combined=True,
                                  structure_curved_show_slices=False):
        """
        Generate before/after comparison plots for optimization results.
        
        Parameters:
        -----------
        initial_model_params : dict
            Model parameters before optimization
        initial_simInt : numpy.ndarray
            Simulated intensity before optimization
        plot_structure : bool
            Whether to plot trapezoid structure comparison
        plot_grid : bool
            Whether to plot the grid of individual Qz cuts
        plot_combined : bool
            Whether to plot the combined view with all cuts
        structure_curved_show_slices : bool
            Passed to curved overlay in structure plot (default False: envelope only).
        """
        import matplotlib.pyplot as plt
        
        # Reconstruct optimized model_params directly from optimization_result.x
        # This bypasses any write-back issues and ensures we use the actual optimized values
        optimized_model_params_for_plot = None
        if hasattr(self, 'optimization_result') and self.optimization_result is not None:
            if hasattr(self, 'param_names') and self.param_names is not None:
                # Reconstruct design_trapezoids from optimization_result.x
                optimized_model_params_for_plot = copy.deepcopy(initial_model_params)
                
                # Get the optimized values from result.x
                result_x = self.optimization_result.x
                param_names = self.param_names
                
                # Update design_trapezoids (or trapezoids) with optimized values
                if 'design_trapezoids' in optimized_model_params_for_plot:
                    design_traps = optimized_model_params_for_plot['design_trapezoids']
                elif 'trapezoids' in optimized_model_params_for_plot:
                    design_traps = optimized_model_params_for_plot['trapezoids']
                else:
                    design_traps = None
                
                if design_traps is not None:
                    for i, param_name in enumerate(param_names):
                        if param_name.startswith('trap_'):
                            parts = param_name.split('_')
                            if len(parts) >= 3:
                                try:
                                    trap_idx = int(parts[1])
                                    param_type = '_'.join(parts[2:])
                                    if trap_idx < len(design_traps):
                                        design_traps[trap_idx][param_type] = result_x[i]
                                except (ValueError, IndexError):
                                    pass
                        elif param_name in ('DW', 'I0', 'Bk'):
                            optimized_model_params_for_plot[param_name] = result_x[i]
                
                # Apply constraints if they exist
                constraints = optimized_model_params_for_plot.get('constraints', None)
                if constraints and 'design_trapezoids' in optimized_model_params_for_plot:
                    tmp_design = {
                        'design_trapezoids': [t.copy() for t in optimized_model_params_for_plot['design_trapezoids']],
                        'design_slds': copy.deepcopy(optimized_model_params_for_plot.get('design_slds', [])),
                        'DW': optimized_model_params_for_plot.get('DW', getattr(self, 'DW', None)),
                        'I0': optimized_model_params_for_plot.get('I0', getattr(self, 'I0', None)),
                        'Bk': optimized_model_params_for_plot.get('Bk', getattr(self, 'Bk', None)),
                        'layers': optimized_model_params_for_plot.get('design_layers', len(optimized_model_params_for_plot['design_trapezoids']) - 1),
                    }
                    self._apply_constraints(tmp_design, constraints)
                    optimized_model_params_for_plot['design_trapezoids'] = tmp_design['design_trapezoids']
        
        # Use reconstructed optimized params if available, otherwise fall back to self.model_params
        optimized_params_to_plot = optimized_model_params_for_plot if optimized_model_params_for_plot is not None else self.model_params
        
        # Plot trapezoid structure comparison on the same plot (center stack + curved overlay)
        if plot_structure:
            _, ax = plt.subplots(figsize=(10, 6))

            def _draw_structure_series(model_params_for_series, linestyle, color, alpha, legend_label, env_linestyle):
                plt.sca(ax)
                plot_prepared, overlay_b = self._prepare_structure_plot_params(model_params_for_series)
                self._plot_trapezoid_structure(
                    plot_prepared,
                    linestyle=linestyle,
                    color=color,
                    alpha=alpha,
                    label=legend_label,
                )
                if overlay_b is not None and self._design_has_curved_sides(overlay_b['design_traps']):
                    env_alpha = 0.12 if color == 'blue' else 0.17
                    self._plot_curved_sidewalls_overlay(
                        ax,
                        overlay_b,
                        show_slices=structure_curved_show_slices,
                        poly_alpha=env_alpha,
                        envelope_edgecolor=color,
                        envelope_linestyle=env_linestyle,
                        envelope_linewidth=1.35,
                        left_slice_color=color,
                        right_slice_color=color,
                    )

            _draw_structure_series(initial_model_params, '--', 'blue', 0.7, 'Initial', '--')
            _draw_structure_series(optimized_params_to_plot, '-', 'red', 1.0, 'Optimized', '-')

            plt.title('Trapezoid Structure Comparison')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Recalculate optimized intensity using reconstructed optimized parameters
        # This ensures we're plotting the correct optimized intensity even if write-back failed
        optimized_simInt = None
        if hasattr(self, 'optimization_result') and self.optimization_result is not None:
            if hasattr(self, 'param_names') and self.param_names is not None:
                # Temporarily store current model_params
                original_model_params = self.model_params.copy()
                
                try:
                    # Use the reconstructed optimized_params we created for plotting
                    if optimized_params_to_plot is not None:
                        # Temporarily set model_params to optimized values for simulation
                        self.model_params = copy.deepcopy(optimized_params_to_plot)
                        # Recalculate intensity with optimized parameters
                        optimized_simInt = self.SimTrap_SM()
                except Exception as e:
                    print(f"[WARNING] Could not recalculate optimized intensity: {e}")
                    print("  Falling back to self.SimInt")
                    optimized_simInt = self.SimInt
                finally:
                    # Restore original model_params
                    self.model_params = original_model_params
        
        # Use recalculated intensity if available, otherwise fall back to self.SimInt
        optimized_simInt_to_plot = optimized_simInt if optimized_simInt is not None else self.SimInt
        
        # Plot QzCut comparisons - grid of individual cuts
        if plot_grid:
            self._plot_qzcut_grid(initial_simInt, optimized_simInt_to_plot)
        
        # Plot combined view with all cuts
        if plot_combined:
            self._plot_qzcut_combined(initial_simInt, optimized_simInt_to_plot)
    
    def _plot_trapezoid_structure(
        self,
        model_params,
        linestyle='-',
        color='black',
        alpha=1.0,
        label=None,
        linewidth=2,
        equal_aspect=True,
        shade_by_sld=False,
        grey_range=(0.85, 0.05),
        shading_alpha=0.6,
        **kwargs
    ):
        """
        Plot the trapezoid structure from the given model parameters.
        
        This function handles design_trapezoids, constraints, and typed layer expansion
        to ensure the plotted structure reflects the actual model state.
        
        Parameters:
        -----------
        model_params : dict
            Dictionary containing model parameters (may include design_trapezoids)
        linestyle : str, optional
            Line style for the plot
        color : str, optional
            Color for the plot
        alpha : float, optional
            Transparency for the plot
        label : str, optional
            Label for the legend
        linewidth : float, optional
            Width of the lines
        equal_aspect : bool, optional
            Whether to use equal aspect ratio
        shade_by_sld : bool, optional
            If True, fills each trapezoid with a greyscale shade based on per-layer SLD values
            from `model_params['slds']` (or `self.sld_values` fallback). Default: False.
        grey_range : tuple(float, float), optional
            (light, dark) greyscale intensities in [0, 1], where 0=black and 1=white.
            Default (0.85, 0.05) is light grey → near-black.
        shading_alpha : float, optional
            Alpha for the trapezoid fill when `shade_by_sld=True`. Default: 0.6.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions
        """
        # Prepare model_params for plotting: handle design_trapezoids, constraints, expansion
        plot_params = copy.deepcopy(model_params)
        constraints = plot_params.get('constraints', None)
        
        # Always prefer design_trapezoids if it exists (regardless of constraints)
        # Note: design_slds is optional - if not present, will use slds or default
        if 'design_trapezoids' in plot_params:
            # Get design_slds if available, otherwise fall back to slds
            design_slds = plot_params.get('design_slds', None)
            if design_slds is None:
                design_slds = plot_params.get('slds', None)
            
            tmp_design = {
                'design_trapezoids': [t.copy() for t in plot_params['design_trapezoids']],
                'design_slds': copy.deepcopy(design_slds) if design_slds is not None else None,
                'DW': plot_params.get('DW', getattr(self, 'DW', None)),
                'I0': plot_params.get('I0', getattr(self, 'I0', None)),
                'Bk': plot_params.get('Bk', getattr(self, 'Bk', None)),
                'layers': plot_params.get('design_layers', len(plot_params['design_trapezoids']) - 1),
            }
            # Apply constraints if they exist
            if constraints:
                self._apply_constraints(tmp_design, constraints)
            tmp_model_params = {
                'trapezoids': tmp_design['design_trapezoids'],
                'layers': tmp_design['layers'],
                'slds': tmp_design.get('design_slds', None),
            }
            expanded_traps, expanded_slds, expanded_layers, _, _, _, _ = self._expand_typed_layers(tmp_model_params)
            plot_params['trapezoids'] = expanded_traps
            plot_params['layers'] = expanded_layers
            plot_params['slds'] = expanded_slds
        elif constraints:
            # No design_trapezoids; apply constraints directly and expand if typed layers exist
            self._apply_constraints(plot_params, constraints)
            expanded_traps, expanded_slds, expanded_layers, _, _, _, _ = self._expand_typed_layers(plot_params)
            if expanded_traps:  # Only update if expansion occurred
                plot_params['trapezoids'] = expanded_traps
                plot_params['layers'] = expanded_layers
                plot_params['slds'] = expanded_slds
        else:
            # No design_trapezoids and no constraints; just expand if typed layers exist
            expanded_traps, expanded_slds, expanded_layers, _, _, _, _ = self._expand_typed_layers(plot_params)
            if expanded_traps:  # Only update if expansion occurred
                plot_params['trapezoids'] = expanded_traps
                plot_params['layers'] = expanded_layers
                plot_params['slds'] = expanded_slds
        
        ax = plt.gca()
        trapezoids = plot_params['trapezoids']
        layers = plot_params['layers']

        # Validate/clip shading params
        try:
            grey_light, grey_dark = float(grey_range[0]), float(grey_range[1])
        except Exception:
            grey_light, grey_dark = 0.85, 0.05
        grey_light = float(np.clip(grey_light, 0.0, 1.0))
        grey_dark = float(np.clip(grey_dark, 0.0, 1.0))
        shading_alpha = float(np.clip(float(shading_alpha), 0.0, 1.0))

        # Optional greyscale fill (draw first so outline stays on top)
        if shade_by_sld and layers > 0 and len(trapezoids) >= layers + 1:
            if 'slds' in model_params:
                slds = np.array(model_params['slds'], dtype=float)
            elif hasattr(self, 'sld_values'):
                slds = np.array(self.sld_values, dtype=float)
            else:
                slds = np.ones(layers, dtype=float)

            # Keep all layers+1 SLD values if provided (one per trapezoid entry including top)
            if len(slds) == layers + 1:
                # Keep all values - one SLD per trapezoid entry
                pass
            elif len(slds) == 1 and layers > 0:
                slds = np.full(layers + 1, float(slds[0]), dtype=float)
            elif len(slds) != layers + 1:
                # Resize to match trapezoid count (layers + 1)
                slds = np.resize(slds, layers + 1).astype(float)

            finite_slds = slds[np.isfinite(slds)]
            if finite_slds.size == 0:
                sld_min, sld_max = 0.0, 1.0
            else:
                sld_min, sld_max = float(np.min(finite_slds)), float(np.max(finite_slds))
            denom = (sld_max - sld_min) if not np.isclose(sld_max, sld_min) else None

            base_w = float(trapezoids[0]['width'])
            height0 = 0.0
            # Shade all trapezoids including the top one (layers + 1 total)
            for i in range(int(layers) + 1):
                h = float(trapezoids[i]['height'])
                if h <= 0:
                    continue

                w0 = float(trapezoids[i]['width'])
                if trapezoids[i]['twidth'] is None:
                    if i < layers:
                        w1 = float(trapezoids[i + 1]['width'])
                    else:
                        w1 = w0  # Top trapezoid: use same width if no twidth
                else:
                    w1 = float(trapezoids[i]['twidth'])

                xL0 = (base_w - w0) / 2.0
                xR0 = xL0 + w0
                xL1 = (base_w - w1) / 2.0
                xR1 = xL1 + w1

                y0 = height0
                y1 = height0 + h

                sld_val = float(slds[i]) if i < len(slds) else 1.0
                if denom is None:
                    t = 0.5
                else:
                    t = (sld_val - sld_min) / denom
                t = float(np.clip(t, 0.0, 1.0))

                grey = grey_light + t * (grey_dark - grey_light)  # larger SLD -> darker (by default)
                grey = float(np.clip(grey, 0.0, 1.0))

                ax.add_patch(
                    Polygon(
                        [(xL0, y0), (xR0, y0), (xR1, y1), (xL1, y1)],
                        closed=True,
                        facecolor=(grey, grey, grey),
                        edgecolor='none',
                        alpha=shading_alpha,
                        zorder=1,
                    )
                )

                height0 = y1
        
        # Plot base
        ax.plot([0, trapezoids[0]['width']], [0, 0], 
                linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        if trapezoids[0]['twidth'] is None:
            ax.plot([0.5*(trapezoids[0]['width']-trapezoids[1]['width']), 0.5*(trapezoids[0]['width']+trapezoids[1]['width'])], [trapezoids[0]['height'], trapezoids[0]['height']], 
                     linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        else:
            ax.plot([0.5*(trapezoids[0]['width']-trapezoids[0]['twidth']), 0.5*(trapezoids[0]['width']+trapezoids[0]['twidth'])], [trapezoids[0]['height'], trapezoids[0]['height']], 
                     linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        
        height = 0
        for i in range(layers + 1):
            if i > 0:
                height += trapezoids[i-1]['height']
            
            width = trapezoids[i]['width']
            x_left = (trapezoids[0]['width'] - width) / 2
            x_right = x_left + width
            if trapezoids[i]['twidth'] is None:
                twidth = trapezoids[i+1]['width']
            else:
                twidth = trapezoids[i]['twidth']
            x_tleft = (trapezoids[0]['width'] - twidth) / 2
            x_tright = x_tleft + twidth
            
            ax.plot([x_left, x_right], [height, height], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
            ax.plot([x_tleft, x_tright], [height+trapezoids[i]['height'], height+trapezoids[i]['height']], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
            
            if i < layers+1:
                if trapezoids[i]['twidth'] is None:
                    next_width = trapezoids[i+1]['width']
                else:
                    next_width = trapezoids[i]['twidth']
                x_next_left = (trapezoids[0]['width'] - next_width) / 2
                x_next_right = x_next_left + next_width
                
                ax.plot([x_left, x_next_left], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
                ax.plot([x_right, x_next_right], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        
        # Add a line to the legend
        if label:
            ax.plot([], [], linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        # Set aspect ratio - only use equal if not overridden by user limits
        if equal_aspect:
            ax.axis('equal')
        
        ax.set_xlabel('Width (Å)')
        ax.set_ylabel('Height (Å)')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    def _plot_qzcut_grid(self, initial_simInt, optimized_simInt=None):
        """
        Plot a grid of QzCut comparisons with both initial and optimized results.
        
        Parameters:
        -----------
        initial_simInt : numpy.ndarray
            Simulated intensity before optimization
        optimized_simInt : numpy.ndarray, optional
            Simulated intensity after optimization. If None, uses self.SimInt
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Determine number of cuts to display
        n_cuts = self.Intensity.shape[1]
        
        # Create a grid of plots
        if n_cuts <= 3:
            n_rows, n_cols = 1, n_cuts
        else:
            n_rows = int(np.ceil(np.sqrt(n_cuts)))
            n_cols = int(np.ceil(n_cuts / n_rows))
        
        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                                squeeze=False, sharex=True)
        axes = axes.flatten()
        
        # Plot each cut
        for i in range(n_cuts):
            ax = axes[i]
            
            # Get data for this cut
            qz_values = self.Qz[:, i]
            qx_value = self.Qx[0, i]
            
            # Plot measured data
            ax.semilogy(qz_values, self.Intensity[:, i], 'o', 
                    color='grey', alpha=0.7, markersize=4, label='Measured')
            
            # Plot initial simulation
            ax.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.8, 
                     linewidth=1.5, label='Initial')
            
            # Plot optimized simulation
            opt_intensity = optimized_simInt if optimized_simInt is not None else self.SimInt
            ax.semilogy(qz_values, opt_intensity[:, i], 'r-', alpha=1.0, 
                     linewidth=1.5, label='Optimized')
            
            # Set labels and title
            ax.set_title(f'Cut at Qx = {qx_value:.4f}')
            ax.set_xlabel('Qz (Å$^{-1}$)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # Only show legend on the first plot to save space
            if i == 0:
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_cuts, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle('Intensity Comparison: Initial vs. Optimized', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.show()
    
    def _plot_qzcut_combined(self, initial_simInt, optimized_simInt=None):
        """
        Plot all QzCuts on one axis with both initial and optimized results.
        
        Parameters:
        -----------
        initial_simInt : numpy.ndarray
            Simulated intensity before optimization
        optimized_simInt : numpy.ndarray, optional
            Simulated intensity after optimization. If None, uses self.SimInt
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create legend items
        plt.plot([], [], 'ko', markersize=4, label='Measured')
        plt.plot([], [], 'b--', linewidth=1.5, label='Initial')
        plt.plot([], [], 'r-', linewidth=1.5, label='Optimized')
        
        # Determine number of cuts
        n_cuts = self.Intensity.shape[1]
        
        # Plot each cut
        for i in range(n_cuts):
            qz_values = self.Qz[:, i]
            
            # Plot measured data
            plt.semilogy(qz_values, self.Intensity[:, i], 'o', 
                    color='grey', alpha=0.5, markersize=4)
            
            # Plot initial simulation
            plt.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.5, linewidth=1.5)
            
            # Plot optimized simulation
            opt_intensity = optimized_simInt if optimized_simInt is not None else self.SimInt
            plt.semilogy(qz_values, opt_intensity[:, i], 'r-', alpha=0.6, linewidth=1.5)
        
        plt.title('Intensity Comparison - All Cuts')
        plt.xlabel('Qz (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def _design_has_curved_sides(design_traps):
        if not design_traps:
            return False
        for t in design_traps:
            if isinstance(t, dict) and str(t.get('Layer_Type', '')).strip().lower() == 'curved_sides':
                return True
        return False

    def _plot_curved_sidewalls_overlay(
        self,
        ax,
        overlay_bundle,
        show_slices=True,
        poly_alpha=0.15,
        left_slice_color='#1f77b4',
        right_slice_color='#ff7f0e',
        envelope_edgecolor='#8b0000',
        envelope_linewidth=1.2,
        envelope_linestyle='-',
        slice_linewidth=0.7,
        slice_line_alpha=0.85,
    ):
        """
        Draw full curved-sidewall polygon envelope(s) and optional side-slice outlines on an existing structure axes.
        """
        design_traps = overlay_bundle['design_traps']
        expanded_traps = overlay_bundle['expanded_traps']
        expanded_slds = overlay_bundle['expanded_slds']
        expanded_design_index = overlay_bundle['expanded_design_index']
        expanded_layers = int(overlay_bundle['expanded_layers'])

        temp_PAR = np.zeros((expanded_layers + 1, 3), dtype=float)
        for ii, trap in enumerate(expanded_traps):
            if ii <= expanded_layers:
                temp_PAR[ii, 0] = trap.get('width')
                temp_PAR[ii, 1] = trap.get('height')
                tw = trap.get('twidth', None)
                temp_PAR[ii, 2] = np.nan if tw is None else tw

        _, traps_by_design, _ = self._curved_sides_compute_for_expanded_stack(
            design_traps=design_traps,
            expanded_design_index=expanded_design_index,
            PAR=temp_PAR,
            layers=expanded_layers,
            expanded_slds=expanded_slds,
        )

        center_x = 0.5 * float(temp_PAR[0, 0])
        heights = temp_PAR[: expanded_layers + 1, 1].astype(float)
        cum_heights = np.concatenate(([0.0], np.cumsum(heights)))
        design_to_expanded = {}
        for exp_idx, d_idx in enumerate(expanded_design_index):
            design_to_expanded.setdefault(int(d_idx), []).append(int(exp_idx))

        for design_idx, d in enumerate(design_traps):
            if not isinstance(d, dict) or str(d.get('Layer_Type', '')).strip().lower() != 'curved_sides':
                continue
            side_length = float(d.get('side_length', 0.0))
            tip_deflection = float(d.get('tip_deflection', 0.0))
            n_side_pts = int(d.get('n_side_pts', 400))
            for exp_idx in design_to_expanded.get(int(design_idx), []):
                if exp_idx > expanded_layers:
                    continue
                yb = float(cum_heights[exp_idx])
                h = float(temp_PAR[exp_idx, 1])
                if h <= 0:
                    continue
                w0 = float(temp_PAR[exp_idx, 0])
                if not np.isnan(temp_PAR[exp_idx, 2]):
                    w1 = float(temp_PAR[exp_idx, 2])
                else:
                    w1 = float(temp_PAR[exp_idx + 1, 0]) if exp_idx < expanded_layers else w0
                layer = self._curved_sides_build_layer_center_trapezoid_incompressible_sides(
                    center_x=center_x,
                    center_y_bottom=yb,
                    center_height=h,
                    center_bottom_width=w0,
                    center_top_width=w1,
                    side_length=side_length,
                    side_tip_deflection=tip_deflection,
                    n_side_pts=n_side_pts,
                )
                ax.fill(
                    layer['poly_x'],
                    layer['poly_y'],
                    facecolor=envelope_edgecolor,
                    edgecolor=envelope_edgecolor,
                    linewidth=float(envelope_linewidth),
                    linestyle=envelope_linestyle,
                    alpha=poly_alpha,
                    zorder=2,
                )

        if show_slices:
            for side, col in (('left', left_slice_color), ('right', right_slice_color)):
                for _, tlist in traps_by_design.get(side, {}).items():
                    if not tlist:
                        continue
                    for t in tlist:
                        xs = [
                            t['x_left_bottom'],
                            t['x_right_bottom'],
                            t['x_right_top'],
                            t['x_left_top'],
                            t['x_left_bottom'],
                        ]
                        ys = [
                            t['y_bottom'],
                            t['y_bottom'],
                            t['y_top'],
                            t['y_top'],
                            t['y_bottom'],
                        ]
                        ax.plot(
                            xs, ys, color=col, linewidth=float(slice_linewidth),
                            alpha=float(slice_line_alpha), zorder=3,
                        )

    def _prepare_structure_plot_params(self, model_params):
        """
        Expand design/typed layers for structure plotting and build overlay_bundle for curved_sides.

        Mirrors the layout logic in plot_structure so optimization comparison plots stay consistent.

        Returns
        -------
        plot_params : dict
            Copy of model_params with trapezoids/layers/slds replaced by expanded simulation stack.
        overlay_bundle : dict or None
            Keys: design_traps, expanded_traps, expanded_slds, expanded_design_index, expanded_layers;
            None if no typed-layer overlay is needed.
        """
        plot_params = copy.deepcopy(model_params)
        constraints = plot_params.get('constraints', None)
        overlay_bundle = None

        if 'design_trapezoids' in plot_params and 'design_slds' in plot_params:
            tmp_design = {
                'design_trapezoids': [t.copy() for t in plot_params['design_trapezoids']],
                'design_slds': copy.deepcopy(plot_params.get('design_slds', [])),
                'DW': plot_params.get('DW', self.DW),
                'I0': plot_params.get('I0', self.I0),
                'Bk': plot_params.get('Bk', self.Bk),
                'layers': plot_params.get('design_layers', len(plot_params['design_trapezoids']) - 1),
            }
            if constraints:
                self._apply_constraints(tmp_design, constraints)
            tmp_model_params = {
                'trapezoids': tmp_design['design_trapezoids'],
                'layers': tmp_design['layers'],
                'slds': tmp_design.get('design_slds', None),
            }
            expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(
                tmp_model_params
            )
            plot_params['trapezoids'] = expanded_traps
            plot_params['layers'] = expanded_layers
            plot_params['slds'] = expanded_slds
            overlay_bundle = {
                'design_traps': list(tmp_design['design_trapezoids']),
                'expanded_traps': expanded_traps,
                'expanded_slds': expanded_slds,
                'expanded_design_index': expanded_design_index,
                'expanded_layers': expanded_layers,
            }
        elif constraints:
            self._apply_constraints(plot_params, constraints)
            design_traps_for_overlay = copy.deepcopy(plot_params['trapezoids'])
            expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(
                plot_params
            )
            plot_params['trapezoids'] = expanded_traps
            plot_params['layers'] = expanded_layers
            plot_params['slds'] = expanded_slds
            if self._has_typed_layers(design_traps_for_overlay):
                overlay_bundle = {
                    'design_traps': design_traps_for_overlay,
                    'expanded_traps': expanded_traps,
                    'expanded_slds': expanded_slds,
                    'expanded_design_index': expanded_design_index,
                    'expanded_layers': expanded_layers,
                }
        else:
            design_traps_for_overlay = copy.deepcopy(plot_params['trapezoids'])
            expanded_traps, expanded_slds, expanded_layers, _, _, _, expanded_design_index = self._expand_typed_layers(
                plot_params
            )
            if expanded_traps:
                plot_params['trapezoids'] = expanded_traps
                plot_params['layers'] = expanded_layers
                plot_params['slds'] = expanded_slds
            if self._has_typed_layers(design_traps_for_overlay):
                overlay_bundle = {
                    'design_traps': design_traps_for_overlay,
                    'expanded_traps': expanded_traps,
                    'expanded_slds': expanded_slds,
                    'expanded_design_index': expanded_design_index,
                    'expanded_layers': expanded_layers,
                }

        return plot_params, overlay_bundle

    
    def plot_structure(
        self,
        figsize=(10, 6),
        xlim=None,
        ylim=None,
        title='Trapezoid Structure',
        show_dimensions=False,
        color='blue',
        linewidth=2,
        equal_aspect=True,
        shade_by_sld=False,
        grey_range=(0.85, 0.05),
        shading_alpha=0.6,
        sld_label_map=None,
        show_sld_legend=True,
        sld_legend_precision=3,
        show_curved_sidewalls=None,
        curved_sidewall_show_slices=True,
        curved_sidewall_poly_alpha=0.15,
        **kwargs
    ):
        """
        Plots the current trapezoid structure with customizable figure properties.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default: (10, 6)
        xlim : tuple, optional
            X-axis limits as (xmin, xmax). If None, uses automatic scaling
        ylim : tuple, optional
            Y-axis limits as (ymin, ymax). If None, uses automatic scaling
        title : str, optional
            Title for the plot. Default: 'Trapezoid Structure'
        show_dimensions : bool, optional
            Whether to show dimension annotations on the plot. Default: False
        color : str, optional
            Color for the structure lines. Default: 'blue'
        linewidth : float, optional
            Width of the structure lines. Default: 2
        equal_aspect : bool, optional
            Whether to use equal aspect ratio. Set to False if xlim/ylim are ignored. Default: True
        shade_by_sld : bool, optional
            If True, fills each trapezoid with a greyscale shade based on per-layer SLD values
            from `model_params['slds']` (or `self.sld_values` fallback). Default: False.
        grey_range : tuple(float, float), optional
            (light, dark) greyscale intensities in [0, 1], where 0=black and 1=white.
            Default (0.85, 0.05) is light grey → near-black.
        shading_alpha : float, optional
            Alpha for the trapezoid fill when `shade_by_sld=True`. Default: 0.6.
        sld_label_map : dict | None, optional
            Mapping from SLD value to label to display in the legend, e.g. {1.0: "SiO2", 1.28: "SiGe"}.
            If provided, legend labels will be "{label} (SLD={value})".
        show_sld_legend : bool, optional
            If True and `shade_by_sld=True`, adds a legend entry for each unique SLD. Default: True.
        sld_legend_precision : int, optional
            Decimal rounding used for grouping and mapping float SLD values in the legend. Default: 3.
        show_curved_sidewalls : bool or None, optional
            If True, overlays the full curved-sidewall envelope (`Layer_Type: curved_sides`) and optional
            side-slice outlines on top of the center trapezoid stack. If None (default), overlay runs only when
            the design contains a `curved_sides` entry.
        curved_sidewall_show_slices : bool, optional
            When True, draws individual side-slice trapezoids (left/right) used in the coherent form factor.
        curved_sidewall_poly_alpha : float, optional
            Fill alpha for the sidewall envelope polygon(s). Default: 0.15.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=figsize)
        # Validate/clip shading params (kept permissive to avoid breaking notebooks)
        try:
            grey_light, grey_dark = float(grey_range[0]), float(grey_range[1])
        except Exception:
            grey_light, grey_dark = 0.85, 0.05
        grey_light = float(np.clip(grey_light, 0.0, 1.0))
        grey_dark = float(np.clip(grey_dark, 0.0, 1.0))
        shading_alpha = float(np.clip(float(shading_alpha), 0.0, 1.0))

        plot_params, overlay_bundle = self._prepare_structure_plot_params(self.model_params)

        ax = self._plot_trapezoid_structure(
            plot_params,
            linestyle='-',
            color=color,
            alpha=1.0,
            linewidth=linewidth,
            label=None,
            equal_aspect=equal_aspect,
            shade_by_sld=shade_by_sld,
            grey_range=(grey_light, grey_dark),
            shading_alpha=shading_alpha,
            **kwargs
        )

        do_curved = show_curved_sidewalls
        if do_curved is None:
            do_curved = overlay_bundle is not None and self._design_has_curved_sides(overlay_bundle['design_traps'])
        if do_curved and overlay_bundle is not None and self._design_has_curved_sides(overlay_bundle['design_traps']):
            try:
                self._plot_curved_sidewalls_overlay(
                    ax,
                    overlay_bundle,
                    show_slices=curved_sidewall_show_slices,
                    poly_alpha=float(curved_sidewall_poly_alpha),
                )
            except Exception as e:
                print(f"WARNING: curved sidewall structure overlay failed: {str(e)}")
        
        # Set axis limits if provided (after plotting to override equal aspect if needed)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
            
        # If user specified limits and equal_aspect is True, adjust aspect
        if (xlim is not None or ylim is not None) and equal_aspect:
            # Use 'auto' to allow limits while maintaining reasonable aspect
            plt.axis('auto')
            # Re-apply limits to ensure they stick
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            
        # Add dimension annotations if requested
        if show_dimensions:
            self._add_dimension_annotations()

        # Add SLD/material legend if requested
        layers = plot_params['layers']
        if shade_by_sld and show_sld_legend and layers > 0:
            if 'slds' in plot_params:
                slds_all = np.array(plot_params['slds'], dtype=float)
            elif hasattr(self, 'sld_values'):
                slds_all = np.array(self.sld_values, dtype=float)
            else:
                slds_all = np.ones(layers, dtype=float)

            if len(slds_all) == layers + 1:
                slds_all = slds_all[:layers]
            elif len(slds_all) == 1 and layers > 1:
                slds_all = np.full(layers, float(slds_all[0]), dtype=float)
            elif len(slds_all) != layers:
                slds_all = np.resize(slds_all, layers).astype(float)

            finite_slds = slds_all[np.isfinite(slds_all)]
            if finite_slds.size == 0:
                sld_min, sld_max = 0.0, 1.0
            else:
                sld_min, sld_max = float(np.min(finite_slds)), float(np.max(finite_slds))
            denom = (sld_max - sld_min) if not np.isclose(sld_max, sld_min) else None

            # Normalize label map keys using the same rounding precision
            normalized_label_map = {}
            if isinstance(sld_label_map, dict):
                for k, v in sld_label_map.items():
                    try:
                        rk = round(float(k), int(sld_legend_precision))
                        normalized_label_map[rk] = v
                    except Exception:
                        continue

            rounded = np.array([round(float(v), int(sld_legend_precision)) for v in slds_all], dtype=float)
            unique_vals = sorted(set(rounded.tolist()))

            handles = []
            for rv in unique_vals:
                if denom is None:
                    t = 0.5
                else:
                    t = (rv - sld_min) / denom
                t = float(np.clip(t, 0.0, 1.0))
                grey = grey_light + t * (grey_dark - grey_light)
                grey = float(np.clip(grey, 0.0, 1.0))

                mapped = normalized_label_map.get(rv, None)
                if mapped is None:
                    lbl = f"SLD={rv:.{int(sld_legend_precision)}f}"
                else:
                    lbl = f"{mapped} (SLD={rv:.{int(sld_legend_precision)}f})"

                handles.append(Patch(facecolor=(grey, grey, grey), edgecolor='black', linewidth=0.5, label=lbl))

            if handles:
                ax.legend(handles=handles, title="SLD", frameon=True)
        
        plt.title(title)
        plt.tight_layout()
        
        return ax
    
    def _add_dimension_annotations(self):
        """
        Add dimension annotations to the trapezoid structure plot.
        """
        trapezoids = self.model_params['trapezoids']
        layers = self.model_params['layers']
        
        height = 0
        for i in range(layers + 1):
            if i > 0:
                height += trapezoids[i-1]['height']
            
            width = trapezoids[i]['width']
            x_left = (trapezoids[0]['width'] - width) / 2
            x_right = x_left + width
            
            # Annotate width
            plt.annotate(f'W{i}: {width:.1f}', 
                        xy=((x_left + x_right) / 2, height - 5),
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Annotate height (except for top layer)
            if i < layers and trapezoids[i]['height'] > 0:
                height_val = trapezoids[i]['height']
                plt.annotate(f'H{i}: {height_val:.1f}', 
                           xy=(x_right + 5, height + height_val/2),
                           ha='left', va='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


    def _plot_width_dw_sweep_results(self, results, figsize, metric='GF'):
        """
        Plot 1-layer width+DW sweep results as heatmap.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_width_dw_1layer
        figsize : tuple
            Figure size
        metric : str
            Metric to plot ('GF' or 'BIC')
        """
        # Choose which matrix to plot
        if metric.upper() == 'GF':
            data_matrix = results['gf_matrix']
            title = 'Goodness of Fit (GF)'
            cmap = 'viridis'
        else:
            data_matrix = results['bic_matrix']
            title = 'Bayesian Information Criterion (BIC)'
            cmap = 'viridis'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        plot_data = np.copy(data_matrix)
        plot_data[np.isinf(plot_data)] = np.nan
        
        # Use log scale if the range is large
        if np.nanmax(plot_data) / np.nanmin(plot_data) > 100:
            norm = LogNorm(vmin=np.nanmin(plot_data), vmax=np.nanmax(plot_data))
        else:
            norm = None
        
        im = ax.imshow(plot_data, cmap=cmap, aspect='auto', origin='lower', norm=norm)
        
        # Set axis labels and ticks
        ax.set_xlabel('Trapezoid Width (both layers)')
        ax.set_ylabel('Debye-Waller Factor (DW)')
        ax.set_title(f'{title} Heatmap: Width vs DW (1-Layer Model)')
        
        # Set tick labels
        n_ticks = 5
        x_tick_indices = np.linspace(0, len(results['width_values'])-1, n_ticks, dtype=int)
        y_tick_indices = np.linspace(0, len(results['dw_values'])-1, n_ticks, dtype=int)
        
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels([f'{results["width_values"][i]:.1f}' for i in x_tick_indices])
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels([f'{results["dw_values"][i]:.1f}' for i in y_tick_indices])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title)
        
        # Mark minimum
        min_idx = np.unravel_index(np.nanargmin(plot_data), plot_data.shape)
        ax.plot(min_idx[1], min_idx[0], 'r*', markersize=15, 
                label=f'Min {metric}: {plot_data[min_idx]:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        min_val = plot_data[min_idx]
        min_width = results['width_values'][min_idx[1]]
        min_dw = results['dw_values'][min_idx[0]]
        
        print("\n1-Layer Width+DW Sweep Summary:")
        print(f"Grid size: {len(results['width_values'])} x {len(results['dw_values'])}")
        print(f"Width range: {results['width_values'][0]:.1f} to {results['width_values'][-1]:.1f}")
        print(f"DW range: {results['dw_values'][0]:.1f} to {results['dw_values'][-1]:.1f}")
        print(f"Best {metric}: {min_val:.4f}")
        print(f"  at Width = {min_width:.1f}, DW = {min_dw:.1f}")
        print(f"  (Both trap_0_width and trap_1_width set to {min_width:.1f})")
    
    def get_optimal_width_dw_1layer(self, results):
        """
        Extract the optimal width and DW values from 1-layer sweep results.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_width_dw_1layer
            
        Returns:
        --------
        dict
            Dictionary with optimal values and corresponding GF/BIC
        """
        if results['sweep_type'] != 'width_dw_1layer':
            raise ValueError("Results must be from parameter_sweep_width_dw_1layer function")
        
        # Find minimum GF
        gf_matrix = results['gf_matrix']
        gf_matrix_clean = np.copy(gf_matrix)
        gf_matrix_clean[np.isinf(gf_matrix_clean)] = np.nan
        
        min_gf_idx = np.unravel_index(np.nanargmin(gf_matrix_clean), gf_matrix_clean.shape)
        min_gf = gf_matrix_clean[min_gf_idx]
        min_gf_width = results['width_values'][min_gf_idx[1]]
        min_gf_dw = results['dw_values'][min_gf_idx[0]]
        
        # Find minimum BIC
        bic_matrix = results['bic_matrix']
        bic_matrix_clean = np.copy(bic_matrix)
        bic_matrix_clean[np.isinf(bic_matrix_clean)] = np.nan
        
        min_bic_idx = np.unravel_index(np.nanargmin(bic_matrix_clean), bic_matrix_clean.shape)
        min_bic = bic_matrix_clean[min_bic_idx]
        min_bic_width = results['width_values'][min_bic_idx[1]]
        min_bic_dw = results['dw_values'][min_bic_idx[0]]
        
        return {
            'best_gf': {
                'width': min_gf_width,
                'dw': min_gf_dw,
                'gf_value': min_gf,
                'bic_value': results['bic_matrix'][min_gf_idx]
            },
            'best_bic': {
                'width': min_bic_width,
                'dw': min_bic_dw,
                'gf_value': results['gf_matrix'][min_bic_idx],
                'bic_value': min_bic
            }
        }
    
    def apply_optimal_width_dw_1layer(self, results, criterion='GF'):
        """
        Apply the optimal width and DW values from sweep results to the model.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_width_dw_1layer
        criterion : str, optional
            Criterion for selecting optimal values ('GF' or 'BIC'). Default: 'GF'
        """
        if self.layers != 1:
            raise ValueError("This function is only for 1-layer models")
        
        optimal_vals = self.get_optimal_width_dw_1layer(results)
        
        if criterion.upper() == 'GF':
            width = optimal_vals['best_gf']['width']
            dw = optimal_vals['best_gf']['dw']
            print(f"Applying optimal GF values: Width = {width:.1f}, DW = {dw:.1f}")
        else:
            width = optimal_vals['best_bic']['width']
            dw = optimal_vals['best_bic']['dw']
            print(f"Applying optimal BIC values: Width = {width:.1f}, DW = {dw:.1f}")
        
        # Update model parameters
        self.model_params['trapezoids'][0]['width'] = width
        self.model_params['trapezoids'][1]['width'] = width
        self.model_params['DW'] = dw
        
        # Update traditional parameters
        self.update_traditional_from_model_params()
        
        # Recalculate simulation
        self.SimInt = self.simulate_structure()
        self.GF = self.GF_calc(self.SimInt)
        self.BIC = self.BIC_calc(self.GF)
        
        print(f"Model updated. New GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
    
    def simulate_structure(self, *args, **kwargs):
        """
        Simulate trapezoid structure intensity.
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (also sets self.SimInt)
        """
        return self.SimTrap_SM(*args, **kwargs)

    def _update_model_with_optimization_result(self, result, param_names, initial_model_params):
        """
        Apply optimizer result to ``model_params``.

        Base CDSAXS_Model only updates expanded ``trapezoids``; SiGe ``trap_{i}_*`` names refer to
        **design** indices when ``design_trapezoids`` is present. Writing to expanded segments
        leaves design geometry unchanged so ``simulate_structure()`` and ``print_parameter_changes``
        GF/BIC can disagree with the objective seen during optimization.
        """
        if 'design_trapezoids' not in initial_model_params:
            return super()._update_model_with_optimization_result(
                result, param_names, initial_model_params
            )

        if hasattr(result, 'x'):
            optimal_values = result.x
        elif hasattr(result, 'best_x'):
            optimal_values = result.best_x
        else:
            raise ValueError("Could not extract optimal values from optimization result")

        optimized_params = copy.deepcopy(initial_model_params)

        if isinstance(self.Bk, np.ndarray):
            optimized_bk = self.Bk.copy()
        else:
            optimized_bk = self.Bk

        for i, param_name in enumerate(param_names):
            if param_name.startswith('trap_'):
                parts = param_name.split('_')
                trap_idx = int(parts[1])
                param_type = '_'.join(parts[2:])
                if trap_idx < len(optimized_params['design_trapezoids']):
                    optimized_params['design_trapezoids'][trap_idx][param_type] = optimal_values[i]
            elif param_name.startswith('Bk_'):
                bk_idx = int(param_name.split('_')[1])
                if isinstance(optimized_bk, np.ndarray):
                    optimized_bk[bk_idx] = optimal_values[i]
                else:
                    n_columns = self.Intensity.shape[1]
                    optimized_bk = np.full(n_columns, optimized_bk)
                    optimized_bk[bk_idx] = optimal_values[i]
            elif param_name == 'Bk':
                optimized_bk = optimal_values[i]
            else:
                optimized_params[param_name] = optimal_values[i]

        optimized_params['Bk'] = optimized_bk.tolist() if isinstance(optimized_bk, np.ndarray) else optimized_bk

        tmp_model_params = {
            'trapezoids': [t.copy() for t in optimized_params['design_trapezoids']],
            'layers': optimized_params.get('design_layers', len(optimized_params['design_trapezoids']) - 1),
            'slds': optimized_params.get('design_slds', optimized_params.get('slds', None)),
        }
        constraints = optimized_params.get('constraints', None)
        if constraints:
            self._apply_constraints(tmp_model_params, constraints)
            optimized_params['design_trapezoids'] = [t.copy() for t in tmp_model_params['trapezoids']]

        expanded_traps, expanded_slds, expanded_layers, _, _, _, _ = self._expand_typed_layers(tmp_model_params)
        optimized_params['trapezoids'] = expanded_traps
        optimized_params['layers'] = int(expanded_layers)
        optimized_params['slds'] = expanded_slds

        return optimized_params

    def _apply_mcmc_parameters(self, params, param_names):
        """
        Apply MCMC sample (or MAP/best) to the model.

        Base implementation writes ``trap_*`` into expanded ``trapezoids`` only, which breaks
        typed design stacks (ellipse, ``curved_sides``, etc.). Reuse design-level apply + expand
        so post-MCMC ``SimInt``/structure plots match the likelihood path.
        """
        if 'design_trapezoids' not in self.model_params:
            return super()._apply_mcmc_parameters(params, param_names)

        res = SimpleNamespace(x=np.asarray(params, dtype=float))
        self.model_params = self._update_model_with_optimization_result(
            res, param_names, copy.deepcopy(self.model_params)
        )
        self.update_traditional_from_model_params()
        self.SimInt = self.simulate_structure()
        self.GF = self.GF_calc(self.SimInt)
        self.BIC = self.BIC_calc(self.GF)

    def _trapezoid_optimization_wrapper(self, optimization_values):
            """
            Enhanced wrapper function for trapezoid optimization with SLD support.
            """
            try:
                # CRITICAL FIX: Always convert to numpy array first
                if not isinstance(optimization_values, np.ndarray):
                    optimization_values = np.array(optimization_values, dtype=float)
                
                # Get parameter names from available sources
                if hasattr(self, 'param_names'):
                    param_names = self.param_names
                elif hasattr(self, 'mcmc_param_names'):
                    param_names = self.mcmc_param_names
                else:
                    # Generate parameter names from optimization parameters
                    param_names = list(self.model_params.get('optimization', {}).keys())
                
                if len(optimization_values) != len(param_names):
                    raise ValueError(f"Parameter count mismatch: got {len(optimization_values)}, expected {len(param_names)}")
                
                # Call SimTrap_GF with numpy array
                return self.SimTrap_GF(optimization_values, param_names, self.Intensity, self.Qx, self.Qz)
                
            except Exception as e:
                print(f"Error in trapezoid wrapper: {e}")
                return float('inf')

        
    
# Create an alias for backward compatibility
SiGeModel = SiGeModelArray
