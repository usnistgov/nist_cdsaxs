# Updated Trapezoid_model.py with common functions moved to base class

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Patch
import copy
from scipy.optimize import differential_evolution

from CDSAXS_base_model import CDSAXS_Model

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
        Currently supports `Layer_Type: 'Ellipse'` (case-insensitive).
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
            else:
                if trap_width is None:
                    raise ValueError(f"Trapezoid {idx} missing 'width'")
                if trap_height is None:
                    raise ValueError(f"Trapezoid {idx} missing 'height'")
                expanded_traps.append({'width': trap_width, 'height': trap_height, 'twidth': trap_twidth})
                expanded_slds.append(sld_val)

        # Ensure the last segment has an explicit top width to avoid PAR[T+1] out-of-range in legacy logic
        if expanded_traps:
            if expanded_traps[-1].get('twidth', None) is None:
                expanded_traps[-1]['twidth'] = expanded_traps[-1]['width']

        expanded_layers = max(0, len(expanded_traps) - 1)

        return expanded_traps, expanded_slds, expanded_layers, design_layers, design_traps, design_slds_list

    def _ensure_expanded_model_params(self):
        """
        Ensure `self.model_params` is in expanded (simulation) form.
        If typed layers are present, this mutates `self.model_params` in-place and preserves
        the original user-provided design structure under `design_*` keys.
        """
        if not hasattr(self, 'model_params') or self.model_params is None:
            return

        # If we already expanded and current trapezoids contain no Layer_Type markers, do nothing
        trapezoids = self.model_params.get('trapezoids', None)
        if not self._has_typed_layers(trapezoids):
            return

        # Avoid re-expanding an already-expanded structure unless the current list is still typed
        if self.model_params.get('_expanded_from_typed_layers', False) and not self._has_typed_layers(trapezoids):
            return

        (
            expanded_traps,
            expanded_slds,
            expanded_layers,
            design_layers,
            design_traps,
            design_slds_list,
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
            
        # Create default limits if not provided
        if param_limits is None:
            param_limits = {}
            
            # Add trapezoid parameters
            for i, trap in enumerate(self.model_params['trapezoids']):
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
        
        # Add SLD parameters - they're treated just like other parameters
        if hasattr(self, 'sld_values'):
            for i, sld_val in enumerate(self.sld_values):
                param_name = f'sld_{i}'
                
                # Only add to optimization if not already specified
                if param_name not in param_limits:
                    # Set reasonable default bounds for SLD values
                    param_limits[param_name] = {
                        'min': max(0.1, sld_val * 0.5),  # Positive SLD with 50% range
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
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = parts[2]
            return self.model_params['trapezoids'][trap_idx][param_type]
        
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
            # Trapezoid parameter
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = parts[2]
            self.model_params['trapezoids'][trap_idx][param_type] = value
            
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
        
        # Update traditional parameters
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
            
            # Calculate form factor using the enhanced coordinates with SLD
            form = self.FreeFormTrapezoid(Coord, layers, Qx, Qz)
            if form is None:
                raise RuntimeError("Failed to calculate form factor in FreeFormTrapezoid")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qx, 2) + np.power(Qz, 2)) * np.power(DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = form * M
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
            
            # Initialize SLD array
            if hasattr(self, 'sld_values'):
                temp_sld_values = self.sld_values.copy()
            else:
                temp_sld_values = np.ones(self.layers + 1)
            
            # Update parameters with optimization values
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]  # 'width', 'height', or 'twidth'
                    
                    # Make sure we have a deep copy of trapezoids to avoid modifying the original
                    if 'trapezoids' not in params or params['trapezoids'] is self.model_params['trapezoids']:
                        params['trapezoids'] = [trap.copy() for trap in self.model_params['trapezoids']]
                    
                    params['trapezoids'][trap_idx][param_type] = optimization_values[i]
                    
                elif param_name.startswith('sld_'):
                    # SLD parameter - treat just like any other parameter
                    sld_idx = int(param_name.split('_')[1])
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
            
            # Create temporary PAR array for compatibility
            temp_PAR = np.zeros((self.layers + 1, 3))
            for i, trap in enumerate(params['trapezoids']):
                if i <= self.layers:
                    temp_PAR[i, 0] = trap['width']
                    temp_PAR[i, 1] = trap['height']
                    # Handle twidth - use None if not present (will be handled by SymCoordAssign)
                    temp_PAR[i, 2] = trap.get('twidth', None)
            
            # Extract global parameters
            temp_DW = params['DW']
            temp_I0 = params['I0']
            
            # Use SymCoordAssign with current SLD values
            Coord = self.SymCoordAssign(temp_PAR, self.layers, sld_values=temp_sld_values)
            if Coord is None:
                raise RuntimeError("Failed to assign coordinates with SLD values")
            
            # Calculate form factor
            form = self.FreeFormTrapezoid(Coord, self.layers, Qx, Qz)
            if form is None:
                raise RuntimeError("Failed to calculate form factor")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qx, 2) + np.power(Qz, 2)) * np.power(temp_DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = form * M
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
                    verbose=False,**kwargs):
        """
        Performs differential evolution optimization for CDSAXS trapezoid model fitting
        with array background support and shows before/after comparison plots.
        
        Fixed to respect verbose parameter properly.
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
            
            # Update default parameters with any provided kwargs
            optimization_params = {**default_params, **kwargs}
            
            # Store current parameters and simulation results for before/after comparison
            initial_model_params = copy.deepcopy(self.model_params)
            
            # Calculate initial simulated intensity if not already done
            if not hasattr(self, 'SimInt') or self.SimInt is None:
                self.SimInt = self.SimTrap_SM()
                
            # Store initial simulation results
            initial_simInt = copy.deepcopy(self.SimInt)
            
            # Calculate initial goodness of fit if not already done
            if not hasattr(self, 'GF_Initial') or self.GF_Initial is None:
                self.GF_Initial = self.GF_calc(self.SimInt)
            
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
            
            # Update model parameters with optimized values
            optimized_params = self.model_params.copy()
            
            # Make a deep copy of trapezoids to avoid modifying the original
            optimized_params['trapezoids'] = [trap.copy() for trap in self.model_params['trapezoids']]
            
            # Initialize background array for updates
            if isinstance(self.Bk, np.ndarray):
                optimized_bk = self.Bk.copy()
            else:
                optimized_bk = self.Bk
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]  # 'width', 'height', or 'twidth'
                    
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
                self._plot_optimization_results(initial_model_params, initial_simInt,
                                            plot_structure, plot_grid, plot_combined)
            
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
                                  plot_structure=True, plot_grid=True, plot_combined=True):
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
        """
        import matplotlib.pyplot as plt
        
        # Plot trapezoid structure comparison on the same plot
        if plot_structure:
            plt.figure(figsize=(10, 6))
            
            # Plot initial trapezoid structure with dashed lines and transparency
            self._plot_trapezoid_structure(initial_model_params, 
                                        linestyle='--', 
                                        color='blue', 
                                        alpha=0.7,
                                        label='Initial')
            
            # Plot optimized trapezoid structure with solid lines
            self._plot_trapezoid_structure(self.model_params, 
                                        linestyle='-', 
                                        color='red', 
                                        alpha=1.0,
                                        label='Optimized')
            
            plt.title('Trapezoid Structure Comparison')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Plot QzCut comparisons - grid of individual cuts
        if plot_grid:
            self._plot_qzcut_grid(initial_simInt)
        
        # Plot combined view with all cuts
        if plot_combined:
            self._plot_qzcut_combined(initial_simInt)
    
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
        
        Parameters:
        -----------
        model_params : dict
            Dictionary containing model parameters
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
        ax = plt.gca()
        trapezoids = model_params['trapezoids']
        layers = model_params['layers']

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

            if len(slds) == layers + 1:
                slds = slds[:layers]
            elif len(slds) == 1 and layers > 1:
                slds = np.full(layers, float(slds[0]), dtype=float)
            elif len(slds) != layers:
                slds = np.resize(slds, layers).astype(float)

            finite_slds = slds[np.isfinite(slds)]
            if finite_slds.size == 0:
                sld_min, sld_max = 0.0, 1.0
            else:
                sld_min, sld_max = float(np.min(finite_slds)), float(np.max(finite_slds))
            denom = (sld_max - sld_min) if not np.isclose(sld_max, sld_min) else None

            base_w = float(trapezoids[0]['width'])
            height0 = 0.0
            for i in range(int(layers)):
                h = float(trapezoids[i]['height'])
                if h <= 0:
                    continue

                w0 = float(trapezoids[i]['width'])
                if trapezoids[i]['twidth'] is None:
                    w1 = float(trapezoids[i + 1]['width'])
                else:
                    w1 = float(trapezoids[i]['twidth'])

                xL0 = (base_w - w0) / 2.0
                xR0 = xL0 + w0
                xL1 = (base_w - w1) / 2.0
                xR1 = xL1 + w1

                y0 = height0
                y1 = height0 + h

                sld_val = float(slds[i])
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
    
    def _plot_qzcut_grid(self, initial_simInt):
        """
        Plot a grid of QzCut comparisons with both initial and optimized results.
        
        Parameters:
        -----------
        initial_simInt : numpy.ndarray
            Simulated intensity before optimization
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
            ax.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=1.0, 
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
    
    def _plot_qzcut_combined(self, initial_simInt):
        """
        Plot all QzCuts on one axis with both initial and optimized results.
        
        Parameters:
        -----------
        initial_simInt : numpy.ndarray
            Simulated intensity before optimization
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
            plt.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=0.6, linewidth=1.5)
        
        plt.title('Intensity Comparison - All Cuts')
        plt.xlabel('Qz (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
    
    
    
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

        ax = self._plot_trapezoid_structure(
            self.model_params,
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
        layers = self.model_params['layers']
        if shade_by_sld and show_sld_legend and layers > 0:
            if 'slds' in self.model_params:
                slds_all = np.array(self.model_params['slds'], dtype=float)
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

