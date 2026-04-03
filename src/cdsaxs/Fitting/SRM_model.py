# Updated Trapezoid_model.py with common functions moved to base class

import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.patches import Polygon, Patch
from scipy.optimize import (
    differential_evolution, 
    dual_annealing, 
    shgo, 
    basinhopping, 
    minimize
)
from tqdm import tqdm
import seaborn as sns

from .CDSAXS_base_model import CDSAXS_Model

class TrapezoidModelArray(CDSAXS_Model):
    """
    CDSAXS model for trapezoid structures with array-based background support.
    Each column can have its own background value.
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
        super().__init__('trapezoid', model, layers, PAR, SLD, DW, I0, Bk, Pitch, 
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
            trapezoid = {
                'width': self.PAR[i, 0],
                'height': self.PAR[i, 1]
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
        
        # Multi-stack (SRM) support: store number of stacks and spacing
        if hasattr(self, 'n_trapezoid_stacks'):
            try:
                self.model_params['n_stacks'] = int(self.n_trapezoid_stacks)
            except Exception:
                pass
        if hasattr(self, 'stack_spacing'):
            try:
                self.model_params['stack_spacing'] = float(self.stack_spacing)
            except Exception:
                pass
            
        return self.model_params
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary including SLD support.
        """
        if not hasattr(self, 'model_params'):
            return
            
        # Update PAR from trapezoids
        trapezoids = self.model_params['trapezoids']
        if not hasattr(self, 'PAR') or self.PAR is None or self.PAR.shape[0] != len(trapezoids):
            self.PAR = np.zeros((len(trapezoids), 2))
            
        for i, trap in enumerate(trapezoids):
            self.PAR[i, 0] = trap['width']
            self.PAR[i, 1] = trap['height']
        
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
        
        # Multi-stack (SRM) support: restore number of stacks and spacing
        if 'n_stacks' in self.model_params:
            try:
                self.n_trapezoid_stacks = int(self.model_params['n_stacks'])
            except Exception:
                pass
        if 'stack_spacing' in self.model_params:
            try:
                self.stack_spacing = float(self.model_params['stack_spacing'])
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Second independent stack family (geometry + SLD) from model_params
        # ------------------------------------------------------------------
        if 'layers_2' in self.model_params and 'trapezoids_2' in self.model_params:
            try:
                self.layers2 = int(self.model_params['layers_2'])
                traps2 = self.model_params['trapezoids_2']
                # Build PAR2 with same convention as primary PAR
                self.PAR2 = np.zeros((self.layers2 + 1, 2), dtype=float)
                for i, trap in enumerate(traps2):
                    if i <= self.layers2:
                        self.PAR2[i, 0] = trap['width']
                        self.PAR2[i, 1] = trap['height']
            except Exception:
                # If anything goes wrong, just skip defining PAR2/layers2
                pass

        if 'slds_2' in self.model_params:
            try:
                sld2 = self.model_params['slds_2']
                if isinstance(sld2, list):
                    self.sld_values2 = np.array(sld2, dtype=float)
                else:
                    self.sld_values2 = np.array([float(sld2)])
            except Exception:
                pass

        if 'n_stacks_2' in self.model_params:
            try:
                self.n_trapezoid_stacks_2 = int(self.model_params['n_stacks_2'])
            except Exception:
                pass
        if 'stack_spacing_2' in self.model_params:
            try:
                self.stack_spacing_2 = float(self.model_params['stack_spacing_2'])
            except Exception:
                pass
        if 'x_offset_2' in self.model_params:
            try:
                self.x_offset_2 = float(self.model_params['x_offset_2'])
            except Exception:
                pass
            
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
        Initialize optimization parameters with bounds.

        Bk / Bk_i and sld_i are opt-in via an explicit optimization dict.
        """
        if not hasattr(self, 'model_params'):
            self.build_model_params_from_traditional()
            
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
            
        trapezoids = self.model_params['trapezoids']
        layers = self.model_params['layers']
        
        # Create PAR array
        PAR = np.zeros([layers + 1, 2])
        for i, trap in enumerate(trapezoids):
            if i <= layers:
                PAR[i, 0] = trap['width']
                PAR[i, 1] = trap['height']
                
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
            Coord = np.zeros([layers+1, 5, 1])
            for T in range(layers+1):
                if T == 0:
                    Coord[T, 0, 0] = 0
                    Coord[T, 1, 0] = PAR[0, 0]
                    Coord[T, 2, 0] = PAR[0, 1]
                    Coord[T, 3, 0] = 0
                    Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
                else:
                    Coord[T, 0, 0] = Coord[T-1, 0, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
                    Coord[T, 1, 0] = Coord[T, 0, 0] + PAR[T, 0]
                    Coord[T, 2, 0] = PAR[T, 1]
                    Coord[T, 3, 0] = 0
                    Coord[T, 4, 0] = 1  # SLD - assigned to be 1 for a single material
            
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
    
    def SymCoordAssign(self, PAR=None, layers=None, sld_values=None,
                       n_stacks=None, stack_spacing=None):
        """
        Coordinate assignment for one or more identical trapezoid stacks.
        
        The base (untranslated) stack is built exactly as before in the
        third‑dimension slice 0. Additional stacks are created by translating
        the x‑coordinates by a constant offset `stack_spacing`:
        
            Coord[:, :, 0]  -> base stack
            Coord[:, :, 1]  -> base stack shifted by +1 * stack_spacing
            Coord[:, :, 2]  -> base stack shifted by +2 * stack_spacing
            ...
        
        For backward compatibility:
        - If `n_stacks` is None, it falls back to `self.n_trapezoid_stacks`
          when present, otherwise 1.
        - If `stack_spacing` is None and multiple stacks are requested, it
          falls back to `self.stack_spacing` or `self.Pitch` if available.
        """
        try:
            using_self = False
            
            # --- Input / attribute fallbacks ---------------------------------
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
                sld_array = np.ones(layers, dtype=float)
            
            # STRICT VALIDATION
            if len(sld_array) != layers:
                raise ValueError(
                    f"SLD array length ({len(sld_array)}) must exactly match number of layers ({layers}). "
                    f"Each layer requires its own SLD value."
                )
            
            # Validate PAR
            if not isinstance(PAR, np.ndarray) or len(PAR) < layers + 1 or PAR.shape[1] < 2:
                raise ValueError("Invalid PAR array dimensions")
            
            # --- Determine number of stacks and spacing -----------------------
            if n_stacks is None:
                if hasattr(self, 'n_trapezoid_stacks'):
                    n_stacks = int(self.n_trapezoid_stacks)
                elif hasattr(self, 'model_params') and 'n_stacks' in self.model_params:
                    n_stacks = int(self.model_params['n_stacks'])
                else:
                    n_stacks = 1
            else:
                n_stacks = int(n_stacks)
            
            if n_stacks < 1:
                n_stacks = 1
            
            if stack_spacing is None and n_stacks > 1:
                if hasattr(self, 'stack_spacing'):
                    stack_spacing = float(self.stack_spacing)
                elif hasattr(self, 'model_params') and 'stack_spacing' in self.model_params:
                    stack_spacing = float(self.model_params['stack_spacing'])
                elif hasattr(self, 'Pitch') and self.Pitch is not None:
                    stack_spacing = float(self.Pitch)
            
            # If no meaningful spacing is available, silently collapse to 1 stack
            if n_stacks > 1 and (stack_spacing is None or np.isclose(stack_spacing, 0.0)):
                n_stacks = 1
            
            # --- Build base (untranslated) stack in 2D -----------------------
            base_coord = np.zeros([layers + 1, 5])
            
            for layer_idx in range(layers):
                T = layer_idx
                
                if T == 0:
                    # Bottom layer
                    base_coord[T, 0] = 0
                    base_coord[T, 1] = PAR[0, 0]
                    base_coord[T, 2] = PAR[0, 1]
                    base_coord[T, 3] = 0
                else:
                    # Upper layers
                    base_coord[T, 0] = base_coord[T-1, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
                    base_coord[T, 1] = base_coord[T, 0] + PAR[T, 0]
                    base_coord[T, 2] = PAR[T, 1]
                    base_coord[T, 3] = 0
                
                # SLD for this layer
                base_coord[T, 4] = sld_array[layer_idx]
            
            # Handle the top vertex (T = layers)
            T = layers
            base_coord[T, 0] = base_coord[T-1, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
            base_coord[T, 1] = base_coord[T, 0] + PAR[T, 0]
            base_coord[T, 2] = PAR[T, 1]
            base_coord[T, 3] = 0
            base_coord[T, 4] = 0.0  # Top vertex - no layer associated
            
            # --- Replicate into multiple translated stacks -------------------
            if n_stacks == 1:
                Coord = base_coord[:, :, np.newaxis]
            else:
                Coord = np.zeros([layers + 1, 5, n_stacks], dtype=float)
                for stack_idx in range(n_stacks):
                    Coord[:, :, stack_idx] = base_coord
                    dx = stack_idx * stack_spacing
                    Coord[:, 0, stack_idx] += dx  # left x
                    Coord[:, 1, stack_idx] += dx  # right x
            
            # Always return the coordinate array; when called with self
            # attributes we also store it on the instance.
            if using_self:
                self.Coord = Coord
            
            return Coord
            
        except Exception as e:
            print(f"Error in SymCoordAssign_Alternative: {str(e)}")
            if using_self:
                return False
            return None

    def SymCoordAssign_SecondStack(self, PAR=None, layers=None, sld_values=None,
                                   n_stacks=None, stack_spacing=None,
                                   x_offset=None):
        """
        Coordinate assignment for a second, independent trapezoid stack family.
        
        This builds a completely separate set of coordinates (Coord2) that can
        have its *own* PAR/layers and its own number of stacks and spacing.
        The entire second family is positioned relative to the first by an
        additional x‑offset:
        
            - Base second stack is shifted by `x_offset` relative to the
              coordinate frame of the first stack.
            - Additional stacks in this second family are spaced by
              `stack_spacing` just like in SymCoordAssign.
        """
        try:
            using_self = False
            
            # --- Input / attribute fallbacks ---------------------------------
            if PAR is None:
                if hasattr(self, 'PAR2'):
                    PAR = self.PAR2
                    using_self = True
                else:
                    raise AttributeError("Missing PAR for second stack (PAR2)")
            
            if layers is None:
                if hasattr(self, 'layers2'):
                    layers = self.layers2
                else:
                    raise AttributeError("Missing layers for second stack (layers2)")
            
            if sld_values is not None:
                sld_array = np.array(sld_values, dtype=float)
            elif hasattr(self, 'sld_values2'):
                sld_array = self.sld_values2.copy()
            elif hasattr(self, 'sld_values'):
                # Fallback: reuse primary SLDs
                sld_array = self.sld_values.copy()
            else:
                sld_array = np.ones(layers, dtype=float)
            
            if len(sld_array) != layers:
                raise ValueError(
                    f"Second stack SLD array length ({len(sld_array)}) must exactly match "
                    f"number of layers ({layers})."
                )
            
            if not isinstance(PAR, np.ndarray) or len(PAR) < layers + 1 or PAR.shape[1] < 2:
                raise ValueError("Invalid PAR array dimensions for second stack")
            
            # --- Determine number of stacks and spacing -----------------------
            if n_stacks is None:
                if hasattr(self, 'n_trapezoid_stacks_2'):
                    n_stacks = int(self.n_trapezoid_stacks_2)
                elif hasattr(self, 'model_params') and 'n_stacks_2' in self.model_params:
                    n_stacks = int(self.model_params['n_stacks_2'])
                else:
                    n_stacks = 1
            else:
                n_stacks = int(n_stacks)
            
            if n_stacks < 1:
                n_stacks = 1
            
            if stack_spacing is None and n_stacks > 1:
                if hasattr(self, 'stack_spacing_2'):
                    stack_spacing = float(self.stack_spacing_2)
                elif hasattr(self, 'model_params') and 'stack_spacing_2' in self.model_params:
                    stack_spacing = float(self.model_params['stack_spacing_2'])
                elif hasattr(self, 'stack_spacing'):
                    # Fallback: reuse primary spacing
                    stack_spacing = float(self.stack_spacing)
            
            if n_stacks > 1 and (stack_spacing is None or np.isclose(stack_spacing, 0.0)):
                n_stacks = 1
            
            # Overall horizontal offset of this second family
            if x_offset is None:
                if hasattr(self, 'x_offset_2'):
                    x_offset = float(self.x_offset_2)
                elif hasattr(self, 'model_params') and 'x_offset_2' in self.model_params:
                    x_offset = float(self.model_params['x_offset_2'])
                else:
                    x_offset = 0.0
            else:
                x_offset = float(x_offset)
            
            # --- Build base (untranslated) second stack in 2D ----------------
            base_coord = np.zeros([layers + 1, 5])
            
            for layer_idx in range(layers):
                T = layer_idx
                
                if T == 0:
                    base_coord[T, 0] = 0
                    base_coord[T, 1] = PAR[0, 0]
                    base_coord[T, 2] = PAR[0, 1]
                    base_coord[T, 3] = 0
                else:
                    base_coord[T, 0] = base_coord[T-1, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
                    base_coord[T, 1] = base_coord[T, 0] + PAR[T, 0]
                    base_coord[T, 2] = PAR[T, 1]
                    base_coord[T, 3] = 0
                
                base_coord[T, 4] = sld_array[layer_idx]
            
            T = layers
            base_coord[T, 0] = base_coord[T-1, 0] + 0.5 * (PAR[T-1, 0] - PAR[T, 0])
            base_coord[T, 1] = base_coord[T, 0] + PAR[T, 0]
            base_coord[T, 2] = PAR[T, 1]
            base_coord[T, 3] = 0
            base_coord[T, 4] = 0.0
            
            # Apply global offset for this second family
            base_coord[:, 0] += x_offset
            base_coord[:, 1] += x_offset
            
            # --- Replicate into multiple translated stacks -------------------
            if n_stacks == 1:
                Coord2 = base_coord[:, :, np.newaxis]
            else:
                Coord2 = np.zeros([layers + 1, 5, n_stacks], dtype=float)
                for stack_idx in range(n_stacks):
                    Coord2[:, :, stack_idx] = base_coord
                    dx = stack_idx * stack_spacing
                    Coord2[:, 0, stack_idx] += dx
                    Coord2[:, 1, stack_idx] += dx
            
            # Always return the coordinate array; when called with self
            # attributes we also store it on the instance.
            if using_self:
                self.Coord2 = Coord2
            
            return Coord2
        
        except Exception as e:
            print(f"Error in SymCoordAssign_SecondStack: {str(e)}")
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
        
        elif param_name.startswith('trap2_'):
            # Second-stack trapezoid geometry, e.g. trap2_0_width / trap2_0_height
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = parts[2]  # 'width' or 'height'
            # Prefer model_params['trapezoids_2'] if present
            if hasattr(self, 'model_params') and 'trapezoids_2' in self.model_params:
                traps2 = self.model_params['trapezoids_2']
                if trap_idx < len(traps2) and param_type in traps2[trap_idx]:
                    return traps2[trap_idx][param_type]
            # Fallback: use PAR2 if defined
            if hasattr(self, 'PAR2'):
                col = 0 if param_type == 'width' else 1
                if trap_idx < self.PAR2.shape[0]:
                    return float(self.PAR2[trap_idx, col])
            raise ValueError(f"Unknown second-stack trapezoid parameter: {param_name}")
        
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
        
        # SRM-specific scalar parameters that may be optimized
        elif param_name in ['n_stacks', 'stack_spacing',
                            'n_stacks_2', 'stack_spacing_2', 'x_offset_2']:
            if hasattr(self, 'model_params') and param_name in self.model_params:
                return self.model_params[param_name]
            if hasattr(self, param_name):
                return getattr(self, param_name)
            raise ValueError(f"Unknown SRM parameter: {param_name}")
        
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
        n_sld_values = self.layers  # Number of actual trapezoids/layers
        
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
            
            # Determine how many trapezoid stacks we have in the 3rd dimension
            if Coord.ndim == 3:
                n_stacks = Coord.shape[2]
            else:
                # Fallback for any unexpected shapes – treat as a single stack
                n_stacks = 1
                Coord = Coord.reshape(Coord.shape[0], Coord.shape[1], 1)
            
            # Initialize form factor array
            form = np.zeros([len(Qx[:,1]), len(Qx[1,:])])
            
            # Calculate form factor: sum contribution from each trapezoid stack
            for stack_idx in range(n_stacks):
                H1 = Coord[0, 3, stack_idx]
                H2 = H1
                
                for i in range(int(layers)):
                    H2 = H2 + Coord[i, 2, stack_idx]
                    if i > 0:
                        H1 = H1 + Coord[i-1, 2, stack_idx]
                        
                    x1 = Coord[i, 0, stack_idx]
                    x4 = Coord[i, 1, stack_idx]
                    x2 = Coord[i+1, 0, stack_idx]
                    x3 = Coord[i+1, 1, stack_idx]
                    
                    # Avoid division by zero
                    x2 = x1 - 1e-6 if np.isclose(x2, x1) else x2
                    x4 = x3 - 1e-6 if np.isclose(x4, x3) else x4
                    
                    SL = Coord[i, 2, stack_idx] / (x2 - x1)
                    SR = -Coord[i, 2, stack_idx] / (x4 - x3)
                    
                    A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
                    A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
                    form = form + (1j/Qx)*(A1-A2)*Coord[i, 4, stack_idx]
            
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
            
            # Generate coordinates for the primary stack family
            if PAR is not None:
                Coord = self.SymCoordAssign(PAR, layers)
                if Coord is None or (using_self and Coord is False):
                    raise RuntimeError("Failed to assign coordinates in SymCoordAssign")
            else:
                if not hasattr(self, 'Coord'):
                    raise AttributeError("Missing required attribute: Coord")
                Coord = self.Coord
            
            # Calculate form factor for the primary stack family
            form = self.FreeFormTrapezoid(Coord, layers, Qx, Qz)
            if form is None:
                raise RuntimeError("Failed to calculate form factor in FreeFormTrapezoid")

            # Optionally add a second, independent stack family if configured
            try:
                has_second = hasattr(self, 'PAR2') and hasattr(self, 'layers2')
            except Exception:
                has_second = False

            if has_second:
                Coord2 = self.SymCoordAssign_SecondStack()
                if Coord2 is None or (isinstance(Coord2, bool) and not Coord2):
                    raise RuntimeError("Failed to assign coordinates for second stack family")
                form2 = self.FreeFormTrapezoid(Coord2, int(self.layers2), Qx, Qz)
                if form2 is None:
                    raise RuntimeError("Failed to calculate form factor for second stack family")
                # Total form factor is coherent sum of both families
                form = form + form2
            
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
            
            # Initialize SLD array (primary stack family)
            if hasattr(self, 'sld_values'):
                temp_sld_values = self.sld_values.copy()
            else:
                temp_sld_values = np.ones(self.layers + 1)

            # ------------------------------------------------------------------
            # SRM: multi-stack + second-stack temporary variables
            # ------------------------------------------------------------------
            # Primary stack family
            temp_n_stacks = getattr(self, 'n_trapezoid_stacks', None)
            temp_stack_spacing = getattr(self, 'stack_spacing', None)

            # Second independent stack family
            has_second = hasattr(self, 'PAR2') and hasattr(self, 'layers2')
            if has_second:
                temp_PAR2 = self.PAR2.copy()
                temp_layers2 = int(self.layers2)
                if hasattr(self, 'sld_values2'):
                    temp_sld_values2 = self.sld_values2.copy()
                else:
                    temp_sld_values2 = np.ones(temp_layers2, dtype=float)
                temp_n_stacks_2 = getattr(self, 'n_trapezoid_stacks_2', None)
                temp_stack_spacing_2 = getattr(self, 'stack_spacing_2', None)
                temp_x_offset_2 = getattr(self, 'x_offset_2', 0.0)
            else:
                temp_PAR2 = None
                temp_layers2 = 0
                temp_sld_values2 = None
                temp_n_stacks_2 = None
                temp_stack_spacing_2 = None
                temp_x_offset_2 = 0.0
            
            # Update parameters with optimization values
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]  # 'width' or 'height'
                    
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

                # --------------------------------------------------------------
                # SRM-specific optimization parameters
                # --------------------------------------------------------------
                elif param_name == 'n_stacks':
                    temp_n_stacks = int(optimization_values[i])
                elif param_name == 'stack_spacing':
                    temp_stack_spacing = float(optimization_values[i])
                elif param_name == 'n_stacks_2':
                    temp_n_stacks_2 = int(optimization_values[i])
                elif param_name == 'stack_spacing_2':
                    temp_stack_spacing_2 = float(optimization_values[i])
                elif param_name == 'x_offset_2':
                    temp_x_offset_2 = float(optimization_values[i])
                elif param_name.startswith('trap2_'):
                    # Geometry of second stack family: trap2_{i}_{width|height}
                    parts = param_name.split('_')
                    if len(parts) >= 3 and temp_PAR2 is not None:
                        trap2_idx = int(parts[1])
                        param_type2 = parts[2]
                        if trap2_idx <= temp_layers2:
                            if param_type2 == 'width':
                                temp_PAR2[trap2_idx, 0] = optimization_values[i]
                            elif param_type2 == 'height':
                                temp_PAR2[trap2_idx, 1] = optimization_values[i]
                elif param_name.startswith('sld2_'):
                    # SLD for second stack family
                    if temp_sld_values2 is not None:
                        sld2_idx = int(param_name.split('_')[1])
                        if sld2_idx < len(temp_sld_values2):
                            temp_sld_values2[sld2_idx] = optimization_values[i]

                else:
                    # Global parameter (DW, I0) or anything stored in params
                    params[param_name] = optimization_values[i]
            
            # Create temporary PAR array for compatibility
            temp_PAR = np.zeros((self.layers + 1, 2))
            for i, trap in enumerate(params['trapezoids']):
                if i <= self.layers:
                    temp_PAR[i, 0] = trap['width']
                    temp_PAR[i, 1] = trap['height']
            
            # Extract global parameters
            temp_DW = params['DW']
            temp_I0 = params['I0']
            
            # Use SymCoordAssign for the primary stack family, including
            # potentially optimized stack count / spacing.
            Coord = self.SymCoordAssign(
                temp_PAR,
                self.layers,
                sld_values=temp_sld_values,
                n_stacks=temp_n_stacks,
                stack_spacing=temp_stack_spacing,
            )
            if Coord is None:
                raise RuntimeError("Failed to assign coordinates with SLD values")
            
            # Calculate form factor for primary family
            form = self.FreeFormTrapezoid(Coord, self.layers, Qx, Qz)
            if form is None:
                raise RuntimeError("Failed to calculate form factor")

            # Add contribution from second independent stack family if defined
            if temp_PAR2 is not None and temp_layers2 > 0:
                Coord2 = self.SymCoordAssign_SecondStack(
                    PAR=temp_PAR2,
                    layers=temp_layers2,
                    sld_values=temp_sld_values2,
                    n_stacks=temp_n_stacks_2,
                    stack_spacing=temp_stack_spacing_2,
                    x_offset=temp_x_offset_2,
                )
                if Coord2 is None or (isinstance(Coord2, bool) and not Coord2):
                    raise RuntimeError("Failed to assign coordinates for second stack family")
                form2 = self.FreeFormTrapezoid(Coord2, temp_layers2, Qx, Qz)
                if form2 is None:
                    raise RuntimeError("Failed to calculate form factor for second stack family")
                form = form + form2
            
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
                    param_type = parts[2]  # 'width' or 'height'
                    
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
        import numpy as np
        
        # Plot trapezoid + SRM structure comparison on the same plot
        if plot_structure:
            plt.figure(figsize=(10, 6))
            
            # Plot initial primary trapezoid structure with dashed lines
            self._plot_trapezoid_structure(initial_model_params, 
                                           linestyle='--', 
                                           color='blue', 
                                           alpha=0.7,
                                           label='Initial')
            
            # Plot optimized primary trapezoid structure with solid lines
            self._plot_trapezoid_structure(self.model_params, 
                                           linestyle='-', 
                                           color='red', 
                                           alpha=1.0,
                                           label='Optimized')
            
            # ------------------------------------------------------------------
            # SRM extension: overlay second-stack family (if defined) for both
            # initial and optimized models on the same axes.
            # ------------------------------------------------------------------
            ax = plt.gca()

            def _draw_second_family(mp, linestyle, color, alpha):
                if 'layers_2' not in mp or 'trapezoids_2' not in mp:
                    return
                try:
                    layers2 = int(mp['layers_2'])
                    traps2 = mp['trapezoids_2']
                    if layers2 < 1 or len(traps2) < layers2 + 1:
                        return
                except Exception:
                    return

                x_offset_2 = float(mp.get('x_offset_2', 0.0))

                height2 = 0.0
                for i in range(layers2 + 1):
                    if i > 0:
                        height2 += traps2[i-1]['height']

                    width_i = traps2[i]['width']
                    # Center second family around x_offset_2
                    x_left = x_offset_2 - width_i / 2.0
                    x_right = x_left + width_i

                    # Horizontal segment at this height
                    ax.plot(
                        [x_left, x_right],
                        [height2, height2],
                        linestyle=linestyle,
                        color=color,
                        alpha=alpha,
                    )

                    # Vertical/diagonal sides up to next layer
                    if i < layers2:
                        next_width = traps2[i + 1]['width']
                        x_next_left = x_offset_2 - next_width / 2.0
                        x_next_right = x_next_left + next_width

                        ax.plot(
                            [x_left, x_next_left],
                            [height2, height2 + traps2[i]['height']],
                            linestyle=linestyle,
                            color=color,
                            alpha=alpha,
                        )
                        ax.plot(
                            [x_right, x_next_right],
                            [height2, height2 + traps2[i]['height']],
                            linestyle=linestyle,
                            color=color,
                            alpha=alpha,
                        )

            # Draw second-stack family for initial (blue dashed) and optimized (red solid)
            _draw_second_family(initial_model_params, linestyle='--', color='blue', alpha=0.7)
            _draw_second_family(self.model_params, linestyle='-', color='red', alpha=1.0)
            
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
                w1 = float(trapezoids[i + 1]['width'])

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
        
        height = 0
        for i in range(layers + 1):
            if i > 0:
                height += trapezoids[i-1]['height']
            
            width = trapezoids[i]['width']
            x_left = (trapezoids[0]['width'] - width) / 2
            x_right = x_left + width
            
            ax.plot([x_left, x_right], [height, height], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
            
            if i < layers:
                next_width = trapezoids[i+1]['width']
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
        # geometry / styling
        show_dimensions=False,
        color='blue',
        linewidth=2,
        equal_aspect=True,
        # material shading
        shade_by_sld=False,
        grey_range=(0.85, 0.05),
        shading_alpha=0.6,
        # SLD legend options
        sld_label_map=None,
        show_sld_legend=True,
        sld_legend_precision=3,
        # Pitch / vacuum / periodic visualization
        show_vacuum_region=True,
        n_trapezoid_stacks=None,
        vacuum_edgecolor='tab:blue',
        vacuum_linestyle='-',
        vacuum_linewidth=1.5,
        vacuum_alpha=0.08,
        **kwargs
    ):
        """
        Plots the current trapezoid structure with customizable figure properties.
        Optionally extends the x-axis to the full pitch and visually indicates the
        vacuum/air region that occupies the rest of the period, as in a line–space grating.
        
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
        show_vacuum_region : bool, optional
            If True and a pitch can be determined (from `model_params['Pitch']`,
            `self.Pitch`, or inferred from the Qx sampling), the x‑axis is extended
            to cover the requested number of trapezoid stacks and the vacuum/air
            regions between neighboring stacks are implicitly shown as the gaps
            between them.
        n_trapezoid_stacks : int, optional
            Number of trapezoid stacks (periods) to draw, including the original
            one at x=0. Must be ≥ 1. Default: 2.
        vacuum_edgecolor : str, optional
            Edge color for the outline of the vacuum region. Default: 'tab:blue'.
        vacuum_linestyle : str, optional
            Line style for the vacuum region outline. Default: '-'.
        vacuum_linewidth : float, optional
            Line width for the vacuum region outline. Default: 1.5.
        vacuum_alpha : float, optional
            Alpha for the light shading of the vacuum region. Default: 0.08.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=figsize)
        layers = int(self.model_params.get('layers', 0))
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

        # ------------------------------------------------------------------
        # Optional: show neighboring cell at +Pitch so vacuum is flush
        # ------------------------------------------------------------------
        trapezoids = self.model_params.get('trapezoids', [])
        base_width = float(trapezoids[0]['width']) if trapezoids else None

        # ------------------------------------------------------------------
        # Determine how many stacks to draw and what spacing to use
        # so that the geometry plot matches the SRM multi-stack model.
        # ------------------------------------------------------------------
        # Number of stacks:
        if n_trapezoid_stacks is not None:
            try:
                n_stacks_plot = int(n_trapezoid_stacks)
            except Exception:
                n_stacks_plot = 1
        elif 'n_stacks' in self.model_params:
            try:
                n_stacks_plot = int(self.model_params['n_stacks'])
            except Exception:
                n_stacks_plot = 1
        elif hasattr(self, 'n_trapezoid_stacks'):
            try:
                n_stacks_plot = int(self.n_trapezoid_stacks)
            except Exception:
                n_stacks_plot = 1
        else:
            n_stacks_plot = 1

        if n_stacks_plot < 1:
            n_stacks_plot = 1

        # First try to get an explicit SRM stack spacing
        stack_spacing = None
        if 'stack_spacing' in self.model_params:
            try:
                stack_spacing = float(self.model_params['stack_spacing'])
            except Exception:
                stack_spacing = None
        if stack_spacing is None and hasattr(self, 'stack_spacing'):
            try:
                stack_spacing = float(self.stack_spacing)
            except Exception:
                stack_spacing = None

        # If no explicit SRM spacing, fall back to pitch (periodicity)
        pitch = None
        if 'Pitch' in self.model_params and self.model_params['Pitch'] is not None:
            try:
                pitch = float(self.model_params['Pitch'])
            except Exception:
                pitch = None
        if pitch is None and hasattr(self, 'Pitch') and self.Pitch is not None:
            try:
                pitch = float(self.Pitch)
            except Exception:
                pitch = None
        # 3) As a last resort, estimate pitch from Qx sampling (ΔQx ≈ 2π / pitch)
        if (
            pitch is None
            and hasattr(self, 'Qx')
            and self.Qx is not None
            and isinstance(self.Qx, np.ndarray)
            and self.Qx.size > 1
        ):
            try:
                qx_line = self.Qx[0, :]
                qx_line = qx_line[np.isfinite(qx_line)]
                # Use non‑zero values and unique sampling points
                qx_line = qx_line[np.abs(qx_line) > 1e-6]
                qx_unique = np.unique(np.round(qx_line, decimals=6))
                if qx_unique.size >= 2:
                    dq = np.median(np.diff(np.sort(qx_unique)))
                    if dq > 0:
                        pitch = float(2 * np.pi / dq)
                        # Cache for later use
                        self.Pitch = pitch
                        self.model_params['Pitch'] = pitch
            except Exception:
                pitch = None

        # If SRM spacing is not set, use pitch as spacing for visualization
        if stack_spacing is None:
            stack_spacing = pitch

        # Draw additional copies of the *primary* structure to show the
        # periodic array of trapezoid stacks.
        if (
            show_vacuum_region
            and stack_spacing is not None
            and not np.isclose(stack_spacing, 0.0)
            and base_width is not None
        ):
            # Ensure valid, at least one extra stack
            n_stacks = int(max(1, n_stacks_plot))

            rightmost_x = base_width

            for stack_idx in range(1, n_stacks):
                offset = stack_idx * stack_spacing
                rightmost_x = max(rightmost_x, offset + base_width)

            height = 0.0
            # Precompute SLD-based greys if needed (same mapping as primary)
            if shade_by_sld and layers > 0 and len(trapezoids) >= layers + 1:
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
            else:
                slds_all = None
                denom = None
                sld_min = sld_max = 0.0

            # Draw each additional stack
            for stack_idx in range(1, n_stacks):
                offset = stack_idx * stack_spacing

                # Optional shading for this stack
                if slds_all is not None:
                    base_w_n = base_width
                    height0_n = 0.0
                    for i in range(int(layers)):
                        h = float(trapezoids[i]['height'])
                        if h <= 0:
                            continue

                        w0 = float(trapezoids[i]['width'])
                        w1 = float(trapezoids[i + 1]['width'])

                        xL0 = (base_w_n - w0) / 2.0 + offset
                        xR0 = xL0 + w0
                        xL1 = (base_w_n - w1) / 2.0 + offset
                        xR1 = xL1 + w1

                        y0 = height0_n
                        y1 = height0_n + h

                        sld_val = float(slds_all[i])
                        if denom is None:
                            t = 0.5
                        else:
                            t = (sld_val - sld_min) / denom
                        t = float(np.clip(t, 0.0, 1.0))

                        grey = grey_range[0] + t * (grey_range[1] - grey_range[0])
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

                        height0_n = y1

                # Outline of neighboring stack
                height = 0.0
                ax.plot(
                    [offset, offset + base_width],
                    [0.0, 0.0],
                    linestyle='-',
                    color=color,
                    linewidth=linewidth,
                    alpha=1.0,
                )

                for i in range(int(layers) + 1):
                    if i > 0:
                        height += trapezoids[i - 1]['height']

                    width_i = trapezoids[i]['width']
                    x_left = (base_width - width_i) / 2.0 + offset
                    x_right = x_left + width_i

                    # Horizontal segment at this height
                    ax.plot(
                        [x_left, x_right],
                        [height, height],
                        linestyle='-',
                        color=color,
                        linewidth=linewidth,
                        alpha=1.0,
                    )

                    # Vertical/diagonal sides up to next layer
                    if i < layers:
                        next_width = trapezoids[i + 1]['width']
                        x_next_left = (base_width - next_width) / 2.0 + offset
                        x_next_right = x_next_left + next_width

                        ax.plot(
                            [x_left, x_next_left],
                            [height, height + trapezoids[i]['height']],
                            linestyle='-',
                            color=color,
                            linewidth=linewidth,
                            alpha=1.0,
                        )
                        ax.plot(
                            [x_right, x_next_right],
                            [height, height + trapezoids[i]['height']],
                            linestyle='-',
                            color=color,
                            linewidth=linewidth,
                            alpha=1.0,
                        )

            # If user did not explicitly request x-limits, extend to include all stacks
            if xlim is None:
                plt.xlim(0.0, rightmost_x)

        # ------------------------------------------------------------------
        # Optional: draw the second, independent stack family used in SRM.
        # This uses PAR2/layers2 (or model_params entries if you add them)
        # along with its own stack count, spacing, and x‑offset.
        # ------------------------------------------------------------------
        has_second = hasattr(self, 'PAR2') and hasattr(self, 'layers2')
        if has_second:
            try:
                PAR2 = self.PAR2
                layers2 = int(self.layers2)
            except Exception:
                PAR2 = None
                layers2 = 0

            if isinstance(PAR2, np.ndarray) and layers2 > 0 and len(PAR2) >= layers2 + 1:
                # Number of stacks for the second family
                if hasattr(self, 'n_trapezoid_stacks_2'):
                    try:
                        n_stacks_2 = int(self.n_trapezoid_stacks_2)
                    except Exception:
                        n_stacks_2 = 1
                elif hasattr(self, 'model_params') and 'n_stacks_2' in self.model_params:
                    try:
                        n_stacks_2 = int(self.model_params['n_stacks_2'])
                    except Exception:
                        n_stacks_2 = 1
                else:
                    n_stacks_2 = 1

                if n_stacks_2 < 1:
                    n_stacks_2 = 1

                # Spacing within the second family
                if hasattr(self, 'stack_spacing_2'):
                    try:
                        stack_spacing_2 = float(self.stack_spacing_2)
                    except Exception:
                        stack_spacing_2 = None
                elif hasattr(self, 'model_params') and 'stack_spacing_2' in self.model_params:
                    try:
                        stack_spacing_2 = float(self.model_params['stack_spacing_2'])
                    except Exception:
                        stack_spacing_2 = None
                else:
                    # Fallback: reuse primary spacing if available
                    stack_spacing_2 = stack_spacing

                # Overall horizontal offset of the second family
                if hasattr(self, 'x_offset_2'):
                    try:
                        x_offset_2 = float(self.x_offset_2)
                    except Exception:
                        x_offset_2 = 0.0
                elif hasattr(self, 'model_params') and 'x_offset_2' in self.model_params:
                    try:
                        x_offset_2 = float(self.model_params['x_offset_2'])
                    except Exception:
                        x_offset_2 = 0.0
                else:
                    x_offset_2 = 0.0

                # Build base second-stack geometry (same rules as SymCoordAssign_SecondStack)
                base_coord2 = np.zeros([layers2 + 1, 5], dtype=float)
                for layer_idx in range(layers2):
                    T2 = layer_idx
                    if T2 == 0:
                        base_coord2[T2, 0] = 0.0
                        base_coord2[T2, 1] = PAR2[0, 0]
                        base_coord2[T2, 2] = PAR2[0, 1]
                        base_coord2[T2, 3] = 0.0
                    else:
                        base_coord2[T2, 0] = (
                            base_coord2[T2 - 1, 0]
                            + 0.5 * (PAR2[T2 - 1, 0] - PAR2[T2, 0])
                        )
                        base_coord2[T2, 1] = base_coord2[T2, 0] + PAR2[T2, 0]
                        base_coord2[T2, 2] = PAR2[T2, 1]
                        base_coord2[T2, 3] = 0.0
                T2 = layers2
                base_coord2[T2, 0] = (
                    base_coord2[T2 - 1, 0]
                    + 0.5 * (PAR2[T2 - 1, 0] - PAR2[T2, 0])
                )
                base_coord2[T2, 1] = base_coord2[T2, 0] + PAR2[T2, 0]
                base_coord2[T2, 2] = PAR2[T2, 1]
                base_coord2[T2, 3] = 0.0

                # Apply global offset to this family
                base_coord2[:, 0] += x_offset_2
                base_coord2[:, 1] += x_offset_2

                # Width of the base of the second family
                base_width_2 = base_coord2[0, 1] - base_coord2[0, 0]

                # Draw each stack in the second family
                max_x_second = base_coord2[:, 1].max()
                for stack_idx in range(n_stacks_2):
                    dx2 = stack_idx * (0.0 if stack_spacing_2 is None else stack_spacing_2)
                    # base line
                    ax.plot(
                        [base_coord2[0, 0] + dx2, base_coord2[0, 1] + dx2],
                        [0.0, 0.0],
                        linestyle='-',
                        color=color,
                        linewidth=linewidth,
                        alpha=1.0,
                    )

                    height2 = 0.0
                    for i2 in range(layers2 + 1):
                        if i2 > 0:
                            height2 += PAR2[i2 - 1, 1]

                        # Horizontal segment at this height
                        x_left2 = base_coord2[i2, 0] + dx2
                        x_right2 = base_coord2[i2, 1] + dx2
                        ax.plot(
                            [x_left2, x_right2],
                            [height2, height2],
                            linestyle='-',
                            color=color,
                            linewidth=linewidth,
                            alpha=1.0,
                        )

                        # Upward sides to next layer
                        if i2 < layers2:
                            next_left2 = base_coord2[i2 + 1, 0] + dx2
                            next_right2 = base_coord2[i2 + 1, 1] + dx2
                            ax.plot(
                                [x_left2, next_left2],
                                [height2, height2 + PAR2[i2, 1]],
                                linestyle='-',
                                color=color,
                                linewidth=linewidth,
                                alpha=1.0,
                            )
                            ax.plot(
                                [x_right2, next_right2],
                                [height2, height2 + PAR2[i2, 1]],
                                linestyle='-',
                                color=color,
                                linewidth=linewidth,
                                alpha=1.0,
                            )

                        max_x_second = max(max_x_second, x_right2)

                # Ensure x-limits include the second family if user did not fix them
                if xlim is None and np.isfinite(max_x_second):
                    cur_xlim = plt.xlim()
                    new_max = max(cur_xlim[1], max_x_second)
                    plt.xlim(min(cur_xlim[0], 0.0), new_max)
        
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
        
        print(f"\n1-Layer Width+DW Sweep Summary:")
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
TrapezoidModel = TrapezoidModelArray

