import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import differential_evolution
from tqdm import tqdm
import seaborn as sns

from CDSAXS_base_model import CDSAXS_Model
class TrapezoidModel(CDSAXS_Model):
    """
    CDSAXS model for trapezoid structures.
    """
    
    def __init__(self, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Initialize the trapezoid model.
        
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
        Bk : float, optional
            Background intensity
        Pitch : float, optional
            Pitch parameter
        model_params : dict, optional
            Dictionary-based parameters
        """
        super().__init__('trapezoid', model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    
    def build_model_params_from_traditional(self):
        """
        Build model_params dictionary from traditional parameters.
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
            'Bk': self.Bk
        }
        
        # Add optional parameters if they exist
        if hasattr(self, 'SLD') and self.SLD is not None:
            self.model_params['SLD'] = self.SLD
            
        if hasattr(self, 'Pitch') and self.Pitch is not None:
            self.model_params['Pitch'] = self.Pitch
            
        return self.model_params
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary.
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
        self.Bk = self.model_params['Bk']
        
        # Update optional parameters
        if 'SLD' in self.model_params:
            self.SLD = self.model_params['SLD']
            
        if 'Pitch' in self.model_params:
            self.Pitch = self.model_params['Pitch']
            
        # Update SimPar
        self.SimPar = np.append(self.PAR.ravel(), [self.I0, self.DW, self.Bk])
        
        return True
    
    def process_imported_data(self):
        """
        Process imported data for trapezoid model.
        """
        try:
            # For trapezoid model, run initial simulation
            self.SymCoordAssign_SingleMaterial()
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
        
        Parameters:
        -----------
        param_limits : dict, optional
            Dictionary of parameters to optimize with their limits
            If None, creates default limits of ±10% for all parameters
            
        Returns:
        --------
        dict
            Dictionary of optimization parameters with limits
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
            
            param_limits['Bk'] = {
                'min': self.Bk * 0.9,
                'max': self.Bk * 1.1,
                'default': self.Bk
            }
        else:
            # Ensure default values are set if not provided
            for param, limits in param_limits.items():
                if 'default' not in limits:
                    if param.startswith('trap_'):
                        parts = param.split('_')
                        trap_idx = int(parts[1])
                        param_type = parts[2]
                        limits['default'] = self.model_params['trapezoids'][trap_idx][param_type]
                    else:
                        limits['default'] = getattr(self, param)
        
        # Store optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits
    
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
            for i in range(int(layers)):
                H2 = H2 + Coord[i, 2, 0]
                if i > 0:
                    H1 = H1 + Coord[i-1, 2, 0]
                    
                x1 = Coord[i, 0, 0]
                x4 = Coord[i, 1, 0]
                x2 = Coord[i+1, 0, 0]
                x3 = Coord[i+1, 1, 0]
                
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
        Simulates the intensity for a single material trapezoid structure.
        
        Parameters:
        -----------
        PAR : numpy.ndarray, optional
            Array with parameters for each layer
            If None, uses self.PAR
        layers : int, optional
            Number of layers in the trapezoid structure
            If None, uses self.layers
        Qx : numpy.ndarray, optional
            X-component of scattering vector, 2D array
            If None, uses self.Qx
        Qz : numpy.ndarray, optional
            Z-component of scattering vector, 2D array
            If None, uses self.Qz
        DW : float, optional
            Debye-Waller factor
            If None, uses self.DW
        I0 : float, optional
            Intensity scaling factor
            If None, uses self.I0
        Bk : float, optional
            Background intensity
            If None, uses self.Bk
            
        Returns:
        --------
        numpy.ndarray
            The simulated intensity
            If called with self attributes, also sets self.SimInt
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
                        # If PAR is not available, we'll try to use existing Coord
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
            
            # Generate coordinates if PAR is provided
            if PAR is not None:
                Coord = self.SymCoordAssign_SingleMaterial(PAR, layers)
                if Coord is None or (using_self and Coord is False):
                    raise RuntimeError("Failed to assign coordinates in SymCoordAssign_SingleMaterial")
            else:
                # Use existing Coord
                if not hasattr(self, 'Coord'):
                    raise AttributeError("Missing required attribute: Coord")
                Coord = self.Coord
            
            # Calculate form factor
            form = self.FreeFormTrapezoid(Coord, layers, Qx, Qz)
            if form is None:
                raise RuntimeError("Failed to calculate form factor in FreeFormTrapezoid")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qx, 2) + np.power(Qz, 2)) * np.power(DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = form * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity
            SimInt = np.power(Formfactor, 2) * I0 + Bk
            
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
        Simulates a trapezoid structure and calculates goodness of fit (GF).
        Used by the differential evolution algorithm.
        
        Parameters:
        -----------
        optimization_values : numpy.ndarray
            1D array containing values for the parameters being optimized
        param_names : list, optional
            List of parameter names corresponding to optimization_values
            If None, uses self.param_names
        Intensity : numpy.ndarray, optional
            Measured intensity data for comparison
            If None, uses self.Intensity
        Qx : numpy.ndarray, optional
            X-component of scattering vector, 2D array
            If None, uses self.Qx
        Qz : numpy.ndarray, optional
            Z-component of scattering vector, 2D array
            If None, uses self.Qz
            
        Returns:
        --------
        float
            Chi-square value representing goodness of fit
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
                else:
                    # Global parameter (DW, I0, Bk)
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
            temp_Bk = params['Bk']
            
            # Simulate intensity
            SimInt = self.SimTrap_SM(temp_PAR, self.layers, Qx, Qz, temp_DW, temp_I0, temp_Bk)
            if SimInt is None:
                raise RuntimeError("Failed to simulate intensity in SimTrap_SM")
            
            # Calculate goodness of fit
            Chi2 = self.GF_calc(SimInt, Intensity)
            
            return Chi2
            
        except Exception as e:
            print(f"Error in SimTrap_GF: {str(e)}")
            return float('inf')  # Return infinity as worst-case fit
    
    def CDSAXS_DiffEvolution(self, params_to_optimize=None, plot_results=True, 
                            plot_structure=True, plot_grid=True, plot_combined=True, **kwargs):
        """
        Performs differential evolution optimization for CDSAXS trapezoid model fitting
        and shows before/after comparison plots.
        
        Parameters:
        -----------
        params_to_optimize : dict, optional
            Dictionary containing parameters to optimize with their bounds
            If None, uses self.model_params['optimization']
        plot_results : bool, optional
            Whether to generate any plots (master switch for all plotting)
        plot_structure : bool, optional
            Whether to plot trapezoid structure comparison
        plot_grid : bool, optional
            Whether to plot the grid of individual Qz cuts
        plot_combined : bool, optional
            Whether to plot the combined view with all cuts
        **kwargs : dict
            Additional keyword arguments to pass to scipy's differential_evolution function
            
        Returns:
        --------
        dict
            Optimized parameter dictionary with the same structure as the input model_params
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
            
            # Create parameter names list and bounds list
            param_names = []
            bounds = []
            initial_values = []
            
            for param_name, param_config in params_to_optimize.items():
                param_names.append(param_name)
                bounds.append((param_config['min'], param_config['max']))
                initial_values.append(param_config['default'])
            
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
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    # Parse trapezoid parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]  # 'width' or 'height'
                    
                    optimized_params['trapezoids'][trap_idx][param_type] = result.x[i]
                else:
                    # Global parameter (DW, I0, Bk)
                    optimized_params[param_name] = result.x[i]
            
            # Update class attributes with optimized values
            self.model_params = optimized_params
            self.update_traditional_from_model_params()
            
            # Simulate with optimized parameters
            self.SimInt = self.SimTrap_SM()
            
            # Calculate goodness of fit and BIC
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            # Print optimization results
            print(f"Optimization complete after {result.nfev} function evaluations")
            print(f"Initial goodness of fit: {self.GF_Initial:.4f}")
            print(f"Final goodness of fit: {self.GF:.4f}")
            print(f"Improvement: {self.GF_Initial - self.GF:.4f} ({(1 - self.GF/self.GF_Initial)*100:.2f}%)")
            
            # Generate before/after comparison plots if requested
            if plot_results:
                self._plot_optimization_results(initial_model_params, initial_simInt,
                                              plot_structure, plot_grid, plot_combined)
            
            # Print parameter changes
            self._print_parameter_changes(initial_model_params)
            
            return self.model_params
                
        except Exception as e:
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
    
    def _plot_trapezoid_structure(self, model_params, linestyle='-', color='black', alpha=1.0, label=None):
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
        """
        trapezoids = model_params['trapezoids']
        layers = model_params['layers']
        
        # Plot base
        plt.plot([0, trapezoids[0]['width']], [0, 0], 
                linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
        
        height = 0
        for i in range(layers + 1):
            if i > 0:
                height += trapezoids[i-1]['height']
            
            width = trapezoids[i]['width']
            x_left = (trapezoids[0]['width'] - width) / 2
            x_right = x_left + width
            
            plt.plot([x_left, x_right], [height, height], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
            
            if i < layers:
                next_width = trapezoids[i+1]['width']
                x_next_left = (trapezoids[0]['width'] - next_width) / 2
                x_next_right = x_next_left + next_width
                
                plt.plot([x_left, x_next_left], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
                plt.plot([x_right, x_next_right], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
        
        # Add a line to the legend
        if label:
            plt.plot([], [], linestyle=linestyle, color=color, alpha=alpha, linewidth=2, label=label)
        
        plt.axis('equal')
        plt.xlabel('Width (nm)')
        plt.ylabel('Height (nm)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        return plt.gca()
    
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
            ax.semilogy(qz_values, self.Intensity[:, i], 'ko', alpha=0.7, 
                      markersize=4, label='Measured')
            
            # Plot initial simulation
            ax.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.8, 
                     linewidth=1.5, label='Initial')
            
            # Plot optimized simulation
            ax.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=1.0, 
                     linewidth=1.5, label='Optimized')
            
            # Set labels and title
            ax.set_title(f'Cut at Qx = {qx_value:.4f}')
            ax.set_xlabel('Qz (nm$^{-1}$)')
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
            plt.semilogy(qz_values, self.Intensity[:, i], 'ko', alpha=0.5, markersize=4)
            
            # Plot initial simulation
            plt.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.5, linewidth=1.5)
            
            # Plot optimized simulation
            plt.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=0.6, linewidth=1.5)
        
        plt.title('Intensity Comparison - All Cuts')
        plt.xlabel('Qz (nm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
    
    def _print_parameter_changes(self, initial_model_params):
        """
        Print a table of parameter changes from optimization.
        
        Parameters:
        -----------
        initial_model_params : dict
            Model parameters before optimization
        """
        print("\nParameter Changes:")
        print("=" * 60)
        print(f"{'Parameter':<20} {'Initial':<15} {'Optimized':<15} {'Change %':<10}")
        print("-" * 60)
        
        # Print trapezoid parameters
        initial_traps = initial_model_params['trapezoids']
        optimized_traps = self.model_params['trapezoids']
        max_traps = max(len(initial_traps), len(optimized_traps))
        
        for i in range(max_traps):
            # Handle the case where the trapezoid exists in both models
            if i < len(initial_traps) and i < len(optimized_traps):
                # Print width
                width_init = initial_traps[i]['width']
                width_optim = optimized_traps[i]['width']
                width_change = (width_optim - width_init) / width_init * 100 if width_init != 0 else float('inf')
                print(f"Trap {i} Width{'':<10} {width_init:<15.4f} {width_optim:<15.4f} {width_change:+.2f}%")
                
                # Print height if this isn't the top-most trapezoid (which might not have a height)
                if 'height' in initial_traps[i] and 'height' in optimized_traps[i]:
                    height_init = initial_traps[i]['height']
                    height_optim = optimized_traps[i]['height']
                    height_change = (height_optim - height_init) / height_init * 100 if height_init != 0 else float('inf')
                    print(f"Trap {i} Height{'':<9} {height_init:<15.4f} {height_optim:<15.4f} {height_change:+.2f}%")
            
            # Handle the case where the trapezoid only exists in the initial model
            elif i < len(initial_traps):
                width_init = initial_traps[i]['width']
                print(f"Trap {i} Width{'':<10} {width_init:<15.4f} {'N/A':<15} {'N/A':<10}")
                
                if 'height' in initial_traps[i]:
                    height_init = initial_traps[i]['height']
                    print(f"Trap {i} Height{'':<9} {height_init:<15.4f} {'N/A':<15} {'N/A':<10}")
            
            # Handle the case where the trapezoid only exists in the optimized model
            elif i < len(optimized_traps):
                width_optim = optimized_traps[i]['width']
                print(f"Trap {i} Width{'':<10} {'N/A':<15} {width_optim:<15.4f} {'N/A':<10}")
                
                if 'height' in optimized_traps[i]:
                    height_optim = optimized_traps[i]['height']
                    print(f"Trap {i} Height{'':<9} {'N/A':<15} {height_optim:<15.4f} {'N/A':<10}")
        
        # Print global parameters
        for param in ['DW', 'I0', 'Bk']:
            if param in initial_model_params and param in self.model_params:
                init_val = initial_model_params[param]
                optim_val = self.model_params[param]
                change = (optim_val - init_val) / init_val * 100 if init_val != 0 else float('inf')
                
                print(f"{param:<20} {init_val:<15.6f} {optim_val:<15.6f} {change:+.2f}%")
        
        print("=" * 60)
    
    def plot_structure(self):
        """
        Plots the current trapezoid structure.
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=(10, 6))
        ax = self._plot_trapezoid_structure(self.model_params)
        plt.title('Trapezoid Structure')
        return ax

    def PlotQzCut(self, cut_index=None, SimInt=None, log_scale='yes'):
        """
        Plots intensity vs Qz for specific Qx cut(s)
        
        Parameters:
        -----------
        cut_index : int or list or None, optional
            Index or indices of the Qx cut(s) to plot
            If None, plots all available cuts
        SimInt : numpy.ndarray, optional
            Simulated intensity to plot alongside measured data
            If None, uses self.SimInt if available
        log_scale : str, optional
            Whether to use logarithmic scale for intensity ('yes' or 'no')
        
        Returns:
        --------
        matplotlib.axes.Axes or list of Axes
            The axes object(s) containing the plot(s)
        """
        # Check if required attributes exist
        if not hasattr(self, 'Qz') or not hasattr(self, 'Intensity'):
            raise AttributeError("Missing required attributes: Qz and/or Intensity")
        
        # Determine which cuts to plot
        if cut_index is None:
            # Plot all cuts
            cut_indices = list(range(self.Intensity.shape[1]))
        elif isinstance(cut_index, (list, tuple, np.ndarray)):
            # Plot multiple specified cuts
            cut_indices = cut_index
        else:
            # Plot a single cut
            cut_indices = [cut_index]
        
        # Create a figure with appropriate size
        n_cuts = len(cut_indices)
        if n_cuts == 1:
            # Single plot
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
        else:
            # Multiple plots
            fig_width = min(16, n_cuts * 5)  # Limit maximum width
            fig_height = min(10, n_cuts * 3)  # Limit maximum height
            
            if n_cuts <= 4:
                # Use a single row for 2-4 plots
                n_rows = 1
                n_cols = n_cuts
            else:
                # Create a grid for many plots
                n_rows = int(np.ceil(np.sqrt(n_cuts)))
                n_cols = int(np.ceil(n_cuts / n_rows))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            if n_rows * n_cols > 1:
                axes = axes.flatten()
        
        # Plot each cut
        for i, (ax, idx) in enumerate(zip(axes, cut_indices)):
            # Check if the index is valid
            if idx < 0 or idx >= self.Intensity.shape[1]:
                ax.text(0.5, 0.5, f"Invalid cut index: {idx}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get Qz values for the selected cut
            qz_values = self.Qz[:, idx]
            qx_value = self.Qx[0, idx]
            
            # Plot measured intensity
            measured_line, = ax.plot(qz_values, self.Intensity[:, idx], 'bo-', label='Measured')
            
            # Plot simulated intensity if available
            if SimInt is not None:
                simulated_line, = ax.plot(qz_values, SimInt[:, idx], 'r-', label='Simulated')
            elif hasattr(self, 'SimInt') and self.SimInt is not None:
                simulated_line, = ax.plot(qz_values, self.SimInt[:, idx], 'r-', label='Simulated')
            
            # Set logarithmic scale if requested
            if log_scale.lower() == 'yes':
                ax.set_yscale('log')
            
            # Set labels and title
            ax.set_title(f'Cut at Qx = {qx_value:.4f}')
            ax.set_xlabel('Qz (nm$^{-1}$)')
            ax.set_ylabel('Intensity (counts)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(cut_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Return a single axis for a single plot, or list of axes for multiple plots
        return axes[0] if len(axes) == 1 else axes
    
    def parameter_sweep_1d(self, sweep_param, sweep_range, n_points=20, 
                          exclude_from_fit=None, plot_results=True, 
                          figsize=(10, 6), save_results=False, filename=None,
                          optimization_kwargs=None, verbose=True):
        """
        Perform a 1D parameter sweep, holding one parameter constant while optimizing others.
        
        Parameters:
        -----------
        sweep_param : str
            Name of the parameter to sweep (e.g., 'trap_0_width', 'DW', 'I0')
        sweep_range : tuple
            (min_value, max_value) for the sweep parameter
        n_points : int, optional
            Number of points to sweep. Default: 20
        exclude_from_fit : list, optional
            List of parameter names to exclude from optimization (in addition to sweep_param)
        plot_results : bool, optional
            Whether to plot the results. Default: True
        figsize : tuple, optional
            Figure size for the plot. Default: (10, 6)
        save_results : bool, optional
            Whether to save results to file. Default: False
        filename : str, optional
            Filename for saving results. If None, auto-generates name
        optimization_kwargs : dict, optional
            Additional kwargs for CDSAXS_DiffEvolution
        verbose : bool, optional
            Whether to print progress. Default: True
            
        Returns:
        --------
        dict
            Dictionary with sweep values, GF values, BIC values, and optimized parameters
        """
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {'maxiter': 30, 'popsize': 10, 'plot_results': False}
        
        # Create sweep values
        sweep_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
        
        # Initialize results storage
        results = {
            'sweep_param': sweep_param,
            'sweep_values': sweep_values,
            'gf_values': [],
            'bic_values': [],
            'optimized_params': [],
            'convergence_flags': []
        }
        
        # Store original parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Setup progress bar
        if verbose:
            pbar = tqdm(sweep_values, desc=f"Sweeping {sweep_param}")
        else:
            pbar = sweep_values
        
        for i, value in enumerate(pbar):
            try:
                # Reset to original parameters
                self.model_params = copy.deepcopy(original_params)
                self.update_traditional_from_model_params()
                
                # Set the sweep parameter value
                self._set_parameter_value(sweep_param, value)
                
                # Create optimization parameters excluding the sweep parameter
                opt_params = self._create_optimization_params_excluding(
                    [sweep_param] + (exclude_from_fit or [])
                )
                
                if not opt_params:
                    # No parameters to optimize, just calculate GF
                    self.SimInt = self.SimTrap_SM()
                    gf = self.GF_calc(self.SimInt)
                    bic = self.BIC_calc(gf)
                    converged = True
                    opt_result = {}
                else:
                    # Run optimization
                    opt_result = self.CDSAXS_DiffEvolution(
                        params_to_optimize=opt_params,
                        **optimization_kwargs
                    )
                    gf = self.GF
                    bic = self.BIC
                    converged = opt_result is not None
                
                # Store results
                results['gf_values'].append(gf)
                results['bic_values'].append(bic)
                results['optimized_params'].append(copy.deepcopy(self.model_params))
                results['convergence_flags'].append(converged)
                
                if verbose:
                    pbar.set_postfix({'GF': f'{gf:.4f}', 'BIC': f'{bic:.4f}'})
                    
            except Exception as e:
                if verbose:
                    print(f"Error at {sweep_param}={value}: {str(e)}")
                results['gf_values'].append(float('inf'))
                results['bic_values'].append(float('inf'))
                results['optimized_params'].append(None)
                results['convergence_flags'].append(False)
        
        # Restore original parameters
        self.model_params = original_params
        self.update_traditional_from_model_params()
        
        # Plot results
        if plot_results:
            self._plot_1d_sweep_results(results, figsize)
        
        # Save results
        if save_results:
            self._save_sweep_results(results, filename or f"{sweep_param}_sweep_1d")
        
        return results
    
    def parameter_sweep_2d(self, sweep_params, sweep_ranges, n_points=(10, 10),
                          exclude_from_fit=None, plot_results=True, 
                          figsize=(10, 8), save_results=False, filename=None,
                          optimization_kwargs=None, verbose=True, metric='GF'):
        """
        Perform a 2D parameter sweep with heatmap visualization.
        
        Parameters:
        -----------
        sweep_params : tuple
            (param1_name, param2_name) to sweep
        sweep_ranges : tuple
            ((min1, max1), (min2, max2)) for the sweep parameters
        n_points : tuple, optional
            (n_points1, n_points2) for each parameter. Default: (10, 10)
        exclude_from_fit : list, optional
            List of parameter names to exclude from optimization
        plot_results : bool, optional
            Whether to plot the heatmap. Default: True
        figsize : tuple, optional
            Figure size for the plot. Default: (10, 8)
        save_results : bool, optional
            Whether to save results to file. Default: False
        filename : str, optional
            Filename for saving results
        optimization_kwargs : dict, optional
            Additional kwargs for CDSAXS_DiffEvolution
        verbose : bool, optional
            Whether to print progress. Default: True
        metric : str, optional
            Metric to plot ('GF' or 'BIC'). Default: 'GF'
            
        Returns:
        --------
        dict
            Dictionary with sweep values, GF/BIC matrices, and optimized parameters
        """
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {'maxiter': 20, 'popsize': 8, 'plot_results': False}
        
        # Create sweep values
        param1_values = np.linspace(sweep_ranges[0][0], sweep_ranges[0][1], n_points[0])
        param2_values = np.linspace(sweep_ranges[1][0], sweep_ranges[1][1], n_points[1])
        
        # Initialize results storage
        results = {
            'sweep_params': sweep_params,
            'param1_values': param1_values,
            'param2_values': param2_values,
            'gf_matrix': np.full((n_points[1], n_points[0]), np.inf),
            'bic_matrix': np.full((n_points[1], n_points[0]), np.inf),
            'optimized_params': [[None for _ in range(n_points[0])] for _ in range(n_points[1])],
            'convergence_matrix': np.full((n_points[1], n_points[0]), False, dtype=bool)
        }
        
        # Store original parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Setup progress bar
        total_points = n_points[0] * n_points[1]
        if verbose:
            pbar = tqdm(total=total_points, desc=f"2D Sweep: {sweep_params[0]} vs {sweep_params[1]}")
        
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                try:
                    # Reset to original parameters
                    self.model_params = copy.deepcopy(original_params)
                    self.update_traditional_from_model_params()
                    
                    # Set the sweep parameter values
                    self._set_parameter_value(sweep_params[0], val1)
                    self._set_parameter_value(sweep_params[1], val2)
                    
                    # Create optimization parameters excluding the sweep parameters
                    opt_params = self._create_optimization_params_excluding(
                        list(sweep_params) + (exclude_from_fit or [])
                    )
                    
                    if not opt_params:
                        # No parameters to optimize, just calculate GF
                        self.SimInt = self.SimTrap_SM()
                        gf = self.GF_calc(self.SimInt)
                        bic = self.BIC_calc(gf)
                        converged = True
                    else:
                        # Run optimization
                        opt_result = self.CDSAXS_DiffEvolution(
                            params_to_optimize=opt_params,
                            **optimization_kwargs
                        )
                        gf = self.GF
                        bic = self.BIC
                        converged = opt_result is not None
                    
                    # Store results
                    results['gf_matrix'][j, i] = gf
                    results['bic_matrix'][j, i] = bic
                    results['optimized_params'][j][i] = copy.deepcopy(self.model_params)
                    results['convergence_matrix'][j, i] = converged
                    
                    if verbose:
                        pbar.set_postfix({
                            f'{sweep_params[0]}': f'{val1:.3f}',
                            f'{sweep_params[1]}': f'{val2:.3f}',
                            'GF': f'{gf:.4f}'
                        })
                        pbar.update(1)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error at {sweep_params[0]}={val1}, {sweep_params[1]}={val2}: {str(e)}")
                        pbar.update(1)
        
        if verbose:
            pbar.close()
        
        # Restore original parameters
        self.model_params = original_params
        self.update_traditional_from_model_params()
        
        # Plot results
        if plot_results:
            self._plot_2d_sweep_results(results, figsize, metric)
        
        # Save results
        if save_results:
            self._save_sweep_results(results, filename or f"{sweep_params[0]}_{sweep_params[1]}_sweep_2d")
        
        return results
    
    def _set_parameter_value(self, param_name, value):
        """Set a parameter value in the model."""
        if param_name.startswith('trap_'):
            # Trapezoid parameter
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = parts[2]
            self.model_params['trapezoids'][trap_idx][param_type] = value
        elif param_name.startswith('Bk_'):
            # Background parameter for specific column
            bk_idx = int(param_name.split('_')[1])
            if isinstance(self.model_params['Bk'], list):
                self.model_params['Bk'][bk_idx] = value
            else:
                # Convert to list if needed
                n_cols = len(self.Bk) if isinstance(self.Bk, np.ndarray) else 1
                self.model_params['Bk'] = [self.model_params['Bk']] * n_cols
                self.model_params['Bk'][bk_idx] = value
        else:
            # Global parameter
            self.model_params[param_name] = value
        
        # Update traditional parameters
        self.update_traditional_from_model_params()
    
    def _create_optimization_params_excluding(self, excluded_params):
        """Create optimization parameters excluding specified parameters."""
        if not hasattr(self, 'model_params') or 'optimization' not in self.model_params:
            self.initialize_optimization_params()
        
        opt_params = {}
        for param_name, param_config in self.model_params['optimization'].items():
            if param_name not in excluded_params:
                opt_params[param_name] = param_config
        
        return opt_params
    
    def _plot_1d_sweep_results(self, results, figsize):
        """Plot 1D sweep results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot GF vs parameter
        ax1.plot(results['sweep_values'], results['gf_values'], 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel(results['sweep_param'])
        ax1.set_ylabel('Goodness of Fit (GF)')
        ax1.set_title(f'GF vs {results["sweep_param"]}')
        ax1.grid(True, alpha=0.3)
        
        # Find and mark minimum
        min_idx = np.argmin(results['gf_values'])
        min_gf = results['gf_values'][min_idx]
        min_param = results['sweep_values'][min_idx]
        ax1.plot(min_param, min_gf, 'ro', markersize=10, label=f'Min GF: {min_gf:.4f}')
        ax1.legend()
        
        # Plot BIC vs parameter
        ax2.plot(results['sweep_values'], results['bic_values'], 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel(results['sweep_param'])
        ax2.set_ylabel('Bayesian Information Criterion (BIC)')
        ax2.set_title(f'BIC vs {results["sweep_param"]}')
        ax2.grid(True, alpha=0.3)
        
        # Find and mark minimum BIC
        min_bic_idx = np.argmin(results['bic_values'])
        min_bic = results['bic_values'][min_bic_idx]
        min_bic_param = results['sweep_values'][min_bic_idx]
        ax2.plot(min_bic_param, min_bic, 'ro', markersize=10, label=f'Min BIC: {min_bic:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n1D Parameter Sweep Summary:")
        print(f"Parameter: {results['sweep_param']}")
        print(f"Range: {results['sweep_values'][0]:.4f} to {results['sweep_values'][-1]:.4f}")
        print(f"Best GF: {min_gf:.4f} at {results['sweep_param']} = {min_param:.4f}")
        print(f"Best BIC: {min_bic:.4f} at {results['sweep_param']} = {min_bic_param:.4f}")
    
    def _plot_2d_sweep_results(self, results, figsize, metric='GF'):
        """Plot 2D sweep results as heatmap."""
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
        # Replace inf values for better visualization
        plot_data = np.copy(data_matrix)
        plot_data[np.isinf(plot_data)] = np.nan
        
        # Use log scale if the range is large
        if np.nanmax(plot_data) / np.nanmin(plot_data) > 100:
            norm = LogNorm(vmin=np.nanmin(plot_data), vmax=np.nanmax(plot_data))
        else:
            norm = None
        
        im = ax.imshow(plot_data, cmap=cmap, aspect='auto', origin='lower', norm=norm)
        
        # Set axis labels and ticks
        param1_name, param2_name = results['sweep_params']
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title(f'{title} Heatmap: {param1_name} vs {param2_name}')
        
        # Set tick labels
        n_ticks = 5
        x_tick_indices = np.linspace(0, len(results['param1_values'])-1, n_ticks, dtype=int)
        y_tick_indices = np.linspace(0, len(results['param2_values'])-1, n_ticks, dtype=int)
        
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels([f'{results["param1_values"][i]:.3f}' for i in x_tick_indices])
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels([f'{results["param2_values"][i]:.3f}' for i in y_tick_indices])
        
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
        min_param1 = results['param1_values'][min_idx[1]]
        min_param2 = results['param2_values'][min_idx[0]]
        
        print(f"\n2D Parameter Sweep Summary:")
        print(f"Parameters: {param1_name} vs {param2_name}")
        print(f"Grid size: {len(results['param1_values'])} x {len(results['param2_values'])}")
        print(f"Best {metric}: {min_val:.4f}")
        print(f"  at {param1_name} = {min_param1:.4f}, {param2_name} = {min_param2:.4f}")
    
    def _save_sweep_results(self, results, filename):
        """Save sweep results to file."""
        import pickle
        
        # Add timestamp to filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.pkl"
        
        with open(full_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to: {full_filename}")
    
    def load_sweep_results(self, filename):
        """Load sweep results from file."""
        import pickle
        
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def compare_sweep_results(self, results_list, labels=None, figsize=(12, 8)):
        """
        Compare multiple 1D sweep results on the same plot.
        
        Parameters:
        -----------
        results_list : list
            List of results dictionaries from parameter_sweep_1d
        labels : list, optional
            Labels for each result set
        figsize : tuple, optional
            Figure size
        """
        if labels is None:
            labels = [f"Sweep {i+1}" for i in range(len(results_list))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
        
        for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
            # Plot GF
            ax1.plot(results['sweep_values'], results['gf_values'], 
                    'o-', color=color, label=label, linewidth=2, markersize=4)
            
            # Plot BIC
            ax2.plot(results['sweep_values'], results['bic_values'], 
                    'o-', color=color, label=label, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Goodness of Fit (GF)')
        ax1.set_title('GF Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_ylabel('BIC')
        ax2.set_title('BIC Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        
    def parameter_sweep_width_dw_1layer(self, width_range, dw_range, n_points=(10, 10),
                                    exclude_from_fit=None, plot_results=True, 
                                    figsize=(10, 8), save_results=False, filename=None,
                                    optimization_kwargs=None, verbose=True, metric='GF'):
        """
        Special 2D parameter sweep for 1-layer models: sweep both widths simultaneously 
        (width_0 = width_1) along with Debye-Waller factor.
        
        This function is only available for single-layer (layers=1) trapezoid models.
        It sweeps the common width value for both the bottom and top trapezoid widths
        while also sweeping the Debye-Waller factor.
        
        Parameters:
        -----------
        width_range : tuple
            (min_width, max_width) for both trapezoid widths
        dw_range : tuple
            (min_dw, max_dw) for the Debye-Waller factor
        n_points : tuple, optional
            (n_width_points, n_dw_points) for each parameter. Default: (10, 10)
        exclude_from_fit : list, optional
            List of parameter names to exclude from optimization (beyond width and DW)
        plot_results : bool, optional
            Whether to plot the heatmap. Default: True
        figsize : tuple, optional
            Figure size for the plot. Default: (10, 8)
        save_results : bool, optional
            Whether to save results to file. Default: False
        filename : str, optional
            Filename for saving results
        optimization_kwargs : dict, optional
            Additional kwargs for CDSAXS_DiffEvolution
        verbose : bool, optional
            Whether to print progress. Default: True
        metric : str, optional
            Metric to plot ('GF' or 'BIC'). Default: 'GF'
            
        Returns:
        --------
        dict
            Dictionary with sweep values, GF/BIC matrices, and optimized parameters
            
        Raises:
        -------
        ValueError
            If the model doesn't have exactly 1 layer
        """
        # Check if this is a 1-layer model
        if self.layers != 1:
            raise ValueError(f"This function is only for 1-layer models. Current model has {self.layers} layers.")
        
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {'maxiter': 20, 'popsize': 8, 'plot_results': False}
        
        # Create sweep values
        width_values = np.linspace(width_range[0], width_range[1], n_points[0])
        dw_values = np.linspace(dw_range[0], dw_range[1], n_points[1])
        
        # Initialize results storage
        results = {
            'sweep_type': 'width_dw_1layer',
            'width_values': width_values,
            'dw_values': dw_values,
            'gf_matrix': np.full((n_points[1], n_points[0]), np.inf),
            'bic_matrix': np.full((n_points[1], n_points[0]), np.inf),
            'optimized_params': [[None for _ in range(n_points[0])] for _ in range(n_points[1])],
            'convergence_matrix': np.full((n_points[1], n_points[0]), False, dtype=bool)
        }
        
        # Store original parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Setup progress bar
        total_points = n_points[0] * n_points[1]
        if verbose:
            pbar = tqdm(total=total_points, desc="1-Layer Width+DW Sweep")
        
        for i, width_val in enumerate(width_values):
            for j, dw_val in enumerate(dw_values):
                try:
                    # Reset to original parameters
                    self.model_params = copy.deepcopy(original_params)
                    self.update_traditional_from_model_params()
                    
                    # Set both widths to the same value
                    self.model_params['trapezoids'][0]['width'] = width_val  # Bottom width
                    self.model_params['trapezoids'][1]['width'] = width_val  # Top width
                    
                    # Set DW value
                    self.model_params['DW'] = dw_val
                    
                    # Update traditional parameters
                    self.update_traditional_from_model_params()
                    
                    # Create optimization parameters excluding width and DW parameters
                    excluded_params = ['trap_0_width', 'trap_1_width', 'DW']
                    if exclude_from_fit:
                        excluded_params.extend(exclude_from_fit)
                    
                    opt_params = self._create_optimization_params_excluding(excluded_params)
                    
                    if not opt_params:
                        # No parameters to optimize, just calculate GF
                        self.SimInt = self.SimTrap_SM()
                        gf = self.GF_calc(self.SimInt)
                        bic = self.BIC_calc(gf)
                        converged = True
                    else:
                        # Run optimization
                        opt_result = self.CDSAXS_DiffEvolution(
                            params_to_optimize=opt_params,
                            **optimization_kwargs
                        )
                        gf = self.GF
                        bic = self.BIC
                        converged = opt_result is not None
                    
                    # Store results
                    results['gf_matrix'][j, i] = gf
                    results['bic_matrix'][j, i] = bic
                    results['optimized_params'][j][i] = copy.deepcopy(self.model_params)
                    results['convergence_matrix'][j, i] = converged
                    
                    if verbose:
                        pbar.set_postfix({
                            'Width': f'{width_val:.1f}',
                            'DW': f'{dw_val:.1f}',
                            'GF': f'{gf:.4f}'
                        })
                        pbar.update(1)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error at Width={width_val}, DW={dw_val}: {str(e)}")
                        pbar.update(1)
        
        if verbose:
            pbar.close()
        
        # Restore original parameters
        self.model_params = original_params
        self.update_traditional_from_model_params()
        
        # Plot results
        if plot_results:
            self._plot_width_dw_sweep_results(results, figsize, metric)
        
        # Save results
        if save_results:
            self._save_sweep_results(results, filename or "width_dw_1layer_sweep")
        
        return results
    
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
        self.SimInt = self.SimTrap_SM()
        self.GF = self.GF_calc(self.SimInt)
        self.BIC = self.BIC_calc(self.GF)
        
        print(f"Model updated. New GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
        
                
        
    def get_optimal_parameters_1d(self, results, criterion='GF'):
        """
        Extract the optimal parameters from 1D sweep results.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_1d
        criterion : str, optional
            Criterion for selecting optimal values ('GF' or 'BIC'). Default: 'GF'
            
        Returns:
        --------
        dict
            Dictionary with optimal parameter value and corresponding metrics
        """
        if criterion.upper() == 'GF':
            values_array = np.array(results['gf_values'])
            min_idx = np.nanargmin(values_array)
            min_value = values_array[min_idx]
            corresponding_bic = results['bic_values'][min_idx]
        else:
            values_array = np.array(results['bic_values'])
            min_idx = np.nanargmin(values_array)
            min_value = values_array[min_idx]
            corresponding_bic = min_value
            corresponding_gf = results['gf_values'][min_idx]
        
        optimal_param_value = results['sweep_values'][min_idx]
        optimal_params = results['optimized_params'][min_idx]
        
        return {
            'sweep_parameter': results['sweep_param'],
            'optimal_value': optimal_param_value,
            'gf': results['gf_values'][min_idx] if criterion.upper() == 'GF' else corresponding_gf,
            'bic': corresponding_bic if criterion.upper() == 'GF' else min_value,
            'optimized_parameters': optimal_params,
            'converged': results['convergence_flags'][min_idx],
            'criterion_used': criterion.upper()
        }
    
    def apply_optimal_parameters_1d(self, results, criterion='GF', update_simulation=True):
        """
        Apply the optimal parameters from 1D sweep results to the model.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_1d
        criterion : str, optional
            Criterion for selecting optimal values ('GF' or 'BIC'). Default: 'GF'
        update_simulation : bool, optional
            Whether to recalculate simulation and metrics. Default: True
        """
        optimal = self.get_optimal_parameters_1d(results, criterion)
        
        if optimal['optimized_parameters'] is None:
            raise ValueError("No valid optimized parameters found in results")
        
        print(f"Applying optimal {criterion} parameters:")
        print(f"  {optimal['sweep_parameter']} = {optimal['optimal_value']:.4f}")
        print(f"  {criterion}: {optimal[criterion.lower()]:.4f}")
        
        # Update model parameters
        self.model_params = copy.deepcopy(optimal['optimized_parameters'])
        self.update_traditional_from_model_params()
        
        if update_simulation:
            # Recalculate simulation and metrics
            self.SimInt = self.SimTrap_SM()
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            print(f"Model updated. New GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
        else:
            print("Model parameters updated (simulation not recalculated)")
    
    def get_optimal_parameters_2d(self, results, criterion='GF'):
        """
        Extract the optimal parameters from 2D sweep results.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_2d
        criterion : str, optional
            Criterion for selecting optimal values ('GF' or 'BIC'). Default: 'GF'
            
        Returns:
        --------
        dict
            Dictionary with optimal parameter values and corresponding metrics
        """
        if criterion.upper() == 'GF':
            values_matrix = results['gf_matrix']
            corresponding_matrix = results['bic_matrix']
        else:
            values_matrix = results['bic_matrix']
            corresponding_matrix = results['gf_matrix']
        
        # Find minimum (handle inf values)
        clean_matrix = np.copy(values_matrix)
        clean_matrix[np.isinf(clean_matrix)] = np.nan
        
        min_idx = np.unravel_index(np.nanargmin(clean_matrix), clean_matrix.shape)
        min_value = clean_matrix[min_idx]
        corresponding_value = corresponding_matrix[min_idx]
        
        # Get parameter values
        param1_value = results['param1_values'][min_idx[1]]
        param2_value = results['param2_values'][min_idx[0]]
        
        # Get optimized parameters
        optimal_params = results['optimized_params'][min_idx[0]][min_idx[1]]
        converged = results['convergence_matrix'][min_idx]
        
        return {
            'sweep_parameters': results['sweep_params'],
            'optimal_values': {
                results['sweep_params'][0]: param1_value,
                results['sweep_params'][1]: param2_value
            },
            'gf': min_value if criterion.upper() == 'GF' else corresponding_value,
            'bic': corresponding_value if criterion.upper() == 'GF' else min_value,
            'optimized_parameters': optimal_params,
            'converged': converged,
            'criterion_used': criterion.upper()
        }
    
    def apply_optimal_parameters_2d(self, results, criterion='GF', update_simulation=True):
        """
        Apply the optimal parameters from 2D sweep results to the model.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_2d
        criterion : str, optional
            Criterion for selecting optimal values ('GF' or 'BIC'). Default: 'GF'
        update_simulation : bool, optional
            Whether to recalculate simulation and metrics. Default: True
        """
        optimal = self.get_optimal_parameters_2d(results, criterion)
        
        if optimal['optimized_parameters'] is None:
            raise ValueError("No valid optimized parameters found in results")
        
        print(f"Applying optimal {criterion} parameters:")
        for param, value in optimal['optimal_values'].items():
            print(f"  {param} = {value:.4f}")
        print(f"  {criterion}: {optimal[criterion.lower()]:.4f}")
        
        # Update model parameters
        self.model_params = copy.deepcopy(optimal['optimized_parameters'])
        self.update_traditional_from_model_params()
        
        if update_simulation:
            # Recalculate simulation and metrics
            self.SimInt = self.SimTrap_SM()
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            print(f"Model updated. New GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
        else:
            print("Model parameters updated (simulation not recalculated)")
    
    def compare_sweep_optima(self, results_list, labels=None, criterion='GF'):
        """
        Compare optimal values from multiple sweep results.
        
        Parameters:
        -----------
        results_list : list
            List of results dictionaries from parameter sweeps
        labels : list, optional
            Labels for each result set
        criterion : str, optional
            Criterion for comparison ('GF' or 'BIC'). Default: 'GF'
        """
        if labels is None:
            labels = [f"Sweep {i+1}" for i in range(len(results_list))]
        
        print(f"\nSweep Results Comparison (by {criterion}):")
        print("=" * 80)
        print(f"{'Label':<15} {'Type':<10} {'Parameters':<25} {'GF':<10} {'BIC':<10} {'Converged':<10}")
        print("-" * 80)
        
        all_optima = []
        
        for result, label in zip(results_list, labels):
            try:
                if 'sweep_param' in result:
                    # 1D sweep
                    optimal = self.get_optimal_parameters_1d(result, criterion)
                    sweep_type = "1D"
                    param_str = f"{optimal['sweep_parameter']}={optimal['optimal_value']:.3f}"
                elif 'sweep_params' in result:
                    # 2D sweep
                    optimal = self.get_optimal_parameters_2d(result, criterion)
                    sweep_type = "2D"
                    param_str = f"{optimal['sweep_parameters'][0]}={optimal['optimal_values'][optimal['sweep_parameters'][0]]:.3f}, " + \
                               f"{optimal['sweep_parameters'][1]}={optimal['optimal_values'][optimal['sweep_parameters'][1]]:.3f}"
                elif 'sweep_type' in result and result['sweep_type'] == 'width_dw_1layer':
                    # 1-layer specialized sweep
                    optimal = self.get_optimal_width_dw_1layer(result)
                    sweep_type = "1L-WD"
                    if criterion.upper() == 'GF':
                        param_str = f"W={optimal['best_gf']['width']:.1f}, DW={optimal['best_gf']['dw']:.1f}"
                        gf_val = optimal['best_gf']['gf_value']
                        bic_val = optimal['best_gf']['bic_value']
                    else:
                        param_str = f"W={optimal['best_bic']['width']:.1f}, DW={optimal['best_bic']['dw']:.1f}"
                        gf_val = optimal['best_bic']['gf_value']
                        bic_val = optimal['best_bic']['bic_value']
                    
                    print(f"{label:<15} {sweep_type:<10} {param_str:<25} {gf_val:<10.4f} {bic_val:<10.4f} {'Yes':<10}")
                    all_optima.append((label, gf_val if criterion.upper() == 'GF' else bic_val))
                    continue
                else:
                    print(f"{label:<15} {'Unknown':<10} {'---':<25} {'---':<10} {'---':<10} {'---':<10}")
                    continue
                
                converged_str = "Yes" if optimal['converged'] else "No"
                
                print(f"{label:<15} {sweep_type:<10} {param_str:<25} {optimal['gf']:<10.4f} {optimal['bic']:<10.4f} {converged_str:<10}")
                all_optima.append((label, optimal[criterion.lower()]))
                
            except Exception as e:
                print(f"{label:<15} {'Error':<10} {str(e)[:25]:<25} {'---':<10} {'---':<10} {'---':<10}")
        
        # Find and highlight best overall
        if all_optima:
            best_label, best_value = min(all_optima, key=lambda x: x[1])
            print("-" * 80)
            print(f"Best overall {criterion}: {best_label} with {criterion} = {best_value:.4f}")
        
        print("=" * 80)
    
    def get_sweep_summary(self, results):
        """
        Get a summary of sweep results.
        
        Parameters:
        -----------
        results : dict
            Results from any parameter sweep
            
        Returns:
        --------
        dict
            Summary information about the sweep
        """
        summary = {'sweep_type': 'unknown'}
        
        try:
            if 'sweep_param' in results:
                # 1D sweep
                summary.update({
                    'sweep_type': '1D',
                    'parameter': results['sweep_param'],
                    'range': (results['sweep_values'][0], results['sweep_values'][-1]),
                    'n_points': len(results['sweep_values']),
                    'best_gf': np.nanmin(results['gf_values']),
                    'best_bic': np.nanmin(results['bic_values']),
                    'convergence_rate': np.mean(results['convergence_flags'])
                })
            elif 'sweep_params' in results:
                # 2D sweep
                gf_matrix = np.copy(results['gf_matrix'])
                gf_matrix[np.isinf(gf_matrix)] = np.nan
                
                bic_matrix = np.copy(results['bic_matrix'])
                bic_matrix[np.isinf(bic_matrix)] = np.nan
                
                summary.update({
                    'sweep_type': '2D',
                    'parameters': results['sweep_params'],
                    'ranges': [
                        (results['param1_values'][0], results['param1_values'][-1]),
                        (results['param2_values'][0], results['param2_values'][-1])
                    ],
                    'grid_size': (len(results['param1_values']), len(results['param2_values'])),
                    'best_gf': np.nanmin(gf_matrix),
                    'best_bic': np.nanmin(bic_matrix),
                    'convergence_rate': np.mean(results['convergence_matrix'])
                })
            elif 'sweep_type' in results and results['sweep_type'] == 'width_dw_1layer':
                # 1-layer specialized
                gf_matrix = np.copy(results['gf_matrix'])
                gf_matrix[np.isinf(gf_matrix)] = np.nan
                
                summary.update({
                    'sweep_type': '1-layer Width+DW',
                    'parameters': ['width_both', 'DW'],
                    'ranges': [
                        (results['width_values'][0], results['width_values'][-1]),
                        (results['dw_values'][0], results['dw_values'][-1])
                    ],
                    'grid_size': (len(results['width_values']), len(results['dw_values'])),
                    'best_gf': np.nanmin(gf_matrix),
                    'best_bic': np.nanmin(results['bic_matrix']),
                    'convergence_rate': np.mean(results['convergence_matrix'])
                })
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import differential_evolution

from CDSAXS_base_model import CDSAXS_Model

class TrapezoidModelArrayBg(CDSAXS_Model):
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
        Build model_params dictionary from traditional parameters.
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
        
        # Add optional parameters if they exist
        if hasattr(self, 'SLD') and self.SLD is not None:
            self.model_params['SLD'] = self.SLD
            
        if hasattr(self, 'Pitch') and self.Pitch is not None:
            self.model_params['Pitch'] = self.Pitch
            
        return self.model_params
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary.
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
            self.SymCoordAssign_SingleMaterial()
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
        Initialize optimization parameters with bounds, including array background support.
        
        Parameters:
        -----------
        param_limits : dict, optional
            Dictionary of parameters to optimize with their limits
            If None, creates default limits of ±10% for all parameters
            
        Returns:
        --------
        dict
            Dictionary of optimization parameters with limits
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
            
            # Add background parameters (one per column)
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
            # Ensure default values are set if not provided
            for param, limits in param_limits.items():
                if 'default' not in limits:
                    if param.startswith('trap_'):
                        parts = param.split('_')
                        trap_idx = int(parts[1])
                        param_type = parts[2]
                        limits['default'] = self.model_params['trapezoids'][trap_idx][param_type]
                    elif param.startswith('Bk_'):
                        bk_idx = int(param.split('_')[1])
                        limits['default'] = self.Bk[bk_idx] if isinstance(self.Bk, np.ndarray) else self.Bk
                    else:
                        limits['default'] = getattr(self, param)
        
        # Store optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits
    
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
            for i in range(int(layers)):
                H2 = H2 + Coord[i, 2, 0]
                if i > 0:
                    H1 = H1 + Coord[i-1, 2, 0]
                    
                x1 = Coord[i, 0, 0]
                x4 = Coord[i, 1, 0]
                x2 = Coord[i+1, 0, 0]
                x3 = Coord[i+1, 1, 0]
                
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
        Simulates the intensity for a single material trapezoid structure with array background support.
        
        Parameters:
        -----------
        PAR : numpy.ndarray, optional
            Array with parameters for each layer
            If None, uses self.PAR
        layers : int, optional
            Number of layers in the trapezoid structure
            If None, uses self.layers
        Qx : numpy.ndarray, optional
            X-component of scattering vector, 2D array
            If None, uses self.Qx
        Qz : numpy.ndarray, optional
            Z-component of scattering vector, 2D array
            If None, uses self.Qz
        DW : float, optional
            Debye-Waller factor
            If None, uses self.DW
        I0 : float, optional
            Intensity scaling factor
            If None, uses self.I0
        Bk : float or numpy.ndarray, optional
            Background intensity (scalar or array with one value per column)
            If None, uses self.Bk
            
        Returns:
        --------
        numpy.ndarray
            The simulated intensity
            If called with self attributes, also sets self.SimInt
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
                        # If PAR is not available, we'll try to use existing Coord
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
            
            # Generate coordinates if PAR is provided
            if PAR is not None:
                Coord = self.SymCoordAssign_SingleMaterial(PAR, layers)
                if Coord is None or (using_self and Coord is False):
                    raise RuntimeError("Failed to assign coordinates in SymCoordAssign_SingleMaterial")
            else:
                # Use existing Coord
                if not hasattr(self, 'Coord'):
                    raise AttributeError("Missing required attribute: Coord")
                Coord = self.Coord
            
            # Calculate form factor
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
        Simulates a trapezoid structure and calculates goodness of fit (GF) with array background support.
        Used by the differential evolution algorithm.
        
        Parameters:
        -----------
        optimization_values : numpy.ndarray
            1D array containing values for the parameters being optimized
        param_names : list, optional
            List of parameter names corresponding to optimization_values
            If None, uses self.param_names
        Intensity : numpy.ndarray, optional
            Measured intensity data for comparison
            If None, uses self.Intensity
        Qx : numpy.ndarray, optional
            X-component of scattering vector, 2D array
            If None, uses self.Qx
        Qz : numpy.ndarray, optional
            Z-component of scattering vector, 2D array
            If None, uses self.Qz
            
        Returns:
        --------
        float
            Chi-square value representing goodness of fit
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
            temp_PAR = np.zeros((self.layers + 1, 2))
            for i, trap in enumerate(params['trapezoids']):
                if i <= self.layers:
                    temp_PAR[i, 0] = trap['width']
                    temp_PAR[i, 1] = trap['height']
            
            # Extract global parameters
            temp_DW = params['DW']
            temp_I0 = params['I0']
            
            # Simulate intensity
            SimInt = self.SimTrap_SM(temp_PAR, self.layers, Qx, Qz, temp_DW, temp_I0, temp_Bk)
            if SimInt is None:
                raise RuntimeError("Failed to simulate intensity in SimTrap_SM")
            
            # Calculate goodness of fit
            Chi2 = self.GF_calc(SimInt, Intensity)
            
            return Chi2
            
        except Exception as e:
            print(f"Error in SimTrap_GF: {str(e)}")
            return float('inf')  # Return infinity as worst-case fit
    
    def CDSAXS_DiffEvolution(self, params_to_optimize=None, plot_results=True, 
                            plot_structure=True, plot_grid=True, plot_combined=True, **kwargs):
        """
        Performs differential evolution optimization for CDSAXS trapezoid model fitting
        with array background support and shows before/after comparison plots.
        
        Parameters:
        -----------
        params_to_optimize : dict, optional
            Dictionary containing parameters to optimize with their bounds
            If None, uses self.model_params['optimization']
        plot_results : bool, optional
            Whether to generate any plots (master switch for all plotting)
        plot_structure : bool, optional
            Whether to plot trapezoid structure comparison
        plot_grid : bool, optional
            Whether to plot the grid of individual Qz cuts
        plot_combined : bool, optional
            Whether to plot the combined view with all cuts
        **kwargs : dict
            Additional keyword arguments to pass to scipy's differential_evolution function
            
        Returns:
        --------
        dict
            Optimized parameter dictionary with the same structure as the input model_params
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
            
            # Create parameter names list and bounds list
            param_names = []
            bounds = []
            initial_values = []
            
            for param_name, param_config in params_to_optimize.items():
                param_names.append(param_name)
                bounds.append((param_config['min'], param_config['max']))
                initial_values.append(param_config['default'])
            
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
            
            # Print optimization results
            print(f"Optimization complete after {result.nfev} function evaluations")
            print(f"Initial goodness of fit: {self.GF_Initial:.4f}")
            print(f"Final goodness of fit: {self.GF:.4f}")
            print(f"Improvement: {self.GF_Initial - self.GF:.4f} ({(1 - self.GF/self.GF_Initial)*100:.2f}%)")
            
            # Generate before/after comparison plots if requested
            if plot_results:
                self._plot_optimization_results(initial_model_params, initial_simInt,
                                              plot_structure, plot_grid, plot_combined)
            
            # Print parameter changes
            self._print_parameter_changes(initial_model_params)
            
            return self.model_params
                
        except Exception as e:
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
    
    def _plot_trapezoid_structure(self, model_params, linestyle='-', color='black', alpha=1.0, label=None, linewidth=2, equal_aspect=True, **kwargs):
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
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions
        """
        trapezoids = model_params['trapezoids']
        layers = model_params['layers']
        
        # Plot base
        plt.plot([0, trapezoids[0]['width']], [0, 0], 
                linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        
        height = 0
        for i in range(layers + 1):
            if i > 0:
                height += trapezoids[i-1]['height']
            
            width = trapezoids[i]['width']
            x_left = (trapezoids[0]['width'] - width) / 2
            x_right = x_left + width
            
            plt.plot([x_left, x_right], [height, height], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
            
            if i < layers:
                next_width = trapezoids[i+1]['width']
                x_next_left = (trapezoids[0]['width'] - next_width) / 2
                x_next_right = x_next_left + next_width
                
                plt.plot([x_left, x_next_left], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
                plt.plot([x_right, x_next_right], [height, height + trapezoids[i]['height']], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
        
        # Add a line to the legend
        if label:
            plt.plot([], [], linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        # Set aspect ratio - only use equal if not overridden by user limits
        if equal_aspect:
            plt.axis('equal')
        
        plt.xlabel('Width (nm)')
        plt.ylabel('Height (nm)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        return plt.gca()
    
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
            ax.semilogy(qz_values, self.Intensity[:, i], 'ko', alpha=0.7, 
                      markersize=4, label='Measured')
            
            # Plot initial simulation
            ax.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.8, 
                     linewidth=1.5, label='Initial')
            
            # Plot optimized simulation
            ax.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=1.0, 
                     linewidth=1.5, label='Optimized')
            
            # Set labels and title
            ax.set_title(f'Cut at Qx = {qx_value:.4f}')
            ax.set_xlabel('Qz (nm$^{-1}$)')
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
            plt.semilogy(qz_values, self.Intensity[:, i], 'ko', alpha=0.5, markersize=4)
            
            # Plot initial simulation
            plt.semilogy(qz_values, initial_simInt[:, i], 'b--', alpha=0.5, linewidth=1.5)
            
            # Plot optimized simulation
            plt.semilogy(qz_values, self.SimInt[:, i], 'r-', alpha=0.6, linewidth=1.5)
        
        plt.title('Intensity Comparison - All Cuts')
        plt.xlabel('Qz (nm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
    
    def _print_parameter_changes(self, initial_model_params):
        """
        Print a table of parameter changes from optimization, including array background.
        
        Parameters:
        -----------
        initial_model_params : dict
            Model parameters before optimization
        """
        print("\nParameter Changes:")
        print("=" * 60)
        print(f"{'Parameter':<20} {'Initial':<15} {'Optimized':<15} {'Change %':<10}")
        print("-" * 60)
        
        # Print trapezoid parameters
        initial_traps = initial_model_params['trapezoids']
        optimized_traps = self.model_params['trapezoids']
        max_traps = max(len(initial_traps), len(optimized_traps))
        
        for i in range(max_traps):
            # Handle the case where the trapezoid exists in both models
            if i < len(initial_traps) and i < len(optimized_traps):
                # Print width
                width_init = initial_traps[i]['width']
                width_optim = optimized_traps[i]['width']
                width_change = (width_optim - width_init) / width_init * 100 if width_init != 0 else float('inf')
                print(f"Trap {i} Width{'':<10} {width_init:<15.4f} {width_optim:<15.4f} {width_change:+.2f}%")
                
                # Print height if this isn't the top-most trapezoid (which might not have a height)
                if 'height' in initial_traps[i] and 'height' in optimized_traps[i]:
                    height_init = initial_traps[i]['height']
                    height_optim = optimized_traps[i]['height']
                    height_change = (height_optim - height_init) / height_init * 100 if height_init != 0 else float('inf')
                    print(f"Trap {i} Height{'':<9} {height_init:<15.4f} {height_optim:<15.4f} {height_change:+.2f}%")
            
            # Handle the case where the trapezoid only exists in the initial model
            elif i < len(initial_traps):
                width_init = initial_traps[i]['width']
                print(f"Trap {i} Width{'':<10} {width_init:<15.4f} {'N/A':<15} {'N/A':<10}")
                
                if 'height' in initial_traps[i]:
                    height_init = initial_traps[i]['height']
                    print(f"Trap {i} Height{'':<9} {height_init:<15.4f} {'N/A':<15} {'N/A':<10}")
            
            # Handle the case where the trapezoid only exists in the optimized model
            elif i < len(optimized_traps):
                width_optim = optimized_traps[i]['width']
                print(f"Trap {i} Width{'':<10} {'N/A':<15} {width_optim:<15.4f} {'N/A':<10}")
                
                if 'height' in optimized_traps[i]:
                    height_optim = optimized_traps[i]['height']
                    print(f"Trap {i} Height{'':<9} {'N/A':<15} {height_optim:<15.4f} {'N/A':<10}")
        
        # Print global parameters
        for param in ['DW', 'I0']:
            if param in initial_model_params and param in self.model_params:
                init_val = initial_model_params[param]
                optim_val = self.model_params[param]
                change = (optim_val - init_val) / init_val * 100 if init_val != 0 else float('inf')
                
                print(f"{param:<20} {init_val:<15.6f} {optim_val:<15.6f} {change:+.2f}%")
        
        # Print background parameters
        initial_bk = initial_model_params['Bk']
        optimized_bk = self.model_params['Bk']
        
        if isinstance(initial_bk, list) and isinstance(optimized_bk, list):
            # Array background
            for i, (init_val, optim_val) in enumerate(zip(initial_bk, optimized_bk)):
                change = (optim_val - init_val) / init_val * 100 if init_val != 0 else float('inf')
                print(f"Bk_{i:<17} {init_val:<15.6f} {optim_val:<15.6f} {change:+.2f}%")
        elif not isinstance(initial_bk, list) and not isinstance(optimized_bk, list):
            # Scalar background
            change = (optimized_bk - initial_bk) / initial_bk * 100 if initial_bk != 0 else float('inf')
            print(f"Bk{'':<18} {initial_bk:<15.6f} {optimized_bk:<15.6f} {change:+.2f}%")
        else:
            # Mixed case (shouldn't happen in normal use)
            print(f"Bk{'':<18} {str(initial_bk):<15} {str(optimized_bk):<15} {'Mixed':<10}")
        
        print("=" * 60)
    
    def plot_structure(self, figsize=(10, 6), xlim=None, ylim=None, title='Trapezoid Structure', 
                      show_dimensions=False, color='blue', linewidth=2, equal_aspect=True, **kwargs):
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
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=figsize)
        ax = self._plot_trapezoid_structure(self.model_params, 
                                          linestyle='-', 
                                          color=color, 
                                          alpha=1.0,
                                          linewidth=linewidth,
                                          label=None,
                                          equal_aspect=equal_aspect,
                                          **kwargs)
        
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

    def PlotQzCut(self, cut_index=None, SimInt=None, log_scale='yes'):
        """
        Plots intensity vs Qz for specific Qx cut(s)
        
        Parameters:
        -----------
        cut_index : int or list or None, optional
            Index or indices of the Qx cut(s) to plot
            If None, plots all available cuts
        SimInt : numpy.ndarray, optional
            Simulated intensity to plot alongside measured data
            If None, uses self.SimInt if available
        log_scale : str, optional
            Whether to use logarithmic scale for intensity ('yes' or 'no')
        
        Returns:
        --------
        matplotlib.axes.Axes or list of Axes
            The axes object(s) containing the plot(s)
        """
        # Check if required attributes exist
        if not hasattr(self, 'Qz') or not hasattr(self, 'Intensity'):
            raise AttributeError("Missing required attributes: Qz and/or Intensity")
        
        # Determine which cuts to plot
        if cut_index is None:
            # Plot all cuts
            cut_indices = list(range(self.Intensity.shape[1]))
        elif isinstance(cut_index, (list, tuple, np.ndarray)):
            # Plot multiple specified cuts
            cut_indices = cut_index
        else:
            # Plot a single cut
            cut_indices = [cut_index]
        
        # Create a figure with appropriate size
        n_cuts = len(cut_indices)
        if n_cuts == 1:
            # Single plot
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
        else:
            # Multiple plots
            fig_width = min(16, n_cuts * 5)  # Limit maximum width
            fig_height = min(10, n_cuts * 3)  # Limit maximum height
            
            if n_cuts <= 4:
                # Use a single row for 2-4 plots
                n_rows = 1
                n_cols = n_cuts
            else:
                # Create a grid for many plots
                n_rows = int(np.ceil(np.sqrt(n_cuts)))
                n_cols = int(np.ceil(n_cuts / n_rows))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            if n_rows * n_cols > 1:
                axes = axes.flatten()
        
        # Plot each cut
        for i, (ax, idx) in enumerate(zip(axes, cut_indices)):
            # Check if the index is valid
            if idx < 0 or idx >= self.Intensity.shape[1]:
                ax.text(0.5, 0.5, f"Invalid cut index: {idx}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get Qz values for the selected cut
            qz_values = self.Qz[:, idx]
            qx_value = self.Qx[0, idx]
            
            # Plot measured intensity
            measured_line, = ax.plot(qz_values, self.Intensity[:, idx], 'bo-', label='Measured')
            
            # Plot simulated intensity if available
            if SimInt is not None:
                simulated_line, = ax.plot(qz_values, SimInt[:, idx], 'r-', label='Simulated')
            elif hasattr(self, 'SimInt') and self.SimInt is not None:
                simulated_line, = ax.plot(qz_values, self.SimInt[:, idx], 'r-', label='Simulated')
            
            # Set logarithmic scale if requested
            if log_scale.lower() == 'yes':
                ax.set_yscale('log')
            
            # Set labels and title
            ax.set_title(f'Cut at Qx = {qx_value:.4f}')
            ax.set_xlabel('Qz (nm$^{-1}$)')
            ax.set_ylabel('Intensity (counts)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(cut_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Return a single axis for a single plot, or list of axes for multiple plots
        return axes[0] if len(axes) == 1 else axes
    
    
      



 
    
    
    def simulate_structure(self, *args, **kwargs):
        """
        Simulate trapezoid structure intensity.
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (also sets self.SimInt)
        """
        return self.SimTrap_SM(*args, **kwargs)