import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import copy
from scipy.optimize import differential_evolution

from CDSAXS_base_model import CDSAXS_Model

class CylinderModel(CDSAXS_Model):
    """
    CDSAXS model for cylindrical structures.
    """
    
    def __init__(self, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Initialize the cylinder model.
        
        Parameters:
        -----------
        model : str
            Model type
        layers : int
            Number of layers
        PAR : numpy.ndarray, optional
            Traditional parameter array with radius and height
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
        super().__init__('cylinder', model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
        
        # Initialize cylinder-specific attributes
        self.Qr = None
        self.Alpha = None
        
        # Set default discretization if not provided
        if not hasattr(self, 'discretization'):
            self.discretization = [10] * self.layers
            if hasattr(self, 'model_params'):
                self.model_params['discretization'] = self.discretization
    
    def build_model_params_from_traditional(self):
        """
        Build model_params dictionary from traditional parameters.
        """
        if not hasattr(self, 'PAR') or self.PAR is None:
            return
            
        # Create cylinders list from PAR
        cylinders = []
        for i in range(self.layers + 1):
            cylinder = {
                'radius': self.PAR[i, 0],
                'height': self.PAR[i, 1]
            }
            cylinders.append(cylinder)
        
        # Create the model_params dictionary
        self.model_params = {
            'layers': self.layers,
            'cylinders': cylinders,
            'DW': self.DW,
            'I0': self.I0,
            'Bk': self.Bk,
            'discretization': self.discretization if hasattr(self, 'discretization') else [10] * self.layers
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
            
        # Update PAR from cylinders
        cylinders = self.model_params['cylinders']
        if not hasattr(self, 'PAR') or self.PAR is None or self.PAR.shape[0] != len(cylinders):
            self.PAR = np.zeros((len(cylinders), 2))
            
        for i, cyl in enumerate(cylinders):
            self.PAR[i, 0] = cyl['radius']
            self.PAR[i, 1] = cyl['height']
        
        # Update global parameters
        self.DW = self.model_params['DW']
        self.I0 = self.model_params['I0']
        self.Bk = self.model_params['Bk']
        
        # Update discretization
        if 'discretization' in self.model_params:
            self.discretization = self.model_params['discretization']
        
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
        Process imported data for cylinder model.
        """
        try:
            # Convert Cartesian to cylindrical coordinates
            self.convert_Cartesian_Cylindrical()
            
            # Run initial simulation
            if not hasattr(self, 'discretization'):
                self.discretization = [10] * self.layers
                if hasattr(self, 'model_params'):
                    self.model_params['discretization'] = self.discretization
                    
            self.SimCyl_SM(self.discretization)
            self.SimInt_Initial = self.SimInt.copy() if hasattr(self, 'SimInt') else None
            self.GF = self.GF_calc(self.SimInt)
            self.GF_Initial = self.GF
            self.BIC = self.BIC_calc(self.GF)
            self.BIC_Initial = self.BIC
        except Exception as e:
            print(f"Warning: Error in cylinder initialization: {str(e)}")
            print("Data import successful, but cylinder initialization failed.")
    
    def convert_Cartesian_Cylindrical(self):
        """
        Converts Cartesian coordinates (Qx, Qy) to Cylindrical coordinates (Qr, Alpha)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qy'):
                raise AttributeError("Missing required attributes: Qx and Qy must be defined")
            
            # Check if arrays are properly initialized and have compatible shapes
            if not isinstance(self.Qx, np.ndarray) or not isinstance(self.Qy, np.ndarray):
                raise TypeError("Qx and Qy must be numpy arrays")
                
            if self.Qx.shape != self.Qy.shape:
                raise ValueError(f"Shape mismatch: Qx shape {self.Qx.shape} different from Qy shape {self.Qy.shape}")
                
            # Check for NaN or empty arrays
            if np.all(np.isnan(self.Qx)) or np.all(np.isnan(self.Qy)):
                raise ValueError("Input arrays contain only NaN values")
                
            # Handle division by zero (when Qx = 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                Alpha = np.arctan(self.Qy / self.Qx)
                # Replace NaN resulting from 0/0 with 0 or appropriate value
                Alpha = np.nan_to_num(Alpha, nan=0.0)
            
            # Compute Qr 
            self.Qr = self.Qx / np.cos(Alpha)
            
            # Store Alpha for future use
            self.Alpha = Alpha
            
            return True
            
        except Exception as e:
            print(f"Error in convert_Cartesian_Cylindrical: {str(e)}")
            return False
    
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
            
            # Add cylinder parameters
            for i, cyl in enumerate(self.model_params['cylinders']):
                param_limits[f'cyl_{i}_radius'] = {
                    'min': cyl['radius'] * 0.9,
                    'max': cyl['radius'] * 1.1,
                    'default': cyl['radius']
                }
                
                param_limits[f'cyl_{i}_height'] = {
                    'min': cyl['height'] * 0.9,
                    'max': cyl['height'] * 1.1,
                    'default': cyl['height']
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
                    if param.startswith('cyl_'):
                        parts = param.split('_')
                        cyl_idx = int(parts[1])
                        param_type = parts[2]
                        limits['default'] = self.model_params['cylinders'][cyl_idx][param_type]
                    else:
                        limits['default'] = getattr(self, param)
        
        # Store optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits
    
    def _extract_PAR_from_model_params(self):
        """
        Helper method to extract PAR array from model_params.
        
        Returns:
        --------
        numpy.ndarray
            PAR array extracted from model_params
        """
        if not hasattr(self, 'model_params'):
            raise AttributeError("Missing required attribute: model_params")
            
        cylinders = self.model_params['cylinders']
        layers = self.model_params['layers']
        
        # Create PAR array
        PAR = np.zeros([layers + 1, 2])
        for i, cyl in enumerate(cylinders):
            if i <= layers:
                PAR[i, 0] = cyl['radius']
                PAR[i, 1] = cyl['height']
                
        return PAR
    
    def ConeFourierTransform(self, Discretization=None):
        """
        Fourier transform for a cone in cylindrical coordinates (Qr,Qz) 
        
        Parameters:
        -----------
        Discretization : list or numpy.ndarray, optional
            Number of discretization steps for each layer
            If None, uses self.discretization
            
        Returns:
        --------
        numpy.ndarray
            The calculated form factor (also sets self.form)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qr') or not hasattr(self, 'Qz'):
                raise AttributeError("Missing required attributes: Qr and Qz")
                
            if not hasattr(self, 'PAR'):
                raise AttributeError("Missing required attribute: PAR")
                
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
                
            # Use provided discretization or default
            if Discretization is None:
                if not hasattr(self, 'discretization'):
                    # Create default discretization
                    self.discretization = [10] * self.layers
                Discretization = self.discretization
                
            # Check discretization length
            if len(Discretization) < self.layers:
                raise ValueError(f"Discretization array must have at least {self.layers} elements")
            
            # Initialize variables
            H1 = 0
            H2 = 0
            self.form = np.zeros([int(len(self.Qr[:,0])), int(len(self.Qr[0,:]))])
            
            # Perform Fourier transform
            for i in range(self.layers):
                H2 = H2 + self.PAR[i, 1]
                stepsize = self.PAR[i, 1] / Discretization[i]
                
                if i > 0:
                    H1 = H1 + self.PAR[i-1, 1]
                    
                z = np.arange(H1, H2 + 0.01, stepsize)
                R1 = self.PAR[i, 0]
                R2 = self.PAR[i+1, 0]
                
                # Avoid division by zero
                if R1 == R2:
                    R1 = R1 + 0.000001
                    
                Slope = (H2 - H1) / (R2 - R1)
                
                for ii in range(len(z) - 1):
                    RI1 = (z[ii] - H1) / Slope + R1
                    RI2 = (z[ii+1] - H1) / Slope + R1
                    fa = 2 * np.pi * RI1 / self.Qr * sp.jv(1, self.Qr * RI1) * np.exp(1j * self.Qz * z[ii])
                    fb = 2 * np.pi * RI2 / self.Qr * sp.jv(1, self.Qr * RI2) * np.exp(1j * self.Qz * z[ii+1])
                    self.form = self.form + stepsize * (fb + fa) / 2  # If you had an SLD variation you would multiply by the SLD here
            
            return self.form
            
        except Exception as e:
            print(f"Error in ConeFourierTransform: {str(e)}")
            self.form = None
            return None
    
    def ConeFourierTransformOpt(self, PAR, layers, Qz, Qr, Discretization):
        """
        Optimized Fourier transform for a cone in cylindrical coordinates (Qr, Qz).
        
        Parameters:
        -----------
        PAR : numpy.ndarray
            Parameter array with shape (n, 2) containing radius and height information
        layers : int
            Number of layers in the cone structure
        Qz : numpy.ndarray
            Z-component of scattering vector, 2D array
        Qr : numpy.ndarray
            Radial component of scattering vector, 2D array
        Discretization : list or numpy.ndarray
            Number of discretization steps for each layer
            
        Returns:
        --------
        numpy.ndarray
            The calculated form factor
        """
        try:
            # Validate input parameters
            if PAR is None or not isinstance(PAR, np.ndarray):
                raise TypeError("PAR must be a numpy array")
                
            if layers is None or not isinstance(layers, (int, float)) or layers <= 0:
                raise ValueError(f"layers must be a positive number, got {layers}")
                
            if Qr is None or Qz is None:
                raise ValueError("Qr and Qz must not be None")
                
            if not isinstance(Qr, np.ndarray) or not isinstance(Qz, np.ndarray):
                raise TypeError("Qr and Qz must be numpy arrays")
                
            if len(Qr.shape) != 2 or len(Qz.shape) != 2:
                raise ValueError(f"Qr and Qz must be 2D arrays, got shapes {Qr.shape} and {Qz.shape}")
                
            if Discretization is None or len(Discretization) < layers:
                raise ValueError(f"Discretization array must have at least {layers} elements")
                
            # Check PAR dimensions for indexing
            if layers + 1 > len(PAR):
                raise IndexError(f"Not enough rows in PAR ({len(PAR)}) for {layers} layers")
            
            # Initialize variables
            H1 = 0
            H2 = 0
            Form = np.zeros([int(len(Qr[:,0])), int(len(Qr[0,:]))])
            
            # Perform Fourier transform
            for i in range(layers):
                H2 = H2 + PAR[i, 1]
                stepsize = PAR[i, 1] / Discretization[i]
                
                if i > 0:
                    H1 = H1 + PAR[i-1, 1]
                    
                z = np.arange(H1, H2 + 0.01, stepsize)
                R1 = PAR[i, 0]
                R2 = PAR[i+1, 0]
                
                # Avoid division by zero
                if R1 == R2:
                    R1 = R1 + 0.000001
                    
                Slope = (H2 - H1) / (R2 - R1)
                
                for ii in range(len(z) - 1):
                    RI1 = (z[ii] - H1) / Slope + R1
                    RI2 = (z[ii+1] - H1) / Slope + R1
                    fa = 2 * np.pi * RI1 / Qr * sp.jv(1, Qr * RI1) * np.exp(1j * Qz * z[ii])
                    fb = 2 * np.pi * RI2 / Qr * sp.jv(1, Qr * RI2) * np.exp(1j * Qz * z[ii+1])
                    Form = Form + stepsize * (fb + fa) / 2
                    
            return Form
            
        except Exception as e:
            print(f"Error in ConeFourierTransformOpt: {str(e)}")
            return None
    
    def SimCyl_SM(self, Discretization=None):
        """
        Simulates the intensity for a single material cylindrical structure.
        
        Parameters:
        -----------
        Discretization : list or numpy.ndarray, optional
            Number of discretization steps for each layer
            If None, uses self.discretization
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (self.SimInt)
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Qr') or not hasattr(self, 'Qz'):
                if hasattr(self, 'Qx') and hasattr(self, 'Qy'):
                    # Try to convert from Cartesian to cylindrical
                    self.convert_Cartesian_Cylindrical()
                else:
                    raise AttributeError("Missing required scattering vector attributes: Qr and/or Qz")
            
            if not hasattr(self, 'DW'):
                raise AttributeError("Missing required attribute: DW (Debye-Waller factor)")
                
            if not hasattr(self, 'I0'):
                raise AttributeError("Missing required attribute: I0 (Intensity scaling factor)")
                
            if not hasattr(self, 'Bk'):
                raise AttributeError("Missing required attribute: Bk (Background intensity)")
                
            # Check if layers attribute exists for ConeFourierTransform
            if not hasattr(self, 'layers'):
                raise AttributeError("Missing required attribute: layers")
                
            # Use provided discretization or default
            if Discretization is None:
                if not hasattr(self, 'discretization'):
                    # Create default discretization
                    self.discretization = [10] * self.layers
                Discretization = self.discretization
                
            # Check discretization length
            if len(Discretization) < self.layers:
                raise ValueError(f"Discretization array must have at least {self.layers} elements")
            
            # Execute form factor calculation
            self.ConeFourierTransform(Discretization)
            if not hasattr(self, 'form') or self.form is None:
                raise RuntimeError("Failed to calculate form factor in ConeFourierTransform")
            
            # Calculate Debye-Waller factor
            # For cylindrical geometry, use Qr instead of Qx
            M = np.power(np.exp(-1 * (np.power(self.Qr, 2) + np.power(self.Qz, 2)) * np.power(self.DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = self.form * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity
            self.SimInt = np.power(Formfactor, 2) * self.I0 + self.Bk
            
            return self.SimInt
            
        except Exception as e:
            print(f"Error in SimCyl_SM: {str(e)}")
            self.SimInt = None
            return None
    
    def SimCyl_GF(self, SimPar, layers, Intensity, Qr, Qz, Discretization):
        """
        Simulates a cylindrical structure and calculates goodness of fit (GF).
        
        Parameters:
        -----------
        SimPar : numpy.ndarray
            1D array containing all simulation parameters (layer dimensions, I0, DW, Bk)
        layers : int
            Number of layers in the cylindrical structure
        Intensity : numpy.ndarray
            Measured intensity data for comparison
        Qr : numpy.ndarray
            Radial component of scattering vector, 2D array
        Qz : numpy.ndarray
            Z-component of scattering vector, 2D array
        Discretization : list or numpy.ndarray
            Number of discretization steps for each layer
            
        Returns:
        --------
        float
            Chi-square value representing goodness of fit
        """
        try:
            # Validate input parameters
            if SimPar is None or not isinstance(SimPar, np.ndarray):
                raise TypeError("SimPar must be a numpy array")
                
            if layers is None or not isinstance(layers, (int, float)) or layers < 0:
                raise ValueError(f"layers must be a non-negative number, got {layers}")
                
            if Intensity is None or not isinstance(Intensity, np.ndarray):
                raise TypeError("Intensity must be a numpy array")
                
            if Qr is None or Qz is None:
                raise ValueError("Qr and Qz must not be None")
                
            if not isinstance(Qr, np.ndarray) or not isinstance(Qz, np.ndarray):
                raise TypeError("Qr and Qz must be numpy arrays")
                
            if Discretization is None:
                raise ValueError("Discretization must not be None")
                
            if len(Discretization) < layers:
                raise ValueError(f"Discretization array must have at least {layers} elements")
                
            # Check if SimPar has sufficient elements
            required_length = (layers + 1) * 2 + 3  # For PARs, I0, DW, Bk
            if len(SimPar) < required_length:
                raise ValueError(f"SimPar array must have at least {required_length} elements, but has {len(SimPar)}")
            
 # Reshape parameters
            PARs = np.zeros([layers + 1, 2])
            PARs[:, 0:2] = np.reshape(SimPar[0:(layers + 1) * 2], (layers + 1, 2))
            
            # Extract I0, DW, Bk parameters
            I0 = SimPar[(layers + 1) * 2]
            DW = SimPar[(layers + 1) * 2 + 1]
            Bk = SimPar[(layers + 1) * 2 + 2]
            
            # Calculate form factor
            F1 = self.ConeFourierTransformOpt(PARs, layers, Qz, Qr, Discretization)
            if F1 is None:
                raise RuntimeError("Failed to calculate form factor in ConeFourierTransformOpt")
            
            # Calculate Debye-Waller factor
            M = np.power(np.exp(-1 * (np.power(Qr, 2) + np.power(Qz, 2)) * np.power(DW, 2)), 0.5)
            
            # Apply Debye-Waller factor to form factor
            Formfactor = F1 * M
            Formfactor = abs(Formfactor)
            
            # Calculate intensity
            SimInt = np.power(Formfactor, 2) * I0 + Bk
            
            # Calculate Chi-square
            Chi2 = abs(np.log(Intensity) - np.log(SimInt))
            Chi2[np.isnan(Chi2)] = 0
            Chi2 = np.sum(Chi2)
            
            return Chi2
            
        except Exception as e:
            print(f"Error in SimCyl_GF: {str(e)}")
            return float('inf')  # Return infinity as worst-case fit
    
    def _cylinder_optimization_wrapper(self, optimization_values):
        """
        Wrapper function for cylindrical optimization that can be pickled.
        """
        # Create PAR array from optimization values
        temp_PAR = np.zeros((self.layers + 1, 2))
        temp_DW = self.DW
        temp_I0 = self.I0
        temp_Bk = self.Bk
        
        for i, param_name in enumerate(self.param_names):
            if param_name.startswith('cyl_'):
                parts = param_name.split('_')
                cyl_idx = int(parts[1])
                param_type = parts[2]
                
                if param_type == 'radius':
                    temp_PAR[cyl_idx, 0] = optimization_values[i]
                elif param_type == 'height':
                    temp_PAR[cyl_idx, 1] = optimization_values[i]
            elif param_name == 'DW':
                temp_DW = optimization_values[i]
            elif param_name == 'I0':
                temp_I0 = optimization_values[i]
            elif param_name == 'Bk':
                temp_Bk = optimization_values[i]
        
        # Create SimPar array for cylindrical GF function
        SimPar = np.append(temp_PAR.ravel(), [temp_I0, temp_DW, temp_Bk])
        
        # Call cylindrical GF function
        return self.SimCyl_GF(SimPar, self.layers, self.Intensity, self.Qr, self.Qz, self.discretization)

    def CDSAXS_DiffEvolution(self, params_to_optimize=None, plot_results=True, 
                            plot_structure=True, plot_grid=True, plot_combined=True, **kwargs):
        """
        Performs differential evolution optimization for CDSAXS cylindrical model fitting
        and shows before/after comparison plots.
        
        Parameters:
        -----------
        params_to_optimize : dict, optional
            Dictionary containing parameters to optimize with their bounds
            If None, uses self.model_params['optimization']
        plot_results : bool, optional
            Whether to generate any plots (master switch for all plotting)
        plot_structure : bool, optional
            Whether to plot cylinder structure comparison
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
                
            if not hasattr(self, 'Qr') or not hasattr(self, 'Qz'):
                # Try to convert from Cartesian to cylindrical
                if hasattr(self, 'Qx') and hasattr(self, 'Qy'):
                    self.convert_Cartesian_Cylindrical()
                else:
                    raise AttributeError("Missing required scattering vector attributes: Qr and/or Qz")
                    
            # Ensure discretization is set
            if not hasattr(self, 'discretization'):
                self.discretization = [10] * self.layers
                self.model_params['discretization'] = self.discretization
            
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
            
            # Store for use in optimization
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
                self.SimInt = self.SimCyl_SM(self.discretization)
                
            # Store initial simulation results
            initial_simInt = copy.deepcopy(self.SimInt)
            
            # Calculate initial goodness of fit if not already done
            if not hasattr(self, 'GF_Initial') or self.GF_Initial is None:
                self.GF_Initial = self.GF_calc(self.SimInt)
            
            # Run differential evolution optimization
            print(f"Starting optimization with {len(param_names)} parameters...")
            
            # Run the optimization using the method-level wrapper (can be pickled)
            result = differential_evolution(
                self._cylinder_optimization_wrapper,
                bounds,
                **optimization_params
            )
            
            # Store the optimization result
            self.optimization_result = result
            
            # Update model parameters with optimized values
            optimized_params = self.model_params.copy()
            
            # Make a deep copy of cylinders to avoid modifying the original
            optimized_params['cylinders'] = [cyl.copy() for cyl in self.model_params['cylinders']]
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('cyl_'):
                    # Parse cylinder parameter
                    parts = param_name.split('_')
                    cyl_idx = int(parts[1])
                    param_type = parts[2]  # 'radius' or 'height'
                    
                    optimized_params['cylinders'][cyl_idx][param_type] = result.x[i]
                else:
                    # Global parameter (DW, I0, Bk)
                    optimized_params[param_name] = result.x[i]
            
            # Update class attributes with optimized values
            self.model_params = optimized_params
            self.update_traditional_from_model_params()
            
            # Simulate with optimized parameters
            self.SimInt = self.SimCyl_SM(self.discretization)
            
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
            Whether to plot cylinder structure comparison
        plot_grid : bool
            Whether to plot the grid of individual Qz cuts
        plot_combined : bool
            Whether to plot the combined view with all cuts
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Plot cylinder structure comparison on the same plot
        if plot_structure:
            plt.figure(figsize=(10, 6))
            
            # Plot initial cylinder structure with dashed lines and transparency
            self._plot_cylinder_structure(initial_model_params, 
                                      linestyle='--', 
                                      color='blue', 
                                      alpha=0.7,
                                      label='Initial')
            
            # Plot optimized cylinder structure with solid lines
            self._plot_cylinder_structure(self.model_params, 
                                      linestyle='-', 
                                      color='red', 
                                      alpha=1.0,
                                      label='Optimized')
            
            plt.title('Cylinder Structure Comparison')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Plot QzCut comparisons - grid of individual cuts
        if plot_grid:
            self._plot_qzcut_grid(initial_simInt)
        
        # Plot combined view with all cuts
        if plot_combined:
            self._plot_qzcut_combined(initial_simInt)
    
    def _plot_cylinder_structure(self, model_params, linestyle='-', color='black', alpha=1.0, label=None):
        """
        Plot the cylinder structure from the given model parameters.
        
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
        cylinders = model_params['cylinders']
        layers = model_params['layers']
        
        # Plot cylinder side view (stacked cylinders)
        height = 0
        for i in range(layers + 1):
            radius = cylinders[i]['radius']
            
            # Plot bottom line of cylinder
            plt.plot([-radius, radius], [height, height], 
                    linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
            
            # Plot vertical lines for this layer
            if i < layers:
                cyl_height = cylinders[i]['height']
                
                plt.plot([-radius, -cylinders[i+1]['radius']], [height, height + cyl_height], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
                        
                plt.plot([radius, cylinders[i+1]['radius']], [height, height + cyl_height], 
                        linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
                        
                height += cyl_height
        
        # Plot top line of the last cylinder
        plt.plot([-cylinders[layers]['radius'], cylinders[layers]['radius']], [height, height], 
                linestyle=linestyle, color=color, alpha=alpha, linewidth=2)
        
        # Add a line to the legend
        if label:
            plt.plot([], [], linestyle=linestyle, color=color, alpha=alpha, linewidth=2, label=label)
        
        plt.axis('equal')
        plt.xlabel('Radius (nm)')
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
            qr_value = self.Qr[0, i]
            
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
            ax.set_title(f'Cut at Qr = {qr_value:.4f}')
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
        
        # Print cylinder parameters
        initial_cyls = initial_model_params['cylinders']
        optimized_cyls = self.model_params['cylinders']
        max_cyls = max(len(initial_cyls), len(optimized_cyls))
        
        for i in range(max_cyls):
            # Handle the case where the cylinder exists in both models
            if i < len(initial_cyls) and i < len(optimized_cyls):
                # Print radius
                radius_init = initial_cyls[i]['radius']
                radius_optim = optimized_cyls[i]['radius']
                radius_change = (radius_optim - radius_init) / radius_init * 100 if radius_init != 0 else float('inf')
                print(f"Cyl {i} Radius{'':<10} {radius_init:<15.4f} {radius_optim:<15.4f} {radius_change:+.2f}%")
                
                # Print height if this isn't the top-most cylinder (which might not have a height)
                if 'height' in initial_cyls[i] and 'height' in optimized_cyls[i]:
                    height_init = initial_cyls[i]['height']
                    height_optim = optimized_cyls[i]['height']
                    height_change = (height_optim - height_init) / height_init * 100 if height_init != 0 else float('inf')
                    print(f"Cyl {i} Height{'':<9} {height_init:<15.4f} {height_optim:<15.4f} {height_change:+.2f}%")
            
            # Handle the case where the cylinder only exists in the initial model
            elif i < len(initial_cyls):
                radius_init = initial_cyls[i]['radius']
                print(f"Cyl {i} Radius{'':<10} {radius_init:<15.4f} {'N/A':<15} {'N/A':<10}")
                
                if 'height' in initial_cyls[i]:
                    height_init = initial_cyls[i]['height']
                    print(f"Cyl {i} Height{'':<9} {height_init:<15.4f} {'N/A':<15} {'N/A':<10}")
            
            # Handle the case where the cylinder only exists in the optimized model
            elif i < len(optimized_cyls):
                radius_optim = optimized_cyls[i]['radius']
                print(f"Cyl {i} Radius{'':<10} {'N/A':<15} {radius_optim:<15.4f} {'N/A':<10}")
                
                if 'height' in optimized_cyls[i]:
                    height_optim = optimized_cyls[i]['height']
                    print(f"Cyl {i} Height{'':<9} {'N/A':<15} {height_optim:<15.4f} {'N/A':<10}")
        
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
        Plots the current cylinder structure.
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=(10, 6))
        ax = self._plot_cylinder_structure(self.model_params)
        plt.title('Cylinder Structure')
        return ax

    def PlotQzCut(self, cut_index=None, SimInt=None, log_scale='yes'):
        """
        Plots intensity vs Qz for specific Qr cut(s)
        
        Parameters:
        -----------
        cut_index : int or list or None, optional
            Index or indices of the Qr cut(s) to plot
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
            qr_value = self.Qr[0, idx]
            
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
            ax.set_title(f'Cut at Qr = {qr_value:.4f}')
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
    

        
    def simulate_structure(self, *args, **kwargs):
        """
        Simulate cylinder structure intensity.
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (also sets self.SimInt)
        """
        return self.SimCyl_SM(*args, **kwargs)