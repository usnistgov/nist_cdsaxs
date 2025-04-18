import os
import re
import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import scipy.special as sp
import copy

class CDSAXS_Model:
    
    def __init__(self, geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Initialize the CDSAXS model with either traditional parameters or a dictionary-based approach.
        
        Parameters:
        -----------
        geometry : str
            Geometry type (e.g., 'trapezoid' or 'cylinder')
        model : str
            Model type
        layers : int
            Number of layers
        PAR : numpy.ndarray, optional
            Traditional parameter array (used if model_params is None)
        SLD : numpy.ndarray, optional
            Scattering length density array (used if model_params is None)
        DW : float, optional
            Debye-Waller factor (used if model_params is None)
        I0 : float, optional
            Intensity scaling factor (used if model_params is None)
        Bk : float, optional
            Background intensity (used if model_params is None)
        Pitch : float, optional
            Pitch parameter (used if model_params is None)
        model_params : dict, optional
            Dictionary-based parameters (if provided, overrides traditional parameters)
        """
        self.geometry = geometry
        self.model = model
        self.layers = layers
        
        # If model_params is provided, use the dictionary-based approach
        if model_params is not None:
            self.model_params = model_params
            
            # Extract traditional parameters from model_params for compatibility
            if self.geometry == 'trapezoid':
                trapezoids = model_params['trapezoids']
                
                # Create PAR array from trapezoids
                self.PAR = np.zeros((layers + 1, 2))
                for i, trap in enumerate(trapezoids):
                    if i <= layers:
                        self.PAR[i, 0] = trap['width']
                        self.PAR[i, 1] = trap['height']
            elif self.geometry == 'cylinder':
                cylinders = model_params['cylinders']
                
                # Create PAR array from cylinders (radius, height)
                self.PAR = np.zeros((layers + 1, 2))
                for i, cyl in enumerate(cylinders):
                    if i <= layers:
                        self.PAR[i, 0] = cyl['radius']
                        self.PAR[i, 1] = cyl['height']
            
            # Set global parameters
            self.DW = model_params['DW']
            self.I0 = model_params['I0']
            self.Bk = model_params['Bk']
            
            # Set SLD if provided
            if 'SLD' in model_params:
                self.SLD = model_params['SLD']
            elif SLD is not None:
                self.SLD = SLD
            
            # Set Pitch if provided
            if 'Pitch' in model_params:
                self.Pitch = model_params['Pitch']
            elif Pitch is not None:
                self.Pitch = Pitch
                
            # Set discretization for cylindrical geometry if provided
            if 'discretization' in model_params:
                self.discretization = model_params['discretization']
        else:
            # Traditional parameter initialization
            self.PAR = PAR
            self.SLD = SLD
            self.DW = DW
            self.I0 = I0
            self.Bk = Bk
            self.Pitch = Pitch
            
            # Create model_params from traditional parameters
            self.build_model_params_from_traditional()
        
        # Store initial values
        self.PAR_Initial = np.copy(self.PAR) if self.PAR is not None else None
        self.DW_Initial = self.DW
        self.I0_Initial = self.I0
        self.Bk_Initial = self.Bk
        
        # Create SimPar for compatibility with existing code
        if self.PAR is not None:
            self.SimPar = np.append(self.PAR.ravel(), [self.I0, self.DW, self.Bk])
            
        # Initialize Qr for cylindrical geometry
        if self.geometry == 'cylinder':
            self.Qr = None

    def build_model_params_from_traditional(self):
        """
        Build model_params dictionary from traditional parameters.
        """
        if not hasattr(self, 'PAR') or self.PAR is None:
            return
            
        # Create appropriate structure based on geometry
        if self.geometry == 'trapezoid':
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
        elif self.geometry == 'cylinder':
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
                'Bk': self.Bk
            }
            
            # Add discretization if available
            if hasattr(self, 'discretization'):
                self.model_params['discretization'] = self.discretization
        
        # Add optional parameters if they exist
        if hasattr(self, 'SLD') and self.SLD is not None:
            self.model_params['SLD'] = self.SLD
            
        if hasattr(self, 'Pitch') and self.Pitch is not None:
            self.model_params['Pitch'] = self.Pitch
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary.
        """
        if not hasattr(self, 'model_params'):
            return
        
        # Update PAR based on geometry
        if self.geometry == 'trapezoid':
            trapezoids = self.model_params['trapezoids']
            if not hasattr(self, 'PAR') or self.PAR is None or self.PAR.shape[0] != len(trapezoids):
                self.PAR = np.zeros((len(trapezoids), 2))
                
            for i, trap in enumerate(trapezoids):
                self.PAR[i, 0] = trap['width']
                self.PAR[i, 1] = trap['height']
        elif self.geometry == 'cylinder':
            cylinders = self.model_params['cylinders']
            if not hasattr(self, 'PAR') or self.PAR is None or self.PAR.shape[0] != len(cylinders):
                self.PAR = np.zeros((len(cylinders), 2))
                
            for i, cyl in enumerate(cylinders):
                self.PAR[i, 0] = cyl['radius']
                self.PAR[i, 1] = cyl['height']
            
            # Update discretization if available
            if 'discretization' in self.model_params:
                self.discretization = self.model_params['discretization']
        
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
    
    def importCDSAXS_GUI(self, Datafile):
        """
        Imports CDSAXS data from a GUI-created file with input validation
        
        Parameters:
        -----------
        Datafile : str
            Path to the data file (CSV format)
        """
        # Check if input variable exists and is valid
        if Datafile is None or not isinstance(Datafile, str):
            raise ValueError("Datafile must be a valid file path")
        
        # Check if file exists
        if not os.path.isfile(Datafile):
            raise FileNotFoundError(f"File not found: {Datafile}")
        
        try:
            # Import data using pandas
            Data = pd.read_csv(Datafile)
            
            # Check if file has content
            if Data.empty:
                raise ValueError("The data file is empty")
            
            # Check the number of cuts
            num_columns = len(Data.columns)
            if num_columns < 2:
                raise ValueError("Data must have at least 2 columns")
                
            numbercuts = num_columns // 2
            headers = Data.columns.tolist()
            
            # Extract qx values from headers
            qxlist = []
            for i in range(1, len(headers), 2):
                # Look for pattern 'qx = number' in the header
                match = re.search(r'qx\s*=\s*(\d+\.?\d*)', headers[i])
                if match:
                    number = float(match.group(1))
                    # Convert to int if it's a whole number
                    if number.is_integer():
                        number = int(number)
                    qxlist.append(number)
            
            # Check if we found any qx values
            if not qxlist:
                raise ValueError("No qx values found in headers")
                
            # Convert to numpy array
            Data1 = Data.to_numpy()
            
            # Initialize arrays
            data_rows = len(Data1[:,0])
            self.Intensity = np.zeros([data_rows, numbercuts])
            self.Qz = np.zeros([data_rows, numbercuts])
            
            # Fill arrays with data
            for i in range(0, numbercuts):
                if (i*2+1) < num_columns:  # Check if column exists
                    self.Intensity[:,i] = Data1[:,(i*2+1)]
                    self.Qz[:,i] = Data1[:,(i*2)]
            
            # Create Qx array
            self.Qx = self.Qz.copy()
            self.Qx[~np.isnan(self.Qx)] = 1
            
            # Apply qx values to each column
            for k, v in enumerate(qxlist):
                if k < self.Qx.shape[1]:  # Check if column exists
                    self.Qx[:,k] = self.Qx[:,k] * v
            
            # Check if Qx values are increasing along the rows and sort if needed
            # Get the first row with valid data to check order
            for row_idx in range(self.Qx.shape[0]):
                if np.all(np.isfinite(self.Qx[row_idx, :])):
                    # Check if Qx is not in ascending order
                    if not np.all(np.diff(self.Qx[row_idx, :]) >= 0):
                        # Get sort indices
                        sort_indices = np.argsort(self.Qx[row_idx, :])
                        
                        # Apply sorting to all arrays
                        self.Qx = self.Qx[:, sort_indices]
                        self.Qz = self.Qz[:, sort_indices]
                        self.Intensity = self.Intensity[:, sort_indices]
                    break
            
            # Create Qy array with zeros where Qx has positive values
            self.Qy = np.zeros_like(self.Qx)
            # Only set values to zero where Qx is positive (keep NaN values as they were)
            self.Qy[self.Qx > 0] = 0
            
            # For cylindrical geometry, convert to cylindrical coordinates
            if self.geometry == 'cylinder':
                self.convert_Cartesian_Cylindrical()
            
            # Calculate number of valid points
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            # Execute geometry-specific code
            if hasattr(self, 'geometry'):
                try:
                    if self.geometry == 'trapezoid':
                        # Trapezoid initialization
                        self.SymCoordAssign_SingleMaterial()
                        self.SimTrap_SM()
                        self.SimInt_Initial = self.SimInt.copy() if hasattr(self, 'SimInt') else None
                        self.GF = self.GF_calc(self.SimInt)
                        self.GF_Initial = self.GF
                        self.BIC = self.BIC_calc(self.GF)
                        self.BIC_Initial = self.BIC
                    elif self.geometry == 'cylinder':
                        # Cylindrical initialization
                        if not hasattr(self, 'discretization'):
                            # Default discretization if not specified
                            self.discretization = [10] * self.layers
                            self.model_params['discretization'] = self.discretization
                            
                        self.SimCyl_SM(self.discretization)
                        self.SimInt_Initial = self.SimInt.copy() if hasattr(self, 'SimInt') else None
                        self.GF = self.GF_calc(self.SimInt)
                        self.GF_Initial = self.GF
                        self.BIC = self.BIC_calc(self.GF)
                        self.BIC_Initial = self.BIC
                    
                except Exception as e:
                    print(f"Warning: Error in {self.geometry} initialization: {str(e)}")
                    print("Data import successful, but structure initialization failed.")
                    # Continue without raising, as the import itself was successful
                
        except pd.errors.EmptyDataError:
            raise ValueError("The data file is empty or not properly formatted")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV file. Check the file format")
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")
        
        return True  # Return success

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
            self.Qr = self.Qx/np.cos(Alpha)
            
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
            Dictionary of parameters to optimize with their limits:
            {
                'trap_0_width': {'min': float, 'max': float},
                'trap_0_height': {'min': float, 'max': float},
                ...
                'DW': {'min': float, 'max': float},
                'I0': {'min': float, 'max': float},
                'Bk': {'min': float, 'max': float}
            }
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
            
            # Add parameters based on geometry
            if self.geometry == 'trapezoid':
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
            elif self.geometry == 'cylinder':
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
                    if param.startswith('trap_'):
                        parts = param.split('_')
                        trap_idx = int(parts[1])
                        param_type = parts[2]
                        limits['default'] = self.model_params['trapezoids'][trap_idx][param_type]
                    elif param.startswith('cyl_'):
                        parts = param.split('_')
                        cyl_idx = int(parts[1])
                        param_type = parts[2]
                        limits['default'] = self.model_params['cylinders'][cyl_idx][param_type]
                    else:
                        limits['default'] = getattr(self, param)
        
        # Store optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits

    def GenBounds(self, limit):
        """
        Generates bounds for optimization parameters based on a percentage limit.
        
        Parameters:
        -----------
        limit : float
            Fraction that determines how much parameters can vary (e.g., 0.1 for ±10%)
        
        Returns:
        --------
        list of tuples
            List of (min, max) bounds for each parameter
            Also sets self.bounds
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'SimPar'):
                raise AttributeError("Missing required attribute: SimPar")
                
            # Validate input parameter
            if limit is None:
                raise ValueError("limit must not be None")
                
            if not isinstance(limit, (int, float)):
                raise TypeError(f"limit must be a number, got {type(limit)}")
                
            if limit <= 0:
                raise ValueError(f"limit must be positive, got {limit}")
            
            # Generate lower and upper bounds
            lower_bounds = self.SimPar * (1 - limit)
            upper_bounds = self.SimPar * (1 + limit)
            
            # Create list of bound tuples
            self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
            
            return True
            
        except Exception as e:
            print(f"Error in GenBounds: {str(e)}")
            self.bounds = None
            return False

    def GF_calc(self, SimInt, Intensity=None):
        """
        Calculates the goodness of fit (GF) metric between experimental and simulated intensities.
        Uses log intensity.
        
        Parameters:
        -----------
        SimInt : numpy.ndarray
            Simulated intensity array 
        Intensity : numpy.ndarray, optional
            Experimental intensity array
            If None, uses self.Intensity
            
        Returns:
        --------
        float
            The goodness of fit value; lower values indicate better fit
        """
        try:
            # Determine whether to use passed parameter or class attribute
            if Intensity is None:
                # Check if required attribute exists
                if not hasattr(self, 'Intensity'):
                    raise AttributeError("Missing required attribute: Intensity")
                Intensity = self.Intensity
                
            # Check if input is valid
            if SimInt is None:
                raise ValueError("SimInt must not be None")
                
            # Check if shapes are compatible
            if Intensity.shape != SimInt.shape:
                raise ValueError(f"Shape mismatch: Intensity shape {Intensity.shape} different from SimInt shape {SimInt.shape}")
            
            # Compute the goodness of fit
            GF_M = abs(np.log(Intensity) - np.log(SimInt))
            
            # Replace NaN values with zeros
            GF_M[np.isnan(GF_M)] = 0
            
            # Sum to get the overall goodness of fit
            GF = np.sum(GF_M)
            
            return GF
            
        except Exception as e:
            print(f"Error in GF_calc: {str(e)}")
            return float('inf')  # Return infinity as a worst-case fit value

    def BIC_calc(self, GF, layers=None, numberpoints=None):
        """
        Calculates the Bayesian Information Criterion (BIC) based on goodness of fit.
        
        Parameters:
        -----------
        GF : float
            Goodness of fit value obtained from GF_calc method
        layers : int, optional
            Number of layers in the model
            If None, uses self.layers
        numberpoints : int, optional
            Number of data points
            If None, uses self.numberpoints
        
        Returns:
        --------
        float
            BIC value; lower values indicate better models considering both fit and complexity
        """
        try:
            # Determine whether to use passed parameters or class attributes
            if layers is None:
                if not hasattr(self, 'layers'):
                    raise AttributeError("Missing required attribute: layers")
                layers = self.layers
                
            if numberpoints is None:
                if not hasattr(self, 'numberpoints'):
                    raise AttributeError("Missing required attribute: numberpoints")
                numberpoints = self.numberpoints
                
            # Check if input is valid
            if GF is None or not isinstance(GF, (int, float)):
                raise ValueError(f"GF must be a numeric value, got {type(GF)}")
                
            # Check if we have sufficient data points
            if numberpoints <= 0:
                raise ValueError(f"Invalid number of data points: {numberpoints}")
            
            # Calculate number of fitting parameters
            k = 2 * layers + 2  # number of fitting parameters
            
            # Calculate BIC
            BIC = (numberpoints - k) * GF / numberpoints + k * math.log(numberpoints)
            
            return BIC
            
        except Exception as e:
            print(f"Error in BIC_calc: {str(e)}")
            return float('inf')  # Return infinity as a worst-case BIC value

    # Trapezoid-specific methods
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
        
        # Extract parameters based on geometry
        if self.geometry == 'trapezoid':
            trapezoids = self.model_params['trapezoids']
            layers = self.model_params['layers']
            
            # Create PAR array
            PAR = np.zeros([layers + 1, 2])
            for i, trap in enumerate(trapezoids):
                if i <= layers:
                    PAR[i, 0] = trap['width']
                    PAR[i, 1] = trap['height']
        elif self.geometry == 'cylinder':
            cylinders = self.model_params['cylinders']
            layers = self.model_params['layers']
            
            # Create PAR array
            PAR = np.zeros([layers + 1, 2])
            for i, cyl in enumerate(cylinders):
                if i <= layers:
                    PAR[i, 0] = cyl['radius']
                    PAR[i, 1] = cyl['height']
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
                
        return PAR

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

    # Cylindrical-specific methods
    def ConeFourierTransform(self, Discretization=None):
        """
        Fourier transform for a cone in cylindrical coordinates (Qr, Qz)
        
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
            [I0, DW, Bk] = SimPar[layers * 2 + 2:layers * 2 + 5]
            
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
            # Check if this is a cylindrical model - if so, use the cylindrical GF function
            if self.geometry == 'cylinder':
                # For cylindrical geometry, we need to convert or use Qr instead of Qx
                if not hasattr(self, 'Qr'):
                    # Convert Cartesian to cylindrical if needed
                    self.convert_Cartesian_Cylindrical()
                
                # Get discretization
                if not hasattr(self, 'discretization'):
                    self.discretization = [10] * self.layers
                
                # Create PAR array from optimization values
                temp_PAR = np.zeros((self.layers + 1, 2))
                for i, param_name in enumerate(param_names):
                    if param_name.startswith('cyl_'):
                        parts = param_name.split('_')
                        cyl_idx = int(parts[1])
                        param_type = parts[2]
                        
                        if param_type == 'radius':
                            temp_PAR[cyl_idx, 0] = optimization_values[i]
                        elif param_type == 'height':
                            temp_PAR[cyl_idx, 1] = optimization_values[i]
                    elif param_name == 'DW':
                        DW = optimization_values[i]
                    elif param_name == 'I0':
                        I0 = optimization_values[i]
                    elif param_name == 'Bk':
                        Bk = optimization_values[i]
                
                # Create SimPar array for cylindrical GF function
                SimPar = np.append(temp_PAR.ravel(), [I0, DW, Bk])
                
                # Call cylindrical GF function
                return self.SimCyl_GF(SimPar, self.layers, self.Intensity, self.Qr, self.Qz, self.discretization)
            
            # Original trapezoid GF function follows
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
        Performs differential evolution optimization for CDSAXS model fitting
        and shows before/after comparison plots.
        
        Parameters:
        -----------
        params_to_optimize : dict, optional
            Dictionary containing parameters to optimize with their bounds
            If None, uses self.model_params['optimization']
        plot_results : bool, optional
            Whether to generate any plots (master switch for all plotting)
        plot_structure : bool, optional
            Whether to plot structure comparison
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
                
            if self.geometry == 'trapezoid':
                if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                    raise AttributeError("Missing required scattering vector attributes: Qx and/or Qz")
            elif self.geometry == 'cylinder':
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
                if self.geometry == 'trapezoid':
                    self.SimInt = self.SimTrap_SM()
                elif self.geometry == 'cylinder':
                    self.SimInt = self.SimCyl_SM(self.discretization)
                    
            # Store initial simulation results
            initial_simInt = copy.deepcopy(self.SimInt)
            
            # Calculate initial goodness of fit if not already done
            if not hasattr(self, 'GF_Initial') or self.GF_Initial is None:
                self.GF_Initial = self.GF_calc(self.SimInt)
            
            # Run differential evolution optimization
            print(f"Starting optimization with {len(param_names)} parameters...")
            
            # Use the appropriate GF function based on geometry
            if self.geometry == 'trapezoid':
                result = differential_evolution(
                    self.SimTrap_GF,
                    bounds, 
                    args=(param_names, self.Intensity, self.Qx, self.Qz),
                    **optimization_params
                )
            elif self.geometry == 'cylinder':
                # For cylindrical geometry, we need to modify the optimization approach
                # to use SimCyl_GF directly with the discretization parameter
                
                # Create a wrapper function for cylindrical optimization
                def cyl_wrapper(optimization_values):
                    # Create PAR array from optimization values
                    temp_PAR = np.zeros((self.layers + 1, 2))
                    temp_DW = self.DW
                    temp_I0 = self.I0
                    temp_Bk = self.Bk
                    
                    for i, param_name in enumerate(param_names):
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
                
                result = differential_evolution(
                    cyl_wrapper,
                    bounds,
                    **optimization_params
                )
            
            # Store the optimization result
            self.optimization_result = result
            
            # Update model parameters with optimized values
            optimized_params = self.model_params.copy()
            
            # Make a deep copy of structure-specific parameters to avoid modifying the original
            if self.geometry == 'trapezoid':
                optimized_params['trapezoids'] = [trap.copy() for trap in self.model_params['trapezoids']]
            elif self.geometry == 'cylinder':
                optimized_params['cylinders'] = [cyl.copy() for cyl in self.model_params['cylinders']]
            
            for i, param_name in enumerate(param_names):
                if self.geometry == 'trapezoid' and param_name.startswith('trap_'):
                    # Parse trapezoid parameter
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]  # 'width' or 'height'
                    
                    optimized_params['trapezoids'][trap_idx][param_type] = result.x[i]
                elif self.geometry == 'cylinder' and param_name.startswith('cyl_'):
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
            if self.geometry == 'trapezoid':
                self.SimInt = self.SimTrap_SM()
            elif self.geometry == 'cylinder':
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
            Whether to plot structure comparison
        plot_grid : bool
            Whether to plot the grid of individual Qz cuts
        plot_combined : bool
            Whether to plot the combined view with all cuts
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Plot structure comparison on the same plot
        if plot_structure:
            plt.figure(figsize=(10, 6))
            
            if self.geometry == 'trapezoid':
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
            elif self.geometry == 'cylinder':
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
        
        # Set up figure for side view of stacked cylinders
        #plt.subplot(1, 2, 1)
        
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
            if self.geometry == 'trapezoid':
                q_values = self.Qz[:, i]
                q_label = f'Qx = {self.Qx[0, i]:.4f}'
            elif self.geometry == 'cylinder':
                q_values = self.Qz[:, i]
                q_label = f'Qr = {self.Qr[0, i]:.4f}'
            
            # Plot measured data
            ax.semilogy(q_values, self.Intensity[:, i], 'ko', alpha=0.7, 
                      markersize=4, label='Measured')
            
            # Plot initial simulation
            ax.semilogy(q_values, initial_simInt[:, i], 'b--', alpha=0.8, 
                     linewidth=1.5, label='Initial')
            
            # Plot optimized simulation
            ax.semilogy(q_values, self.SimInt[:, i], 'r-', alpha=1.0, 
                     linewidth=1.5, label='Optimized')
            
            # Set labels and title
            ax.set_title(f'Cut at {q_label}')
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
            if self.geometry == 'trapezoid':
                q_values = self.Qz[:, i]
            elif self.geometry == 'cylinder':
                q_values = self.Qz[:, i]
            
            # Plot measured data
            plt.semilogy(q_values, self.Intensity[:, i], 'ko', alpha=0.5, markersize=4)
            
            # Plot initial simulation
            plt.semilogy(q_values, initial_simInt[:, i], 'b--', alpha=0.5, linewidth=1.5)
            
            # Plot optimized simulation
            plt.semilogy(q_values, self.SimInt[:, i], 'r-', alpha=0.6, linewidth=1.5)
        
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
        
        # Print structure-specific parameters based on geometry
        if self.geometry == 'trapezoid':
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
        
        elif self.geometry == 'cylinder':
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
        Plots the current structure based on geometry.
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        plt.figure(figsize=(10, 6))
        
        if self.geometry == 'trapezoid':
            return self._plot_trapezoid_structure(self.model_params)
        elif self.geometry == 'cylinder':
            return self._plot_cylinder_structure(self.model_params)
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")