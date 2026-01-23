import os
import re
import math
import numpy as np
import pandas as pd
from scipy.optimize import (
    differential_evolution, 
    dual_annealing, 
    shgo, 
    basinhopping, 
    minimize
)
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import warnings
from matplotlib.colors import LogNorm
class CDSAXS_Model:
    """
    Base class for CDSAXS modeling.
    """
    
    @staticmethod
    def create_model(geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Factory method to create the appropriate model based on geometry.
        
        Parameters:
        -----------
        geometry : str
            Geometry type ('trapezoid', 'sige', or 'cylinder')
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
            
        Returns:
        --------
        CDSAXS_Model
            Appropriate model instance based on geometry
        """
        if geometry == 'trapezoid':
            from .trapezoid_model import TrapezoidModel
            return TrapezoidModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
        elif geometry == 'sige':
            from .SiGe_model import SiGeModelArray
            return SiGeModelArray(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
        elif geometry == 'cylinder':
            from .cylinder_model import CylinderModel
            return CylinderModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")
    
    def __init__(self, geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
        """
        Initialize the CDSAXS model.
        
        Parameters:
        -----------
        geometry : str
            Geometry type (e.g., 'trapezoid', 'sige', or 'cylinder')
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
        self.geometry = geometry
        self.model = model
        self.layers = layers
        self.PAR = PAR
        self.SLD = SLD
        self.DW = DW
        self.I0 = I0
        self.Bk = Bk
        self.Pitch = Pitch
        
        # Store initial values
        self.PAR_Initial = np.copy(self.PAR) if self.PAR is not None else None
        self.DW_Initial = self.DW
        self.I0_Initial = self.I0
        self.Bk_Initial = self.Bk
        
        # Set model_params if provided, or build from traditional parameters
        if model_params is not None:
            self.model_params = model_params
            self.update_traditional_from_model_params()
        else:
            self.build_model_params_from_traditional()
        
        # Create SimPar for compatibility with existing code
        if self.PAR is not None:
            self.SimPar = np.append(self.PAR.ravel(), [self.I0, self.DW, self.Bk])
            
        # Initialize callback data storage
        self._callback_data = {
            'iteration': [],
            'objective_values': [],
            'best_objective': [],
            'parameter_values': [],
            'convergence': [],
            'acceptance_flags': [],  # For dual_annealing
            'optimizer_type': None
        }
        
        # Simple callback settings
        self._callback_enabled = False
        self._callback_print_frequency = 10
    
    def build_model_params_from_traditional(self):
        """
        Build model_params dictionary from traditional parameters.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_traditional_from_model_params(self):
        """
        Update traditional parameters from model_params dictionary.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
            
            # Calculate number of valid points
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            # Process data according to geometry (implemented by subclasses)
            self.process_imported_data()
                
        except pd.errors.EmptyDataError:
            raise ValueError("The data file is empty or not properly formatted")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV file. Check the file format")
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")
        
        return True  # Return success
    
    def importCDSAXS_txtlegacy(self, intensity_file, qx_file, qz_file):
        """
        Imports CDSAXS data from three separate tab-delimited .txt files with input validation
        
        Parameters:
        -----------
        intensity_file : str
            Path to the intensity data file (tab-delimited .txt format)
        qx_file : str
            Path to the Qx data file (tab-delimited .txt format)
        qz_file : str
            Path to the Qz data file (tab-delimited .txt format)
        """
        # Check if input variables exist and are valid
        for file_path, param_name in [(intensity_file, 'intensity_file'), 
                                       (qx_file, 'qx_file'), 
                                       (qz_file, 'qz_file')]:
            if file_path is None or not isinstance(file_path, str):
                raise ValueError(f"{param_name} must be a valid file path")
            
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Read the three tab-delimited files
            # Using numpy.loadtxt for tab-delimited files
            Intensity_data = np.loadtxt(intensity_file, delimiter='\t')
            Qx_data = np.loadtxt(qx_file, delimiter='\t')
            Qz_data = np.loadtxt(qz_file, delimiter='\t')
            
            # Check if files have content
            if Intensity_data.size == 0 or Qx_data.size == 0 or Qz_data.size == 0:
                raise ValueError("One or more data files are empty")
            
            # Ensure 2D arrays (handle 1D case)
            if Intensity_data.ndim == 1:
                Intensity_data = Intensity_data.reshape(-1, 1)
            if Qx_data.ndim == 1:
                Qx_data = Qx_data.reshape(-1, 1)
            if Qz_data.ndim == 1:
                Qz_data = Qz_data.reshape(-1, 1)
            
            # Check that all files have the same dimensions
            if Intensity_data.shape != Qx_data.shape or Intensity_data.shape != Qz_data.shape:
                raise ValueError(f"Dimension mismatch: Intensity shape {Intensity_data.shape}, "
                              f"Qx shape {Qx_data.shape}, Qz shape {Qz_data.shape}. "
                              f"All files must have the same dimensions.")
            
            # Assign data to instance variables
            self.Intensity = Intensity_data.copy()
            self.Qx = Qx_data.copy()
            self.Qz = Qz_data.copy()
            
            # Convert zeros in intensity array to NaNs
            self.Intensity[self.Intensity == 0] = np.nan
            
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
            
            # Calculate number of valid points
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            # Process data according to geometry (implemented by subclasses)
            self.process_imported_data()
                
        except IOError as e:
            raise IOError(f"Error reading one or more files: {str(e)}")
        except ValueError as e:
            # Re-raise ValueError with more context if needed
            if "could not convert string to float" in str(e).lower():
                raise ValueError("Error parsing data files. Check that files contain only numeric values.")
            raise
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")
        
        return True  # Return success
    
    def importScaledCDSAXS_Data(self, datafile, format='auto'):
        """
        Imports scaled CDSAXS data that was exported by export_scaled_data function.
        This function can be added to the CDSAXS_Model base class.
        
        Parameters:
        -----------
        datafile : str
            Path to the scaled data file (CSV or NPZ format)
        format : str, optional
            File format ('csv', 'numpy', or 'auto' for auto-detection)
            
        Returns:
        --------
        bool
            True if import was successful, False otherwise
        """
        # Check if input variable exists and is valid
        if datafile is None or not isinstance(datafile, str):
            raise ValueError("Datafile must be a valid file path")
        
        # Check if file exists
        if not os.path.isfile(datafile):
            raise FileNotFoundError(f"File not found: {datafile}")
        
        # Auto-detect format if needed
        if format == 'auto':
            if datafile.lower().endswith('.csv'):
                format = 'csv'
            elif datafile.lower().endswith('.npz'):
                format = 'numpy'
            else:
                raise ValueError("Cannot auto-detect format. Please specify 'csv' or 'numpy'")
        
        try:
            if format.lower() == 'csv':
                return self._import_scaled_csv(datafile)
            elif format.lower() in ['numpy', 'npz']:
                return self._import_scaled_numpy(datafile)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'numpy'")
                
        except Exception as e:
            print(f"Error importing scaled data: {str(e)}")
            return False


    def _import_scaled_csv(self, datafile):
        """
        Import scaled data from CSV format.
        
        Parameters:
        -----------
        datafile : str
            Path to the CSV file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Read the CSV file
            data = pd.read_csv(datafile)
            
            if data.empty:
                raise ValueError("The data file is empty")
            
            # Parse column headers to extract cut information
            # Expected format: "Qz_cut_N" and "Intensity_cut_N_qx_VALUE"
            headers = data.columns.tolist()
            
            # Find unique cut indices and qx values
            cut_info = {}
            for header in headers:
                if header.startswith('Intensity_cut_'):
                    # Parse: "Intensity_cut_N_qx_VALUE"
                    match = re.match(r'Intensity_cut_(\d+)_qx_([+-]?\d*\.?\d*)', header)
                    if match:
                        cut_idx = int(match.group(1))
                        qx_value = float(match.group(2))
                        cut_info[cut_idx] = qx_value
            
            if not cut_info:
                raise ValueError("No valid intensity columns found in CSV file")
            
            # Sort cuts by index
            sorted_cuts = sorted(cut_info.keys())
            numbercuts = len(sorted_cuts)
            
            # Get data dimensions
            data_rows = len(data)
            
            # Initialize arrays
            self.Intensity = np.zeros([data_rows, numbercuts])
            self.Qz = np.zeros([data_rows, numbercuts])
            self.Qx = np.zeros([data_rows, numbercuts])
            
            # Fill arrays with data
            for i, cut_idx in enumerate(sorted_cuts):
                qz_col = f"Qz_cut_{cut_idx}"
                intensity_col = f"Intensity_cut_{cut_idx}_qx_{cut_info[cut_idx]:.4f}"
                
                if qz_col not in data.columns:
                    raise ValueError(f"Missing Qz column: {qz_col}")
                if intensity_col not in data.columns:
                    # Try to find a close match (handle floating point precision)
                    intensity_candidates = [col for col in data.columns 
                                        if col.startswith(f"Intensity_cut_{cut_idx}_qx_")]
                    if not intensity_candidates:
                        raise ValueError(f"Missing intensity column for cut {cut_idx}")
                    intensity_col = intensity_candidates[0]
                
                self.Qz[:, i] = data[qz_col].values
                self.Intensity[:, i] = data[intensity_col].values
                self.Qx[:, i] = cut_info[cut_idx]
            
            # Create Qy array with zeros
            self.Qy = np.zeros_like(self.Qx)
            
            # Calculate number of valid points and cuts
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            self.numbercuts = numbercuts
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            print(f"Successfully imported scaled CSV data:")
            print(f"  - {numbercuts} cuts")
            print(f"  - {data_rows} data points per cut")
            print(f"  - {self.numberpoints} total valid points")
            
            # Process data according to geometry (if method exists)
            if hasattr(self, 'process_imported_data'):
                self.process_imported_data()
            
            return True
            
        except Exception as e:
            print(f"Error importing scaled CSV data: {str(e)}")
            return False


    def _import_scaled_numpy(self, datafile):
        """
        Import scaled data from NumPy NPZ format.
        
        Parameters:
        -----------
        datafile : str
            Path to the NPZ file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Load the NPZ file
            data = np.load(datafile)
            
            # Check required arrays
            required_arrays = ['Intensity', 'Qz', 'Qx', 'Qy']
            for array_name in required_arrays:
                if array_name not in data:
                    raise ValueError(f"Missing required array: {array_name}")
            
            # Load arrays
            self.Intensity = data['Intensity']
            self.Qz = data['Qz']
            self.Qx = data['Qx']
            self.Qy = data['Qy']
            
            # Validate shapes
            if not all(arr.shape == self.Intensity.shape for arr in [self.Qz, self.Qx, self.Qy]):
                raise ValueError("Array shapes are inconsistent")
            
            # Calculate derived values
            self.numberpoints = np.sum(np.isfinite(self.Intensity))
            self.numbercuts = self.Intensity.shape[1]
            
            # Check if we have valid data
            if self.numberpoints == 0:
                raise ValueError("No valid data points found after processing")
            
            print(f"Successfully imported scaled NumPy data:")
            print(f"  - {self.numbercuts} cuts")
            print(f"  - {self.Intensity.shape[0]} data points per cut")
            print(f"  - {self.numberpoints} total valid points")
            
            # Process data according to geometry (if method exists)
            if hasattr(self, 'process_imported_data'):
                self.process_imported_data()
            
            return True
            
        except Exception as e:
            print(f"Error importing scaled NumPy data: {str(e)}")
            return False
    
    def process_imported_data(self):
        """
        Process imported data according to geometry.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def initialize_optimization_params(self, param_limits=None):
        """
        Initialize optimization parameters with bounds.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

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

    def CDSAXS_DiffEvolution(self, params_to_optimize=None, plot_results=True, 
                             plot_structure=True, plot_grid=True, plot_combined=True, **kwargs):
        """
        Performs differential evolution optimization for CDSAXS model fitting.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _plot_qzcut_grid(self, initial_simInt):
        """
        Plot a grid of QzCut comparisons with both initial and optimized results.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _plot_qzcut_combined(self, initial_simInt):
        """
        Plot all QzCuts on one axis with both initial and optimized results.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    
    def print_parameter_changes(self, initial_model_params=None, boundary_threshold=1.0, use_colors=True):
        """
        Print a table of parameter changes from optimization with bounds and color coding.
        Now includes original and final GF values at the top.
        
        Parameters:
        -----------
        initial_model_params : dict, optional
            Model parameters before optimization
            If None, uses self._initial_model_params if available
        boundary_threshold : float, optional
            Percentage threshold for boundary warning (default: 1.0%)
        use_colors : bool, optional
            Whether to use color coding (default: True)
        """
        if initial_model_params is None:
            if hasattr(self, '_initial_model_params'):
                initial_model_params = self._initial_model_params
            else:
                print("Error: No initial parameters available for comparison.")
                print("Either provide initial_model_params or run optimization first.")
                return
        
        # Color codes for terminal output
        if use_colors:
            RED = '\033[91m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            BLUE = '\033[94m'
            MAGENTA = '\033[95m'
            CYAN = '\033[96m'
            RESET = '\033[0m'
            BOLD = '\033[1m'
        else:
            RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = RESET = BOLD = ''
        
        # Get optimization parameters to extract bounds
        optimization_params = getattr(self, 'model_params', {}).get('optimization', {})
        if not optimization_params:
            # Try to get from stored optimization info
            optimization_params = getattr(self, 'mcmc_param_info', {})
        
        print(f"\n{BOLD}Parameter Changes with Optimization Bounds:{RESET}")
        print("=" * 80)
        
        # Add GF comparison at the top
        if hasattr(self, 'GF_Initial') and hasattr(self, 'GF'):
            initial_gf = self.GF_Initial
            final_gf = self.GF
            improvement = initial_gf - final_gf
            improvement_pct = (improvement / initial_gf) * 100 if initial_gf > 0 else 0
            
            print(f"{BOLD}Goodness of Fit Summary:{RESET}")
            print(f"{'Original GF:':<15} {CYAN}{initial_gf:<12.6f}{RESET}")
            print(f"{'Final GF:':<15} {GREEN if improvement > 0 else RED}{final_gf:<12.6f}{RESET}")
            print(f"{'Improvement:':<15} {GREEN if improvement > 0 else RED}{improvement:<12.6f}{RESET} ({improvement_pct:+.2f}%)")
            
            # Add BIC if available
            if hasattr(self, 'BIC_Initial') and hasattr(self, 'BIC'):
                initial_bic = self.BIC_Initial
                final_bic = self.BIC
                bic_improvement = initial_bic - final_bic
                bic_improvement_pct = (bic_improvement / initial_bic) * 100 if initial_bic > 0 else 0
                
                print(f"{'Original BIC:':<15} {CYAN}{initial_bic:<12.6f}{RESET}")
                print(f"{'Final BIC:':<15} {GREEN if bic_improvement > 0 else RED}{final_bic:<12.6f}{RESET}")
                print(f"{'BIC Improvement:':<15} {GREEN if bic_improvement > 0 else RED}{bic_improvement:<12.6f}{RESET} ({bic_improvement_pct:+.2f}%)")
            
            print("=" * 80)
        
        # Parameter change table
        print(f"{'Parameter':<20} {'Initial':<12} {'Lower':<12} {'Optimized':<12} {'Upper':<12}")
        print("-" * 80)
        
        try:
            # Print trapezoid parameters
            if self.geometry in ['trapezoid', 'sige']:
                initial_traps = initial_model_params.get('trapezoids', [])
                current_traps = self.model_params.get('trapezoids', [])
                
                max_traps = max(len(initial_traps), len(current_traps))
                
                for i in range(max_traps):
                    if i < len(initial_traps) and i < len(current_traps):
                        initial_trap = initial_traps[i]
                        current_trap = current_traps[i]
                        
                        # Print width
                        if 'width' in initial_trap and 'width' in current_trap:
                            param_name = f'trap_{i}_width'
                            initial_val = initial_trap['width']
                            current_val = current_trap['width']
                            
                            # Get bounds and apply color coding
                            param_info = optimization_params.get(param_name, {})
                            lower_bound = param_info.get('min')
                            upper_bound = param_info.get('max')
                            colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                            boundary_threshold, use_colors)
                            
                            lower_str = f"{lower_bound:.4f}" if lower_bound is not None else "N/A"
                            upper_str = f"{upper_bound:.4f}" if upper_bound is not None else "N/A"
                            
                            print(f"Trap {i} Width{'':<8} {initial_val:<12.4f} {lower_str:<12} "
                                f"{colored_value:<12} {upper_str:<12}")
                        
                        # Print height
                        if 'height' in initial_trap and 'height' in current_trap:
                            param_name = f'trap_{i}_height'
                            initial_val = initial_trap['height']
                            current_val = current_trap['height']
                            
                            param_info = optimization_params.get(param_name, {})
                            lower_bound = param_info.get('min')
                            upper_bound = param_info.get('max')
                            colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                            boundary_threshold, use_colors)
                            
                            lower_str = f"{lower_bound:.4f}" if lower_bound is not None else "N/A"
                            upper_str = f"{upper_bound:.4f}" if upper_bound is not None else "N/A"
                            
                            print(f"Trap {i} Height{'':<7} {initial_val:<12.4f} {lower_str:<12} "
                                f"{colored_value:<12} {upper_str:<12}")
                        
                        # Print twidth (if present - used by SiGe model)
                        if 'twidth' in initial_trap or 'twidth' in current_trap:
                            # Check if twidth exists in both or if we should print anyway
                            if 'twidth' in initial_trap and 'twidth' in current_trap:
                                param_name = f'trap_{i}_twidth'
                                initial_val = initial_trap['twidth']
                                current_val = current_trap['twidth']
                                
                                # Skip if twidth is None (not all trapezoids have twidth)
                                if initial_val is None or current_val is None:
                                    continue
                                
                                param_info = optimization_params.get(param_name, {})
                                lower_bound = param_info.get('min')
                                upper_bound = param_info.get('max')
                                colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                                boundary_threshold, use_colors)
                                
                                lower_str = f"{lower_bound:.4f}" if lower_bound is not None else "N/A"
                                upper_str = f"{upper_bound:.4f}" if upper_bound is not None else "N/A"
                                
                                print(f"Trap {i} TWidth{'':<6} {initial_val:<12.4f} {lower_str:<12} "
                                    f"{colored_value:<12} {upper_str:<12}")
            
            # Print cylinder parameters
            elif self.geometry == 'cylinder':
                initial_cyls = initial_model_params.get('cylinders', [])
                current_cyls = self.model_params.get('cylinders', [])
                
                max_cyls = max(len(initial_cyls), len(current_cyls))
                
                for i in range(max_cyls):
                    if i < len(initial_cyls) and i < len(current_cyls):
                        initial_cyl = initial_cyls[i]
                        current_cyl = current_cyls[i]
                        
                        # Print radius
                        if 'radius' in initial_cyl and 'radius' in current_cyl:
                            param_name = f'cyl_{i}_radius'
                            initial_val = initial_cyl['radius']
                            current_val = current_cyl['radius']
                            
                            param_info = optimization_params.get(param_name, {})
                            lower_bound = param_info.get('min')
                            upper_bound = param_info.get('max')
                            colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                            boundary_threshold, use_colors)
                            
                            lower_str = f"{lower_bound:.4f}" if lower_bound is not None else "N/A"
                            upper_str = f"{upper_bound:.4f}" if upper_bound is not None else "N/A"
                            
                            print(f"Cyl {i} Radius{'':<8} {initial_val:<12.4f} {lower_str:<12} "
                                f"{colored_value:<12} {upper_str:<12}")
                        
                        # Print height
                        if 'height' in initial_cyl and 'height' in current_cyl:
                            param_name = f'cyl_{i}_height'
                            initial_val = initial_cyl['height']
                            current_val = current_cyl['height']
                            
                            param_info = optimization_params.get(param_name, {})
                            lower_bound = param_info.get('min')
                            upper_bound = param_info.get('max')
                            colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                            boundary_threshold, use_colors)
                            
                            lower_str = f"{lower_bound:.4f}" if lower_bound is not None else "N/A"
                            upper_str = f"{upper_bound:.4f}" if upper_bound is not None else "N/A"
                            
                            print(f"Cyl {i} Height{'':<8} {initial_val:<12.4f} {lower_str:<12} "
                                f"{colored_value:<12} {upper_str:<12}")
            
            # Print global parameters (DW, I0)
            for param in ['DW', 'I0']:
                if param in initial_model_params and param in self.model_params:
                    initial_val = initial_model_params[param]
                    current_val = self.model_params[param]
                    
                    param_info = optimization_params.get(param, {})
                    lower_bound = param_info.get('min')
                    upper_bound = param_info.get('max')
                    colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                    boundary_threshold, use_colors)
                    
                    lower_str = f"{lower_bound:.6f}" if lower_bound is not None else "N/A"
                    upper_str = f"{upper_bound:.6f}" if upper_bound is not None else "N/A"
                    
                    print(f"{param:<20} {initial_val:<12.6f} {lower_str:<12} "
                        f"{colored_value:<12} {upper_str:<12}")
            
            # Print background parameters
            initial_bk = initial_model_params.get('Bk')
            current_bk = self.model_params.get('Bk')
            
            if initial_bk is not None and current_bk is not None:
                # Handle array background
                if isinstance(initial_bk, (list, np.ndarray)) and isinstance(current_bk, (list, np.ndarray)):
                    initial_bk = np.array(initial_bk)
                    current_bk = np.array(current_bk)
                    
                    for i in range(min(len(initial_bk), len(current_bk))):
                        param_name = f'Bk_{i}'
                        initial_val = initial_bk[i]
                        current_val = current_bk[i]
                        
                        param_info = optimization_params.get(param_name, {})
                        lower_bound = param_info.get('min')
                        upper_bound = param_info.get('max')
                        colored_value = self._get_colored_value(current_val, lower_bound, upper_bound, 
                                                        boundary_threshold, use_colors)
                        
                        lower_str = f"{lower_bound:.6f}" if lower_bound is not None else "N/A"
                        upper_str = f"{upper_bound:.6f}" if upper_bound is not None else "N/A"
                        
                        print(f"Bk_{i:<17} {initial_val:<12.6f} {lower_str:<12} "
                            f"{colored_value:<12} {upper_str:<12}")
                
                # Handle scalar background
                elif not isinstance(initial_bk, (list, np.ndarray)) and not isinstance(current_bk, (list, np.ndarray)):
                    param_info = optimization_params.get('Bk', {})
                    lower_bound = param_info.get('min')
                    upper_bound = param_info.get('max')
                    colored_value = self._get_colored_value(current_bk, lower_bound, upper_bound, 
                                                    boundary_threshold, use_colors)
                    
                    lower_str = f"{lower_bound:.6f}" if lower_bound is not None else "N/A"
                    upper_str = f"{upper_bound:.6f}" if upper_bound is not None else "N/A"
                    
                    print(f"Bk{'':<18} {initial_bk:<12.6f} {lower_str:<12} "
                        f"{colored_value:<12} {upper_str:<12}")
        
        except Exception as e:
            print(f"Error printing parameter changes: {str(e)}")
            print("Falling back to basic display...")
            
            # Basic fallback - just show current parameters
            print("Current parameters:")
            for key, value in self.model_params.items():
                if isinstance(value, (int, float)):
                    print(f"{key:<20} {value:<12.6f}")
        
        print("=" * 80)
        
        if use_colors:
            print(f"\n{GREEN}Green{RESET}: Parameter safely within bounds")
            print(f"{RED}Red{RESET}: Parameter within {boundary_threshold}% of optimization boundary") 
            print(f"{YELLOW}Yellow{RESET}: No bounds information available")


    def _get_colored_value(self, value, lower_bound, upper_bound, threshold_percent=1.0, use_colors=True):
        """
        Get colored value string based on proximity to bounds.
        
        Parameters:
        -----------
        value : float
            Current parameter value
        lower_bound : float or None
            Lower optimization bound
        upper_bound : float or None
            Upper optimization bound  
        threshold_percent : float
            Percentage threshold for boundary warning
        use_colors : bool
            Whether to use color codes
            
        Returns:
        --------
        str
            Colored value string
        """
        if use_colors:
            RED = '\033[91m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RESET = '\033[0m'
        else:
            RED = GREEN = YELLOW = RESET = ''
        
        # Format the value
        if abs(value) < 1:
            value_str = f"{value:.6f}"
        else:
            value_str = f"{value:.4f}"
        
        if lower_bound is None or upper_bound is None:
            return f"{YELLOW}{value_str}{RESET}"
        
        # Calculate threshold distances
        bound_range = upper_bound - lower_bound
        threshold_distance = bound_range * (threshold_percent / 100)
        
        # Check proximity to bounds
        distance_to_lower = value - lower_bound
        distance_to_upper = upper_bound - value
        
        if distance_to_lower <= threshold_distance or distance_to_upper <= threshold_distance:
            return f"{RED}{value_str}{RESET}"
        else:
            return f"{GREEN}{value_str}{RESET}"
    def plot_structure(self):
        """
        Plots the current structure.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def simulate_structure(self, *args, **kwargs):
        """
        Abstract method to simulate the structure intensity.
        To be implemented by subclasses with their specific simulation methods.
        
        Returns:
        --------
        numpy.ndarray
            The simulated intensity (also sets self.SimInt)
        """
      
            
        self.SimInt = self.simulate_structure()
      
        gf = self.GF_calc(self.SimInt)

        
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def parameter_sweep_1d(self, sweep_param, sweep_range, n_points=20, 
                            exclude_from_fit=None, plot_results=True, 
                            figsize=(10, 6), save_results=False, filename=None,
                            optimization_kwargs=None, verbose=True):
        """
        Clean version of parameter_sweep_1d with minimal progress bar updates.
        """
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {'maxiter': 30, 'popsize': 10}
        
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
        
        # Setup progress bar with fixed description (no updates)
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
                
                # Initialize variables for this iteration
                gf = float('inf')
                bic = float('inf')
                converged = False
                
                if not opt_params:
                    # No parameters to optimize, just calculate GF
                    try:
                        # Handle both geometries correctly
                        if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                            sim_result = self.simulate_structure(self.discretization)
                        else:
                            sim_result = self.simulate_structure()
                        
                        if sim_result is not None:
                            gf = self.GF_calc(sim_result)
                            bic = self.BIC_calc(gf)
                            converged = True
                            
                    except Exception as e:
                        if verbose and i < 3:  # Only print errors for first few points
                            print(f"Warning: Simulation failed at {sweep_param}={value}")
                            
                else:
                    # Run optimization with suppressed output
                    try:
                        self.CDSAXS_DiffEvolution(
                            params_to_optimize=opt_params,
                            plot_results=False,  # Suppress plots during sweep
                            verbose=False,       # Suppress optimization output
                            **optimization_kwargs
                        )
                        
                        if hasattr(self, 'GF') and hasattr(self, 'BIC'):
                            gf = self.GF
                            bic = self.BIC
                            converged = True
                            
                    except Exception as e:
                        if verbose and i < 3:  # Only print errors for first few points
                            print(f"Warning: Optimization failed at {sweep_param}={value}: {e}")
                
                # Store results
                results['gf_values'].append(gf)
                results['bic_values'].append(bic)
                results['optimized_params'].append(copy.deepcopy(self.model_params))
                results['convergence_flags'].append(converged)
                
                # No progress bar updates - just let it show the percentage
                    
            except Exception as e:
                if verbose and i < 3:  # Only print errors for first few points
                    print(f"Error at {sweep_param}={value}: {str(e)}")
                # Still store something to maintain array lengths
                results['gf_values'].append(float('inf'))
                results['bic_values'].append(float('inf'))
                results['optimized_params'].append(None)
                results['convergence_flags'].append(False)
        
        if verbose and hasattr(pbar, 'close'):
            pbar.close()
        
        # Verify results before proceeding
        expected_length = len(sweep_values)
        actual_length = len(results['gf_values'])
        
        if actual_length != expected_length:
            print(f"WARNING: Results length mismatch. Expected {expected_length}, got {actual_length}")
            while len(results['gf_values']) < expected_length:
                results['gf_values'].append(float('inf'))
                results['bic_values'].append(float('inf'))
                results['optimized_params'].append(None)
                results['convergence_flags'].append(False)
        
        # Print summary with best results
        if verbose:
            self._print_sweep_summary_1d(results)
        
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
    
    
    
    def parameter_sweep_2d_clean(self, sweep_params, sweep_ranges, n_points=(10, 10),
                      exclude_from_fit=None, plot_results=True, 
                      figsize=(10, 8), save_results=False, filename=None,
                      optimization_kwargs=None, verbose=True, metric='GF'):
        """
        Clean version of 2D parameter sweep with minimal progress bar updates.
        """
        import numpy as np
        import copy
        from tqdm import tqdm
        
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {'maxiter': 20, 'popsize': 8}
        
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
        
        # Setup progress bar with fixed description (no updates)
        total_points = n_points[0] * n_points[1]
        if verbose:
            pbar = tqdm(total=total_points, desc=f"2D Sweep: {sweep_params[0]} vs {sweep_params[1]}")
        
        point_count = 0
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                point_count += 1
                
                # Initialize variables for this iteration
                gf = float('inf')
                bic = float('inf')
                converged = False
                
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
                        try:
                            if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                                sim_result = self.simulate_structure(self.discretization)
                            else:
                                sim_result = self.simulate_structure()
                            
                            if sim_result is not None:
                                gf = self.GF_calc(sim_result)
                                bic = self.BIC_calc(gf)
                                converged = True
                                
                        except Exception as e:
                            if verbose and total_points <= 25 and point_count <= 3:
                                print(f"Warning: Simulation failed at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}")
                    else:
                        # Run optimization with suppressed output
                        try:
                            self.CDSAXS_DiffEvolution(
                                params_to_optimize=opt_params,
                                plot_results=False,
                                verbose=False,
                                **optimization_kwargs
                            )
                            
                            if hasattr(self, 'GF') and hasattr(self, 'BIC'):
                                gf = self.GF
                                bic = self.BIC
                                converged = True
                                
                        except Exception as e:
                            if verbose and total_points <= 25 and point_count <= 3:
                                print(f"Warning: Optimization failed at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}: {e}")
                    
                    # Store results
                    results['gf_matrix'][j, i] = gf
                    results['bic_matrix'][j, i] = bic
                    results['optimized_params'][j][i] = copy.deepcopy(self.model_params)
                    results['convergence_matrix'][j, i] = converged
                    
                    # Just update progress, no description changes
                    if verbose:
                        pbar.update(1)
                        
                except Exception as e:
                    if verbose and point_count <= 3:
                        print(f"Error at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}: {str(e)}")
                    
                    # Store failed results
                    results['gf_matrix'][j, i] = float('inf')
                    results['bic_matrix'][j, i] = float('inf')
                    results['optimized_params'][j][i] = None
                    results['convergence_matrix'][j, i] = False
                    
                    if verbose:
                        pbar.update(1)
        
        if verbose:
            pbar.close()
        
        # Print summary with best results
        if verbose:
            self._print_sweep_summary_2d(results)
        
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
            if hasattr(self, 'discretization'):
                # Cylinder models need discretization parameter
                self.SimInt = self.simulate_structure(self.discretization)
            else:
                # Trapezoid models don't need discretization
                self.SimInt = self.simulate_structure()
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
  
    def _set_parameter_value(self, param_name, value):
        """Set a parameter value in the model."""
        if param_name.startswith('trap_'):
            # Trapezoid parameter
            parts = param_name.split('_')
            trap_idx = int(parts[1])
            param_type = parts[2]
            self.model_params['trapezoids'][trap_idx][param_type] = value
        elif param_name.startswith('cyl_'):
            # Cylinder parameter
            parts = param_name.split('_')
            cyl_idx = int(parts[1])
            param_type = parts[2]
            self.model_params['cylinders'][cyl_idx][param_type] = value
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
    
    def PlotQzCut(self, cut_index=None, SimInt=None, log_scale='yes'):
        """
        Plots intensity vs Qz for specific cuts (Qx for trapezoid, Qr for cylinder)
        
        Parameters:
        -----------
        cut_index : int or list or None, optional
            Index or indices of the cut(s) to plot
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
        
        # Determine the appropriate Q-component for labeling
        q_component = self.Qx if self.geometry in ['trapezoid', 'sige'] else self.Qr
        q_label = 'Qx' if self.geometry in ['trapezoid', 'sige'] else 'Qr'
        
        # Plot each cut
        for i, (ax, idx) in enumerate(zip(axes, cut_indices)):
            # Check if the index is valid
            if idx < 0 or idx >= self.Intensity.shape[1]:
                ax.text(0.5, 0.5, f"Invalid cut index: {idx}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get Qz values for the selected cut
            qz_values = self.Qz[:, idx]
            q_value = q_component[0, idx]
            
            # Plot measured intensity
            measured_line, = ax.plot(qz_values, self.Intensity[:, idx], 'o-', 
                                color='grey', alpha=0.7, label='Measured')
            
            # Plot simulated intensity if available
            if SimInt is not None:
                simulated_line, = ax.plot(qz_values, SimInt[:, idx], 'b-', label='Simulated')
            elif hasattr(self, 'SimInt') and self.SimInt is not None:
                simulated_line, = ax.plot(qz_values, self.SimInt[:, idx], 'b-', label='Simulated')
            
            # Set logarithmic scale if requested
            if log_scale.lower() == 'yes':
                ax.set_yscale('log')
            
            # Set labels and title
            ax.set_title(f'Cut at {q_label} = {q_value:.4f}')
            ax.set_xlabel('Qz (Å$^{-1}$)')
            ax.set_ylabel('Intensity (counts)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(cut_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Return a single axis for a single plot, or list of axes for multiple plots
        return axes[0] if len(axes) == 1 else axes
    
    def export_scaled_data(self, output_file, format='csv', scaling_factor=None, 
                          data_subset=None, metadata=None):
        """
        Export the current intensity data with optional scaling and subsetting.
        
        Parameters:
        -----------
        output_file : str
            Path for the output file (without extension)
        format : str, optional
            Output format ('csv' or 'numpy'). Default: 'csv'
        scaling_factor : float or numpy.ndarray, optional
            Factor(s) to scale the intensity data
            If array, must match the number of cuts
        data_subset : dict, optional
            Dictionary specifying data subset to export
            e.g., {'qz_range': (min_qz, max_qz), 'cuts': [0, 2, 4]}
        metadata : dict, optional
            Additional metadata to include in the export
            
        Returns:
        --------
        str
            Path of the exported file
        """
        # Check if required attributes exist
        if not hasattr(self, 'Intensity') or not hasattr(self, 'Qz'):
            raise AttributeError("Missing required attributes: Intensity and/or Qz")
        
        # Prepare data for export
        intensity_data = self.Intensity.copy()
        qz_data = self.Qz.copy()
        
        # Get Q-component data based on geometry
        if self.geometry in ['trapezoid', 'sige']:
            q_data = self.Qx.copy()
            q_label = 'qx'
        else:
            q_data = self.Qr.copy()
            q_label = 'qr'
        
        # Apply scaling if provided
        if scaling_factor is not None:
            if np.isscalar(scaling_factor):
                intensity_data *= scaling_factor
            else:
                # Array scaling - must match number of cuts
                if len(scaling_factor) != intensity_data.shape[1]:
                    raise ValueError(f"Scaling factor array length ({len(scaling_factor)}) "
                                   f"must match number of cuts ({intensity_data.shape[1]})")
                intensity_data *= scaling_factor[np.newaxis, :]
        
        # Apply data subset if provided
        if data_subset is not None:
            # Subset by Qz range
            if 'qz_range' in data_subset:
                qz_min, qz_max = data_subset['qz_range']
                valid_rows = []
                for col in range(qz_data.shape[1]):
                    col_mask = (qz_data[:, col] >= qz_min) & (qz_data[:, col] <= qz_max)
                    if col == 0:
                        valid_rows = col_mask
                    else:
                        valid_rows = valid_rows | col_mask
                
                intensity_data = intensity_data[valid_rows, :]
                qz_data = qz_data[valid_rows, :]
                q_data = q_data[valid_rows, :]
            
            # Subset by specific cuts
            if 'cuts' in data_subset:
                cut_indices = data_subset['cuts']
                intensity_data = intensity_data[:, cut_indices]
                qz_data = qz_data[:, cut_indices]
                q_data = q_data[:, cut_indices]
        
        # Export based on format
        if format.lower() == 'csv':
            return self._export_csv(output_file, intensity_data, qz_data, q_data, q_label, metadata)
        elif format.lower() in ['numpy', 'npz']:
            return self._export_numpy(output_file, intensity_data, qz_data, q_data, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, output_file, intensity_data, qz_data, q_data, q_label, metadata):
        """Export data in CSV format."""
        import pandas as pd
        
        # Create column names and data
        columns = []
        data_dict = {}
        
        for i in range(intensity_data.shape[1]):
            q_value = q_data[0, i]  # Assuming q is constant per cut
            
            qz_col = f"Qz_cut_{i}"
            intensity_col = f"Intensity_cut_{i}_{q_label}_{q_value:.4f}"
            
            columns.extend([qz_col, intensity_col])
            data_dict[qz_col] = qz_data[:, i]
            data_dict[intensity_col] = intensity_data[:, i]
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Add timestamp to filename

        filename = f"{output_file}_.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Add metadata as comments if provided
        if metadata:
            self._add_csv_metadata(filename, metadata)
        
        print(f"Data exported to: {filename}")
        print(f"  - {intensity_data.shape[1]} cuts")
        print(f"  - {intensity_data.shape[0]} data points per cut")
        
        return filename
    
    def _export_numpy(self, output_file, intensity_data, qz_data, q_data, metadata):
        """Export data in NumPy format."""
        # Create Qy data (zeros for both geometries in this context)
        qy_data = np.zeros_like(q_data)
        
        # Prepare data dictionary
        data_dict = {
            'Intensity': intensity_data,
            'Qz': qz_data,
            'Qy': qy_data
        }
        
        # Add the appropriate Q component
        if self.geometry in ['trapezoid', 'sige']:
            data_dict['Qx'] = q_data
        else:
            data_dict['Qr'] = q_data
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    data_dict[f'metadata_{key}'] = value
        
        # Add timestamp to filename
    
        filename = f"{output_file}.npz"
        
        # Save to NPZ
        np.savez_compressed(filename, **data_dict)
        
        print(f"Data exported to: {filename}")
        print(f"  - {intensity_data.shape[1]} cuts")
        print(f"  - {intensity_data.shape[0]} data points per cut")
        
        return filename
    
    def _add_csv_metadata(self, filename, metadata):
        """Add metadata as comments to CSV file."""
        # Read existing content
        with open(filename, 'r') as f:
            content = f.read()
        
        # Prepare metadata comments
        metadata_lines = ["# Metadata:"]
        for key, value in metadata.items():
            metadata_lines.append(f"# {key}: {value}")
        metadata_lines.append("# ")  # Empty line before data
        
        # Write metadata + content
        with open(filename, 'w') as f:
            f.write('\n'.join(metadata_lines) + '\n')
            f.write(content)
    
    def validate_model_parameters(self, verbose=True):
        """
        Validate the current model parameters for consistency and physical reasonableness.
        
        Parameters:
        -----------
        verbose : bool, optional
            Whether to print detailed validation results. Default: True
            
        Returns:
        --------
        dict
            Dictionary with validation results and any issues found
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'parameter_summary': {}
        }
        
        try:
            # Check if model_params exists
            if not hasattr(self, 'model_params'):
                validation_results['errors'].append("No model_params found")
                validation_results['valid'] = False
                return validation_results
            
            # Validate basic structure
            required_keys = ['layers', 'DW', 'I0', 'Bk']
            for key in required_keys:
                if key not in self.model_params:
                    validation_results['errors'].append(f"Missing required parameter: {key}")
                    validation_results['valid'] = False
            
            # Validate geometry-specific parameters
            if self.geometry in ['trapezoid', 'sige']:
                self._validate_trapezoid_params(validation_results)
            elif self.geometry == 'cylinder':
                self._validate_cylinder_params(validation_results)
            
            # Validate global parameters
            self._validate_global_params(validation_results)
            
            # Generate parameter summary
            self._generate_parameter_summary(validation_results)
            
            if verbose:
                self._print_validation_results(validation_results)
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['valid'] = False
        
        return validation_results
    
    def _validate_trapezoid_params(self, validation_results):
        """Validate trapezoid-specific parameters."""
        if 'trapezoids' not in self.model_params:
            validation_results['errors'].append("Missing trapezoids parameter")
            return
        
        trapezoids = self.model_params['trapezoids']
        layers = self.model_params['layers']
        
        # Check number of trapezoids
        if len(trapezoids) != layers + 1:
            validation_results['errors'].append(
                f"Number of trapezoids ({len(trapezoids)}) should be layers + 1 ({layers + 1})"
            )
        
        # Validate each trapezoid
        for i, trap in enumerate(trapezoids):
            if 'width' not in trap:
                validation_results['errors'].append(f"Trapezoid {i} missing width")
                continue
            
            width = trap['width']
            if width <= 0:
                validation_results['errors'].append(f"Trapezoid {i} width ({width}) must be positive")
            
            # Check height for non-top trapezoids
            if i < layers:
                if 'height' not in trap:
                    validation_results['errors'].append(f"Trapezoid {i} missing height")
                    continue
                
                height = trap['height']
                if height <= 0:
                    validation_results['warnings'].append(f"Trapezoid {i} height ({height}) should be positive")
        
        # Check for tapering consistency
        widths = [trap['width'] for trap in trapezoids]
        if len(widths) > 1:
            if not all(w1 >= w2 for w1, w2 in zip(widths[:-1], widths[1:])):
                validation_results['warnings'].append("Trapezoid widths are not monotonically decreasing")
    
    def _validate_cylinder_params(self, validation_results):
        """Validate cylinder-specific parameters."""
        if 'cylinders' not in self.model_params:
            validation_results['errors'].append("Missing cylinders parameter")
            return
        
        cylinders = self.model_params['cylinders']
        layers = self.model_params['layers']
        
        # Check number of cylinders
        if len(cylinders) != layers + 1:
            validation_results['errors'].append(
                f"Number of cylinders ({len(cylinders)}) should be layers + 1 ({layers + 1})"
            )
        
        # Validate each cylinder
        for i, cyl in enumerate(cylinders):
            if 'radius' not in cyl:
                validation_results['errors'].append(f"Cylinder {i} missing radius")
                continue
            
            radius = cyl['radius']
            if radius <= 0:
                validation_results['errors'].append(f"Cylinder {i} radius ({radius}) must be positive")
            
            # Check height for non-top cylinders
            if i < layers:
                if 'height' not in cyl:
                    validation_results['errors'].append(f"Cylinder {i} missing height")
                    continue
                
                height = cyl['height']
                if height <= 0:
                    validation_results['warnings'].append(f"Cylinder {i} height ({height}) should be positive")
        
        # Check for tapering consistency
        radii = [cyl['radius'] for cyl in cylinders]
        if len(radii) > 1:
            if not all(r1 >= r2 for r1, r2 in zip(radii[:-1], radii[1:])):
                validation_results['warnings'].append("Cylinder radii are not monotonically decreasing")
    
    def _validate_global_params(self, validation_results):
        """Validate global parameters."""
        # Validate DW (Debye-Waller factor)
        dw = self.model_params.get('DW', None)
        if dw is not None:
            if dw < 0:
                validation_results['errors'].append(f"DW ({dw}) must be non-negative")
            elif dw > 10:
                validation_results['warnings'].append(f"DW ({dw}) is unusually large")
        
        # Validate I0 (intensity scaling)
        i0 = self.model_params.get('I0', None)
        if i0 is not None:
            if i0 <= 0:
                validation_results['errors'].append(f"I0 ({i0}) must be positive")
        
        # Validate background
        bk = self.model_params.get('Bk', None)
        if bk is not None:
            if np.isscalar(bk):
                if bk < 0:
                    validation_results['warnings'].append(f"Background ({bk}) is negative")
            else:
                # Array background
                if np.any(np.array(bk) < 0):
                    validation_results['warnings'].append("Some background values are negative")
    
    def _generate_parameter_summary(self, validation_results):
        """Generate a summary of model parameters."""
        summary = {}
        
        if hasattr(self, 'model_params'):
            summary['geometry'] = self.geometry
            summary['layers'] = self.model_params.get('layers', 'Unknown')
            summary['DW'] = self.model_params.get('DW', 'Unknown')
            summary['I0'] = self.model_params.get('I0', 'Unknown')
            summary['Bk'] = self.model_params.get('Bk', 'Unknown')
            
            if self.geometry in ['trapezoid', 'sige'] and 'trapezoids' in self.model_params:
                widths = [trap.get('width', 0) for trap in self.model_params['trapezoids']]
                heights = [trap.get('height', 0) for trap in self.model_params['trapezoids'][:-1]]
                summary['widths'] = widths
                summary['heights'] = heights
                summary['total_height'] = sum(heights)
                summary['aspect_ratio'] = max(widths) / summary['total_height'] if summary['total_height'] > 0 else float('inf')
                
                # Include twidth if present (used by SiGe model)
                twidths = [trap.get('twidth') for trap in self.model_params['trapezoids'] if 'twidth' in trap]
                if twidths:
                    summary['twidths'] = twidths
            
            elif self.geometry == 'cylinder' and 'cylinders' in self.model_params:
                radii = [cyl.get('radius', 0) for cyl in self.model_params['cylinders']]
                heights = [cyl.get('height', 0) for cyl in self.model_params['cylinders'][:-1]]
                summary['radii'] = radii
                summary['heights'] = heights
                summary['total_height'] = sum(heights)
                summary['aspect_ratio'] = max(radii) / summary['total_height'] if summary['total_height'] > 0 else float('inf')
        
        validation_results['parameter_summary'] = summary
    
    def _print_validation_results(self, validation_results):
        """Print validation results in a formatted way."""
        print("\n" + "="*60)
        print("MODEL PARAMETER VALIDATION")
        print("="*60)
        
        # Print overall status
        status = "VALID" if validation_results['valid'] else "INVALID"
        print(f"Status: {status}")
        
        # Print errors
        if validation_results['errors']:
            print(f"\nERRORS ({len(validation_results['errors'])}):")
            for error in validation_results['errors']:
                print(f"  ❌ {error}")
        
        # Print warnings
        if validation_results['warnings']:
            print(f"\nWARNINGS ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Print parameter summary
        if validation_results['parameter_summary']:
            print(f"\nPARAMETER SUMMARY:")
            summary = validation_results['parameter_summary']
            print(f"  Geometry: {summary.get('geometry', 'Unknown')}")
            print(f"  Layers: {summary.get('layers', 'Unknown')}")
            print(f"  DW: {summary.get('DW', 'Unknown')}")
            print(f"  I0: {summary.get('I0', 'Unknown')}")
            print(f"  Background: {summary.get('Bk', 'Unknown')}")
            
            if 'total_height' in summary:
                print(f"  Total Height: {summary['total_height']:.2f}")
            if 'aspect_ratio' in summary:
                ratio = summary['aspect_ratio']
                if ratio != float('inf'):
                    print(f"  Aspect Ratio: {ratio:.2f}")
        
        if not validation_results['errors'] and not validation_results['warnings']:
            print("\n✅ No issues found!")
        
        print("="*60)
    
    def copy_model(self, deep=True):
        """
        Create a copy of the current model.
        
        Parameters:
        -----------
        deep : bool, optional
            Whether to create a deep copy. Default: True
            
        Returns:
        --------
        CDSAXS_Model
            Copy of the current model
        """
        if deep:
            # Create new model with copied parameters
            new_model = self.__class__(
                self.model,
                self.layers,
                model_params=copy.deepcopy(self.model_params) if hasattr(self, 'model_params') else None
            )
            
            # Copy data if it exists
            if hasattr(self, 'Intensity'):
                new_model.Intensity = self.Intensity.copy()
            if hasattr(self, 'Qz'):
                new_model.Qz = self.Qz.copy()
            if hasattr(self, 'Qx'):
                new_model.Qx = self.Qx.copy()
            if hasattr(self, 'Qy'):
                new_model.Qy = self.Qy.copy()
            if hasattr(self, 'SimInt'):
                new_model.SimInt = self.SimInt.copy()
            
            # Copy other attributes
            for attr in ['GF', 'BIC', 'numberpoints', 'numbercuts']:
                if hasattr(self, attr):
                    setattr(new_model, attr, getattr(self, attr))
                    
        else:
            # Shallow copy
            new_model = copy.copy(self)
        
        return new_model
    
    def reset_to_initial(self):
        """
        Reset model parameters to their initial values.
        """
        if hasattr(self, 'PAR_Initial') and self.PAR_Initial is not None:
            self.PAR = self.PAR_Initial.copy()
        
        if hasattr(self, 'DW_Initial') and self.DW_Initial is not None:
            self.DW = self.DW_Initial
            
        if hasattr(self, 'I0_Initial') and self.I0_Initial is not None:
            self.I0 = self.I0_Initial
            
        if hasattr(self, 'Bk_Initial') and self.Bk_Initial is not None:
            self.Bk = self.Bk_Initial
        
        # Rebuild model_params from initial traditional parameters
        self.build_model_params_from_traditional()
        
        print("Model parameters reset to initial values")
    
    def get_model_info(self):
        """
        Get a comprehensive summary of the model.
        
        Returns:
        --------
        dict
            Dictionary containing model information
        """
        info = {
            'geometry': self.geometry,
            'model_type': self.model,
            'layers': self.layers,
            'has_data': hasattr(self, 'Intensity'),
            'has_simulation': hasattr(self, 'SimInt'),
            'parameters': {}
        }
        
        # Add parameter information
        if hasattr(self, 'model_params'):
            info['parameters'] = copy.deepcopy(self.model_params)
        
        # Add data information
        if hasattr(self, 'Intensity'):
            info['data_shape'] = self.Intensity.shape
            info['number_points'] = getattr(self, 'numberpoints', 'Unknown')
            info['number_cuts'] = getattr(self, 'numbercuts', self.Intensity.shape[1])
        
        # Add fitting results
        if hasattr(self, 'GF'):
            info['goodness_of_fit'] = self.GF
        if hasattr(self, 'BIC'):
            info['bic'] = self.BIC
        
        return info
    
    def _print_sweep_summary_1d(self, results):
        """
        Print a summary of 1D sweep results including the best fit details.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_1d
        """
        print(f"\n{'='*60}")
        print(f"1D PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        
        # Find best GF and BIC
        gf_values = np.array(results['gf_values'])
        bic_values = np.array(results['bic_values'])
        
        # Handle inf values
        finite_gf_mask = np.isfinite(gf_values)
        finite_bic_mask = np.isfinite(bic_values)
        
        if np.any(finite_gf_mask):
            best_gf_idx = np.argmin(gf_values[finite_gf_mask])
            best_gf_global_idx = np.where(finite_gf_mask)[0][best_gf_idx]
            best_gf = gf_values[best_gf_global_idx]
            best_gf_param = results['sweep_values'][best_gf_global_idx]
            best_gf_bic = bic_values[best_gf_global_idx]
        else:
            best_gf = float('inf')
            best_gf_param = None
            best_gf_bic = float('inf')
        
        if np.any(finite_bic_mask):
            best_bic_idx = np.argmin(bic_values[finite_bic_mask])
            best_bic_global_idx = np.where(finite_bic_mask)[0][best_bic_idx]
            best_bic = bic_values[best_bic_global_idx]
            best_bic_param = results['sweep_values'][best_bic_global_idx]
            best_bic_gf = gf_values[best_bic_global_idx]
        else:
            best_bic = float('inf')
            best_bic_param = None
            best_bic_gf = float('inf')
        
        # Print sweep info
        param_name = results['sweep_param']
        n_points = len(results['sweep_values'])
        param_range = (results['sweep_values'][0], results['sweep_values'][-1])
        convergence_rate = np.mean(results['convergence_flags']) * 100
        
        print(f"Parameter: {param_name}")
        print(f"Range: {param_range[0]:.4f} to {param_range[1]:.4f}")
        print(f"Points: {n_points}")
        print(f"Convergence rate: {convergence_rate:.1f}%")
        print()
        
        # Print best results
        print(f"BEST GOODNESS OF FIT:")
        if best_gf_param is not None:
            print(f"  {param_name} = {best_gf_param:.4f}")
            print(f"  GF = {best_gf:.4f}")
            print(f"  BIC = {best_gf_bic:.4f}")
        else:
            print("  No valid fits found")
        print()
        
        print(f"BEST BIC:")
        if best_bic_param is not None:
            print(f"  {param_name} = {best_bic_param:.4f}")
            print(f"  GF = {best_bic_gf:.4f}")
            print(f"  BIC = {best_bic:.4f}")
        else:
            print("  No valid fits found")
        
        print(f"{'='*60}")
        print("TIP: Use model.show_best_fit_results(results) to see detailed optimization results for the best fit")


    def _print_sweep_summary_2d(self, results):
        """
        Print a summary of 2D sweep results including the best fit details.
        
        Parameters:
        -----------
        results : dict
            Results from parameter_sweep_2d
        """
        print(f"\n{'='*60}")
        print(f"2D PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        
        # Get matrices and handle inf values
        gf_matrix = np.copy(results['gf_matrix'])
        bic_matrix = np.copy(results['bic_matrix'])
        
        gf_matrix[np.isinf(gf_matrix)] = np.nan
        bic_matrix[np.isinf(bic_matrix)] = np.nan
        
        # Find best results
        if not np.all(np.isnan(gf_matrix)):
            best_gf_idx = np.unravel_index(np.nanargmin(gf_matrix), gf_matrix.shape)
            best_gf = gf_matrix[best_gf_idx]
            best_gf_param1 = results['param1_values'][best_gf_idx[1]]
            best_gf_param2 = results['param2_values'][best_gf_idx[0]]
            best_gf_bic = results['bic_matrix'][best_gf_idx]
        else:
            best_gf = np.nan
            best_gf_param1 = None
            best_gf_param2 = None
            best_gf_bic = np.nan
        
        if not np.all(np.isnan(bic_matrix)):
            best_bic_idx = np.unravel_index(np.nanargmin(bic_matrix), bic_matrix.shape)
            best_bic = bic_matrix[best_bic_idx]
            best_bic_param1 = results['param1_values'][best_bic_idx[1]]
            best_bic_param2 = results['param2_values'][best_bic_idx[0]]
            best_bic_gf = results['gf_matrix'][best_bic_idx]
        else:
            best_bic = np.nan
            best_bic_param1 = None
            best_bic_param2 = None
            best_bic_gf = np.nan
        
        # Print sweep info
        param1_name, param2_name = results['sweep_params']
        grid_size = (len(results['param1_values']), len(results['param2_values']))
        param1_range = (results['param1_values'][0], results['param1_values'][-1])
        param2_range = (results['param2_values'][0], results['param2_values'][-1])
        convergence_rate = np.mean(results['convergence_matrix']) * 100
        
        print(f"Parameters: {param1_name} vs {param2_name}")
        print(f"Grid size: {grid_size[0]} x {grid_size[1]}")
        print(f"{param1_name} range: {param1_range[0]:.4f} to {param1_range[1]:.4f}")
        print(f"{param2_name} range: {param2_range[0]:.4f} to {param2_range[1]:.4f}")
        print(f"Convergence rate: {convergence_rate:.1f}%")
        print()
        
        # Print best results
        print(f"BEST GOODNESS OF FIT:")
        if best_gf_param1 is not None and not np.isnan(best_gf):
            print(f"  {param1_name} = {best_gf_param1:.4f}")
            print(f"  {param2_name} = {best_gf_param2:.4f}")
            print(f"  GF = {best_gf:.4f}")
            print(f"  BIC = {best_gf_bic:.4f}")
        else:
            print("  No valid fits found")
        print()
        
        print(f"BEST BIC:")
        if best_bic_param1 is not None and not np.isnan(best_bic):
            print(f"  {param1_name} = {best_bic_param1:.4f}")
            print(f"  {param2_name} = {best_bic_param2:.4f}")
            print(f"  GF = {best_bic_gf:.4f}")
            print(f"  BIC = {best_bic:.4f}")
        else:
            print("  No valid fits found")
        
        print(f"{'='*60}")
        print("TIP: Use model.show_best_fit_results(results) to see detailed optimization results for the best fit")
        
    def show_best_fit_results(self, results, criterion='GF', run_optimization=True):
        """
        Apply the best parameters from sweep results and show optimization details.
        
        Parameters:
        -----------
        results : dict
            Results from any parameter sweep function
        criterion : str, optional
            Criterion for selecting best parameters ('GF' or 'BIC'). Default: 'GF'
        run_optimization : bool, optional
            Whether to re-run optimization with best parameters. Default: True
        """
        print(f"\n{'='*60}")
        print(f"BEST FIT DETAILS ({criterion.upper()} CRITERION)")
        print(f"{'='*60}")
        
        # Get the best parameters based on sweep type
        if 'sweep_param' in results:
            # 1D sweep
            optimal = self.get_optimal_parameters_1d(results, criterion)
            if optimal['optimized_parameters'] is None:
                print("No valid optimized parameters found in results")
                return
            
            print(f"Best parameter value:")
            print(f"  {optimal['sweep_parameter']} = {optimal['optimal_value']:.4f}")
            print(f"  GF = {optimal['gf']:.4f}")
            print(f"  BIC = {optimal['bic']:.4f}")
            print()
            
            # Apply the best parameters
            self.model_params = copy.deepcopy(optimal['optimized_parameters'])
            
        elif 'sweep_params' in results:
            # 2D sweep
            optimal = self.get_optimal_parameters_2d(results, criterion)
            if optimal['optimized_parameters'] is None:
                print("No valid optimized parameters found in results")
                return
            
            print(f"Best parameter values:")
            for param, value in optimal['optimal_values'].items():
                print(f"  {param} = {value:.4f}")
            print(f"  GF = {optimal['gf']:.4f}")
            print(f"  BIC = {optimal['bic']:.4f}")
            print()
            
            # Apply the best parameters
            self.model_params = copy.deepcopy(optimal['optimized_parameters'])
            
        elif 'sweep_type' in results and results['sweep_type'] == 'width_dw_1layer':
            # 1-layer specialized sweep
            optimal = self.get_optimal_width_dw_1layer(results)
            
            if criterion.upper() == 'GF':
                best_result = optimal['best_gf']
            else:
                best_result = optimal['best_bic']
                
            print(f"Best parameter values:")
            print(f"  Width = {best_result['width']:.1f} Å (both trap_0_width and trap_1_width)")
            print(f"  DW = {best_result['dw']:.3f}")
            print(f"  GF = {best_result['gf_value']:.4f}")
            print(f"  BIC = {best_result['bic_value']:.4f}")
            print()
            
            # Apply the best parameters manually for 1-layer case
            self.model_params['trapezoids'][0]['width'] = best_result['width']
            self.model_params['trapezoids'][1]['width'] = best_result['width']
            self.model_params['DW'] = best_result['dw']
        else:
            print("Unknown sweep type")
            return
        
        # Update traditional parameters
        self.update_traditional_from_model_params()
        
        if run_optimization:
            print("Re-running optimization with best parameters to show detailed results...")
            print("-" * 60)
            
            # Get all optimizable parameters for the final detailed run
            self.initialize_optimization_params()
            all_params = self.model_params.get('optimization', {})
            
            if all_params:
                # Run optimization with full output
                final_result = self.CDSAXS_DiffEvolution(
                    params_to_optimize=all_params,
                    plot_results=True,  # Show all plots for best fit
                    verbose=True       # Show all output details
                )
            else:
                # Just simulate if no parameters to optimize
                self.SimInt = self.simulate_structure()
                self.GF = self.GF_calc(self.SimInt)
                self.BIC = self.BIC_calc(self.GF)
                
                print(f"Final results:")
                print(f"  GF = {self.GF:.4f}")
                print(f"  BIC = {self.BIC:.4f}")
        else:
            # Just update simulation without showing optimization details
            self.SimInt = self.simulate_structure()
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            print(f"Applied best parameters. Final metrics:")
            print(f"  GF = {self.GF:.4f}")
            print(f"  BIC = {self.BIC:.4f}")
        
        print(f"{'='*60}")


    # Complete layer insertion methods for CDSAXS_base_model.py
# These are fully self-contained and don't require any external imports

    def calculate_width_at_height(self, height_position: float):
        """Calculate the width (trapezoid) or radius (cylinder) at a given height position."""
        ### Not clear if this will work for the SiGe Model
        if not hasattr(self, 'model_params'):
            raise AttributeError("Model must have model_params attribute")
        
        # Get structure data based on geometry
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
            width_key = 'width'
        elif self.geometry == 'cylinder':
            structures = self.model_params['cylinders']
            width_key = 'radius'
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        # Calculate total height
        total_height = sum(struct['height'] for struct in structures[:-1])
        
        # Validate height position
        if height_position < 0 or height_position > total_height:
            raise ValueError(f"Height position {height_position:.2f} is outside valid range [0, {total_height:.2f}]")
        
        # Special cases
        if height_position == 0:
            return structures[0][width_key]
        if height_position == total_height:
            return structures[-1][width_key]
        
        # Find which layer contains this height
        current_height = 0
        for i in range(len(structures) - 1):
            layer_height = structures[i]['height']
            
            if current_height <= height_position <= current_height + layer_height:
                # Found the layer - interpolate between bottom and top widths
                bottom_width = structures[i][width_key]
                top_width = structures[i + 1][width_key]
                
                # Calculate position within this layer (0 = bottom, 1 = top)
                layer_position = (height_position - current_height) / layer_height
                
                # Linear interpolation
                interpolated_width = bottom_width + layer_position * (top_width - bottom_width)
                return interpolated_width
            
            current_height += layer_height
        
        # Should never reach here if height_position is valid
        raise ValueError(f"Could not find layer containing height {height_position:.2f}")

    def get_total_structure_height(self):
        """Get the total height of the model structure."""
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
        elif self.geometry == 'cylinder':
            structures = self.model_params['cylinders']
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        return sum(struct['height'] for struct in structures[:-1])

    def _find_layer_at_height(self, height_position: float):
        """
        Find which layer contains the given height and the position within that layer.
        
        Parameters:
        -----------
        height_position : float
            Height position from bottom
            
        Returns:
        --------
        tuple
            (layer_index, position_in_layer) where position_in_layer is 0-1
        """
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
        elif self.geometry == 'cylinder':
            structures = self.model_params['cylinders']
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        current_height = 0
        for i in range(len(structures) - 1):
            layer_height = structures[i]['height']
            
            if current_height <= height_position <= current_height + layer_height:
                position_in_layer = (height_position - current_height) / layer_height
                return i, position_in_layer
            
            current_height += layer_height
        
        raise ValueError(f"Could not find layer containing height {height_position:.2f}")

    def _copy_model_data(self, target_model):
        """
        Copy experimental data from this model to target model.
        Also store reference to source model for optimization inheritance.
        """
        data_attributes = ['Intensity', 'Qz', 'Qx', 'Qy', 'Qr', 'Alpha', 'numberpoints', 'numbercuts']
        
        for attr in data_attributes:
            if hasattr(self, attr):
                setattr(target_model, attr, getattr(self, attr))
        
        # Store reference to source model for optimization parameter inheritance
        target_model._source_model = self

    def _create_new_model_from_params(self, new_model_params):
        """
        Create a new model from parameters using available methods.
        This avoids import issues by using the class's existing capabilities.
        """
        # Try different approaches to create the new model
        
        # Method 1: Try using the create_model static method if available
        if hasattr(self.__class__, 'create_model'):
            try:
                return self.__class__.create_model(
                    geometry=self.geometry,
                    model=self.model,
                    layers=new_model_params['layers'],
                    model_params=new_model_params
                )
            except:
                pass
        
        # Method 2: Try creating using the class constructor directly
        try:
            new_model = self.__class__(
                model=self.model,
                layers=new_model_params['layers'],
                model_params=new_model_params
            )
            return new_model
        except:
            pass
        
        # Method 3: Try using cdsaxs.create_model if available
        try:
            import cdsaxs
            return cdsaxs.create_model(
                geometry=self.geometry,
                model=self.model,
                layers=new_model_params['layers'],
                model_params=new_model_params
            )
        except:
            pass
        
        # Method 4: Manual class selection (fallback)
        if self.geometry in ['trapezoid', 'sige']:
            # Try to get TrapezoidModel class
            try:
                # First try to get it from the same module
                import sys
                current_module = sys.modules[self.__module__]
                if hasattr(current_module, 'TrapezoidModel'):
                    TrapezoidModel = getattr(current_module, 'TrapezoidModel')
                else:
                    # Try importing from parent package
                    module_parts = self.__module__.split('.')
                    if len(module_parts) > 1:
                        parent_module = '.'.join(module_parts[:-1])
                        try:
                            import importlib
                            parent = importlib.import_module(parent_module)
                            TrapezoidModel = getattr(parent, 'TrapezoidModel')
                        except:
                            raise ImportError("Could not find TrapezoidModel")
                    else:
                        raise ImportError("Could not find TrapezoidModel")
                
                return TrapezoidModel(
                    model=self.model,
                    layers=new_model_params['layers'],
                    model_params=new_model_params
                )
            except:
                raise ImportError("Could not create TrapezoidModel")
        
        elif self.geometry == 'cylinder':
            # Try to get CylinderModel class
            try:
                # Similar approach for cylinder model
                import sys
                current_module = sys.modules[self.__module__]
                if hasattr(current_module, 'CylinderModel'):
                    CylinderModel = getattr(current_module, 'CylinderModel')
                else:
                    module_parts = self.__module__.split('.')
                    if len(module_parts) > 1:
                        parent_module = '.'.join(module_parts[:-1])
                        try:
                            import importlib
                            parent = importlib.import_module(parent_module)
                            CylinderModel = getattr(parent, 'CylinderModel')
                        except:
                            raise ImportError("Could not find CylinderModel")
                    else:
                        raise ImportError("Could not find CylinderModel")
                
                return CylinderModel(
                    model=self.model,
                    layers=new_model_params['layers'],
                    model_params=new_model_params
                )
            except:
                raise ImportError("Could not create CylinderModel")
        
        raise ValueError(f"Could not create new model for geometry: {self.geometry}")

    def add_layer_at_percentage(self, height_percentage: float, auto_setup_optimization: bool = True, 
                           optimization_margin: float = 0.2, discretization_per_nm: float = 5.0,
                           new_layer_height: float = None, inherit_global_limits: bool = True):
        """
        Add exactly one layer at the specified height percentage.
        
        Parameters:
        -----------
        height_percentage : float
            Percentage of total height where to insert new layer (0-100)
        auto_setup_optimization : bool, optional
            Whether to automatically setup optimization parameters. Default: True
        optimization_margin : float, optional
            Margin for optimization bounds as a fraction for new structure parameters. Default: 0.2 (±20%)
        discretization_per_nm : float, optional
            For cylinder models: discretization points per nanometer. Default: 5.0
        new_layer_height : float, optional
            Height for the new layer in Angstroms. If None, calculates as 5% of total height
        inherit_global_limits : bool, optional
            Whether to inherit DW, I0, Bk limits from the original model. Default: True
            
        Returns:
        --------
        CDSAXS_Model
            New model with exactly one additional layer
        """
        import copy
        import numpy as np
        
        if not 0 <= height_percentage <= 100:
            raise ValueError("Height percentage must be between 0 and 100")
        
        # Get structures
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
            width_key = 'width'
        else:
            structures = self.model_params['cylinders']
            width_key = 'radius'
        
        # Extract width and height arrays
        widths = [s[width_key] for s in structures]
        heights = [s['height'] for s in structures[:-1]]
        
        # Calculate insertion parameters
        total_height = sum(heights)
        insertion_height = (height_percentage / 100) * total_height
        if new_layer_height is None:
            new_layer_height = max(2.0, total_height * 0.05)
        
        # Find which layer contains the insertion height
        current_height = 0.0
        insert_after_layer = None
        height_into_target_layer = 0.0
        
        for i, layer_height in enumerate(heights):
            layer_bottom = current_height
            layer_top = current_height + layer_height
            
            if layer_bottom <= insertion_height <= layer_top:
                insert_after_layer = i
                height_into_target_layer = insertion_height - layer_bottom
                break
                
            current_height += layer_height
        
        if insert_after_layer is None:
            raise ValueError(f"Could not find layer containing insertion height {insertion_height:.1f}")
        
        # Calculate width at insertion point
        bottom_width = widths[insert_after_layer]
        top_width = widths[insert_after_layer + 1]
        target_layer_height = heights[insert_after_layer]
        
        if target_layer_height > 0:
            position_in_layer = height_into_target_layer / target_layer_height
            insertion_width = bottom_width + position_in_layer * (top_width - bottom_width)
        else:
            insertion_width = bottom_width
        
        # Create new width and height arrays
        new_widths = widths.copy()
        new_heights = heights.copy()
        
        # Split the target layer height
        remaining_height = target_layer_height - height_into_target_layer
        
        # Modify the target layer to only go up to insertion point
        new_heights[insert_after_layer] = height_into_target_layer
        
        # Insert width at insertion point
        new_widths.insert(insert_after_layer + 1, insertion_width)
        
        # Insert new layer height
        new_heights.insert(insert_after_layer + 1, new_layer_height)
        
        # Add remaining height to the new layer if significant
        if remaining_height > 0.01:
            new_heights[insert_after_layer + 1] += remaining_height
        
        # Verify we added exactly 1 width + 1 height
        if len(new_widths) - len(widths) != 1 or len(new_heights) - len(heights) != 1:
            raise RuntimeError("Layer insertion failed: incorrect width/height count")
        
        # Build new structures
        new_structures = []
        for i in range(len(new_widths)):
            if i < len(new_heights):
                structure = {width_key: new_widths[i], 'height': new_heights[i]}
            else:
                structure = {width_key: new_widths[i], 'height': 0.0}
            new_structures.append(structure)
        
        # Create new model parameters
        new_model_params = copy.deepcopy(self.model_params)
        
        if self.geometry in ['trapezoid', 'sige']:
            new_model_params['trapezoids'] = new_structures
        else:
            new_model_params['cylinders'] = new_structures
            # Generate discretization for cylinders
            new_discretization = []
            for i in range(len(new_heights)):
                h_nm = new_structures[i]['height'] / 10.0
                disc = max(5, min(50, int(h_nm * discretization_per_nm)))
                new_discretization.append(disc)
            new_model_params['discretization'] = new_discretization
        
        new_model_params['layers'] = len(new_heights)
        
        # Create new model
        new_model = self._create_new_model_from_params(new_model_params)
        self._copy_model_data(new_model)
        
        # Setup optimization parameters if requested
        if auto_setup_optimization:
            new_model._setup_optimization_for_new_layers(optimization_margin, inherit_global_limits)
        
        return new_model




    

    def add_multiple_layers(self, height_percentages: list, sequential: bool = False, 
                       auto_setup_optimization: bool = True, optimization_margin: float = 0.2,
                       discretization_per_nm: float = 5.0, inherit_global_limits: bool = True):
        """
        Add multiple layers at different height percentages.
        
        Parameters:
        -----------
        height_percentages : list
            List of height percentages where to insert new layers (0-100)
        sequential : bool, optional
            If True, add all layers to a single model sequentially
            If False, create separate models each with one additional layer
        auto_setup_optimization : bool, optional
            Whether to automatically setup optimization parameters. Default: True
        optimization_margin : float, optional
            Margin for optimization bounds as a fraction for new structure parameters. Default: 0.2 (±20%)
        discretization_per_nm : float, optional
            For cylinder models: discretization points per nanometer. Default: 5.0
        inherit_global_limits : bool, optional
            Whether to inherit DW, I0, Bk limits from the original model. Default: True
            
        Returns:
        --------
        CDSAXS_Model or List[CDSAXS_Model]
            If sequential=True: Single model with all additional layers
            If sequential=False: List of models, each with one additional layer
        """
        if sequential:
            # Sort percentages to ensure proper insertion order (top to bottom)
            sorted_percentages = sorted(height_percentages, reverse=True)
            
            # Start with the original model
            current_model = self
            
            # Add layers one by one, working from top to bottom
            for height_percentage in sorted_percentages:
                current_model = current_model.add_layer_at_percentage(
                    height_percentage, 
                    auto_setup_optimization=False,
                    optimization_margin=optimization_margin,
                    discretization_per_nm=discretization_per_nm,
                    inherit_global_limits=inherit_global_limits
                )
            
            # Setup optimization for the final model
            if auto_setup_optimization:
                current_model._setup_optimization_for_new_layers(optimization_margin, inherit_global_limits)
            
            return current_model
        else:
            # Create separate models
            new_models = []
            
            for height_percentage in height_percentages:
                new_model = self.add_layer_at_percentage(
                    height_percentage, 
                    auto_setup_optimization=auto_setup_optimization,
                    optimization_margin=optimization_margin,
                    discretization_per_nm=discretization_per_nm,
                    inherit_global_limits=inherit_global_limits
                )
                new_models.append(new_model)
            
            return new_models

    def visualize_layer_addition(self, height_percentage: float, figsize=(12, 6)):
        """
        Visualize the original model and the model with inserted layer side by side.
        
        Parameters:
        -----------
        height_percentage : float
            Height percentage where layer will be inserted
        figsize : tuple
            Figure size for the plot
            
        Returns:
        --------
        CDSAXS_Model
            New model with the inserted layer
        """
        import matplotlib.pyplot as plt
        
        # Create new model with inserted layer
        new_model = self.add_layer_at_percentage(height_percentage, auto_setup_optimization=False)
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot original model
        plt.sca(ax1)
        self.plot_structure()
        ax1.set_title(f'Original Model ({self.layers} layers)')
        
        # Add line showing insertion point
        total_height = self.get_total_structure_height()
        insertion_height = (height_percentage / 100) * total_height
        insertion_width = self.calculate_width_at_height(insertion_height)
        
        if self.geometry in ['trapezoid', 'sige']:
            ax1.axhline(y=insertion_height, color='red', linestyle='--', alpha=0.7, 
                    label=f'Insertion at {height_percentage}%')
            ax1.plot([-insertion_width/2, insertion_width/2], [insertion_height, insertion_height], 
                    'ro', markersize=8)
        else:  # cylinder
            ax1.axhline(y=insertion_height, color='red', linestyle='--', alpha=0.7, 
                    label=f'Insertion at {height_percentage}%')
            ax1.plot([-insertion_width, insertion_width], [insertion_height, insertion_height], 
                    'ro', markersize=8)
        
        ax1.legend()
        
        # Plot new model
        plt.sca(ax2)
        new_model.plot_structure()
        ax2.set_title(f'Modified Model ({new_model.layers} layers)')
        
        plt.tight_layout()
        plt.show()
        
        return new_model
    
    def _setup_optimization_for_new_layers(self, optimization_margin: float = 0.2, inherit_global_limits: bool = True):
        """
        Set up optimization parameters for the current model structure.
        
        Parameters:
        -----------
        optimization_margin : float, optional
            Margin for optimization bounds as a fraction for new structure parameters. Default: 0.2 (±20%)
        inherit_global_limits : bool, optional
            Whether to inherit DW, I0, Bk limits from previous model if available. Default: True
        """
        import numpy as np
        
        # Initialize optimization parameters with the specified margin
        param_limits = {}
        
        if self.geometry in ['trapezoid', 'sige']:
            # Add trapezoid parameters
            for i, trap in enumerate(self.model_params['trapezoids']):
                # Width parameters
                width_val = trap['width']
                param_limits[f'trap_{i}_width'] = {
                    'min': width_val * (1 - optimization_margin),
                    'max': width_val * (1 + optimization_margin),
                    'default': width_val
                }
                
                # Height parameters (skip the last trapezoid which has height 0)
                if i < len(self.model_params['trapezoids']) - 1:
                    height_val = trap['height']
                    param_limits[f'trap_{i}_height'] = {
                        'min': height_val * (1 - optimization_margin),
                        'max': height_val * (1 + optimization_margin),
                        'default': height_val
                    }
                
                # TWidth parameters (if present - used by SiGe model)
                if 'twidth' in trap and trap['twidth'] is not None:
                    twidth_val = trap['twidth']
                    param_limits[f'trap_{i}_twidth'] = {
                        'min': twidth_val * (1 - optimization_margin),
                        'max': twidth_val * (1 + optimization_margin),
                        'default': twidth_val
                    }
        
        elif self.geometry == 'cylinder':
            # Add cylinder parameters
            for i, cyl in enumerate(self.model_params['cylinders']):
                # Radius parameters
                radius_val = cyl['radius']
                param_limits[f'cyl_{i}_radius'] = {
                    'min': radius_val * (1 - optimization_margin),
                    'max': radius_val * (1 + optimization_margin),
                    'default': radius_val
                }
                
                # Height parameters (skip the last cylinder which has height 0)
                if i < len(self.model_params['cylinders']) - 1:
                    height_val = cyl['height']
                    param_limits[f'cyl_{i}_height'] = {
                        'min': height_val * (1 - optimization_margin),
                        'max': height_val * (1 + optimization_margin),
                        'default': height_val
                    }
        
        # Add global parameters with inheritance option
        if inherit_global_limits and hasattr(self, '_source_model') and hasattr(self._source_model, 'model_params'):
            # Try to inherit from source model optimization parameters
            source_optimization = self._source_model.model_params.get('optimization', {})
            
            # Inherit DW limits if available
            if 'DW' in source_optimization:
                inherited_dw = source_optimization['DW'].copy()
                inherited_dw['default'] = self.DW  # Update default to current value
                param_limits['DW'] = inherited_dw
            else:
                # Fallback to margin-based
                param_limits['DW'] = {
                    'min': self.DW * (1 - optimization_margin),
                    'max': self.DW * (1 + optimization_margin),
                    'default': self.DW
                }
            
            # Inherit I0 limits if available
            if 'I0' in source_optimization:
                inherited_i0 = source_optimization['I0'].copy()
                inherited_i0['default'] = self.I0  # Update default to current value
                param_limits['I0'] = inherited_i0
            else:
                # Fallback to margin-based
                param_limits['I0'] = {
                    'min': self.I0 * (1 - optimization_margin),
                    'max': self.I0 * (1 + optimization_margin),
                    'default': self.I0
                }
            
            # Inherit background limits if available
            if isinstance(self.Bk, np.ndarray):
                # Array background - inherit individual limits
                for i, bk_val in enumerate(self.Bk):
                    bk_param_name = f'Bk_{i}'
                    if bk_param_name in source_optimization:
                        inherited_bk = source_optimization[bk_param_name].copy()
                        inherited_bk['default'] = bk_val  # Update default to current value
                        param_limits[bk_param_name] = inherited_bk
                    else:
                        # Fallback to margin-based
                        param_limits[bk_param_name] = {
                            'min': bk_val * (1 - optimization_margin),
                            'max': bk_val * (1 + optimization_margin),
                            'default': bk_val
                        }
            else:
                # Scalar background
                if 'Bk' in source_optimization:
                    inherited_bk = source_optimization['Bk'].copy()
                    inherited_bk['default'] = self.Bk  # Update default to current value
                    param_limits['Bk'] = inherited_bk
                else:
                    # Fallback to margin-based
                    param_limits['Bk'] = {
                        'min': self.Bk * (1 - optimization_margin),
                        'max': self.Bk * (1 + optimization_margin),
                        'default': self.Bk
                    }
        else:
            # No inheritance - use margin-based approach for global parameters
            param_limits['DW'] = {
                'min': self.DW * (1 - optimization_margin),
                'max': self.DW * (1 + optimization_margin),
                'default': self.DW
            }
            
            param_limits['I0'] = {
                'min': self.I0 * (1 - optimization_margin),
                'max': self.I0 * (1 + optimization_margin),
                'default': self.I0
            }
            
            # Handle background parameters
            if isinstance(self.Bk, np.ndarray):
                for i, bk_val in enumerate(self.Bk):
                    param_limits[f'Bk_{i}'] = {
                        'min': bk_val * (1 - optimization_margin),
                        'max': bk_val * (1 + optimization_margin),
                        'default': bk_val
                    }
            else:
                param_limits['Bk'] = {
                    'min': self.Bk * (1 - optimization_margin),
                    'max': self.Bk * (1 + optimization_margin),
                    'default': self.Bk
                }
        
        # Store optimization parameters
        self.model_params['optimization'] = param_limits
        
        return param_limits


    def _extract_width_height_relationship(self):
        """
        Extract the width-height relationship from the current model.
        
        Returns:
        --------
        tuple
            (height_points, width_points) arrays defining the structure profile
        """
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
            width_key = 'width'
        elif self.geometry == 'cylinder':
            structures = self.model_params['cylinders']
            width_key = 'radius'
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        # Build arrays of height points and corresponding widths
        height_points = [0.0]  # Start at bottom
        width_points = [structures[0][width_key]]  # Bottom width
        
        current_height = 0.0
        for i in range(len(structures) - 1):  # Exclude the top point
            layer_height = structures[i]['height']
            current_height += layer_height
            height_points.append(current_height)
            width_points.append(structures[i + 1][width_key])
        
        
        return np.array(height_points), np.array(width_points)

    def _calculate_width_from_relationship(self, height, height_points, width_points):
        """
        Calculate width at given height using the extracted relationship.
        
        Parameters:
        -----------
        height : float
            Height where to calculate width
        height_points : numpy.ndarray
            Array of height reference points
        width_points : numpy.ndarray
            Array of corresponding widths
            
        Returns:
        --------
        float
            Interpolated width at the given height
        """
        # Handle edge cases
        if height <= height_points[0]:
            return width_points[0]
        if height >= height_points[-1]:
            return width_points[-1]
        
        # Find the interval containing the height
        for i in range(len(height_points) - 1):
            if height_points[i] <= height <= height_points[i + 1]:
                # Linear interpolation between the two points
                h1, h2 = height_points[i], height_points[i + 1]
                w1, w2 = width_points[i], width_points[i + 1]
                
                if h2 == h1:  # Avoid division by zero
                    return w1
                
                # Linear interpolation
                fraction = (height - h1) / (h2 - h1)
                width = w1 + fraction * (w2 - w1)
                return width
        
        # Should never reach here
        raise ValueError(f"Could not interpolate width for height {height}")



    def batch_initialize_and_fit(self, initialization_params, n_points_per_param=5, 
                            optimization_kwargs=None, max_fits=None, verbose=True,
                            save_results=True, results_filename=None):
        """
        Perform batch initialization and fitting with different starting conditions.
        
        Parameters:
        -----------
        initialization_params : dict
            Dictionary specifying parameters to vary and their ranges
            Format: {'param_name': {'min': value, 'max': value, 'n_points': int}}
            or: {'param_name': {'min': value, 'max': value}} (uses n_points_per_param)
        n_points_per_param : int, optional
            Default number of points per parameter if not specified. Default: 5
        optimization_kwargs : dict, optional
            Additional kwargs for CDSAXS_DiffEvolution
        max_fits : int, optional
            Maximum number of fits to perform (useful for large grids). Default: None (all)
        verbose : bool, optional
            Whether to print progress. Default: True
        save_results : bool, optional
            Whether to save results to file. Default: True
        results_filename : str, optional
            Filename for saved results. If None, auto-generates with timestamp
            
        Returns:
        --------
        dict
            Dictionary containing batch fitting results
        """
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing batch fitting")
        
        # Set default optimization parameters
        if optimization_kwargs is None:
            optimization_kwargs = {
                'maxiter': 50,
                'popsize': 15,
                'plot_results': False,  # Don't plot individual fits
                'verbose': False       # Don't print individual fit details
            }
        
        # Generate initialization grid
        grid_points = self._generate_initialization_grid(initialization_params, n_points_per_param)
        
        # Limit number of fits if requested
        if max_fits is not None and len(grid_points) > max_fits:
            if verbose:
                print(f"Limiting fits to {max_fits} out of {len(grid_points)} possible combinations")
            # Randomly sample to get diverse coverage
            indices = np.random.choice(len(grid_points), max_fits, replace=False)
            grid_points = [grid_points[i] for i in sorted(indices)]
        
        if verbose:
            print(f"Starting batch fitting with {len(grid_points)} different initializations...")
            print(f"Varying parameters: {list(initialization_params.keys())}")
        
        # Store original parameters for restoration
        original_params = copy.deepcopy(self.model_params)
        
        # Initialize results storage
        results = {
            'initialization_params': initialization_params,
            'n_total_fits': len(grid_points),
            'fits': [],
            'geometry': self.geometry,
            'layers': self.layers
        }
        
        # Progress bar
        if verbose:
            pbar = tqdm(enumerate(grid_points), total=len(grid_points), 
                    desc="Batch Fitting")
        else:
            pbar = enumerate(grid_points)
        
        # Perform fits
        for fit_idx, init_values in pbar:
            try:
                # Reset to original parameters
                self.model_params = copy.deepcopy(original_params)
                self.update_traditional_from_model_params()
                
                # Apply initialization values
                for param_name, value in init_values.items():
                    self._set_parameter_value(param_name, value)
                
                # Create optimization parameters (all fittable parameters)
                opt_params = self._create_full_optimization_params()
                
                # Run optimization
                optimized_params = self.CDSAXS_DiffEvolution(
                    params_to_optimize=opt_params,
                    **optimization_kwargs
                )
                
                if optimized_params is not None:
                    # Store fit result
                    fit_result = {
                        'fit_id': fit_idx + 1,
                        'initialization': init_values.copy(),
                        'final_params': copy.deepcopy(self.model_params),
                        'gf': self.GF,
                        'bic': self.BIC,
                        'converged': True,
                        'optimization_result': getattr(self, 'optimization_result', None)
                    }
                    
                    # Check for parameters near bounds
                    fit_result['near_bounds'] = self._check_parameters_near_bounds(opt_params)
                    
                    results['fits'].append(fit_result)
                    
                    # Update progress bar with current best
                    if len(results['fits']) > 0:
                        best_gf = min(fit['gf'] for fit in results['fits'])
                        if verbose and hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix({
                                'Best_GF': f'{best_gf:.4f}',
                                'Current_GF': f'{self.GF:.4f}',
                                'Fits': len(results['fits'])
                            })
                else:
                    # Failed fit
                    fit_result = {
                        'fit_id': fit_idx + 1,
                        'initialization': init_values.copy(),
                        'final_params': None,
                        'gf': float('inf'),
                        'bic': float('inf'),
                        'converged': False,
                        'near_bounds': {}
                    }
                    results['fits'].append(fit_result)
                    
            except Exception as e:
                if verbose:
                    print(f"Error in fit {fit_idx + 1}: {str(e)}")
                
                # Store failed fit
                fit_result = {
                    'fit_id': fit_idx + 1,
                    'initialization': init_values.copy(),
                    'final_params': None,
                    'gf': float('inf'),
                    'bic': float('inf'),
                    'converged': False,
                    'error': str(e),
                    'near_bounds': {}
                }
                results['fits'].append(fit_result)
        
        if verbose and hasattr(pbar, 'close'):
            pbar.close()
        
        # Sort results by GF (best first)
        results['fits'].sort(key=lambda x: x['gf'])
        
        # Add rankings
        for i, fit in enumerate(results['fits']):
            fit['rank'] = i + 1
        
        # Restore original parameters
        self.model_params = original_params
        self.update_traditional_from_model_params()
        
        # Print summary
        if verbose:
            self._print_batch_summary(results)
        
        # Save results
        if save_results:
            filename = self._save_batch_results(results, results_filename)
            if verbose:
                print(f"Results saved to: {filename}")
        
        return results

    def _generate_initialization_grid(self, initialization_params, n_points_per_param):
        """
        Generate grid of initialization points from parameter ranges.
        
        Parameters:
        -----------
        initialization_params : dict
            Parameter specifications
        n_points_per_param : int
            Default number of points per parameter
            
        Returns:
        --------
        list
            List of dictionaries, each containing initialization values
        """
        # Create arrays for each parameter
        param_arrays = {}
        param_names = []
        
        for param_name, param_spec in initialization_params.items():
            param_names.append(param_name)
            
            # Determine number of points
            if 'n_points' in param_spec:
                n_points = param_spec['n_points']
            else:
                n_points = n_points_per_param
            
            # Generate values
            if n_points == 1:
                # Single point - use midpoint
                mid_val = (param_spec['min'] + param_spec['max']) / 2
                param_arrays[param_name] = [mid_val]
            else:
                # Multiple points - linspace
                param_arrays[param_name] = np.linspace(
                    param_spec['min'], 
                    param_spec['max'], 
                    n_points
                )
        
        # Generate all combinations
        grid_points = []
        
        def generate_combinations(param_idx, current_combo):
            if param_idx == len(param_names):
                grid_points.append(current_combo.copy())
                return
            
            param_name = param_names[param_idx]
            for value in param_arrays[param_name]:
                current_combo[param_name] = value
                generate_combinations(param_idx + 1, current_combo)
        
        generate_combinations(0, {})
        
        return grid_points

    def _create_full_optimization_params(self, margin=0.3):
        """
        Create optimization parameters for all fittable parameters.
        
        Parameters:
        -----------
        margin : float, optional
            Margin for bounds as fraction (0.3 = ±30%). Default: 0.3
            
        Returns:
        --------
        dict
            Dictionary of optimization parameters
        """
        opt_params = {}
        
        if self.geometry in ['trapezoid', 'sige']:
            # Add all trapezoid parameters
            for i, trap in enumerate(self.model_params['trapezoids']):
                # Width parameters
                width_val = trap['width']
                opt_params[f'trap_{i}_width'] = {
                    'min': width_val * (1 - margin),
                    'max': width_val * (1 + margin),
                    'default': width_val
                }
                
                # Height parameters (skip last trapezoid)
                if i < len(self.model_params['trapezoids']) - 1:
                    height_val = trap['height']
                    opt_params[f'trap_{i}_height'] = {
                        'min': height_val * (1 - margin),
                        'max': height_val * (1 + margin),
                        'default': height_val
                    }
                
                # TWidth parameters (if present - used by SiGe model)
                if 'twidth' in trap and trap['twidth'] is not None:
                    twidth_val = trap['twidth']
                    opt_params[f'trap_{i}_twidth'] = {
                        'min': twidth_val * (1 - margin),
                        'max': twidth_val * (1 + margin),
                        'default': twidth_val
                    }
        
        elif self.geometry == 'cylinder':
            # Add all cylinder parameters
            for i, cyl in enumerate(self.model_params['cylinders']):
                # Radius parameters
                radius_val = cyl['radius']
                opt_params[f'cyl_{i}_radius'] = {
                    'min': radius_val * (1 - margin),
                    'max': radius_val * (1 + margin),
                    'default': radius_val
                }
                
                # Height parameters (skip last cylinder)
                if i < len(self.model_params['cylinders']) - 1:
                    height_val = cyl['height']
                    opt_params[f'cyl_{i}_height'] = {
                        'min': height_val * (1 - margin),
                        'max': height_val * (1 + margin),
                        'default': height_val
                    }
        
        # Add global parameters
        opt_params['DW'] = {
            'min': self.DW * (1 - margin),
            'max': self.DW * (1 + margin),
            'default': self.DW
        }
        
        opt_params['I0'] = {
            'min': self.I0 * (1 - margin),
            'max': self.I0 * (1 + margin),
            'default': self.I0
        }
        
        # Handle background parameters
        if isinstance(self.Bk, np.ndarray):
            for i, bk_val in enumerate(self.Bk):
                opt_params[f'Bk_{i}'] = {
                    'min': bk_val * (1 - margin),
                    'max': bk_val * (1 + margin),
                    'default': bk_val
                }
        else:
            opt_params['Bk'] = {
                'min': self.Bk * (1 - margin),
                'max': self.Bk * (1 + margin),
                'default': self.Bk
            }
        
        return opt_params

    def _check_parameters_near_bounds(self, opt_params, tolerance=0.01):
        """
        Check which parameters are near their optimization bounds.
        
        Parameters:
        -----------
        opt_params : dict
            Optimization parameters with bounds
        tolerance : float, optional
            Tolerance for "near bounds" (1% = 0.01). Default: 0.01
            
        Returns:
        --------
        dict
            Dictionary indicating which parameters are near bounds
        """
        near_bounds = {}
        
        for param_name, param_config in opt_params.items():
            try:
                current_value = self._get_current_parameter_value(param_name)
                param_range = param_config['max'] - param_config['min']
                
                # Check distance to bounds as fraction of range
                dist_to_min = (current_value - param_config['min']) / param_range
                dist_to_max = (param_config['max'] - current_value) / param_range
                
                near_min = dist_to_min < tolerance
                near_max = dist_to_max < tolerance
                
                near_bounds[param_name] = {
                    'near_min': near_min,
                    'near_max': near_max,
                    'near_either': near_min or near_max,
                    'current_value': current_value,
                    'min_bound': param_config['min'],
                    'max_bound': param_config['max']
                }
                
            except Exception as e:
                near_bounds[param_name] = {'error': str(e)}
        
        return near_bounds

    def _print_batch_summary(self, results):
        """
        Print a summary of batch fitting results.
        
        Parameters:
        -----------
        results : dict
            Batch fitting results
        """
        fits = results['fits']
        successful_fits = [f for f in fits if f['converged'] and f['gf'] != float('inf')]
        
        print(f"\n{'='*80}")
        print(f"BATCH FITTING SUMMARY")
        print(f"{'='*80}")
        print(f"Total initializations: {results['n_total_fits']}")
        print(f"Successful fits: {len(successful_fits)}")
        print(f"Failed fits: {results['n_total_fits'] - len(successful_fits)}")
        
        if successful_fits:
            print(f"Best GF: {successful_fits[0]['gf']:.6f}")
            print(f"Best BIC: {successful_fits[0]['bic']:.6f}")
            
            # Show distribution of GF values
            gf_values = [f['gf'] for f in successful_fits]
            print(f"GF range: {min(gf_values):.6f} to {max(gf_values):.6f}")
            print(f"GF std dev: {np.std(gf_values):.6f}")
        
        print(f"{'='*80}")

    def display_top_fits(self, results, n_top=10, show_near_bounds=True, 
                        colorize=True, save_table=False, table_filename=None):
        """
        Display top fitting results in a formatted table.
        
        Parameters:
        -----------
        results : dict
            Results from batch_initialize_and_fit
        n_top : int, optional
            Number of top fits to display. Default: 10
        show_near_bounds : bool, optional
            Whether to highlight parameters near bounds. Default: True
        colorize : bool, optional
            Whether to use color coding (red for near bounds). Default: True
        save_table : bool, optional
            Whether to save table to file. Default: False
        table_filename : str, optional
            Filename for saved table
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the top fits
        """
        fits = results['fits']
        top_fits = fits[:min(n_top, len(fits))]
        
        print(f"\n{'='*100}")
        print(f"TOP {len(top_fits)} FITS (Ranked by Goodness of Fit)")
        print(f"{'='*100}")
        
        # Create DataFrame for better formatting
        table_data = []
        
        for fit in top_fits:
            row = {
                'Rank': fit['rank'],
                'GF': fit['gf'],
                'BIC': fit['bic'],
                'Converged': '✓' if fit['converged'] else '✗'
            }
            
            # Add parameter values
            if fit['final_params'] is not None:
                # Add key parameters based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    # Show first few trapezoid widths and heights
                    for i in range(min(3, len(fit['final_params']['trapezoids']))):
                        trap = fit['final_params']['trapezoids'][i]
                        row[f'W{i}'] = trap['width']
                        if i < len(fit['final_params']['trapezoids']) - 1:
                            row[f'H{i}'] = trap['height']
                
                elif self.geometry == 'cylinder':
                    # Show first few cylinder radii and heights
                    for i in range(min(3, len(fit['final_params']['cylinders']))):
                        cyl = fit['final_params']['cylinders'][i]
                        row[f'R{i}'] = cyl['radius']
                        if i < len(fit['final_params']['cylinders']) - 1:
                            row[f'H{i}'] = cyl['height']
                
                # Add global parameters
                row['DW'] = fit['final_params']['DW']
                row['I0'] = fit['final_params']['I0']
                
                # Add background (first value if array)
                bk = fit['final_params']['Bk']
                if isinstance(bk, list):
                    row['Bk'] = bk[0]
                else:
                    row['Bk'] = bk
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Format numeric columns
        numeric_cols = [col for col in df.columns if col not in ['Rank', 'Converged']]
        for col in numeric_cols:
            if col in ['GF', 'BIC']:
                df[col] = df[col].apply(lambda x: f'{x:.6f}' if x != float('inf') else 'Failed')
            else:
                df[col] = df[col].apply(lambda x: f'{x:.3f}' if pd.notnull(x) else 'N/A')
        
        # Display table with color coding if requested
        if colorize and show_near_bounds:
            self._display_colorized_table(df, top_fits, results)
        else:
            print(df.to_string(index=False))
        
        # Save table if requested
        if save_table:
            filename = table_filename or f"top_fits_batch_results.csv"
            df.to_csv(filename, index=False)
            print(f"\nTable saved to: {filename}")
        
        return df

    def _display_colorized_table(self, df, top_fits, results):
        """
        Display table with color coding for parameters near bounds.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Table data
        top_fits : list
            List of top fit results
        results : dict
            Full batch results
        """
        # Print header
        header = "  ".join(f"{col:>10}" for col in df.columns)
        print(header)
        print("-" * len(header))
        
        # Print each row with color coding
        for i, (_, row) in enumerate(df.iterrows()):
            fit = top_fits[i]
            row_str = ""
            
            for j, (col, value) in enumerate(row.items()):
                # Check if this parameter is near bounds
                near_bounds = False
                if fit['converged'] and col not in ['Rank', 'GF', 'BIC', 'Converged']:
                    # Map display column to parameter name
                    param_name = self._map_column_to_param(col, fit)
                    if param_name and param_name in fit.get('near_bounds', {}):
                        near_bounds_info = fit['near_bounds'][param_name]
                        near_bounds = near_bounds_info.get('near_either', False)
                
                # Format value with color
                value_str = f"{value:>10}"
                if near_bounds:
                    # Red color for parameters near bounds
                    value_str = f"\033[91m{value_str}\033[0m"
                
                row_str += value_str + "  "
            
            print(row_str)
        
        # Print legend
        print("\n\033[91m■\033[0m = Parameter within 1% of optimization bounds")

    def _map_column_to_param(self, col, fit):
        """
        Map display column name to parameter name.
        
        Parameters:
        -----------
        col : str
            Column name from display table
        fit : dict
            Fit result
            
        Returns:
        --------
        str or None
            Parameter name, or None if not found
        """
        # Map display columns to parameter names
        if col.startswith('W') and col[1:].isdigit():
            idx = int(col[1:])
            return f'trap_{idx}_width'
        elif col.startswith('H') and col[1:].isdigit():
            idx = int(col[1:])
            return f'trap_{idx}_height'
        elif col.startswith('R') and col[1:].isdigit():
            idx = int(col[1:])
            return f'cyl_{idx}_radius'
        elif col in ['DW', 'I0', 'Bk']:
            return col
        
        return None

    def plot_fit_comparison(self, results, fit_ranks=[1, 2, 3], figsize=(15, 10),
                        show_structure=True, show_intensity=True):
        """
        Plot comparison of selected fit results side by side.
        
        Parameters:
        -----------
        results : dict
            Results from batch_initialize_and_fit
        fit_ranks : list, optional
            List of fit rankings to compare (1-indexed). Default: [1, 2, 3]
        figsize : tuple, optional
            Figure size. Default: (15, 10)
        show_structure : bool, optional
            Whether to show structure plots. Default: True
        show_intensity : bool, optional
            Whether to show intensity comparison plots. Default: True
        """
        fits = results['fits']
        
        # Validate fit ranks
        valid_ranks = []
        for rank in fit_ranks:
            if 1 <= rank <= len(fits) and fits[rank-1]['converged']:
                valid_ranks.append(rank)
            else:
                print(f"Warning: Rank {rank} is invalid or failed - skipping")
        
        if not valid_ranks:
            print("No valid fits to plot")
            return
        
        n_fits = len(valid_ranks)
        
        # Store current model state
        original_params = copy.deepcopy(self.model_params)
        
        try:
            if show_structure and show_intensity:
                # Create 2x3 grid (structure on top, intensity on bottom)
                fig, axes = plt.subplots(2, n_fits, figsize=figsize)
                if n_fits == 1:
                    axes = axes.reshape(2, 1)
            elif show_structure or show_intensity:
                # Create 1xN grid
                fig, axes = plt.subplots(1, n_fits, figsize=(figsize[0], figsize[1]//2))
                if n_fits == 1:
                    axes = [axes]
            else:
                print("Nothing to plot - both show_structure and show_intensity are False")
                return
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            for i, rank in enumerate(valid_ranks):
                fit = fits[rank-1]
                color = colors[i % len(colors)]
                
                # Apply fit parameters
                self.model_params = copy.deepcopy(fit['final_params'])
                self.update_traditional_from_model_params()
                
                # Plot structure
                if show_structure:
                    if show_intensity:
                        ax_struct = axes[0, i]
                    else:
                        ax_struct = axes[i]
                    
                    plt.sca(ax_struct)
                    self.plot_structure()
                    ax_struct.set_title(f'Rank #{rank}\nGF: {fit["gf"]:.4f}', 
                                    fontsize=12, color=color)
                    
                    # Highlight structure with color
                    for line in ax_struct.get_lines():
                        line.set_color(color)
                        line.set_linewidth(2)
                
                # Plot intensity comparison
                if show_intensity:
                    if show_structure:
                        ax_int = axes[1, i]
                    else:
                        ax_int = axes[i]
                    
                    # Simulate with current parameters
                    if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                        self.SimInt = self.simulate_structure(self.discretization)
                    else:
                        self.SimInt = self.simulate_structure()
                    
                    # Plot first few cuts
                    n_cuts_to_show = min(3, self.Intensity.shape[1])
                    
                    for cut_idx in range(n_cuts_to_show):
                        if self.geometry in ['trapezoid', 'sige']:
                            qz_values = self.Qz[:, cut_idx]
                            q_value = self.Qx[0, cut_idx]
                            q_label = 'Qx'
                        else:
                            qz_values = self.Qz[:, cut_idx]
                            q_value = self.Qr[0, cut_idx]
                            q_label = 'Qr'
                        
                        # Plot measured (gray) and simulated (colored)
                        alpha = 0.7 - cut_idx * 0.2
                        ax_int.semilogy(qz_values, self.Intensity[:, cut_idx], 
                                    'o', color='gray', alpha=alpha, markersize=3,
                                    label='Measured' if cut_idx == 0 else '')
                        ax_int.semilogy(qz_values, self.SimInt[:, cut_idx], 
                                    '-', color=color, alpha=alpha, linewidth=2,
                                    label=f'Rank #{rank}' if cut_idx == 0 else '')
                    
                    ax_int.set_xlabel('Qz (Å⁻¹)')
                    ax_int.set_ylabel('Intensity')
                    ax_int.set_title(f'Intensity Fit\nBIC: {fit["bic"]:.4f}', 
                                fontsize=12, color=color)
                    ax_int.grid(True, alpha=0.3)
                    
                    if i == 0:  # Only show legend on first plot
                        ax_int.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print fit details
            print(f"\nFit Comparison Details:")
            print(f"{'Rank':<6} {'GF':<12} {'BIC':<12} {'Notes'}")
            print("-" * 50)
            
            for rank in valid_ranks:
                fit = fits[rank-1]
                notes = []
                
                # Check for parameters near bounds
                if fit.get('near_bounds'):
                    near_bound_params = [name for name, info in fit['near_bounds'].items() 
                                    if info.get('near_either', False)]
                    if near_bound_params:
                        notes.append(f"{len(near_bound_params)} params near bounds")
                
                notes_str = "; ".join(notes) if notes else "Good"
                print(f"{rank:<6} {fit['gf']:<12.6f} {fit['bic']:<12.6f} {notes_str}")
        
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()

    def _save_batch_results(self, results, filename=None):
        """
        Save batch fitting results to file.
        
        Parameters:
        -----------
        results : dict
            Batch fitting results
        filename : str, optional
            Filename for saving. If None, auto-generates
            
        Returns:
        --------
        str
            Filename where results were saved
        """
        if filename is None:
            filename = f"batch_fits_{self.geometry}_{self.layers}L.pkl"
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        return filename

    def load_batch_results(self, filename):
        """
        Load batch fitting results from file.
        
        Parameters:
        -----------
        filename : str
            Filename to load
            
        Returns:
        --------
        dict
            Loaded batch fitting results
        """
        import pickle
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        return results

    def apply_batch_fit_result(self, results, rank=1, recalculate=True):
        """
        Apply parameters from a specific batch fit result to the model.
        
        Parameters:
        -----------
        results : dict
            Results from batch_initialize_and_fit
        rank : int, optional
            Rank of fit to apply (1 = best). Default: 1
        recalculate : bool, optional
            Whether to recalculate simulation and metrics. Default: True
        """
        if rank < 1 or rank > len(results['fits']):
            raise ValueError(f"Rank {rank} is out of range (1 to {len(results['fits'])})")
        
        fit = results['fits'][rank-1]
        
        if not fit['converged'] or fit['final_params'] is None:
            raise ValueError(f"Rank {rank} fit failed or has no parameters")
        
        print(f"Applying parameters from rank #{rank} fit:")
        print(f"  GF: {fit['gf']:.6f}")
        print(f"  BIC: {fit['bic']:.6f}")
        
        # Apply parameters
        self.model_params = copy.deepcopy(fit['final_params'])
        self.update_traditional_from_model_params()
        
        if recalculate:
            # Recalculate simulation and metrics
            if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                self.SimInt = self.simulate_structure(self.discretization)
            else:
                self.SimInt = self.simulate_structure()
            
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            print(f"Model updated with rank #{rank} parameters")
            print(f"Recalculated GF: {self.GF:.6f}")
            print(f"Recalculated BIC: {self.BIC:.6f}")
        else:
            print(f"Model parameters updated (simulation not recalculated)")

    def get_batch_fit_summary(self, results, criterion='GF', n_top=5):
        """
        Get a summary of batch fitting results.
        
        Parameters:
        -----------
        results : dict
            Results from batch_initialize_and_fit
        criterion : str, optional
            Criterion for ranking ('GF' or 'BIC'). Default: 'GF'
        n_top : int, optional
            Number of top results to include in summary. Default: 5
            
        Returns:
        --------
        dict
            Summary information
        """
        fits = results['fits']
        successful_fits = [f for f in fits if f['converged'] and f['gf'] != float('inf')]
        
        if not successful_fits:
            return {
                'error': 'No successful fits found',
                'total_fits': len(fits),
                'successful_fits': 0
            }
        
        # Sort by criterion
        if criterion.upper() == 'BIC':
            successful_fits.sort(key=lambda x: x['bic'])
        else:
            successful_fits.sort(key=lambda x: x['gf'])
        
        # Calculate statistics
        gf_values = [f['gf'] for f in successful_fits]
        bic_values = [f['bic'] for f in successful_fits]
        
        summary = {
            'total_fits': len(fits),
            'successful_fits': len(successful_fits),
            'failed_fits': len(fits) - len(successful_fits),
            'success_rate': len(successful_fits) / len(fits),
            'best_fit': {
                'rank': 1,
                'gf': successful_fits[0]['gf'],
                'bic': successful_fits[0]['bic'],
                'params': successful_fits[0]['final_params']
            },
            'statistics': {
                'gf': {
                    'min': min(gf_values),
                    'max': max(gf_values),
                    'mean': np.mean(gf_values),
                    'std': np.std(gf_values),
                    'median': np.median(gf_values)
                },
                'bic': {
                    'min': min(bic_values),
                    'max': max(bic_values),
                    'mean': np.mean(bic_values),
                    'std': np.std(bic_values),
                    'median': np.median(bic_values)
                }
            },
            'top_fits': successful_fits[:n_top],
            'criterion_used': criterion.upper()
        }
        
        # Count parameters near bounds
        total_near_bounds = 0
        for fit in successful_fits:
            if 'near_bounds' in fit:
                near_count = sum(1 for info in fit['near_bounds'].values() 
                            if isinstance(info, dict) and info.get('near_either', False))
                total_near_bounds += near_count
        
        summary['parameters_near_bounds'] = {
            'total_instances': total_near_bounds,
            'average_per_fit': total_near_bounds / len(successful_fits) if successful_fits else 0
        }
        
        return summary

    def export_batch_results_table(self, results, filename=None, format='csv', 
                                include_all_params=False, n_fits=None):
        """
        Export batch fitting results to a table file.
        
        Parameters:
        -----------
        results : dict
            Results from batch_initialize_and_fit
        filename : str, optional
            Output filename. If None, auto-generates
        format : str, optional
            Output format ('csv', 'xlsx', or 'json'). Default: 'csv'
        include_all_params : bool, optional
            Whether to include all parameter values. Default: False (summary only)
        n_fits : int, optional
            Number of fits to export. If None, exports all
            
        Returns:
        --------
        str
            Filename where table was exported
        """
        fits = results['fits']
        
        if n_fits is not None:
            fits = fits[:n_fits]
        
        # Create table data
        table_data = []
        
        for fit in fits:
            row = {
                'Rank': fit['rank'],
                'Fit_ID': fit['fit_id'],
                'GF': fit['gf'],
                'BIC': fit['bic'],
                'Converged': fit['converged']
            }
            
            # Add initialization values
            for param, value in fit['initialization'].items():
                row[f'Init_{param}'] = value
            
            # Add final parameter values
            if fit['final_params'] is not None and include_all_params:
                if self.geometry in ['trapezoid', 'sige']:
                    for i, trap in enumerate(fit['final_params']['trapezoids']):
                        row[f'Final_trap_{i}_width'] = trap['width']
                        if 'height' in trap:
                            row[f'Final_trap_{i}_height'] = trap['height']
                elif self.geometry == 'cylinder':
                    for i, cyl in enumerate(fit['final_params']['cylinders']):
                        row[f'Final_cyl_{i}_radius'] = cyl['radius']
                        if 'height' in cyl:
                            row[f'Final_cyl_{i}_height'] = cyl['height']
                
                # Add global parameters
                row['Final_DW'] = fit['final_params']['DW']
                row['Final_I0'] = fit['final_params']['I0']
                row['Final_Bk'] = fit['final_params']['Bk']
            
            # Add near bounds information
            if 'near_bounds' in fit:
                near_bound_count = sum(1 for info in fit['near_bounds'].values() 
                                    if isinstance(info, dict) and info.get('near_either', False))
                row['Params_Near_Bounds'] = near_bound_count
                
                # List parameters near bounds
                near_params = [name for name, info in fit['near_bounds'].items() 
                            if isinstance(info, dict) and info.get('near_either', False)]
                row['Near_Bounds_List'] = '; '.join(near_params) if near_params else ''
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"batch_results_{self.geometry}_{self.layers}L"
        
        # Export based on format
        if format.lower() == 'csv':
            full_filename = f"{filename}.csv"
            df.to_csv(full_filename, index=False)
        elif format.lower() == 'xlsx':
            full_filename = f"{filename}.xlsx"
            df.to_excel(full_filename, index=False)
        elif format.lower() == 'json':
            full_filename = f"{filename}.json"
            df.to_json(full_filename, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Table exported to: {full_filename}")
        print(f"Exported {len(df)} fits with {len(df.columns)} columns")
        
        return full_filename

    def compare_batch_results(self, results_list, labels=None, figsize=(12, 8)):
        """
        Compare multiple batch fitting results.
        
        Parameters:
        -----------
        results_list : list
            List of results dictionaries from batch_initialize_and_fit
        labels : list, optional
            Labels for each result set
        figsize : tuple, optional
            Figure size
        """
        if labels is None:
            labels = [f"Batch {i+1}" for i in range(len(results_list))]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
        
        for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
            fits = results['fits']
            successful_fits = [f for f in fits if f['converged'] and f['gf'] != float('inf')]
            
            if not successful_fits:
                continue
            
            gf_values = [f['gf'] for f in successful_fits]
            bic_values = [f['bic'] for f in successful_fits]
            ranks = list(range(1, len(successful_fits) + 1))
            
            # Plot 1: GF vs Rank
            ax1.semilogy(ranks[:20], gf_values[:20], 'o-', color=color, label=label, alpha=0.7)
            ax1.set_xlabel('Rank')
            ax1.set_ylabel('Goodness of Fit (GF)')
            ax1.set_title('GF vs Rank (Top 20)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: BIC vs Rank
            ax2.semilogy(ranks[:20], bic_values[:20], 'o-', color=color, label=label, alpha=0.7)
            ax2.set_xlabel('Rank')
            ax2.set_ylabel('BIC')
            ax2.set_title('BIC vs Rank (Top 20)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: GF Distribution
            ax3.hist(gf_values, bins=20, alpha=0.6, color=color, label=label, density=True)
            ax3.set_xlabel('Goodness of Fit (GF)')
            ax3.set_ylabel('Density')
            ax3.set_title('GF Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Success Rate and Statistics
            stats = {
                'Total Fits': len(fits),
                'Successful': len(successful_fits),
                'Success Rate': len(successful_fits) / len(fits),
                'Best GF': min(gf_values),
                'Best BIC': min(bic_values)
            }
            
            y_pos = len(results_list) - i - 1
            ax4.text(0.1, y_pos, f"{label}:", fontweight='bold', color=color)
            ax4.text(0.3, y_pos, f"Success: {stats['Success Rate']:.1%} ({stats['Successful']}/{stats['Total Fits']})")
            ax4.text(0.7, y_pos, f"Best GF: {stats['Best GF']:.4f}")
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(-0.5, len(results_list) - 0.5)
        ax4.set_title('Batch Comparison Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
    def CDSAXS_Optimize(self, params_to_optimize=None, optimizer='differential_evolution', 
                       plot_results=True, plot_structure=True, plot_grid=True, 
                       plot_combined=False, verbose=False, use_callbacks=False, 
                       callback_frequency=10, **kwargs):
        """
        Flexible optimization method for CDSAXS model fitting with optional callback monitoring.
        
        Parameters:
        -----------
        params_to_optimize : dict, optional
            Dictionary containing parameters to optimize with their bounds
        optimizer : str, optional
            Optimizer to use ('differential_evolution', 'dual_annealing', etc.). Default: 'differential_evolution'
        plot_results : bool, optional
            Whether to generate before/after comparison plots. Default: True
        plot_structure : bool, optional
            Whether to plot structure comparison. Default: True
        plot_grid : bool, optional
            Whether to plot grid of individual cuts. Default: True
        plot_combined : bool, optional
            Whether to plot combined view with all cuts. Default: True
        verbose : bool, optional
            Whether to print detailed output. Default: False
        use_callbacks : bool, optional
            Whether to enable callback monitoring. Default: False
        callback_frequency : int, optional
            Print progress every N iterations when using callbacks. Default: 10
        **kwargs : dict
            Additional arguments passed to the scipy optimizer
            
        Returns:
        --------
        dict or None
            Optimized model parameters or None if failed
        """
        try:
            # Check if required attributes exist
            if not hasattr(self, 'Intensity'):
                raise AttributeError("Missing required attribute: Intensity")
                
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                if self.geometry == 'cylinder' and hasattr(self, 'Qy'):
                    self.convert_Cartesian_Cylindrical()
                else:
                    raise AttributeError("Missing required scattering vector attributes")
            
            # Initialize optimization parameters if needed
            if not hasattr(self, 'model_params') or 'optimization' not in self.model_params:
                self.initialize_optimization_params()
            
            # Determine parameters to optimize
            if params_to_optimize is None:
                params_to_optimize = self.model_params['optimization']
            
            # Ensure all parameters have default values
            params_to_optimize = self._ensure_defaults_in_params(params_to_optimize)
            
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
            
            # Setup callbacks if requested
            if use_callbacks:
                self._callback_enabled = True
                self._callback_print_frequency = callback_frequency
                self._clear_callback_data()
                
                # Add appropriate callback to kwargs based on optimizer
                if optimizer == 'differential_evolution':
                    kwargs['callback'] = self._create_differential_evolution_callback()
                elif optimizer == 'dual_annealing':
                    kwargs['callback'] = self._create_dual_annealing_callback()
                
                if verbose:
                    print(f"Callbacks enabled for {optimizer} (print every {callback_frequency} iterations)")
            else:
                self._callback_enabled = False
            
            # Store current parameters and simulation results for before/after comparison
            initial_model_params = copy.deepcopy(self.model_params)
            
            # Calculate initial simulated intensity if not already done
            if not hasattr(self, 'SimInt') or self.SimInt is None:
                self.SimInt = self.simulate_structure()
                
            # Store initial simulation results
            initial_simInt = copy.deepcopy(self.SimInt)
            self._initial_model_params = initial_model_params
            
            # Calculate initial goodness of fit if not already done
            if not hasattr(self, 'GF_Initial') or self.GF_Initial is None:
                self.GF_Initial = self.GF_calc(self.SimInt)
            
            # Choose wrapper function based on geometry
            if self.geometry == 'cylinder':
                wrapper_func = self._cylinder_optimization_wrapper
            elif self.geometry in ['trapezoid', 'sige']:
                wrapper_func = self._trapezoid_optimization_wrapper
            else:
                raise ValueError(f"Unsupported geometry: {self.geometry}")
            
            # Run optimization
            if verbose:
                print(f"Starting optimization with {optimizer} using {len(param_names)} parameters...")
            
            result = self._run_scipy_optimizer(
                optimizer, wrapper_func, bounds, initial_values, verbose, **kwargs
            )
            
            # Store the optimization result
            self.optimization_result = result
            
            # Update model parameters with optimized values
            optimized_params = self._update_model_with_optimization_result(
                result, param_names, initial_model_params
            )
            
            # Update class attributes with optimized values
            self.model_params = optimized_params
            self.update_traditional_from_model_params()
            
            # Simulate with optimized parameters
            self.SimInt = self.simulate_structure()
            
            # Calculate goodness of fit and BIC
            self.GF = self.GF_calc(self.SimInt)
            self.BIC = self.BIC_calc(self.GF)
            
            # Print callback summary if callbacks were used
            if use_callbacks and verbose:
                self._print_callback_summary()
            
            # Generate callback plots if callbacks were used
            if use_callbacks:
                self._plot_callback_results()
            
            # Generate before/after comparison plots if requested
            if plot_results:
                self._plot_optimization_results(initial_model_params, initial_simInt,
                                            plot_structure, plot_grid, plot_combined)
            
            # Print parameter changes only if verbose
            if verbose:
                self.print_parameter_changes(initial_model_params)
            
            return self.model_params
                
        except Exception as e:
            if verbose:
                print(f"Error in CDSAXS_Optimize: {str(e)}")
                import traceback
                traceback.print_exc()
            return None
        
        
    def _run_scipy_optimizer(self, optimizer, objective_func, bounds, initial_values, verbose, **kwargs):
        """
        Run the specified scipy optimizer with appropriate parameters.
        
        Parameters:
        -----------
        optimizer : str
            Name of the scipy optimizer
        objective_func : callable
            Objective function to minimize
        bounds : list
            Parameter bounds
        initial_values : list
            Initial parameter values
        verbose : bool
            Whether to print progress
        **kwargs : dict
            Additional optimizer-specific arguments
            
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization result object
        """
        
        if optimizer == 'differential_evolution':
            # Default parameters for differential_evolution
            default_params = {
                'polish': True,
                'x0': np.array(initial_values),
                'maxiter': 100,
                'popsize': 15
            }
            default_params.update(kwargs)
            
            result = differential_evolution(
                objective_func, bounds, **default_params
            )
            
        elif optimizer == 'dual_annealing':
            # Default parameters for dual_annealing
            default_params = {
                'x0': np.array(initial_values),
                'maxiter': 1000,
                #'local_search_options': {'method': 'L-BFGS-B'}
            }
            default_params.update(kwargs)
            
            result = dual_annealing(
                objective_func, bounds, **default_params
            )
            
        elif optimizer == 'shgo':
            # Default parameters for SHGO (Simplicial Homology Global Optimization)
            default_params = {
                'n': 100,  # Number of sampling points
                'iters': 3,  # Number of iterations
                'sampling_method': 'sobol'
            }
            default_params.update(kwargs)
            
            result = shgo(
                objective_func, bounds, **default_params
            )
            
        elif optimizer == 'basinhopping':
            # Basin hopping requires an initial point and local minimizer
            default_params = {
                'niter': 100,
                'T': 1.0,
                'stepsize': 0.5,
                'minimizer_kwargs': {
                    'method': 'L-BFGS-B',
                    'bounds': bounds
                }
            }
            default_params.update(kwargs)
            
            # Start from initial values
            x0 = np.array(initial_values)
            
            result = basinhopping(
                objective_func, x0, **default_params
            )
            
        elif optimizer == 'minimize':
            # Local optimization - requires method to be specified
            method = kwargs.pop('method', 'L-BFGS-B')
            
            default_params = {
                'method': method,
                'bounds': bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                'options': {'maxiter': 1000}
            }
            default_params.update(kwargs)
            
            x0 = np.array(initial_values)
            
            result = minimize(
                objective_func, x0, **default_params
            )
            
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. "
                            f"Supported options: 'differential_evolution', 'dual_annealing', "
                            f"'shgo', 'basinhopping', 'minimize'")
        
        return result
    
    def _update_model_with_optimization_result(self, result, param_names, initial_model_params):
        """
        Update model parameters with optimization results.
        
        Parameters:
        -----------
        result : scipy.optimize.OptimizeResult
            Optimization result
        param_names : list
            List of parameter names
        initial_model_params : dict
            Initial model parameters
            
        Returns:
        --------
        dict
            Updated model parameters
        """
        optimized_params = copy.deepcopy(initial_model_params)
        
        # Handle different result types
        if hasattr(result, 'x'):
            optimal_values = result.x
        elif hasattr(result, 'best_x'):  # Some optimizers use this
            optimal_values = result.best_x
        else:
            raise ValueError("Could not extract optimal values from optimization result")
        
        # Update parameters based on geometry
        if self.geometry in ['trapezoid', 'sige']:
            # Make a deep copy of trapezoids to avoid modifying the original
            optimized_params['trapezoids'] = [trap.copy() for trap in initial_model_params['trapezoids']]
            
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
                    
                    optimized_params['trapezoids'][trap_idx][param_type] = optimal_values[i]
                elif param_name.startswith('Bk_'):
                    # Background parameter for specific column
                    bk_idx = int(param_name.split('_')[1])
                    if isinstance(optimized_bk, np.ndarray):
                        optimized_bk[bk_idx] = optimal_values[i]
                    else:
                        # Convert scalar to array if needed
                        n_columns = self.Intensity.shape[1]
                        optimized_bk = np.full(n_columns, optimized_bk)
                        optimized_bk[bk_idx] = optimal_values[i]
                elif param_name == 'Bk':
                    # Scalar background parameter
                    optimized_bk = optimal_values[i]
                else:
                    # Global parameter (DW, I0)
                    optimized_params[param_name] = optimal_values[i]
            
            # Update background in optimized parameters
            optimized_params['Bk'] = optimized_bk.tolist() if isinstance(optimized_bk, np.ndarray) else optimized_bk
            
        elif self.geometry == 'cylinder':
            # Make a deep copy of cylinders to avoid modifying the original
            optimized_params['cylinders'] = [cyl.copy() for cyl in initial_model_params['cylinders']]
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('cyl_'):
                    # Parse cylinder parameter
                    parts = param_name.split('_')
                    cyl_idx = int(parts[1])
                    param_type = parts[2]  # 'radius' or 'height'
                    
                    optimized_params['cylinders'][cyl_idx][param_type] = optimal_values[i]
                else:
                    # Global parameter (DW, I0, Bk)
                    optimized_params[param_name] = optimal_values[i]
        
        return optimized_params
    
    
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import corner  # For corner plots
    from tqdm import tqdm
    import warnings

    def CDSAXS_MCMC(self, params_to_sample=None, n_walkers=50, n_steps=1000, 
                    burn_in=200, thin=1, progress=True, plot_results=True,
                    plot_chains=True, plot_corner=True, plot_structure=False,
                    save_chains=False, chain_filename=None, verbose=True,
                    prior_type='uniform', sigma_multiplier=10.0, **emcee_kwargs):
        """
        Perform MCMC sampling using emcee to estimate parameters and uncertainties.
        
        Parameters:
        -----------
        params_to_sample : dict, optional
            Dictionary containing parameters to sample with their bounds
            If None, uses self.model_params['optimization']
        n_walkers : int, optional
            Number of MCMC walkers. Default: 50
        n_steps : int, optional
            Number of MCMC steps per walker. Default: 1000
        burn_in : int, optional
            Number of burn-in steps to discard. Default: 200
        thin : int, optional
            Thinning factor for chains. Default: 1 (no thinning)
        progress : bool, optional
            Whether to show progress bar. Default: True
        plot_results : bool, optional
            Whether to generate plots. Default: True
        plot_chains : bool, optional
            Whether to plot walker chains. Default: True
        plot_corner : bool, optional
            Whether to plot corner plot. Default: True
        plot_structure : bool, optional
            Whether to plot structure with uncertainties. Default: True
        save_chains : bool, optional
            Whether to save chains to file. Default: False
        chain_filename : str, optional
            Filename for saved chains
        verbose : bool, optional
            Whether to print detailed output. Default: True
        prior_type : str, optional
            Type of prior: 'uniform', 'gaussian'. Default: 'uniform'
        sigma_multiplier : float, optional
            For Gaussian priors: std = (max-min)/sigma_multiplier. Default: 10.0
        **emcee_kwargs : dict
            Additional arguments passed to emcee.EnsembleSampler
            
        Returns:
        --------
        dict
            Dictionary containing MCMC results, chains, and statistics
        """
        try:
            # Check if emcee is available
            try:
                import emcee
            except ImportError:
                raise ImportError("emcee package is required. Install with: pip install emcee")
            
            # Check if required attributes exist
            if not hasattr(self, 'Intensity'):
                raise AttributeError("Missing required attribute: Intensity")
                
            if not hasattr(self, 'Qx') or not hasattr(self, 'Qz'):
                if self.geometry == 'cylinder' and hasattr(self, 'Qy'):
                    self.convert_Cartesian_Cylindrical()
                else:
                    raise AttributeError("Missing required scattering vector attributes")
            
            # Initialize optimization parameters if needed
            if not hasattr(self, 'model_params') or 'optimization' not in self.model_params:
                self.initialize_optimization_params()
            
            # Determine parameters to sample
            if params_to_sample is None:
                params_to_sample = self.model_params['optimization']
            
            # Ensure all parameters have default values
            params_to_sample = self._ensure_defaults_in_params(params_to_sample)
            
            # Setup MCMC parameters
            param_names = list(params_to_sample.keys())
            n_params = len(param_names)
            
            if verbose:
                print(f"Setting up MCMC with {n_params} parameters and {n_walkers} walkers")
                print(f"Parameters to sample: {param_names}")
            
            # Store parameter info
            self.mcmc_param_names = param_names
            self.mcmc_param_info = params_to_sample
            
            # Setup priors and initial positions
            bounds, initial_positions, log_prior_func = self._setup_mcmc_priors(
                params_to_sample, n_walkers, prior_type, sigma_multiplier
            )
            
            # Create log probability function
            def log_probability(theta):
                # Check priors
                lp = log_prior_func(theta)
                if not np.isfinite(lp):
                    return -np.inf
                
                # Calculate likelihood
                ll = self._mcmc_log_likelihood(theta)
                if not np.isfinite(ll):
                    return -np.inf
                    
                return lp + ll
            
            # Handle thin parameter: extract from emcee_kwargs if present, function param takes precedence
            emcee_kwargs_clean = emcee_kwargs.copy()
            if 'thin' in emcee_kwargs_clean:
                # Function parameter thin takes precedence
                del emcee_kwargs_clean['thin']
            # Use function parameter thin (defaults to 1)
            thin_to_use = thin
            
            # Initialize sampler
            sampler = emcee.EnsembleSampler(
                n_walkers, n_params, log_probability, **emcee_kwargs
            )
            
            if verbose:
                print(f"Running MCMC: {n_steps} steps with {n_walkers} walkers")
                print(f"Burn-in: {burn_in} steps, Thinning: {thin}")
            
            # Run MCMC
            if progress:
                # Run with progress bar
                with tqdm(total=n_steps, desc="MCMC Progress") as pbar:
                    for i, state in enumerate(sampler.sample(initial_positions, iterations=n_steps)):
                        pbar.update(1)
                        if i % 100 == 0 and verbose:
                            acceptance = np.mean(sampler.acceptance_fraction)
                            pbar.set_postfix({"Accept": f"{acceptance:.3f}"})
            else:
                # Run without progress bar
                sampler.run_mcmc(initial_positions, n_steps)
            
            # Extract results
            chains = sampler.get_chain()
            log_prob = sampler.get_log_prob()
            
            # Apply burn-in and thinning
            if burn_in > 0:
                chains_burned = chains[burn_in:]
                log_prob_burned = log_prob[burn_in:]
            else:
                chains_burned = chains
                log_prob_burned = log_prob
            
            if thin > 1:
                chains_final = chains_burned[::thin]
                log_prob_final = log_prob_burned[::thin]
            else:
                chains_burned = chains
                log_prob_burned = log_prob
            
            # Thinning is already handled by emcee, so no need for post-processing thinning
            chains_final = chains_burned
            log_prob_final = log_prob_burned
            
            # Flatten chains for analysis
            flat_chains = chains_final.reshape(-1, n_params)
            flat_log_prob = log_prob_final.flatten()
            
            # Calculate statistics
            param_stats = self._calculate_mcmc_statistics(flat_chains, param_names)
            
            # Find best-fit parameters
            best_idx = np.argmax(flat_log_prob)
            best_params = flat_chains[best_idx]
            
            # Create results dictionary
            results = {
                'chains': chains,
                'chains_burned': chains_burned,
                'chains_final': chains_final,
                'flat_chains': flat_chains,
                'log_prob': log_prob,
                'log_prob_final': log_prob_final,
                'param_names': param_names,
                'param_stats': param_stats,
                'best_params': best_params,
                'best_log_prob': flat_log_prob[best_idx],
                'n_walkers': n_walkers,
                'n_steps': n_steps,
                'burn_in': burn_in,
                'thin': thin,
                'acceptance_fraction': sampler.acceptance_fraction,
                'mean_acceptance': np.mean(sampler.acceptance_fraction),
                'autocorr_time': None,  # Will calculate if possible
                'effective_samples': len(flat_chains)
            }
            
            # Calculate autocorrelation time if possible
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    autocorr_time = sampler.get_autocorr_time(quiet=True)
                    results['autocorr_time'] = autocorr_time
                    results['mean_autocorr_time'] = np.mean(autocorr_time)
            except Exception:
                if verbose:
                    print("Warning: Could not calculate autocorrelation time")
            
            # Apply best-fit parameters to model
            self._apply_mcmc_parameters(best_params, param_names)
            
            # Print summary
            if verbose:
                self._print_mcmc_summary(results)
            
            # Generate plots
            if plot_results:
                self._plot_mcmc_results(results, plot_chains, plot_corner, plot_structure)
            
            # Save chains if requested
            if save_chains:
                filename = self._save_mcmc_chains(results, chain_filename)
                if verbose:
                    print(f"Chains saved to: {filename}")
            
            return results
            
        except Exception as e:
            print(f"Error in CDSAXS_MCMC: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _setup_mcmc_priors(self, params_to_sample, n_walkers, prior_type, sigma_multiplier):
        """
        Setup priors and initial walker positions for MCMC.
        
        Parameters:
        -----------
        params_to_sample : dict
            Parameters to sample with bounds
        n_walkers : int
            Number of walkers
        prior_type : str
            Type of prior ('uniform' or 'gaussian')
        sigma_multiplier : float
            For Gaussian priors
            
        Returns:
        --------
        tuple
            (bounds, initial_positions, log_prior_function)
        """
        param_names = list(params_to_sample.keys())
        n_params = len(param_names)
        
        bounds = []
        defaults = []
        
        for param_name in param_names:
            param_info = params_to_sample[param_name]
            bounds.append((param_info['min'], param_info['max']))
            defaults.append(param_info['default'])
        
        bounds = np.array(bounds)
        defaults = np.array(defaults)
        
        # Generate initial positions
        if prior_type == 'uniform':
            # Uniform distribution around default values
            widths = bounds[:, 1] - bounds[:, 0]
            initial_positions = []
            
            for _ in range(n_walkers):
                pos = defaults + 0.1 * widths * (np.random.random(n_params) - 0.5)
                # Ensure within bounds
                pos = np.clip(pos, bounds[:, 0], bounds[:, 1])
                initial_positions.append(pos)
            
            initial_positions = np.array(initial_positions)
            
            # Define uniform log prior
            def log_prior_uniform(theta):
                if np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])):
                    return 0.0
                else:
                    return -np.inf
                    
            log_prior_func = log_prior_uniform
            
        elif prior_type == 'gaussian':
            # Gaussian priors centered on defaults
            sigmas = (bounds[:, 1] - bounds[:, 0]) / sigma_multiplier
            
            # Generate initial positions from Gaussian around defaults
            initial_positions = []
            for _ in range(n_walkers):
                pos = np.random.normal(defaults, sigmas * 0.5)
                # Ensure within bounds
                pos = np.clip(pos, bounds[:, 0], bounds[:, 1])
                initial_positions.append(pos)
            
            initial_positions = np.array(initial_positions)
            
            # Define Gaussian log prior
            def log_prior_gaussian(theta):
                if np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])):
                    # Gaussian prior
                    log_prior = -0.5 * np.sum(((theta - defaults) / sigmas) ** 2)
                    return log_prior
                else:
                    return -np.inf
                    
            log_prior_func = log_prior_gaussian
            
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
        
        return bounds, initial_positions, log_prior_func

    def _mcmc_log_likelihood(self, theta):
        """
        Calculate log likelihood for MCMC with better error handling.
        
        Parameters:
        -----------
        theta : array_like
            Parameter values
            
        Returns:
        --------
        float
            Log likelihood
        """
        try:
            # Ensure param_names is available for the wrapper functions
            if not hasattr(self, 'param_names') and not hasattr(self, 'mcmc_param_names'):
                if hasattr(self, 'model_params') and 'optimization' in self.model_params:
                    self.mcmc_param_names = list(self.model_params['optimization'].keys())
                else:
                    return -np.inf
            
            # Choose wrapper function based on geometry
            if self.geometry == 'cylinder':
                gf = self._cylinder_optimization_wrapper(theta)
            elif self.geometry in ['trapezoid', 'sige']:
                gf = self._trapezoid_optimization_wrapper(theta)
            else:
                return -np.inf
            
            if not np.isfinite(gf) or gf <= 0:
                return -np.inf
            
            # Convert goodness of fit to log likelihood
            # Assuming Chi-square likelihood: log_likelihood = -0.5 * chi2
            log_likelihood = -0.5 * gf
            
            return log_likelihood
            
        except Exception as e:
            # Don't print errors during MCMC as it will spam the output
            return -np.inf

    def _apply_mcmc_parameters(self, params, param_names):
        """
        Apply MCMC parameter values to the model.
        
        Parameters:
        -----------
        params : array_like
            Parameter values
        param_names : list
            Parameter names
        """
        # Create a copy of current model parameters
        updated_params = copy.deepcopy(self.model_params)
        
        # Update parameters based on geometry
        if self.geometry in ['trapezoid', 'sige']:
            # Make a deep copy of trapezoids
            updated_params['trapezoids'] = [trap.copy() for trap in self.model_params['trapezoids']]
            
            # Initialize background
            if isinstance(self.Bk, np.ndarray):
                updated_bk = self.Bk.copy()
            else:
                updated_bk = self.Bk
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('trap_'):
                    parts = param_name.split('_')
                    trap_idx = int(parts[1])
                    param_type = parts[2]
                    updated_params['trapezoids'][trap_idx][param_type] = params[i]
                elif param_name.startswith('Bk_'):
                    bk_idx = int(param_name.split('_')[1])
                    if isinstance(updated_bk, np.ndarray):
                        updated_bk[bk_idx] = params[i]
                    else:
                        n_columns = self.Intensity.shape[1]
                        updated_bk = np.full(n_columns, updated_bk)
                        updated_bk[bk_idx] = params[i]
                elif param_name == 'Bk':
                    updated_bk = params[i]
                else:
                    updated_params[param_name] = params[i]
            
            # Update background
            updated_params['Bk'] = updated_bk.tolist() if isinstance(updated_bk, np.ndarray) else updated_bk
            
        elif self.geometry == 'cylinder':
            # Make a deep copy of cylinders
            updated_params['cylinders'] = [cyl.copy() for cyl in self.model_params['cylinders']]
            
            for i, param_name in enumerate(param_names):
                if param_name.startswith('cyl_'):
                    parts = param_name.split('_')
                    cyl_idx = int(parts[1])
                    param_type = parts[2]
                    updated_params['cylinders'][cyl_idx][param_type] = params[i]
                else:
                    updated_params[param_name] = params[i]
        
        # Apply updated parameters
        self.model_params = updated_params
        self.update_traditional_from_model_params()
        
        # Update simulation
        self.SimInt = self.simulate_structure()
        self.GF = self.GF_calc(self.SimInt)
        self.BIC = self.BIC_calc(self.GF)

    def _calculate_mcmc_statistics(self, flat_chains, param_names):
        """
        Calculate statistics from MCMC chains.
        
        Parameters:
        -----------
        flat_chains : ndarray
            Flattened MCMC chains
        param_names : list
            Parameter names
            
        Returns:
        --------
        dict
            Dictionary of parameter statistics
        """
        param_stats = {}
        
        for i, param_name in enumerate(param_names):
            chain = flat_chains[:, i]
            
            # Calculate percentiles
            percentiles = np.percentile(chain, [2.5, 16, 50, 84, 97.5])
            
            param_stats[param_name] = {
                'mean': np.mean(chain),
                'median': percentiles[2],
                'std': np.std(chain),
                'percentile_2.5': percentiles[0],
                'percentile_16': percentiles[1],
                'percentile_84': percentiles[3],
                'percentile_97.5': percentiles[4],
                'confidence_68': [percentiles[1], percentiles[3]],
                'confidence_95': [percentiles[0], percentiles[4]],
                'samples': chain
            }
        
        return param_stats

    def _print_mcmc_summary(self, results):
        """
        Print a summary of MCMC results.
        
        Parameters:
        -----------
        results : dict
            MCMC results dictionary
        """
        print(f"\n{'='*80}")
        print(f"MCMC SAMPLING SUMMARY")
        print(f"{'='*80}")
        
        print(f"Walkers: {results['n_walkers']}")
        print(f"Steps: {results['n_steps']} (burn-in: {results['burn_in']}, thin: {results['thin']})")
        print(f"Effective samples: {results['effective_samples']}")
        print(f"Mean acceptance fraction: {results['mean_acceptance']:.3f}")
        
        if results['autocorr_time'] is not None:
            print(f"Mean autocorrelation time: {results['mean_autocorr_time']:.1f}")
            
            # Check convergence
            n_effective = results['n_steps'] - results['burn_in']
            if results['mean_autocorr_time'] > 0:
                n_independent = n_effective / results['mean_autocorr_time']
                print(f"Independent samples per walker: ~{n_independent:.0f}")
                
                if n_independent < 50:
                    print("⚠️  Warning: Low number of independent samples. Consider longer chains.")
                elif n_independent > 100:
                    print("✓ Good number of independent samples")
        
        print(f"\nBest-fit log probability: {results['best_log_prob']:.3f}")
        print(f"Best-fit GF: {self.GF:.6f}")
        print(f"Best-fit BIC: {self.BIC:.6f}")
        
        print(f"\nParameter Estimates (68% confidence intervals):")
        print(f"{'Parameter':<20} {'Median':<12} {'68% CI':<20} {'95% CI':<20}")
        print("-" * 80)
        
        for param_name, stats in results['param_stats'].items():
            median = stats['median']
            ci_68 = stats['confidence_68']
            ci_95 = stats['confidence_95']
            
            ci_68_str = f"[{ci_68[0]:.4f}, {ci_68[1]:.4f}]"
            ci_95_str = f"[{ci_95[0]:.4f}, {ci_95[1]:.4f}]"
            
            print(f"{param_name:<20} {median:<12.4f} {ci_68_str:<20} {ci_95_str:<20}")
        
        print(f"{'='*80}")

    def _plot_mcmc_results(self, results, plot_chains=True, plot_corner=True, plot_structure=True):
        """
        Generate plots for MCMC results.
        
        Parameters:
        -----------
        results : dict
            MCMC results
        plot_chains : bool
            Whether to plot walker chains
        plot_corner : bool
            Whether to plot corner plot
        plot_structure : bool
            Whether to plot structure with uncertainties
        """
        
        if plot_chains:
            self._plot_mcmc_chains(results)
        
        if plot_corner:
            self._plot_mcmc_corner(results)
        
        if plot_structure:
            self._plot_mcmc_structure_uncertainty(results)

    def _plot_mcmc_chains(self, results):
        """
        Plot MCMC walker chains to check convergence.
        
        Parameters:
        -----------
        results : dict
            MCMC results
        """
        chains = results['chains']
        param_names = results['param_names']
        burn_in = results['burn_in']
        n_params = len(param_names)
        
        # Create subplots
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 2.5 * n_params), sharex=True)
        if n_params == 1:
            axes = [axes]
        
        for i, (ax, param_name) in enumerate(zip(axes, param_names)):
            # Plot all walker chains
            for walker in range(results['n_walkers']):
                ax.plot(chains[:, walker, i], alpha=0.3, color='steelblue', linewidth=0.5)
            
            # Mark burn-in
            if burn_in > 0:
                ax.axvline(burn_in, color='red', linestyle='--', alpha=0.7, label='Burn-in')
            
            ax.set_ylabel(param_name)
            if i == 0 and burn_in > 0:
                ax.legend()
            
            # Add statistics
            stats = results['param_stats'][param_name]
            ax.axhline(stats['median'], color='orange', linestyle='-', alpha=0.8, linewidth=1)
            ax.axhline(stats['confidence_68'][0], color='orange', linestyle=':', alpha=0.6)
            ax.axhline(stats['confidence_68'][1], color='orange', linestyle=':', alpha=0.6)
        
        axes[-1].set_xlabel('Step')
        plt.suptitle('MCMC Walker Chains', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _plot_mcmc_corner(self, results):
        """
        Plot corner plot showing parameter correlations.
        
        Parameters:
        -----------
        results : dict
            MCMC results
        """
        try:
            import corner
        except ImportError:
            print("Corner package not available. Install with: pip install corner")
            return
        
        flat_chains = results['flat_chains']
        param_names = results['param_names']
        
        # Create labels with units (customize as needed)
        labels = []
        for name in param_names:
            if 'width' in name or 'radius' in name or 'height' in name:
                labels.append(f"{name} (Å)")
            elif name == 'DW':
                labels.append("DW (Å)")
            elif name == 'I0':
                labels.append("I0")
            elif name.startswith('Bk'):
                labels.append(f"{name}")
            else:
                labels.append(name)
        
        # Calculate quantiles for plotting
        quantiles = [0.16, 0.5, 0.84]
        
        fig = corner.corner(
            flat_chains,
            labels=labels,
            quantiles=quantiles,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            color='steelblue',
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=1.0
        )
        
        plt.suptitle('Parameter Posterior Distributions', fontsize=16, y=0.98)
        plt.show()

    def _plot_mcmc_structure_uncertainty(self, results):
        """
        Fixed version: Plot structure with uncertainty bands from MCMC samples.
        
        Parameters:
        -----------
        results : dict
            MCMC results
        """
        # Sample parameter sets from posterior
        flat_chains = results['flat_chains']
        param_names = results['param_names']
        n_samples = min(100, len(flat_chains))  # Limit for performance
        
        # Randomly select parameter sets
        indices = np.random.choice(len(flat_chains), n_samples, replace=False)
        
        # Store original parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Calculate structures for sampled parameters
        structures_samples = []
        
        for idx in indices:
            params = flat_chains[idx]
            self._apply_mcmc_parameters(params, param_names)
            
            # Extract structure points for plotting
            if self.geometry in ['trapezoid', 'sige']:
                heights, widths = self._extract_width_height_relationship()
                structures_samples.append((heights, widths))
            elif self.geometry == 'cylinder':
                heights, radii = self._extract_width_height_relationship()
                structures_samples.append((heights, radii))
        
        # Restore original (best-fit) parameters
        self.model_params = original_params
        self.update_traditional_from_model_params()
        
        # Apply best-fit parameters
        best_params = results['best_params']
        self._apply_mcmc_parameters(best_params, param_names)
        
        # Plot structure uncertainty
        plt.figure(figsize=(10, 6))
        
        # Plot sample structures properly
        for heights, widths in structures_samples:
            if self.geometry in ['trapezoid', 'sige']:
                # Plot proper trapezoid shape
                base_width = widths[0]
                
                # Create trapezoid outline coordinates
                x_coords = []
                y_coords = []
                
                # Bottom edge
                x_coords.extend([-base_width/2, base_width/2])
                y_coords.extend([0, 0])
                
                # Right edge going up
                for i in range(len(heights)-1):
                    h1, h2 = heights[i], heights[i+1]
                    w1, w2 = widths[i], widths[i+1]
                    x_coords.extend([w1/2, w2/2])
                    y_coords.extend([h1, h2])
                
                # Top edge
                top_width = widths[-1]
                x_coords.extend([top_width/2, -top_width/2])
                y_coords.extend([heights[-1], heights[-1]])
                
                # Left edge going down
                for i in range(len(heights)-1, 0, -1):
                    h1, h2 = heights[i], heights[i-1]
                    w1, w2 = widths[i], widths[i-1]
                    x_coords.extend([-w1/2, -w2/2])
                    y_coords.extend([h1, h2])
                
                # Close the shape
                x_coords.append(-base_width/2)
                y_coords.append(0)
                
                plt.plot(x_coords, y_coords, 'b-', alpha=0.05, linewidth=0.5)
                    
            elif self.geometry == 'cylinder':
                # Plot cylinder outline (both sides)
                for i in range(len(heights)):
                    radius = widths[i]  # widths are actually radii for cylinders
                    height = heights[i]
                    plt.plot([-radius, radius], [height, height], 'b-', alpha=0.05, linewidth=0.5)
                    
                    # Connect layers with vertical lines
                    if i > 0:
                        prev_radius = widths[i-1]
                        prev_height = heights[i-1]
                        plt.plot([prev_radius, radius], [prev_height, height], 'b-', alpha=0.05, linewidth=0.5)
                        plt.plot([-prev_radius, -radius], [prev_height, height], 'b-', alpha=0.05, linewidth=0.5)
        
        # Plot best-fit structure on top (on the same axes)
        best_heights, best_widths = self._extract_width_height_relationship()
        
        if self.geometry in ['trapezoid', 'sige']:
            # Plot best-fit trapezoid in red
            base_width = best_widths[0]
            
            # Create trapezoid outline coordinates
            x_coords = []
            y_coords = []
            
            # Bottom edge
            x_coords.extend([-base_width/2, base_width/2])
            y_coords.extend([0, 0])
            
            # Right edge going up
            for i in range(len(best_heights)-1):
                h1, h2 = best_heights[i], best_heights[i+1]
                w1, w2 = best_widths[i], best_widths[i+1]
                x_coords.extend([w1/2, w2/2])
                y_coords.extend([h1, h2])
            
            # Top edge
            top_width = best_widths[-1]
            x_coords.extend([top_width/2, -top_width/2])
            y_coords.extend([best_heights[-1], best_heights[-1]])
            
            # Left edge going down
            for i in range(len(best_heights)-1, 0, -1):
                h1, h2 = best_heights[i], best_heights[i-1]
                w1, w2 = best_widths[i], best_widths[i-1]
                x_coords.extend([-w1/2, -w2/2])
                y_coords.extend([h1, h2])
            
            # Close the shape
            x_coords.append(-base_width/2)
            y_coords.append(0)
            
            plt.plot(x_coords, y_coords, 'r-', linewidth=2, label='Best fit')
            
        elif self.geometry == 'cylinder':
            # Plot best-fit cylinder in red
            for i in range(len(best_heights)):
                radius = best_widths[i]
                height = best_heights[i]
                plt.plot([-radius, radius], [height, height], 'r-', linewidth=2)
                
                # Connect layers with vertical lines
                if i > 0:
                    prev_radius = best_widths[i-1]
                    prev_height = best_heights[i-1]
                    plt.plot([prev_radius, radius], [prev_height, height], 'r-', linewidth=2)
                    plt.plot([-prev_radius, -radius], [prev_height, height], 'r-', linewidth=2)
        
        plt.title(f'Structure Uncertainty from MCMC\n({n_samples} posterior samples)', fontsize=14)
        plt.xlabel('Width/Radius (Å)')
        plt.ylabel('Height (Å)')
        
        # Add text with confidence info
        plt.text(0.02, 0.98, f'Blue envelope: Posterior uncertainty\nRed line: Best fit', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def _save_mcmc_chains(self, results, filename=None):
        """
        Save MCMC chains and results to file.
        
        Parameters:
        -----------
        results : dict
            MCMC results
        filename : str, optional
            Output filename
            
        Returns:
        --------
        str
            Filename where data was saved
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcmc_chains_{self.geometry}_{self.layers}L_{timestamp}.npz"
        
        # Save chains and key results
        np.savez_compressed(
            filename,
            chains=results['chains'],
            flat_chains=results['flat_chains'],
            log_prob=results['log_prob'],
            param_names=results['param_names'],
            best_params=results['best_params'],
            acceptance_fraction=results['acceptance_fraction'],
            autocorr_time=results['autocorr_time'] if results['autocorr_time'] is not None else np.array([]),
            n_walkers=results['n_walkers'],
            n_steps=results['n_steps'],
            burn_in=results['burn_in'],
            thin=results['thin']
        )
        
        return filename

    def load_mcmc_chains(filename):
        """
        Load MCMC chains from file.
        
        Parameters:
        -----------
        filename : str
            Filename to load
            
        Returns:
        --------
        dict
            Loaded MCMC results
        """
        data = np.load(filename, allow_pickle=True)
        
        results = {
            'chains': data['chains'],
            'flat_chains': data['flat_chains'],
            'log_prob': data['log_prob'],
            'param_names': data['param_names'].tolist(),
            'best_params': data['best_params'],
            'acceptance_fraction': data['acceptance_fraction'],
            'n_walkers': int(data['n_walkers']),
            'n_steps': int(data['n_steps']),
            'burn_in': int(data['burn_in']),
            'thin': int(data['thin'])
        }
        
        if 'autocorr_time' in data and len(data['autocorr_time']) > 0:
            results['autocorr_time'] = data['autocorr_time']
        else:
            results['autocorr_time'] = None
        
        return results
    



    def plot_mcmc_uncertainty_envelope(self, mcmc_results, n_samples=100, n_slices=101, 
                                 confidence_level=0.95, plot_results=True, 
                                 figsize=(10, 6), show_best_fit=True, show_mean=True,
                                 show_base=True, colors=None):
        """
        Plot uncertainty envelope around structure from emcee MCMC results.
        
        Parameters:
        -----------
        mcmc_results : dict
            Results from CDSAXS_MCMC containing chains and parameter info
        n_samples : int, optional
            Number of MCMC samples to use for uncertainty calculation. Default: 100
        n_slices : int, optional
            Number of height slices for uncertainty envelope. Default: 101
        confidence_level : float, optional
            Confidence level for uncertainty envelope (0.68 or 0.95). Default: 0.95
        plot_results : bool, optional
            Whether to plot the results. Default: True
        figsize : tuple, optional
            Figure size. Default: (10, 6)
        show_best_fit : bool, optional
            Whether to overlay the best-fit structure. Default: True
        show_mean : bool, optional
            Whether to show the mean structure. Default: True
        show_base : bool, optional
            Whether to show the base line at y=0. Default: True
        colors : dict, optional
            Custom colors for plotting. Keys: 'envelope', 'mean', 'best_fit', 'structure'
            
        Returns:
        --------
        tuple
            (center_line, inner_envelope, outer_envelope) arrays for plotting
        """
        
        # Set default colors
        if colors is None:
            colors = {
                'envelope': 'cornflowerblue',
                'mean': 'red',
                'best_fit': 'darkgreen',
                'structure': 'red'  # Same color for base and structure lines
            }
        
        # Extract flattened chains and parameter names
        flat_chains = mcmc_results['flat_chains']
        param_names = mcmc_results['param_names']
        
        # Limit number of samples for performance
        total_samples = len(flat_chains)
        if n_samples > total_samples:
            n_samples = total_samples
        
        # Randomly select samples for diversity
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        selected_samples = flat_chains[sample_indices]
        
        # Store original model parameters
        original_params = self.model_params.copy()
        
        # Initialize arrays for uncertainty calculation
        xi = np.zeros([n_slices, 2, n_samples])  # [slice, side(left/right), sample]
        yi = np.zeros([n_slices, 1, n_samples])  # [slice, 1, sample]
        
        try:
            # Process each MCMC sample
            for pop_number, sample_params in enumerate(selected_samples):
                # Apply MCMC parameters to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    trap_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        trap_heights[i + 1] = trap_heights[i] + heights[i]
                    total_height = trap_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_trapezoid_envelope_sample(
                        widths, heights, trap_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
                    
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    cyl_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        cyl_heights[i + 1] = cyl_heights[i] + heights[i]
                    total_height = cyl_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_cylinder_envelope_sample(
                        radii, heights, cyl_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
            
            # Calculate statistics across all samples
            center_line, inner_envelope, outer_envelope = self._calculate_uncertainty_statistics(
                xi, yi, confidence_level, n_slices
            )
            
            # Plot results if requested
            if plot_results:
                self._plot_uncertainty_envelope_enhanced(
                    center_line, inner_envelope, outer_envelope, figsize, 
                    show_best_fit, show_mean, show_base, colors, confidence_level, mcmc_results
                )
            
            return center_line, inner_envelope, outer_envelope
            
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()

    def _extract_structure_for_uncertainty(self):
        """
        Extract structure information for uncertainty calculation.
        
        Returns:
        --------
        tuple
            (heights, widths_or_radii) arrays
        """
        if self.geometry in ['trapezoid', 'sige']:
            structures = self.model_params['trapezoids']
            widths = [trap['width'] for trap in structures]
            heights = [trap['height'] for trap in structures[:-1]]  # Skip last (top)
            return heights, widths
        
        elif self.geometry == 'cylinder':
            structures = self.model_params['cylinders']
            radii = [cyl['radius'] for cyl in structures]
            heights = [cyl['height'] for cyl in structures[:-1]]  # Skip last (top)
            return heights, radii
        
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

    def _calculate_trapezoid_envelope_sample(self, widths, heights, trap_heights, total_height,
                                        xi, yi, pop_number, n_slices):
        """
        Calculate uncertainty envelope for a single trapezoid sample.
        """
        # Convert to coordinates similar to your SymCoordAssign function
        coord = np.zeros([len(widths), 2])  # [layer, left/right]
        
        for i in range(len(widths)):
            coord[i, 0] = -widths[i] / 2  # Left side
            coord[i, 1] = widths[i] / 2   # Right side
        
        # Calculate envelope for each height slice
        for c in range(n_slices):
            if total_height > 0:
                disc_height = c * total_height / (n_slices - 1)
            else:
                disc_height = 0
                
            yi[c, 0, pop_number] = disc_height
            
            # Find which trapezoid layer we're in
            layer_idx = 0
            for i in range(len(trap_heights) - 1):
                if disc_height >= trap_heights[i] and disc_height <= trap_heights[i + 1]:
                    layer_idx = i
                    break
            
            # Ensure we don't go out of bounds
            layer_idx = min(layer_idx, len(coord) - 2)
            layer_idx = max(layer_idx, 0)
            
            # Get coordinates for this layer
            if layer_idx < len(coord) - 1 and layer_idx >= 0:
                x1_left = coord[layer_idx, 0]
                x2_left = coord[layer_idx + 1, 0]
                x1_right = coord[layer_idx, 1]
                x2_right = coord[layer_idx + 1, 1]
                y1 = trap_heights[layer_idx]
                y2 = trap_heights[layer_idx + 1]
                
                # Linear interpolation to find x positions at disc_height
                if abs(y2 - y1) > 1e-10:  # Avoid division by zero
                    # Calculate position fraction within this layer
                    frac = (disc_height - y1) / (y2 - y1)
                    frac = max(0, min(1, frac))  # Clamp between 0 and 1
                    
                    # Linear interpolation
                    xi[c, 0, pop_number] = x1_left + frac * (x2_left - x1_left)
                    xi[c, 1, pop_number] = x1_right + frac * (x2_right - x1_right)
                else:
                    xi[c, 0, pop_number] = x1_left
                    xi[c, 1, pop_number] = x1_right
            else:
                # Top of structure or edge case
                if len(coord) > 0:
                    xi[c, 0, pop_number] = coord[-1, 0]
                    xi[c, 1, pop_number] = coord[-1, 1]
                else:
                    xi[c, 0, pop_number] = 0
                    xi[c, 1, pop_number] = 0

    def _calculate_cylinder_envelope_sample(self, radii, heights, cyl_heights, total_height,
                                        xi, yi, pop_number, n_slices):
        """
        Calculate uncertainty envelope for a single cylinder sample.
        """
        # For cylinders, we treat them similar to trapezoids but with radius instead of half-width
        for c in range(n_slices):
            if total_height > 0:
                disc_height = c * total_height / (n_slices - 1)
            else:
                disc_height = 0
                
            yi[c, 0, pop_number] = disc_height
            
            # Find which cylinder layer we're in
            layer_idx = 0
            for i in range(len(cyl_heights) - 1):
                if disc_height >= cyl_heights[i] and disc_height <= cyl_heights[i + 1]:
                    layer_idx = i
                    break
            
            # Ensure we don't go out of bounds
            layer_idx = min(layer_idx, len(radii) - 2)
            layer_idx = max(layer_idx, 0)
            
            # Linear interpolation between cylinder radii
            if layer_idx < len(radii) - 1 and layer_idx >= 0:
                r1 = radii[layer_idx]
                r2 = radii[layer_idx + 1]
                y1 = cyl_heights[layer_idx]
                y2 = cyl_heights[layer_idx + 1]
                
                if abs(y2 - y1) > 1e-10:  # Avoid division by zero
                    # Calculate position fraction within this layer
                    frac = (disc_height - y1) / (y2 - y1)
                    frac = max(0, min(1, frac))  # Clamp between 0 and 1
                    
                    # Linear interpolation of radius
                    radius_at_height = r1 + frac * (r2 - r1)
                else:
                    radius_at_height = r1
                
                xi[c, 0, pop_number] = -radius_at_height  # Left side
                xi[c, 1, pop_number] = radius_at_height   # Right side
            else:
                # Top of structure or edge case
                if len(radii) > 0:
                    xi[c, 0, pop_number] = -radii[-1]
                    xi[c, 1, pop_number] = radii[-1]
                else:
                    xi[c, 0, pop_number] = 0
                    xi[c, 1, pop_number] = 0

    def _calculate_uncertainty_statistics(self, xi, yi, confidence_level, n_slices):
        """
        Calculate uncertainty statistics from all samples.
        """
        # Calculate confidence interval multiplier
        if confidence_level == 0.68:
            z_score = 1.0  # 1 sigma
        elif confidence_level == 0.95:
            z_score = 1.96  # 2 sigma
        else:
            # Custom confidence level
            from scipy.stats import norm
            z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Calculate statistics
        center_x = np.mean(xi, axis=2)  # Average across samples
        std_x = np.std(xi, axis=2) * z_score  # Standard deviation with confidence multiplier
        
        center_y = np.mean(yi, axis=2)
        std_y = np.std(yi, axis=2) * z_score
        
        # Create envelope arrays
        outer_edge = center_x.copy()
        outer_edge[:, 0] = outer_edge[:, 0] - std_x[:, 0]  # Left side outward
        outer_edge[:, 1] = outer_edge[:, 1] + std_x[:, 1]  # Right side outward
        
        inner_edge = center_x.copy()
        inner_edge[:, 0] = inner_edge[:, 0] + std_x[:, 0]  # Left side inward
        inner_edge[:, 1] = inner_edge[:, 1] - std_x[:, 1]  # Right side inward
        
        y_inner = center_y - std_y
        y_outer = center_y + std_y
        
        # Create plotting arrays (similar to your original LinePlot, InnerPlot, OuterPlot)
        center_line = np.zeros([2 * n_slices, 2])
        inner_envelope = np.zeros([2 * n_slices, 2])
        outer_envelope = np.zeros([2 * n_slices, 2])
        
        # Center line
        center_line[0:n_slices, 0] = center_x[:, 0]  # Left side
        center_line[n_slices:2*n_slices, 0] = np.flipud(center_x[:, 1])  # Right side (flipped)
        center_line[0:n_slices, 1] = center_y[:, 0]  # Heights
        center_line[n_slices:2*n_slices, 1] = np.flipud(center_y[:, 0])  # Heights (flipped)
        
        # Inner envelope
        inner_envelope[0:n_slices, 0] = inner_edge[:, 0]
        inner_envelope[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
        inner_envelope[0:n_slices, 1] = y_inner[:, 0]
        inner_envelope[n_slices:2*n_slices, 1] = np.flipud(y_inner[:, 0])
        
        # Outer envelope
        outer_envelope[0:n_slices, 0] = outer_edge[:, 0]
        outer_envelope[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
        outer_envelope[0:n_slices, 1] = y_outer[:, 0]
        outer_envelope[n_slices:2*n_slices, 1] = np.flipud(y_outer[:, 0])
        
        return center_line, inner_envelope, outer_envelope

    def _plot_uncertainty_envelope_enhanced(self, center_line, inner_envelope, outer_envelope, 
                                        figsize, show_best_fit, show_mean, show_base, colors, confidence_level, mcmc_results):
        """
        Enhanced plot of the uncertainty envelope with better visualization.
        """
        plt.figure(figsize=figsize)
        
        # Convert confidence level to percentage for label
        conf_percent = int(confidence_level * 100)
        
        # Plot outer envelope (filled with semi-transparent color)
        plt.fill(outer_envelope[:, 0], outer_envelope[:, 1], 
                alpha=0.4, color=colors['envelope'], 
                label=f'{conf_percent}% Confidence Interval', 
                zorder=2)
        
        # Plot dashed lines around the outside of the confidence interval
        plt.plot(outer_envelope[:, 0], outer_envelope[:, 1], 
                color='steelblue', linewidth=1.5, linestyle='--', 
                alpha=0.8, zorder=4)
        
        # Plot inner envelope (filled with white to create "hole" effect)
        plt.fill(inner_envelope[:, 0], inner_envelope[:, 1], 
                alpha=1.0, color='white', zorder=3)
        
        # Plot mean structure if requested
        if show_mean:
            plt.plot(center_line[:, 0], center_line[:, 1], 
                    color=colors['mean'], linewidth=1.5, 
                    label='Mean Structure', zorder=5)
        
        # Add base line connecting left and right sides (not spanning whole plot)
        if show_base:
            # Get the leftmost and rightmost points at the base (y=0)
            base_indices = np.where(np.abs(center_line[:, 1]) < 1e-6)[0]  # Find points at y≈0
            if len(base_indices) >= 2:
                # Find the leftmost and rightmost base points
                base_x_coords = center_line[base_indices, 0]
                base_y_coords = center_line[base_indices, 1]
                
                # Connect the extreme points
                x_left = np.min(base_x_coords)
                x_right = np.max(base_x_coords)
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)  # Removed label='Base'
            else:
                # Fallback: use the width of the structure at the base
                # Find the first and last points (should be at the base)
                n_slices = len(center_line) // 2
                x_left = center_line[0, 0]    # First point (left side at base)
                x_right = center_line[n_slices, 0]  # Middle point (right side at base)
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)  # Removed label='Base'
        
        # Optionally overlay the best-fit structure from MCMC
        if show_best_fit:
            try:
                # Apply best-fit parameters and plot structure
                best_params = mcmc_results['best_params']
                param_names = mcmc_results['param_names']
                
                # Temporarily apply best parameters
                original_params = self.model_params.copy()
                self._apply_mcmc_parameters(best_params, param_names)
                
                # Extract best-fit structure
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    self._plot_trapezoid_outline(widths, heights, 
                                            color=colors['best_fit'], 
                                            linewidth=1.0, linestyle='-',
                                            label='Best Fit (MCMC)', zorder=6)
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    self._plot_cylinder_outline(radii, heights,
                                            color=colors['best_fit'],
                                            linewidth=1.0, linestyle='-',
                                            label='Best Fit (MCMC)', zorder=6)
                
                # Restore original parameters
                self.model_params = original_params
                self.update_traditional_from_model_params()
                
            except Exception as e:
                print(f"Warning: Could not plot best-fit structure: {e}")
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, zorder=0)
        
        # Formatting
        plt.title(f'Structure Uncertainty from MCMC\n({self.geometry.title()} Model)', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Width/Radius (Å)', fontsize=12)
        plt.ylabel('Height (Å)', fontsize=12)
        
        # Improve legend
        plt.legend(frameon=True, framealpha=0.9, fontsize=11, 
                fancybox=True, shadow=True, loc='best')
        
        # Set equal aspect ratio for better shape visualization
        plt.axis('equal')
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    def _plot_trapezoid_outline(self, widths, heights, **kwargs):
        """
        Plot trapezoid structure outline.
        """
        # Calculate cumulative heights
        trap_heights = np.zeros(len(heights) + 1)
        for i in range(len(heights)):
            trap_heights[i + 1] = trap_heights[i] + heights[i]
        
        # Create outline coordinates
        x_coords = []
        y_coords = []
        
        # Bottom edge
        x_coords.extend([-widths[0]/2, widths[0]/2])
        y_coords.extend([0, 0])
        
        # Right edge going up
        for i in range(len(heights)):
            x_coords.extend([widths[i]/2, widths[i+1]/2])
            y_coords.extend([trap_heights[i], trap_heights[i+1]])
        
        # Top edge
        x_coords.extend([widths[-1]/2, -widths[-1]/2])
        y_coords.extend([trap_heights[-1], trap_heights[-1]])
        
        # Left edge going down
        for i in range(len(heights)-1, -1, -1):
            x_coords.extend([-widths[i+1]/2, -widths[i]/2])
            y_coords.extend([trap_heights[i+1], trap_heights[i]])
        
        # Close the shape
        x_coords.append(-widths[0]/2)
        y_coords.append(0)
        
        plt.plot(x_coords, y_coords, **kwargs)

    def _plot_cylinder_outline(self, radii, heights, **kwargs):
        """
        Plot cylinder structure outline.
        """
        # Calculate cumulative heights
        cyl_heights = np.zeros(len(heights) + 1)
        for i in range(len(heights)):
            cyl_heights[i + 1] = cyl_heights[i] + heights[i]
        
        # Plot right side
        for i in range(len(heights)):
            plt.plot([radii[i], radii[i+1]], [cyl_heights[i], cyl_heights[i+1]], **kwargs)
            if i == 0:  # Remove label for subsequent lines
                kwargs.pop('label', None)
        
        # Plot left side
        for i in range(len(heights)):
            plt.plot([-radii[i], -radii[i+1]], [cyl_heights[i], cyl_heights[i+1]], **kwargs)
        
        # Plot horizontal lines
        for i in range(len(radii)):
            plt.plot([-radii[i], radii[i]], [cyl_heights[i], cyl_heights[i]], **kwargs)
    
    def compute_structure_slice_samples(self, mcmc_results, n_samples=1000, n_slices=201, 
                                        random_seed=None, verbose=False):
        """Compute x/y samples for each height slice from MCMC chains.

        Uses the same internal machinery as plot_mcmc_uncertainty_envelope to
        reconstruct the trapezoid (or cylinder) structure for many MCMC samples,
        but returns the raw slice samples (xi, yi) for further analysis.

        This function applies each MCMC sample to the model, extracts the structure
        geometry, and computes the x positions (left/right edges) and y positions
        (heights) at evenly spaced slices through the structure.

        Parameters
        ----------
        mcmc_results : dict
            Output from CDSAXS_MCMC containing:
            - 'flat_chains': flattened MCMC chain samples
            - 'param_names': list of parameter names
        n_samples : int, optional
            Number of MCMC samples to draw from the flattened chains. Default: 1000
        n_slices : int, optional
            Number of height slices (vertical sampling points). Default: 201
        random_seed : int, optional
            Random seed for sample selection. If None, uses current random state. Default: None
        verbose : bool, optional
            Whether to print progress information. Default: False

        Returns
        -------
        xi : ndarray, shape (n_slices, 2, n_samples)
            x positions for left/right edges at each slice and sample.
            xi[slice_idx, 0, :] = left edge positions
            xi[slice_idx, 1, :] = right edge positions
        yi : ndarray, shape (n_slices, 1, n_samples)
            y (height) for each slice and sample.
            Note: yi values are typically constant across samples for a given slice,
            but may vary slightly due to different total heights in different samples.
        sample_indices : ndarray, shape (n_samples,)
            Indices of the MCMC samples that were used (for reference)
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Extract flattened chains and parameter names
        flat_chains = mcmc_results['flat_chains']
        param_names = mcmc_results['param_names']
        
        # Limit number of samples for performance
        total_samples = len(flat_chains)
        if n_samples > total_samples:
            n_samples = total_samples
            if verbose:
                print(f"Warning: Requested {n_samples} samples but only {total_samples} available. Using all samples.")
        
        # Randomly select samples for diversity
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        selected_samples = flat_chains[sample_indices]
        
        if verbose:
            print(f"Computing structure samples for {n_samples} MCMC samples with {n_slices} slices...")
        
        # Store original model parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Allocate arrays
        xi = np.zeros((n_slices, 2, n_samples))
        yi = np.zeros((n_slices, 1, n_samples))
        
        try:
            for k, sample_params in enumerate(selected_samples):
                if verbose and (k + 1) % 100 == 0:
                    print(f"  Processing sample {k + 1}/{n_samples}...")
                
                # Apply MCMC params to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    trap_heights = np.zeros(len(heights) + 1)
                    for i, h in enumerate(heights):
                        trap_heights[i + 1] = trap_heights[i] + h
                    total_height = trap_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_trapezoid_envelope_sample(
                        widths, heights, trap_heights, total_height,
                        xi, yi, k, n_slices
                    )
                    
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    cyl_heights = np.zeros(len(heights) + 1)
                    for i, h in enumerate(heights):
                        cyl_heights[i + 1] = cyl_heights[i] + h
                    total_height = cyl_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_cylinder_envelope_sample(
                        radii, heights, cyl_heights, total_height,
                        xi, yi, k, n_slices
                    )
                else:
                    raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()
        
        if verbose:
            print(f"Completed computing {n_samples} structure samples.")
            print(f"  xi shape: {xi.shape}")
            print(f"  yi shape: {yi.shape}")
            print(f"  Height range: [{np.min(yi):.2f}, {np.max(yi):.2f}] Å")
            print(f"  X range: [{np.min(xi):.2f}, {np.max(xi):.2f}] Å")
        
        return xi, yi, sample_indices
    
    def plot_mcmc_uncertainty_envelope_percentile(self, mcmc_results, n_samples=100, n_slices=101, 
                                                   confidence_level=0.95, plot_results=True, 
                                                   figsize=(10, 6), show_best_fit=True, show_mean=True,
                                                   show_base=True, colors=None):
        """
        Plot uncertainty envelope around structure from MCMC results using percentiles.
        
        This is a percentile-based version of plot_mcmc_uncertainty_envelope that uses
        percentiles instead of standard deviation to calculate the uncertainty envelope.
        This is more robust for non-normal distributions.
        
        Parameters:
        -----------
        mcmc_results : dict
            Results from CDSAXS_MCMC containing chains and parameter info
        n_samples : int, optional
            Number of MCMC samples to use for uncertainty calculation. Default: 100
        n_slices : int, optional
            Number of height slices for uncertainty envelope. Default: 101
        confidence_level : float, optional
            Confidence level for uncertainty envelope (0-1). Default: 0.95 (95% CI)
        plot_results : bool, optional
            Whether to plot the results. Default: True
        figsize : tuple, optional
            Figure size. Default: (10, 6)
        show_best_fit : bool, optional
            Whether to overlay the best-fit structure. Default: True
        show_mean : bool, optional
            Whether to show the mean structure. Default: True
        show_base : bool, optional
            Whether to show the base line at y=0. Default: True
        colors : dict, optional
            Custom colors for plotting. Keys: 'envelope', 'mean', 'best_fit', 'structure'
            
        Returns:
        --------
        tuple
            (center_line, inner_envelope, outer_envelope) arrays for plotting
        """
        # Set default colors
        if colors is None:
            colors = {
                'envelope': 'cornflowerblue',
                'mean': 'red',
                'best_fit': 'darkgreen',
                'structure': 'red'
            }
        
        # Extract flattened chains and parameter names
        flat_chains = mcmc_results['flat_chains']
        param_names = mcmc_results['param_names']
        
        # Limit number of samples for performance
        total_samples = len(flat_chains)
        if n_samples > total_samples:
            n_samples = total_samples
        
        # Randomly select samples for diversity
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        selected_samples = flat_chains[sample_indices]
        
        # Store original model parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Initialize arrays for uncertainty calculation
        xi = np.zeros([n_slices, 2, n_samples])  # [slice, side(left/right), sample]
        yi = np.zeros([n_slices, 1, n_samples])  # [slice, 1, sample]
        
        try:
            # Process each MCMC sample
            for pop_number, sample_params in enumerate(selected_samples):
                # Apply MCMC parameters to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    trap_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        trap_heights[i + 1] = trap_heights[i] + heights[i]
                    total_height = trap_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_trapezoid_envelope_sample(
                        widths, heights, trap_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
                    
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    cyl_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        cyl_heights[i + 1] = cyl_heights[i] + heights[i]
                    total_height = cyl_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_cylinder_envelope_sample(
                        radii, heights, cyl_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
            
            # Calculate percentiles from confidence level
            alpha = 1 - confidence_level
            p_low = 100 * alpha / 2
            p_high = 100 * (1 - alpha / 2)
            
            # Calculate percentile-based statistics
            center_x = np.mean(xi, axis=2)  # Mean across samples (for center line)
            center_y = np.mean(yi, axis=2)
            
            # Calculate percentile bounds for x (left and right edges separately)
            x_left_percentiles = np.percentile(xi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            x_right_percentiles = np.percentile(xi[:, 1, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            
            # Calculate percentile bounds for y
            y_percentiles = np.percentile(yi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            
            # Create envelope arrays using percentiles
            # Outer envelope: p_low for left edge, p_high for right edge
            outer_edge = np.zeros_like(center_x)
            outer_edge[:, 0] = x_left_percentiles[:, 0]  # Left side: lower percentile (outward)
            outer_edge[:, 1] = x_right_percentiles[:, 2]  # Right side: upper percentile (outward)
            
            # Inner envelope: p_high for left edge, p_low for right edge
            inner_edge = np.zeros_like(center_x)
            inner_edge[:, 0] = x_left_percentiles[:, 2]  # Left side: upper percentile (inward)
            inner_edge[:, 1] = x_right_percentiles[:, 0]  # Right side: lower percentile (inward)
            
            y_inner = y_percentiles[:, 0]  # Lower percentile
            y_outer = y_percentiles[:, 2]   # Upper percentile
            
            # Create plotting arrays (similar to original function)
            center_line = np.zeros([2 * n_slices, 2])
            inner_envelope = np.zeros([2 * n_slices, 2])
            outer_envelope = np.zeros([2 * n_slices, 2])
            
            # Center line (using mean)
            center_line[0:n_slices, 0] = center_x[:, 0]  # Left side
            center_line[n_slices:2*n_slices, 0] = np.flipud(center_x[:, 1])  # Right side (flipped)
            center_line[0:n_slices, 1] = center_y[:, 0]  # Heights
            center_line[n_slices:2*n_slices, 1] = np.flipud(center_y[:, 0])  # Heights (flipped)
            
            # Inner envelope
            inner_envelope[0:n_slices, 0] = inner_edge[:, 0]
            inner_envelope[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
            inner_envelope[0:n_slices, 1] = y_inner
            inner_envelope[n_slices:2*n_slices, 1] = np.flipud(y_inner)
            
            # Outer envelope
            outer_envelope[0:n_slices, 0] = outer_edge[:, 0]
            outer_envelope[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
            outer_envelope[0:n_slices, 1] = y_outer
            outer_envelope[n_slices:2*n_slices, 1] = np.flipud(y_outer)
            
            # Plot results if requested
            if plot_results:
                self._plot_uncertainty_envelope_percentile(
                    center_line, inner_envelope, outer_envelope, figsize, 
                    show_best_fit, show_mean, show_base, colors, confidence_level, mcmc_results
                )
            
            return center_line, inner_envelope, outer_envelope
            
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()
    
    def plot_mcmc_uncertainty_envelope_percentile2(self, mcmc_results, n_samples=100, n_slices=101, 
                                                   confidence_level=0.95, plot_results=True, 
                                                   figsize=(10, 6), show_best_fit=True, show_mean=True,
                                                   show_base=True, colors=None, combine_envelopes=False):
        """
        Plot uncertainty envelope around structure from MCMC results using percentiles with multiple envelopes.
        
        This function creates 6 separate envelopes:
        - Inner envelope: inner_lower (y_inner), inner_middle (median y), inner_upper (y_outer)
        - Outer envelope: outer_lower (y_inner), outer_middle (median y), outer_upper (y_outer)
        
        This is an extended version of plot_mcmc_uncertainty_envelope_percentile that provides
        more detailed uncertainty visualization by showing envelopes at different y percentiles.
        
        When combine_envelopes=True, the three inner and three outer envelopes are combined by
        selecting points that are furthest from the center line, handling crossover points between envelopes.
        
        Parameters:
        -----------
        mcmc_results : dict
            Results from CDSAXS_MCMC containing chains and parameter info
        n_samples : int, optional
            Number of MCMC samples to use for uncertainty calculation. Default: 100
        n_slices : int, optional
            Number of height slices for uncertainty envelope. Default: 101
        confidence_level : float, optional
            Confidence level for uncertainty envelope (0-1). Default: 0.95 (95% CI)
        plot_results : bool, optional
            Whether to plot the results. Default: True
        figsize : tuple, optional
            Figure size. Default: (10, 6)
        show_best_fit : bool, optional
            Whether to overlay the best-fit structure. Default: True
        show_mean : bool, optional
            Whether to show the mean structure. Default: True
        show_base : bool, optional
            Whether to show the base line at y=0. Default: True
        colors : dict, optional
            Custom colors for plotting. Keys: 'envelope', 'mean', 'best_fit', 'structure'
        combine_envelopes : bool, optional
            If True, combine the three inner and three outer envelopes by selecting points
            furthest from center line. Default: False
            
        Returns:
        --------
        tuple
            If combine_envelopes=False:
                (center_line, inner_lower, inner_middle, inner_upper, outer_lower, outer_middle, outer_upper)
            If combine_envelopes=True:
                (center_line, inner_lower, inner_middle, inner_upper, outer_lower, outer_middle, outer_upper,
                 inner_combined, outer_combined)
        """
        # Set default colors
        if colors is None:
            colors = {
                'envelope': 'cornflowerblue',
                'mean': 'red',
                'best_fit': 'darkgreen',
                'structure': 'red'
            }
        
        # Extract flattened chains and parameter names
        flat_chains = mcmc_results['flat_chains']
        param_names = mcmc_results['param_names']
        
        # Limit number of samples for performance
        total_samples = len(flat_chains)
        if n_samples > total_samples:
            n_samples = total_samples
        
        # Randomly select samples for diversity
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        selected_samples = flat_chains[sample_indices]
        
        # Store original model parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Initialize arrays for uncertainty calculation
        xi = np.zeros([n_slices, 2, n_samples])  # [slice, side(left/right), sample]
        yi = np.zeros([n_slices, 1, n_samples])  # [slice, 1, sample]
        
        try:
            # Process each MCMC sample
            for pop_number, sample_params in enumerate(selected_samples):
                # Apply MCMC parameters to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    trap_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        trap_heights[i + 1] = trap_heights[i] + heights[i]
                    total_height = trap_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_trapezoid_envelope_sample(
                        widths, heights, trap_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
                    
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    cyl_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        cyl_heights[i + 1] = cyl_heights[i] + heights[i]
                    total_height = cyl_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_cylinder_envelope_sample(
                        radii, heights, cyl_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
            
            # Calculate percentiles from confidence level
            alpha = 1 - confidence_level
            p_low = 100 * alpha / 2
            p_high = 100 * (1 - alpha / 2)
            
            # Calculate percentile-based statistics
            center_x = np.mean(xi, axis=2)  # Mean across samples (for center line)
            center_y = np.mean(yi, axis=2)
            
            # Calculate percentile bounds for x (left and right edges separately)
            x_left_percentiles = np.percentile(xi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            x_right_percentiles = np.percentile(xi[:, 1, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            
            # Calculate percentile bounds for y
            y_percentiles = np.percentile(yi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
            
            # Create envelope arrays using percentiles
            # Outer envelope: p_low for left edge, p_high for right edge
            outer_edge = np.zeros_like(center_x)
            outer_edge[:, 0] = x_left_percentiles[:, 0]  # Left side: lower percentile (outward)
            outer_edge[:, 1] = x_right_percentiles[:, 2]  # Right side: upper percentile (outward)
            
            # Inner envelope: p_high for left edge, p_low for right edge
            inner_edge = np.zeros_like(center_x)
            inner_edge[:, 0] = x_left_percentiles[:, 2]  # Left side: upper percentile (inward)
            inner_edge[:, 1] = x_right_percentiles[:, 0]  # Right side: lower percentile (inward)
            
            y_inner = y_percentiles[:, 0]  # Lower percentile
            y_median = y_percentiles[:, 1]  # Median (50th percentile)
            y_outer = y_percentiles[:, 2]   # Upper percentile
            
            # Create plotting arrays for center line
            center_line = np.zeros([2 * n_slices, 2])
            center_line[0:n_slices, 0] = center_x[:, 0]  # Left side
            center_line[n_slices:2*n_slices, 0] = np.flipud(center_x[:, 1])  # Right side (flipped)
            center_line[0:n_slices, 1] = center_y[:, 0]  # Heights
            center_line[n_slices:2*n_slices, 1] = np.flipud(center_y[:, 0])  # Heights (flipped)
            
            # Create 6 envelope arrays
            inner_lower = np.zeros([2 * n_slices, 2])
            inner_middle = np.zeros([2 * n_slices, 2])
            inner_upper = np.zeros([2 * n_slices, 2])
            outer_lower = np.zeros([2 * n_slices, 2])
            outer_middle = np.zeros([2 * n_slices, 2])
            outer_upper = np.zeros([2 * n_slices, 2])
            
            # Inner envelope - lower (y_inner)
            inner_lower[0:n_slices, 0] = inner_edge[:, 0]
            inner_lower[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
            inner_lower[0:n_slices, 1] = y_inner
            inner_lower[n_slices:2*n_slices, 1] = np.flipud(y_inner)
            
            # Inner envelope - middle (median y)
            inner_middle[0:n_slices, 0] = inner_edge[:, 0]
            inner_middle[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
            inner_middle[0:n_slices, 1] = y_median
            inner_middle[n_slices:2*n_slices, 1] = np.flipud(y_median)
            
            # Inner envelope - upper (y_outer)
            inner_upper[0:n_slices, 0] = inner_edge[:, 0]
            inner_upper[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
            inner_upper[0:n_slices, 1] = y_outer
            inner_upper[n_slices:2*n_slices, 1] = np.flipud(y_outer)
            
            # Outer envelope - lower (y_inner)
            outer_lower[0:n_slices, 0] = outer_edge[:, 0]
            outer_lower[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
            outer_lower[0:n_slices, 1] = y_inner
            outer_lower[n_slices:2*n_slices, 1] = np.flipud(y_inner)
            
            # Outer envelope - middle (median y)
            outer_middle[0:n_slices, 0] = outer_edge[:, 0]
            outer_middle[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
            outer_middle[0:n_slices, 1] = y_median
            outer_middle[n_slices:2*n_slices, 1] = np.flipud(y_median)
            
            # Outer envelope - upper (y_outer)
            outer_upper[0:n_slices, 0] = outer_edge[:, 0]
            outer_upper[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
            outer_upper[0:n_slices, 1] = y_outer
            outer_upper[n_slices:2*n_slices, 1] = np.flipud(y_outer)
            
            # Combine envelopes if requested
            inner_combined = None
            outer_combined = None
            if combine_envelopes:
                # Combine inner envelopes (furthest from center)
                # Use lower y range to limit to y_inner max
                inner_combined = self._combine_envelopes_furthest_from_center(
                    inner_lower, inner_middle, inner_upper, center_line, use_lower_y_range=True
                )
                # Combine outer envelopes (furthest from center)
                # Use upper y range to extend to y_outer max
                outer_combined = self._combine_envelopes_furthest_from_center(
                    outer_lower, outer_middle, outer_upper, center_line, use_lower_y_range=False
                )
            
            # Plot results if requested
            if plot_results:
                self._plot_uncertainty_envelope_percentile2(
                    center_line, inner_lower, inner_middle, inner_upper, 
                    outer_lower, outer_middle, outer_upper, figsize, 
                    show_best_fit, show_mean, show_base, colors, confidence_level, mcmc_results,
                    inner_combined, outer_combined
                )
            
            # Return appropriate values based on combine_envelopes flag
            if combine_envelopes:
                return center_line, inner_lower, inner_middle, inner_upper, outer_lower, outer_middle, outer_upper, inner_combined, outer_combined
            else:
                return center_line, inner_lower, inner_middle, inner_upper, outer_lower, outer_middle, outer_upper
            
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()
    
    def plot_mcmc_uncertainty_envelope_percentile3(self, mcmc_results, n_samples=100, n_slices=101, 
                                                   confidence_levels=[0.5, 0.9, 0.95], plot_results=True, 
                                                   figsize=(10, 6), show_best_fit=True, show_mean=True,
                                                   show_base=True, colors=None, layer_indices=None, layer_positions=None,
                                                   arbitrary_heights=None):
        """
        Plot uncertainty envelope around structure from MCMC results with multiple confidence levels.
        
        This function creates combined envelopes for multiple confidence levels, plotting them with
        increasingly light shades of blue. Only the outermost confidence level has a dashed outline.
        
        Parameters:
        -----------
        mcmc_results : dict
            Results from CDSAXS_MCMC containing chains and parameter info
        n_samples : int, optional
            Number of MCMC samples to use for uncertainty calculation. Default: 100
        n_slices : int, optional
            Number of height slices for uncertainty envelope. Default: 101
        confidence_levels : list of float, optional
            List of confidence levels for uncertainty envelopes (0-1). Default: [0.5, 0.9, 0.95]
            Should be provided in order from inner to outer, or will be sorted automatically.
        plot_results : bool, optional
            Whether to plot the results. Default: True
        figsize : tuple, optional
            Figure size. Default: (10, 6)
        show_best_fit : bool, optional
            Whether to overlay the best-fit structure. Default: True
        show_mean : bool, optional
            Whether to show the mean structure. Default: True
        show_base : bool, optional
            Whether to show the base line at y=0. Default: True
        colors : dict, optional
            Custom colors for plotting. Keys: 'mean', 'best_fit', 'structure'
        layer_indices : int or list of int, optional
            Index/indices of layer(s) to mark (0-based). If None, no layer lines are drawn. Default: None
        layer_positions : str or list of str, optional
            Position(s) on layer(s) to mark: 'top' or 'bottom'. If single string, applies to all layers.
            If list, must match length of layer_indices. Default: 'top'
        arbitrary_heights : float or list of float, optional
            Arbitrary height(s) at which to draw horizontal lines and calculate uncertainties.
            Heights should be in Å. Default: None
            
        Returns:
        --------
        dict
            Dictionary with keys for each confidence level containing (center_line, inner_combined, outer_combined)
        """
        # Set default colors
        if colors is None:
            colors = {
                'mean': 'red',
                'best_fit': 'darkgreen',
                'structure': 'red'
            }
        
        # Ensure confidence_levels is a list and sort from smallest to largest
        if not isinstance(confidence_levels, list):
            confidence_levels = [confidence_levels]
        confidence_levels = sorted(confidence_levels)
        
        # Extract flattened chains and parameter names
        flat_chains = mcmc_results['flat_chains']
        param_names = mcmc_results['param_names']
        
        # Limit number of samples for performance
        total_samples = len(flat_chains)
        if n_samples > total_samples:
            n_samples = total_samples
        
        # Randomly select samples for diversity
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        selected_samples = flat_chains[sample_indices]
        
        # Store original model parameters
        original_params = copy.deepcopy(self.model_params)
        
        # Initialize arrays for uncertainty calculation
        xi = np.zeros([n_slices, 2, n_samples])  # [slice, side(left/right), sample]
        yi = np.zeros([n_slices, 1, n_samples])  # [slice, 1, sample]
        
        try:
            # Process each MCMC sample (only need to do this once for all confidence levels)
            for pop_number, sample_params in enumerate(selected_samples):
                # Apply MCMC parameters to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information based on geometry
                if self.geometry in ['trapezoid', 'sige']:
                    heights, widths = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    trap_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        trap_heights[i + 1] = trap_heights[i] + heights[i]
                    total_height = trap_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_trapezoid_envelope_sample(
                        widths, heights, trap_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
                    
                elif self.geometry == 'cylinder':
                    heights, radii = self._extract_structure_for_uncertainty()
                    
                    # Create cumulative heights array (including 0 at start)
                    cyl_heights = np.zeros(len(heights) + 1)
                    for i in range(len(heights)):
                        cyl_heights[i + 1] = cyl_heights[i] + heights[i]
                    total_height = cyl_heights[-1]
                    
                    # Calculate uncertainty envelope for this sample
                    self._calculate_cylinder_envelope_sample(
                        radii, heights, cyl_heights, total_height,
                        xi, yi, pop_number, n_slices
                    )
            
            # Calculate percentile-based statistics (same for all confidence levels)
            center_x = np.mean(xi, axis=2)  # Mean across samples (for center line)
            center_y = np.mean(yi, axis=2)
            
            # Create plotting arrays for center line
            center_line = np.zeros([2 * n_slices, 2])
            center_line[0:n_slices, 0] = center_x[:, 0]  # Left side
            center_line[n_slices:2*n_slices, 0] = np.flipud(center_x[:, 1])  # Right side (flipped)
            center_line[0:n_slices, 1] = center_y[:, 0]  # Heights
            center_line[n_slices:2*n_slices, 1] = np.flipud(center_y[:, 0])  # Heights (flipped)
            
            # Calculate mean layer boundary heights if requested
            layer_heights = None
            layer_info = None
            if layer_indices is not None:
                layer_heights, layer_info = self._calculate_mean_layer_heights(
                    selected_samples, param_names, layer_indices, layer_positions
                )
            
            # Add arbitrary heights to layer_info if provided
            if arbitrary_heights is not None:
                if not isinstance(arbitrary_heights, (list, tuple, np.ndarray)):
                    arbitrary_heights = [arbitrary_heights]
                
                if layer_info is None:
                    layer_info = []
                
                for height in arbitrary_heights:
                    layer_info.append({
                        'layer_index': None,
                        'position': 'arbitrary',
                        'mean_height': height
                    })
            
            # Calculate and combine envelopes for each confidence level
            results = {}
            for conf_level in confidence_levels:
                # Calculate percentiles from confidence level
                alpha = 1 - conf_level
                p_low = 100 * alpha / 2
                p_high = 100 * (1 - alpha / 2)
                
                # Calculate percentile bounds for x (left and right edges separately)
                x_left_percentiles = np.percentile(xi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
                x_right_percentiles = np.percentile(xi[:, 1, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
                
                # Calculate percentile bounds for y
                y_percentiles = np.percentile(yi[:, 0, :], [p_low, 50, p_high], axis=1).T  # [n_slices, 3]
                
                # Create envelope arrays using percentiles
                # Outer envelope: p_low for left edge, p_high for right edge
                outer_edge = np.zeros_like(center_x)
                outer_edge[:, 0] = x_left_percentiles[:, 0]  # Left side: lower percentile (outward)
                outer_edge[:, 1] = x_right_percentiles[:, 2]  # Right side: upper percentile (outward)
                
                # Inner envelope: p_high for left edge, p_low for right edge
                inner_edge = np.zeros_like(center_x)
                inner_edge[:, 0] = x_left_percentiles[:, 2]  # Left side: upper percentile (inward)
                inner_edge[:, 1] = x_right_percentiles[:, 0]  # Right side: lower percentile (inward)
                
                y_inner = y_percentiles[:, 0]  # Lower percentile
                y_median = y_percentiles[:, 1]  # Median (50th percentile)
                y_outer = y_percentiles[:, 2]   # Upper percentile
                
                # Create 6 envelope arrays
                inner_lower = np.zeros([2 * n_slices, 2])
                inner_middle = np.zeros([2 * n_slices, 2])
                inner_upper = np.zeros([2 * n_slices, 2])
                outer_lower = np.zeros([2 * n_slices, 2])
                outer_middle = np.zeros([2 * n_slices, 2])
                outer_upper = np.zeros([2 * n_slices, 2])
                
                # Inner envelope - lower (y_inner)
                inner_lower[0:n_slices, 0] = inner_edge[:, 0]
                inner_lower[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
                inner_lower[0:n_slices, 1] = y_inner
                inner_lower[n_slices:2*n_slices, 1] = np.flipud(y_inner)
                
                # Inner envelope - middle (median y)
                inner_middle[0:n_slices, 0] = inner_edge[:, 0]
                inner_middle[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
                inner_middle[0:n_slices, 1] = y_median
                inner_middle[n_slices:2*n_slices, 1] = np.flipud(y_median)
                
                # Inner envelope - upper (y_outer)
                inner_upper[0:n_slices, 0] = inner_edge[:, 0]
                inner_upper[n_slices:2*n_slices, 0] = np.flipud(inner_edge[:, 1])
                inner_upper[0:n_slices, 1] = y_outer
                inner_upper[n_slices:2*n_slices, 1] = np.flipud(y_outer)
                
                # Outer envelope - lower (y_inner)
                outer_lower[0:n_slices, 0] = outer_edge[:, 0]
                outer_lower[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
                outer_lower[0:n_slices, 1] = y_inner
                outer_lower[n_slices:2*n_slices, 1] = np.flipud(y_inner)
                
                # Outer envelope - middle (median y)
                outer_middle[0:n_slices, 0] = outer_edge[:, 0]
                outer_middle[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
                outer_middle[0:n_slices, 1] = y_median
                outer_middle[n_slices:2*n_slices, 1] = np.flipud(y_median)
                
                # Outer envelope - upper (y_outer)
                outer_upper[0:n_slices, 0] = outer_edge[:, 0]
                outer_upper[n_slices:2*n_slices, 0] = np.flipud(outer_edge[:, 1])
                outer_upper[0:n_slices, 1] = y_outer
                outer_upper[n_slices:2*n_slices, 1] = np.flipud(y_outer)
                
                # Combine envelopes (always combine for V3)
                inner_combined = self._combine_envelopes_furthest_from_center(
                    inner_lower, inner_middle, inner_upper, center_line, use_lower_y_range=True
                )
                outer_combined = self._combine_envelopes_furthest_from_center(
                    outer_lower, outer_middle, outer_upper, center_line, use_lower_y_range=False
                )
                
                results[conf_level] = {
                    'center_line': center_line,
                    'inner_combined': inner_combined,
                    'outer_combined': outer_combined
                }
            
            # Plot results if requested
            if plot_results:
                self._plot_uncertainty_envelope_percentile3(
                    results, confidence_levels, figsize, 
                    show_best_fit, show_mean, show_base, colors, mcmc_results,
                    layer_heights, layer_info, center_line, xi, yi, confidence_levels
                )
            
            return results
            
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()
    
    def _calculate_mean_layer_heights(self, selected_samples, param_names, layer_indices, layer_positions):
        """
        Calculate the mean heights of layer boundaries across MCMC samples.
        
        Parameters:
        -----------
        selected_samples : array
            Selected MCMC samples
        param_names : list
            Parameter names from MCMC results
        layer_indices : int or list of int
            Index/indices of the layer(s) (0-based)
        layer_positions : str or list of str
            'top' or 'bottom' of the layer(s)
            
        Returns:
        --------
        tuple
            (layer_heights_dict, layer_info_list)
            layer_heights_dict: dict mapping (layer_index, position) to mean height
            layer_info_list: list of dicts with layer_index, position, and mean_height
        """
        # Normalize inputs to lists
        if not isinstance(layer_indices, (list, tuple, np.ndarray)):
            layer_indices = [layer_indices]
        if not isinstance(layer_positions, (list, tuple, np.ndarray)):
            layer_positions = [layer_positions] * len(layer_indices)
        
        if len(layer_positions) != len(layer_indices):
            raise ValueError(f"layer_positions length ({len(layer_positions)}) must match layer_indices length ({len(layer_indices)})")
        
        original_params = copy.deepcopy(self.model_params)
        layer_heights_dict = {}  # (layer_index, position) -> list of heights
        
        try:
            for sample_params in selected_samples:
                # Apply MCMC parameters to model
                self._apply_mcmc_parameters(sample_params, param_names)
                
                # Extract structure information
                if self.geometry in ['trapezoid', 'sige']:
                    structures = self.model_params['trapezoids']
                    heights = [trap['height'] for trap in structures[:-1]]  # Skip last (top)
                elif self.geometry == 'cylinder':
                    structures = self.model_params['cylinders']
                    heights = [cyl['height'] for cyl in structures[:-1]]  # Skip last (top)
                else:
                    continue
                
                # Calculate cumulative heights
                cumulative_heights = np.zeros(len(heights) + 1)
                for i in range(len(heights)):
                    cumulative_heights[i + 1] = cumulative_heights[i] + heights[i]
                
                # Get heights for each requested layer/position
                for layer_index, layer_position in zip(layer_indices, layer_positions):
                    key = (layer_index, layer_position)
                    if key not in layer_heights_dict:
                        layer_heights_dict[key] = []
                    
                    # Get the appropriate height based on layer_index and position
                    if layer_position == 'bottom':
                        # Bottom of layer_index is at cumulative_heights[layer_index]
                        if layer_index < len(cumulative_heights):
                            layer_heights_dict[key].append(cumulative_heights[layer_index])
                    elif layer_position == 'top':
                        # Top of layer_index is at cumulative_heights[layer_index + 1]
                        if layer_index + 1 < len(cumulative_heights):
                            layer_heights_dict[key].append(cumulative_heights[layer_index + 1])
                    else:
                        raise ValueError(f"layer_position must be 'top' or 'bottom', got '{layer_position}'")
        
        finally:
            # Restore original parameters
            self.model_params = original_params
            self.update_traditional_from_model_params()
        
        # Calculate means and create info list
        layer_info_list = []
        for (layer_index, layer_position), heights_list in layer_heights_dict.items():
            if len(heights_list) > 0:
                mean_height = np.mean(heights_list)
                layer_heights_dict[(layer_index, layer_position)] = mean_height
                layer_info_list.append({
                    'layer_index': layer_index,
                    'position': layer_position,
                    'mean_height': mean_height
                })
            else:
                layer_heights_dict[(layer_index, layer_position)] = None
        
        return layer_heights_dict, layer_info_list
    
    def _calculate_distance_from_center_line(self, point, center_line):
        """
        Calculate the distance from a point to the nearest point on the center line.
        
        Parameters:
        -----------
        point : array-like, shape (2,)
            Point coordinates [x, y]
        center_line : array-like, shape (n_points, 2)
            Center line points [[x, y], ...]
            
        Returns:
        --------
        float
            Minimum distance from point to center line
        """
        point = np.array(point)
        center_line = np.array(center_line)
        
        # Calculate distances to all points on center line
        distances = np.sqrt(np.sum((center_line - point)**2, axis=1))
        
        # Also check distances to line segments between consecutive points
        min_dist = np.min(distances)
        
        # Check line segments for potentially closer points
        for i in range(len(center_line) - 1):
            p1 = center_line[i]
            p2 = center_line[i + 1]
            
            # Vector from p1 to p2
            v = p2 - p1
            # Vector from p1 to point
            w = point - p1
            
            # Project point onto line segment
            c1 = np.dot(w, v)
            if c1 <= 0:
                # Closest to p1
                dist = np.linalg.norm(point - p1)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    # Closest to p2
                    dist = np.linalg.norm(point - p2)
                else:
                    # Closest to point on segment
                    b = c1 / c2
                    proj = p1 + b * v
                    dist = np.linalg.norm(point - proj)
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _find_envelope_crossovers(self, envelope1, envelope2, center_line):
        """
        Find crossover points where two envelopes intersect based on distance from center line.
        
        Parameters:
        -----------
        envelope1 : array-like, shape (n_points, 2)
            First envelope points
        envelope2 : array-like, shape (n_points, 2)
            Second envelope points
        center_line : array-like, shape (n_points, 2)
            Center line points
            
        Returns:
        --------
        array
            Indices where envelopes cross (where distance ordering changes)
        """
        n_points = len(envelope1)
        crossovers = []
        
        # Calculate distances for both envelopes
        dist1 = np.array([self._calculate_distance_from_center_line(envelope1[i], center_line) 
                          for i in range(n_points)])
        dist2 = np.array([self._calculate_distance_from_center_line(envelope2[i], center_line) 
                          for i in range(n_points)])
        
        # Find where the difference changes sign (crossover)
        diff = dist1 - dist2
        for i in range(n_points - 1):
            if diff[i] * diff[i + 1] < 0:  # Sign change indicates crossover
                crossovers.append(i)
        
        return np.array(crossovers)
    
    def _combine_envelopes_furthest_from_center(self, envelope_lower, envelope_middle, envelope_upper, center_line, use_lower_y_range=False):
        """
        Combine three envelopes by selecting points that are furthest from the center line.
        
        This function handles the case where y positions differ between envelopes by
        interpolating to find corresponding points at similar y values.
        
        Parameters:
        -----------
        envelope_lower : array-like, shape (n_points, 2)
            Lower envelope points
        envelope_middle : array-like, shape (n_points, 2)
            Middle envelope points
        envelope_upper : array-like, shape (n_points, 2)
            Upper envelope points
        center_line : array-like, shape (n_points, 2)
            Center line points
        use_lower_y_range : bool, optional
            If True, use lower envelope's y range (for inner envelopes, limits to y_inner max).
            If False, use upper envelope's y range (for outer envelopes, extends to y_outer max).
            Default: False
            
        Returns:
        --------
        array, shape (n_points, 2)
            Combined envelope with points furthest from center line
        """
        envelope_lower = np.array(envelope_lower)
        envelope_middle = np.array(envelope_middle)
        envelope_upper = np.array(envelope_upper)
        center_line = np.array(center_line)
        
        n_points = len(envelope_lower)
        n_slices = n_points // 2
        
        # For the combined envelope, we need to use the full y range, especially extending
        # to the maximum y_outer. We'll use the upper envelope's y values as reference
        # since it has the highest y values, ensuring we capture the full extent.
        
        # Process left and right sides separately
        combined = np.zeros_like(envelope_lower)
        
        for side_idx in range(2):
            if side_idx == 0:
                # Left side
                start_idx = 0
                end_idx = n_slices
            else:
                # Right side (flipped)
                start_idx = n_slices
                end_idx = n_points
            
            # Get y values for each envelope on this side
            y_lower = envelope_lower[start_idx:end_idx, 1]
            y_middle = envelope_middle[start_idx:end_idx, 1]
            y_upper = envelope_upper[start_idx:end_idx, 1]
            
            # Choose y reference based on envelope type
            # For inner envelopes: use lower envelope's y range (limits to y_inner max)
            # For outer envelopes: use upper envelope's y range (extends to y_outer max)
            if use_lower_y_range:
                y_ref = y_lower.copy()  # Inner envelope: limit to y_inner max
            else:
                y_ref = y_upper.copy()  # Outer envelope: extend to y_outer max
            
            # For each reference y point, find corresponding points on all three envelopes
            for i, y_val in enumerate(y_ref):
                # Find indices on each envelope closest to this y value
                if use_lower_y_range:
                    idx_lower = i  # Already aligned with lower envelope
                    idx_middle = np.argmin(np.abs(y_middle - y_val))
                    idx_upper = np.argmin(np.abs(y_upper - y_val))
                else:
                    idx_lower = np.argmin(np.abs(y_lower - y_val))
                    idx_middle = np.argmin(np.abs(y_middle - y_val))
                    idx_upper = i  # Already aligned with upper envelope
                
                # Get points from each envelope
                pt_lower = envelope_lower[start_idx + idx_lower]
                pt_middle = envelope_middle[start_idx + idx_middle]
                pt_upper = envelope_upper[start_idx + idx_upper]
                
                # Special handling based on envelope type
                if use_lower_y_range:
                    # For inner envelope: if we're beyond the range of middle/upper, use lower envelope point
                    if y_val > np.max(y_middle) or y_val > np.max(y_upper):
                        # At heights beyond other envelopes, use lower envelope point
                        combined[start_idx + i] = pt_lower
                    else:
                        # Calculate distances and select furthest from center
                        dist_lower = self._calculate_distance_from_center_line(pt_lower, center_line)
                        dist_middle = self._calculate_distance_from_center_line(pt_middle, center_line)
                        dist_upper = self._calculate_distance_from_center_line(pt_upper, center_line)
                        distances = [dist_lower, dist_middle, dist_upper]
                        points = [pt_lower, pt_middle, pt_upper]
                        max_idx = np.argmax(distances)
                        combined[start_idx + i] = points[max_idx]
                else:
                    # For outer envelope: if we're at a high y value where upper envelope extends beyond others,
                    # prioritize using the upper envelope point to preserve the maximum height
                    if y_val > np.max(y_middle) and y_val > np.max(y_lower):
                        # At heights beyond other envelopes, use upper envelope point
                        combined[start_idx + i] = pt_upper
                    else:
                        # Calculate distances from center line for all three points
                        dist_lower = self._calculate_distance_from_center_line(pt_lower, center_line)
                        dist_middle = self._calculate_distance_from_center_line(pt_middle, center_line)
                        dist_upper = self._calculate_distance_from_center_line(pt_upper, center_line)
                        
                        # Select point with maximum distance
                        # This ensures we get the furthest point from center at each y level
                        distances = [dist_lower, dist_middle, dist_upper]
                        points = [pt_lower, pt_middle, pt_upper]
                        max_idx = np.argmax(distances)
                        
                        # Use the selected point, which preserves its y value
                        combined[start_idx + i] = points[max_idx]
        
        return combined
    
    def _plot_uncertainty_envelope_percentile(self, center_line, inner_envelope, outer_envelope, 
                                              figsize, show_best_fit, show_mean, show_base, 
                                              colors, confidence_level, mcmc_results):
        """Helper function to plot the uncertainty envelope (percentile-based)."""
        plt.figure(figsize=figsize)
        
        # Convert confidence level to percentage for label
        conf_percent = int(confidence_level * 100)
        
        # Plot outer envelope (filled with semi-transparent color)
        plt.fill(outer_envelope[:, 0], outer_envelope[:, 1], 
                alpha=0.4, color=colors['envelope'], 
                label=f'{conf_percent}% CI (percentile-based)', 
                zorder=2)
        
        # Plot dashed lines around the outside of the confidence interval
        plt.plot(outer_envelope[:, 0], outer_envelope[:, 1], 
                color='steelblue', linewidth=1.5, linestyle='--', 
                alpha=0.8, zorder=4)
        
        # Plot inner envelope (filled with white to create "hole" effect)
        plt.fill(inner_envelope[:, 0], inner_envelope[:, 1], 
                alpha=1.0, color='white', zorder=3)
        
        # Plot mean structure if requested
        if show_mean:
            plt.plot(center_line[:, 0], center_line[:, 1], 
                    color=colors['mean'], linewidth=1.5, 
                    label='Mean Structure', zorder=5)
        
        # Add base line connecting left and right sides
        if show_base:
            # Find the leftmost and rightmost points at the base (y=0)
            base_indices = np.where(np.abs(center_line[:, 1]) < 1e-6)[0]
            if len(base_indices) >= 2:
                base_x_coords = center_line[base_indices, 0]
                x_left = np.min(base_x_coords)
                x_right = np.max(base_x_coords)
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
            else:
                # Fallback: use the width of the structure at the base
                n_slices = len(center_line) // 2
                x_left = center_line[0, 0]
                x_right = center_line[n_slices, 0]
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
        
        # Optionally overlay the best-fit structure from MCMC
        if show_best_fit:
            try:
                best_params = mcmc_results.get('best_params', None)
                if best_params is not None:
                    param_names = mcmc_results['param_names']
                    original_params = self.model_params.copy()
                    try:
                        self._apply_mcmc_parameters(best_params, param_names)
                        # Extract best-fit structure
                        if self.geometry in ['trapezoid', 'sige']:
                            heights, widths = self._extract_structure_for_uncertainty()
                            # Plot trapezoid outline using existing method
                            self._plot_trapezoid_outline(widths, heights, 
                                                       color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                       label='Best Fit (MCMC)', zorder=6)
                        elif self.geometry == 'cylinder':
                            heights, radii = self._extract_structure_for_uncertainty()
                            # Plot cylinder outline using existing method
                            self._plot_cylinder_outline(radii, heights,
                                                       color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                       label='Best Fit (MCMC)', zorder=6)
                    finally:
                        self.model_params = original_params
                        self.update_traditional_from_model_params()
            except Exception as e:
                print(f"Warning: Could not plot best-fit structure: {e}")
        
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title('MCMC Uncertainty Envelope (Percentile-based)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def _plot_uncertainty_envelope_percentile2(self, center_line, inner_lower, inner_middle, inner_upper,
                                               outer_lower, outer_middle, outer_upper, figsize, 
                                               show_best_fit, show_mean, show_base, colors, 
                                               confidence_level, mcmc_results, inner_combined=None, outer_combined=None):
        """Helper function to plot the uncertainty envelope with 6 separate envelopes (percentile-based)."""
        # Convert confidence level to percentage for label
        conf_percent = int(confidence_level * 100)
        
        # If combined envelopes are provided, create two separate plots
        if inner_combined is not None or outer_combined is not None:
            # First plot: Component envelopes
            plt.figure(figsize=figsize)
            self._plot_component_envelopes(center_line, inner_lower, inner_middle, inner_upper,
                                          outer_lower, outer_middle, outer_upper, 
                                          show_best_fit, show_mean, show_base, colors, 
                                          conf_percent, mcmc_results)
            
            # Second plot: Combined envelopes
            plt.figure(figsize=figsize)
            self._plot_combined_envelopes(center_line, inner_combined, outer_combined,
                                         show_best_fit, show_mean, show_base, colors,
                                         conf_percent, mcmc_results)
        else:
            # Single plot: Component envelopes only
            plt.figure(figsize=figsize)
            self._plot_component_envelopes(center_line, inner_lower, inner_middle, inner_upper,
                                          outer_lower, outer_middle, outer_upper, 
                                          show_best_fit, show_mean, show_base, colors, 
                                          conf_percent, mcmc_results)
    
    def _plot_component_envelopes(self, center_line, inner_lower, inner_middle, inner_upper,
                                 outer_lower, outer_middle, outer_upper, 
                                 show_best_fit, show_mean, show_base, colors, 
                                 conf_percent, mcmc_results):
        """Plot the 6 component envelopes."""
        # Plot outer envelopes (from outermost to innermost)
        # Outer upper envelope (y_outer)
        plt.fill(outer_upper[:, 0], outer_upper[:, 1], 
                alpha=0.2, color='lightblue', 
                label=f'Outer Upper (y_outer)', 
                zorder=2)
        plt.plot(outer_upper[:, 0], outer_upper[:, 1], 
                color='steelblue', linewidth=1.0, linestyle='--', 
                alpha=0.6, zorder=3)
        
        # Outer middle envelope (median y)
        plt.fill(outer_middle[:, 0], outer_middle[:, 1], 
                alpha=0.2, color='cornflowerblue', 
                label=f'Outer Middle (median)', 
                zorder=3)
        plt.plot(outer_middle[:, 0], outer_middle[:, 1], 
                color='steelblue', linewidth=1.0, linestyle='--', 
                alpha=0.6, zorder=4)
        
        # Outer lower envelope (y_inner)
        plt.fill(outer_lower[:, 0], outer_lower[:, 1], 
                alpha=0.2, color='royalblue', 
                label=f'Outer Lower (y_inner)', 
                zorder=4)
        plt.plot(outer_lower[:, 0], outer_lower[:, 1], 
                color='steelblue', linewidth=1.0, linestyle='--', 
                alpha=0.6, zorder=5)
        
        # Plot inner envelopes (from outermost to innermost)
        # Inner upper envelope (y_outer)
        plt.fill(inner_upper[:, 0], inner_upper[:, 1], 
                alpha=0.3, color='lightcoral', 
                label=f'Inner Upper (y_outer)', 
                zorder=5)
        plt.plot(inner_upper[:, 0], inner_upper[:, 1], 
                color='crimson', linewidth=1.0, linestyle=':', 
                alpha=0.7, zorder=6)
        
        # Inner middle envelope (median y)
        plt.fill(inner_middle[:, 0], inner_middle[:, 1], 
                alpha=0.3, color='salmon', 
                label=f'Inner Middle (median)', 
                zorder=6)
        plt.plot(inner_middle[:, 0], inner_middle[:, 1], 
                color='crimson', linewidth=1.0, linestyle=':', 
                alpha=0.7, zorder=7)
        
        # Inner lower envelope (y_inner)
        plt.fill(inner_lower[:, 0], inner_lower[:, 1], 
                alpha=0.3, color='indianred', 
                label=f'Inner Lower (y_inner)', 
                zorder=7)
        plt.plot(inner_lower[:, 0], inner_lower[:, 1], 
                color='crimson', linewidth=1.0, linestyle=':', 
                alpha=0.7, zorder=8)
        
        # Add common plot elements
        self._add_common_plot_elements(center_line, show_best_fit, show_mean, show_base, 
                                      colors, conf_percent, mcmc_results, 
                                      'Component Envelopes')
    
    def _plot_combined_envelopes(self, center_line, inner_combined, outer_combined,
                                show_best_fit, show_mean, show_base, colors,
                                conf_percent, mcmc_results):
        """Plot the combined envelopes."""
        # Shade only the region between outer and inner envelopes
        if outer_combined is not None and inner_combined is not None:
            # Create a polygon that goes around outer envelope, then back along inner envelope (reversed)
            # This creates a filled region between the two envelopes
            combined_polygon = np.vstack([
                outer_combined,  # Outer envelope (forward)
                inner_combined[::-1]  # Inner envelope (reversed to close the polygon)
            ])
            plt.fill(combined_polygon[:, 0], combined_polygon[:, 1], 
                    alpha=0.3, color='cornflowerblue', 
                    label=f'{conf_percent}% Uncertainty Envelope', 
                    zorder=1, edgecolor=None)
        
        # Plot outer combined envelope outline (thin dashed blue)
        if outer_combined is not None:
            plt.plot(outer_combined[:, 0], outer_combined[:, 1], 
                    color='steelblue', linewidth=1, linestyle='--', 
                    alpha=0.8, zorder=2)
        
        # Plot inner combined envelope outline (thin dashed blue)
        if inner_combined is not None:
            plt.plot(inner_combined[:, 0], inner_combined[:, 1], 
                    color='steelblue', linewidth=1, linestyle='--', 
                    alpha=0.8, zorder=3)
        
        # Add common plot elements
        self._add_common_plot_elements(center_line, show_best_fit, show_mean, show_base, 
                                      colors, conf_percent, mcmc_results, 
                                      'Combined Envelopes')
    
    def _plot_uncertainty_envelope_percentile3(self, results, confidence_levels, figsize,
                                               show_best_fit, show_mean, show_base, colors, mcmc_results,
                                               layer_heights=None, layer_info=None, center_line=None, 
                                               xi=None, yi=None, conf_levels=None):
        """Plot multiple confidence level envelopes with increasingly light shades of blue."""
        plt.figure(figsize=figsize)
        
        # Sort confidence levels from smallest to largest (inner to outer)
        sorted_levels = sorted(confidence_levels)
        n_levels = len(sorted_levels)
        
        # Generate increasingly light shades of blue
        # Start with darker blue for inner, lighter for outer
        base_colors = ['darkblue', 'steelblue', 'cornflowerblue', 'lightblue', 'lavender']
        if n_levels > len(base_colors):
            # Generate colors if we have more levels using colormap
            color_values = plt.cm.Blues(np.linspace(0.7, 0.3, n_levels))
            fill_colors = [color_values[i] for i in range(n_levels)]
        else:
            fill_colors = base_colors[:n_levels]
        
        # Plot envelopes from inner to outer (smallest to largest confidence level)
        for idx, conf_level in enumerate(sorted_levels):
            conf_percent = int(conf_level * 100)
            result = results[conf_level]
            inner_combined = result['inner_combined']
            outer_combined = result['outer_combined']
            center_line = result['center_line']
            
            # Get color for this level (darker for inner, lighter for outer)
            fill_color = fill_colors[idx]
            
            # Determine if this is the outermost level
            is_outermost = (idx == n_levels - 1)
            
            # Shade the region between outer and inner envelopes
            if outer_combined is not None and inner_combined is not None:
                # Create a polygon that goes around outer envelope, then back along inner envelope (reversed)
                combined_polygon = np.vstack([
                    outer_combined,  # Outer envelope (forward)
                    inner_combined[::-1]  # Inner envelope (reversed to close the polygon)
                ])
                plt.fill(combined_polygon[:, 0], combined_polygon[:, 1], 
                        alpha=0.3, color=fill_color, 
                        label=f'{conf_percent}% Uncertainty Envelope', 
                        zorder=n_levels - idx, edgecolor=None)
            
            # Only the outermost level gets the dashed outline
            if is_outermost:
                # Plot outer combined envelope outline (thin dashed blue)
                if outer_combined is not None:
                    plt.plot(outer_combined[:, 0], outer_combined[:, 1], 
                            color='steelblue', linewidth=1, linestyle='--', 
                            alpha=0.8, zorder=n_levels + 1)
                
                # Plot inner combined envelope outline (thin dashed blue)
                if inner_combined is not None:
                    plt.plot(inner_combined[:, 0], inner_combined[:, 1], 
                            color='steelblue', linewidth=1, linestyle='--', 
                            alpha=0.8, zorder=n_levels + 2)
        
        # Use the center line from the first (or any) result
        if center_line is None:
            center_line = results[sorted_levels[0]]['center_line']
        
        # Draw layer boundary lines if requested (no labels in legend)
        if layer_info is not None:
            for layer_data in layer_info:
                layer_height = layer_data['mean_height']
                # Don't add to legend - plot without label
                self._plot_layer_boundary_line(layer_height, center_line, label='')
            
            # Print x positions with uncertainty at layer heights
            if xi is not None and yi is not None and conf_levels is not None:
                self._print_layer_x_positions(layer_info, center_line, xi, yi, conf_levels)
        
        # Add common plot elements
        self._add_common_plot_elements_v3(center_line, show_best_fit, show_mean, show_base, 
                                         colors, sorted_levels, mcmc_results)
    
    def _plot_layer_boundary_line(self, layer_height, center_line, label='Layer Boundary'):
        """
        Draw a horizontal line at the specified layer height, connecting the mean structure.
        
        Parameters:
        -----------
        layer_height : float
            Height at which to draw the line
        center_line : array
            Center line points to determine x extent
        label : str, optional
            Label for the legend. Default: 'Layer Boundary'
        """
        # Find the x coordinates of the mean structure at this height
        # Interpolate to find where the center line crosses this height
        n_slices = len(center_line) // 2
        
        # Get left and right sides separately
        left_side = center_line[0:n_slices, :]
        right_side = center_line[n_slices:2*n_slices, :]
        
        # Find x coordinates at layer_height for left and right sides
        x_left = None
        x_right = None
        
        # Check left side (going up)
        for i in range(len(left_side) - 1):
            y1, y2 = left_side[i, 1], left_side[i + 1, 1]
            if (y1 <= layer_height <= y2) or (y2 <= layer_height <= y1):
                # Interpolate x position
                if abs(y2 - y1) > 1e-10:
                    frac = (layer_height - y1) / (y2 - y1)
                    x_left = left_side[i, 0] + frac * (left_side[i + 1, 0] - left_side[i, 0])
                else:
                    x_left = left_side[i, 0]
                break
        
        # Check right side (going down, but we need to check in reverse)
        for i in range(len(right_side) - 1):
            y1, y2 = right_side[i, 1], right_side[i + 1, 1]
            if (y1 <= layer_height <= y2) or (y2 <= layer_height <= y1):
                # Interpolate x position
                if abs(y2 - y1) > 1e-10:
                    frac = (layer_height - y1) / (y2 - y1)
                    x_right = right_side[i, 0] + frac * (right_side[i + 1, 0] - right_side[i, 0])
                else:
                    x_right = right_side[i, 0]
                break
        
        # If we couldn't find exact matches, use the closest points
        if x_left is None:
            # Find closest point on left side
            left_idx = np.argmin(np.abs(left_side[:, 1] - layer_height))
            x_left = left_side[left_idx, 0]
        
        if x_right is None:
            # Find closest point on right side
            right_idx = np.argmin(np.abs(right_side[:, 1] - layer_height))
            x_right = right_side[right_idx, 0]
        
        # Draw horizontal line connecting left and right
        plt.plot([x_left, x_right], [layer_height, layer_height], 
                color='black', linewidth=1.5, linestyle='--', 
                alpha=0.7, zorder=200, label=label)
    
    def _print_layer_x_positions(self, layer_info, center_line, xi, yi, confidence_levels):
        """
        Print x positions of center line and uncertainty envelopes at layer heights.
        
        Parameters:
        -----------
        layer_info : list of dict
            List containing layer_index, position, and mean_height for each layer
        center_line : array
            Center line points
        xi : array
            x positions from MCMC samples [n_slices, 2, n_samples]
        yi : array
            y positions from MCMC samples [n_slices, 1, n_samples]
        confidence_levels : list
            List of confidence levels
        """
        print("\n" + "=" * 80)
        print("X Positions with Uncertainty at Specified Heights")
        print("=" * 80)
        
        n_slices = len(center_line) // 2
        
        for layer_data in layer_info:
            layer_index = layer_data['layer_index']
            layer_position = layer_data['position']
            layer_height = layer_data['mean_height']
            
            # Create header based on whether it's a layer or arbitrary height
            if layer_index is not None:
                header = f"Layer {layer_index} {layer_position}"
            else:
                header = "Arbitrary Height"
            print(f"\n{header} (height = {layer_height:.3f} Å)")
            print("-" * 80)
            
            # Find slice index closest to layer_height
            y_values = center_line[0:n_slices, 1]  # Left side y values
            slice_idx = np.argmin(np.abs(y_values - layer_height))
            
            # Get x positions at this slice for all samples
            x_left_samples = xi[slice_idx, 0, :]  # Left edge across all samples
            x_right_samples = xi[slice_idx, 1, :]  # Right edge across all samples
            x_center_samples = (x_left_samples + x_right_samples) / 2  # Center across all samples
            
            # Calculate statistics for center position
            center_mean = np.mean(x_center_samples)
            center_std = np.std(x_center_samples)
            
            # Calculate statistics for width
            width_samples = x_right_samples - x_left_samples
            width_mean = np.mean(width_samples)
            width_std = np.std(width_samples)
            
            # Calculate percentiles for each confidence level
            print(f"  Center X Position:")
            print(f"    Mean:     {center_mean:>10.3f} Å")
            print(f"    Std:      {center_std:>10.3f} Å")
            print(f"    ±1σ:      [{center_mean - center_std:.3f}, {center_mean + center_std:.3f}] Å")
            
            for conf_level in sorted(confidence_levels):
                alpha = 1 - conf_level
                p_low = 100 * alpha / 2
                p_high = 100 * (1 - alpha / 2)
                conf_percent = int(conf_level * 100)
                
                center_percentiles = np.percentile(x_center_samples, [p_low, p_high])
                print(f"    {conf_percent}% CI:  [{center_percentiles[0]:.3f}, {center_percentiles[1]:.3f}] Å")
            
            print(f"\n  Width (x_right - x_left):")
            print(f"    Mean:     {width_mean:>10.3f} Å")
            print(f"    Std:      {width_std:>10.3f} Å")
            print(f"    ±1σ:      [{width_mean - width_std:.3f}, {width_mean + width_std:.3f}] Å")
            
            for conf_level in sorted(confidence_levels):
                alpha = 1 - conf_level
                p_low = 100 * alpha / 2
                p_high = 100 * (1 - alpha / 2)
                conf_percent = int(conf_level * 100)
                
                width_percentiles = np.percentile(width_samples, [p_low, p_high])
                print(f"    {conf_percent}% CI:  [{width_percentiles[0]:.3f}, {width_percentiles[1]:.3f}] Å")
            
            # Calculate statistics for left and right edges
            left_mean = np.mean(x_left_samples)
            left_std = np.std(x_left_samples)
            right_mean = np.mean(x_right_samples)
            right_std = np.std(x_right_samples)
            
            print(f"\n  Left Edge X Position:")
            print(f"    Mean:     {left_mean:>10.3f} Å")
            print(f"    Std:      {left_std:>10.3f} Å")
            print(f"    ±1σ:      [{left_mean - left_std:.3f}, {left_mean + left_std:.3f}] Å")
            
            for conf_level in sorted(confidence_levels):
                alpha = 1 - conf_level
                p_low = 100 * alpha / 2
                p_high = 100 * (1 - alpha / 2)
                conf_percent = int(conf_level * 100)
                
                left_percentiles = np.percentile(x_left_samples, [p_low, p_high])
                print(f"    {conf_percent}% CI:  [{left_percentiles[0]:.3f}, {left_percentiles[1]:.3f}] Å")
            
            print(f"\n  Right Edge X Position:")
            print(f"    Mean:     {right_mean:>10.3f} Å")
            print(f"    Std:      {right_std:>10.3f} Å")
            print(f"    ±1σ:      [{right_mean - right_std:.3f}, {right_mean + right_std:.3f}] Å")
            
            for conf_level in sorted(confidence_levels):
                alpha = 1 - conf_level
                p_low = 100 * alpha / 2
                p_high = 100 * (1 - alpha / 2)
                conf_percent = int(conf_level * 100)
                
                right_percentiles = np.percentile(x_right_samples, [p_low, p_high])
                print(f"    {conf_percent}% CI:  [{right_percentiles[0]:.3f}, {right_percentiles[1]:.3f}] Å")
        
        print("\n" + "=" * 80)
    
    def _add_common_plot_elements_v3(self, center_line, show_best_fit, show_mean, show_base,
                                    colors, confidence_levels, mcmc_results):
        """Add common plot elements for V3 (mean structure, base line, best fit) with filtered legend."""
        # Plot mean structure if requested
        if show_mean:
            plt.plot(center_line[:, 0], center_line[:, 1], 
                    color=colors['mean'], linewidth=1.5, 
                    label='Mean Structure', zorder=100)
        
        # Add base line connecting left and right sides
        if show_base:
            # Find the leftmost and rightmost points at the base (y=0)
            base_indices = np.where(np.abs(center_line[:, 1]) < 1e-6)[0]
            if len(base_indices) >= 2:
                base_x_coords = center_line[base_indices, 0]
                x_left = np.min(base_x_coords)
                x_right = np.max(base_x_coords)
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
            else:
                # Fallback: use the width of the structure at the base
                n_slices = len(center_line) // 2
                x_left = center_line[0, 0]
                x_right = center_line[n_slices, 0]
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
        
        # Optionally overlay the best-fit structure from MCMC (no label)
        if show_best_fit:
            try:
                best_params = mcmc_results.get('best_params', None)
                if best_params is not None:
                    param_names = mcmc_results['param_names']
                    original_params = self.model_params.copy()
                    try:
                        self._apply_mcmc_parameters(best_params, param_names)
                        # Extract best-fit structure
                        if self.geometry in ['trapezoid', 'sige']:
                            heights, widths = self._extract_structure_for_uncertainty()
                            # Plot trapezoid outline using existing method (no label)
                            self._plot_trapezoid_outline(widths, heights, 
                                                       color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                       label='', zorder=101)
                        elif self.geometry == 'cylinder':
                            heights, radii = self._extract_structure_for_uncertainty()
                            # Plot cylinder outline using existing method (no label)
                            self._plot_cylinder_outline(radii, heights,
                                                       color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                       label='', zorder=101)
                    finally:
                        self.model_params = original_params
                        self.update_traditional_from_model_params()
            except Exception as e:
                print(f"Warning: Could not plot best-fit structure: {e}")
        
        # Get max confidence level for title
        max_conf = int(max(confidence_levels) * 100)
        
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title(f'MCMC Uncertainty Envelopes (Percentile-based, up to {max_conf}% CI)')
        
        # Filter legend to only show mean structure and uncertainty envelopes (no layer boundaries)
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if 'Mean Structure' in label or 'Uncertainty Envelope' in label:
                filtered_handles.append(handle)
                filtered_labels.append(label)
        plt.legend(filtered_handles, filtered_labels, loc='best', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def _add_common_plot_elements(self, center_line, show_best_fit, show_mean, show_base,
                                 colors, conf_percent, mcmc_results, plot_type):
        """Add common plot elements (mean structure, base line, best fit) to the current plot."""
        # Plot mean structure if requested
        if show_mean:
            plt.plot(center_line[:, 0], center_line[:, 1], 
                    color=colors['mean'], linewidth=1.5, 
                    label='Mean Structure', zorder=9)
        
        # Add base line connecting left and right sides
        if show_base:
            # Find the leftmost and rightmost points at the base (y=0)
            base_indices = np.where(np.abs(center_line[:, 1]) < 1e-6)[0]
            if len(base_indices) >= 2:
                base_x_coords = center_line[base_indices, 0]
                x_left = np.min(base_x_coords)
                x_right = np.max(base_x_coords)
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
            else:
                # Fallback: use the width of the structure at the base
                n_slices = len(center_line) // 2
                x_left = center_line[0, 0]
                x_right = center_line[n_slices, 0]
                plt.plot([x_left, x_right], [0, 0], 
                        color=colors['structure'], linewidth=1.5, 
                        alpha=0.8, zorder=1)
        
        # Optionally overlay the best-fit structure from MCMC (no label for combined plot)
        if show_best_fit:
            try:
                best_params = mcmc_results.get('best_params', None)
                if best_params is not None:
                    param_names = mcmc_results['param_names']
                    original_params = self.model_params.copy()
                    try:
                        self._apply_mcmc_parameters(best_params, param_names)
                        # Extract best-fit structure
                        if self.geometry in ['trapezoid', 'sige']:
                            heights, widths = self._extract_structure_for_uncertainty()
                            # Plot trapezoid outline using existing method (no label for combined plot)
                            if plot_type == 'Combined Envelopes':
                                self._plot_trapezoid_outline(widths, heights, 
                                                           color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                           label='', zorder=10)
                            else:
                                self._plot_trapezoid_outline(widths, heights, 
                                                           color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                           label='Best Fit (MCMC)', zorder=10)
                        elif self.geometry == 'cylinder':
                            heights, radii = self._extract_structure_for_uncertainty()
                            # Plot cylinder outline using existing method (no label for combined plot)
                            if plot_type == 'Combined Envelopes':
                                self._plot_cylinder_outline(radii, heights,
                                                           color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                           label='', zorder=10)
                            else:
                                self._plot_cylinder_outline(radii, heights,
                                                           color=colors['best_fit'], linewidth=1.0, linestyle='-',
                                                           label='Best Fit (MCMC)', zorder=10)
                    finally:
                        self.model_params = original_params
                        self.update_traditional_from_model_params()
            except Exception as e:
                print(f"Warning: Could not plot best-fit structure: {e}")
        
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title(f'MCMC Uncertainty Envelope - {plot_type} (Percentile-based, {conf_percent}% CI)')
        
        # For combined plot, only show mean structure and uncertainty envelope in legend
        if plot_type == 'Combined Envelopes':
            # Get handles and labels, filter to only show mean structure and uncertainty envelope
            handles, labels = plt.gca().get_legend_handles_labels()
            filtered_handles = []
            filtered_labels = []
            for handle, label in zip(handles, labels):
                if 'Mean Structure' in label or 'Uncertainty Envelope' in label:
                    filtered_handles.append(handle)
                    filtered_labels.append(label)
            plt.legend(filtered_handles, filtered_labels, loc='best', fontsize=8)
        else:
            plt.legend(loc='best', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_slice_histograms_enhanced(self, xi, yi, slice_indices, plot_xi=True, plot_yi=True, 
                                       bins=50, confidence_level=0.95, percentiles=None, figsize=None):
        """Plot histograms of xi and/or yi values for selected height slices.
        
        This function plots histograms for:
        - x-left: left edge positions
        - x-right: right edge positions  
        - y: height positions (when plot_yi=True)
        
        Note: x-center is NOT plotted. Only left, right, and y are shown.
        
        Uncertainty indicators:
        - Blue dashed lines: ±1σ (standard deviation based, ~68% CI for normal distribution)
        - Green dotted lines: Percentile-based confidence interval (default 95% CI)
        
        Parameters
        ----------
        xi : ndarray, shape (n_slices, 2, n_samples)
            x positions (left/right) from compute_structure_slice_samples
        yi : ndarray, shape (n_slices, 1, n_samples)
            y positions (heights) corresponding to xi
        slice_indices : list[int] or int
            Indices of slices to plot (0 .. n_slices-1). Can be a single int or list.
        plot_xi : bool, optional
            Whether to plot xi histograms (left and right edges). Default: True
        plot_yi : bool, optional
            Whether to plot yi histograms. Default: True
        bins : int, optional
            Number of histogram bins. Default: 50
        confidence_level : float, optional
            Confidence level for percentile calculation (0-1). Default: 0.95 (95% CI)
            If percentiles is provided, this is ignored.
        percentiles : tuple, optional
            Lower and upper percentiles to mark (e.g. (2.5, 97.5)). 
            If None, calculated from confidence_level. Default: None
        figsize : tuple, optional
            Figure size. If None, automatically determined. Default: None
        """
        # Calculate percentiles from confidence_level if not provided
        if percentiles is None:
            alpha = 1 - confidence_level
            p_low = 100 * alpha / 2
            p_high = 100 * (1 - alpha / 2)
            percentiles = (p_low, p_high)
        
        # Convert single slice index to list
        if isinstance(slice_indices, int):
            slice_indices = [slice_indices]
        slice_indices = list(slice_indices)
        
        n_slices, _, n_samples = xi.shape
        
        # Determine what to plot: left, right (when plot_xi=True), and y (when plot_yi=True)
        n_xi_plots = 2 if plot_xi else 0  # left and right only (no center)
        n_yi_plots = 1 if plot_yi else 0
        n_plots_per_slice = n_xi_plots + n_yi_plots
        n_slices_to_plot = len(slice_indices)
        
        if n_plots_per_slice == 0:
            print("Warning: Both plot_xi and plot_yi are False. Nothing to plot.")
            return
        
        # Calculate subplot layout
        n_cols = min(3, n_plots_per_slice)
        n_rows = n_slices_to_plot
        
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_slices_to_plot == 1 and n_plots_per_slice == 1:
            axes = np.array([[axes]])
        elif n_slices_to_plot == 1:
            axes = axes.reshape(1, -1)
        elif n_plots_per_slice == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(n_rows, n_cols)
        
        # Plot for each slice
        for row_idx, slice_idx in enumerate(slice_indices):
            if slice_idx < 0 or slice_idx >= n_slices:
                # Hide invalid slices
                for col_idx in range(n_cols):
                    if row_idx < axes.shape[0] and col_idx < axes.shape[1]:
                        axes[row_idx, col_idx].set_visible(False)
                continue
            
            col_idx = 0
            
            # Extract data for this slice
            x_left = xi[slice_idx, 0, :]
            x_right = xi[slice_idx, 1, :]
            y_val = yi[slice_idx, 0, :]
            y_mean = np.mean(y_val)
            
            # Plot xi histograms (left and right only, no center)
            if plot_xi:
                # Left edge
                if col_idx < axes.shape[1]:
                    ax = axes[row_idx, col_idx]
                    mean = np.mean(x_left)
                    std = np.std(x_left)
                    p_low_val, p_high_val = np.percentile(x_left, percentiles)
                    
                    ax.hist(x_left, bins=bins, color='lightblue', edgecolor='black', alpha=0.7)
                    ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'mean={mean:.2f}')
                    ax.axvline(mean - std, color='blue', linestyle='--', linewidth=1.5, 
                              label=f'±1σ (std-based, ~68% CI)')
                    ax.axvline(mean + std, color='blue', linestyle='--', linewidth=1.5)
                    ax.axvline(p_low_val, color='green', linestyle=':', linewidth=1.5, 
                              label=f'{percentiles[0]:.1f}% (percentile)')
                    ax.axvline(p_high_val, color='green', linestyle=':', linewidth=1.5, 
                              label=f'{percentiles[1]:.1f}% (percentile)')
                    
                    ax.set_title(f"Slice {slice_idx} (y≈{y_mean:.1f}): Left Edge")
                    ax.set_xlabel("x-left (Å)")
                    ax.set_ylabel("Count")
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    col_idx += 1
                
                # Right edge
                if col_idx < axes.shape[1]:
                    ax = axes[row_idx, col_idx]
                    mean = np.mean(x_right)
                    std = np.std(x_right)
                    p_low_val, p_high_val = np.percentile(x_right, percentiles)
                    
                    ax.hist(x_right, bins=bins, color='lightcoral', edgecolor='black', alpha=0.7)
                    ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'mean={mean:.2f}')
                    ax.axvline(mean - std, color='blue', linestyle='--', linewidth=1.5, 
                              label=f'±1σ (std-based, ~68% CI)')
                    ax.axvline(mean + std, color='blue', linestyle='--', linewidth=1.5)
                    ax.axvline(p_low_val, color='green', linestyle=':', linewidth=1.5, 
                              label=f'{percentiles[0]:.1f}% (percentile)')
                    ax.axvline(p_high_val, color='green', linestyle=':', linewidth=1.5, 
                              label=f'{percentiles[1]:.1f}% (percentile)')
                    
                    ax.set_title(f"Slice {slice_idx} (y≈{y_mean:.1f}): Right Edge")
                    ax.set_xlabel("x-right (Å)")
                    ax.set_ylabel("Count")
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    col_idx += 1
            
            # Plot yi histogram
            if plot_yi and col_idx < axes.shape[1]:
                ax = axes[row_idx, col_idx]
                mean = np.mean(y_val)
                std = np.std(y_val)
                p_low_val, p_high_val = np.percentile(y_val, percentiles)
                
                ax.hist(y_val, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
                ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'mean={mean:.2f}')
                if std > 1e-10:  # Only plot std if there's variation
                    ax.axvline(mean - std, color='blue', linestyle='--', linewidth=1.5, 
                              label=f'±1σ (std-based, ~68% CI)')
                    ax.axvline(mean + std, color='blue', linestyle='--', linewidth=1.5)
                ax.axvline(p_low_val, color='green', linestyle=':', linewidth=1.5, 
                          label=f'{percentiles[0]:.1f}% (percentile)')
                ax.axvline(p_high_val, color='green', linestyle=':', linewidth=1.5, 
                          label=f'{percentiles[1]:.1f}% (percentile)')
                
                ax.set_title(f"Slice {slice_idx}: Height (y)")
                ax.set_xlabel("y (Å)")
                ax.set_ylabel("Count")
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                col_idx += 1
            
            # Hide unused subplots in this row
            while col_idx < axes.shape[1]:
                axes[row_idx, col_idx].set_visible(False)
                col_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    def print_slice_uncertainties(self, mcmc_results, slice_indices, n_samples=1000, n_slices=1001,
                                  confidence_level=0.95, use_percentiles=True, verbose=True):
        """
        Print uncertainty statistics (mean, std, percentiles) at specific height slices.
        
        This function computes the structure samples and then calculates and displays
        uncertainty statistics for the requested slices.
        
        Parameters
        ----------
        mcmc_results : dict
            Results from CDSAXS_MCMC containing chains and parameter info
        slice_indices : list[int] or int
            Indices of slices to analyze (0 .. n_slices-1). Can be a single int or list.
        n_samples : int, optional
            Number of MCMC samples to use. Default: 1000
        n_slices : int, optional
            Number of height slices used in computation. Default: 1001
        confidence_level : float, optional
            Confidence level for percentile calculation (0-1). Default: 0.95
        use_percentiles : bool, optional
            If True, use percentile-based statistics. If False, use std-based. Default: True
        verbose : bool, optional
            Whether to print the results. Default: True
            
        Returns
        -------
        dict
            Dictionary containing uncertainty statistics for each slice:
            {
                slice_idx: {
                    'y': {'mean': float, 'std': float, 'percentiles': (low, high)},
                    'x_left': {'mean': float, 'std': float, 'percentiles': (low, high)},
                    'x_right': {'mean': float, 'std': float, 'percentiles': (low, high)},
                    'width': {'mean': float, 'std': float, 'percentiles': (low, high)}
                }
            }
        """
        # Convert single slice index to list
        if isinstance(slice_indices, int):
            slice_indices = [slice_indices]
        slice_indices = list(slice_indices)
        
        # Compute structure samples
        xi, yi, sample_indices = self.compute_structure_slice_samples(
            mcmc_results, n_samples=n_samples, n_slices=n_slices, verbose=False
        )
        
        # Calculate percentiles from confidence level
        alpha = 1 - confidence_level
        p_low = 100 * alpha / 2
        p_high = 100 * (1 - alpha / 2)
        
        results = {}
        
        if verbose:
            print("=" * 80)
            print(f"UNCERTAINTY STATISTICS AT SPECIFIC SLICES")
            print("=" * 80)
            print(f"Confidence Level: {confidence_level*100:.1f}%")
            print(f"Number of samples: {n_samples}")
            print(f"Total slices: {n_slices}")
            print("=" * 80)
        
        for slice_idx in slice_indices:
            if slice_idx < 0 or slice_idx >= n_slices:
                if verbose:
                    print(f"\nWarning: Slice {slice_idx} is out of range (0-{n_slices-1}). Skipping.")
                continue
            
            # Extract data for this slice
            x_left = xi[slice_idx, 0, :]
            x_right = xi[slice_idx, 1, :]
            y_val = yi[slice_idx, 0, :]
            width = x_right - x_left  # Width at this slice
            
            # Calculate statistics
            y_mean = np.mean(y_val)
            y_std = np.std(y_val)
            y_percentiles = np.percentile(y_val, [p_low, p_high])
            
            x_left_mean = np.mean(x_left)
            x_left_std = np.std(x_left)
            x_left_percentiles = np.percentile(x_left, [p_low, p_high])
            
            x_right_mean = np.mean(x_right)
            x_right_std = np.std(x_right)
            x_right_percentiles = np.percentile(x_right, [p_low, p_high])
            
            width_mean = np.mean(width)
            width_std = np.std(width)
            width_percentiles = np.percentile(width, [p_low, p_high])
            
            # Store results
            results[slice_idx] = {
                'y': {
                    'mean': y_mean,
                    'std': y_std,
                    'percentiles': (y_percentiles[0], y_percentiles[1])
                },
                'x_left': {
                    'mean': x_left_mean,
                    'std': x_left_std,
                    'percentiles': (x_left_percentiles[0], x_left_percentiles[1])
                },
                'x_right': {
                    'mean': x_right_mean,
                    'std': x_right_std,
                    'percentiles': (x_right_percentiles[0], x_right_percentiles[1])
                },
                'width': {
                    'mean': width_mean,
                    'std': width_std,
                    'percentiles': (width_percentiles[0], width_percentiles[1])
                }
            }
            
            if verbose:
                print(f"\nSlice {slice_idx} (y ≈ {y_mean:.2f} Å):")
                print("-" * 80)
                
                # Height (y)
                print(f"  Height (y):")
                print(f"    Mean:     {y_mean:>10.3f} Å")
                print(f"    Std:      {y_std:>10.3f} Å")
                if use_percentiles:
                    print(f"    {p_low:.1f}% percentile: {y_percentiles[0]:>10.3f} Å")
                    print(f"    {p_high:.1f}% percentile: {y_percentiles[1]:>10.3f} Å")
                    print(f"    Range ({confidence_level*100:.0f}% CI): [{y_percentiles[0]:.3f}, {y_percentiles[1]:.3f}] Å")
                else:
                    print(f"    ±1σ:      [{y_mean - y_std:.3f}, {y_mean + y_std:.3f}] Å")
                
                # Left edge
                print(f"\n  Left Edge (x_left):")
                print(f"    Mean:     {x_left_mean:>10.3f} Å")
                print(f"    Std:      {x_left_std:>10.3f} Å")
                if use_percentiles:
                    print(f"    {p_low:.1f}% percentile: {x_left_percentiles[0]:>10.3f} Å")
                    print(f"    {p_high:.1f}% percentile: {x_left_percentiles[1]:>10.3f} Å")
                    print(f"    Range ({confidence_level*100:.0f}% CI): [{x_left_percentiles[0]:.3f}, {x_left_percentiles[1]:.3f}] Å")
                else:
                    print(f"    ±1σ:      [{x_left_mean - x_left_std:.3f}, {x_left_mean + x_left_std:.3f}] Å")
                
                # Right edge
                print(f"\n  Right Edge (x_right):")
                print(f"    Mean:     {x_right_mean:>10.3f} Å")
                print(f"    Std:      {x_right_std:>10.3f} Å")
                if use_percentiles:
                    print(f"    {p_low:.1f}% percentile: {x_right_percentiles[0]:>10.3f} Å")
                    print(f"    {p_high:.1f}% percentile: {x_right_percentiles[1]:>10.3f} Å")
                    print(f"    Range ({confidence_level*100:.0f}% CI): [{x_right_percentiles[0]:.3f}, {x_right_percentiles[1]:.3f}] Å")
                else:
                    print(f"    ±1σ:      [{x_right_mean - x_right_std:.3f}, {x_right_mean + x_right_std:.3f}] Å")
                
                # Width
                print(f"\n  Width (x_right - x_left):")
                print(f"    Mean:     {width_mean:>10.3f} Å")
                print(f"    Std:      {width_std:>10.3f} Å")
                if use_percentiles:
                    print(f"    {p_low:.1f}% percentile: {width_percentiles[0]:>10.3f} Å")
                    print(f"    {p_high:.1f}% percentile: {width_percentiles[1]:>10.3f} Å")
                    print(f"    Range ({confidence_level*100:.0f}% CI): [{width_percentiles[0]:.3f}, {width_percentiles[1]:.3f}] Å")
                else:
                    print(f"    ±1σ:      [{width_mean - width_std:.3f}, {width_mean + width_std:.3f}] Å")
        
        if verbose:
            print("\n" + "=" * 80)
        
        return results
            
            
    def _clear_callback_data(self):
        """Clear stored callback data."""
        self._callback_data = {
            'iteration': [],
            'objective_values': [],
            'best_objective': [],
            'parameter_values': [],
            'convergence': [],
            'acceptance_flags': [],
            'optimizer_type': None
        }
    
    def _create_differential_evolution_callback(self):
        """Create callback for differential_evolution."""
        def de_callback(xk, convergence=None):
            if not self._callback_enabled:
                return False
            
            iteration = len(self._callback_data['iteration']) + 1
            self._callback_data['iteration'].append(iteration)
            self._callback_data['optimizer_type'] = 'differential_evolution'
            
            # Calculate objective function value
            try:
                if hasattr(self, '_cylinder_optimization_wrapper'):
                    objective_value = self._cylinder_optimization_wrapper(xk)
                elif hasattr(self, '_trapezoid_optimization_wrapper'):
                    objective_value = self._trapezoid_optimization_wrapper(xk)
                else:
                    objective_value = float('inf')
            except Exception as e:
                objective_value = float('inf')
            
            # Store results
            self._callback_data['objective_values'].append(objective_value)
            self._callback_data['parameter_values'].append(xk.copy())
            self._callback_data['convergence'].append(convergence)
            
            # Track best objective
            if self._callback_data['best_objective']:
                best_so_far = min(self._callback_data['best_objective'][-1], objective_value)
            else:
                best_so_far = objective_value
            self._callback_data['best_objective'].append(best_so_far)
            
            # Print progress
            if iteration % self._callback_print_frequency == 0:
                conv_str = f", Conv = {convergence:.6f}" if convergence is not None else ""
                print(f"DE Iter {iteration:4d}: Objective = {objective_value:.6f}, "
                      f"Best = {best_so_far:.6f}{conv_str}")
            
            return False
        
        return de_callback
    
    def _create_dual_annealing_callback(self):
        """Create callback for dual_annealing."""
        def da_callback(x, f, accept):
            if not self._callback_enabled:
                return False
            
            iteration = len(self._callback_data['iteration']) + 1
            self._callback_data['iteration'].append(iteration)
            self._callback_data['optimizer_type'] = 'dual_annealing'
            
            # Store results
            self._callback_data['objective_values'].append(f)
            self._callback_data['parameter_values'].append(x.copy())
            self._callback_data['acceptance_flags'].append(accept)
            
            # Track best objective
            if self._callback_data['best_objective']:
                best_so_far = min(self._callback_data['best_objective'][-1], f)
            else:
                best_so_far = f
            self._callback_data['best_objective'].append(best_so_far)
            
            # Print progress with acceptance rate
            if iteration % self._callback_print_frequency == 0:
                recent_accepts = sum(self._callback_data['acceptance_flags'][-self._callback_print_frequency:])
                accept_rate = recent_accepts / min(self._callback_print_frequency, 
                                                 len(self._callback_data['acceptance_flags'])) * 100
                print(f"DA Iter {iteration:4d}: Objective = {f:.6f}, Best = {best_so_far:.6f}, "
                      f"Accept = {accept}, Recent Accept Rate = {accept_rate:.1f}%")
            
            return False
        
        return da_callback
    
    def _plot_callback_results(self):
        """Plot callback results after optimization."""
        if len(self._callback_data['objective_values']) < 2:
            return
        
        optimizer_type = self._callback_data.get('optimizer_type', 'unknown')
        
        if optimizer_type == 'dual_annealing':
            self._plot_dual_annealing_results()
        else:
            self._plot_differential_evolution_results()
    
    def _plot_differential_evolution_results(self):
        """Plot differential_evolution results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Convergence
        axes[0].plot(self._callback_data['iteration'], self._callback_data['objective_values'], 
                    'b-', alpha=0.7, label='Current')
        axes[0].plot(self._callback_data['iteration'], self._callback_data['best_objective'], 
                    'r-', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective Function (GF)')
        axes[0].set_title('Differential Evolution Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Plot 2: Parameter evolution (first 4 parameters)
        param_array = np.array(self._callback_data['parameter_values'])
        n_params_to_show = min(4, param_array.shape[1])
        
        for i in range(n_params_to_show):
            param_name = getattr(self, 'param_names', [f'Param_{i}'])[i] if hasattr(self, 'param_names') else f'Param_{i}'
            axes[1].plot(self._callback_data['iteration'], param_array[:, i], 
                        label=param_name, alpha=0.8)
        
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Parameter Evolution (First 4)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Improvement rate
        improvements = []
        for i in range(1, len(self._callback_data['best_objective'])):
            if self._callback_data['best_objective'][i-1] > 0:
                improvement = (self._callback_data['best_objective'][i-1] - 
                             self._callback_data['best_objective'][i]) / self._callback_data['best_objective'][i-1]
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        if improvements:
            axes[2].plot(self._callback_data['iteration'][1:], improvements, 'g-', alpha=0.7)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Relative Improvement')
            axes[2].set_title('Best Objective Improvement Rate')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_dual_annealing_results(self):
        """Plot dual_annealing results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Convergence
        axes[0, 0].plot(self._callback_data['iteration'], self._callback_data['objective_values'], 
                       'b-', alpha=0.7, label='Current')
        axes[0, 0].plot(self._callback_data['iteration'], self._callback_data['best_objective'], 
                       'r-', linewidth=2, label='Best so far')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Function (GF)')
        axes[0, 0].set_title('Dual Annealing Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Acceptance pattern
        window_size = min(20, len(self._callback_data['acceptance_flags']) // 4)
        if window_size > 1:
            accept_array = np.array(self._callback_data['acceptance_flags'])
            moving_accept = np.convolve(accept_array, np.ones(window_size)/window_size, mode='valid')
            moving_iterations = self._callback_data['iteration'][window_size-1:]
            axes[0, 1].plot(moving_iterations, moving_accept * 100, 'g-', linewidth=2, 
                           label=f'Moving Average (window={window_size})')
        
        # Scatter plot of accepts/rejects
        accepts = [i for i, flag in enumerate(self._callback_data['acceptance_flags']) if flag == 1]
        rejects = [i for i, flag in enumerate(self._callback_data['acceptance_flags']) if flag == 0]
        
        if accepts:
            axes[0, 1].scatter([self._callback_data['iteration'][i] for i in accepts], [100] * len(accepts), 
                              c='green', alpha=0.6, s=10, label='Accepted')
        if rejects:
            axes[0, 1].scatter([self._callback_data['iteration'][i] for i in rejects], [0] * len(rejects), 
                              c='red', alpha=0.6, s=10, label='Rejected')
        
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Acceptance (%)')
        axes[0, 1].set_title('Acceptance Pattern')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(-5, 105)
        
        # Plot 3: Parameter evolution
        param_array = np.array(self._callback_data['parameter_values'])
        n_params_to_show = min(4, param_array.shape[1])
        
        for i in range(n_params_to_show):
            param_name = getattr(self, 'param_names', [f'Param_{i}'])[i] if hasattr(self, 'param_names') else f'Param_{i}'
            axes[1, 0].plot(self._callback_data['iteration'], param_array[:, i], 
                           label=param_name, alpha=0.8)
        
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('Parameter Evolution (First 4)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Running acceptance rate
        if len(self._callback_data['acceptance_flags']) > 20:
            window = min(50, len(self._callback_data['acceptance_flags']) // 4)
            running_accept = []
            for i in range(window, len(self._callback_data['acceptance_flags'])):
                recent_rate = sum(self._callback_data['acceptance_flags'][i-window:i]) / window * 100
                running_accept.append(recent_rate)
            
            axes[1, 1].plot(self._callback_data['iteration'][window:], running_accept, 'orange', linewidth=2)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Running Acceptance Rate (%)')
            axes[1, 1].set_title(f'Running Acceptance Rate (window={window})')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _print_callback_summary(self):
        """Print summary of callback results."""
        if len(self._callback_data['objective_values']) == 0:
            return
        
        optimizer_type = self._callback_data.get('optimizer_type', 'unknown')
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION SUMMARY ({optimizer_type.upper()})")
        print(f"{'='*60}")
        
        total_iterations = len(self._callback_data['iteration'])
        initial_obj = self._callback_data['objective_values'][0]
        final_obj = self._callback_data['objective_values'][-1]
        best_obj = min(self._callback_data['objective_values'])
        
        print(f"Total iterations: {total_iterations}")
        print(f"Initial objective: {initial_obj:.6f}")
        print(f"Final objective: {final_obj:.6f}")
        print(f"Best objective: {best_obj:.6f}")
        
        total_improvement = initial_obj - best_obj
        relative_improvement = total_improvement / initial_obj * 100 if initial_obj > 0 else 0
        
        print(f"Total improvement: {total_improvement:.6f}")
        print(f"Relative improvement: {relative_improvement:.2f}%")
        
        # Dual annealing specific stats
        if optimizer_type == 'dual_annealing' and self._callback_data['acceptance_flags']:
            total_accepts = sum(self._callback_data['acceptance_flags'])
            accept_rate = total_accepts / len(self._callback_data['acceptance_flags']) * 100
            print(f"Overall acceptance rate: {accept_rate:.1f}% ({total_accepts}/{len(self._callback_data['acceptance_flags'])})")
        
        print(f"{'='*60}")