import os
import re
import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import copy

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
            Geometry type ('trapezoid' or 'cylinder')
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
            Geometry type (e.g., 'trapezoid' or 'cylinder')
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

    def _print_parameter_changes(self, initial_model_params):
        """
        Print a table of parameter changes from optimization.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

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
        raise NotImplementedError("Subclasses must implement this method")