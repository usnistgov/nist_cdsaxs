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
        if not opt_params:
            print(f"Debug: About to call simulate_structure")
            print(f"Debug: self.Intensity shape before sim: {self.Intensity.shape}")
            
            self.SimInt = self.simulate_structure()
            
            print(f"Debug: SimInt shape after sim: {self.SimInt.shape if self.SimInt is not None else 'None'}")
            print(f"Debug: SimInt sample: {self.SimInt[0,0] if self.SimInt is not None else 'None'}")
            
            gf = self.GF_calc(self.SimInt)
            print(f"Debug: GF value: {gf}")
        
        raise NotImplementedError("Subclasses must implement this method")
    
    
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
        #### dbug
        
        print(f"Debug: Intensity shape: {self.Intensity.shape if hasattr(self, 'Intensity') else 'No Intensity attr'}")
        print(f"Debug: Intensity sample: {self.Intensity[0,0] if hasattr(self, 'Intensity') else 'N/A'}")
        print(f"Debug: Has NaN values: {np.any(np.isnan(self.Intensity)) if hasattr(self, 'Intensity') else 'N/A'}")
        
        if not hasattr(self, 'Intensity'):
            raise ValueError("Data must be imported before performing parameter sweep")
        ### debug
        
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
                    if hasattr(self, 'discretization'):
                        # Cylinder models need discretization parameter
                        self.SimInt = self.simulate_structure(self.discretization)
                    else:
                        # Trapezoid models don't need discretization
                        self.SimInt = self.simulate_structure()
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
                        if hasattr(self, 'discretization'):
                            # Cylinder models need discretization parameter
                            self.SimInt = self.simulate_structure(self.discretization)
                        else:
                            # Trapezoid models don't need discretization
                            self.SimInt = self.simulate_structure()
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