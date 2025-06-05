import os
import re
import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

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
        Fixed version of parameter_sweep_1d with proper error handling and results storage.
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
                
                # Initialize variables for this iteration
                gf = float('inf')
                bic = float('inf')
                converged = False
                
                if not opt_params:
                    # No parameters to optimize, just calculate GF
                    try:
                        # Handle both geometries correctly
                        if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                            # Cylinder models need discretization parameter
                            sim_result = self.simulate_structure(self.discretization)
                        else:
                            # Trapezoid models don't need discretization
                            sim_result = self.simulate_structure()
                        
                        # Check if simulation succeeded
                        if sim_result is not None:
                            gf = self.GF_calc(sim_result)
                            bic = self.BIC_calc(gf)
                            converged = True
                            
                            # Debug output for first few points
                            if verbose and i < 3:
                                print(f"DEBUG: Point {i+1}: {sweep_param}={value:.1f}, GF={gf:.4f}")
                        else:
                            if verbose:
                                print(f"Warning: Simulation failed at {sweep_param}={value}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Simulation error at {sweep_param}={value}: {e}")
                else:
                    # Run optimization with suppressed output
                    try:
                        # Don't assign the return value to avoid dictionary display
                        self.CDSAXS_DiffEvolution(
                            params_to_optimize=opt_params,
                            plot_results=False,  # Suppress plots during sweep
                            verbose=False,       # Suppress optimization output
                            **optimization_kwargs
                        )
                        
                        # Get results from model attributes
                        if hasattr(self, 'GF') and hasattr(self, 'BIC'):
                            gf = self.GF
                            bic = self.BIC
                            converged = True
                        else:
                            if verbose:
                                print(f"Warning: No GF/BIC attributes after optimization at {sweep_param}={value}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Optimization failed at {sweep_param}={value}: {e}")
                
                # Store results (make sure we always store something)
                results['gf_values'].append(gf)
                results['bic_values'].append(bic)
                results['optimized_params'].append(copy.deepcopy(self.model_params))
                results['convergence_flags'].append(converged)
                
                # Update progress bar with current best
                if verbose and hasattr(pbar, 'set_postfix'):
                    finite_gfs = [g for g in results['gf_values'] if np.isfinite(g)]
                    current_best_gf = min(finite_gfs) if finite_gfs else float('inf')
                    pbar.set_postfix({
                        f'{sweep_param}': f'{value:.3f}',
                        'Best_GF': f'{current_best_gf:.4f}' if current_best_gf != float('inf') else 'inf'
                    })
                    
            except Exception as e:
                if verbose:
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
            # Pad with inf values if needed
            while len(results['gf_values']) < expected_length:
                results['gf_values'].append(float('inf'))
                results['bic_values'].append(float('inf'))
                results['optimized_params'].append(None)
                results['convergence_flags'].append(False)
        
        # Debug: Print a few results
        if verbose:
            print(f"DEBUG: First few results:")
            for i in range(min(3, len(results['gf_values']))):
                print(f"  {sweep_param}={sweep_values[i]:.1f} -> GF={results['gf_values'][i]}")
        
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
        
        # Setup progress bar
        total_points = n_points[0] * n_points[1]
        if verbose:
            pbar = tqdm(total=total_points, desc=f"2D Sweep: {sweep_params[0]} vs {sweep_params[1]}")
        
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
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
                            # Handle both geometries correctly
                            if hasattr(self, 'discretization') and self.geometry == 'cylinder':
                                # Cylinder models need discretization parameter
                                sim_result = self.simulate_structure(self.discretization)
                            else:
                                # Trapezoid models don't need discretization
                                sim_result = self.simulate_structure()
                            
                            # Check if simulation succeeded
                            if sim_result is not None:
                                gf = self.GF_calc(sim_result)
                                bic = self.BIC_calc(gf)
                                converged = True
                            else:
                                if verbose and total_points <= 25:  # Only print for small grids
                                    print(f"Warning: Simulation failed at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}")
                                
                        except Exception as e:
                            if verbose and total_points <= 25:  # Only print for small grids
                                print(f"Warning: Simulation error at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}: {e}")
                    else:
                        # Run optimization with suppressed output
                        try:
                            # Don't assign the return value to avoid dictionary display
                            self.CDSAXS_DiffEvolution(
                                params_to_optimize=opt_params,
                                plot_results=False,  # Suppress plots during sweep
                                verbose=False,       # Suppress optimization output
                                **optimization_kwargs
                            )
                            
                            # Get results from model attributes
                            if hasattr(self, 'GF') and hasattr(self, 'BIC'):
                                gf = self.GF
                                bic = self.BIC
                                converged = True
                            else:
                                if verbose and total_points <= 25:  # Only print for small grids
                                    print(f"Warning: No GF/BIC attributes after optimization at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}")
                                
                        except Exception as e:
                            if verbose and total_points <= 25:  # Only print for small grids
                                print(f"Warning: Optimization failed at {sweep_params[0]}={val1:.3f}, {sweep_params[1]}={val2:.3f}: {e}")
                    
                    # Store results
                    results['gf_matrix'][j, i] = gf
                    results['bic_matrix'][j, i] = bic
                    results['optimized_params'][j][i] = copy.deepcopy(self.model_params)
                    results['convergence_matrix'][j, i] = converged
                    
                    # Update progress bar
                    if verbose:
                        # Calculate current best for progress display
                        current_gf_matrix = results['gf_matrix'][:j+1, :i+1] if j > 0 or i > 0 else results['gf_matrix'][j:j+1, i:i+1]
                        finite_gfs = current_gf_matrix[np.isfinite(current_gf_matrix)]
                        current_best_gf = np.min(finite_gfs) if len(finite_gfs) > 0 else float('inf')
                        
                        pbar.set_postfix({
                            f'{sweep_params[0]}': f'{val1:.3f}',
                            f'{sweep_params[1]}': f'{val2:.3f}',
                            'Best_GF': f'{current_best_gf:.4f}' if current_best_gf != float('inf') else 'inf'
                        })
                        pbar.update(1)
                        
                except Exception as e:
                    if verbose:
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
        
        # Verify results matrices
        expected_shape = (n_points[1], n_points[0])
        if results['gf_matrix'].shape != expected_shape:
            print(f"WARNING: GF matrix shape mismatch. Expected {expected_shape}, got {results['gf_matrix'].shape}")
        if results['bic_matrix'].shape != expected_shape:
            print(f"WARNING: BIC matrix shape mismatch. Expected {expected_shape}, got {results['bic_matrix'].shape}")
        
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
        q_component = self.Qx if self.geometry == 'trapezoid' else self.Qr
        q_label = 'Qx' if self.geometry == 'trapezoid' else 'Qr'
        
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
        if self.geometry == 'trapezoid':
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_file}_{timestamp}.csv"
        
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
        if self.geometry == 'trapezoid':
            data_dict['Qx'] = q_data
        else:
            data_dict['Qr'] = q_data
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    data_dict[f'metadata_{key}'] = value
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_file}_{timestamp}.npz"
        
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
            if self.geometry == 'trapezoid':
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
            
            if self.geometry == 'trapezoid' and 'trapezoids' in self.model_params:
                widths = [trap.get('width', 0) for trap in self.model_params['trapezoids']]
                heights = [trap.get('height', 0) for trap in self.model_params['trapezoids'][:-1]]
                summary['widths'] = widths
                summary['heights'] = heights
                summary['total_height'] = sum(heights)
                summary['aspect_ratio'] = max(widths) / summary['total_height'] if summary['total_height'] > 0 else float('inf')
            
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
        if not hasattr(self, 'model_params'):
            raise AttributeError("Model must have model_params attribute")
        
        # Get structure data based on geometry
        if self.geometry == 'trapezoid':
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
        if self.geometry == 'trapezoid':
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
        if self.geometry == 'trapezoid':
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
        if self.geometry == 'trapezoid':
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
        if self.geometry == 'trapezoid':
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
        
        if self.geometry == 'trapezoid':
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


    def _add_trapezoid_layer_continuous(self, layer_index: int, position_in_layer: float, insertion_width: float):
        """
        Add a layer to a trapezoid model by splitting an existing layer while maintaining continuity.
        The key insight: we insert a NEW layer but don't duplicate the boundary point.
        """
        # Deep copy the original model parameters
        new_model_params = copy.deepcopy(self.model_params)
        
        # Get original trapezoids
        orig_trapezoids = new_model_params['trapezoids']
        
        # Calculate heights for the split layer
        original_height = orig_trapezoids[layer_index]['height']
        bottom_height = original_height * position_in_layer
        top_height = original_height * (1 - position_in_layer)
        
        print(f"DEBUG: Splitting layer {layer_index}")
        print(f"DEBUG: Original height: {original_height:.1f} Å")
        print(f"DEBUG: Bottom part: {bottom_height:.1f} Å, Top part: {top_height:.1f} Å")
        print(f"DEBUG: Insertion width: {insertion_width:.1f} Å")
        
        # Create new trapezoids list - the key is to properly handle the insertion
        new_trapezoids = []
        
        # Add all trapezoids up to (but not including) the split point
        for i in range(layer_index):
            new_trapezoids.append(orig_trapezoids[i].copy())
        
        # Add the bottom part of the split layer (modified height)
        bottom_trap = orig_trapezoids[layer_index].copy()
        bottom_trap['height'] = bottom_height
        new_trapezoids.append(bottom_trap)
        
        # Add the insertion point as a new trapezoid structure
        # This becomes the "top" of the bottom layer and "bottom" of the next layer
        insertion_trap = {
            'width': insertion_width,
            'height': 0.1  # Small height for the new layer
        }
        new_trapezoids.append(insertion_trap)
        
        # Add the top part of the split layer (if it has meaningful height)
        if top_height > 0.01:  # Only if there's meaningful height left
            top_trap = {
                'width': insertion_width,  # Start from insertion width
                'height': top_height
            }
            new_trapezoids.append(top_trap)
        
        # Add all remaining trapezoids after the split layer
        for i in range(layer_index + 1, len(orig_trapezoids)):
            new_trapezoids.append(orig_trapezoids[i].copy())
        
        # Update model parameters
        new_model_params['trapezoids'] = new_trapezoids
        new_model_params['layers'] = len(new_trapezoids) - 1
        
        print(f"DEBUG: Created {len(new_trapezoids)} trapezoids = {new_model_params['layers']} layers")
        for i, trap in enumerate(new_trapezoids):
            print(f"DEBUG: Trapezoid {i}: width={trap['width']:.1f}, height={trap['height']:.1f}")
        
        # Create new model using the helper method
        new_model = self._create_new_model_from_params(new_model_params)
        
        # Copy data if present
        self._copy_model_data(new_model)
        
        return new_model

    def _add_cylinder_layer_continuous(self, layer_index: int, position_in_layer: float, insertion_radius: float):
        """
        Add a layer to a cylinder model by splitting an existing layer while maintaining continuity.
        """
        # Deep copy the original model parameters
        new_model_params = copy.deepcopy(self.model_params)
        
        # Get discretization setting (from temporary setting or default)
        discretization_per_nm = getattr(self, '_temp_discretization_per_nm', 5.0)
        
        # Get original cylinders
        orig_cylinders = new_model_params['cylinders']
        
        # Calculate heights for the split layer
        original_height = orig_cylinders[layer_index]['height']
        bottom_height = original_height * position_in_layer
        top_height = original_height * (1 - position_in_layer)
        
        print(f"DEBUG: Splitting cylinder layer {layer_index}")
        print(f"DEBUG: Original height: {original_height:.1f} Å")
        print(f"DEBUG: Bottom part: {bottom_height:.1f} Å, Top part: {top_height:.1f} Å")
        print(f"DEBUG: Insertion radius: {insertion_radius:.1f} Å")
        
        # Create new cylinders list
        new_cylinders = []
        
        # Add all cylinders up to (but not including) the split point
        for i in range(layer_index):
            new_cylinders.append(orig_cylinders[i].copy())
        
        # Add the bottom part of the split layer (modified height)
        bottom_cyl = orig_cylinders[layer_index].copy()
        bottom_cyl['height'] = bottom_height
        new_cylinders.append(bottom_cyl)
        
        # Add the insertion point as a new cylinder structure
        insertion_cyl = {
            'radius': insertion_radius,
            'height': 0.1  # Small height for the new layer
        }
        new_cylinders.append(insertion_cyl)
        
        # Add the top part of the split layer (if it has meaningful height)
        if top_height > 0.01:  # Only if there's meaningful height left
            top_cyl = {
                'radius': insertion_radius,  # Start from insertion radius
                'height': top_height
            }
            new_cylinders.append(top_cyl)
        
        # Add all remaining cylinders after the split layer
        for i in range(layer_index + 1, len(orig_cylinders)):
            new_cylinders.append(orig_cylinders[i].copy())
        
        # Update model parameters
        new_model_params['cylinders'] = new_cylinders
        new_model_params['layers'] = len(new_cylinders) - 1
        
        # Generate new discretization array based on layer heights
        new_discretization = self._generate_discretization_array(new_cylinders, discretization_per_nm)
        new_model_params['discretization'] = new_discretization
        
        print(f"DEBUG: Created {len(new_cylinders)} cylinders = {new_model_params['layers']} layers")
        for i, cyl in enumerate(new_cylinders):
            height_str = f"{cyl['height']:.1f}" if 'height' in cyl else "0.0"
            print(f"DEBUG: Cylinder {i}: radius={cyl['radius']:.1f}, height={height_str}")
        
        # Debug discretization
        print(f"Generated discretization array for {new_model_params['layers']} layers:")
        for i, (cyl, disc) in enumerate(zip(new_cylinders[:-1], new_discretization)):
            height_nm = cyl['height'] / 10.0
            print(f"  Layer {i}: height = {cyl['height']:.1f} Å ({height_nm:.2f} nm) → discretization = {disc}")
        
        # Create new model using the helper method
        new_model = self._create_new_model_from_params(new_model_params)
        
        # Copy data if present
        self._copy_model_data(new_model)
        
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
        
        if self.geometry == 'trapezoid':
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
        
        if self.geometry == 'trapezoid':
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
        if self.geometry == 'trapezoid':
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
        
        print(f"DEBUG: Extracted {len(height_points)} height points: {height_points}")
        print(f"DEBUG: Corresponding widths: {width_points}")
        
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

  