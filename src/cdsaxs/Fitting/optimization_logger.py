import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class OptimizationLogger:
    """
    Class for logging optimization results and maintaining a database of best fits.
    """
    
    def __init__(self, sample_name, output_dir='results'):
        """
        Initialize the optimization logger.
        
        Parameters:
        -----------
        sample_name : str
            Name of the sample being analyzed (used for database filename)
        output_dir : str, optional
            Directory to store the database and plots
        """
        self.sample_name = sample_name
        self.output_dir = output_dir
        self.db_filename = os.path.join(output_dir, f"{sample_name}_fits.json")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load or create the database
        self.database = self.load_database()
    
    def load_database(self):
        """
        Load the existing database or create a new one if it doesn't exist.
        
        Returns:
        --------
        dict
            Database of optimization results
        """
        if os.path.exists(self.db_filename):
            try:
                with open(self.db_filename, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {self.db_filename}. Creating a new database.")
        
        # Create new database if file doesn't exist or is invalid
        return {
            "sample_name": self.sample_name,
            "fits": {},
            "history": []
        }
    
    def save_database(self):
        """
        Save the database to disk.
        """
        with open(self.db_filename, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    def log_result(self, model, save_model=True):
        """
        Log optimization result and update database if this is the best fit for this number of layers.
        
        Parameters:
        -----------
        model : CDSAXS_Model
            The optimized model
        save_model : bool, optional
            Whether to save the full model parameters (True) or just metrics (False)
            
        Returns:
        --------
        bool
            True if this was a new best fit, False otherwise
        """
        # Extract key information
        layers = model.layers
        geometry = model.geometry
        gf = model.GF if hasattr(model, 'GF') else float('inf')
        bic = model.BIC if hasattr(model, 'BIC') else float('inf')
        
        # Create result entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "timestamp": timestamp,
            "layers": layers,
            "geometry": geometry,
            "goodness_of_fit": gf,
            "bic": bic,
        }
        
        # Add parameters if requested
        if save_model:
            if hasattr(model, 'model_params'):
                # Convert numpy arrays to lists for JSON serialization
                params = self._convert_to_serializable(model.model_params)
                result["parameters"] = params
        
        # Check if this is a new best fit for this number of layers
        layers_key = str(layers)
        is_best = False
        
        if layers_key not in self.database["fits"]:
            # First fit with this number of layers
            self.database["fits"][layers_key] = result
            is_best = True
        elif gf < self.database["fits"][layers_key]["goodness_of_fit"]:
            # Better fit than current best
            self.database["fits"][layers_key] = result
            is_best = True
        
        # Add to history regardless
        self.database["history"].append({
            "timestamp": timestamp,
            "layers": layers,
            "geometry": geometry,
            "goodness_of_fit": gf,
            "bic": bic,
            "is_best": is_best
        })
        
        # Save database
        self.save_database()
        
        # Generate plots if this was a new best fit
        if is_best:
            self.plot_metrics()
        
        return is_best
    
    def _convert_to_serializable(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.
        
        Parameters:
        -----------
        obj : object
            Object to convert
            
        Returns:
        --------
        object
            Serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def plot_metrics(self):
        """
        Plot GF vs layers and BIC vs layers from the database of best fits.
        """
        fits = self.database["fits"]
        if not fits:
            print("No fits available to plot.")
            return
        
        # Extract data
        layers = []
        gf_values = []
        bic_values = []
        geometries = []
        
        for layer_str, fit in fits.items():
            layers.append(int(layer_str))
            gf_values.append(fit["goodness_of_fit"])
            bic_values.append(fit["bic"])
            geometries.append(fit["geometry"])
        
        # Sort by number of layers
        indices = np.argsort(layers)
        layers = [layers[i] for i in indices]
        gf_values = [gf_values[i] for i in indices]
        bic_values = [bic_values[i] for i in indices]
        geometries = [geometries[i] for i in indices]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot GF vs layers
        for geom in set(geometries):
            geom_indices = [i for i, g in enumerate(geometries) if g == geom]
            geom_layers = [layers[i] for i in geom_indices]
            geom_gf = [gf_values[i] for i in geom_indices]
            ax1.plot(geom_layers, geom_gf, 'o-', label=geom)
        
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('Goodness of Fit')
        ax1.set_title('Goodness of Fit vs. Number of Layers')
        ax1.grid(True, linestyle='--', alpha=0.7)
        if len(set(geometries)) > 1:
            ax1.legend()
        
        # Plot BIC vs layers
        for geom in set(geometries):
            geom_indices = [i for i, g in enumerate(geometries) if g == geom]
            geom_layers = [layers[i] for i in geom_indices]
            geom_bic = [bic_values[i] for i in geom_indices]
            ax2.plot(geom_layers, geom_bic, 'o-', label=geom)
        
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('Bayesian Information Criterion (BIC)')
        ax2.set_title('BIC vs. Number of Layers')
        ax2.grid(True, linestyle='--', alpha=0.7)
        if len(set(geometries)) > 1:
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(self.output_dir, f"{self.sample_name}_metrics.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)
        
        print(f"Metrics plot saved to {plot_filename}")
    
    def load_best_model(self, layers, model_class, **kwargs):
        """
        Load the best model for a given number of layers.
        
        Parameters:
        -----------
        layers : int
            Number of layers to load
        model_class : class
            Class to use for creating the model (e.g., TrapezoidModel or CylinderModel)
        **kwargs : dict
            Additional keyword arguments to pass to the model constructor
            
        Returns:
        --------
        CDSAXS_Model or None
            The best model for the given number of layers, or None if not found
        """
        layers_key = str(layers)
        if layers_key not in self.database["fits"]:
            print(f"No fits found for {layers} layers.")
            return None
        
        fit = self.database["fits"][layers_key]
        if "parameters" not in fit:
            print(f"Parameters not saved for {layers} layers.")
            return None
        
        # Create model from saved parameters
        model_params = fit["parameters"]
        geometry = fit["geometry"]
        
        # Create model
        model = model_class(
            model=kwargs.get("model", "single_material"),
            layers=layers,
            model_params=model_params,
            **{k: v for k, v in kwargs.items() if k != "model_params" and k != "model" and k != "layers"}
        )
        
        print(f"Loaded best model for {layers} layers (GF: {fit['goodness_of_fit']:.4f}, BIC: {fit['bic']:.4f})")
        return model

# Example of how to integrate with CDSAXS_Model classes
def add_logging_to_model_class():
    """
    Add logging functionality to CDSAXS_Model base class.
    """
    from CDSAXS_base_model import CDSAXS_Model
    
    # Original CDSAXS_DiffEvolution method
    original_diff_evolution = CDSAXS_Model.CDSAXS_DiffEvolution
    
    def log_diff_evolution(self, params_to_optimize=None, plot_results=True, 
                           plot_structure=True, plot_grid=True, plot_combined=True, 
                           log_results=True, sample_name=None, **kwargs):
        """
        Extended CDSAXS_DiffEvolution that logs optimization results.
        
        Parameters:
        -----------
        log_results : bool, optional
            Whether to log the optimization results
        sample_name : str, optional
            Name of the sample for logging
            If None, uses class name and current timestamp
        
        Additional parameters same as original CDSAXS_DiffEvolution.
        """
        # Call original method
        result = original_diff_evolution(self, params_to_optimize, plot_results, 
                                        plot_structure, plot_grid, plot_combined, **kwargs)
        
        # Log results if requested
        if log_results and result is not None:
            if sample_name is None:
                sample_name = f"{self.geometry}_{datetime.now().strftime('%Y%m%d')}"
            
            # Create logger and log result
            logger = OptimizationLogger(sample_name)
            is_best = logger.log_result(self)
            
            if is_best:
                print(f"New best fit for {self.layers} layers! GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
            else:
                print(f"Optimization complete. GF: {self.GF:.4f}, BIC: {self.BIC:.4f}")
        
        return result
    
    # Replace method
    CDSAXS_Model.CDSAXS_DiffEvolution = log_diff_evolution
    
    print("Logging functionality added to CDSAXS_Model class.")

# Example usage for multi-layer optimization
def optimize_multiple_layers(geometry, data_file, min_layers=1, max_layers=5, sample_name=None):
    """
    Optimize models with different numbers of layers and track the best fits.
    
    Parameters:
    -----------
    geometry : str
        Geometry to use ('trapezoid' or 'cylinder')
    data_file : str
        Path to the data file
    min_layers : int, optional
        Minimum number of layers to try
    max_layers : int, optional
        Maximum number of layers to try
    sample_name : str, optional
        Name of the sample for logging
        If None, uses geometry and current date
        
    Returns:
    --------
    dict
        Summary of optimization results
    """
    from cdsaxs import create_model
    
    if sample_name is None:
        sample_name = f"{geometry}_{datetime.now().strftime('%Y%m%d')}"
    
    # Create logger
    logger = OptimizationLogger(sample_name)
    
    results = {}
    
    # Try each number of layers
    for layers in range(min_layers, max_layers + 1):
        print(f"\n=== Optimizing {geometry.capitalize()} Model with {layers} Layers ===")
        
        # Create initial model parameters
        if geometry == 'trapezoid':
            # Create trapezoid parameters with appropriate tapering
            structures = []
            base_width = 100.0
            total_height = 50.0
            
            # Distribute height evenly
            layer_height = total_height / layers
            
            # Create structures with linear tapering
            for i in range(layers + 1):
                width = base_width * (1 - 0.3 * i / layers)  # 30% reduction from bottom to top
                structures.append({
                    'width': width,
                    'height': layer_height if i < layers else 0.0
                })
            
            model_params = {
                'layers': layers,
                'trapezoids': structures,
                'DW': 0.5,
                'I0': 1.0,
                'Bk': 0.01
            }
        elif geometry == 'cylinder':
            # Create cylinder parameters with appropriate tapering
            structures = []
            base_radius = 50.0
            total_height = 50.0
            
            # Distribute height evenly
            layer_height = total_height / layers
            
            # Create structures with linear tapering
            for i in range(layers + 1):
                radius = base_radius * (1 - 0.3 * i / layers)  # 30% reduction from bottom to top
                structures.append({
                    'radius': radius,
                    'height': layer_height if i < layers else 0.0
                })
            
            model_params = {
                'layers': layers,
                'cylinders': structures,
                'DW': 0.5,
                'I0': 1.0,
                'Bk': 0.01,
                'discretization': [10] * layers
            }
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")
        
        # Create model
        model = create_model(
            geometry=geometry,
            model='single_material',
            layers=layers,
            model_params=model_params
        )
        
        # Import data
        model.importCDSAXS_GUI(data_file)
        
        # Run optimization
        optimized_params = model.CDSAXS_DiffEvolution(
            popsize=15,
            maxiter=50,
            log_results=True,
            sample_name=sample_name
        )
        
        # Store results
        results[layers] = {
            'GF': model.GF,
            'BIC': model.BIC
        }
    
    # Print summary
    print("\n=== Optimization Summary ===")
    print(f"{'Layers':<10} {'Goodness of Fit':<20} {'BIC':<20}")
    print("-" * 50)
    for layers, result in sorted(results.items()):
        print(f"{layers:<10} {result['GF']:<20.4f} {result['BIC']:<20.4f}")
    
    return results