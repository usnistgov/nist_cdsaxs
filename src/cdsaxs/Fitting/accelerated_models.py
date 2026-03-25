"""
Accelerated model classes that integrate Cython optimization with fallback to pure Python.
"""

import numpy as np
import copy

# Try relative imports first, then absolute imports as fallback
try:
    from .cython_integration import (
        gf_calc_accelerated,
        debye_waller_factor_accelerated,
        intensity_calculation_accelerated,
        sym_coord_assign_accelerated,
        free_form_trapezoid_accelerated,
        sim_trap_sm_accelerated,
        cone_fourier_transform_accelerated,
        sim_cyl_sm_accelerated,
        sim_cyl_gf_accelerated,
        convert_cartesian_cylindrical_accelerated,
        USE_CYTHON,
        CYTHON_AVAILABLE
    )
except ImportError:
    from cython_integration import (
        gf_calc_accelerated,
        debye_waller_factor_accelerated,
        intensity_calculation_accelerated,
        sym_coord_assign_accelerated,
        free_form_trapezoid_accelerated,
        sim_trap_sm_accelerated,
        cone_fourier_transform_accelerated,
        sim_cyl_sm_accelerated,
        sim_cyl_gf_accelerated,
        convert_cartesian_cylindrical_accelerated,
        USE_CYTHON,
        CYTHON_AVAILABLE
    )

# Import original models
try:
    from .Trapezoid_model import TrapezoidModelArray
    from .Cylinder_model import CylinderModel
except ImportError:
    from .Trapezoid_model import TrapezoidModelArray
    from .Cylinder_model import CylinderModel

class AcceleratedTrapezoidModel(TrapezoidModelArray):
    """
    Trapezoid model with Cython acceleration and fallback to pure Python.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_acceleration = True
        self._acceleration_status = {
            'coord_assignment': False,
            'form_factor': False,
            'simulation': False,
            'gf_calculation': False
        }
    
    def enable_acceleration(self):
        """Enable Cython acceleration for this model instance."""
        self._use_acceleration = True
        if CYTHON_AVAILABLE:
            print("Acceleration enabled for this trapezoid model")
        else:
            print("Cython not available - acceleration cannot be enabled")
    
    def disable_acceleration(self):
        """Disable Cython acceleration for this model instance."""
        self._use_acceleration = False
        print("Acceleration disabled for this trapezoid model")
    
    def get_acceleration_status(self):
        """Get status of which functions are using acceleration."""
        return {
            'cython_available': CYTHON_AVAILABLE,
            'acceleration_enabled': self._use_acceleration,
            'function_status': self._acceleration_status.copy()
        }
    
    def GF_calc(self, SimInt, Intensity=None):
        """
        Accelerated goodness of fit calculation with fallback.
        """
        if self._use_acceleration and CYTHON_AVAILABLE:
            target_intensity = self.Intensity if Intensity is None else Intensity
            result = gf_calc_accelerated(SimInt, target_intensity)
            if result is not None:
                self._acceleration_status['gf_calculation'] = True
                return result
        
        # Fallback to original implementation
        self._acceleration_status['gf_calculation'] = False
        return super().GF_calc(SimInt, Intensity)
    
    def SymCoordAssign(self, PAR=None, layers=None, sld_values=None):
        """
        Accelerated coordinate assignment with fallback.
        """
        # Determine parameters
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
            layers = self.layers
        
        if sld_values is None:
            if hasattr(self, 'sld_values'):
                sld_values = self.sld_values
            else:
                sld_values = np.ones(layers, dtype=np.float64)
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE:
            result = sym_coord_assign_accelerated(PAR, layers, sld_values)
            if result is not None:
                self._acceleration_status['coord_assignment'] = True
                if using_self:
                    self.Coord = result
                    return True
                return result
        
        # Fallback to original implementation
        self._acceleration_status['coord_assignment'] = False
        return super().SymCoordAssign(PAR, layers, sld_values)
    
    def FreeFormTrapezoid(self, Coord=None, layers=None, Qx=None, Qz=None):
        """
        Accelerated form factor calculation with fallback.
        """
        # Determine parameters
        using_self = False
        if Coord is None:
            if not hasattr(self, 'Coord'):
                raise AttributeError("Missing required attribute: Coord")
            Coord = self.Coord
            using_self = True
        
        if layers is None:
            layers = self.layers
        
        if Qx is None:
            Qx = self.Qx
        
        if Qz is None:
            Qz = self.Qz
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE:
            result = free_form_trapezoid_accelerated(Coord, layers, Qx, Qz)
            if result is not None:
                self._acceleration_status['form_factor'] = True
                if using_self:
                    self.form = result
                return result
        
        # Fallback to original implementation
        self._acceleration_status['form_factor'] = False
        return super().FreeFormTrapezoid(Coord, layers, Qx, Qz)
    
    def SimTrap_SM(self, PAR=None, layers=None, Qx=None, Qz=None, DW=None, I0=None, Bk=None):
        """
        Accelerated trapezoid simulation with fallback.
        """
        # Determine parameters (same logic as original)
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
            layers = self.layers
        if Qx is None:
            Qx = self.Qx
        if Qz is None:
            Qz = self.Qz
        if DW is None:
            DW = self.DW
        if I0 is None:
            I0 = self.I0
        if Bk is None:
            Bk = self.Bk
        
        # Get SLD values
        if hasattr(self, 'sld_values'):
            sld_values = self.sld_values
        else:
            sld_values = np.ones(layers, dtype=np.float64)
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE and PAR is not None:
            # Convert background to array if needed for Cython
            if isinstance(Bk, np.ndarray):
                background = Bk
            else:
                background = None  # Will be handled in Cython function
            
            result = sim_trap_sm_accelerated(PAR, layers, Qx, Qz, DW, I0, sld_values, background)
            if result is not None:
                self._acceleration_status['simulation'] = True
                if using_self:
                    self.SimInt = result
                return result
        
        # Fallback to original implementation
        self._acceleration_status['simulation'] = False
        return super().SimTrap_SM(PAR, layers, Qx, Qz, DW, I0, Bk)
    
    def SimTrap_GF(self, optimization_values, param_names=None, Intensity=None, Qx=None, Qz=None):
        """
        Enhanced GF calculation that may use acceleration in the future.
        For now, uses the original implementation but tracks acceleration status.
        """
        # Note: Full Cython integration for optimization wrapper requires more complex parameter mapping
        # For now, use original implementation but prepare for future acceleration
        self._acceleration_status['gf_calculation'] = False
        return super().SimTrap_GF(optimization_values, param_names, Intensity, Qx, Qz)


class AcceleratedCylinderModel(CylinderModel):
    """
    Cylinder model with Cython acceleration and fallback to pure Python.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_acceleration = True
        self._acceleration_status = {
            'coord_conversion': False,
            'form_factor': False,
            'simulation': False,
            'gf_calculation': False
        }
    
    def enable_acceleration(self):
        """Enable Cython acceleration for this model instance."""
        self._use_acceleration = True
        if CYTHON_AVAILABLE:
            print("Acceleration enabled for this cylinder model")
        else:
            print("Cython not available - acceleration cannot be enabled")
    
    def disable_acceleration(self):
        """Disable Cython acceleration for this model instance."""
        self._use_acceleration = False
        print("Acceleration disabled for this cylinder model")
    
    def get_acceleration_status(self):
        """Get status of which functions are using acceleration."""
        return {
            'cython_available': CYTHON_AVAILABLE,
            'acceleration_enabled': self._use_acceleration,
            'function_status': self._acceleration_status.copy()
        }
    
    def GF_calc(self, SimInt, Intensity=None):
        """
        Accelerated goodness of fit calculation with fallback.
        """
        if self._use_acceleration and CYTHON_AVAILABLE:
            target_intensity = self.Intensity if Intensity is None else Intensity
            result = gf_calc_accelerated(SimInt, target_intensity)
            if result is not None:
                self._acceleration_status['gf_calculation'] = True
                return result
        
        # Fallback to original implementation
        self._acceleration_status['gf_calculation'] = False
        return super().GF_calc(SimInt, Intensity)
    
    def convert_Cartesian_Cylindrical(self):
        """
        Accelerated coordinate conversion with fallback.
        """
        if self._use_acceleration and CYTHON_AVAILABLE:
            result = convert_cartesian_cylindrical_accelerated(self.Qx, self.Qy)
            if result is not None:
                self._acceleration_status['coord_conversion'] = True
                self.Qr, self.Alpha = result
                return True
        
        # Fallback to original implementation
        self._acceleration_status['coord_conversion'] = False
        return super().convert_Cartesian_Cylindrical()
    
    def ConeFourierTransform(self, Discretization=None, sld_values=None):
        """
        Accelerated Fourier transform with fallback.
        """
        # Determine parameters
        if Discretization is None:
            if not hasattr(self, 'discretization'):
                self.discretization = [10] * self.layers
            Discretization = self.discretization
        
        if sld_values is None:
            if hasattr(self, 'sld_values'):
                sld_values = self.sld_values
            else:
                sld_values = np.ones(self.layers, dtype=np.float64)
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE:
            # Convert discretization to int array
            discretization_array = np.array(Discretization, dtype=np.int32)
            result = cone_fourier_transform_accelerated(
                self.PAR, self.layers, self.Qr, self.Qz, discretization_array, sld_values
            )
            if result is not None:
                self._acceleration_status['form_factor'] = True
                self.form = result
                return result
        
        # Fallback to original implementation
        self._acceleration_status['form_factor'] = False
        return super().ConeFourierTransform(Discretization, sld_values)
    
    def SimCyl_SM(self, Discretization=None):
        """
        Accelerated cylinder simulation with fallback.
        """
        if Discretization is None:
            if not hasattr(self, 'discretization'):
                self.discretization = [10] * self.layers
            Discretization = self.discretization
        
        # Get SLD values
        if hasattr(self, 'sld_values'):
            sld_values = self.sld_values
        else:
            sld_values = np.ones(self.layers, dtype=np.float64)
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE:
            discretization_array = np.array(Discretization, dtype=np.int32)
            result = sim_cyl_sm_accelerated(
                self.PAR, self.layers, self.Qr, self.Qz, self.DW, self.I0, self.Bk, 
                discretization_array, sld_values
            )
            if result is not None:
                self._acceleration_status['simulation'] = True
                self.SimInt = result
                return result
        
        # Fallback to original implementation
        self._acceleration_status['simulation'] = False
        return super().SimCyl_SM(Discretization)
    
    def SimCyl_GF(self, SimPar, layers, Intensity, Qr, Qz, Discretization):
        """
        Accelerated GF calculation with fallback.
        """
        # Get SLD values
        if hasattr(self, 'sld_values'):
            sld_values = self.sld_values
        else:
            sld_values = np.ones(layers, dtype=np.float64)
        
        # Try Cython acceleration
        if self._use_acceleration and CYTHON_AVAILABLE:
            discretization_array = np.array(Discretization, dtype=np.int32)
            result = sim_cyl_gf_accelerated(SimPar, layers, Intensity, Qr, Qz, discretization_array, sld_values)
            if result is not None:
                self._acceleration_status['gf_calculation'] = True
                return result
        
        # Fallback to original implementation
        self._acceleration_status['gf_calculation'] = False
        return super().SimCyl_GF(SimPar, layers, Intensity, Qr, Qz, Discretization)


# Factory function to create accelerated models
def create_accelerated_model(geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
    """
    Create the appropriate accelerated model based on geometry.
    
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
        Appropriate accelerated model instance based on geometry
    """
    if geometry == 'trapezoid':
        return AcceleratedTrapezoidModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    elif geometry == 'cylinder':
        return AcceleratedCylinderModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

# Convenience function to check acceleration status across models
def print_acceleration_summary(*models):
    """
    Print acceleration status summary for multiple models.
    
    Parameters:
    -----------
    *models : AcceleratedModel instances
        Models to check
    """
    print("\n" + "="*60)
    print("ACCELERATION STATUS SUMMARY")
    print("="*60)
    print(f"Cython available: {CYTHON_AVAILABLE}")
    print(f"Global Cython enabled: {USE_CYTHON}")
    print()
    
    for i, model in enumerate(models):
        if hasattr(model, 'get_acceleration_status'):
            status = model.get_acceleration_status()
            print(f"Model {i+1} ({model.geometry}, {model.layers} layers):")
            print(f"  Acceleration enabled: {status['acceleration_enabled']}")
            
            for func_name, is_accelerated in status['function_status'].items():
                icon = "✅" if is_accelerated else "❌"
                print(f"  {icon} {func_name.replace('_', ' ').title()}")
            print()
        else:
            print(f"Model {i+1}: Not an accelerated model")
    
    print("="*60)
