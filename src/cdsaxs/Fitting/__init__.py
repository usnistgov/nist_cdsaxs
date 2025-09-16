# cdsaxs/Fitting/__init__.py

# Import base model class
from .CDSAXS_base_model import CDSAXS_Model

# Import model classes
from .Trapezoid_model import TrapezoidModel, TrapezoidModelArray
from .Cylinder_model import CylinderModel

# Import accelerated models
from .accelerated_models import (
    AcceleratedTrapezoidModel, 
    AcceleratedCylinderModel, 
    create_accelerated_model,
    print_acceleration_summary
)

# Import core functions from CDSAXSFunctions
from .CDSAXSFunctions import (
    FreeFormTrapezoid,
    ConeFourierTransform,
    importCDSAXS1D,
    SymCoordAssign,
    SymCoordAssign_SingleMaterial,
    SimTrap,
    SimTrap_SM,
    PBA_SymTrap,
    plotSymTrap,
    PlotQzCut,
    PlotQzCut_NoScale,
    PlotQzCutComp,
    TPARfromFITPAR,
    Misfit
)

# Import optimization utilities
from .optimization_logger import (
    OptimizationLogger,
    add_logging_to_model_class,
    optimize_multiple_layers
)

# Import cython integration utilities
from .cython_integration import (
    enable_cython,
    disable_cython,
    cython_fallback,
    benchmark_function,
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
    run_comprehensive_benchmark
)

def create_model(geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
    """
    Create the appropriate model based on geometry.
    
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
        return TrapezoidModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    elif geometry == 'cylinder':
        return CylinderModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

# Define all exports
__all__ = [
    # Base model class
    'CDSAXS_Model',
    
    # Model classes
    'TrapezoidModel',
    'TrapezoidModelArray', 
    'CylinderModel',
    
    # Accelerated models
    'AcceleratedTrapezoidModel',
    'AcceleratedCylinderModel',
    'create_accelerated_model',
    'print_acceleration_summary',
    
    # Core functions
    'FreeFormTrapezoid',
    'ConeFourierTransform',
    'importCDSAXS1D',
    'SymCoordAssign',
    'SymCoordAssign_SingleMaterial',
    'SimTrap',
    'SimTrap_SM',
    'PBA_SymTrap',
    'plotSymTrap',
    'PlotQzCut',
    'PlotQzCut_NoScale',
    'PlotQzCutComp',
    'TPARfromFITPAR',
    'Misfit',
    
    # Optimization utilities
    'OptimizationLogger',
    'add_logging_to_model_class',
    'optimize_multiple_layers',
    
    # Cython integration
    'enable_cython',
    'disable_cython',
    'cython_fallback',
    'benchmark_function',
    'gf_calc_accelerated',
    'debye_waller_factor_accelerated',
    'intensity_calculation_accelerated',
    'sym_coord_assign_accelerated',
    'free_form_trapezoid_accelerated',
    'sim_trap_sm_accelerated',
    'cone_fourier_transform_accelerated',
    'sim_cyl_sm_accelerated',
    'sim_cyl_gf_accelerated',
    'convert_cartesian_cylindrical_accelerated',
    'run_comprehensive_benchmark',
    
    # Model creation function
    'create_model'
]