# cdsaxs/__init__.py
from .Trapezoid_model import TrapezoidModel
from .Cylinder_model import CylinderModel

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

# Make create_model available at the package level
__all__ = ['TrapezoidModel', 'CylinderModel', 'create_model']