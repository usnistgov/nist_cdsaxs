# cdsaxs/Fitting/__init__.py

from .CDSAXS_base_model import CDSAXS_Model
from .Trapezoid_model import TrapezoidModel, TrapezoidModelArray
from .SRM_model import TrapezoidModelArray as SRMModel
from .Trapezoid_model_dean import TrapezoidModelArrayDean, TrapezoidModelDean
from .Trapezoid_model_dean_gpu import (
    TrapezoidModelArrayDeanGPU,
    TrapezoidModelArrayDeanGPUFused,
    TrapezoidModelDeanGPU,
    TrapezoidModelDeanGPUFused,
)
from .SiGe_model import SiGeModel, SiGeModelArray
from .Cylinder_model import CylinderModel

from .accelerated_models import (
    AcceleratedTrapezoidModel,
    AcceleratedCylinderModel,
    create_accelerated_model,
    print_acceleration_summary,
)

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
    Misfit,
)

from .optimization_logger import (
    OptimizationLogger,
    add_logging_to_model_class,
    optimize_multiple_layers,
)

from .dean_optimization_trapezoid import (
    build_candidate_matrix,
    build_example_trapezoid_model,
    compare_objective_scaling,
    evaluate_batched_objective,
    evaluate_scalar_objective,
    extract_structure_signature,
    get_example_data_path,
    get_objective_batch_profile,
    get_parameter_names_and_defaults,
    profile_objective_components,
    run_objective_scaling_trial,
    run_objective_trial,
    run_optimizer_trial,
    run_structure_similarity_trial,
)

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
    run_comprehensive_benchmark,
)


def create_model(geometry, model, layers, PAR=None, SLD=None, DW=None, I0=None, Bk=None, Pitch=None, model_params=None):
    """
    Create the appropriate model based on geometry.
    """
    if geometry == "trapezoid":
        return TrapezoidModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    elif geometry == 'srm':
        return SRMModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    elif geometry == "sige":
        return SiGeModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    elif geometry == "cylinder":
        return CylinderModel(model, layers, PAR, SLD, DW, I0, Bk, Pitch, model_params)
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

__all__ = [
    "CDSAXS_Model",
    "TrapezoidModel",
    "TrapezoidModelArray",
    "SRMModel",
    "TrapezoidModelArrayDean",
    "TrapezoidModelDean",
    "TrapezoidModelArrayDeanGPU",
    "TrapezoidModelArrayDeanGPUFused",
    "TrapezoidModelDeanGPU",
    "TrapezoidModelDeanGPUFused",
    "SiGeModel",
    "SiGeModelArray",
    "CylinderModel",
    "AcceleratedTrapezoidModel",
    "AcceleratedCylinderModel",
    "create_accelerated_model",
    "print_acceleration_summary",
    "FreeFormTrapezoid",
    "ConeFourierTransform",
    "importCDSAXS1D",
    "SymCoordAssign",
    "SymCoordAssign_SingleMaterial",
    "SimTrap",
    "SimTrap_SM",
    "PBA_SymTrap",
    "plotSymTrap",
    "PlotQzCut",
    "PlotQzCut_NoScale",
    "PlotQzCutComp",
    "TPARfromFITPAR",
    "Misfit",
    "OptimizationLogger",
    "add_logging_to_model_class",
    "optimize_multiple_layers",
    "build_candidate_matrix",
    "build_example_trapezoid_model",
    "compare_objective_scaling",
    "evaluate_batched_objective",
    "evaluate_scalar_objective",
    "extract_structure_signature",
    "get_example_data_path",
    "get_objective_batch_profile",
    "get_parameter_names_and_defaults",
    "profile_objective_components",
    "run_objective_scaling_trial",
    "run_objective_trial",
    "run_optimizer_trial",
    "run_structure_similarity_trial",
    "enable_cython",
    "disable_cython",
    "cython_fallback",
    "benchmark_function",
    "gf_calc_accelerated",
    "debye_waller_factor_accelerated",
    "intensity_calculation_accelerated",
    "sym_coord_assign_accelerated",
    "free_form_trapezoid_accelerated",
    "sim_trap_sm_accelerated",
    "cone_fourier_transform_accelerated",
    "sim_cyl_sm_accelerated",
    "sim_cyl_gf_accelerated",
    "convert_cartesian_cylindrical_accelerated",
    "run_comprehensive_benchmark",
    "create_model",
]
