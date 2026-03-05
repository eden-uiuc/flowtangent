# RCAIDE/Framework/Analyses/Aerodynamics/VLM.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable, Optional

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.Library import Units

from RCAIDE.Framework import Process, ProcessStep

from RCAIDE.Framework.Methods.Aerodynamics.Vortex_Lattice import (check_freestream,
                                                                  compute_coefficients, compute_induced_velocity, 
                                                                  compute_panel_pressures, compute_vlm_rhs, 
                                                                  compute_vortex_strength, update_wing_geometry,
                                                                  initialize_VLM_geometry, discretize_wings,
                                                                  generate_full_vortex_distribution)

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Settings
# ----------------------------------------------------------------------------------------------------------------------


class SupersonicSettings(eqx.Module):

    peak_mach_number                        = 1.04
    begin_drag_rise_mach_number             = 0.95
    end_drag_rise_mach_number               = 1.2
    transonic_drag_multiplier               = 1.25
    volume_wave_drag_scaling                = 3.2
    fuselage_parasite_drag_begin_blend_mach = 0.91
    fuselage_parasite_drag_end_blend_mach   = 0.99
    cross_sectional_area_calculation_type:  str = eqx.field(static=True, default='Fixed')
    wave_drag_type:                         str = eqx.field(static=True, default='Raymer')

class CorrectionFactors(eqx.Module):

    fuselage_lift: float = 1.14
    trim_drag: float = 1.0

    viscous_lift_drag: float = 0.38
    lift_to_drag: float = 0.0
    CL_max: float = 1.0

class EfficiencyFactors(eqx.Module):

    span: float = 1.0
    oswald: float = 1.0

class ParasiteDragFormFactors(eqx.Module):

    wing: float = 1.1
    fuselage: float = 2.3

class Training(eqx.Module):

    angle_of_attack:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.linspace(-5. * Units.deg, 15.* Units.deg, 40))
    Mach:                   jnp.ndarray  = eqx.field(default_factory=lambda: jnp.linspace(0., 0.85, 20))

    sideslip_angle:         jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    aileron_deflection:     jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    elevator_deflection:    jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    rudder_deflection:      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    flap_deflection:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    slat_deflection:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)

    u:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))
    v:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))
    w:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))

    pitch_rate:             jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)
    roll_rate:              jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)
    yaw_rate:               jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)


class VLMVortices(eqx.Module):
    # General Settings
    spanwise_cosine_spacing: bool   = eqx.field(static=True, default=True)
    model_fuselage: bool            = eqx.field(static=True, default=False)
    floating_point_precision: str   = eqx.field(static=True, default="float64")
    verbose: bool                   = eqx.field(static=True, default=False)
    
    # Discretization Inputs (Optional, so the user can choose which to define)
    number_of_spanwise_vortices:    int = eqx.field(static=True, default=5)
    number_of_chordwise_vortices:   int = eqx.field(static=True, default=2)
    
    wing_spanwise_vortices:         Optional[int] = eqx.field(static=True, default=None)
    wing_chordwise_vortices:        Optional[int] = eqx.field(static=True, default=None)
    fuselage_spanwise_vortices:     Optional[int] = eqx.field(static=True, default=None)
    fuselage_chordwise_vortices:    Optional[int] = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Validates discretization inputs and resolves global vs separate routing."""

        # Check if the user explicitly provided separate definitions
        separate_provided = any([
            self.wing_spanwise_vortices is not None,
            self.wing_chordwise_vortices is not None,
            self.fuselage_spanwise_vortices is not None,
            self.fuselage_chordwise_vortices is not None
        ])

        if separate_provided:
            # 1. Validate that ALL separate variables were provided
            missing_separate = any(x is None for x in [
                self.wing_spanwise_vortices, self.wing_chordwise_vortices,
                self.fuselage_spanwise_vortices, self.fuselage_chordwise_vortices
            ])
            if missing_separate:
                raise ValueError('If using separate surface discretization, all n_sw and n_cw values must be defined.')

        else:
            # 2. User didn't provide separate settings, so we fallback to the global defaults
            if not self.number_of_spanwise_vortices or not self.number_of_chordwise_vortices:
                raise ValueError('If using global surface discretization, both n_sw and n_cw must be defined.')

            # Route the global settings to the specific component fields
            object.__setattr__(self, 'wing_spanwise_vortices', self.number_of_spanwise_vortices)
            object.__setattr__(self, 'wing_chordwise_vortices', self.number_of_chordwise_vortices)
            object.__setattr__(self, 'fuselage_spanwise_vortices', self.number_of_spanwise_vortices)
            object.__setattr__(self, 'fuselage_chordwise_vortices', self.number_of_chordwise_vortices)


class VLMTopology(eqx.Module):
    """ Static topological mapping for the VLM mesh. Gradients do NOT flow here. """
    
    # Global Counters
    total_panels: int = eqx.field(static=True, default=100)
    total_wings: int = eqx.field(static=True, default=1)
    
    # 1D Integer/Boolean Arrays (Length = total_panels)
    surface_ID: jnp.ndarray       = eqx.field(default_factory = lambda: jnp.empty(0))# Maps panel 'i' to wing 'j'
    is_leading_edge: jnp.ndarray  = eqx.field(default_factory = lambda: jnp.empty(0))# True if panel 'i' is at the leading edge
    is_trailing_edge: jnp.ndarray = eqx.field(default_factory = lambda: jnp.empty(0))# True if panel 'i' is at the trailing edge
    
    # Slicing Helpers 
    surface_breaks: jnp.ndarray = eqx.field(default_factory = lambda: jnp.empty(0))   # The starting panel index for each wing

    @classmethod
    def build_from_records(cls, vlm_records_list, settings):
        """ 
        Pure Python builder. Runs ONCE during initialization to map the grid. 
        """
        surface_id_list = []
        is_le_list = []
        is_te_list = []
        surface_breaks_list = []
        
        total_panels = 0
        
        # We iterate over the 1D list of VLM records we just built!
        for current_surface_id, record in enumerate(vlm_records_list):
            
            # Record the panel index where this wing starts
            surface_breaks_list.append(total_panels)
            
            # --- Discretization Logic ---
            # NOTE: If we are using span_breaks to distribute panels proportionally, 
            # that calculation goes here! For now, we use the global settings.
            n_sw = settings.vortices.wing_spanwise_vortices
            n_cw = settings.vortices.wing_chordwise_vortices
            n_panels = n_sw * n_cw
            
            # Map every panel in this grid to the current wing/surface ID
            surface_id_list.extend([current_surface_id] * n_panels)
            
            # Build Leading Edge boolean mask
            # The first row of panels (n_sw) is the leading edge
            wing_le = [True] * n_sw + [False] * (n_panels - n_sw)
            is_le_list.extend(wing_le)
            
            # Build Trailing Edge boolean mask
            # The last row of panels (n_sw) is the trailing edge
            wing_te = [False] * (n_panels - n_sw) + [True] * n_sw
            is_te_list.extend(wing_te)
            
            total_panels += n_panels

        # Pack everything into immutable JAX arrays exactly ONCE
        return cls(
            total_panels=total_panels,
            total_wings=len(vlm_records_list),
            surface_ID=jnp.array(surface_id_list, dtype=jnp.int32),
            is_leading_edge=jnp.array(is_le_list, dtype=bool),
            is_trailing_edge=jnp.array(is_te_list, dtype=bool),
            surface_breaks=jnp.array(surface_breaks_list, dtype=jnp.int32)
        )

class VLMSettings(eqx.Module): 

    model_fuselage:                 bool    = eqx.field(static=True, default=False)
    trim_aircraft:                  bool    = eqx.field(static=True, default=False)

    discretize_control_surfaces:    bool    = eqx.field(static=True, default=True)
    recalculate_total_wetted_area:  bool    = eqx.field(static=True, default=False)
    model_propeller_wake:           bool    = eqx.field(static=True, default=False)
    VORLAX_empirical_corrections:   bool    = eqx.field(static=True, default=False)

    CL_max:                         float   = jnp.inf
    CD_increment:                   float   = 0.0
    spoiler_drag_increment:         float   = 0.0

    supersonic:     SupersonicSettings      = eqx.field(default_factory=SupersonicSettings)
    correction:     CorrectionFactors       = eqx.field(default_factory=CorrectionFactors)
    efficiency:     EfficiencyFactors       = eqx.field(default_factory=EfficiencyFactors)
    parasite_drag:  ParasiteDragFormFactors = eqx.field(default_factory=ParasiteDragFormFactors)
    training:       Training                = eqx.field(default_factory=Training)
    
    topology:       VLMTopology             = eqx.field(default_factory=VLMTopology)
    vortices:       VLMVortices             = eqx.field(default_factory=VLMVortices)

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------
def _default_VLM_init_steps():
    return(
        ProcessStep(initialize_VLM_geometry, "Initialize VLM Geometry"),
        ProcessStep(discretize_wings, "Discretize VLM Wings"),
        ProcessStep(generate_full_vortex_distribution, "Generate Wing Vortices"),
    )


class InitializeVLM(Process):
    
    tag: str = eqx.field(static=True, default="Initialize Aerodynamics Analysis")
    steps: tuple = eqx.field(default_factory=_default_VLM_init_steps)


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Process
# ----------------------------------------------------------------------------------------------------------------------

def _default_VLM_steps():
    return(
        ProcessStep(check_freestream, "Check Freestream"),
        ProcessStep(update_wing_geometry, "Update Wing Geometry"), # Updates System for shape optimization
        ProcessStep(compute_vlm_rhs, "Calculate Boundary Conditions"),
        ProcessStep(compute_induced_velocity, "Calculate AICs"),
        ProcessStep(compute_vortex_strength, "Compute Vortex Strength"),
        ProcessStep(compute_panel_pressures, "Compute Pressure Coefficients"),
        ProcessStep(compute_coefficients, "Compute Aerodynamic Coefficients")
    )

class VLM(Process):

    tag: str = eqx.field(static=True, default="Aerodynamics")

    steps : tuple = eqx.field(default_factory=_default_VLM_steps)

