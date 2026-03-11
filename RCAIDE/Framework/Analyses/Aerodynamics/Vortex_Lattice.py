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
                                                                  compute_panel_pressures, compute_boundary_conditions, 
                                                                  compute_vortex_strength, update_wing_geometry,
                                                                  initialize_VLM_geometry, discretize_wings,
                                                                  generate_full_vortex_distribution,
                                                                  apply_aerodynamic_forces)

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

class VLMSettings(eqx.Module): 

    model_fuselage:                 bool    = eqx.field(static=True, default=False)
    trim_aircraft:                  bool    = eqx.field(static=True, default=False)

    discretize_control_surfaces:    bool    = eqx.field(static=True, default=True)
    recalculate_total_wetted_area:  bool    = eqx.field(static=True, default=False)
    model_propeller_wake:           bool    = eqx.field(static=True, default=False)
    VORLAX_empirical_corrections:   bool    = eqx.field(static=True, default=True)

    CL_max:                         float   = jnp.inf
    CD_increment:                   float   = 0.0
    spoiler_drag_increment:         float   = 0.0

    supersonic:     SupersonicSettings      = eqx.field(default_factory=SupersonicSettings)
    correction:     CorrectionFactors       = eqx.field(default_factory=CorrectionFactors)
    efficiency:     EfficiencyFactors       = eqx.field(default_factory=EfficiencyFactors)
    parasite_drag:  ParasiteDragFormFactors = eqx.field(default_factory=ParasiteDragFormFactors)
    training:       Training                = eqx.field(default_factory=Training)
    
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
        # Lift and Induced Drag
        ProcessStep(check_freestream, "Check Freestream"),
        ProcessStep(compute_boundary_conditions, "Calculate Boundary Conditions"),
        ProcessStep(compute_induced_velocity, "Calculate AICs"),
        ProcessStep(compute_vortex_strength, "Compute Vortex Strength"),
        ProcessStep(compute_panel_pressures, "Compute Pressure Coefficients"),
        ProcessStep(compute_coefficients, "Compute Aerodynamic Coefficients"),
        ProcessStep(apply_aerodynamic_forces, "Apply Aerodynamic Forces"),

        # Parasite Drag
        
    )
    # TODO: Add trimming/stability analysis

class VLM(Process):

    tag: str = eqx.field(static=True, default="Aerodynamics")

    steps : tuple = eqx.field(default_factory=_default_VLM_steps)

