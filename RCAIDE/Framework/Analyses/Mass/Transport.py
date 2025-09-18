import chex
from dataclasses import make_dataclass
from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Library.Methods.Mass import Transport as Mass

import RCAIDE.Library as rcl


@chex.dataclass(kw_only=True)
class Transport(Process):
    """
    Transport Mass Analysis Class

    This class is used to perform mass analysis for a transport aircraft. It inherits from the `Process` class.

    Attributes
    ----------
    main_wing_mass_reduction_factor : float
        The mass reduction factor for the main wing.
    fuselage_mass_reduction_factor : float
        The mass reduction factor for the fuselage.
    empennage_mass_reduction_factor : float
        The mass reduction factor for the empennage.
    systems_mass_reduction_factor : float
        The mass reduction factor for the systems.
    rudder_sizing_fraction : float
        The rudder area as a fraction of the main wing area.

    Methods
    -------
    __post_init__():
        Initializes the settings for the mass analysis.

        If the appropriate datastructures aren't already in settings, creates them.
        Maps analysis settings into settings datastructure for later retrieval.
        Adds default process steps for mass calculations.
    """

    # Make settings analysis class attributes so that users can see what settings can/must be set when initializing
    # an instance of this analysis, and so that they appear in the docstring of the analysis

    # Mass Reduction Factors
    main_wing_mass_reduction_factor: float = 0.
    fuselage_mass_reduction_factor: float = 0.
    empennage_mass_reduction_factor: float = 0.
    systems_mass_reduction_factor: float = 0.

    # Rudder Sizing Fraction
    rudder_sizing_fraction: float = 0.25

    def __post_init__(self):
        # If the appropriate datastructures aren't already in settings, create them:
        if 'mass_reduction_factors' not in vars(self.settings).keys():
            self.settings.mass_reduction_factors = make_dataclass(cls_name='MassReductionFactors',
                                                                  fields=[
                                                                      ('main_wing', float),
                                                                      ('fuselage', float),
                                                                      ('empennage', float),
                                                                      ('systems', float)
                                                                  ])
        if 'sizing' not in vars(self.settings).keys():
            self.settings.sizing = make_dataclass(cls_name='Sizing',
                                                  fields=[
                                                      ('rudder_fraction', float)
                                                  ])

        # Map analysis settings into settings datastructure for later retrieval
        self.settings.mass_reduction_factors.main_wing    = self.main_wing_mass_reduction_factor
        self.settings.mass_reduction_factors.fuselage     = self.fuselage_mass_reduction_factor
        self.settings.mass_reduction_factors.empennage    = self.empennage_mass_reduction_factor
        self.settings.mass_reduction_factors.systems      = self.systems_mass_reduction_factor

        self.settings.sizing.rudder_fraction              = self.rudder_sizing_fraction

        ###---Default Process Steps---###

        self.append(ProcessStep(name='Propulsion Mass',
                                function=rcl.Methods.Mass.Energy.Jets.Jet_Mass_from_SLS))
        self.append(ProcessStep(name='Passenger & Payload Mass',
                                function=Mass.passenger_payload))
        self.append(ProcessStep(name='Operating System Mass',
                                function=Mass.operating_systems))
        self.append(ProcessStep(name='Main Wing Mass',
                                function=Mass.segmented_main_wing))
        self.append(ProcessStep(name='Horizontal Tail Mass',
                                function=Mass.horizontal_tail))
        self.append(ProcessStep(name='Vertical Tail Mass',
                                function=Mass.vertical_tail))
        self.append(ProcessStep(name='Fuselage Mass',
                                function=Mass.fuselage))
        self.append(ProcessStep(name='Landing Gear',
                                function=Mass.landing_gear))