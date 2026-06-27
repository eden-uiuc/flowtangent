from RCAIDE.utils import init_field

from RCAIDE.Library.Methods.Mass import Transport as Mass

from RCAIDE.Framework import Process, ProcessStep

# from RCAIDE.Framework.Methods.Mass.Energy import tf_mass_from_SLS


def _default_transport_steps() -> tuple[ProcessStep, ...]:
    """Builds the static pipeline of turbofan cycle analysis steps."""
    return (
        # ProcessStep(tag="Propulsion Mass", function=tf_mass_from_SLS),
        ProcessStep(tag="Passenger & Payload Mass", function=Mass.passenger_payload),
        ProcessStep(tag="Operating System Mass", function=Mass.operating_systems),
        ProcessStep(tag="Main Wing Mass", function=Mass.segmented_main_wing),
        ProcessStep(tag="Horizontal Tail Mass", function=Mass.horizontal_tail),
        ProcessStep(tag="Vertical Tail Mass", function=Mass.vertical_tail),
        ProcessStep(tag="Fuselage Mass", function=Mass.fuselage),
        ProcessStep(tag="Landing Gear", function=Mass.landing_gear),
    )


class Transport(Process):
    tag: str = init_field("Transport Mass Analysis", static=True)
    steps: tuple[ProcessStep, ...] = init_field(_default_transport_steps)
