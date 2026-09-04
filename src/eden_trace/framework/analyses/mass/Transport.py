from eden_trace.utils import field

from eden_trace.library.methods.mass import transport as Mass

from eden_trace.framework import Process, ProcessStep

# from Trace.Framework.Methods.Mass.Energy import tf_mass_from_SLS


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
    tag: str = field("Transport Mass Analysis", static=True)
    steps: tuple[ProcessStep, ...] = field(_default_transport_steps)
