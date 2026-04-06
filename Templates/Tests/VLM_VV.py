# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import subprocess
import os
import re

import jax.numpy as jnp
import equinox as eqx

from RCAIDE.Library.Components import ComponentAreas
from RCAIDE.Library.Components.Wings import Wing, WingChords, WingDimensions

from RCAIDE.Framework import Process, State, Settings
from RCAIDE.Framework.System import Aircraft
from RCAIDE.Framework.Missions.Conditions import Numerics

from RCAIDE.Framework.Analyses.Aerodynamics import VLM, VLMSettings, InitializeVLM, VLMVortices


# AVL Helper Functions -------------------------------------------------------------------------------------------------

def create_avl_geometry(filename, name="Test_Wing", span=10.0, chord=1.0):
    """Writes a simple rectangular wing .avl file."""

    # Global Reference Values
    sref = span * chord
    cref = chord
    bref = span
    xref, yref, zref = 0.0, 0.0, 0.0

    # We define the right half of the wing, so semi-span is span/2
    semi_span = span / 2.0

    avl_text = f"""{name}
#Mach
0.0
#IYsym   IZsym   Zsym
 0       0       0.0
#Sref    Cref    Bref
 {sref}  {cref}  {bref}
#Xref    Yref    Zref
 {xref}  {yref}  {zref}

#======================================================
SURFACE
Main_Wing
#Nchordwise  Cspace  Nspanwise  Sspace
 12          0.0     20         0.0

YDUPLICATE
0.0

#------------------------------------------------------
SECTION
#Xle    Yle    Zle     Chord   Ainc  Nspanwise  Sspace
 0.0    0.0    0.0     {chord}   0.0   0          0

SECTION
#Xle    Yle     Zle    Chord   Ainc  Nspanwise  Sspace
 0.0  {semi_span} 0.0  {chord}   0.0   0          0
"""
    with open(filename, 'w') as f:
        f.write(avl_text)


def run_avl_alpha_sweep(avl_file, alpha, run_name):
    """Executes AVL silently using subprocess and a keystroke macro."""

    stab_file = f"{run_name}_stab.txt"

    # This string represents the EXACT keystrokes you would type in the terminal
    keystrokes = f"""load {avl_file}
oper
a
a
{alpha}
x
st
{stab_file}
O
\n
quit
"""
    # The 'O' above tells AVL to Overwrite the file if it already exists
    # The '\n' drops us out of the OPER menu back to the main menu

    avl_exe = os.path.expanduser("~/.local/bin/avl")
    print(f"\nRunning AVL for {avl_file} at Alpha = {alpha}...")

    # Call AVL natively from your PATH, piping the keystrokes in
    result = subprocess.run(
        [avl_exe],
        input=keystrokes,
        text=True,
        capture_output=True
    )

    # Optional: Check if AVL crashed or threw a Fortran error
    if "Stop" in result.stdout or result.returncode != 0:
        print("AVL encountered an error!")
        print(result.stdout)

    print(f"Done. Saved stability data to {stab_file}")


def parse_avl_stability(stab_file_path):
    """
    Parses an AVL stability (.st) output file and returns a dictionary
    of the core aerodynamic coefficients and stability derivatives.
    """
    results = {}

    with open(stab_file_path, 'r') as f:
        text = f.read()

    # 1. Parse Global Core Coefficients (e.g., CLtot =  0.34980)
    # The regex \s*=\s* handles any number of spaces around the equals sign
    # The ([-.\d]+) captures the actual number, including negatives and decimals
    core_keys = ['Alpha', 'CLtot', 'CDtot', 'Cmtot']
    for key in core_keys:
        match = re.search(fr"{key}\s*=\s*([-.\d]+)", text)
        if match:
            results[key] = float(match.group(1))

    # 2. Parse Stability Derivatives (e.g., CLa   4.500000)
    # Derivatives in AVL don't have an equals sign, just spaces
    # Example format: " CLa      4.500000"
    deriv_keys = ['CLa', 'Cma', 'Clb', 'Cnb', 'Cmq', 'Cnr']
    for key in deriv_keys:
        # Look for the key, followed by spaces, then capture the number
        match = re.search(fr"\b{key}\s+([-.\d]+)", text)
        if match:
            results[key] = float(match.group(1))

    return results

# VORJAX Helper Functions ----------------------------------------------------------------------------------------------


def create_vorjax_geometry(span=10.0, chord=1.0):

    wing_spans = WingDimensions(projected=span)

    wing_chords = WingChords(root=chord, tip=chord, mean_aerodynamic=chord)

    wing_areas = ComponentAreas(reference=span * chord, wetted=2 * span * chord)

    wing = Wing(
        symmetric=True,
        spans=wing_spans,
        chords=wing_chords,
        areas=wing_areas,
        origin=jnp.array([[0.0, 0.0, 0.0]]),
        aerodynamic_center=jnp.array([0.0, 0.0, 0.0])
    )

    system = Aircraft(tag='Test Aircraft', areas=wing_areas).add_subcomponent(wing)
    system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

    return system


def run_vorjax_alpha_sweep(vehicle, alpha):

    state = State(numerics=Numerics(number_of_control_points=1, calculate_integration=False))
    frozen_initials = eqx.tree_at(lambda s: s.initials, state, None, is_leaf=lambda x: x is None)
    state = eqx.tree_at(lambda s: s.initials, state, frozen_initials, is_leaf=lambda x: x is None)

    # Set State Values
    initial_state = eqx.tree_at(
        lambda s: (s.stability.static.roll_rate, s.stability.static.pitch_rate, s.stability.static.yaw_rate),
        state,
        (jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    )

    initial_state = eqx.tree_at(lambda s: s.freestream.speed, initial_state, jnp.array([100.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.mach_number, initial_state, jnp.array([0.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.density, initial_state, jnp.array([1.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.temperature, initial_state, jnp.array([273.15]))
    initial_state = eqx.tree_at(lambda s: s.frames.inertial.velocity_vector, initial_state, jnp.array([100.0, 0., 0.]))
    initial_state = eqx.tree_at(lambda s: s.aerodynamics.angles.alpha, initial_state, jnp.deg2rad(alpha) * jnp.ones(1))

    initial_state = initial_state.expand_rows(1)

    initial_system = vehicle

    vortices = VLMVortices(
        spanwise_cosine_spacing=False,
        number_of_spanwise_vortices=20,
        number_of_chordwise_vortices=12
    )
    aero_settings = VLMSettings(vortices=vortices)
    initial_settings = eqx.tree_at(lambda s: s.analysis.aerodynamics, Settings(DEBUG_MODE=False), aero_settings)

    analysis = Process(
        steps=(
            InitializeVLM(),
            VLM()
        ),
        initial_state=initial_state,
        initial_system=initial_system,
        initial_settings=initial_settings
    )

    final_state, final_system, final_settings = analysis.run(initial_state, initial_system, initial_settings)

    alpha = jnp.rad2deg(final_state.aerodynamics.angles.alpha).item(0)
    CL = final_state.aerodynamics.coefficients.lift.total.item(0)
    CD = final_state.aerodynamics.coefficients.drag.induced.inviscid.total.item(0)
    CM = final_state.aerodynamics.coefficients.moments.pitch.item(0)

    return alpha, CL, CD, CM


if __name__ == "__main__":
    geometry_file = "vnv_test1.avl"

    # 1. Generate the geometry file
    create_avl_geometry(geometry_file, span=10.0, chord=1.0)

    # 2. Run AVL at 2.0 degrees Angle of Attack
    run_avl_alpha_sweep(geometry_file, alpha=2.0, run_name="test1")

    parsed_data = parse_avl_stability("test1_stab.txt")

    print("\n--- Extracted AVL Results ---")
    for k, v in parsed_data.items():
        print(f"{k}: {v}")

    vehicle = create_vorjax_geometry(span=10.0, chord=1.0)
    alpha, CL, CD, CM = run_vorjax_alpha_sweep(vehicle, alpha=2.0)
    print(f"\n--- Extracted VORJAX Results ---")
    print(f"Alpha: {alpha}")
    print(f"CL: {CL:.5f}")
    print(f"CD: {CD:.5f}")
    print(f"CM: {CM:.5f}")