# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import subprocess
import os
import re

from pathlib import Path

import jax.numpy as jnp
import equinox as eqx
import plotly.graph_objects as go
import numpy as np

from RCAIDE.Library.Components import ComponentAreas
from RCAIDE.Library.Components.Wings import Wing, WingChords, WingDimensions

from RCAIDE.Framework import Process, State, Settings
from RCAIDE.Framework.System import Aircraft
from RCAIDE.Framework.Missions.Conditions import Numerics

from RCAIDE.Framework.Analyses.Aerodynamics import VLM, VLMSettings, InitializeVLM, VLMVortices

from RCAIDE.Framework.Interfaces.AVL import parse_avl_file, convert_to_RCAIDE
from RCAIDE.Framework.Plotting import plot_vlm_panels


# AVL Helper Functions -------------------------------------------------------------------------------------------------

def AVL_straight_wing(filename, name="Test_Wing", span=10.0, chord=1.0):
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


def run_AVL_alpha_sweep(avl_file, alpha, run_name, oper_mode="st"):
    """Executes AVL silently using subprocess and a keystroke macro."""

    file_name = f"{run_name}_{oper_mode}.txt"

    # This string represents the EXACT keystrokes you would type in the terminal
    keystrokes = f"""load {avl_file}
oper
a
a
{alpha}
x
{oper_mode}
{file_name}
O

quit
"""
    # The 'O' above tells AVL to Overwrite the file if it already exists
    # The '\n' drops us out of the OPER menu back to the main menu

    avl_exe = os.path.expanduser("~/.local/bin/avl")
    print(f"\nRunning AVL for {avl_file} at Alpha = {alpha}...")

    # Call AVL natively from your PATH, piping the keystrokes in
    try:
        result = subprocess.run(
            [avl_exe],
            input=keystrokes,
            text=True,
            capture_output=True,
            timeout=5.0
        )
    except subprocess.TimeoutExpired:
        print("CRITICAL: AVL hung in an infinite loop and was terminated.")
        # You can optionally run a system kill command here just to be safe
        os.system("pkill -9 avl")
        return

    # Optional: Check if AVL crashed or threw a Fortran error
    if "Stop" in result.stdout or result.returncode != 0:
        print("AVL encountered an error!")
        print(result.stdout)

    print(f"Done. Saved stability data to {file_name}")


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
def VORJAX_straight_wing(span=10.0, chord=1.0):

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


def VORJAX_test_run(vehicle, alpha, Mach, debug_mode=False):

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
    initial_state = eqx.tree_at(lambda s: s.freestream.mach_number, initial_state, jnp.array([Mach]))
    initial_state = eqx.tree_at(lambda s: s.freestream.density, initial_state, jnp.array([1.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.temperature, initial_state, jnp.array([273.15]))
    initial_state = eqx.tree_at(lambda s: s.frames.inertial.velocity_vector, initial_state, jnp.array([100.0, 0., 0.]))
    initial_state = eqx.tree_at(lambda s: s.aerodynamics.angles.alpha, initial_state, jnp.deg2rad(alpha) * jnp.ones(1))

    initial_state = initial_state.expand_rows(1)

    initial_system = vehicle

    vortices = VLMVortices(
        spanwise_cosine_spacing=False,
        spanwise_vortices=(26, 5, 5),
        chordwise_vortices=(12, 3, 3)
    )
    aero_settings = VLMSettings(vortices=vortices)
    initial_settings = eqx.tree_at(lambda s: s.analysis.aerodynamics, Settings(DEBUG_MODE=debug_mode), aero_settings)

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

    analysis_data = final_system.analysis_data
    alpha = jnp.rad2deg(final_state.aerodynamics.angles.alpha).item(0)
    CL = final_state.aerodynamics.coefficients.lift.total.item(0)
    CD = final_state.aerodynamics.coefficients.drag.induced.inviscid.total.item(0)
    CM = final_state.aerodynamics.coefficients.moments.pitch.item(0)

    return alpha, CL, CD, CM, analysis_data



def AVL_basic_test(geometry_file=None, run_name=None, oper_mode="st", alpha=2.0, span=10.0, chord=1.0):
    # 1. Generate the geometry file
    if geometry_file is None:
        AVL_straight_wing(geometry_file, span=span, chord=chord)
        geometry_file = f"{run_name}.avl"

    if run_name is None:
        run_name = Path(geometry_file).stem

    run_AVL_alpha_sweep(geometry_file, alpha=alpha, run_name=run_name, oper_mode=oper_mode)

    if oper_mode == "st":
        parsed_data = parse_avl_stability(f"{run_name}_{oper_mode}.txt")

        print("\n--- Extracted AVL Results ---")
        for k, v in parsed_data.items():
            print(f"{k}: {v}")
        print("\n")


if __name__ == "__main__":

    geometry_file = './AVL Test Cases/b737_wings_flat.avl'

    avl_b737_data = parse_avl_file(Path('./AVL Test Cases/b737_wings_flat.avl'))
    vehicle = convert_to_RCAIDE(avl_b737_data)
    # vehicle = VORJAX_straight_wing(span=10.0, chord=1.0)

    AVL_basic_test(geometry_file, oper_mode="st")

    alpha, CL, CD, CM, data = VORJAX_test_run(vehicle, alpha=2.0, Mach=0.00, debug_mode=True)

    print(f"\n--- Extracted VORJAX Results ---")
    print(f"Alpha: {alpha}")
    print(f"CL: {CL:.5f}")
    print(f"CD: {CD:.5f}")
    print(f"CM: {CM:.5f}")

    # 3. Plot the Vortex Distribution
    VD = data['vortex_distribution']
    fig = plot_vlm_panels(VD, panel_values=np.asarray(data['pressure_coefficients'].squeeze(0)))
    fig.show()

    print("Done!")