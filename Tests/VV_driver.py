# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import subprocess
import os
import re
import json

from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import trange
from plotly.subplots import make_subplots

import RCAIDE.utils as ru

from RCAIDE.Library import Units
from RCAIDE.Library.Components import ComponentAreas
from RCAIDE.Library.Components.Wings import Wing, WingSegment, WingChords, WingDimensions, WingSweeps
from RCAIDE.Library.Components.Airfoils import Airfoil, Airfoil_Data

from RCAIDE.Library.Methods.Aerodynamics.Transonic import ensemble_CL_spline

from RCAIDE.Framework import Process, State, Settings, GradientMap
from RCAIDE.Framework.System import Aircraft
from RCAIDE.Framework.Missions.Conditions import Numerics

from RCAIDE.Framework.Analyses.Aerodynamics import VORJAX, VLMSettings, InitializeVORJAX, Vortices, SupersonicSettings

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

def parse_avl_fe_dcp(filepath):
    """
    Parses an AVL 'FE' output file and extracts the panel dCp values.
    Returns a flattened 1D NumPy array of the dCp values.
    """
    dcp_list = []
    dcp_col_idx = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1. Dynamically find the column index for dCp
    for line in lines:
        if 'dCp' in line:
            tokens = line.split()
            try:
                dcp_col_idx = tokens.index('dCp')
            except ValueError:
                continue
            break

    if dcp_col_idx is None:
        raise ValueError("Could not find 'dCp' header in the AVL FE file.")

    # 2. Extract the panel data
    for line in lines:
        tokens = line.split()

        # A valid panel data line will have enough columns and start with one integer (I)
        if len(tokens) > dcp_col_idx and tokens[0].isdigit():
            try:
                dcp_value = float(tokens[dcp_col_idx])
                dcp_list.append(dcp_value)
            except ValueError:
                # Catch any weird formatting edges where the column isn't a float
                continue

    return np.array(dcp_list)

def AVL_basic_test(geometry_file=None, run_name=None, oper_mode="st", alpha=2.0, span=10.0, chord=1.0):
    # 1. Generate the geometry file
    if run_name is None:
        run_name = Path(geometry_file).stem

    if geometry_file is None:
        geometry_file = f"{run_name}.avl"
        AVL_straight_wing(geometry_file, span=span, chord=chord)

    run_AVL_alpha_sweep(geometry_file, alpha=alpha, run_name=run_name, oper_mode=oper_mode)

    if oper_mode == "st":
        parsed_data = parse_avl_stability(f"{run_name}_{oper_mode}.txt")

        print("\n--- Extracted AVL Results ---")
        for k, v in parsed_data.items():
            print(f"{k}: {v}")
        print("\n")
    elif oper_mode == "fe":
        parsed_data = parse_avl_fe_dcp(f"{run_name}_{oper_mode}.txt")
    else:
        parsed_data = None

    return parsed_data

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

def VORJAX_elliptical_wing(AR=10., n_segments=1):

    span = AR
    c_root = (4.0 * span) / (jnp.pi * AR)
    S_ref = span**2 / AR

    def chord_frac_at_eta(eta):
        # Using 0.99999 to prevent divide-by-zero or NaNs at the absolute tip
        return jnp.sqrt(1.0 - jnp.clip(eta, 0.0, 0.99999)**2)
    
    segments = ()
    eta = jnp.cos(jnp.linspace(jnp.pi/2, 0, n_segments + 1))
    
    for i in range(n_segments):
        eta_start = eta[i]
        eta_end   = eta[i+1]
        
        chord_frac_start = chord_frac_at_eta(eta_start)
        chord_frac_end   = chord_frac_at_eta(eta_end)

        # 1. Calculate the required setback of the quarter-chord
        # (Positive delta_x means the c/4 line is moving backward/sweeping aft)
        delta_x_c4 = 0.25 * 1.0 * (chord_frac_start - chord_frac_end) 
        
        # 2. Calculate the spanwise distance of this segment
        delta_y = 0.5 * AR * (eta_end - eta_start)
        
        # 3. Get the sweep angle
        sweep_c4 = jnp.arctan2(delta_x_c4, delta_y)

        segments += (WingSegment(
            tag=f"{i}", 
            percent_span_location=eta_start, 
            root_chord_percent=chord_frac_start,
            sweeps=WingSweeps(quarter_chord=sweep_c4)  # Inject sweep here!
        ),)

    # Tip segment doesn't need a sweep since there's no geometry after it
    segments += (WingSegment(
        tag="Tip", 
        percent_span_location=1.0, 
        root_chord_percent=0.01
    ),)

    wing_areas = ComponentAreas(reference=S_ref, wetted=2.0 * S_ref)

    wing = Wing(tag=f"Elliptical {n_segments}",
                segments=segments,
                symmetric=True,
                spans=WingDimensions(projected=span),
                chords=WingChords(root=c_root, tip=0.01 * c_root, mean_aerodynamic=c_root * 8.0 / (3 * jnp.pi)),
                areas=wing_areas,
                taper=0.01,
                origin=jnp.array([[0.0, 0.0, 0.0]]),
                aerodynamic_center=jnp.array([0.0, 0.0, 0.0])).update_geometry()
    
    system = Aircraft(tag='Test Aircraft', areas=wing_areas).add_subcomponent(wing)
    system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

    return system  

def VORJAX_delta_wing(AR=2.0):
    
    # Define a fixed span, calculate the rest
    span = 10.0
    S_ref = span**2 / AR
    c_root = 2.0 * S_ref / span
    
    # Prevent divide-by-zero singularities at the tip
    c_tip_ratio = 0.001 
    
    # Calculate Quarter-Chord Sweep for a straight trailing edge
    # tan(sweep) = (0.75 * c_root) / (span / 2)
    sweep_c4 = jnp.arctan((0.75 * c_root) / (span / 2.0))
    
    segments = (
        WingSegment(
            tag="Root_to_Tip",
            percent_span_location=0.0,
            root_chord_percent=1.0,
            sweeps=WingSweeps(quarter_chord=sweep_c4)
        ),
        WingSegment(
            tag="Tip",
            percent_span_location=1.0,
            root_chord_percent=c_tip_ratio
        )
    )

    wing_areas = ComponentAreas(reference=S_ref, wetted=2.0 * S_ref)

    wing = Wing(tag=f"Delta_AR_{AR}",
                segments=segments,
                symmetric=True,
                spans=WingDimensions(projected=span),
                chords=WingChords(root=c_root, tip=c_root * c_tip_ratio, mean_aerodynamic=2.0/3.0 * c_root),
                areas=wing_areas,
                taper=c_tip_ratio,
                origin=jnp.array([[0.0, 0.0, 0.0]]),
                aerodynamic_center=jnp.array([0.0, 0.0, 0.0])).update_geometry()
    
    system = Aircraft(tag='Delta Aircraft', areas=wing_areas).add_subcomponent(wing)
    system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

    return system

def VORJAX_ONERA_M6():
    """
    ONERA M6 Geometry Definition.
    src: https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html
    """

    AR = 3.8
    taper = 0.56
    
    sweep_qc = 26.7 * Units.deg
    sweep_le = 30.0 * Units.deg
    sweep_te = 15.8 * Units.deg

    c_root = 805.9 * Units.mm
    semispan = 1196.3 * Units.mm

    mac = 0.64607 * Units.m

    segments = (
        WingSegment(
            tag="ONERA M6",
            percent_span_location=0.0,
            root_chord_percent=1.0,
            sweeps=WingSweeps(leading_edge=sweep_le, quarter_chord=sweep_qc),
            airfoil=Airfoil.from_file(Airfoil_Data/"ONERA_M6.txt")
        ),
        WingSegment(
            tag="Tip",
            percent_span_location=1.0,
            root_chord_percent=taper,
            airfoil=Airfoil.from_file(Airfoil_Data/"ONERA_M6.txt")
        )
    )
    
    onera_wing = Wing(
        tag="Main Wing",
        symmetric=True,
        segments=segments,
        aspect_ratio=AR,
        taper=0.56,
        origin=jnp.array([[0.0, 0.0, 0.0]]),
        chords=WingChords(root=c_root, mean_aerodynamic=mac),
        spans=WingDimensions(projected=2 * semispan),
    ).update_geometry(calculate_reference_area=True, calculate_wetted_area=True)

    system = Aircraft(tag='ONERA M6 Container', areas=onera_wing.areas).add_subcomponent(onera_wing)
    system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

    return system

def VORJAX_test_run(vehicle, alpha, Mach, n_sw=20, n_cw=6, cos_sw=True, shock=False, grad_map=None, near_field=False, debug_mode=False):

    state = State(numerics=Numerics(number_of_control_points=1, calculate_integration=False))
    frozen_initials = eqx.tree_at(lambda s: s.initials, state, None, is_leaf=lambda x: x is None)
    state = eqx.tree_at(lambda s: s.initials, state, frozen_initials, is_leaf=lambda x: x is None)

    # Set State Values
    initial_state = eqx.tree_at(
        lambda s: (s.stability.static.roll_rate, s.stability.static.pitch_rate, s.stability.static.yaw_rate),
        state,
        (jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    )

    if isinstance(alpha, list | jnp.ndarray) and isinstance(Mach, list | jnp.ndarray):
        assert len(alpha) == len(Mach)
        alpha = jnp.array(alpha).reshape(-1, 1)
        Mach = jnp.array(Mach).reshape(-1, 1)
    else:
        alpha = jnp.array([alpha])
        Mach = jnp.array([Mach])
    
    initial_state = eqx.tree_at(lambda s: s.aerodynamics.angles.alpha, initial_state, alpha)
    initial_state = eqx.tree_at(lambda s: s.freestream.mach_number, initial_state, Mach)

    initial_state = eqx.tree_at(lambda s: s.freestream.speed, initial_state, jnp.array([100.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.density, initial_state, jnp.array([1.0]))
    initial_state = eqx.tree_at(lambda s: s.freestream.gamma, initial_state, jnp.array([1.4]))
    initial_state = eqx.tree_at(lambda s: s.freestream.temperature, initial_state, jnp.array([273.15]))
    initial_state = eqx.tree_at(lambda s: s.frames.inertial.velocity_vector, initial_state, jnp.array([100.0, 0., 0.]))

    initial_state = initial_state.expand_rows(len(alpha))

    initial_system = vehicle

    vortices = Vortices(
        spanwise_cosine=cos_sw,
        n_spanwise=n_sw,
        n_chordwise=n_cw
    )

    mach_settings = SupersonicSettings(
        peak_mach_number=2.0,
        begin_blend_mach=0.7,
        end_blend_mach=1.2,
        shock_correction=shock
    )
    
    aero_settings = VLMSettings(vortices=vortices, supersonic=mach_settings, near_field_drag=near_field)
    initial_settings = eqx.tree_at(lambda s: s.analysis.aerodynamics, Settings(DEBUG_MODE=debug_mode), aero_settings)

    analysis = Process(
        tag="VORJAX Test Run",
        steps=(
            InitializeVORJAX(),
            VORJAX()
        ),
        initial_state=initial_state,
        initial_system=initial_system,
        initial_settings=initial_settings
    )

    results = analysis.run(
        initial_state,
        initial_system,
        initial_settings,
        grad_map=grad_map
    )

    return results

# Plotting Helper Functions --------------------------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to seamlessly convert NumPy arrays to JSON lists."""
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def save_plot_cache(plot_key, filepath="./Tests/VORJAX/plotting.json", **kwargs):
    """Saves kwargs to a JSON file under a specific plot_key."""
    cache = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            cache = json.load(f)
            
    cache[plot_key] = kwargs
    
    with open(filepath, 'w') as f:
        json.dump(cache, f, cls=NumpyEncoder, indent=4)
    print(f"Cached data for '{plot_key}' to {filepath}")

def load_plot_cache(key, filepath="./Tests/VORJAX/plotting.json"):
    """Loads the entire JSON cache dictionary."""
    if not os.path.exists(filepath):
        print(f"Warning: Cache file {filepath} not found.")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)[key]

# Plotting Functions ---------------------------------------------------------------------------------------------------

def plot_spanwise_loading_mpl(eta, gamma, CL, b, AR, v_inf):
    # Set global font to match LaTeX standard
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 8  # 10pt for clear readability at column width

    # Convert inputs to numpy arrays just in case they are lists
    eta = np.array(eta)
    gamma = np.array(gamma) * v_inf

    # 1. Calculate Theoretical Distribution for Error Checking
    # Avoid div-by-zero at the exact tip by masking eta < 0.999
    valid_mask = np.abs(eta) < 0.999
    gamma_theory_discrete = (2 * v_inf * b * CL) / (np.pi * AR) * np.sqrt(1 - eta[valid_mask]**2)
    
    # Calculate Mean Absolute Relative Error (as a percentage)
    mean_error = np.mean(np.abs(gamma[valid_mask] - gamma_theory_discrete) / gamma_theory_discrete) * 100.0

    # 2. Calculate Dense Theoretical Curve for Smooth Plotting
    # Assuming half-span plotting from root (0) to tip (1)
    eta_dense = np.linspace(0, 1, 500)
    gamma_dense = (2 * v_inf * b * CL) / (np.pi * AR) * np.sqrt(1 - eta_dense**2)

    # 3. Initialize Figure (3.5 inch width for AIAA standard single column)
    _, ax = plt.subplots(figsize=(3.5, 2.6))

    # 4. Add Trace: Theoretical Distribution (Smooth Dashed Line)
    ax.plot(eta_dense, gamma_dense, color='black', linestyle='--', linewidth=1.5, 
            label='Lifting Surface Theory')

    # 5. Add Trace: VORJAX Actual Results (Scatter points, no connecting line)
    ax.plot(eta, gamma, color='black', linestyle='None', marker='o', 
            markersize=5, markerfacecolor='black', label='VORJAX Integration')

    # 6. Formatting X and Y axes
    ax.set_xlabel(r"Non-dimensional Semi-Span ($\eta = 2y/b$)", fontweight='bold')
    ax.set_ylabel(r"Circulation ($\Gamma$) [m$^2$/s]", fontweight='bold')
    
    ax.set_xlim(left=0, right=1.05)
    ax.set_ylim(bottom=0, top=max(gamma_dense)*1.15) # Leave room for legend/callout
    
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='both')
    ax.grid(True, linestyle='-', color='#E5E5E5', alpha=0.7)

    # 7. Add Error Callout Box
    bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.7)
    ax.text(0.05, 0.08, f"Mean Error: {mean_error:.2f}%", 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=bbox_props)

    # 8. Legend
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

    # Title is optional, can be commented out for the \caption{} in LaTeX
    plt.title("Spanwise Lift Distribution: Elliptical Wing", fontweight='bold')
    
    # 9. Export to native PDF vector graphic
    plt.tight_layout()
    plt.savefig("./Tests/VORJAX/plots/spanwise_loading.pdf", format='pdf', bbox_inches='tight')

def plot_elliptical_convergence_mpl(n_segments, grad_AD, error, grad_truth):
    # Set global font to match LaTeX standard
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 8 # 11pt maps perfectly to AIAA column widths
    
    # Convert error to percentage
    error_percent = np.array(error) * 100.0

    # 1. Initialize figure (AIAA single column is ~3.5 inches wide. 3.5 x 2.6 is a good aspect ratio)
    _, ax1 = plt.subplots(figsize=(3.5, 2.6))

    # 2. Add Trace: AD Gradient (Primary Y) - Solid line, filled circles
    ax1.plot(n_segments, grad_AD, color='black', linestyle='-', linewidth=1.5, 
             marker='o', markersize=5, markerfacecolor='black', label='VORJAX AD Gradient')

    # 3. Add Trace: Analytical Truth (Primary Y) - Dashed line, no markers
    ax1.axhline(y=grad_truth, color='black', linestyle='--', linewidth=1.5, label="Hembold's Equation")

    # 4. Initialize Secondary Y-axis
    ax2 = ax1.twinx()

    # 5. Add Trace: Relative Error (Secondary Y) - Dotted line, open squares
    ax2.plot(n_segments, error_percent, color='black', linestyle=':', linewidth=1.5, 
             marker='s', markersize=5, markerfacecolor='none', markeredgecolor='black', 
             label='Relative Error')

    # 6. Formatting X-axis
    ax1.set_xlabel("Number of Spanwise Segments", fontweight='bold')
    ax1.set_xlim(left=0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.tick_params(axis='both', direction='in', top=True)

    # 7. Formatting Primary Y-axis
    ax1.set_ylabel(r"Autodiff Gradient ($C_{L_\alpha}$) [rad$^{-1}$]", fontweight='bold')
    ax1.set_ylim(bottom=0, top=5.1)
    ax1.grid(True, linestyle='-', color='#E5E5E5', alpha=0.7)

    # 8. Formatting Secondary Y-axis
    ax2.set_ylabel("Relative Error [%]", fontweight='bold')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', direction='in')

    # 9. Combine Legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', 
               frameon=True, edgecolor='black', fancybox=False)

    plt.title("Elliptical Wing Lift Convergence", fontweight='bold')
    
    # 10. Export to native PDF vector graphic
    plt.tight_layout()
    plt.savefig("./Tests/VORJAX/plots/elliptical_lift.pdf", format='pdf', bbox_inches='tight')

def plot_elliptical_drag_mpl(n_segments, grad_AD, field):
    # Set global font to match LaTeX standard
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 8 # 11pt maps perfectly to AIAA column widths

    AR = 8.0
    alpha = 2.0 * Units.deg
    grad_truth = (2.0  * alpha / (jnp.pi * AR)) * (2.0 * jnp.pi/(jnp.sqrt(1 + (2 / AR)**2) + 2/AR))**2

    # Convert error to percentage
    error = abs(np.asarray(grad_AD) - grad_truth)/grad_truth
    error_percent = np.array(error) * 100.0

    # 1. Initialize figure (AIAA single column is ~3.5 inches wide. 3.5 x 2.6 is a good aspect ratio)
    _, ax1 = plt.subplots(figsize=(3.5, 2.6))

    # 2. Add Trace: AD Gradient (Primary Y) - Solid line, filled circles
    ax1.plot(n_segments, grad_AD, color='black', linestyle='-', linewidth=1.5, 
             marker='o', markersize=5, markerfacecolor='black', label='VORJAX AD Gradient')

    # 3. Add Trace: Analytical Truth (Primary Y) - Dashed line, no markers
    ax1.axhline(y=grad_truth, color='black', linestyle='--', linewidth=1.5, label="Munk's Stagger Thm.")

    # 4. Initialize Secondary Y-axis
    ax2 = ax1.twinx()

    # 5. Add Trace: Relative Error (Secondary Y) - Dotted line, open squares
    ax2.plot(n_segments, error_percent, color='black', linestyle=':', linewidth=1.5, 
             marker='s', markersize=5, markerfacecolor='none', markeredgecolor='black', 
             label='Relative Error')

    # 6. Formatting X-axis
    ax1.set_xlabel("Number of Spanwise Segments", fontweight='bold')
    ax1.set_xlim(left=0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.tick_params(axis='both', direction='in', top=True)

    # 7. Formatting Primary Y-axis
    ax1.set_ylabel(r"Autodiff Gradient ($C_{Di_\alpha}$) [rad$^{-1}$]", fontweight='bold')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle='-', color='#E5E5E5', alpha=0.7)

    # 8. Formatting Secondary Y-axis
    ax2.set_ylabel("Relative Error [%]", fontweight='bold')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', direction='in')

    # 9. Combine Legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', 
               frameon=True, edgecolor='black', fancybox=False)

    plt.title("Elliptical Wing Drag Convergence", fontweight='bold')
    
    # 10. Export to native PDF vector graphic
    plt.tight_layout()
    plt.savefig(f"./Tests/VORJAX/plots/elliptical_drag_{field}.pdf", format='pdf', bbox_inches='tight')

def plot_elliptical_convergence_plotly(n_segments, grad_AD, error, grad_truth):
    
    # 1. Initialize figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Convert error to a percentage for cleaner reading
    error_percent = np.array(error) * 100.0

    # 2. Add Trace: AD Gradient (Primary Y)
    fig.add_trace(
        go.Scatter(
            x=n_segments, 
            y=grad_AD, 
            name="VORJAX AD Gradient",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )

    # 3. Add Trace: Analytical Truth (Primary Y)
    # Drawing a line from the first to the last x-coordinate
    fig.add_trace(
        go.Scatter(
            x=[n_segments[0], n_segments[-1]],
            y=[grad_truth, grad_truth],
            name="Analytical Truth",
            mode="lines",
            line=dict(color="black", width=3, dash="dash")
        ),
        secondary_y=False,
    )

    # 4. Add Trace: Relative Error (Secondary Y)
    fig.add_trace(
        go.Scatter(
            x=n_segments, 
            y=error_percent, 
            name="Relative Error",
            mode="lines+markers",
            line=dict(color="#d62728", width=3),
            marker=dict(symbol="square", size=8)
        ),
        secondary_y=True,
    )

    # 5. Styling and Layout
    fig.update_layout(
        title_text="<b>Elliptical Wing Grid Convergence: VORJAX vs. Lifting-Line Theory</b>",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified", # Combines all hover info into one box
        legend=dict(
            x=0.75,
            y=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    # Format X-axis (Force integer ticks)
    fig.update_xaxes(
        title_text="<b>Number of Spanwise Segments</b>", 
        tickmode="linear", 
        tick0=0, 
        dtick=2,
        showgrid=True,
        gridcolor="lightgrey"
    )

    # Format Primary Y-axis
    fig.update_yaxes(
        title_text="<b>Autodiff Gradient (C<sub>L&alpha;</sub>) [rad<sup>-1</sup>]</b>", 
        color="#1f77b4",
        secondary_y=False,
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
        rangemode="tozero"

    )

    # Format Secondary Y-axis
    fig.update_yaxes(
        title_text="<b>Relative Error [%]</b>", 
        color="#d62728",
        secondary_y=True,
        showgrid=False, # Disabled to prevent clashing with primary gridlines
        zeroline=False,
        rangemode="tozero"
    )

    fig.show()

def plot_fd_v_curve_plotly(step_sizes, fd_errors):
    
    fig = go.Figure()

    # 1. Add Trace: FD Absolute Error
    fig.add_trace(
        go.Scatter(
            x=step_sizes, 
            y=fd_errors, 
            name="Central Difference Error",
            mode="lines+markers",
            line=dict(color="#9467bd", width=3),  # Muted purple
            marker=dict(symbol="circle", size=8),
            hovertemplate='Step Size (h): %{x:.1e}<br>Absolute Error: %{y:.1e}<extra></extra>'
        )
    )

    # 2. Add an invisible "floor" trace just to show where AD lives (machine zero)
    # This emphasizes that AD doesn't have a step size; it just sits at ~0 error.
    fig.add_trace(
        go.Scatter(
            x=[min(step_sizes), max(step_sizes)],
            y=[1e-16, 1e-16],
            name="VORJAX AD Error Bound (Machine Zero)",
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            hoverinfo="skip"
        )
    )

    # 3. Styling and Layout
    fig.update_layout(
        title_text="<b>Finite Difference vs. Autodiff: The Step-Size Dilemma</b>",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        margin=dict(l=60, r=60, t=100, b=60)
    )

    # 4. Format X-axis (Log Scale)
    fig.update_xaxes(
        title_text="<b>Finite Difference Step Size (h)</b>", 
        type="log",
        exponentformat="e",
        showgrid=True,
        gridcolor="lightgrey",
        # Ascending order is standard (1e-15 -> 1e-1)
        autorange="reversed" if step_sizes[0] > step_sizes[-1] else True 
    )

    # 5. Format Y-axis (Log Scale)
    fig.update_yaxes(
        title_text="<b>Absolute Error vs. AD Gradient</b>", 
        type="log",
        exponentformat="e",
        showgrid=True,
        gridcolor="lightgrey"
    )

    # 6. Educational Annotations
    # Floating on the left side (Round-off region)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.85,
        text="<b>Round-off Error</b><br>Dominates",
        showarrow=False,
        font=dict(color="#d62728", size=13),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#d62728",
        borderwidth=1
    )

    # Floating on the right side (Truncation region)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.85,
        text="<b>Truncation Error</b><br>Dominates",
        showarrow=False,
        font=dict(color="#1f77b4", size=13),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#1f77b4",
        borderwidth=1
    )

    fig.show()

def plot_fd_v_curve_mpl(step_sizes, fd_errors):
    # Set global font to match LaTeX standard (AIAA prefers 8pt for figure text)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 8 

    # 1. Initialize figure (AIAA single column is ~3.5 inches wide)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # 2. Add Trace: FD Absolute Error (Solid line, filled circles)
    ax.loglog(step_sizes, fd_errors, color='black', linestyle='-', linewidth=1.2, 
              marker='o', markersize=4, markerfacecolor='black', 
              label='Central Difference Error')

    # 3. Add Trace: AD Error Bound (Dotted line, no markers)
    # Using 1e-16 as the theoretical machine zero floor
    ax.axhline(y=1e-16, color='black', linestyle=':', linewidth=1.5, 
               label='VORJAX AD Error Bound (Machine Zero)')

    # 4. Formatting X and Y axes
    ax.set_xlabel("Finite Difference Step Size ($h$)", fontweight='bold')
    ax.set_ylabel("Absolute Error vs. AD Gradient", fontweight='bold')
    
    # Force X-axis to standard ascending order if needed
    if step_sizes[0] > step_sizes[-1]:
        ax.invert_xaxis()

    # Apply AIAA-style inward ticks and bounding box
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='both')
    ax.grid(True, which="major", linestyle='-', color='#E5E5E5', alpha=0.7)

    # 5. Educational Annotations
    # Set up a clean bounding box style for the text
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7)

    # Round-off region (Left side / Small h)
    ax.text(0.05, 0.4, "Round-off\nError", 
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='left',
            bbox=bbox_props)

    # Truncation region (Right side / Large h)
    ax.text(0.95, 0.4, "Truncation\nError", 
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=bbox_props)

    # 6. Legend
    # Placed at the top center, slightly above the plot box
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, 
              frameon=True, edgecolor='black', fancybox=False)

    # Note: You can comment this title out for the final LaTeX export 
    # to let the \caption{} handle it.
    plt.title("Central Difference Accuracy vs Step Size ", 
              fontweight='bold', pad=35) # Pad to make room for legend
    
    # 7. Export to native PDF vector graphic
    plt.tight_layout()
    plt.savefig("./Tests/VORJAX/plots/fd_v_curve.pdf", format='pdf', bbox_inches='tight')

def plot_theoretical_error_comparison_plotly(step_sizes, fd_grads, exact_grad, grad_truth):
    
    # Calculate Relative Errors against Lifting-Line Theory
    fd_rel_errors = np.abs((np.array(fd_grads) - grad_truth) / grad_truth)
    ad_rel_error = float(np.abs((np.asarray(exact_grad) - grad_truth) / grad_truth))

    fig = go.Figure()

    # 1. Add Trace: FD Relative Error vs Theory
    fig.add_trace(
        go.Scatter(
            x=step_sizes, 
            y=fd_rel_errors, 
            name="Finite Difference Error",
            mode="lines+markers",
            line=dict(color="#d62728", width=3),  # Red
            marker=dict(symbol="circle", size=8),
            hovertemplate='Step Size (h): %{x:.1e}<br>FD Rel Error: %{y:.2e}<extra></extra>'
        )
    )

    # 2. Add Trace: AD Relative Error vs Theory (Flat Line)
    fig.add_trace(
        go.Scatter(
            x=[min(step_sizes), max(step_sizes)],
            y=[ad_rel_error, ad_rel_error],
            name=f"Autodiff Error Floor (Grid Limit)",
            mode="lines",
            line=dict(color="#1f77b4", width=3, dash="dash"), # Blue dashed
            hovertemplate=f'AD Rel Error (No Step Size): {ad_rel_error:.2e}<extra></extra>'
        )
    )

    # 3. Styling and Layout
    fig.update_layout(
        title_text="<b>Gradient Accuracy vs. Lifting-Line Theory</b>",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        margin=dict(l=60, r=60, t=100, b=60)
    )

    # 4. Format X-axis (Log Scale)
    fig.update_xaxes(
        title_text="<b>Finite Difference Step Size (h)</b>", 
        type="log",
        exponentformat="e",
        showgrid=True,
        gridcolor="lightgrey",
        autorange="reversed" if step_sizes[0] > step_sizes[-1] else True 
    )

    # 5. Format Y-axis (Log Scale)
    fig.update_yaxes(
        title_text="<b>Relative Error vs. Theoretical Truth</b>", 
        type="log",
        exponentformat="e",
        showgrid=True,
        gridcolor="lightgrey"
    )

    # 6. Annotations
    fig.add_annotation(
        x=np.log10(step_sizes[len(step_sizes)//2]), # Middle of the plot
        y=np.log10(ad_rel_error),
        text="VLM Grid Discretization Limit",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="#1f77b4", size=12)
    )

    fig.show()

def plot_delta_ar_sweep_plotly(ARs, grad_AD, error_AD, grad_jones, grad_FD=None, error_FD=None):
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # --- Primary Y: Gradients ---
    fig.add_trace(
        go.Scatter(
            x=ARs, y=grad_AD, 
            name="VORJAX AD Gradient",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            marker=dict(symbol="circle", size=8)
        ), secondary_y=False
    )
    
    if grad_FD is not None and len(grad_FD) > 0:
        fig.add_trace(
            go.Scatter(
                x=ARs, y=grad_FD, 
                name="Central Difference (FD)",
                mode="markers",
                marker=dict(color="orange", symbol="x", size=8, line=dict(width=2))
            ), secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=ARs, y=grad_jones, 
            name="Jones Slender Wing Theory",
            mode="lines",
            line=dict(color="black", width=3, dash="dash")
        ), secondary_y=False
    )

    # --- Secondary Y: Errors ---
    error_AD_pct = np.array(error_AD) * 100.0
    fig.add_trace(
        go.Scatter(
            x=ARs, y=error_AD_pct, 
            name="AD Relative Error vs Jones",
            mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dot"),
            marker=dict(symbol="square", size=6)
        ), secondary_y=True
    )

    if error_FD is not None and len(error_FD) > 0:
        error_FD_pct = np.array(error_FD) * 100.0
        fig.add_trace(
            go.Scatter(
                x=ARs, y=error_FD_pct, 
                name="FD Relative Error vs Jones",
                mode="lines+markers",
                line=dict(color="orange", width=2, dash="dot"),
                marker=dict(symbol="x", size=6)
            ), secondary_y=True
        )

    # --- Layout & Styling ---
    fig.update_layout(
        title_text="<b>Delta Wing Aspect Ratio Sweep: VLM vs. Slender Wing Theory</b>",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=1),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    fig.update_xaxes(title_text="<b>Aspect Ratio (AR)</b>", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="<b>Gradient (C<sub>L&alpha;</sub>) [rad<sup>-1</sup>]</b>", 
                     color="#1f77b4", secondary_y=False, showgrid=True, gridcolor="lightgrey", rangemode="tozero")
    fig.update_yaxes(title_text="<b>Relative Error [%]</b>", 
                     color="#d62728", secondary_y=True, showgrid=False, rangemode="tozero")

    return fig

def plot_delta_convergence_and_memory_plotly(n_panels, grad_AD, memory_gb, grad_truth):
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Primary Y: AD Gradient Convergence ---
    fig.add_trace(
        go.Scatter(
            x=n_panels, 
            y=grad_AD, 
            name="VORJAX AD Gradient",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3), # Blue
            marker=dict(size=8),
            hovertemplate='Panels: %{x}<br>Gradient: %{y:.4f}<extra></extra>'
        ), secondary_y=False
    )

    # Primary Y: Truth Line
    fig.add_trace(
        go.Scatter(
            x=[n_panels[0], n_panels[-1]], 
            y=[grad_truth, grad_truth],
            name="Jones Theory Truth",
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            hoverinfo="skip"
        ), secondary_y=False
    )

    # --- Secondary Y: VRAM Memory Usage ---
    fig.add_trace(
        go.Scatter(
            x=n_panels, 
            y=memory_gb, 
            name="Peak VRAM Allocation",
            mode="lines+markers",
            line=dict(color="#2ca02c", width=3), # Green
            marker=dict(symbol="square", size=8),
            hovertemplate='Panels: %{x}<br>VRAM: %{y:.2f} GB<extra></extra>'
        ), secondary_y=True
    )

    # --- Layout & Styling ---
    fig.update_layout(
        title_text="<b>Delta Wing: Gradient Convergence vs. Memory Cost</b>",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.05, y=0.5, 
            bgcolor="rgba(255, 255, 255, 0.9)", 
            bordercolor="black", borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    # Format X-axis (Log Scale for doubling panels)
    fig.update_xaxes(
        title_text="<b>Number of Panels (N)</b>", 
        type="log",
        tickvals=n_panels, # Force ticks only on the values you tested
        ticktext=[str(val) for val in n_panels],
        showgrid=True, 
        gridcolor="lightgrey"
    )
    
    # Format Primary Y-axis (Linear for Gradient)
    fig.update_yaxes(
        title_text="<b>Gradient (C<sub>L&alpha;</sub>) [rad<sup>-1</sup>]</b>", 
        color="#1f77b4", 
        secondary_y=False, 
        showgrid=True, 
        gridcolor="lightgrey", 
    )
    
    # Format Secondary Y-axis (Log Scale for O(N^2) memory)
    fig.update_yaxes(
        title_text="<b>Peak GPU Memory [GB]</b>", 
        color="#2ca02c", 
        type="log",
        secondary_y=True, 
        showgrid=False # Disabled to prevent gridline clashing
    )

    return fig

def plot_transonic_tuning(mach, cl_su2, cl_vorjax, M_sub, M_sup):
    """
    Plots SU2 data against raw VORJAX and a dynamically calculated Hermite spline.
    Includes a hidden trace to display the % Error in the unified hover menu.
    """
    
    # 1. Find VORJAX boundary values
    idx_sub = np.argmin(np.abs(mach - M_sub))
    idx_sup = np.argmin(np.abs(mach - M_sup))
    val_sub = cl_vorjax[idx_sub]
    val_sup = cl_vorjax[idx_sup]
    
    # 2. Boundary Gradients
    grad_sub = val_sub * (M_sub / (1.0 - M_sub**2))
    grad_sup = -val_sup * (M_sup / (M_sup**2 - 1.0))
    delta_M = M_sup - M_sub
    
    # --- Helper to evaluate the spline at any Mach array ---
    def eval_spline(m_array):
        t = (m_array - M_sub) / delta_M
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        return (val_sub * h00) + (val_sup * h01) + (grad_sub * delta_M * h10) + (grad_sup * delta_M * h11)

    # 3. Generate High-Res Spline for the visible line
    mach_spline = np.linspace(M_sub, M_sup, 100)
    cl_spline = ensemble_CL_spline(mach_spline, M_sub, M_sup, val_sub, val_sup)
                
    # 4. Calculate Error exactly at SU2 points
    # (We only care about the error where SU2 overlaps with the spline bounds)
    mask = (mach >= M_sub) & (mach <= M_sup)
    mach_su2_masked = mach[mask]
    cl_su2_masked = cl_su2[mask]
    
    spline_at_su2 = ensemble_CL_spline(mach_su2_masked, M_sub, M_sup, val_sub, val_sup)
    error_pct = ((spline_at_su2 - cl_su2_masked) / cl_su2_masked) * 100.0

    # 5. Build the Plotly Figure
    fig = go.Figure()
    
    # Raw VORJAX 
    fig.add_trace(go.Scatter(
        x=mach, y=cl_vorjax, 
        mode='lines', name='Raw VORJAX',
        line=dict(color='lightgray', width=2, dash='dash')
    ))
    
    # The Transonic Spline
    fig.add_trace(go.Scatter(
        x=mach_spline, y=cl_spline, 
        mode='lines', name=f'Spline ({M_sub} - {M_sup})',
        line=dict(color='blue', width=4)
    ))

    # SU2 Data 
    fig.add_trace(go.Scatter(
        x=mach, y=cl_su2, 
        mode='markers', name='SU2 (Euler)',
        marker=dict(color='red', size=8, symbol='diamond', line=dict(width=1, color='darkred'))
    ))
    
    # --- The Invisible Hover Trace ---
    fig.add_trace(go.Scatter(
        x=mach_su2_masked, y=cl_su2_masked, # Align with SU2 points
        mode='markers', name='Spline Error',
        marker=dict(opacity=0), # Invisible!
        showlegend=False,
        customdata=error_pct,
        hovertemplate='%{customdata:+.2f}%' # Shows like "+5.23%" or "-2.10%"
    ))
    
    # Vertical bounds
    fig.add_vline(x=M_sub, line_dash="dot", line_color="black")
    fig.add_vline(x=M_sup, line_dash="dot", line_color="black")
    
    y_max_su2 = np.max(cl_su2)
    fig.update_layout(
        title="Transonic Spline Tuning vs SU2 Validation",
        xaxis_title="Mach Number",
        yaxis_title="Lift Coefficient (CL)",
        yaxis=dict(range=[0, y_max_su2 * 1.5]), 
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

# Execution ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":


    # geometry_file = '/home/jordan/dev/RCAIDE/Templates/Tests/V_and_V/AVL Test Cases/b737_wings_flat_no_af.avl'

    # avl_b737_data = parse_avl_file(Path(geometry_file))
    # vehicle = convert_to_RCAIDE(avl_b737_data)

    # AVL_basic_test(geometry_file, oper_mode="st")

    alpha_path  = ru.PathTuple(("aerodynamics", "angles", "alpha"))
    mach_path   = ru.PathTuple(("freestream", "mach_number"))
    lift_path   = ru.PathTuple(("aerodynamics", "coefficients", "lift", "total"))
    drag_path   = ru.PathTuple(("aerodynamics", "coefficients", "drag", "total"))
    i_drag_path = ru.PathTuple(("aerodynamics", "coefficients", "drag", "induced", "total"))
    nf_drag_path = ru.PathTuple(("aerodynamics", "coefficients", "drag", "induced", "near_field"))
    ff_drag_path = ru.PathTuple(("aerodynamics", "coefficients", "drag", "induced", "far_field"))

    GRAD_MAP = GradientMap(
        state_inputs=(alpha_path,),
        state_outputs=(lift_path,)
    )

    TEST_AVL        = False
    TEST_ELLIPTICAL = False
    TEST_FD         = False
    TEST_METHOD     = False
    TEST_DELTA      = True
    TEST_ONERA      = False
    
    COSINE_SPC_SW   = True
    PLOT_WINGS      = False
    SHOCK           = False
    
    DEBUG           = False
    NEW_DATA        = True

    # AVL Test Cases ---------------------------------------------------------------------------------------------------
    if TEST_AVL:

        AVL_dCp = AVL_basic_test(run_name="straight_wing", oper_mode="fe", alpha=2.0, span=10.0, chord=1.0)
        parsed_data = AVL_basic_test(run_name="straight_wing", oper_mode="st", alpha=2.0, span=10.0, chord=1.0)

        vehicle = VORJAX_straight_wing(span=10.0, chord=1.0)
        results = VORJAX_test_run(vehicle,
                                  alpha=[2.0 * Units.deg], Mach=[0.00],
                                  n_sw=20, n_cw=12, cos_sw=False,
                                  debug_mode=DEBUG)

        f_st, f_sys, f_setts = results

        CL = ru.get_target(f_st, lift_path).item(0)
        CD = ru.get_target(f_st, ru.PathTuple(("aerodynamics", "coefficients", "drag", "total"))).item(0)
        C_m = ru.get_target(f_st, ru.PathTuple(("aerodynamics", "coefficients", "moments", "pitch"))).item(0)

        data = f_sys.analysis_data
        VORJAX_dCp = np.round(np.asarray(data["pressure_coefficients"]), 5)
        err_max = np.max((AVL_dCp - VORJAX_dCp) / AVL_dCp)

        print("\n--- Extracted VORJAX Results ---")
        print(f"CL: {CL:.5f}")
        print(f"CD: {CD:.5f}")
        print(f"CM: {C_m:.5f}")
        print(f"Max dCp Error: {err_max:.5f}")

        if PLOT_WINGS:
            fig = plot_vlm_panels(data['vortex_distribution'], data['pressure_coefficients'][0])
            fig.show()

    # Elliptical Wing Test ---------------------------------------------------------------------------------------------
    if TEST_ELLIPTICAL:
        
        print("\n--- Elliptical Gradients Test ---")

        if NEW_DATA:
            max_segments = 20
            AR = 8.0
            grad_truth = 2.0 * jnp.pi/(jnp.sqrt(1 + (2 / AR)**2) + 2/AR)
            step_sizes = jnp.logspace(-1, -15, 15)
            
            CL = []
            grad_AD = []
            error_AD = []
            
            grad_FD = []
            error_FD = []

            for n_seg in trange(1, max_segments+1, desc="Running Elliptical Lift Tests"):
                vehicle = VORJAX_elliptical_wing(AR=AR, n_segments=n_seg)

                results = VORJAX_test_run(
                    vehicle,
                    alpha=2.0 * Units.deg , Mach=0.00,
                    n_sw=max_segments,
                    grad_map=GRAD_MAP,
                    debug_mode=DEBUG
                )
                f_st, f_sys, f_setts, jac = results

                CL.append(ru.get_target(f_st, lift_path).item(0))
                grad_AD.append(jac.item(0))
                error_AD.append(abs(jac.item(0) - grad_truth)/grad_truth)

                if PLOT_WINGS:
                    if n_seg == 1 or n_seg % 5 == 0:
                        fig = plot_vlm_panels(VD=f_sys.analysis_data['vortex_distribution'])
                        fig.show()
                
            data = f_sys.analysis_data
            VD = data['vortex_distribution']
            Gamma=data['vortex_strengths']
            
            le_mask_float = VD.is_leading_edge.astype(jnp.float32)
            eta = jax.ops.segment_sum(VD.collocation_points[:, 1] * le_mask_float, VD.strip_ids, num_segments=VD.total_strips) / (AR/2.0)
            gamma=jax.ops.segment_sum(Gamma[0], VD.strip_ids, num_segments=VD.total_strips)

            save_plot_cache(
                "elliptical_convergence",
                n_segments=list(range(1, max_segments+1)),
                grad_AD=grad_AD,
                error=error_AD,
                grad_truth=grad_truth
            )
            
            save_plot_cache(
                "elliptical_gamma",
                eta=eta,
                b=8.,
                CL=0.1712,
                AR=8.,
                v_inf=100.,
                gamma=gamma,
            )
            
            if TEST_FD:
                vehicle = VORJAX_elliptical_wing(AR=AR, n_segments=max_segments)
                for i in trange(len(step_sizes), desc="Running Elliptical FD Tests"):
                    h = step_sizes[i]

                    # Forward Step
                    res_fwd = VORJAX_test_run(
                        vehicle,
                        alpha=(2.0 * Units.deg) + h, Mach=0.00,
                        debug_mode=DEBUG)
                    CL_fwd = ru.get_target(res_fwd[0], lift_path).item(0)
                    
                    # Backward Step
                    res_bwd = VORJAX_test_run(
                        vehicle,
                        alpha=(2.0 * Units.deg) - h,
                        Mach=0.00,
                        debug_mode=DEBUG)
                    CL_bwd = ru.get_target(res_bwd[0], lift_path).item(0)
                    
                    # Central Difference
                    g_FD = (CL_fwd - CL_bwd) / (2 * h)
                    grad_FD.append(g_FD)
                    
                    # Calculate absolute error against JAX
                    error_FD.append(abs(g_FD - grad_AD[-1]))

                    save_plot_cache(
                        "elliptical_FD",
                        step_sizes=step_sizes,
                        fd_errors=error_FD
                    )

            

            
            
            
        
        # plot_elliptical_convergence_plotly(n_segments_list, grad_AD, error_AD, grad_truth)
        # plot_fd_v_curve_plotly(step_sizes, error_FD)
        
        plot_elliptical_convergence_mpl(**load_plot_cache('elliptical_convergence'))
        plot_fd_v_curve_mpl(**load_plot_cache('elliptical_FD'))
        plot_spanwise_loading_mpl(**load_plot_cache('elliptical_gamma'))

    # Elliptical Methodological Test -----------------------------------------------------------------------------------
    if TEST_METHOD:
        
        print("\n--- Elliptical Drag Methodology Test ---")

        if NEW_DATA:
            max_segments = 20
            AR = 8.0
            alpha = 2.0 * Units.deg
            grad_truth = (2.0  * alpha / (jnp.pi * AR)) * (2.0 * jnp.pi/(jnp.sqrt(1 + (2 / AR)**2) + 2/AR))**2
            
            CDnf = []
            grad_nf = []

            CDff = []
            grad_ff = []
        
            for n_seg in trange(1, max_segments+1, desc="Running Elliptical Test Cases"):
                vehicle = VORJAX_elliptical_wing(AR=AR, n_segments=max_segments)

                results = VORJAX_test_run(
                    vehicle,
                    alpha=alpha , Mach=0.00,
                    n_sw=max_segments,
                    grad_map=GradientMap(
                        state_inputs=(alpha_path,),
                        state_outputs=(nf_drag_path, ff_drag_path)
                    ),
                    debug_mode=DEBUG
                )
                
                f_st, f_sys, f_setts, jac = results

                CDnf.append(ru.get_target(f_st, nf_drag_path).item(0))
                grad_nf.append(jac.item(0))
                
                CDff.append(ru.get_target(f_st, ff_drag_path).item(0))
                grad_ff.append(jac.item(1))
            
            save_plot_cache(
                "elliptical_drag_nf",
                n_segments=list(range(1, max_segments+1)),
                grad_AD=grad_nf,
                field="near",
            )

            save_plot_cache(
                "elliptical_drag_ff",
                n_segments=list(range(1, max_segments+1)),
                grad_AD=grad_ff,
                field="far"
            )
        
        plot_elliptical_drag_mpl(**load_plot_cache('elliptical_drag_nf'))
        plot_elliptical_drag_mpl(**load_plot_cache('elliptical_drag_ff'))

    # Delta Wing Test --------------------------------------------------------------------------------------------------
    if TEST_DELTA:
        CL = []
        grad_AD = []
        error_AD = []

        vram_gb = []

        ARs = jnp.linspace(0.1, 2.5, 25)
        grad_jones = jnp.pi * ARs / 2.0

        n_cws   = [4, 8, 8, 16, 16, 32, 32, 64, 64]
        n_sws   = [4, 4, 8, 8,  16, 16, 32, 32, 64]

        vehicle = VORJAX_delta_wing(AR=ARs[0])
        for i in trange(len(n_sws), desc="Running Delta Wing Panelization Tests"):
            n_sw = n_sws[i]
            n_cw = n_cws[i]

            # Forward Step
            results = VORJAX_test_run(
                vehicle,
                alpha=2.0 * Units.deg,
                Mach=0.00,
                n_sw=n_sw,
                n_cw=n_cw,
                grad_map=GRAD_MAP,
                debug_mode=DEBUG
            )
            
            f_st, f_sys, f_setts, jac = results

            jac.block_until_ready()
            stats = jax.devices()[0].memory_stats()
            peak_vram = stats.get('peak_bytes_in_use', 0) / (1024 ** 3)
            vram_gb.append(peak_vram)
            
            CL.append(ru.get_target(f_st, lift_path).item(0))
            grad_AD.append(jac.item(0))
            error_AD.append(abs(jac.item(0) - grad_jones[0])/grad_jones[0])

        
        fig = plot_delta_convergence_and_memory_plotly([s * c for s, c in zip(n_sws, n_cws)], grad_AD, vram_gb, grad_jones[0])
        fig.show()

        CL = []
        grad_AD = []
        error_AD = []

        for i in trange(len(ARs), desc="Running Delta Wing AR Sweep"):
            vehicle = VORJAX_delta_wing(AR=ARs[i])
            grad_truth = grad_jones[i]

            results = VORJAX_test_run(
                vehicle,
                alpha=2.0 * Units.deg,
                Mach=0.00,
                grad_map=GRAD_MAP,
                debug_mode=DEBUG
            )
            
            f_st, f_sys, f_setts, jac = results

            CL.append(ru.get_target(f_st, lift_path).item(0))
            grad_AD.append(jac.item(0))
            error_AD.append(abs(jac.item(0) - grad_truth)/grad_truth)

            if PLOT_WINGS:
                if int(i) % 10 == 0:
                    fig = plot_vlm_panels(f_sys.analysis_data['vortex_distribution'])
                    fig.show()
        
        fig = plot_delta_ar_sweep_plotly(ARs, grad_AD, error_AD, grad_jones)
        fig.show()
        
    # ONERA M6 Mach Sweep ----------------------------------------------------------------------------------------------
    if TEST_ONERA:

        Mach = [
            0.3, 0.4, 0.5, 0.6, 0.7,  # Subsonic
            0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, # Transonic
            1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0 # Supersonic
            ]
        alpha = [3.06 * Units.deg] * len(Mach)

        vehicle = VORJAX_ONERA_M6()

        results = VORJAX_test_run(
            vehicle,
            alpha, Mach,
            n_sw=24, n_cw=12,
            cos_sw=COSINE_SPC_SW,
            shock=SHOCK,
            grad_map=GRAD_MAP,
            debug_mode=DEBUG
        )
        f_st, f_sys, f_setts, jac = results

        CL  = ru.get_target(f_st, lift_path)
        CDi = ru.get_target(f_st, i_drag_path)
        dCL_dMach = jnp.array([jac[i, 0, i] for i in range(len(alpha))]).reshape(CL.shape)

        cache_path = Path(os.path.join(os.path.dirname(__file__), 'VORJAX/SU2_Test_Cases/su2_run_cache.json'))
        with open(cache_path) as f:
            su2_cache = json.load(f)

        SU2_CLs = jnp.array([v['cl'] for v in su2_cache.values()])

        print(f"\nONERA M6 Mach Sweep - SHOCK: {SHOCK}\n"+"-"*70)
        for i in range(len(Mach)):
            SU2_results = list(su2_cache.values())[i]
            SU2_CL = float(SU2_results['cl'])
            SU2_da = float(SU2_results['dcl_dalpha'])

            VJX_CL = float(CL[i, 0])
            VJX_da = float(dCL_dMach[i, 0]) * Units.deg

            CL_err = (VJX_CL - SU2_CL)/SU2_CL * 100
            da_err = (VJX_da - SU2_da)/SU2_da * 100

            VJX_CDi = float(CDi[i, 0])

            print(f"M: {Mach[i]:.2f}, CL: {VJX_CL:.3e}({CL_err:>5.1f}%), dAlpha: {VJX_da:.3e}({da_err:>5.1f}%), CDi: {VJX_CDi:>7.4f}")

        if PLOT_WINGS:
            data = f_sys.analysis_data
            base_panels = plot_vlm_panels(data["vortex_distribution"], title="ONERA M6 Panelization")
            base_panels.show()

            m11_flags = plot_vlm_panels(data["vortex_distribution"], data['singularities'][12], title="ONERA M6 Flag, M = 1.1")
            m11_flags.show()

            m11_dcp = plot_vlm_panels(data["vortex_distribution"], data['dCp'][12], title="ONERA M6 DCp, M = 1.1")
            m11_dcp.show()

            m20 = plot_vlm_panels(data["vortex_distribution"], data['dCp'][-1], title="ONERA M6 DCp, M = 2.0")
            m20.show()

    print("\nAll tests complete.")