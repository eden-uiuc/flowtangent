# RCAIDE/Framework/Interfaces/AVL.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Apr 2026, RCAIDE Team
# Modified:

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import jax.numpy as jnp
import equinox as eqx

from RCAIDE.Library.Components import ComponentAreas
from RCAIDE.Library.Components.Wings import Wing, WingChords, WingDimensions, WingSegment, WingSweeps
from RCAIDE.Framework.System import Aircraft

# ----------------------------------------------------------------------------------------------------------------------
# AVL Interface Functions
# ----------------------------------------------------------------------------------------------------------------------

def parse_avl_file(filepath: str) -> dict:
    """
    Parses an AVL geometry file into a structured Python dictionary.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1. Clean the file: strip whitespace and comments (# or !)
    cleaned_lines = []
    for line in lines:
        line = line.split('#')[0].split('!')[0].strip()
        if line:  # Keep only non-empty lines
            cleaned_lines.append(line)

    if not cleaned_lines:
        raise ValueError("AVL file is empty or only contains comments.")

    # 2. Extract Global Header Data
    # Format:
    # Line 0: Name
    # Line 1: Mach
    # Line 2: IYsym IZsym Zsym
    # Line 3: Sref Cref Bref
    # Line 4: Xref Yref Zref
    avl_data = {
        "name": cleaned_lines[0],
        "mach": float(cleaned_lines[1].split()[0]),
        "symmetry": [float(x) for x in cleaned_lines[2].split()[:3]],
        "reference_area": [float(x) for x in cleaned_lines[3].split()[:3]],
        "reference_point": [float(x) for x in cleaned_lines[4].split()[:3]],
        "surfaces": []
    }

    # 3. State Machine for Surfaces and Sections
    current_surface = None

    # Iterate through the rest of the lines
    i = 5
    while i < len(cleaned_lines):
        token = cleaned_lines[i].upper()

        if token == "SURFACE":
            # Start a new surface block
            i += 1
            current_surface = {
                "name": cleaned_lines[i],
                "paneling": [],
                "y_duplicate": None,
                "sections": []
            }
            # The line after the name is Nchordwise Cspace Nspanwise Sspace
            i += 1
            current_surface["paneling"] = [float(x) for x in cleaned_lines[i].split()]
            avl_data["surfaces"].append(current_surface)

        elif token == "YDUPLICATE":
            i += 1
            current_surface["y_duplicate"] = float(cleaned_lines[i])


        elif token == "SECTION":

            i += 1

            section_data = [float(x) for x in cleaned_lines[i].split()]

            current_surface["sections"].append({
                "x_le": section_data[0],
                "y_le": section_data[1],
                "z_le": section_data[2],
                "chord": section_data[3],
                "twist": section_data[4],
                "airfoil_naca": None,
                "airfoil_file": None
            })

        elif token == "NACA":
            i += 1
            # Attach to the most recently created section
            current_surface["sections"][-1]["airfoil_naca"] = cleaned_lines[i].strip()

        elif token == "AFILE":
            i += 1
            current_surface["sections"][-1]["airfoil_file"] = cleaned_lines[i].strip()

        i += 1

    return avl_data


def convert_to_RCAIDE(avl_data: dict) -> Aircraft:
    """
    Converts a parsed AVL data dictionary into an RCAIDE Aircraft system,
    translating Cartesian coordinates into parametric fractions.
    """
    sref, cref, bref = avl_data["reference_area"]
    xref, yref, zref = avl_data["reference_point"]

    global_areas = ComponentAreas(reference=sref)
    vehicle = Aircraft(tag=avl_data["name"], areas=global_areas)
    vehicle = eqx.tree_at(lambda v: v.mass_properties.center_of_gravity, vehicle, jnp.array([[xref, yref, zref]]))

    for surf_data in avl_data["surfaces"]:
        sections = surf_data["sections"]
        if len(sections) < 2:
            continue

        is_symmetric = surf_data.get("y_duplicate") is not None

        # 1. Global Wing References
        root_sec = sections[0]
        tip_sec = sections[-1]

        root_chord = root_sec["chord"]
        tip_chord = tip_sec["chord"]
        taper = tip_chord / root_chord if root_chord > 0 else 0.0

        # Check if vertical tail (dy is ~0)
        dy_total = tip_sec["y_le"] - root_sec["y_le"]
        is_vertical = jnp.isclose(dy_total, 0.0).item()

        semispan = tip_sec["z_le"] - root_sec["z_le"] if is_vertical else dy_total
        semispan = jnp.maximum(semispan, 1e-8)  # Prevent div by zero
        total_span = semispan * 2.0 if is_symmetric and not is_vertical else semispan

        # 2. Build the Parametric Segments
        segments_list = []
        for i in range(len(sections) - 1):
            sec_in = sections[i]
            sec_out = sections[i + 1]

            # Parametric Span and Chord Locations
            ds_in = sec_in["z_le"] - root_sec["z_le"] if is_vertical else sec_in["y_le"] - root_sec["y_le"]
            span_fraction = ds_in / semispan
            chord_fraction = sec_in["chord"] / root_chord

            # Geometric Deltas for this specific panel
            dy = sec_out["y_le"] - sec_in["y_le"]
            dz = sec_out["z_le"] - sec_in["z_le"]

            # Quarter Chord Sweep Math
            # X_qc = X_le + 0.25 * Chord
            x_qc_in = sec_in["x_le"] + 0.25 * sec_in["chord"]
            x_qc_out = sec_out["x_le"] + 0.25 * sec_out["chord"]
            dx_qc = x_qc_out - x_qc_in

            qc_sweep = 0.0 if is_vertical else jnp.arctan2(dx_qc, dy).item()
            dihedral = jnp.pi / 2.0 if is_vertical else jnp.arctan2(dz, dy)

            #TODO: Airfoil parsing
            segment = WingSegment(
                tag=f"{surf_data['name']}_Seg_{i + 1}",
                percent_span_location=span_fraction,
                root_chord_percent=chord_fraction,
                twist=sec_in["twist"],  # AVL twist is in degrees, verify if RCAIDE expects radians here!
                dihedral_outboard=dihedral,
                sweeps=WingSweeps(quarter_chord=qc_sweep)
            )
            segments_list.append(segment)

        # 3. Instantiate the Wing Component
        wing = Wing(
            tag=surf_data["name"],
            symmetric=is_symmetric,
            vertical=is_vertical,
            taper=taper,
            segments=tuple(segments_list),  # Pass the pre-built segments here
            spans=WingDimensions(projected=total_span),
            chords=WingChords(root=root_chord, tip=tip_chord, mean_aerodynamic=cref),
            origin=jnp.array([[root_sec["x_le"], root_sec["y_le"], root_sec["z_le"]]]),
            aerodynamic_center=jnp.array([[xref, yref, zref]])
        )

        # 4. Trigger RCAIDE's internal geometry engine to fill in the rest
        wing = wing.update_geometry(calculate_reference_area=True, calculate_wetted_area=True)

        vehicle = vehicle.add_subcomponent(wing)

    return vehicle


def read_and_convert(file_path: str) -> Aircraft:
    avl_data = parse_avl_file(file_path)
    return convert_to_RCAIDE(avl_data)

