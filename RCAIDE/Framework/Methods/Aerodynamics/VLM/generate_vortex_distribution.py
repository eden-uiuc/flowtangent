# RCAIDE/Framework/Methods/Aerodynamics/VLM/generate_panel_coordinates.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System, Aircraft
    from RCAIDE.Framework.Settings import Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Data Structures
# ----------------------------------------------------------------------------------------------------------------------

class VortexDistribution(eqx.Module):
    """ 
    Differentiable float arrays containing the physical 3D mesh for the VLM.
    All arrays represent flattened properties of the panels.
    """
    # --- Horseshoe Vortex Geometry (Shape: total_panels x 3) ---
    bound_vortex_left: jnp.ndarray   # (x, y, z) of the left node (1/4 chord)
    bound_vortex_right: jnp.ndarray  # (x, y, z) of the right node (1/4 chord)
    collocation_points: jnp.ndarray  # (x, y, z) where flow tangency is enforced (3/4 chord)
    normal_vectors: jnp.ndarray      # (nx, ny, nz) unit vector perpendicular to the panel
    
    # --- Panel Properties (Shape: total_panels) ---
    chord_lengths: jnp.ndarray       # Local chord length of the panel
    panel_areas: jnp.ndarray         # Area of the panel
    tangent_incidence_angle: jnp.ndarray # Local twist/camber/deflection at the panel

# ---------------------------------------------------------
# Helper Function
# ---------------------------------------------------------
@jax.jit
def generate_wing_panel_coordinates(vlm_wings: tuple, vlm_vortex_settings):
    
    all_left_nodes = []
    all_right_nodes = []
    all_collocations = []
    all_normals = []
    all_chords = []
    all_areas = []
    all_incidences = []

    for record in vlm_wings:
        wing = record.wing
        
        n_sw = vlm_vortex_settings.wing_spanwise_vortices
        n_cw = vlm_vortex_settings.wing_chordwise_vortices
        
        # Determine spanwise spacing (Uniform or Cosine)
        span = jnp.where(wing.symmetric, wing.spans.projected / 2.0, wing.spans.projected)
        
        seg_etas = jnp.array([seg.percent_span_location for seg in wing.segments])
        seg_y = seg_etas * span
        
        seg_chords = jnp.array([seg.chord for seg in wing.segments])

        seg_camber_xs = jnp.array([seg.airfoil.x_lower_surface for seg in wing.segments])
        seg_camber_zs = jnp.array([seg.airfoil.camber_coordinates for seg in wing.segments])

        seg_twists = jnp.array([seg.twist for seg in wing.segments])
        seg_x_off = jnp.array([seg.x_offset for seg in wing.segments])
        seg_z_off = jnp.array([seg.dih_offset for seg in wing.segments])


        if vlm_vortex_settings.spanwise_cosine_spacing:
            # Cosine spacing: clusters panels near the tips and roots
            thetan = jnp.linspace(jnp.pi/2, 0, n_sw + 1)
            y_coords = span * jnp.cos(thetan)
        else:
            y_coords = jnp.linspace(0, span, n_sw + 1)
        
        req_etas = jnp.append(record.strip_eta_starts, record.strip_eta_ends[-1])
        req_y = req_etas * span

        le_cuts = record.strip_le_cuts
        te_cuts = record.strip_te_cuts

        # The base uniform chordwise fractions (0.0 to 1.0)
        base_x_fractions = jnp.linspace(0.0, 1.0, n_cw + 1)
        
        # Replicate the legacy "shifted_idxs" logic in pure JAX
        shifted_idxs = jnp.zeros(n_sw + 1)
        
        for i in range(len(req_y)):
            y_r = req_y[i]
            # Find closest available y_coord
            diffs = jnp.abs(y_coords - y_r) + shifted_idxs
            idx = jnp.argmin(diffs)
            
            # Snap it and mark it as used (inf)
            y_coords = y_coords.at[idx].set(y_r)
            shifted_idxs = shifted_idxs.at[idx].set(jnp.inf)
            
        # Ensure they are sorted left-to-right
        y_coords = jnp.sort(y_coords)

        # Base chordwise fractions (Uniform)
        # Note: If control surfaces cut into the wing, these fractions are scaled in the original code.
        # For this base implementation, we assume uniform spacing from LE to TE.
        x_fractions = jnp.linspace(0.0, 1.0, n_cw + 1)
        
        # To avoid the inner loop over n_sw, we vmap the strip generator!
        # This is a huge performance boost over the original code.
        def generate_strip(y_a, y_b):
            """ Generates (n_cw) panels for a single spanwise strip. """
            # 1. Interpolate Root/Tip Chords for this strip
            # Since wing geometry is tracked per segment, we need to find which segment we are in.
            # For simplicity in this functional outline, assume a single taper.
            # In your full code, you will interpolate `y_a` along the wing.segments bounds!
            
            # --- Placeholder for segment interpolation ---
            chord_a = jnp.interp(y_a, seg_y, seg_chords)
            chord_b = jnp.interp(y_b, seg_y, seg_chords)
            
            twist_a = jnp.interp(y_a, seg_y, seg_twists)
            twist_b = jnp.interp(y_b, seg_y, seg_twists)
            
            x_offset_a = jnp.interp(y_a, seg_y, seg_x_off)
            x_offset_b = jnp.interp(y_b, seg_y, seg_x_off)
            
            z_offset_a = jnp.interp(y_a, seg_y, seg_z_off)
            z_offset_b = jnp.interp(y_b, seg_y, seg_z_off)

            # Camber Calculation

            camber_x_a = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_a, seg_y, seg_camber_xs)
            camber_z_a = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_a, seg_y, seg_camber_zs)
            
            camber_x_b = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_b, seg_y, seg_camber_xs)
            camber_z_b = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_b, seg_y, seg_camber_zs)
            
            # 2. Look up the cuts for this specific panel strip!
            y_mid = (y_a + y_b) / 2.0
            
            # searchsorted finds which topological interval y_mid belongs to
            strip_idx = jnp.searchsorted(req_y, y_mid) - 1
            strip_idx = jnp.clip(strip_idx, 0, len(le_cuts) - 1)
            
            le_cut = le_cuts[strip_idx]
            te_cut = te_cuts[strip_idx]

            # 3. Rescale the chordwise fractions
            # If le_cut is 0.0 and te_cut is 0.8, x_fractions will now go from 0.0 to 0.8
            local_x_fractions = le_cut + base_x_fractions * (te_cut - le_cut)

            # 4. Apply the scaled fractions to the X coordinates
            # (Note: local_x_fractions replaces x_fractions from the earlier code)
            panel_x_a = x_offset_a + local_x_fractions * chord_a
            panel_x_b = x_offset_b + local_x_fractions * chord_b
            
            # 5. Scale the panel delta_x for the vortex placement
            # The panels are physically smaller now, so the 1/4 and 3/4 points shift!
            dx_a = chord_a * (te_cut - le_cut) / n_cw
            dx_b = chord_b * (te_cut - le_cut) / n_cw
            
            bound_vortex_x_a = panel_x_a[:-1] + 0.25 * dx_a
            bound_vortex_x_b = panel_x_b[:-1] + 0.25 * dx_b
            
            colloc_x_a = panel_x_a[:-1] + 0.75 * dx_a
            colloc_x_b = panel_x_b[:-1] + 0.75 * dx_b
            colloc_x_center = (colloc_x_a + colloc_x_b) / 2.0
            
            # Y locations
            bound_vortex_y_a = jnp.full(n_cw, y_a)
            bound_vortex_y_b = jnp.full(n_cw, y_b)
            colloc_y_center = jnp.full(n_cw, (y_a + y_b) / 2.0)
            
            # Z locations (Flat)
            bound_vortex_z_a = jnp.full(n_cw, z_offset_a)
            bound_vortex_z_b = jnp.full(n_cw, z_offset_b)
            colloc_z_center = jnp.full(n_cw, (z_offset_a + z_offset_b) / 2.0)

            # 6. Control Surface Rescaling (Evaluated at JAX Trace time!)
            if record.is_a_control_surface:
                cf = record.chord_fraction
                if not record.is_slat:
                    # Shift the window to the trailing edge
                    camber_x_a = camber_x_a - (1.0 - cf)
                    camber_x_b = camber_x_b - (1.0 - cf)
                
                # Rescale the nondimensional coordinates to the local CS chord
                camber_x_a = camber_x_a / cf
                camber_x_b = camber_x_b / cf
                camber_z_a = camber_z_a / cf
                camber_z_b = camber_z_b / cf

            # 7. Calculate Local Fractional Panel Locations (x/c)
            dx_frac = 1.0 / n_cw
            
            # The 1/4 chord line of each local panel
            frac_ah = local_x_fractions[:-1] + 0.25 * dx_frac
            # The 3/4 chord line of each local panel
            frac_ac = local_x_fractions[:-1] + 0.75 * dx_frac
            
            # 8. Interpolate the Z camber offsets from the airfoil surface
            z_c_ah_a = jnp.interp(frac_ah, camber_x_a, camber_z_a) * chord_a
            z_c_ah_b = jnp.interp(frac_ah, camber_x_b, camber_z_b) * chord_b
            
            z_c_ac_a = jnp.interp(frac_ac, camber_x_a, camber_z_a) * chord_a
            z_c_ac_b = jnp.interp(frac_ac, camber_x_b, camber_z_b) * chord_b
            
            z_c_ac_center = (z_c_ac_a + z_c_ac_b) / 2.0

            # 9. Add Camber to the Flat Z Coordinates!
            # (We apply this BEFORE twist rotation, exactly as the legacy code did)
            bound_vortex_z_a = bound_vortex_z_a + z_c_ah_a
            bound_vortex_z_b = bound_vortex_z_b + z_c_ah_b
            colloc_z_center = colloc_z_center + z_c_ac_center
            
            # 10. Apply Twist Rotations (Simplified Matrix)
            # Twist rotates the X and Z coordinates around the leading edge (x_offset, z_offset)
            def apply_twist(x, z, pivot_x, pivot_z, twist_angle):
                cos_t = jnp.cos(twist_angle)
                sin_t = jnp.sin(twist_angle)
                new_x = pivot_x + cos_t * (x - pivot_x) + sin_t * (z - pivot_z)
                new_z = pivot_z - sin_t * (x - pivot_x) + cos_t * (z - pivot_z)
                return new_x, new_z
                
            bv_x_a_rot, bv_z_a_rot = apply_twist(bound_vortex_x_a, bound_vortex_z_a, x_offset_a, z_offset_a, twist_a)
            bv_x_b_rot, bv_z_b_rot = apply_twist(bound_vortex_x_b, bound_vortex_z_b, x_offset_b, z_offset_b, twist_b)
            
            col_twist = (twist_a + twist_b) / 2.0
            col_x_rot, col_z_rot = apply_twist(colloc_x_center, colloc_z_center, (x_offset_a+x_offset_b)/2.0, (z_offset_a+z_offset_b)/2.0, col_twist)

            # 11. Pack into (N, 3) arrays
            left_nodes = jnp.stack([bv_x_a_rot, bound_vortex_y_a, bv_z_a_rot], axis=1)
            right_nodes = jnp.stack([bv_x_b_rot, bound_vortex_y_b, bv_z_b_rot], axis=1)
            collocations = jnp.stack([col_x_rot, colloc_y_center, col_z_rot], axis=1)
            
            # Normal vectors (Simplified flat panel assumption for now)
            normals = jnp.stack([jnp.sin(col_twist), jnp.zeros(n_cw), jnp.cos(col_twist)], axis=1)
            
            # Geometric properties
            chords = jnp.full(n_cw, (chord_a + chord_b) / 2.0)
            areas = chords * (y_b - y_a)
            incidences = jnp.full(n_cw, col_twist)
            
            return left_nodes, right_nodes, collocations, normals, chords, areas, incidences

        # We map the strip generator over the left and right edges of every spanwise station!
        y_lefts = y_coords[:-1]
        y_rights = y_coords[1:]
        
        # JAX vmap is perfectly safe here because n_cw is static!
        # The output of vmap is shape (n_sw, n_cw, 3). We use jnp.reshape to flatten it to (total_panels, 3).
        strip_results = jax.vmap(generate_strip)(y_lefts, y_rights)
        
        all_left_nodes.append(jnp.reshape(strip_results[0], (-1, 3)) + wing.origin)
        all_right_nodes.append(jnp.reshape(strip_results[1], (-1, 3)) + wing.origin)
        all_collocations.append(jnp.reshape(strip_results[2], (-1, 3)) + wing.origin)
        all_normals.append(jnp.reshape(strip_results[3], (-1, 3)))
        all_chords.append(jnp.reshape(strip_results[4], (-1,)))
        all_areas.append(jnp.reshape(strip_results[5], (-1,)))
        all_incidences.append(jnp.reshape(strip_results[6], (-1,)))
        
        # Symmetry Calculation

        # 1. Grab the generated arrays for the right side of the aircraft
        strip_L_nodes = jnp.reshape(strip_results[0], (-1, 3)) + wing.origin
        strip_R_nodes = jnp.reshape(strip_results[1], (-1, 3)) + wing.origin
        strip_C_nodes = jnp.reshape(strip_results[2], (-1, 3)) + wing.origin
        strip_normals = jnp.reshape(strip_results[3], (-1, 3))
        strip_chords  = jnp.reshape(strip_results[4], (-1,))
        strip_areas   = jnp.reshape(strip_results[5], (-1,))
        strip_inc     = jnp.reshape(strip_results[6], (-1,))

        # 2. Append the original (Right side) geometry
        all_left_nodes.append(strip_L_nodes)
        all_right_nodes.append(strip_R_nodes)
        all_collocations.append(strip_C_nodes)
        all_normals.append(strip_normals)
        all_chords.append(strip_chords)
        all_areas.append(strip_areas)
        all_incidences.append(strip_inc)

        # 3. If symmetric, mirror and append!
        if wing.symmetric:
            # Mirror Y coordinates by multiplying by -1
            mirrored_L_nodes = strip_L_nodes.at[:, 1].multiply(-1.0)
            mirrored_R_nodes = strip_R_nodes.at[:, 1].multiply(-1.0)
            mirrored_C_nodes = strip_C_nodes.at[:, 1].multiply(-1.0)
            
            # Normals flip in the Y direction
            mirrored_normals = strip_normals.at[:, 1].multiply(-1.0)
            
            # CRITICAL: To maintain the Right-Hand Rule for vortex circulation, 
            # the geometric "Left" and "Right" nodes must swap!
            all_left_nodes.append(mirrored_R_nodes)
            all_right_nodes.append(mirrored_L_nodes)
            
            all_collocations.append(mirrored_C_nodes)
            all_normals.append(mirrored_normals)
            all_chords.append(strip_chords)
            all_areas.append(strip_areas)
            all_incidences.append(strip_inc)

    # Concatenate all wings into the final VortexDistribution
    return VortexDistribution(
        bound_vortex_left=jnp.concatenate(all_left_nodes, axis=0),
        bound_vortex_right=jnp.concatenate(all_right_nodes, axis=0),
        collocation_points=jnp.concatenate(all_collocations, axis=0),
        normal_vectors=jnp.concatenate(all_normals, axis=0),
        chord_lengths=jnp.concatenate(all_chords, axis=0),
        panel_areas=jnp.concatenate(all_areas, axis=0),
        tangent_incidence_angle=jnp.concatenate(all_incidences, axis=0)
    )

# ---------------------------------------------------------
# Stateful Version
# ---------------------------------------------------------

def generate_wing_vortex_distribution(state: "State", system: "Aircraft", settings: "Settings"):

    vlm_wings = system.analysis_data["vlm_wings"]
    vlm_vortex_settings = settings.analysis.aerodynamics.vortices

    VD = generate_wing_panel_coordinates(vlm_wings, vlm_vortex_settings)

    current_system = eqx.tree_at(lambda s: s.analysis_data["vortex_distribution"], system, VD)

    return state, current_system, settings