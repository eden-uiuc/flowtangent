# RCAIDE/Framework/Methods/Aerodynamics/VLM/generate_panel_coordinates.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
import dataclasses

# pacakge imports
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

    # --- Physical Panel Corners (total_panels x 3) ---
    panel_corner_a1: jnp.ndarray  # Top Left
    panel_corner_a2: jnp.ndarray  # Bottom Left
    panel_corner_b1: jnp.ndarray  # Top Right
    panel_corner_b2: jnp.ndarray  # Bottom Right
    
    # --- Solver Metadata ---
    is_leading_edge: jnp.ndarray
    is_trailing_edge: jnp.ndarray
    surface_id: jnp.ndarray
    panels_per_strip: jnp.ndarray
    total_strips: int = eqx.field(static=True, default=0)

# ---------------------------------------------------------
# Helper Functions
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
    
    all_a1 = []
    all_a2 = []
    all_b1 = []
    all_b2 = []
    
    all_le = []
    all_te = []

    all_surface_ids = []
    all_panels_per_strip = []

    total_strips = 0

    for id_index, record in enumerate(vlm_wings):
        
        current_id = id_index + 1
        
        wing = record.wing
        
        n_sw = record.n_sw
        n_cw = record.n_cw
        n_af_pts = record.n_af_pts
        
        total_strips += n_sw

        # Determine spanwise spacing (Uniform or Cosine)
        span = jnp.where(wing.symmetric, wing.spans.projected / 2.0, wing.spans.projected)
        
        seg_etas = jnp.array([seg.percent_span_location for seg in wing.segments])
        seg_y = seg_etas * span
        
        seg_chords = jnp.array([seg.chords.root for seg in wing.segments])
        seg_twists = jnp.array([seg.twist for seg in wing.segments])

        camber_xs = []
        camber_zs = []

        for seg in wing.segments:
            if hasattr(seg, 'airfoil') and seg.airfoil is not None:
                camber_xs.append(seg.airfoil.x_lower_surface)
                camber_zs.append(seg.airfoil.camber)
            else:
                camber_xs.append(jnp.linspace(0., 1., n_af_pts))
                camber_zs.append(jnp.zeros((n_af_pts)))

        seg_camber_xs = jnp.array(camber_xs)
        seg_camber_zs = jnp.array(camber_zs)

        seg_x_off = record.segment_x_offsets
        seg_z_off = record.segment_z_offsets

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
        
        # To avoid the inner loop over n_sw, we vmap the strip generator
        def generate_strip(y_a, y_b):
            """ Generates (n_cw) panels for a single spanwise strip. """
            # 1. Interpolate Root/Tip Chords for this strip
            chord_a = jnp.interp(y_a, seg_y, seg_chords)
            chord_b = jnp.interp(y_b, seg_y, seg_chords)
            
            twist_a = jnp.interp(y_a, seg_y, seg_twists)
            twist_b = jnp.interp(y_b, seg_y, seg_twists)
            
            x_offset_a = jnp.interp(y_a, seg_y, seg_x_off)
            x_offset_b = jnp.interp(y_b, seg_y, seg_x_off)
            
            z_offset_a = jnp.interp(y_a, seg_y, seg_z_off)
            z_offset_b = jnp.interp(y_b, seg_y, seg_z_off)

            # Camber Calculation ---------------------------------------------------------------------------------------

            camber_x_a = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_a, seg_y, seg_camber_xs)
            camber_z_a = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_a, seg_y, seg_camber_zs)
            
            camber_x_b = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_b, seg_y, seg_camber_xs)
            camber_z_b = jax.vmap(jnp.interp, in_axes=(None, None, 1))(y_b, seg_y, seg_camber_zs)
            
            # Look up the cuts for this specific panel strip
            y_mid = (y_a + y_b) / 2.0
            
            # searchsorted finds which topological interval y_mid belongs to
            strip_idx = jnp.searchsorted(req_y, y_mid) - 1
            strip_idx = jnp.clip(strip_idx, 0, len(le_cuts) - 1)
            
            le_cut = le_cuts[strip_idx]
            te_cut = te_cuts[strip_idx]

            # Rescale the chordwise fractions
            # If le_cut is 0.0 and te_cut is 0.8, x_fractions will now go from 0.0 to 0.8
            local_x_fractions = le_cut + base_x_fractions * (te_cut - le_cut)

            # Apply the scaled fractions to the X coordinates
            # (Note: local_x_fractions replaces x_fractions from the earlier code)
            panel_x_a = x_offset_a + local_x_fractions * chord_a
            panel_x_b = x_offset_b + local_x_fractions * chord_b
            
            # Scale the panel delta_x for the vortex placement
            # The panels are physically smaller now, so the 1/4 and 3/4 points shift
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

            # Control Surface Rescaling (Evaluated at JAX Trace time)
            if record.is_a_control_surface:
                cf = record.cs_meta.chord_fraction
                if not record.is_slat:
                    # Shift the window to the trailing edge
                    camber_x_a = camber_x_a - (1.0 - cf)
                    camber_x_b = camber_x_b - (1.0 - cf)
                
                # Rescale the nondimensional coordinates to the local CS chord
                camber_x_a = camber_x_a / cf
                camber_x_b = camber_x_b / cf
                camber_z_a = camber_z_a / cf
                camber_z_b = camber_z_b / cf

            # Calculate Local Fractional Panel Locations (x/c)
            dx_frac = (te_cut - le_cut) / n_cw
            
            # The 1/4 chord line of each local panel
            frac_ah = local_x_fractions[:-1] + 0.25 * dx_frac
            # The 3/4 chord line of each local panel
            frac_ac = local_x_fractions[:-1] + 0.75 * dx_frac
            # Left edge of each local panel
            frac_a1 = local_x_fractions[:-1]
            # Right edge of each local panel
            frac_a2 = local_x_fractions[1:]
            
            # Interpolate the Z camber offsets from the airfoil surface
            z_c_ah_a = jnp.interp(frac_ah, camber_x_a, camber_z_a) * chord_a
            z_c_ah_b = jnp.interp(frac_ah, camber_x_b, camber_z_b) * chord_b
            
            z_c_ac_a = jnp.interp(frac_ac, camber_x_a, camber_z_a) * chord_a
            z_c_ac_b = jnp.interp(frac_ac, camber_x_b, camber_z_b) * chord_b

            z_c_a1_a = jnp.interp(frac_a1, camber_x_a, camber_z_a) * chord_a
            z_c_a2_a = jnp.interp(frac_a2, camber_x_a, camber_z_a) * chord_a
            
            z_c_b1_b = jnp.interp(frac_a1, camber_x_b, camber_z_b) * chord_b
            z_c_b2_b = jnp.interp(frac_a2, camber_x_b, camber_z_b) * chord_b
            
            # Add dihedral offsets to physical corners
            zeta_a1 = z_offset_a + z_c_a1_a
            zeta_a2 = z_offset_a + z_c_a2_a
            zeta_b1 = z_offset_b + z_c_b1_b
            zeta_b2 = z_offset_b + z_c_b2_b

            z_c_ac_center = (z_c_ac_a + z_c_ac_b) / 2.0

            # Add Camber to the Flat Z Coordinates
            bound_vortex_z_a = bound_vortex_z_a + z_c_ah_a
            bound_vortex_z_b = bound_vortex_z_b + z_c_ah_b
            colloc_z_center = colloc_z_center + z_c_ac_center
            
            # Apply Twist Rotations
            # Twist rotates the X and Z coordinates around the leading edge (x_offset, z_offset)
            def apply_twist(x, z, pivot_x, pivot_z, twist_angle):
                cos_t = jnp.cos(twist_angle)
                sin_t = jnp.sin(twist_angle)
                new_x = pivot_x + cos_t * (x - pivot_x) + sin_t * (z - pivot_z)
                new_z = pivot_z - sin_t * (x - pivot_x) + cos_t * (z - pivot_z)
                return new_x, new_z
                
            bv_x_a_rot, bv_z_a_rot = apply_twist(bound_vortex_x_a, bound_vortex_z_a, x_offset_a, z_offset_a, twist_a)
            bv_x_b_rot, bv_z_b_rot = apply_twist(bound_vortex_x_b, bound_vortex_z_b, x_offset_b, z_offset_b, twist_b)

            x_a1_rot, z_a1_rot = apply_twist(panel_x_a[:-1], zeta_a1, x_offset_a, z_offset_a, twist_a)
            x_a2_rot, z_a2_rot = apply_twist(panel_x_a[1:],  zeta_a2, x_offset_a, z_offset_a, twist_a)
            
            x_b1_rot, z_b1_rot = apply_twist(panel_x_b[:-1], zeta_b1, x_offset_b, z_offset_b, twist_b)
            x_b2_rot, z_b2_rot = apply_twist(panel_x_b[1:],  zeta_b2, x_offset_b, z_offset_b, twist_b)
            
            col_twist = (twist_a + twist_b) / 2.0
            col_x_rot, col_z_rot = apply_twist(colloc_x_center, colloc_z_center, (x_offset_a+x_offset_b)/2.0, (z_offset_a+z_offset_b)/2.0, col_twist)

            # Pack into (N, 3) arrays
            left_nodes = jnp.stack([bv_x_a_rot, bound_vortex_y_a, bv_z_a_rot], axis=1)
            right_nodes = jnp.stack([bv_x_b_rot, bound_vortex_y_b, bv_z_b_rot], axis=1)
            collocations = jnp.stack([col_x_rot, colloc_y_center, col_z_rot], axis=1)
            
            corner_a1 = jnp.stack([x_a1_rot, bound_vortex_y_a, z_a1_rot], axis=1)
            corner_a2 = jnp.stack([x_a2_rot, bound_vortex_y_a, z_a2_rot], axis=1)
            corner_b1 = jnp.stack([x_b1_rot, bound_vortex_y_b, z_b1_rot], axis=1)
            corner_b2 = jnp.stack([x_b2_rot, bound_vortex_y_b, z_b2_rot], axis=1)
            
            # Normal vectors (Simplified flat panel assumption for now)
            nx = jnp.full(n_cw, jnp.sin(col_twist))
            ny = jnp.zeros(n_cw)
            nz = jnp.full(n_cw, jnp.cos(col_twist))
            
            normals = jnp.stack([nx, ny, nz], axis=1)
            
            # Geometric properties
            chords = jnp.full(n_cw, (chord_a + chord_b) / 2.0)
            areas = chords * (y_b - y_a)
            incidences = jnp.full(n_cw, col_twist)
            panels = jnp.full(n_cw, n_cw, dtype=jnp.int32)
            
            le_flags = jnp.zeros(n_cw, dtype=bool).at[0].set(True)
            te_flags = jnp.zeros(n_cw, dtype=bool).at[-1].set(True)
            
            return (left_nodes, right_nodes, collocations, normals, chords, areas, incidences,
                    corner_a1, corner_a2, corner_b1, corner_b2, le_flags, te_flags, panels)

        # We map the strip generator over the left and right edges of every spanwise station
        y_lefts = y_coords[:-1]
        y_rights = y_coords[1:]
        
        strip_results = jax.vmap(generate_strip)(y_lefts, y_rights)

        # Batch reshape of coordinates
        coord_indices = [0, 1, 2, 7, 8, 9, 10]
        coord_arrays = [jnp.reshape(strip_results[i], (-1, 3)) for i in coord_indices]

        # Reshape normals without origin shift
        strip_normals = jnp.reshape(strip_results[3], (-1, 3))

        # Batch reshape of scalar properties
        scalar_indices = [4, 5, 6, 11, 12, 13]
        strip_chords, strip_areas, strip_inc, strip_le, strip_te, cw_panels = [
            jnp.reshape(strip_results[i], (-1,)) for i in scalar_indices
        ]

        n_panels = strip_chords.shape[0]
        strip_ids = jnp.full(n_panels, current_id)

        arrays_3d = coord_arrays + [strip_normals]

        # 2. Vertical Orientation (90-deg rotation on all 3D arrays)
        if wing.vertical:
            for i in range(len(arrays_3d)):
                y_old, z_old = arrays_3d[i][:, 1], arrays_3d[i][:, 2]
                arrays_3d[i] = arrays_3d[i].at[:, 1].set(-z_old).at[:, 2].set(y_old)

        for i in range(len(coord_arrays)):
            arrays_3d[i] = arrays_3d[i] + wing.origin

        strip_L, strip_R, strip_C, strip_a1, strip_a2, strip_b1, strip_b2, strip_normals = arrays_3d

        # 3. Unconditional Append (Primary Side)
        all_left_nodes.append(strip_L)
        all_right_nodes.append(strip_R)
        all_collocations.append(strip_C)
        all_normals.append(strip_normals)
        all_a1.append(strip_a1)
        all_a2.append(strip_a2)
        all_b1.append(strip_b1)
        all_b2.append(strip_b2)

        all_chords.append(strip_chords)
        all_areas.append(strip_areas)
        all_incidences.append(strip_inc)
        all_le.append(strip_le)
        all_te.append(strip_te)
        all_panels_per_strip.append(cw_panels)
        all_surface_ids.append(strip_ids)

        # 4. Global Symmetry (Mirror across XZ-plane)
        if wing.symmetric:
            # Mirror all 3D arrays by flipping the Y-axis (-1.0)
            m_L, m_R, m_C, m_a1, m_a2, m_b1, m_b2, m_normals = [
                arr.at[:, 1].multiply(-1.0) for arr in arrays_3d
            ]

            # CRITICAL: Swap Left and Right to maintain Right-Hand Rule vortex circulation
            all_left_nodes.append(m_R)
            all_right_nodes.append(m_L)

            all_collocations.append(m_C)
            all_normals.append(m_normals)
            all_a1.append(m_a1)
            all_a2.append(m_a2)
            all_b1.append(m_b1)
            all_b2.append(m_b2)

            # 1D arrays are geometric properties (areas, chords), append them exactly as is
            all_chords.append(strip_chords)
            all_areas.append(strip_areas)
            all_incidences.append(strip_inc)
            all_le.append(strip_le)
            all_te.append(strip_te)
            all_panels_per_strip.append(cw_panels)

            # Symmetrical surfaces get negated IDs
            all_surface_ids.append(-strip_ids)
            total_strips += n_sw

    # Concatenate all wings into the final VortexDistribution
    return VortexDistribution(
        bound_vortex_left=jnp.concatenate(all_left_nodes, axis=0),
        bound_vortex_right=jnp.concatenate(all_right_nodes, axis=0),
        collocation_points=jnp.concatenate(all_collocations, axis=0),
        normal_vectors=jnp.concatenate(all_normals),
        chord_lengths=jnp.concatenate(all_chords, axis=0),
        panel_areas=jnp.concatenate(all_areas, axis=0),
        tangent_incidence_angle=jnp.concatenate(all_incidences, axis=0),
        panel_corner_a1=jnp.concatenate(all_a1, axis=0),
        panel_corner_a2=jnp.concatenate(all_a2, axis=0),
        panel_corner_b1=jnp.concatenate(all_b1, axis=0),
        panel_corner_b2=jnp.concatenate(all_b2, axis=0),
        is_leading_edge=jnp.concatenate(all_le, axis=0),
        is_trailing_edge=jnp.concatenate(all_te, axis=0),
        surface_id=jnp.concatenate(all_surface_ids, axis=0),
        panels_per_strip=jnp.concatenate(all_panels_per_strip, axis=0),
        total_strips=total_strips
    )


@jax.jit
def generate_body_panel_coordinates(body, n_cw: int, n_sw: int, surface_id: int):
    """
    Generates horizontal VLM panels for a Fuselage or Nacelle body.
    """
    # 1. Fineness to Curvature Mapping
    vec1 = jnp.array([2.0, 1.5, 1.2, 1.0])
    vec2 = jnp.array([1.0, 1.57, 3.2, 8.0])
    x_vals = jnp.linspace(0.0, 1.0, 4)
    
    nose_curv = jnp.interp(jnp.interp(body.fineness.nose, vec2, x_vals), x_vals, vec1)
    tail_curv = jnp.interp(jnp.interp(body.fineness.tail, vec2, x_vals), x_vals, vec1)
    
    semispan_h = body.diameters.maximum / 2.0 
    
    # 2. Generate Cosine Spacing for the Y-coordinates
    si = jnp.arange(1, (n_sw * 2) + 2)
    spacing = jnp.cos((2 * si - 1) / (2 * len(si)) * jnp.pi)
    
    half_idx = int((len(si) + 1) / 2)
    y_coords = semispan_h * spacing[0:half_idx][::-1]
    
    # 3. Define the Strip Generator
    def generate_body_strip(y_a, y_b):
        y_mid = (y_a + y_b) / 2.0
        
        def get_chord_and_x(y_val):
            abs_frac = jnp.abs(y_val / semispan_h)
            n_len = ((1.0 - (abs_frac ** nose_curv)) ** (1.0 / nose_curv)) * body.lengths.nose
            t_len = ((1.0 - (abs_frac ** tail_curv)) ** (1.0 / tail_curv)) * body.lengths.tail
            
            cabin_len = body.lengths.total - (body.lengths.nose + body.lengths.tail)
            chord = cabin_len + n_len + t_len
            x_origin = body.lengths.nose - n_len
            return chord, x_origin

        chord_a, x_offset_a = get_chord_and_x(y_a)
        chord_b, x_offset_b = get_chord_and_x(y_b)
        
        x_fractions = jnp.linspace(0.0, 1.0, n_cw + 1)
        panel_x_a = x_offset_a + x_fractions * chord_a
        panel_x_b = x_offset_b + x_fractions * chord_b
        
        dx_a = chord_a / n_cw
        dx_b = chord_b / n_cw
        
        bv_x_a = panel_x_a[:-1] + 0.25 * dx_a
        bv_x_b = panel_x_b[:-1] + 0.25 * dx_b
        bv_z = jnp.zeros(n_cw) 
        
        col_x = (panel_x_a[:-1] + panel_x_b[:-1]) / 2.0 + 0.75 * ((dx_a + dx_b) / 2.0)
        
        corner_a1_x = panel_x_a[:-1]
        corner_a2_x = panel_x_a[1:]
        corner_b1_x = panel_x_b[:-1]
        corner_b2_x = panel_x_b[1:]
        
        # Nodes
        left_nodes = jnp.stack([bv_x_a, jnp.full(n_cw, y_a), bv_z], axis=1)
        right_nodes = jnp.stack([bv_x_b, jnp.full(n_cw, y_b), bv_z], axis=1)
        collocations = jnp.stack([col_x, jnp.full(n_cw, y_mid), bv_z], axis=1)
        
        # Corners
        c_a1 = jnp.stack([corner_a1_x, jnp.full(n_cw, y_a), bv_z], axis=1)
        c_a2 = jnp.stack([corner_a2_x, jnp.full(n_cw, y_a), bv_z], axis=1)
        c_b1 = jnp.stack([corner_b1_x, jnp.full(n_cw, y_b), bv_z], axis=1)
        c_b2 = jnp.stack([corner_b2_x, jnp.full(n_cw, y_b), bv_z], axis=1)
        
        normals = jnp.stack([jnp.zeros(n_cw), jnp.zeros(n_cw), jnp.ones(n_cw)], axis=1)
        chords = jnp.full(n_cw, (chord_a + chord_b) / 2.0)
        areas = chords * (y_b - y_a)
        incidences = jnp.zeros(n_cw)
        panels = jnp.full(n_cw, n_cw, dtype=jnp.int32)
        
        le_flags = jnp.zeros(n_cw, dtype=bool).at[0].set(True)
        te_flags = jnp.zeros(n_cw, dtype=bool).at[-1].set(True)
        
        return (left_nodes, right_nodes, collocations, normals, chords, areas, incidences,
                c_a1, c_a2, c_b1, c_b2, le_flags, te_flags, panels)

    # 4. Vmap across the right-side strips
    y_lefts = y_coords[:-1]
    y_rights = y_coords[1:]
    strip_results = jax.vmap(generate_body_strip)(y_lefts, y_rights)
    
    # 5. Unpack and reshape (+ body.origin translation)
    origin = body.aerodynamic_center if hasattr(body, 'aerodynamic_center') else jnp.zeros(3)

    R_L_nodes = jnp.reshape(strip_results[0], (-1, 3)) + origin
    R_R_nodes = jnp.reshape(strip_results[1], (-1, 3)) + origin
    R_C_nodes = jnp.reshape(strip_results[2], (-1, 3)) + origin
    R_normals = jnp.reshape(strip_results[3], (-1, 3))
    
    chords    = jnp.reshape(strip_results[4], (-1,))
    areas     = jnp.reshape(strip_results[5], (-1,))
    inc       = jnp.reshape(strip_results[6], (-1,))
    
    R_a1      = jnp.reshape(strip_results[7], (-1, 3)) + origin
    R_a2      = jnp.reshape(strip_results[8], (-1, 3)) + origin
    R_b1      = jnp.reshape(strip_results[9], (-1, 3)) + origin
    R_b2      = jnp.reshape(strip_results[10], (-1, 3)) + origin

    le_flags  = jnp.reshape(strip_results[11], (-1,))
    te_flags  = jnp.reshape(strip_results[12], (-1,))

    cw_panels = jnp.reshape(strip_results[13], (-1,))
    
    num_panels = chords.shape[0]
    right_ids = jnp.full(num_panels, surface_id)

    # 6. Mirror for the Left Side (Y -> -Y)
    L_L_nodes = R_L_nodes.at[:, 1].multiply(-1.0)
    L_R_nodes = R_R_nodes.at[:, 1].multiply(-1.0)
    L_C_nodes = R_C_nodes.at[:, 1].multiply(-1.0)
    L_normals = R_normals.at[:, 1].multiply(-1.0)
    
    L_a1      = R_a1.at[:, 1].multiply(-1.0)
    L_a2      = R_a2.at[:, 1].multiply(-1.0)
    L_b1      = R_b1.at[:, 1].multiply(-1.0)
    L_b2      = R_b2.at[:, 1].multiply(-1.0)
    
    left_ids  = jnp.full(num_panels, -surface_id)

    # 7. Concatenate and Return (Swapping L and R nodes on the left side to preserve Right Hand Rule)
    return VortexDistribution(
        bound_vortex_left=jnp.concatenate([R_L_nodes, L_R_nodes], axis=0),
        bound_vortex_right=jnp.concatenate([R_R_nodes, L_L_nodes], axis=0),
        collocation_points=jnp.concatenate([R_C_nodes, L_C_nodes], axis=0),
        normal_vectors=jnp.concatenate([R_normals, L_normals], axis=0),
        chord_lengths=jnp.concatenate([chords, chords], axis=0),
        panel_areas=jnp.concatenate([areas, areas], axis=0),
        tangent_incidence_angle=jnp.concatenate([inc, inc], axis=0),
        panel_corner_a1=jnp.concatenate([R_a1, L_a1], axis=0),
        panel_corner_a2=jnp.concatenate([R_a2, L_a2], axis=0),
        panel_corner_b1=jnp.concatenate([R_b1, L_b1], axis=0),
        panel_corner_b2=jnp.concatenate([R_b2, L_b2], axis=0),
        is_leading_edge=jnp.concatenate([le_flags, le_flags], axis=0),
        is_trailing_edge=jnp.concatenate([te_flags, te_flags], axis=0),
        surface_id=jnp.concatenate([right_ids, left_ids], axis=0),
        panels_per_strip=jnp.concatenate([cw_panels, cw_panels], axis=0),
        total_strips=n_sw*2
    )
def combine_vortex_distributions(vd1: VortexDistribution, vd2: VortexDistribution) -> VortexDistribution:
    merged_kwargs = {}
    
    # Iterate through every field in the Equinox module natively
    for field in dataclasses.fields(vd1):
        key = field.name
        val1 = getattr(vd1, key)
        val2 = getattr(vd2, key)

        if key == "total_strips":
            # Pure Python integer addition! (No JAX tracers involved)
            merged_kwargs[key] = val1 + val2
            
        elif isinstance(val1, jnp.ndarray):
            # Safely concatenate all the dynamic JAX geometry arrays
            merged_kwargs[key] = jnp.concatenate([val1, val2], axis=0)
            
        else:
            # Fallback for any other configuration flags (just passes vd1's value)
            merged_kwargs[key] = val1

    # Instantiate a brand new module, completely circumventing JAX Treedef mismatches
    return VortexDistribution(**merged_kwargs)

# ---------------------------------------------------------
# Stateful Version
# ---------------------------------------------------------

def generate_full_vortex_distribution(state: "State", system: "Aircraft", settings: "Settings"):

    vlm_wings = system.analysis_data["vlm_wings"]
    vlm_vortex_settings = settings.analysis.aerodynamics.vortices #type: ignore

    VD = generate_wing_panel_coordinates(vlm_wings, vlm_vortex_settings)

    if settings.analysis.aerodynamics.model_fuselage: #type: ignore
        
        n_b_cw = vlm_vortex_settings.fuselage_chordwise_vortices
        n_b_sw = vlm_vortex_settings.fuselage_spanwise_vortices

        body_id = len(vlm_wings) + 1
        for fuselage in system.fuselages:
            fus_vd = generate_body_panel_coordinates(fuselage, n_b_cw, n_b_sw, body_id)
            VD = combine_vortex_distributions(VD, fus_vd)
            body_id += 1

        
        for nacelle in system.nacelles:
            nac_vd = generate_body_panel_coordinates(nacelle, n_b_cw, n_b_sw, body_id)
            VD = combine_vortex_distributions(VD, nac_vd)
            
            body_id += 1

    # Normal vector orientation correction
    normal_arr = VD.normal_vectors
    normal_arr = normal_arr.at[:,0].set(-normal_arr[:,0])
    normal_arr = normal_arr.at[:,2].set(-normal_arr[:,2])
    VD = eqx.tree_at(lambda v:v.normal_vectors, VD, normal_arr)

    n_panels    = VD.collocation_points.shape[0]
    n_time      = state.numerics.number_of_control_points
    
    updated_analysis_data = system.analysis_data | {
        "vortex_distribution": VD,
        "boundary_conditions": jnp.zeros((n_time, n_panels)),
        "relative_velocity": jnp.zeros((n_time, n_panels, 3)),
        "AICs": jnp.zeros((n_time, n_panels, n_panels, 3)),
        "singularities": jnp.zeros((n_time, n_panels)),
        "vortex_strengths": jnp.zeros((n_time, n_panels)),
        "pressure_coefficients": jnp.zeros((n_time, n_panels)),
        "VORLAX_EW_matrix": jnp.zeros((n_time, n_panels, n_panels))
    }

    current_system = eqx.tree_at(lambda s: s.analysis_data, system, updated_analysis_data)

    return state, current_system, settings