# RCAIDE/Framework/Methods/Aerodynamics/make_VLM_Wings.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created:  Jun 2021, A. Blaufox
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass

# package imports
import jax
import jax.numpy as jnp
import equinox as eqx

from jax import vmap

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System, Aircraft
    from RCAIDE.Framework.Settings import Settings
    from RCAIDE.Framework.Analyses.Aerodynamics.Vortex_Lattice import VLMSettings

# package imports 
from RCAIDE.Library.Components.Wings import Wing, WingSegment, WingSweeps, WingChords, WingControlSurface, WingDimensions
# from RCAIDE.Library.Components.Wings import All_Moving_Surface

# ----------------------------------------------------------------------------------------------------------------------
# VLM-Specific Data Structures
# ----------------------------------------------------------------------------------------------------------------------


class ControlSurfaceMetadata(eqx.Module):
    parent_wing_index: int = eqx.field(static=True)
    seg_a_index: int = eqx.field(static=True)
    seg_b_index: int = eqx.field(static=True)
    span_fraction_start: float = eqx.field(static=True)
    span_fraction_end: float = eqx.field(static=True)
    chord_fraction: float = eqx.field(static=True)
    is_slat: bool = eqx.field(static=True)


class VLMWingRecord(eqx.Module):
    """ A fully differentiable container for the VLM solver. """
    wing: eqx.Module  # The differentiable geometry
    
    # Number of Spanwise/Chordwise Panels
    n_sw: int = eqx.field(static=True, default=10)
    n_cw: int = eqx.field(static=True, default=10)

    # Number of Airfoil Coordinates:
    n_af_pts: int = eqx.field(static=True, default=2)  # Number of airfoil coordinates, 2 if no airfoil for flat line

    segment_x_offsets: jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty(0))
    segment_z_offsets: jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty(0))

    # Static VLM routing flags
    is_a_control_surface: bool = eqx.field(static=True, default=False)
    is_slat: bool = eqx.field(static=True, default=False)
    cs_ID: int = eqx.field(static=True, default=-1)
    
    # Optional metadata (Only populated if this is a control surface)
    cs_meta: Optional[ControlSurfaceMetadata] = eqx.field(static=True, default=None)
    
    # Span break data
    strip_eta_starts: jnp.ndarray   = eqx.field(default_factory=lambda: jnp.empty(0))
    strip_eta_ends: jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))
    strip_le_cs_ids: jnp.ndarray    = eqx.field(default_factory=lambda: jnp.empty(0))
    strip_te_cs_ids: jnp.ndarray    = eqx.field(default_factory=lambda: jnp.empty(0))
    strip_le_cuts: jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))
    strip_te_cuts: jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))


class Interval(eqx.Module):
    """ Represents a spanwise strip of the wing between two breaks. """
    eta_start: jnp.ndarray
    eta_end: jnp.ndarray
    le_cs_id: int = -1  # -1 means no slat
    te_cs_id: int = -1  # -1 means no flap/aileron

# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_segmented_wing(wing):
    """ Returns a tuple of (root_segment, tip_segment) for unsegmented wings. """
    
    # If it already has segments, just return them as-is
    if hasattr(wing, 'segments') and len(wing.segments) > 0:
        return wing.segments

    # 1. Build Root Segment
    root_sweeps = WingSweeps(
        quarter_chord=wing.sweeps.quarter_chord,
        leading_edge=wing.sweeps.leading_edge
    )

    root_segment = WingSegment(
        tag='root_segment',
        percent_span_location=0.0,
        twist=wing.twists.root,
        root_chord_percent=1.0,
        dihedral_outboard=wing.dihedral,
        sweeps=root_sweeps,
        thickness_to_chord=wing.thickness_to_chord,
    )
    if hasattr(wing, 'airfoil') and wing.airfoil is not None:
        root_segment = eqx.tree_at(lambda s: s.airfoil, root_segment, wing.airfoil)

    # 2. Build Tip Segment
    tip_sweeps = WingSweeps(
        quarter_chord=0.0,
        leading_edge=1e-8,
    )

    tip_segment = WingSegment(
        tag='tip_segment',
        percent_span_location=1.0,
        twist=wing.twists.tip,
        root_chord_percent=wing.taper,
        dihedral_outboard=0.0,
        sweeps=tip_sweeps,
        thickness_to_chord=wing.thickness_to_chord,
    )
    
    if hasattr(wing, 'airfoil') and wing.airfoil is not None:
        tip_segment = eqx.tree_at(lambda s: s.airfoil, tip_segment, wing.airfoil)

    return (root_segment, tip_segment)


def populate_control_sections(wing):
    """
    Evaluates wing control surfaces and maps them to specific wing segments based 
    on spanwise boundaries. Slices control surfaces that span multiple segments.
    """
    # If the wing has no control surfaces, just return it as-is
    if not hasattr(wing, 'control_surfaces') or not wing.control_surfaces:
        return wing

    new_segments = []
    
    # Iterate through the frozen segments
    for i, seg in enumerate(wing.segments):
        if i == 0:
            # The root segment (index 0) cannot have control surfaces.
            # We safely clear any existing ones.
            new_seg = eqx.tree_at(lambda s: s.control_surfaces, seg, [])
            new_segments.append(new_seg)
            continue

        prev_seg = wing.segments[i-1]
        seg_start = prev_seg.percent_span_location
        seg_end = seg.percent_span_location

        seg_cs_list = []
        
        # Check every control surface against this segment's bounds
        for cs in wing.control_surfaces:
            cs_start = cs.span_fraction_start
            cs_end = cs.span_fraction_end

            # 1D Intersection Check: Overlap occurs if start is before seg_end AND end is after seg_start
            if cs_start < seg_end and cs_end > seg_start:
                
                new_start = max(cs_start, seg_start)
                new_end = min(cs_end, seg_end)
                
                # Calculate the proportional span of the sliced control surface
                span_ratio = (new_end - new_start) / (cs_end - cs_start) if (cs_end - cs_start) > 0 else 0.0
                new_span = cs.span * span_ratio
                
                # Create a new, frozen control surface with the updated bounds
                new_cs = eqx.tree_at(
                    lambda c: (c.span_fraction_start, c.span_fraction_end, c.span),
                    cs,
                    (new_start, new_end, new_span)
                )
                seg_cs_list.append(new_cs)
                
        # Attach the valid control surfaces to this segment
        new_seg = eqx.tree_at(lambda s: s.control_surfaces, seg, seg_cs_list)
        new_segments.append(new_seg)

    # Return a new frozen wing with the updated tuple of segments
    return eqx.tree_at(lambda w: w.segments, wing, tuple(new_segments))


def convert_sweep_segments(old_sweep, root_chord_percent_a, root_chord_percent_b,
                           wing_root_chord, wingspan, old_ref=0.0, new_ref=0.25):
    """ Differentiable pure math for sweep conversion. """
    
    # If references are the same, return early (using jnp.where is safer for JIT, 
    # but since refs are usually static floats, a standard if-statement is fine here)
    if old_ref == new_ref:
        return old_sweep

    root_chord = root_chord_percent_a * wing_root_chord
    tip_chord  = root_chord_percent_b * wing_root_chord
    taper      = tip_chord / root_chord
    
    chord_mean_geo = 0.5 * (root_chord + tip_chord)
    ar = wingspan / chord_mean_geo  

    # Convert to Leading Edge sweep
    if old_ref == 0.0:
        sweep_LE = old_sweep
    else:
        sweep_LE = jnp.arctan(jnp.tan(old_sweep) + 4 * old_ref * (1 - taper) / (ar * (1 + taper)))

    # Convert from LE sweep to the desired reference
    new_sweep = jnp.arctan(jnp.tan(sweep_LE) - 4 * new_ref * (1 - taper) / (ar * (1 + taper)))

    return new_sweep


def calculate_segment_offsets(wing):
    """ 
    Calculates the cumulative X and Y/Z offsets for wing segments.
    Returns a new Wing PyTree with updated segment attributes. 
    """
    wingspan = wing.spans.projected if wing.symmetric else wing.spans.projected * 2
    wing_halfspan = wing.spans.projected * 0.5 if wing.symmetric else wing.spans.projected

    new_segments = []
    new_x_offsets = []
    new_z_offsets = []
    
    current_x_offset = 0.0
    current_z_offset = 0.0

    for i, seg in enumerate(wing.segments):
        # Base case: Root segment has no offsets
        if i == 0:
            new_seg = eqx.tree_at(lambda s: s.chords.root, seg, (seg.root_chord_percent * wing.chords.root))
            new_segments.append(new_seg)
            new_x_offsets.append(current_x_offset)
            new_z_offsets.append(current_z_offset)
            continue

        prev_seg = wing.segments[i-1]

        # 1. Grab the provided sweep (Assuming it defaults to 0.0, NOT None)
        le_sweep_provided = prev_seg.sweeps.leading_edge

        # 2. Compute the fallback converted sweep unconditionally
        converted_sweep = convert_sweep_segments(
            prev_seg.sweeps.quarter_chord,
            prev_seg.root_chord_percent,
            seg.root_chord_percent,
            wing.chords.root,
            wingspan,
            old_ref=0.25,
            new_ref=0.0
        )

        # 3. Use Data Flow to select the correct value
        # (Using < 1e-8 is safer than exact == 0.0 for floating point tracers)
        le_sweep = jnp.where(
            jnp.abs(le_sweep_provided) < 1e-8,
            converted_sweep,
            le_sweep_provided
        )
            
        # Update the previous segment in our new list with the calculated LE sweep
        new_segments[i-1] = eqx.tree_at(lambda s: s.sweeps.leading_edge, new_segments[i-1], le_sweep)

        # 2. Cumulative Offsets
        section_span = (seg.percent_span_location - prev_seg.percent_span_location) * wing_halfspan
        current_x_offset = current_x_offset + section_span * jnp.tan(le_sweep)
        current_z_offset = current_z_offset + section_span * jnp.tan(prev_seg.dihedral_outboard)

        # 3. Update current segment
        new_seg = eqx.tree_at(lambda s: s.chords.root, seg, seg.root_chord_percent * wing.chords.root)
        new_segments.append(new_seg)
        new_x_offsets.append(current_x_offset)
        new_z_offsets.append(current_z_offset)

    # Standard VLM cap: force the absolute tip segment's LE sweep to ~0 to prevent singularities
    if new_segments:
        new_segments[-1] = eqx.tree_at(lambda s: s.sweeps.leading_edge, new_segments[-1], 1e-8)

    # Pack the tuple back into the Wing PyTree
    return eqx.tree_at(lambda w: w.segments, wing, tuple(new_segments)), jnp.array(new_x_offsets), jnp.array(new_z_offsets)


def setup_cs_skeleton(
        cs: WingControlSurface,
        parent_wing_idx: int,
        seg_a_idx: int,
        seg_b_idx: int,
        parent_wing: Wing,
        parent_intervals: list[Interval],
        parent_x_offsets: jnp.ndarray,
        parent_z_offsets: jnp.ndarray,
        cs_ID: int,
        vlm_settings: "VLMSettings"
    ):
    """ Builds the empty skeleton and metadata record in pure Python. """
    
    is_slat = "slat" in cs.tag.lower()

    cs_start = cs.span_fraction_start
    cs_end = cs.span_fraction_end

    base_n_sw: int = vlm_settings.vortices.wing_spanwise_vortices #type: ignore

    # Dry run of normalized spacing
    if vlm_settings.vortices.spanwise_cosine_spacing:
        thetan = jnp.linspace(jnp.pi/2, 0, base_n_sw + 1)
        base_etas = jnp.cos(thetan)
    else:
        base_etas = jnp.linspace(0.0, 1.0, base_n_sw + 1)

    # Snap the base etas to the parent wing's topological breaks
    req_etas = jnp.append(jnp.array([i.eta_start for i in parent_intervals]), parent_intervals[-1].eta_end)
    shifted_idxs = jnp.zeros(base_n_sw + 1)
    
    for req_eta in req_etas:
        diffs = jnp.abs(base_etas - req_eta) + shifted_idxs
        idx = jnp.argmin(diffs)
        
        # JAX arrays are immutable, must reassign to save the update.
        base_etas = base_etas.at[idx].set(req_eta)
        shifted_idxs = shifted_idxs.at[idx].set(jnp.inf)
        
    base_etas = jnp.sort(base_etas)
    
    # Extract only the etas that fall inside this control surface
    cs_etas = base_etas[(base_etas >= cs_start - 1e-6) & (base_etas <= cs_end + 1e-6)]
    
    # Calculate the static panel counts
    cs_n_sw = max(len(cs_etas) - 1, 1)
    
    base_n_cw: int = vlm_settings.vortices.wing_chordwise_vortices #type: ignore
    
    # Use .item() to safely extract the standard Python integer from the JAX 0D array
    cs_n_cw = max(int(jnp.ceil(cs.root_chord_percent * base_n_cw).item()), 2)
    
    # Build the metadata map
    cs_meta = ControlSurfaceMetadata(
        parent_wing_index=parent_wing_idx,
        seg_a_index=seg_a_idx,
        seg_b_index=seg_b_idx,
        span_fraction_start=cs.span_fraction_start,
        span_fraction_end=cs.span_fraction_end,
        chord_fraction=cs.root_chord_percent,
        is_slat=is_slat
    )
    
    # Spawn the Skeleton Wing (Floats don't matter here, they get overwritten in JAX
    skeleton_chords = WingChords(
        root=parent_wing.chords.root * cs.root_chord_percent
    )
    skeleton_spans = WingDimensions(
        projected=(cs.span_fraction_end-cs.span_fraction_start) * parent_wing.spans.projected
    )

    skeleton_wing = Wing(
        tag=f"{parent_wing.tag}__cs_id_{cs_ID}",
        symmetric=parent_wing.symmetric,
        vertical=parent_wing.vertical,
        chords=skeleton_chords,
        spans=skeleton_spans,
        taper=1.0 # Dummy taper, gets updated in update geometry step before meshing
    )

    cs_segments = convert_to_segmented_wing(skeleton_wing)
    skeleton_wing = eqx.tree_at(lambda w: w.segments, skeleton_wing, cs_segments)
    skeleton_wing = calculate_cs_geometry(skeleton_wing, parent_wing, parent_x_offsets, parent_z_offsets, cs_meta)
    skeleton_wing, seg_x_offsets, seg_z_offsets = calculate_segment_offsets(skeleton_wing)

    # Pack it into VLMWingRecord
    return VLMWingRecord(
        wing=skeleton_wing,
        is_a_control_surface=True,
        cs_ID=cs_ID,
        cs_meta=cs_meta,
        n_cw=cs_n_cw,
        n_sw=cs_n_sw,
        strip_eta_starts=cs_etas[:-1],
        strip_eta_ends=cs_etas[1:],
        strip_le_cuts=jnp.zeros(2),
        strip_te_cuts=jnp.ones(1),
        segment_x_offsets=seg_x_offsets,
        segment_z_offsets=seg_z_offsets
    )

@jax.jit
def calculate_cs_geometry(skeleton_wing, parent_wing, parent_x_offsets, parent_z_offsets, cs_meta: ControlSurfaceMetadata):
    """ 
    Takes the traced parent wing, calculates the CS geometry, 
    and injects it into the skeleton. 
    """
    # 1. Unpack the exact segments this CS sits between
    # (Because the indices are static, JAX can trace this perfectly)
    seg_a = parent_wing.segments[cs_meta.seg_a_index]
    seg_b = parent_wing.segments[cs_meta.seg_b_index]
    
    span_a = seg_a.percent_span_location
    span_b = seg_b.percent_span_location
    
    cs_start = cs_meta.span_fraction_start
    cs_end = cs_meta.span_fraction_end
    
    # 2. Differentiable Interpolations
    # We use jnp.interp to safely interpolate along the segment bounds
    xp = jnp.array([span_a, span_b])
    
    twist_root = jnp.interp(cs_start, xp, jnp.array([seg_a.twist, seg_b.twist]))
    twist_tip  = jnp.interp(cs_end, xp, jnp.array([seg_a.twist, seg_b.twist]))
    
    local_chord_root = jnp.interp(cs_start, xp, jnp.array([seg_a.chords.root, seg_b.chords.root]))
    local_chord_tip  = jnp.interp(cs_end, xp, jnp.array([seg_a.chords.root, seg_b.chords.root]))
    
    cs_root_chord = local_chord_root * cs_meta.chord_fraction
    cs_tip_chord  = local_chord_tip * cs_meta.chord_fraction
    
    # Safe division for taper
    taper = jnp.where(cs_root_chord != 0.0, cs_tip_chord / cs_root_chord, 0.0)
    
    # 3. Origin Offsets
    wing_halfspan = jnp.where(parent_wing.symmetric, parent_wing.spans.projected * 0.5, parent_wing.spans.projected)
    
    # jax.lax.cond is safer than 'if' inside JIT for booleans that might become dynamic later, 
    # but since is_slat is static, a standard Python if/else works here too.
    le_te_offset = jnp.where(cs_meta.is_slat, 0.0, (1.0 - cs_meta.chord_fraction) * local_chord_root)

    x_off_a = parent_x_offsets[cs_meta.seg_a_index]
    x_off_b = parent_x_offsets[cs_meta.seg_b_index]
    x_off = jnp.interp(cs_start, xp, jnp.array([x_off_a, x_off_b]))

    y_off = cs_start * wing_halfspan

    z_off_a = parent_z_offsets[cs_meta.seg_a_index]
    z_off_b = parent_z_offsets[cs_meta.seg_b_index]
    z_off = jnp.interp(cs_start, xp, jnp.array([z_off_a, z_off_b]))
    
    new_origin = parent_wing.origin + jnp.array([[x_off + le_te_offset, y_off, z_off]])
    
    # 4. Inject back into the skeleton using eqx.tree_at
    updated_cs_wing = eqx.tree_at(
        lambda w: (w.chords.root, w.chords.tip, w.twists.root, w.twists.tip, w.taper, w.origin),
        skeleton_wing,
        (cs_root_chord, cs_tip_chord, twist_root, twist_tip, taper, new_origin)
    )
    
    return updated_cs_wing


def generate_topological_span_breaks(wing: Wing) -> list[Interval]:
    """
    Finds every unique spanwise slicing plane (from segments and control surfaces)
    and builds non-overlapping spanwise intervals.
    """
    # 1. Collect all raw span fractions where a break occurs
    raw_breaks = []
    
    # Add segment boundaries
    for seg in wing.segments:
        raw_breaks.append(seg.percent_span_location)
        
    # Add control surface boundaries
    if hasattr(wing, 'control_surfaces'):
        for cs in wing.control_surfaces:
            raw_breaks.append(cs.span_fraction_start)
            raw_breaks.append(cs.span_fraction_end)
            
    # 2. Sort and deduplicate (merge coincident breaks using a tight tolerance)
    raw_breaks = jnp.array(raw_breaks)
    raw_breaks = jnp.sort(raw_breaks)
    
    unique_breaks = [raw_breaks[0]]
    for val in raw_breaks[1:]:
        if val - unique_breaks[-1] > 1e-6: # 1e-6 tolerance for floating point overlaps
            unique_breaks.append(val)
            
    # 3. Build the intervals and map the Control Surface IDs
    intervals = []
    for i in range(len(unique_breaks) - 1):
        eta_start = unique_breaks[i]
        eta_end = unique_breaks[i+1]
        midpoint = (eta_start + eta_end) / 2.0  # Use midpoint to check who lives here
        
        le_id = -1
        te_id = -1
        
        # Check which control surfaces overlap this interval's midpoint
        if hasattr(wing, 'control_surfaces'):
            for cs_idx, cs in enumerate(wing.control_surfaces):
                if cs.span_fraction_start < midpoint < cs.span_fraction_end:
                    if "slat" in cs.tag.lower():
                        le_id = cs_idx
                    else:
                        te_id = cs_idx
                        
        intervals.append(Interval(eta_start, eta_end, le_id, te_id))
        
    return intervals


def validate_airfoil_resolutions(wing):
    # Semi=proofing against future Airfoil subclasses by checking name instead of isinstance
    is_airfoil = lambda node: hasattr(node, '__class__') and 'Airfoil' in node.__class__.__name__

    # Get leaves with Airfoils as stopping points
    all_leaves = jax.tree_util.tree_leaves(wing, is_leaf=is_airfoil)

    # Filter out non-Airfoil leaves
    airfoils = [leaf for leaf in all_leaves if is_airfoil(leaf)]

    if airfoils:
        resolutions = [af.coordinates.shape[0] for af in airfoils]
        if len(set(resolutions)) > 1:
            raise ValueError(f"VLM discretization requires all airfoils on a wing to have the same number of points. "
                             f"On wing '{wing.tag}' found resolutions: {set(resolutions)}.")
        else:
            return resolutions[0]
    else:
        return 2 # Number of airfoil coordinates, 2 if no airfoil for flat line


def discretize_wings(state: "State", system: "Aircraft", settings: "Settings"):
    # unpack inputs
    vlm_settings: VLMSettings = settings.analysis.aerodynamics #type: ignore
    discretize_cs = vlm_settings.discretize_control_surfaces
        
    # Reformat original wings to have at least 2 segments and additional values for processing later
    vlm_records = []
    updated_system = system
    
    for wing_idx, wing in enumerate(system.wings): #type: ignore
        wing: Wing
        if len(wing.segments) == 0:
            # convert to preferred format for the panelization loop
            new_segments = convert_to_segmented_wing(wing)
            wing = eqx.tree_at(lambda w: w.segments, wing, new_segments)
        else:
            # TODO: Add support for All_Moving_Surface class
            # if issubclass(wing.__class__, All_Moving_Surface): # these cases unsupported due to the way the panelization loop is structured at the moment
            #     if not (wing.hinge_vector == jnp.array([0.,0.,0.])).all() and wing.use_constant_hinge_fraction: #type: ignore
            #         raise ValueError("A hinge_vector is specified, but the surface is set to use a constant hinge fraction")
            #     if len(wing.control_surfaces.subcomponents) > 0:
            #         raise ValueError('Input: control surfaces are not supported on all-moving surfaces at this time')
            for segment in wing.segments:
                if len(segment.control_surfaces) > 0:
                    raise ValueError(f"Found control surfaces on segment '{segment.tag}' of wing '{wing.tag}'. \
                                     Control surfaces must be attributes of the wing itself.")
        
        valid_cs = [cs for cs in wing.control_surfaces if "slat" not in cs.tag.lower()]
        wing = eqx.tree_at(lambda w: w.control_surfaces, wing, valid_cs)

        wing = populate_control_sections(wing) if discretize_cs else wing
        
        wing, seg_x_offsets, seg_z_offsets = calculate_segment_offsets(wing)

        intervals = generate_topological_span_breaks(wing)

        n_af_pts = validate_airfoil_resolutions(wing)

        le_cuts = []
        te_cuts = []
        for i in intervals:
            # Default is no cut (0.0 to 1.0)
            le_c = 0.0
            te_c = 1.0
            if i.le_cs_id != -1:
                le_c = wing.control_surfaces[i.le_cs_id].root_chord_percent
            if i.te_cs_id != -1:
                te_c = 1.0 - wing.control_surfaces[i.te_cs_id].root_chord_percent
            le_cuts.append(le_c)
            te_cuts.append(te_c)

        main_record = VLMWingRecord(
            wing=wing,
            strip_eta_starts=jnp.array([i.eta_start for i in intervals]),
            strip_eta_ends=jnp.array([i.eta_end for i in intervals]),
            strip_le_cs_ids=jnp.array([i.le_cs_id for i in intervals]),
            strip_te_cs_ids=jnp.array([i.te_cs_id for i in intervals]),
            strip_le_cuts=jnp.array(le_cuts),
            strip_te_cuts=jnp.array(te_cuts),
            n_sw=vlm_settings.vortices.wing_spanwise_vortices, #type: ignore
            n_cw=vlm_settings.vortices.wing_chordwise_vortices, #type: ignore
            n_af_pts=n_af_pts,
            segment_x_offsets=seg_x_offsets,
            segment_z_offsets=seg_z_offsets
        )
        vlm_records.append(main_record)

        cs_ID = 0
        for seg_idx, seg in enumerate(wing.segments):
            if len(seg.control_surfaces) > 0:
                for cs in seg.control_surfaces:
                    seg_a_idx = seg_idx - 1
                    seg_b_idx = seg_idx

                    cs_record = setup_cs_skeleton(
                        cs, 
                        parent_wing_idx=wing_idx,
                        seg_a_idx=seg_a_idx, 
                        seg_b_idx=seg_b_idx, 
                        parent_wing=wing,
                        parent_x_offsets=seg_x_offsets,
                        parent_z_offsets=seg_z_offsets,
                        parent_intervals=intervals,
                        cs_ID=cs_ID,
                        vlm_settings=vlm_settings
                    )
                    
                    vlm_records.append(cs_record)
                    cs_ID += 1
        
        updated_analysis_data = system.analysis_data | {"vlm_wings": vlm_records}
        
        updated_system  = eqx.tree_at(lambda s: s.analysis_data, updated_system, updated_analysis_data)

    return state, updated_system, settings


def update_wing_geometry(state: "State", system: "Aircraft", settings: "Settings"):
    old_vlm_records = system.analysis_data["vlm_wings"]
    ready_vlm_records = []
    
    updated_main_wings = []
    updated_x_offsets = []
    updated_z_offsets = []
    main_wing_counter = 0

    for record in old_vlm_records:
        if not record.is_a_control_surface:
            # 1. Pull the raw, differentiable wing from the global system
            fresh_wing = system.wings[main_wing_counter]
            
            # 2. Format it for the VLM dynamically so JAX can trace it
            if len(fresh_wing.segments) == 0:
                new_segments = convert_to_segmented_wing(fresh_wing)
                fresh_wing = eqx.tree_at(lambda w: w.segments, fresh_wing, new_segments)
            
            # 3. Calculate offsets using the bridged geometry
            updated_wing, updated_x_offsets, updated_z_offsets = calculate_segment_offsets(fresh_wing)
            
            updated_main_wings.append(updated_wing)
            ready_record = eqx.tree_at(lambda r: (r.wing, r.segment_x_offsets, r.segment_z_offsets),
                                       record, (updated_wing, updated_x_offsets, updated_z_offsets))
            ready_vlm_records.append(ready_record)
            
            main_wing_counter += 1
            
        else:
            # --- CONTROL SURFACES ---
            parent_idx = record.cs_meta.parent_wing_index
            parent_wing = updated_main_wings[parent_idx] #type: ignore
            parent_x_offsets = updated_x_offsets
            parent_z_offsets = updated_z_offsets
            
            updated_cs_wing = calculate_cs_geometry(
                record.wing,
                parent_wing,
                parent_x_offsets,
                parent_z_offsets,
                record.cs_meta
            )
            
            ready_record = eqx.tree_at(lambda r: r.wing, record, updated_cs_wing)
            ready_vlm_records.append(ready_record)

    # Pack the updated records back into the solver dictionary ONLY.
    # We DO NOT overwrite system.wings, preserving the global vehicle
    new_analysis_data = system.analysis_data | {"vlm_wings": tuple(ready_vlm_records)}
    # TODO: Regenerate vortex distribution with updated wings.
    
    current_system = eqx.tree_at(lambda s: s.analysis_data, system, new_analysis_data)

    return state, current_system, settings