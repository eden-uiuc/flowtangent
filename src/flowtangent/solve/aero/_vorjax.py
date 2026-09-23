# ============================================================================
# initialization.py
# ============================================================================

# flowtangent/Framework/Methods/Aerodynamics/VLM/initialization
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Aircraft, Settings, State

import dataclasses
import warnings
from typing import Any, Callable, Iterable, Optional

# package imports
import equinox as eqx
import jax
import jax.numpy as jnp

# package imports
import sklearn

from ...components._wings import Wing, WingSegment, WingSweeps
from ...core._processes import Process, ProcessStep
from ...data import units as U  # noqa: N812
from ...functional.aero.shocks import oblique_shock, theta_beta_mach
from ...functional.aero.transonic import ensemble_CL_spline, peaked_CL_spline
from ...sim.initialize import initialize_aerodynamics
from ...utils import Module, TreePath, field, io, method_field, static_field, update

# FT imports

# ----------------------------------------------------------------------------------------------------------------------
#  API Setup
# ----------------------------------------------------------------------------------------------------------------------

__all__ = [
    "VORJAX",
    "VORJAXSettings",
]

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs(
    "settings.analysis.aerodynamics: VLMSettings",
    "system.wings.[Wing].chords.mean_aerodynamic",
    "system.wings.[Wing].spans.projected",
    "system.mass_properties.center_of_gravity",
)
@io.outputs("system.reference_geometry", "system.analysis_data")
def initialize_VORJAX_data(state: State, system: Aircraft, settings: Settings):
    """
    Parses the vehicle geometry to find the primary reference parameters
    and packs them into JAX arrays for the VLM solver.
    """

    if "VORJAX" not in settings.analysis.aerodynamics.__class__.__name__:
        raise ValueError(
            "settings.analysis.aerodynamics are not VORJAX Settings."
            "Please use Flowtangent.Framework.Analysis.Vortex_Lattice.VLMSettings"
        )

    # Standard Python Control Flow (Safe outside of @jax.jit)
    wings = system.wings

    ref_wing = None
    if hasattr(wings, "main_wing"):
        ref_wing = wings.main_wing
    elif len(wings) > 0:
        ref_wing = wings[0]

    if ref_wing is not None:
        ref_wing: Wing
        c_bar = ref_wing.chords.mean_aerodynamic
        x_mac = ref_wing.aerodynamic_center[0] + ref_wing.origin[0][0]
        z_mac = ref_wing.aerodynamic_center[2] + ref_wing.origin[0][2]
        b_ref = ref_wing.spans.projected
    else:
        c_bar = 0.0
        x_mac = 0.0
        z_mac = 0.0
        b_ref = 0.0

        for wing in wings:
            if not wing.vertical:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar = wing.chords.mean_aerodynamic
                    x_mac = wing.aerodynamic_center[0] + wing.origin[0][0]
                    z_mac = wing.aerodynamic_center[2] + wing.origin[0][2]
                    b_ref = wing.spans.projected

    # 2. Resolve the Center of Gravity / Moment Reference Center
    # Assuming the legacy shape was a 2D array like [[x, y, z]]
    cg_array = system.mass_properties.center_of_gravity
    x_cg = cg_array[0][0]
    z_cg = cg_array[0][2]

    x_m = jnp.where(x_cg == 0.0, x_mac, x_cg)
    z_m = jnp.where(x_cg == 0.0, z_mac, z_cg)

    # 3. Pack into strict JAX arrays
    # We use jnp.atleast_1d and explicit array shapes to match your jnp.empty structures
    new_ref_geom = system.reference_geometry.__class__(
        mean_aerodynamic_chord=jnp.atleast_1d(c_bar),  # type: ignore
        projected_span=jnp.atleast_1d(b_ref),  # type: ignore
        aerodynamic_center=jnp.array([[x_mac, 0.0, z_mac]]),  # type: ignore
        center_of_gravity=jnp.array([[x_m, 0.0, z_m]]),  # type: ignore
    )

    # Add analysis data keys
    initial_analysis_data = {
        "vortex_distribution": None,
        "VICs": None,
        "induced_wake": None,
        "boundary_conditions": None,
        "relative_velocity": None,
        "singularities": None,
        "vortex_strengths": None,
        "dCp": None,
    }

    updated_system = update(
        system,
        (
            ("reference_geometry", new_ref_geom),
            ("analysis_data", initial_analysis_data),
        ),
    )

    return state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
# VortexDistribution Data Structure
# ----------------------------------------------------------------------------------------------------------------------


class VortexDistribution(Module):
    """
    A globally unstructured VLM mesh.
    N = total number of panels across the entire aircraft.
    """

    # --- Base Geometric State ---
    panel_vertices: jax.Array  # (N, 4, 3), CCW from Front-Left
    camber_slopes: jax.Array  # (N,) Camber slope at each panel
    wedge_angles: jax.Array  # (N_s,) Leading edge wedge angle for supersonic correction

    # --- Identity & Topology (Calculated before flattening) ---
    surface_id: jax.Array  # (N,) ID of the originating wing/fuselage
    control_surface_id: jax.Array  # (N,) ID of the control surface (-1 for solid wing)
    is_leading_edge: jax.Array  # (N,) Boolean mask
    is_trailing_edge: jax.Array  # (N,) Boolean mask

    # --- Static Structural Integers (NOT traced by JAX) ---
    total_panels: int = eqx.field(static=True)
    total_strips: int = eqx.field(static=True)

    def __init__(
        self,
        panel_vertices,
        camber_slopes,
        wedge_angles,
        surface_id,
        control_surface_id,
        is_leading_edge,
        is_trailing_edge,
        total_panels=None,
        total_strips=None,
        **kwargs,
    ):
        self.panel_vertices = panel_vertices
        self.camber_slopes = camber_slopes
        self.wedge_angles = wedge_angles
        self.surface_id = surface_id
        self.control_surface_id = control_surface_id
        self.is_leading_edge = is_leading_edge
        self.is_trailing_edge = is_trailing_edge

        # If passed in from mirror_distribution or unpacking, use them directly
        if total_panels is not None:
            self.total_panels = total_panels
        else:
            self.total_panels = int(panel_vertices.shape[0])

        if total_strips is not None:
            self.total_strips = total_strips
        else:
            self.total_strips = int(jnp.sum(is_leading_edge))

    # --- Derived Physics (@properties) ---
    @property
    def bound_vortex_left(self):
        verts = self.panel_vertices
        return 0.75 * verts[:, 0, :] + 0.25 * verts[:, 1, :]

    @property
    def bound_vortex_right(self):
        verts = self.panel_vertices
        return 0.75 * verts[:, 3, :] + 0.25 * verts[:, 2, :]

    @property
    def bound_vortex_inboard(self):
        left = self.bound_vortex_left
        right = self.bound_vortex_right

        # True if Left is further outboard (larger absolute Y) than Right.
        flip = jnp.abs(left[:, 1]) > jnp.abs(right[:, 1])

        # Expand mask from (N,) to (N, 1) so it broadcasts across X, Y, Z
        return jnp.where(flip[:, None], right, left)

    @property
    def bound_vortex_outboard(self):
        left = self.bound_vortex_left
        right = self.bound_vortex_right

        flip = jnp.abs(left[:, 1]) > jnp.abs(right[:, 1])
        return jnp.where(flip[:, None], left, right)

    @property
    def bound_vortex_center(self):
        # Center is mathematically identical regardless of inboard/outboard flip
        return 0.5 * (self.bound_vortex_left + self.bound_vortex_right)

    @property
    def bound_vortex_A(self):
        """Returns the vortex endpoint with the strictly smaller Y-coordinate for VIC calculation"""
        left = self.bound_vortex_left
        right = self.bound_vortex_right

        flip = left[:, 1] > right[:, 1]
        return jnp.where(flip[:, None], right, left)

    @property
    def bound_vortex_B(self):
        """Returns the bound vortex endpoint with the strictly larger Y-coordinate for VIC calculation"""
        left = self.bound_vortex_left
        right = self.bound_vortex_right

        flip = left[:, 1] > right[:, 1]
        return jnp.where(flip[:, None], left, right)

    @property
    def collocation_points(self):
        verts = self.panel_vertices
        colloc_left = 0.25 * verts[:, 0, :] + 0.75 * verts[:, 1, :]
        colloc_right = 0.25 * verts[:, 3, :] + 0.75 * verts[:, 2, :]
        return 0.5 * (colloc_left + colloc_right)

    @property
    def normal_vectors(self):
        verts = self.panel_vertices
        diag_1 = verts[:, 2, :] - verts[:, 0, :]
        diag_2 = verts[:, 1, :] - verts[:, 3, :]

        # Swapped order: diag_2 x diag_1 forces the right-hand rule to point UP (+Z)
        raw_normals = jnp.cross(diag_2, diag_1)

        return raw_normals / jnp.linalg.norm(raw_normals, axis=1, keepdims=True)

    @property
    def chord_lengths(self):
        verts = self.panel_vertices
        chord_left = jnp.linalg.norm(verts[:, 1, :] - verts[:, 0, :], axis=-1)
        chord_right = jnp.linalg.norm(verts[:, 2, :] - verts[:, 3, :], axis=-1)
        return (chord_left + chord_right) / 2.0

    @property
    def incidence_angle(self):
        verts = self.panel_vertices

        mid_front = 0.5 * (verts[:, 0, :] + verts[:, 3, :])
        mid_back = 0.5 * (verts[:, 1, :] + verts[:, 2, :])

        dx = mid_back[:, 0] - mid_front[:, 0]
        dz = mid_back[:, 2] - mid_front[:, 2]

        physical_twist = jnp.arctan2(-dz, dx)
        camber_angle = jnp.arctan(self.camber_slopes)

        return physical_twist + camber_angle

    @property
    def panel_areas(self):
        verts = self.panel_vertices
        diag_1 = verts[:, 2, :] - verts[:, 0, :]
        diag_2 = verts[:, 1, :] - verts[:, 3, :]
        raw_normals = jnp.cross(diag_2, diag_1)
        return 0.5 * jnp.linalg.norm(raw_normals, axis=1)

    @property
    def strip_ids(self):
        return jnp.cumsum(self.is_leading_edge) - 1

    @property
    def panels_per_strip(self):
        strip_ids = self.strip_ids
        panel_ones = jnp.ones_like(strip_ids, dtype=jnp.float32)
        stripwise_panels = jax.ops.segment_sum(panel_ones, strip_ids, num_segments=self.total_strips)
        return stripwise_panels[strip_ids]


def mirror_distribution(vd: VortexDistribution) -> VortexDistribution:
    """Creates the symmetric left-side counterpart of a right-side wing."""

    # 1. Flip the Y coordinates (Index 1)
    flipped_verts = vd.panel_vertices.at[:, :, 1].multiply(-1.0)

    # 2. Reorder the corners to fix the winding (Maintain UPWARD normals)
    # Original: [0: Front-Left, 1: Back-Left, 2: Back-Right, 3: Front-Right]
    # Mirrored: Swap Left and Right
    # New order: [3, 2, 1, 0]
    mirrored_verts = flipped_verts[:, jnp.array([3, 2, 1, 0]), :]

    # Create the new kwargs dict
    mirrored_kwargs = {}
    for fld in dataclasses.fields(vd):
        key = fld.name
        if key == "panel_vertices":
            mirrored_kwargs[key] = mirrored_verts
        else:
            # Copy all other flags, surface IDs, and strip IDs as-is
            mirrored_kwargs[key] = getattr(vd, key)

    return VortexDistribution(**mirrored_kwargs)


def merge_vortex_distributions(vd_list: list[VortexDistribution]) -> VortexDistribution:
    """
    Merges a list of flattened VortexDistributions into a single global unstructured mesh.
    Highly optimized for JAX by using single-pass concatenations.
    """
    if not vd_list:
        raise ValueError("Cannot merge an empty list of VortexDistributions.")
    if len(vd_list) == 1:
        return vd_list[0]

    merged_kwargs = {}

    # Iterate through the fields defined in the Equinox module
    for fld in dataclasses.fields(vd_list[0]):
        key = fld.name
        first_val = getattr(vd_list[0], key)

        if key == "strip_id":
            # Accumulate strip IDs with a running offset to guarantee global uniqueness
            adjusted_strip_ids = []
            current_offset = 0

            for vd in vd_list:
                val = getattr(vd, key)
                adjusted_strip_ids.append(val + current_offset)

                if val.size > 0:
                    current_offset += jnp.max(val) + 1

            merged_kwargs[key] = jnp.concatenate(adjusted_strip_ids, axis=0)

        elif key in ["total_panels", "total_strips"]:
            # Explicitly sum the structural integers across all meshes
            merged_kwargs[key] = sum(getattr(vd, key) for vd in vd_list)

        elif isinstance(first_val, jax.Array):
            # One-shot concatenation for all geometry, flags, and surface IDs
            arrays_to_concat = [getattr(vd, key) for vd in vd_list]
            merged_kwargs[key] = jnp.concatenate(arrays_to_concat, axis=0)

        else:
            # Fallback for static configuration fields (assumes identical across the list)
            merged_kwargs[key] = first_val

    return VortexDistribution(**merged_kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_segmented_wing(wing):
    """Returns a tuple of (root_segment, tip_segment) for unsegmented wings."""

    # If it already has segments, just return them as-is
    if hasattr(wing, "segments") and len(wing.segments) > 0:
        return wing.segments

    # 1. Build Root Segment
    root_sweeps = WingSweeps(quarter_chord=wing.sweeps.quarter_chord, leading_edge=wing.sweeps.leading_edge)

    root_segment = WingSegment(
        name="root_segment",
        percent_span_location=0.0,
        twist=wing.twists.root,
        root_chord_percent=1.0,
        dihedral_outboard=wing.dihedral,
        sweeps=root_sweeps,
        thickness_to_chord=wing.thickness_to_chord,
    )
    if hasattr(wing, "airfoil") and wing.airfoil is not None:
        root_segment = update(root_segment, "airfoil", wing.airfoil)

    # 2. Build Tip Segment
    tip_sweeps = WingSweeps(
        quarter_chord=0.0,
        leading_edge=1e-8,
    )

    tip_segment = WingSegment(
        name="tip_segment",
        percent_span_location=1.0,
        twist=wing.twists.tip,
        root_chord_percent=wing.taper,
        dihedral_outboard=0.0,
        sweeps=tip_sweeps,
        thickness_to_chord=wing.thickness_to_chord,
    )

    if hasattr(wing, "airfoil") and wing.airfoil is not None:
        tip_segment = update(tip_segment, "airfoil", wing.airfoil)

    return (root_segment, tip_segment)


def validate_airfoil_resolutions(wing):
    # Semi=proofing against future Airfoil subclasses by checking name instead of isinstance
    def is_airfoil(node):
        return hasattr(node, "__class__") and "Airfoil" in node.__class__.__name__

    # Get leaves with Airfoils as stopping points
    all_leaves = jax.tree_util.tree_leaves(wing, is_leaf=is_airfoil)

    # Filter out non-Airfoil leaves
    airfoils = [leaf for leaf in all_leaves if is_airfoil(leaf)]

    if airfoils:
        resolutions = [af.coordinates.shape[0] for af in airfoils]
        if len(set(resolutions)) > 1:
            raise ValueError(
                f"VLM discretization requires all airfoils on a wing to have the same number of points. "
                f"On wing '{wing.name}' found resolutions: {set(resolutions)}."
            )
        else:
            return resolutions[0]
    else:
        return 2  # Number of airfoil coordinates, 2 if no airfoil for flat line


def find_intervals(wing: Wing) -> tuple[jax.Array, jax.Array]:
    """
    Finds every unique spanwise slicing plane (from segments and control surfaces)
    and builds non-overlapping spanwise intervals.
    """
    # 1. Collect all raw span fractions where a break occurs

    segment_boundaries = [seg.percent_span_location for seg in wing.segments] + [1.0]
    cs_span_starts = [cs.span_fraction_start for cs in wing.control_surfaces]
    cs_span_ends = [cs.span_fraction_end for cs in wing.control_surfaces]

    raw_breaks = jnp.sort(jnp.array(segment_boundaries + cs_span_starts + cs_span_ends))

    diffs = jnp.diff(raw_breaks)
    mask = jnp.concatenate([jnp.array([True]), diffs > 1e-6])
    unique_breaks = raw_breaks[mask]

    # Find midpoints between breaks
    eta_starts = unique_breaks[:-1]
    eta_ends = unique_breaks[1:]
    midpoints = (eta_starts + eta_ends) / 2.0
    n_intervals = len(midpoints)

    strip_segment_idx = jnp.searchsorted(jnp.array(segment_boundaries), midpoints, side="right") - 1
    strip_segment_idx = jnp.clip(strip_segment_idx, 0, len(wing.segments) - 1)

    # Escape CS handling if wing has no control surfaces

    if not hasattr(wing, "control_surfaces") or len(wing.control_surfaces) == 0:
        le_cuts = jnp.zeros(n_intervals)
        te_cuts = jnp.ones(n_intervals)
        le_ids = jnp.full(n_intervals, -1, dtype=jnp.int32)
        te_ids = jnp.full(n_intervals, -1, dtype=jnp.int32)
        return jnp.stack([eta_starts, eta_ends, le_cuts, te_cuts, le_ids, te_ids], axis=1), strip_segment_idx

    # Convert CS info to arrays
    cs_span_starts = jnp.array(cs_span_starts)
    cs_span_ends = jnp.array(cs_span_ends)
    cs_chord_starts = jnp.array([cs.chord_fraction_start for cs in wing.control_surfaces])
    cs_chord_ends = jnp.array([cs.chord_fraction_end for cs in wing.control_surfaces])

    # Find which control surfaces intersect each interval
    active_mask = (midpoints[:, None] > cs_span_starts[None, :]) & (midpoints[:, None] < cs_span_ends[None, :])

    # Find leading edge and trailing edge cuts
    # (Chord starts and ends are bound to LE/TE by WingControlSurface post_init validation)
    is_le_cs = cs_chord_starts == 0.0
    is_te_cs = cs_chord_ends == 1.0

    le_cuts = jnp.max(jnp.where(active_mask & is_le_cs[None, :], cs_chord_ends[None, :], 0.0), axis=1, initial=1.0)
    te_cuts = jnp.min(jnp.where(active_mask & is_te_cs[None, :], cs_chord_starts[None, :], 1.0), axis=1, initial=1.0)

    # Map the cuts to particular control surfaces
    le_active = active_mask & is_le_cs[None, :]
    te_active = active_mask & is_te_cs[None, :]

    le_id = jnp.where(jnp.any(le_active, axis=1), jnp.argmax(le_active, axis=1), -1)
    te_id = jnp.where(jnp.any(te_active, axis=1), jnp.argmax(te_active, axis=1), -1)

    return jnp.stack([eta_starts, eta_ends, le_cuts, te_cuts, le_id, te_id], axis=1), strip_segment_idx


def generate_spanwise_coordinates(intervals_data: jax.Array, n_sw: int, cosine_spacing: bool = False) -> jax.Array:
    """
    Generates piecewise spanwise coordinates (eta) guaranteeing breaks at the interval boundaries.

    Args:
        intervals_data: jax.Array of shape (N_intervals, 4) -> [eta_start, eta_end, le_cut, te_cut]
        n_sw: int, total number of spanwise panels requested.
        cosine_spacing: bool, whether to cluster panels at interval boundaries.

    Returns:
        eta_vertices: jax.Array of shape (n_sw + 1,)
    """
    n_intervals = intervals_data.shape[0]

    # Extract the bounds directly from the new interval structure
    eta_starts = intervals_data[:, 0]
    eta_ends = intervals_data[:, 1]

    # Safety catch: Ensure we have at least 1 panel per interval
    n_sw = jnp.maximum(n_sw, n_intervals)

    # 1. Proportional Allocation (Guaranteeing exactly n_sw total panels)
    widths = eta_ends - eta_starts
    cum_fractions = jnp.cumsum(widths) / jnp.sum(widths)

    # Subtract n_intervals to guarantee a baseline of 1 panel per interval
    n_sw_adj = n_sw - n_intervals
    cum_panels = jnp.round(cum_fractions * n_sw_adj).astype(int)

    # Retrieve the exact panels per interval and add the baseline 1 back
    panels_per_interval = jnp.diff(jnp.concatenate([jnp.array([0]), cum_panels])) + 1

    # 2. Global-to-Local Index Mapping
    cum_panels_adj = jnp.concatenate([jnp.array([0]), jnp.cumsum(panels_per_interval)])

    # Create the global vertex indices (0 to n_sw)
    vertex_indices = jnp.arange(n_sw + 1)

    # Find which interval each vertex belongs to
    interval_idx = jnp.searchsorted(cum_panels_adj, vertex_indices, side="right") - 1

    # Clip to prevent out-of-bounds on the very last vertex (n_sw)
    interval_idx = jnp.clip(interval_idx, 0, n_intervals - 1)

    # 3. Calculate local fractions
    local_i = vertex_indices - cum_panels_adj[interval_idx]
    n_local = panels_per_interval[interval_idx]

    # Linear fraction inside the interval (0.0 to 1.0)
    f_linear = local_i / n_local

    # Apply Cosine Spacing if requested
    f_spacing = jnp.where(cosine_spacing, 0.5 * (1.0 - jnp.cos(jnp.pi * f_linear)), f_linear)

    # 4. Map back to global eta coordinates
    interval_starts = eta_starts[interval_idx]
    interval_ends = eta_ends[interval_idx]

    eta_vertices = interval_starts + f_spacing * (interval_ends - interval_starts)

    interval_mapping = interval_idx[:-1]

    return eta_vertices, interval_mapping


def generate_chordwise_coordinates(le_cut: float, te_cut: float, n_cw: int, cosine_spacing: bool = False) -> jax.Array:
    """
    Generates piecewise chordwise coordinates (0.0 to 1.0) for a single strip.
    """
    # Define the 3 potential chordwise sections:
    breaks = jnp.array(
        [0.0, le_cut, te_cut, 1.0]
    )  # [Leading Edge -> le_cut], [le_cut -> te_cut], [te_cut -> Trailing Edge]
    widths = jnp.diff(breaks)  # Calculate physical widths of these sections
    n_cw = jnp.maximum(n_cw, 3)  # Safety catch

    # Proportional Allocation ()
    total_width = jnp.sum(widths)
    cum_fractions = jnp.cumsum(widths) / jnp.maximum(total_width, 1e-8)
    cum_panels = jnp.round(cum_fractions * n_cw).astype(
        int
    )  # We must distribute exactly n_c panels, rounding off 0-widths intervales
    panels_per_interval = jnp.diff(jnp.concatenate([jnp.array([0]), cum_panels]))

    # Global-to-Local Index Mapping
    cum_panels_adj = jnp.concatenate([jnp.array([0]), jnp.cumsum(panels_per_interval)])
    vertex_indices = jnp.arange(n_cw + 1)

    interval_idx = jnp.searchsorted(cum_panels_adj, vertex_indices, side="right") - 1
    interval_idx = jnp.clip(interval_idx, 0, 2)

    # Calculate local fractions
    local_i = vertex_indices - cum_panels_adj[interval_idx]
    n_local = panels_per_interval[interval_idx]
    f_linear = jnp.where(n_local > 0, local_i / jnp.maximum(n_local, 1), 0.0)

    # Chordwise cosine spacing is currently present, but unsupported
    f_spacing = jnp.where(cosine_spacing, 0.5 * (1.0 - jnp.cos(jnp.pi * f_linear)), f_linear)

    # Map back to global chord coordinates (0.0 to 1.0)
    interval_starts = breaks[interval_idx]
    interval_ends = breaks[interval_idx + 1]

    x_c_vertices = interval_starts + f_spacing * (interval_ends - interval_starts)

    return x_c_vertices


def calculate_macro_properties(wing, eta_vertices: jax.Array, semispan: float) -> tuple:
    """
    Vectorized lofting of the structural wing, directly evaluated at the computational grid.
    """
    # 1. Extract Structural Nodes (Assuming len(segments) == N_nodes)
    seg_etas = jnp.stack([seg.percent_span_location for seg in wing.segments])
    seg_c_fracs = jnp.stack([seg.root_chord_percent for seg in wing.segments])
    seg_twists = jnp.stack([seg.twist for seg in wing.segments])

    # Sweeps and dihedrals dictate the interval outboard of the node
    qc_sweeps = jnp.stack([seg.sweeps.quarter_chord for seg in wing.segments])[:-1]
    dihedrals = jnp.stack([seg.dihedral_outboard for seg in wing.segments])[:-1]

    # 2. Calculate Physical Geometry at the Nodes
    seg_Y = seg_etas * semispan
    seg_c = seg_c_fracs * wing.chords.root

    # Deltas between nodes
    dY = jnp.diff(seg_Y)
    dc = jnp.diff(seg_c)

    # 3. Vectorized Sweep & Dihedral projection
    dX_qc = dY * jnp.tan(qc_sweeps)
    dX_LE = dX_qc - 0.25 * dc  # Shift reference frame to LE
    dZ_LE = dY * jnp.tan(dihedrals)

    # 4. Cumulative sum to get actual 3D coordinates of the structural nodes
    node_X_LE = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dX_LE)])
    node_Z_LE = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dZ_LE)])

    # Map to individual strips.

    strip_X_LE = jnp.interp(eta_vertices, seg_etas, node_X_LE)
    strip_Y = jnp.interp(eta_vertices, seg_etas, seg_Y)
    strip_Z_LE = jnp.interp(eta_vertices, seg_etas, node_Z_LE)

    strip_c = jnp.interp(eta_vertices, seg_etas, seg_c)
    strip_twist = jnp.interp(eta_vertices, seg_etas, seg_twists)

    return strip_X_LE, strip_Y, strip_Z_LE, strip_c, strip_twist


def morph_to_3d_mesh(xi_grid, strip_X_LE, strip_Y, strip_Z_LE, strip_c, strip_twist):
    """
    Morphs the non-dimensional topological grid into a 3D VLM panel mesh.
    Returns an array of shape (n_sw, n_cw, 4, 3) containing the 4 corner vertices of every panel.
    """
    # 1. Extract Left (L) and Right (R) macroscopic boundaries for each strip
    # Shapes become (n_sw, 1) so they broadcast against the (n_sw, n_cw + 1) grids
    c_L = strip_c[:-1][:, None]
    c_R = strip_c[1:][:, None]

    twist_L = strip_twist[:-1][:, None]
    twist_R = strip_twist[1:][:, None]

    X_LE_L = strip_X_LE[:-1][:, None]
    X_LE_R = strip_X_LE[1:][:, None]

    Y_L = strip_Y[:-1][:, None]
    Y_R = strip_Y[1:][:, None]

    Z_LE_L = strip_Z_LE[:-1][:, None]
    Z_LE_R = strip_Z_LE[1:][:, None]

    # 2. Scale up to physical 2D coordinates (still relative to the Leading Edge pivot)
    # Shapes: (n_sw, n_cw + 1)
    x_2d_L = xi_grid * c_L
    z_2d_L = jnp.zeros_like(x_2d_L)

    x_2d_R = xi_grid * c_R
    z_2d_R = jnp.zeros_like(x_2d_R)

    # 3. Apply Twist Rotation (Pitching around the Leading Edge pivot)
    x_rot_L = x_2d_L * jnp.cos(twist_L) + z_2d_L * jnp.sin(twist_L)
    z_rot_L = -x_2d_L * jnp.sin(twist_L) + z_2d_L * jnp.cos(twist_L)

    x_rot_R = x_2d_R * jnp.cos(twist_R) + z_2d_R * jnp.sin(twist_R)
    z_rot_R = -x_2d_R * jnp.sin(twist_R) + z_2d_R * jnp.cos(twist_R)

    # 4. Translate to the 3D Swept/Dihedraled Space
    # These contain the exact 3D coordinates for every chordwise vertex line
    X_3D_L = X_LE_L + x_rot_L
    Y_3D_L = jnp.broadcast_to(Y_L, X_3D_L.shape)  # Y is constant along the chord
    Z_3D_L = Z_LE_L + z_rot_L

    X_3D_R = X_LE_R + x_rot_R
    Y_3D_R = jnp.broadcast_to(Y_R, X_3D_R.shape)
    Z_3D_R = Z_LE_R + z_rot_R

    # 5. Assemble the Panels
    # Stack the coordinates into (X, Y, Z) points -> Shape: (n_sw, n_cw + 1, 3)
    verts_L = jnp.stack([X_3D_L, Y_3D_L, Z_3D_L], axis=-1)
    verts_R = jnp.stack([X_3D_R, Y_3D_R, Z_3D_R], axis=-1)

    # Slice them to define the 4 corners of each panel (Front to Back)
    front_left = verts_L[:, :-1, :]
    back_left = verts_L[:, 1:, :]
    back_right = verts_R[:, 1:, :]
    front_right = verts_R[:, :-1, :]

    # Stack into final panel array.
    # Counter-clockwise ordering ensures normal vectors point UP (Right Hand Rule)
    panel_vertices = jnp.stack([front_left, back_left, back_right, front_right], axis=2)

    return panel_vertices

def generate_topology(state: State, system: Aircraft, settings: Settings):

    VD_list = []

    # Reformat original wings to have at least 2 segments and additional values for processing later
    for wing_idx, wing in enumerate(system.wings):  # type: ignore
        wing: Wing
        if len(wing.segments) == 0:
            raise ValueError(
                f"Found wing'{wing.name}' with no segments defined. \
                    Define segments manually or run wing.generate_segments or wing.update_geometry."
            )
        else:
            # TODO: Add support for All_Moving_Surface class
            for segment in wing.segments:
                if len(segment.control_surfaces) > 0:
                    raise ValueError(
                        f"Found control surfaces on segment '{segment.name}' of wing '{wing.name}'. \
                                     Control surfaces must be attributes of the wing itself."
                    )

        # Non-Dimensional Panelization ---------------------------------------------------------------------------------

        interval_data, strip_interval_map = find_intervals(wing)

        try:
            n_sw = vlm_settings.vortices.wings_n_spanwise[wing_idx]
            n_cw = vlm_settings.vortices.wings_n_chordwise[wing_idx]
        except TypeError:
            n_sw = vlm_settings.vortices.wings_n_spanwise
            n_cw = vlm_settings.vortices.wings_n_chordwise

        if len(interval_data) > n_sw or n_cw < 3:  # type: ignore
            warnings.warn(
                f"Specified number of wing vortices ({n_sw}, {n_cw}) "
                f"is less than the required spanwise breaks ({len(interval_data)}). "
                f"Increasing number of wing spanwise vortices to prevent mesh collapse."
            )  # Handled in generation functions below

        # Calculate strip eta (non-dimensional y-coordinate) (Shape: (n_sw +1,))
        eta, strip_interval_map = generate_spanwise_coordinates(
            interval_data, n_sw, vlm_settings.vortices.spanwise_cosine
        )

        # Calculate strip xi (non-dimensional x-coordinate) (Shape: (n_sw, n_cw + 1))
        strip_le_cuts = interval_data[:, 2][strip_interval_map]
        strip_te_cuts = interval_data[:, 3][strip_interval_map]

        vmap_chordwise = jax.vmap(generate_chordwise_coordinates, in_axes=(0, 0, None, None))
        xi_grid = vmap_chordwise(
            strip_le_cuts, strip_te_cuts, n_cw, False
        )  # Force linear chordwise spacing for Pistolesi's theorem (assumed in coefficient integration)

        # Map panels to control surfaces
        xi_mid = (xi_grid[:, :-1] + xi_grid[:, 1:]) / 2.0
        strip_le_ids = interval_data[:, 4][strip_interval_map]
        strip_te_ids = interval_data[:, 5][strip_interval_map]

        panel_cs_id = jnp.full_like(
            xi_mid, -1, dtype=jnp.int32
        )  # Default to -1 to indicate panel belongs to wing itself
        panel_cs_id = jnp.where(
            xi_mid < strip_le_cuts[:, None], strip_le_ids[:, None], panel_cs_id
        )  # If xi < LE cut, assign local LE CS ID
        panel_cs_id = jnp.where(
            xi_mid > strip_te_cuts[:, None], strip_te_ids[:, None], panel_cs_id
        )  # If xi > TE cut, assign local TE CS ID

        # Geometric Corrections ----------------------------------------------------------------------------------------

        # Calculate strip zeta (non-dimensional z-coordinate) (Shape: (n_sw, n_cw + 1))
        n_af_pts = validate_airfoil_resolutions(wing)
        flat_x = jnp.linspace(0.0, 1.0, n_af_pts // 2)
        flat_z = jnp.zeros(n_af_pts // 2)

        seg_camber_x = jnp.stack(
            [seg.airfoil.x_lower_surface if getattr(seg, "airfoil", None) else flat_x for seg in wing.segments]
        )  # type: ignore

        seg_camber_z = jnp.stack(
            [seg.airfoil.camber if getattr(seg, "airfoil", None) else flat_z for seg in wing.segments]
        )  # type: ignore

        seg_wedge_angle = jnp.stack(
            [seg.airfoil.wedge_angle if getattr(seg, "airfoil", None) else 0.0 for seg in wing.segments]
        )  # type: ignore

        strip_camber_x = seg_camber_x[strip_interval_map]
        strip_camber_z = seg_camber_z[strip_interval_map]
        strip_wedge_angle = seg_wedge_angle[strip_interval_map]

        xi_colloc = 0.25 * xi_grid[:, :-1] + 0.75 * xi_grid[:, 1:]

        vmap_interp = jax.vmap(jnp.interp, in_axes=(0, 0, 0))

        # Finite difference local camber slope/incidence angle
        zeta_fwd = vmap_interp(xi_colloc + 1e-4, strip_camber_x, strip_camber_z)
        zeta_bwd = vmap_interp(xi_colloc - 1e-4, strip_camber_x, strip_camber_z)

        camber_slopes = (zeta_fwd - zeta_bwd) / 2e-4

        # Calculate strip macro-level properties
        semispan = wing.spans.projected / 2.0 if wing.symmetric else wing.spans.projected
        strip_X_LE, strip_Y, strip_Z_LE, strip_c, strip_twist = calculate_macro_properties(wing, eta, semispan)

        morph_results = morph_to_3d_mesh(xi_grid, strip_X_LE, strip_Y, strip_Z_LE, strip_c, strip_twist)

        if wing.vertical:
            y_coords = morph_results[:, :, :, 1]
            z_coords = morph_results[:, :, :, 2]

            morph_results = morph_results.at[:, :, :, 1].set(z_coords)
            morph_results = morph_results.at[:, :, :, 2].set(y_coords)

        # Flatten and pack into VortexDistribution ---------------------------------------------------------------------
        flat_vertices = (morph_results + wing.origin).reshape(-1, 4, 3)

        VD = VortexDistribution(
            panel_vertices=flat_vertices,
            camber_slopes=camber_slopes.reshape(-1),
            wedge_angles=strip_wedge_angle.reshape(-1),
            surface_id=jnp.full(flat_vertices.shape[0], wing_idx, dtype=jnp.int32),
            control_surface_id=panel_cs_id.reshape(-1),
            is_leading_edge=jnp.zeros_like(xi_mid, dtype=bool).at[:, 0].set(True).reshape(-1),
            is_trailing_edge=jnp.zeros_like(xi_mid, dtype=bool).at[:, -1].set(True).reshape(-1),
        )

        VD_list.append(VD)

        if wing.symmetric:
            VD_list.append(mirror_distribution(VD))

        return VD_list

@io.inputs(
    "settings.analysis.aerodynamics: VLMSettings",
    "settings.analysis.aerodynamics.discretize_control_surfaces",
    "settings.analysis.aerodynamics.vortices.wing_spanwise_vortices",
    "settings.analysis.aerodynamics.vortices.wing_chordwise_vortices",
    "system.wings",
)
@io.outputs("system.analysis_data['vortex_distribution']", "settings.analysis.aerodynamics.vortices.chordwise_cosine")
def discretize_surfaces(state: State, system: "Aircraft", settings: Settings):

    # Pre-Processing ---------------------------------------------------------------------------------------------------

    # Unpacking
    vlm_settings: VORJAXSettings = settings.analysis.aerodynamics  # type: ignore
    updated_system = system
    VD_list = []

    VD_list = generate_topology(state, system, settings)

    full_VD = merge_vortex_distributions(VD_list)

    updated_analysis_data = system.analysis_data | {"vortex_distribution": full_VD}

    updated_system = update(updated_system, "analysis_data", updated_analysis_data)

    updated_settings = update(settings, "analysis.aerodynamics", vlm_settings)

    return state, updated_system, updated_settings


# ----------------------------------------------------------------------------------------------------------------------
#  Freestream Checking
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs(
    "state.frames.inertial.velocity_vector",
)
@io.outputs(
    "state.frames.inertial.velocity_vector",
    "state.freestream.speed",
)
def check_freestream(state: State, system: Aircraft, settings: Settings):
    """Unpacks PyTrees, calls pure math, repacks PyTrees."""

    velocity = state.frames.inertial.velocity_vector

    safe_velocity = jnp.where(velocity == 0.0, 1e-6, velocity)
    safe_speed = jnp.linalg.norm(safe_velocity, axis=-1, keepdims=True)

    current_state = update(
        state,
        (
            ("frames.inertial.velocity_vector", safe_velocity),
            ("freestream.speed", safe_speed),
        ),
    )

    return current_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Boundary Conditions (Vortex Strength Right Hand Side Matrix)
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs(
    "settings.analysis.aerodynamics: VLMSettings",
    "system.analysis_data['vortex_distribution']",
    "state.aerodynamics.angles.alpha",
    "state.aerodynamics.angles.beta",
    "state.freestream.speed",
    "state.stability.static.roll_rate",
    "state.stability.static.pitch_rate",
    "state.stability.static.yaw_rate",
)
@io.outputs(
    "system.analysis_data['boundary_conditions']",
    "system.analysis_data['relative_velocity']",
)
def compute_boundary_conditions(state: State, system: Aircraft, settings: Settings):
    """
    Computes the Neumann boundary condition (RHS) for the VLM.
    RHS = (V_freestream + V_rotation + V_wake) @ n
    """

    vlm_settings: "VORJAXSettings" = settings.analysis.aerodynamics  # type: ignore
    VD: VortexDistribution = system.analysis_data["vortex_distribution"]  # type: ignore

    # Extract State Conditions
    alpha = state.aerodynamics.angles.alpha
    beta = state.aerodynamics.angles.beta
    v_inf = state.freestream.speed

    p = state.stability.static.roll_rate
    q = state.stability.static.pitch_rate
    r = state.stability.static.yaw_rate

    # Squeeze to drop the dummy dimension from the state vectors,
    # transpose to prepare for cross product -> (n_time, 3)
    omega = jnp.concatenate([p, q, r], axis=1)

    # Build Freestream Velocity Vector (Wind Speed in Body Frame)
    v_fs = v_inf * jnp.concatenate(
        [jnp.cos(alpha) * jnp.cos(beta), jnp.sin(beta), jnp.sin(alpha) * jnp.cos(beta)], axis=1
    )

    # Compute Rotational Velocity at every control point: V_rot = -(Omega x r)
    moment_center = system.reference_geometry.center_of_gravity
    r_giro = VD.collocation_points - moment_center

    v_rot = -jnp.cross(omega[:, None, :], r_giro[None, :, :])

    # Sum the total relative velocity (N, 3)
    v_total = v_fs[:, None, :] + v_rot
    if vlm_settings.model_propeller_wake:
        # TODO: Convert BEMT and add wake calculation to VLM Process
        raise ValueError("Propeller wake modelling is unsupported pending BEMT inclusion in Flowtangent.")
        v_total = v_total + system.analysis_data["induced_wake"]  # type: ignore

    # Take the Dot Product with the Panel Normals
    # Normalize by V_inf to match the standard VLM coefficient formulation
    v_unit = v_total / v_inf[:, None]

    # Dot product: -sum(V * N, axis=1)
    base_rhs_array = -jnp.sum(v_unit * VD.normal_vectors, axis=-1)

    # Camber correction from thin wing assumption
    v_unit_x = v_unit[..., 0]
    rhs_array = base_rhs_array + (v_unit_x * VD.camber_slopes)

    updated_analysis_data = system.analysis_data | {
        "boundary_conditions": rhs_array,
        "relative_velocity": v_total,
    }

    updated_system = update(system, "analysis_data", updated_analysis_data)

    return state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


@jax.jit
def subsonic_induction(z, x1_sq, r_o1, x2_sq, r_o2, x_ty, t, B_sq, z_sq, tol_sq, x1, y1, x2, y2, r_tv1, r_tv2):
    """
    Pure JAX translation of the VORLAX subsonic Biot-Savart induction.

    This kernel computes the induced velocity (U, V, W) at a collocation point
    due to a single swept horseshoe vortex, applying local compressibility scaling.

    Variable Glossary (Miranda-Elliott-Baker Local Swept Coordinate System):
    -------------------------------------------------------------------------
    z               : Vertical distance from the collocation point to the vortex plane.
    x1, x2          : Streamwise distances from the collocation pt to vortex endpoints 1 and 2.
    y1, y2          : Spanwise distances from the collocation pt to vortex endpoints 1 and 2.
    x_sq1, x_sq2    : Squared streamwise distances (X1^2, X2^2).
    r_tv1, r_tv2    : Squared transverse distances (Y^2 + Z^2) to endpoints.
    beta_sq         : Compressibility factor (M^2 - 1). Negative in subsonic flow.
    r_o1, r_o2      : Compressibility-scaled transverse distances (B2 * RTV).
    r1, r2          : "Effective" compressible distances to endpoints sqrt(X^2 - B^2 * RTV^2).
    t               : Tangent of the bound vortex sweep angle.
    x_ty            : Cross-term projection mapping the distance along the swept vortex line.
    tol_sq          : Squared singularity tolerance (prevents div-by-zero near the filament).
    F_b2, F_b2      : Bound vortex influence terms.
    F_t1, F_t2      : Trailing vortex influence terms.
    """
    C_pi = 4.0 * jnp.pi

    # 1. Effective Compressible Distances
    # Using 1e-16 prevents exact 0.0 which would cause NaN gradients in downstream divisions
    r1 = jnp.sqrt(jnp.maximum(x1_sq - r_o1, 1e-16))
    r2 = jnp.sqrt(jnp.maximum(x2_sq - r_o2, 1e-16))

    # 2. Bound Vortex Denominator
    t_Bz = (jnp.square(t) - B_sq) * z_sq
    safe_denom = jnp.maximum(jnp.square(x_ty) + t_Bz, tol_sq)

    # 3. DRY Helper Function with NaN-safe division
    def calc_F(x, y, r, r_tv):
        # F_b: Influence contribution from the bound (swept) segment
        F_b = (t * x - B_sq * y) / r

        # F_t: Influence contribution from the semi-infinite trailing leg
        # safe_denom prevents divide-by-zero in the unselected jnp.where branch
        safe_denom = jnp.where(r_tv < tol_sq, 1.0, r * r_tv)
        F_t = jnp.where(r_tv < tol_sq, 0.0, (x + r) / safe_denom)

        return F_b, F_t

    # Evaluate for Endpoint 1 (Left/A) and Endpoint 2 (Right/B)
    F_b1, F_t1 = calc_F(x1, y1, r1, r_tv1)
    F_b2, F_t2 = calc_F(x2, y2, r2, r_tv2)

    # 4. Final Velocity Assembly
    Q_b = (F_b1 - F_b2) / safe_denom
    z_pi = z / C_pi

    # U: Streamwise induced velocity (Perturbation velocity)
    U = jnp.where(z_sq < tol_sq, 0.0, z_pi * Q_b)

    # V: Spanwise induced velocity (Sidewash)
    V = jnp.where(z_sq < tol_sq, 0.0, z_pi * (F_t1 - F_t2 - Q_b * t))

    # W: Normal induced velocity (Downwash)
    W = -(Q_b * x_ty + F_t1 * y1 - F_t2 * y2) / C_pi

    return U, V, W


@jax.jit
def supersonic_in_plane(r1, r2, y1, y2, tol, x_ty, C_pi):
    """
    Pure JAX translation of the in-plane supersonic induction.
    Evaluates downwash analytically when the collocation point sits exactly
    in the Z=0 plane of the vortex (where RTV -> 0).
    """
    # AD-Safe Denominators (Prevents NaN gradients in unselected branches)
    safe_Y1 = jnp.where(jnp.abs(y1) > tol, y1, 1.0)
    safe_Y2 = jnp.where(jnp.abs(y2) > tol, y2, 1.0)
    safe_XTY = jnp.where(jnp.abs(x_ty) > tol, x_ty, 1.0)

    F1 = jnp.where(jnp.abs(y1) > tol, r1 / safe_Y1, 0.0)
    F2 = jnp.where(jnp.abs(y2) > tol, r2 / safe_Y2, 0.0)

    W_in = jnp.where(jnp.abs(x_ty) > tol, (-F1 + F2) / (safe_XTY * C_pi), 0.0)
    return W_in


@jax.jit
def supersonic_induction(
    z,
    x_sq1,
    r_o1,
    x_sq2,
    r_o2,
    x_ty,
    t,
    B_sq,
    z_sq,
    tol_sq,
    tol,
    tol_sq2,
    x1,
    y1,
    x2,
    y2,
    r_tv1,
    r_tv2,
    c,
    sonic_mask,
    recv_idx,
):
    """
    Pure JAX translation of the VORLAX supersonic Biot-Savart induction.

    Variable Glossary (Supersonic Additions):
    -------------------------------------------------------------------------
    cutoff     : Defines the boundary of the Mach cone interaction.
    reps       : Mach cone proximity threshold.
    valid1/2   : Boolean masks. True if the point lies inside the downstream Mach cone.
    WWAVE      : The Principal Part of the singular integral. Represents the 2D wave
                 drag contribution of the panel on itself (self-induction).
    T2A / T2F  : Aft and Forward panel sweep tangents, used to detect sonic edges.
    TRANS      : Edge condition parameter. If TRANS < 0, the edge is "sonic"
                 (sweep angle exactly matches the Mach angle).
    RFLAG      : Subsonic/Supersonic leading edge flag used downstream for LE suction.
    sonic_mask : Identifies panels exhibiting mathematical singularities at Mach=sec(sweep).
    """
    C_pi = 2.0 * jnp.pi
    t_sq = jnp.square(t)
    z_pi = z / C_pi
    cutoff = 0.8

    # Mach Cone Distances (Real only inside the cone)
    r1 = jnp.where(x_sq1 > r_o1, jnp.sqrt(jnp.maximum(x_sq1 - r_o1, 1e-16)), 0.0)
    r2 = jnp.where(x_sq2 > r_o2, jnp.sqrt(jnp.maximum(x_sq2 - r_o2, 1e-16)), 0.0)

    # Denominator Setup
    safe_denom = jnp.square(x_ty) + (t_sq - B_sq) * z_sq
    sgn = jnp.where(safe_denom < 0, -1.0, 1.0)
    safe_denom = jnp.where(jnp.abs(safe_denom) < tol_sq, sgn * tol_sq, safe_denom)

    def calc_F(x, y, x_sq, r_o, r, r_tv):
        reps = cutoff * x_sq
        valid = (x >= tol) & (r != 0.0) & (r_o <= reps) & (r_tv >= tol_sq)

        # AD-Safe denominators (only applied when 'valid' is True)
        safe_r = jnp.where(valid, r, 1.0)
        safe_rr_tv = jnp.where(valid, r * r_tv, 1.0)

        # 1.0 fallback is mathematically required by VORLAX supersonic integration
        F_b = jnp.where(valid, (t * x - B_sq * y) / safe_r, 1.0)
        F_t = jnp.where(valid, x / safe_rr_tv, 1.0)

        return F_b, F_t

    F_b1, F_t1 = calc_F(x1, y1, x_sq1, r_o1, r1, r_tv1)
    F_b2, F_t2 = calc_F(x2, y2, x_sq2, r_o2, r2, r_tv2)

    # Global Velocity Assembly
    Q_b = (F_b1 - F_b2) / safe_denom
    U = z_pi * Q_b
    V = z_pi * (F_t1 - F_t2 - Q_b * t)
    W = -(Q_b * x_ty + F_t1 * y1 - F_t2 * y2) / C_pi

    # In-Plane Singularity Override
    in_plane = z_sq < tol_sq2
    W_in = supersonic_in_plane(r1, r2, y1, y2, tol, x_ty, C_pi)

    U = jnp.where(in_plane, 0.0, U)
    V = jnp.where(in_plane, 0.0, V)
    W = jnp.where(in_plane, W_in, W)

    # W_wave: Principal Part of the Integral (Self-Influence / Wave Drag)
    N = U.shape[1]
    t_sq = t**2

    W_wave_cond = B_sq > t_sq[None, :]
    W_wave_output = -0.5 * jnp.sqrt(jnp.where(W_wave_cond, B_sq - t_sq[None, :], 1.0)) / jnp.maximum(c, 1e-12)

    # Calculate W_wave for all senders (Shape: n_time, N)
    W_wave_val = jnp.where(W_wave_cond, W_wave_output, 0.0)

    # Create a boolean mask for the diagonal element of this specific row
    j_indices = jnp.arange(N)
    is_diag = (j_indices == recv_idx)[None, :]

    # Only add the W_wave value to the element where sender == receiver
    W = W + jnp.where(is_diag, W_wave_val, 0.0)

    # Build the 1D slice of the Laplacian stencil for THIS receiver row
    sonic_row = jnp.where(
        j_indices == recv_idx,
        2.0,
        jnp.where(j_indices == recv_idx - 1, -1.0, jnp.where(j_indices == recv_idx + 1, -1.0, 0.0)),
    )[None, :]

    is_recv_sonic = sonic_mask[:, recv_idx][:, None]

    # Overwrite the influence of sonic sending panels with the smoothing stencil
    W = jnp.where(is_recv_sonic, sonic_row, W)

    return U, V, W


@jax.jit
@jax.checkpoint
def compute_C_ij(VD, Mach):
    """
    Computes the Aerodynamic Influence Coefficient matrix C_ij.
    Output Shape: (n_time, N, N, 3)
    """

    # Unpack Vortex Distribution Data ----------------------------------------------------------------------------------
    vortex_A = VD.bound_vortex_A.astype(jnp.float32)
    vortex_B = VD.bound_vortex_B.astype(jnp.float32)
    center = VD.bound_vortex_center.astype(jnp.float32)
    colloc = VD.collocation_points.astype(jnp.float32)

    # Local Panel Orientation ------------------------------------------------------------------------------------------
    dy = vortex_B[:, 1] - vortex_A[:, 1]
    dz = vortex_B[:, 2] - vortex_A[:, 2]

    norm_yz = jnp.maximum(jnp.sqrt(dy**2 + dz**2), 1e-16)
    costheta = dy / norm_yz
    sintheta = dz / norm_yz

    # Local Panel Sweep ------------------------------------------------------------------------------------------------
    dx_vortex = vortex_B[:, 0] - center[:, 0]
    dy_vortex = (vortex_B[:, 1] - center[:, 1]) * costheta + (vortex_B[:, 2] - center[:, 2]) * sintheta

    # s = local half-span, t = tangent of the sweep angle
    s = jnp.abs(dy_vortex)
    t = dx_vortex / jnp.maximum(dy_vortex, 1e-16)
    t_sq = t**2

    # Beta-Squared = Mach^2 - 1.0 --------------------------------------------------------------------------------------
    beta_sq = (Mach.squeeze(1) ** 2 - 1.0).astype(jnp.float32)
    beta_sq_exp = beta_sq[:, None] if beta_sq.ndim == 1 else beta_sq
    is_subsonic = beta_sq_exp < 0.0

    # Sonic Mask Pre-Calc ----------------------------------------------------------------------------------------------
    t_sq_fore = jnp.where(VD.is_leading_edge, 0.0, jnp.roll(t_sq, shift=1))
    t_sq_aft = jnp.where(VD.is_trailing_edge, 0.0, jnp.roll(t_sq, shift=-1))

    sonic_check = (beta_sq_exp - t_sq_fore[None, :]) * (beta_sq_exp - t_sq_aft[None, :])
    sonic_mask = (sonic_check < 0) & VD.is_leading_edge

    # Check for singularity (Mach cone passes through panel)
    singularity_flag = jnp.where(sonic_mask, 0, 1)
    singularity_flag = jnp.where(is_subsonic, 1, singularity_flag)

    safe_beta_sq = jnp.maximum(t_sq_fore[None, :], t_sq_aft[None, :]) + 0.01
    beta_sq_exp = jnp.where(sonic_mask, safe_beta_sq, beta_sq_exp)

    # C_mn Calculation -------------------------------------------------------------------------------------------------

    tol = s / 500.0
    tol_sq = tol**2
    tol_sq_scl = 2500.0 * tol_sq

    # Row-wise vector-mapping to minimize peak memory usage
    def compute_row(c_pt, ct_R, st_R, recv_idx):
        dx = c_pt[0] - center[:, 0]
        dy = c_pt[1] - center[:, 1]
        dz = c_pt[2] - center[:, 2]

        y_dist = dy * costheta + dz * sintheta
        z_dist = -dy * sintheta + dz * costheta

        x_dist_left = dx + t * s
        x_dist_right = dx - t * s
        x_dist_center = dx - t * y_dist

        y_dist_left = y_dist + s
        y_dist_right = y_dist - s

        # Arrays are (N,) instead of (N, N)
        x_sq1 = x_dist_left**2
        x_sq2 = x_dist_right**2
        y_sq1 = y_dist_left**2
        y_sq2 = y_dist_right**2
        z_sq = z_dist**2

        r_tv1 = y_sq1 + z_sq
        r_tv2 = y_sq2 + z_sq

        # Broadcast the Mach/Time dimension here: (n_time, 1) * (N,) -> (n_time, N)
        r_o1 = beta_sq_exp * r_tv1[None, :]
        r_o2 = beta_sq_exp * r_tv2[None, :]

        # --- Subsonic Kernel ---
        U_ind, V_ind, W_ind = subsonic_induction(
            x1_sq=x_sq1,
            x2_sq=x_sq2,
            x_ty=x_dist_center,
            x1=x_dist_left,
            x2=x_dist_right,
            y1=y_dist_left,
            y2=y_dist_right,
            z=z_dist,
            z_sq=z_sq,
            r_tv1=r_tv1,
            r_tv2=r_tv2,
            r_o1=r_o1,
            r_o2=r_o2,
            t=t,
            B_sq=beta_sq_exp,
            tol_sq=tol_sq,
        )

        # --- Supersonic Kernel ---
        U_sup, V_sup, W_sup = supersonic_induction(
            x_sq1=x_sq1,
            x_sq2=x_sq2,
            x_ty=x_dist_center,
            x1=x_dist_left,
            x2=x_dist_right,
            y1=y_dist_left,
            y2=y_dist_right,
            z=z_dist,
            z_sq=z_sq,
            r_tv1=r_tv1,
            r_tv2=r_tv2,
            r_o1=r_o1,
            r_o2=r_o2,
            t=t,
            B_sq=beta_sq_exp,
            tol=tol,
            tol_sq=tol_sq,
            tol_sq2=tol_sq_scl,
            c=VD.chord_lengths,
            sonic_mask=sonic_mask,
            recv_idx=recv_idx,
        )

        # --- Blending ---
        U_ind = jnp.where(is_subsonic, U_ind, U_sup)
        V_ind = jnp.where(is_subsonic, V_ind, V_sup)
        W_ind = jnp.where(is_subsonic, W_ind, W_sup)

        # --- EW Calculation ---
        # Note: ct_S and st_S are just the global 'costheta' and 'sintheta' arrays
        COS_RS = ct_R * costheta + st_R * sintheta
        SIN_RS = st_R * costheta - ct_R * sintheta

        EW_row = W_ind * COS_RS[None, :] - V_ind * SIN_RS[None, :]

        # --- Rotate to Global Frame ---
        C_ij_row = jnp.stack(
            [
                U_ind,
                V_ind * costheta[None, :] - W_ind * sintheta[None, :],
                V_ind * sintheta[None, :] + W_ind * costheta[None, :],
            ],
            axis=-1,
        )

        # Return the tuple
        return C_ij_row, EW_row

    C_ij, _ = jax.vmap(compute_row, out_axes=(1, 0))(colloc, costheta, sintheta, jnp.arange(VD.total_panels))
    # C_mn = jnp.swapaxes(C_ij_mapped, 0, 1)

    # If using chordwise cosine spacing, compute leading edge normalwash for Lan's method
    # (Currently unsupported, commented out to minimize memory footprint)

    # front_left = VD.panel_vertices[:, 0, :]
    # front_right = VD.panel_vertices[:, 3, :]
    # front_mid = 0.5 * (front_left + front_right)

    # _, LN_mapped = jax.vmap(compute_row)(front_mid, costheta, sintheta, jnp.arange(VD.total_panels))
    # LN = jnp.swapaxes(LN_mapped, 0, 1)

    return C_ij.astype(jnp.float64), singularity_flag


# ----------------------------------------------------------------------------------------------------------------------
#  Wing Induced Velocity Calculation
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs("system.analysis_data['vortex_distribution']", "state.freestream.mach_number")
@io.outputs(
    "system.analysis_data['VICs']", "system.analysis_data['singularities']", "system.analysis_data['le_normalwash']"
)
def compute_induced_velocity(state: State, system: Aircraft, settings: Settings):

    VD = system.analysis_data["vortex_distribution"]
    Mach = state.freestream.mach_number

    (
        C_ij,
        singularity_flag,
    ) = compute_C_ij(VD, Mach)

    updated_analysis_data = system.analysis_data | {
        "VICs": C_ij,
        "singularities": singularity_flag,
    }

    updated_system = update(system, "analysis_data", updated_analysis_data)

    return state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Compute VLM Vortex Strength
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs(
    "system.analysis_data['vortex_distribution']",
    "system.analysis_data['VICs']",
    "system.analysis_data['boundary_conditions']",
    "system.analysis_data['singularities']",
)
@io.outputs(
    "system.analysis_data['vortex_strengths']",
)
def compute_vortex_strength(state: State, system: Aircraft, settings: Settings):
    """Solves the linear system A * GAMMA = RHS for the vortex strengths."""

    analysis: dict[str, jax.Array] = system.analysis_data
    VD = analysis["vortex_distribution"]

    # Extract the arrays we built in previous steps
    # C_mn shape: (n_time, receiver_N, sender_N, 3)
    C_mn = analysis["VICs"]

    # RHS shape: (n_time, receiver_N)
    RHS = analysis["boundary_conditions"]

    # RFLAG shape: (n_time, receiver_N)
    singularity_flag = analysis["singularities"]

    # Zero out the RHS for supersonic panels swept parallel to the Mach cone
    RHS = RHS * singularity_flag

    # Build the 'A' matrix via Dot Product: sum(C_mn * n)
    # The normal vector belongs to the RECEIVING panel (dim 1).
    # We broadcast it over n_time (dim 0) and the sending panels (dim 2).
    # VD.normal_vectors shape: (N, 3) -> Broadcast to (1, N, 1, 3)
    normals_broadcast = VD.normal_vectors[None, :, None, :]

    # A shape: (n_time, receiver_N, sender_N)
    A = jnp.sum(C_mn * normals_broadcast, axis=-1)

    # Solve the linear system
    # A is (n_time, N, N), RHS is (n_time, N)
    # Output GAMMA is perfectly shaped as (n_time, N)
    GAMMA = jnp.linalg.solve(A, RHS[..., None]).squeeze(-1)

    # Pack the results
    updated_analysis_data = analysis | {"vortex_strengths": GAMMA}

    updated_system = update(system, "analysis_data", updated_analysis_data)

    return state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Apply Aerodynamic Forces
# ----------------------------------------------------------------------------------------------------------------------


@io.inputs(
    "state.aerodynamics.coefficients.lift.total",
    "state.aerodynamics.coefficients.drag.total",
    "state.freestream.density",
    "state.freestream.speed",
    "system.areas.reference",
)
@io.outputs("state.frames.wind.total_force_vector")
def apply_aerodynamic_forces(state: State, system: Aircraft, settings: Settings):

    # Get coefficients from analysis
    C_L = state.aerodynamics.coefficients.lift.total
    C_D = state.aerodynamics.coefficients.drag.total

    rho = state.freestream.density
    flight_speed = state.freestream.speed
    S = system.areas.reference

    qS = 0.5 * rho * (flight_speed**2) * S

    F_Z = -qS * C_L  # Z negative by right hand rule convention
    F_X = qS * C_D

    wind_forces = state.frames.wind.total_force_vector
    wind_forces = wind_forces.at[:, 2].set(F_Z.flatten())
    wind_forces = wind_forces.at[:, 0].set(F_X.flatten())

    state = update(state, "frames.wind.total_force_vector", wind_forces)

    return state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Compute VLM Pressure Coefficients
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@jax.jit
def compute_pressure_coefficients(VD, v_total, Gamma, v_inf):
    """
    Computes the differential pressure coefficient (Delta C_P) for all panels.
    Input shapes: v_total (n_time, N, 3), GAMMA (n_time, N)
    """
    # Local Velocity Components (normalized by V_inf) ------------------------------------------------------------------
    Vx_local = v_total[:, :, 0] / v_inf
    Vy_local = v_total[:, :, 1] / v_inf

    # Local Panel Geometry (Sweep Tangents and Dihedral) ---------------------------------------------------------------
    dx = VD.chord_lengths
    strip_ids = jnp.cumsum(VD.is_leading_edge) - 1
    strip_chord_array = jax.ops.segment_sum(dx, strip_ids, num_segments=VD.total_strips)
    strip_chord = strip_chord_array[strip_ids]

    # Front edge sweep (Front-Left [0] to Front-Right [3])
    dx_A = VD.panel_vertices[:, 3, 0] - VD.panel_vertices[:, 0, 0]
    dy_A = VD.panel_vertices[:, 3, 1] - VD.panel_vertices[:, 0, 1]
    dz_A = VD.panel_vertices[:, 3, 2] - VD.panel_vertices[:, 0, 2]
    dy_z_A = jnp.maximum(jnp.sqrt(dy_A**2 + dz_A**2), 1e-12)  # Prevent DivByZero

    tan_A = dx_A / dy_z_A
    cos_DL = dy_A / dy_z_A  # Cosine of local dihedral

    # Back edge sweep (Back-Left [1] to Back-Right [2])
    dx_B = VD.panel_vertices[:, 2, 0] - VD.panel_vertices[:, 1, 0]
    dy_B = VD.panel_vertices[:, 2, 1] - VD.panel_vertices[:, 1, 1]
    dz_B = VD.panel_vertices[:, 2, 2] - VD.panel_vertices[:, 1, 2]
    dy_z_B = jnp.maximum(jnp.sqrt(dy_B**2 + dz_B**2), 1e-12)

    tan_B = dx_B / dy_z_B

    # Helmholtz' theorem integration of anterior circulation from shed vortices ----------------------------------------

    def scan_fn(a, b):
        # a and b are tuples: (value, is_leading_edge_flag)
        v1, le1 = a
        v2, le2 = b
        # If element 'b' is a leading edge, it resets the sum to just v2
        return jnp.where(le2, v2, v1 + v2), le1 | le2

    gamma_over_c = Gamma / strip_chord[None, :]  # Circulation per unit chord
    is_le = jnp.broadcast_to(
        VD.is_leading_edge[None, :], gamma_over_c.shape
    )  # Broadcast the 1D LE flag to match the (n_time, N) matrix
    gamma_anterior, _ = jax.lax.associative_scan(
        scan_fn, (gamma_over_c, is_le), axis=1
    )  # Associative scan w/ binary switch on leading edge
    Gamma_anterior = jnp.where(is_le, 0.0, jnp.roll(gamma_anterior, shift=1, axis=1))

    # Sweep / Sideslip Correction --------------------------------------------------------------------------------------
    Gamma_lateral = Gamma_anterior * (tan_A - tan_B)[None, :] - gamma_over_c * tan_B[None, :]
    dCp_sideslip = 2.0 * Vy_local * cos_DL[None, :] * Gamma_lateral / dx[None, :]

    # Net Circulation --------------------------------------------------------------------------------------------------
    Gamma_net = Gamma * Vx_local / dx

    # Final Delta Cp
    dCp = 2.0 * Gamma_net + dCp_sideslip

    return dCp


# ---------------------------------------------------------
#  STATEFUL VERSION
# ---------------------------------------------------------
@io.inputs(
    "system.analysis_data['vortex_distribution']",
    "system.analysis_data['relative_velocity']",
    "system.analysis_data['vortex_strengths']",
    "state.freestream.speed",
)
@io.outputs(
    "system.analysis_data['dCp']",
)
def compute_panel_pressures(state: State, system: Aircraft, settings: Settings):
    """Calculates the differential pressure coefficient (Delta C_P) for all VLM panels."""

    analysis = system.analysis_data
    VD = analysis["vortex_distribution"]

    v_total = analysis["relative_velocity"]
    GAMMA = analysis["vortex_strengths"]
    v_inf = state.freestream.speed

    dCp = compute_pressure_coefficients(VD, v_total, GAMMA, v_inf)

    updated_analysis_data = analysis | {"dCp": dCp}

    updated_system = update(system, "analysis_data", updated_analysis_data)

    return state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Lift and Drag Calculation
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------
# Trefftz Plane Induced Drag
# ---------------------------------------------------------
@jax.jit
def _compute_trefftz_drag(tp_y_var, tp_z_var, tp_y_L, tp_y_R, tp_z_L, tp_z_R, gamma_segments, rho):

    # 1. Distance Matrices
    dy_L = tp_y_var[:, :, None] - tp_y_L[:, None, :]
    dz_L = tp_z_var[:, :, None] - tp_z_L[:, None, :]
    r2_L = jnp.maximum(dy_L**2 + dz_L**2, 1e-12)

    dy_R = tp_y_var[:, :, None] - tp_y_R[:, None, :]
    dz_R = tp_z_var[:, :, None] - tp_z_R[:, None, :]
    r2_R = jnp.maximum(dy_R**2 + dz_R**2, 1e-12)

    # 2. Induced Velocity
    v_ind_y = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * ((dz_L / r2_L) - (dz_R / r2_R)), axis=-1)
    v_ind_z = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * (-(dy_L / r2_L) + (dy_R / r2_R)), axis=-1)

    # 3. Strict 2D Unstructured Normal Vectors
    dy_panel = tp_y_R - tp_y_L
    dz_panel = tp_z_R - tp_z_L
    panel_width = jnp.maximum(jnp.sqrt(dy_panel**2 + dz_panel**2), 1e-16)

    n_hat_y = -dz_panel / panel_width
    n_hat_z = dy_panel / panel_width

    v_normal = v_ind_y * n_hat_y + v_ind_z * n_hat_z

    # 4. Drag Integration
    D_induced = -0.5 * rho * jnp.sum(gamma_segments * v_normal * panel_width, axis=1)

    return D_induced, v_normal


# ---------------------------------------------------------
# Full Coefficient Calculation
# ---------------------------------------------------------


@jax.jit
def _compute_aerodynamic_coefficients(VD, dCp, Gamma, state, system, settings):
    """
    Computes CL, CD, C_m, CY_body, Cl, Cn and induced drag using the unstructured VD mesh.
    """

    vlm_settings = settings.analysis.aerodynamics

    alpha = state.aerodynamics.angles.alpha
    beta = state.aerodynamics.angles.beta
    v_inf = state.freestream.speed
    mach = state.freestream.mach_number
    rho = state.freestream.density

    S_ref = system.areas.reference
    c_ref = system.reference_geometry.mean_aerodynamic_chord
    b_ref = system.reference_geometry.projected_span
    cg = system.reference_geometry.center_of_gravity

    x_m, z_m = cg[:, 0][:, None], -cg[:, 2][:, None]

    sin_alpha, cos_alpha = jnp.sin(alpha), jnp.cos(alpha)
    sin_beta, cos_beta = jnp.sin(beta), jnp.cos(beta)
    crosswind_factor = cos_alpha * sin_beta * 2.0

    # ------------------------------------------------------------------
    # Mesh Topology Resolution
    # ------------------------------------------------------------------
    le_mask_float = VD.is_leading_edge.astype(jnp.float32)
    te_mask_float = VD.is_trailing_edge.astype(jnp.float32)
    strip_ids = VD.strip_ids

    stripwise_chords = jax.ops.segment_sum(VD.chord_lengths, strip_ids, num_segments=VD.total_strips)
    panel_dx_nondim = VD.chord_lengths / stripwise_chords[VD.strip_ids]

    # ------------------------------------------------------------------
    # Local Panel Sweep and Dihedral (Using VD.panel_vertices)
    # 0: Front-Left, 1: Back-Left, 2: Back-Right, 3: Front-Right
    # ------------------------------------------------------------------
    # Leading edge vector of each panel: Front-Right minus Front-Left
    dx_all = VD.panel_vertices[:, 3, 0] - VD.panel_vertices[:, 0, 0]
    dy_all = VD.panel_vertices[:, 3, 1] - VD.panel_vertices[:, 0, 1]
    dz_all = VD.panel_vertices[:, 3, 2] - VD.panel_vertices[:, 0, 2]

    dy_LE = jax.ops.segment_sum(dy_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dz_LE = jax.ops.segment_sum(dz_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dx_LE = jax.ops.segment_sum(dx_all * le_mask_float, strip_ids, num_segments=VD.total_strips)

    dihedral_length_LE = jnp.maximum(jnp.sqrt(dy_LE**2 + dz_LE**2), 1e-12)
    tan_sweep_LE = jnp.clip(dx_LE / dihedral_length_LE, min=-3.73, max=3.73)
    cos_dihedral = jnp.abs(dy_LE) / dihedral_length_LE
    sin_dihedral = jnp.sign(dy_LE) * dz_LE / dihedral_length_LE

    # Panel Forces (Assumes uniform spacing for Pistolesi's theorem)
    quarter_chord_offset = 0.25 * panel_dx_nondim
    colloc_offset = 0.75 * panel_dx_nondim
    panel_force_mag = panel_dx_nondim[None, :] * dCp

    panel_inc = VD.incidence_angle
    panel_axial_coeff = panel_force_mag * jnp.sin(panel_inc)[None, :]
    panel_normal_coeff = panel_force_mag * jnp.cos(panel_inc)[None, :]
    le_inc = jax.ops.segment_sum(panel_inc * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]

    panel_indices = jnp.arange(VD.total_panels)
    strip_start_indices = jax.ops.segment_min(panel_indices, strip_ids, num_segments=VD.total_strips)[strip_ids]
    chordwise_indices = panel_indices - strip_start_indices + 1.0

    vortex_x_nondim = (chordwise_indices - 0.75) * panel_dx_nondim
    panel_pitching_moment = (colloc_offset[None, :] - vortex_x_nondim[None, :]) * panel_normal_coeff

    # ------------------------------------------------------------------
    # Rear Quarter Calculation (Using VD.panel_vertices)
    # ------------------------------------------------------------------
    collocation_x = VD.collocation_points[:, 0]
    trailing_edge_x_avg = (VD.panel_vertices[:, 1, 0] + VD.panel_vertices[:, 2, 0]) / 2.0
    rear_quarter_x = trailing_edge_x_avg - collocation_x
    panel_sideslip_couple = panel_normal_coeff * rear_quarter_x[None, :]

    # Integrate Panels into Strips, V-Mapped over time dimension
    strip_sum = jax.vmap(lambda arr: jax.ops.segment_sum(arr, strip_ids, num_segments=VD.total_strips))

    strip_body_x_coeff = strip_sum(panel_axial_coeff) * stripwise_chords[None, :]
    strip_body_z_coeff = strip_sum(panel_normal_coeff) * stripwise_chords[None, :]
    pitching_moment = strip_sum(panel_pitching_moment) * (stripwise_chords[None, :] ** 2)

    sideslip_couple = strip_sum(panel_sideslip_couple) * stripwise_chords[None, :]
    sideslip_couple = sideslip_couple * (-1.0) * crosswind_factor * cos_dihedral[None, :] * 0.5

    # ------------------------------------------------------------------
    # Leading Edge Suction Correction
    # ------------------------------------------------------------------

    B_sq = jnp.square(mach) - 1.0
    t_sq = jnp.square(tan_sweep_LE)[None, :]

    # Guard statement to avoid singularity at B_sq > t_sq
    L_eff = jnp.where(B_sq < t_sq, jnp.sqrt(jnp.maximum(t_sq - B_sq, 1e-16)), 0.0)

    # Hancock's method:
    le_qc = jax.ops.segment_sum(quarter_chord_offset * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
    le_dCp = strip_sum(dCp * le_mask_float[None, :])
    A0 = 0.5 * le_dCp * jnp.sqrt(le_qc)

    # Suction coefficient
    Cs = 0.5 * jnp.pi * jnp.square(A0) * L_eff

    # Update the strip coefficients w/ leading edge geometry
    strip_body_x_coeff = jnp.where(vlm_settings.corrections.suction, strip_body_x_coeff - Cs, strip_body_x_coeff)
    strip_body_z_coeff = jnp.where(
        vlm_settings.corrections.suction, strip_body_z_coeff + Cs * jnp.sqrt(1.0 + t_sq) * le_inc, strip_body_z_coeff
    )

    # ------------------------------------------------------------------
    # Supersonic Shock Pressure Correction
    # ------------------------------------------------------------------
    theta_w = VD.wedge_angles
    a_local = alpha + le_inc
    theta_u = jnp.where(theta_w > 0, theta_w - a_local, theta_w)
    theta_l = jnp.where(theta_w > 0, theta_w + a_local, theta_w)

    flow_g = state.freestream.gamma

    cos_sweep_LE = 1.0 / jnp.sqrt(1.0 + tan_sweep_LE**2)
    m_normal = mach * cos_sweep_LE

    def compute_strip_shock(m, t, g):
        b = jnp.where(t > 0, theta_beta_mach(m, t, g), jnp.pi / 2)  # Calculate beta
        _, _, _, Ptr_s = oblique_shock(m, t, b, g)  # Shock pressure recovery
        Ptr = jnp.where(t >= 0, Ptr_s, 1.0)
        return Ptr

    vmap_strips = jax.vmap(compute_strip_shock, in_axes=(0, 0, None))
    vmap_machs_and_strips = jax.vmap(vmap_strips, in_axes=(0, 0, 0))

    # Compute upper and lower shock pressure recovery
    strip_Ptr_u = vmap_machs_and_strips(m_normal, theta_u, flow_g)
    strip_Ptr_l = vmap_machs_and_strips(m_normal, theta_l, flow_g)

    # Average shock pressure recovery factor
    strip_Ptr = jnp.squeeze((strip_Ptr_u + strip_Ptr_l) / 2.0, axis=-1)

    effective_Ptr = jnp.where(m_normal > 1.0, strip_Ptr, 1.0)
    effective_Ptr = jnp.where(vlm_settings.corrections.shock, effective_Ptr, 1.0)

    # ------------------------------------------------------------------
    # Body Axis Transformation & Strips Integration
    # ------------------------------------------------------------------
    strip_body_force_x = strip_body_x_coeff * effective_Ptr
    strip_body_force_y = -strip_body_z_coeff * sin_dihedral[None, :] * effective_Ptr
    strip_body_force_z = strip_body_z_coeff * cos_dihedral[None, :] * effective_Ptr

    colloc_LE = jax.ops.segment_sum(
        VD.collocation_points * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips
    )
    colloc_LE_x, colloc_LE_y, colloc_LE_z = colloc_LE[:, 0][None, :], colloc_LE[:, 1][None, :], colloc_LE[:, 2][None, :]

    strip_body_moment_x = strip_body_force_z * colloc_LE_y - strip_body_force_y * (colloc_LE_z - z_m) + sideslip_couple
    strip_body_moment_y = (
        pitching_moment * cos_dihedral[None, :]
        + strip_body_force_x * (colloc_LE_z - z_m)
        - strip_body_force_z * (colloc_LE_x - x_m)
    )
    strip_body_moment_z = (
        pitching_moment * sin_dihedral[None, :]
        - strip_body_force_x * colloc_LE_y
        + strip_body_force_y * (colloc_LE_x - x_m)
    )

    # Strip Aerodynamic Integration: Front-Right (3) and Front-Left (0)
    corner_b1_LE = jax.ops.segment_sum(
        VD.panel_vertices[:, 3, :] * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips
    )
    corner_a1_LE = jax.ops.segment_sum(
        VD.panel_vertices[:, 0, :] * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips
    )

    panel_span_LE = jnp.abs(corner_b1_LE[:, 1] - corner_a1_LE[:, 1])
    strip_area = panel_span_LE * stripwise_chords

    strip_lift = (
        strip_body_force_z * cos_alpha - (strip_body_force_x * cos_beta + strip_body_force_y * sin_beta) * sin_alpha
    ) * panel_span_LE[None, :]
    strip_pitching_moment = (strip_body_moment_y * cos_beta - strip_body_moment_x * sin_beta) * panel_span_LE[None, :]

    force_x = strip_body_force_x * strip_area[None, :]
    force_y = (strip_body_force_y * cos_beta - strip_body_force_x * sin_beta) * strip_area[None, :]
    force_z = strip_body_force_z * strip_area[None, :]

    strip_rolling_moment = (
        strip_body_moment_x * cos_alpha * cos_beta
        + strip_body_moment_y * cos_alpha * sin_beta
        + strip_body_moment_z * sin_alpha
    ) * panel_span_LE[None, :]
    strip_yawing_moment = (
        strip_body_moment_z * cos_alpha - (strip_body_moment_x * cos_beta + strip_body_moment_y * sin_beta) * sin_alpha
    ) * panel_span_LE[None, :]

    # ------------------------------------------------------------------
    # Trefftz Plane Execution
    # ------------------------------------------------------------------

    # Project from the TE to infinity
    TE_corner_L = jax.ops.segment_sum(
        VD.panel_vertices[:, 1, :] * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips
    )
    TE_corner_R = jax.ops.segment_sum(
        VD.panel_vertices[:, 2, :] * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips
    )
    TE_mid = (TE_corner_L + TE_corner_R) / 2.0

    tp_z_var = TE_mid[:, 2] * cos_alpha - TE_mid[:, 0] * sin_alpha
    tp_z_L = TE_corner_L[:, 2] * cos_alpha - TE_corner_L[:, 0] * sin_alpha
    tp_z_R = TE_corner_R[:, 2] * cos_alpha - TE_corner_R[:, 0] * sin_alpha

    # Dimensionalized drag computation
    D_trefftz, _ = _compute_trefftz_drag(
        TE_mid[:, 1][None, :],
        tp_z_var,
        TE_corner_L[:, 1][None, :],
        TE_corner_R[:, 1][None, :],
        tp_z_L,
        tp_z_R,
        strip_sum(Gamma) * v_inf,
        rho[:, 0],
    )

    # Wind-Frame Coefficients
    # Body Frame is Back-Right-Up, Wind-Frame is Front-Right-Down, so CX and CZ are negative
    CX_wind = -jnp.sum(force_x, axis=1) / S_ref
    CY_wind = jnp.sum(force_y, axis=1) / S_ref
    CZ_wind = -jnp.sum(force_z, axis=1) / S_ref

    CL_near = jnp.sum(strip_lift, axis=1) / S_ref

    CDi_far = D_trefftz / (0.5 * rho[:, 0] * jnp.square(v_inf[:, 0]) * S_ref)  # Far-Field (Trefftz plane wake integral)
    CDi_near = -CX_wind * cos_alpha[:, 0] - CZ_wind * sin_alpha[:, 0]  # Near-field (direct force integration)

    C_l = -jnp.sum(strip_rolling_moment, axis=1) / (S_ref * b_ref)
    C_m = jnp.sum(strip_pitching_moment, axis=1) / (S_ref * c_ref)
    C_n = -jnp.sum(strip_yawing_moment, axis=1) / (S_ref * b_ref)

    return CL_near, CDi_far, CDi_near, CX_wind, CY_wind, CZ_wind, C_l, C_m, C_n


@io.inputs(
    "system.analysis_data['vortex_distribution']",
    "system.analysis_data[dCp]",
    "system.analysis_data['vortex_strengths']",
    "state.aerodynamics.angles.alpha",
    "state.aerodynamics.angles.beta",
    "state.freestream.speed",
    "state.freestream.mach_number",
    "state.freestream.density",
    "state.freestream.gamma",
    "system.areas.reference",
    "system.reference_geometry.mean_aerodynamic_chord",
    "system.reference_geometry.projected_span",
    "system.reference_geometry.center_of_gravity",
)
@io.outputs(
    "state.aerodynamics.coefficients.lift.total",
    "state.aerodynamics.coefficients.drag.total",
    "state.aerodynamics.coefficients.drag.induced.total",
    "state.aerodynamics.coefficients.drag.induced.inviscid.total",
    "state.aerodynamics.coefficients.X",
    "state.aerodynamics.coefficients.Y",
    "state.aerodynamics.coefficients.Z",
    "state.aerodynamics.coefficients.moments.pitch",
    "state.aerodynamics.coefficients.moments.roll",
    "state.aerodynamics.coefficients.moments.yaw",
)
def compute_coefficients(state: State, system: Aircraft, settings: Settings):
    """Final VLM step to extract global coefficients and append to State."""

    analysis = system.analysis_data

    CL, CDi_far, CDi_near, CX, CY, CZ, C_l, C_m, C_n = _compute_aerodynamic_coefficients(
        analysis["vortex_distribution"], analysis["dCp"], analysis["vortex_strengths"], state, system, settings
    )

    # Apply Correction Factors
    vlm_settings: VORJAXSettings = settings.analysis.aerodynamics  # type: ignore

    CDi = jnp.where(vlm_settings.near_field_drag, CDi_near, CDi_far)
    CL = jnp.where(vlm_settings.model_fuselage, CL * vlm_settings.corrections.fuselage_lift, CL)

    # Update the Vehicle/Segment State with the aerodynamic coefficients
    C = state.aerodynamics.coefficients

    C = update(C, "lift.total", CL[:, None])
    C = update(C, "drag.total", CDi[:, None])
    C = update(C, "drag.induced.total", CDi[:, None])
    C = update(C, "drag.induced.inviscid.total", CDi[:, None])
    C = update(C, "drag.induced.near_field", CDi_near[:, None])
    C = update(C, "drag.induced.far_field", CDi_far[:, None])

    # Wind-Frame Coefficients
    C = update(C, "X", CX[:, None])
    C = update(C, "Y", CY[:, None])
    C = update(C, "Z", CZ[:, None])

    # Moment Coefficients
    C = update(C, "moments.pitch", C_m[:, None])
    C = update(C, "moments.roll", C_l[:, None])
    C = update(C, "moments.yaw", C_n[:, None])

    state = update(state, "aerodynamics.coefficients", C)

    return state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Settings
# ----------------------------------------------------------------------------------------------------------------------


class SupersonicSettings(Module):
    begin_blend_mach: float = 0.5
    end_blend_mach: float = 2.0

    peak_CL_multiplier: float = 1.15  # noqa: N815
    peak_mach_number: Optional[float] = None
    _transonic_CL_blender: Callable = method_field(ensemble_CL_spline)  # noqa: N815

    begin_drag_rise_mach_number: float = 0.95
    end_drag_rise_mach_number: float = 1.2

    transonic_drag_multiplier: float = 1.25
    volume_wave_drag_scaling: float = 3.2

    cross_section_type: str = field("Fixed", static=True)
    wave_drag_type: str = field("Raymer", static=True)

    def __check_init__(self):
        if self.peak_mach_number is not None:
            object.__setattr__(self, "_transonic_CL_blender", peaked_CL_spline)

    def transonic_CL_blender(self, M, val_sub, val_sup):
        return self._transonic_CL_blender(
            M,
            self.begin_blend_mach,
            self.peak_mach_number,
            self.end_blend_mach,
            val_sub,
            val_sup,
            peak_multiplier=self.peak_CL_multiplier,
        )


class CorrectionFactors(Module):
    suction: bool = field(True, static=True)
    shock: bool = field(True, static=True)

    fuselage_lift: float = 1.14
    trim_drag: float = 1.02

    viscous_lift_drag: float = 0.38
    lift_to_drag: float = 0.0
    CL_max: float = 1.0


class FormFactors(Module):
    span_efficiency: float = 1.0
    oswald: float = 1.0

    wing: float = 1.1
    fuselage: float = 2.3
    pylon: float = 0.2


class Surrogate(Module):
    surrogate: Optional[Any] = field(sklearn.gaussian_process.GaussianProcessRegressor, static=True)

    blend_transonic: bool = True

    angle_of_attack: jax.Array = field(lambda: jnp.linspace(-5.0, 15.0, 40) * U.deg)
    sideslip_angle: jax.Array = field(lambda: jnp.linspace(0.0, 15.0, 30) * U.deg)
    mach: jax.Array = field(lambda: jnp.linspace(0.0, 0.85, 20))

    aileron_deflection: jax.Array = field(lambda: jnp.array([30, 10.0, 1e-12]) * U.deg)
    elevator_deflection: jax.Array = field(lambda: jnp.array([30, 10.0, 1e-12]) * U.deg)
    rudder_deflection: jax.Array = field(lambda: jnp.array([30, 10.0, 1e-12]) * U.deg)
    flap_deflection: jax.Array = field(lambda: jnp.array([30, 10.0, 1e-12]) * U.deg)
    slat_deflection: jax.Array = field(lambda: jnp.array([30, 10.0, 1e-12]) * U.deg)

    u: jax.Array = field(lambda: jnp.array([0.2, 0.1, 1e-12]))
    v: jax.Array = field(lambda: jnp.array([0.2, 0.1, 1e-12]))
    w: jax.Array = field(lambda: jnp.array([0.2, 0.1, 1e-12]))

    pitch_rate: jax.Array = field(lambda: jnp.array([0.3, 0.15, 0.0]) * U.rad / U.s)
    roll_rate: jax.Array = field(lambda: jnp.array([0.3, 0.15, 0.0]) * U.rad / U.s)
    yaw_rate: jax.Array = field(lambda: jnp.array([0.3, 0.15, 0.0]) * U.rad / U.s)

    def fit(self, *args, **kwargs):
        return self.surrogate.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.surrogate.predict(*args, **kwargs)


class Vortices(Module):
    model_fuselage: bool = field(False, static=True)
    verbose: bool = field(False, static=True)

    # Discretization Inputs (Optional, so the user can choose which to define)
    spanwise_cosine: bool = field(True, static=True)
    chordwise_cosine: bool = field(False, static=True)  # Currently unsupported

    n_spanwise: Optional[Iterable[int] | int] = field(
        8, static=True
    )  # Min value is number of wing segments (possibly more for control surfaces)
    n_chordwise: Optional[Iterable[int] | int] = field(
        3, static=True
    )  # Min value 3 to allow front and rear control surfaces

    # Can set separate values for each wing/fuselage, else uses global value above
    wings_n_spanwise: Optional[Iterable[int] | int] = field(None, static=True)
    wings_n_chordwise: Optional[Iterable[int] | int] = field(None, static=True)

    bodies_n_spanwise: Optional[Iterable[int] | int] = field(None, static=True)
    bodies_n_chordwise: Optional[Iterable[int] | int] = field(None, static=True)

    def __post_init__(self):
        """Validates discretization inputs and resolves global vs separate routing."""

        if self.chordwise_cosine:
            warnings.warn("Chordwise cosine spacing is currently unsupported. Defaulting to linear spacing.")
            object.__setattr__(self, "chordwise_cosine", False)

        # Check if the user explicitly provided separate definitions
        separate_provided = any(
            [
                self.wings_n_spanwise is not None,
                self.wings_n_chordwise is not None,
                self.bodies_n_spanwise is not None,
                self.bodies_n_chordwise is not None,
            ]
        )

        if separate_provided:
            # Validate that all separate variables were provided
            missing_separate = any(
                x is None
                for x in [
                    self.wings_n_spanwise,
                    self.wings_n_chordwise,
                    self.bodies_n_spanwise,
                    self.bodies_n_chordwise,
                ]
            )
            if missing_separate:
                raise ValueError("If using separate surface discretization, all n_sw and n_cw values must be defined.")

        else:
            # User didn't provide separate settings, so we fallback to the global defaults
            if not self.n_spanwise or not self.n_chordwise:
                raise ValueError("If using global surface discretization, both n_sw and n_cw must be defined.")

            # Route the global settings to the specific component fields
            object.__setattr__(self, "wings_n_spanwise", self.n_spanwise)
            object.__setattr__(self, "wings_n_chordwise", self.n_chordwise)
            object.__setattr__(self, "bodies_n_spanwise", self.n_spanwise)
            object.__setattr__(self, "bodies_n_chordwise", self.n_chordwise)


class VORJAXSettings(Module):
    model_fuselage: bool = field(False, static=True)
    trim_aircraft: bool = field(False, static=True)

    recalculate_wetted_area: bool = field(False, static=True)
    model_propeller_wake: bool = field(False, static=True)
    near_field_drag: bool = field(False, static=True)

    CL_max: float = jnp.inf
    CD_increment: float = 0.0
    spoiler_drag_increment: float = 0.0

    # Sub-Settings

    vortices: Vortices = field(Vortices)

    supersonic: SupersonicSettings = field(SupersonicSettings)
    corrections: CorrectionFactors = field(CorrectionFactors)
    form_factors: FormFactors = field(FormFactors)
    surrogate: Surrogate = field(Surrogate)


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------


def _default_VORJAX_init_steps():
    return (
        ProcessStep(function=initialize_aerodynamics, name="Initialize Component Bookkeeping"),
        ProcessStep(function=initialize_VORJAX_data, name="Initialize Data Structures"),
        ProcessStep(function=discretize_surfaces, name="Discretize Surfaces"),
    )


class InitializeVORJAX(Process):
    name: str = field("Initialize VORJAX", static=True)
    steps: tuple = field(_default_VORJAX_init_steps)

    def __init__(self, name="Initialize VORJAX", steps=_default_VORJAX_init_steps()) -> None:
        super().__init__(name=name, steps=steps)


# ----------------------------------------------------------
#  VORJAX Compute Process
# ----------------------------------------------------------


def _default_VORJAX_compute_steps():
    return (
        # Lift and Induced Drag
        ProcessStep(function=check_freestream, name="Freestream Validation"),
        ProcessStep(function=compute_boundary_conditions, name="Calculate Boundary Conditions"),
        ProcessStep(function=compute_induced_velocity, name="Calculate VICs"),
        ProcessStep(function=compute_vortex_strength, name="Compute Vortex Strength"),
        ProcessStep(function=compute_panel_pressures, name="Compute Pressure Coefficients"),
        ProcessStep(function=compute_coefficients, name="Compute Aerodynamic Coefficients"),
        ProcessStep(function=apply_aerodynamic_forces, name="Apply Aerodynamic Forces"),
    )


class ComputeVORJAX(Process):
    name: str = field("Compute VORJAX", static=True)
    steps: tuple = field(_default_VORJAX_compute_steps)

    def __init__(
        self,
        name: str = "Compute VORJAX",
        steps: tuple = _default_VORJAX_compute_steps(),
    ) -> None:
        super().__init__(name=name, steps=steps)


class VORJAX(Process):
    name: str = static_field("Aerodynamics")
    steps: tuple = field(lambda: (InitializeVORJAX(), ComputeVORJAX()))

    def __init__(
        self,
        name: str = "Aerodynamics",
        steps: tuple = (InitializeVORJAX(), ComputeVORJAX()),
    ) -> None:
        super().__init__(name=name, steps=steps)

    # TODO: Add full drag, trimming, stability analysis


# -----------------------------------------------------------
# Batched VORJAX Analysis
# -----------------------------------------------------------

VORJAX_Inputs = {
    "mach": (TreePath(("freestream", "mach_number")), [0.0]),
    "alpha": (TreePath(("aerodynamics", "angles", "alpha")), [0.0]),
    "beta": (TreePath(("aerodynamics", "angles", "beta")), [0.0]),
    "roll_rate": (TreePath(("stability", "static", "roll_rate")), [0.0]),
    "pitch_rate": (TreePath(("stability", "static", "pitch_rate")), [0.0]),
    "yaw_rate": (TreePath(("stability", "static", "yaw_rate")), [0.0]),
    "density": (TreePath(("freestream", "density")), [1.225]),
    "gamma": (TreePath(("freestream", "gamma")), [1.4]),
    "temperature": (TreePath(("freestream", "temperature")), [288.15]),
}

VORJAX_Outputs = {
    "CL": TreePath(("aerodynamics", "coefficients", "lift", "total")),
    "CD": TreePath(("aerodynamics", "coefficients", "drag", "total")),
    "CX": TreePath(
        (
            "aerodynamics",
            "coefficients",
            "X",
        )
    ),
    "CY": TreePath(
        (
            "aerodynamics",
            "coefficients",
            "Y",
        )
    ),
    "CZ": TreePath(
        (
            "aerodynamics",
            "coefficients",
            "Z",
        )
    ),
    "C_l": TreePath(("aerodynamics", "coefficients", "moments", "roll")),
    "C_m": TreePath(("aerodynamics", "coefficients", "moments", "pitch")),
    "C_n": TreePath(("aerodynamics", "coefficients", "moments", "yaw")),
}


if __name__ == "__main__":
    print(*[VORJAX_Inputs[i][0] for i in ["alpha", "mach"]])

# ----------------------------------------------------------
#  Surrogate VLM Process
# ----------------------------------------------------------

# TODO: Surrogate VLM initialization, steps, and analysis
