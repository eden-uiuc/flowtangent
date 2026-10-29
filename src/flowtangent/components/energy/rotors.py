from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple, Optional

if TYPE_CHECKING:
    from ... import State, System, Settings

from ..wings import Wing, WingDimensions
from .nodes import PACTNode

import jax.numpy as jnp
import jax
import optimistix as optx
from typing import Callable, Tuple

from ...utils import method_field,  update
from ...utils.typing import _

# ----------------------------------------------------------------------------------------------------------------------
# BEMT Pure Functions
# ----------------------------------------------------------------------------------------------------------------------

def prandtl_loss(
        r: jax.Array,
        phi: jax.Array,
        B: int,
        R_root: float,
        R_tip: float,
    ) -> jax.Array:
    """Computes combined hub and tip loss factors."""
    sin_phi = jnp.clip(jnp.abs(jnp.sin(phi)), 1e-8, None)
    
    # Exponent terms
    f_tip = (B / 2.0) * (R_tip - r) / (r * sin_phi)
    f_root = (B / 2.0) * (r - R_root) / (r * sin_phi)
    
    # Loss factors (clipping exponent to avoid NaN in arccos)
    F_tip = (2.0 / jnp.pi) * jnp.arccos(jnp.clip(jnp.exp(-f_tip), -1.0, 1.0))
    F_root = (2.0 / jnp.pi) * jnp.arccos(jnp.clip(jnp.exp(-f_root), -1.0, 1.0))
    
    return jnp.clip(F_tip * F_root, 1e-4, 1.0)

def element_kinematics(
    phi: jax.Array,
    r: jax.Array,
    chord: jax.Array, 
    cl: jax.Array,
    cd: jax.Array,
    B: int,
    R_root: float,
    R_tip: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Calculates induction factors and aerodynamic forcing coefficients."""
    F = prandtl_loss(r, phi, B, R_root, R_tip)
    sigma = (B * chord) / (2.0 * jnp.pi * r)
    
    c_n = cl * jnp.cos(phi) + cd * jnp.sin(phi)
    c_t = cl * jnp.sin(phi) - cd * jnp.cos(phi)
    
    sin_phi = jnp.sin(phi) + 1e-8
    cos_phi = jnp.cos(phi) + 1e-8
    
    a = (sigma * c_n) / (4.0 * F * sin_phi**2 - sigma * c_n)
    a_prime = (sigma * c_t) / (4.0 * F * sin_phi * cos_phi + sigma * c_t)
    
    return a, a_prime, c_n, c_t

def element_inflow_angle(
    phi_guess: jax.Array,
    r: jax.Array,
    chord: jax.Array,
    twist: jax.Array, 
    v_inf: jax.Array,
    omega: jax.Array,
    B: int,
    R_root: float,
    R_tip: float,
    polar_fn: Callable,
    polar_args: tuple
) -> jax.Array:
    """Finds equilibrium inflow angle using Optimistix."""
    
    def residual_fn(phi, args):
        # Unpack arguments needed for the residual
        r, chord, twist, v_inf, omega, B, R_root, R_tip, polar_args = args
        
        alpha = twist - phi
        cl, cd = polar_fn(alpha, *polar_args)
        
        a, a_prime, _, _ = element_kinematics(phi, r, chord, cl, cd, B, R_root, R_tip)
        
        # R(phi) = U_axial_geom - U_axial_induced = 0
        # sin(phi)*(1 - a') - cos(phi)*(V_inf / (omega * r))*(1 + a)
        R = jnp.sin(phi) * (1.0 - a_prime) - jnp.cos(phi) * (v_inf / (omega * r)) * (1.0 + a)
        return R

    solver = optx.Newton(rtol=1e-5, atol=1e-5)
    args = (r, chord, twist, v_inf, omega, B, R_root, R_tip, polar_args)
    
    sol = optx.root_find(
        fn=residual_fn,
        solver=solver,
        y0=phi_guess,
        args=args,
        max_steps=50,
        throw=False
    )
    
    # Bound physically realistic inflow angles
    return jnp.clip(sol.value, 1e-4, jnp.pi/2 - 1e-4)

def evaluate_rotor(
    radii: jax.Array,
    chords: jax.Array,
    twists: jax.Array, 
    v_inf: jax.Array,
    omega: jax.Array,
    rho: jax.Array, 
    B: int,
    R_root: float,
    R_tip: float,
    polar_fn: Callable,
    polar_args: tuple,
    phi_guesses: Optional[jax.Array] = None,
):
    """
    Evaluates total Thrust and Torque for a full rotor.
    All spanwise arrays must be of shape (N_stations,).
    """
    if phi_guesses is None:
        # Simple geometric inflow guess if none provided
        phi_guesses = jnp.arctan2(v_inf, omega * radii)

    # 1. Vectorize the solver across the spanwise dimension
    # (radii, chords, twists, phi_guesses) are mapped (0). Constants are None.
    vmap_inflow = jax.vmap(
        element_inflow_angle,
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
    )
    
    phi = vmap_inflow(
        phi_guesses, radii, chords, twists, 
        v_inf, omega, B, R_root, R_tip, 
        polar_fn, polar_args
    )
    
    # 2. Vectorized evaluation of final aerodynamic forces
    def compute_forces(r, chord, twist, phi):
        alpha = twist - phi
        cl, cd = polar_fn(alpha, *polar_args)
        
        a, a_prime, c_n, c_t = element_kinematics(phi, r, chord, cl, cd, B, R_root, R_tip)
        
        # Local velocity magnitude squared: W^2 = U_axial^2 + U_tangential^2
        # U_axial = v_inf * (1 + a)
        # U_tangential = omega * r * (1 - a_prime)
        W_sq = (v_inf * (1.0 + a))**2 + (omega * r * (1.0 - a_prime))**2
        
        # Differential forces per unit span
        dynamic_pressure = 0.5 * rho * W_sq * chord
        dT_dr = B * dynamic_pressure * c_n
        dQ_dr = B * dynamic_pressure * c_t * r
        
        return dT_dr, dQ_dr

    vmap_forces = jax.vmap(compute_forces, in_axes=(0, 0, 0, 0))
    dT_dr, dQ_dr = vmap_forces(radii, chords, twists, phi)
    
    # 3. Spanwise Integration
    thrust = jnp.trapezoid(dT_dr, radii)
    torque = jnp.trapezoid(dQ_dr, radii)
    
    return thrust, torque

# ----------------------------------------------------------------------------------------------------------------------
# Rotor Class
# ----------------------------------------------------------------------------------------------------------------------

class Rotor(PACTNode):

    blade: Wing = _
    radii: WingDimensions = _

    polar_function: Optional[Callable] = method_field(None)
    
    n_blades: int = 3

    def transmit(self, state: State, system: System, settings: Settings):

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        
        blade = system_node.blade
        elements = blade.segments
        span = blade.spans.projected
        c_root = blade.chords.root

        r = jnp.asarray([e.percent_span_location * span for e in elements])
        c = jnp.asarray([e.root_chord_percent * c_root for e in elements])
        theta = jnp.asarray([e.twist for e in elements])
        
        thrust, torque = evaluate_rotor(
            radii=r,
            chords=c,
            twists=theta,
            v_inf=state.freestream.speed,
            omega=network_state.omega,
            rho=state.freestream.density,
            B=system_node.n_blades,
            R_root=system_node.radii.root,
            R_tip=system_node.radii.tip,
            polar_fn=system_node.polar_function,
            polar_args=(), # TODO: Modify once polar function is established,
            phi_guesses=None,
        )

        outputs = state_node.force
        outputs = update(outputs, (("thrust", thrust), ("torque", torque)))
        
        updated_state = update(state, lambda s: s.energy.nodes[self.network_id].flow, outputs)

        return updated_state, system, settings