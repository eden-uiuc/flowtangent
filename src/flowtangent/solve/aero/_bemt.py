from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import State, System, Settings


import jax.numpy as jnp
import jax
import optimistix as optx
from typing import Callable, Tuple

from ... import Process, ProcessStep

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
    phi_guesses: jax.Array = None,
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
    
    return thrust, torque, phi

# ----------------------------------------------------------------------------------------------------------------------
# BEMT Process Functions
# ----------------------------------------------------------------------------------------------------------------------

