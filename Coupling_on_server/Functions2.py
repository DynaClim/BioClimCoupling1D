#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve
from scipy.integrate import simpson
import numpy as np
from Constants import *


# In[2]:


def compute_altitudes(P, T, M, g=3.71):
    """
    Calculate altitudes based on pressure and temperature profiles, accounting for
    the molar mass of the atmosphere. Inputs and outputs are ordered from highest
    altitude (lowest pressure) to surface (highest pressure).

    :param P: List or array of pressures (Pa), from highest altitude to surface.
    :param T: List or array of temperatures (K), same order as P.
    :param M: Molar mass of the atmosphere (kg/mol).
    :param g: Gravitational acceleration (m/s^2), default is 3.71 (Mars).
    :return: Tuple of (altitudes (m), altitude increments (m)), where altitudes
             decrease from highest altitude (Z[0] = 0) to surface (Z[-1]).
    """
    # Universal gas constant (J/(mol·K))
    R = 8.314

    # Convert inputs to numpy arrays
    P = np.array(P, dtype=float)
    T = np.array(T, dtype=float)

    # Input validation
    if len(P) != len(T):
        raise ValueError("P and T must have the same length.")
    if len(P) < 2:
        raise ValueError("At least two points are required.")
    if not np.all(T > 0):
        raise ValueError("Temperatures must be positive (in Kelvin).")
    if not np.all(P > 0):
        raise ValueError("Pressures must be positive.")
    if M <= 0:
        raise ValueError("Molar mass must be positive.")
    if g <= 0:
        raise ValueError("Gravitational acceleration must be positive.")
    if not np.all(np.diff(P) > 0):
        raise ValueError("Pressures must increase monotonically from highest altitude to surface.")

    # Initialize arrays
    Z = np.zeros(len(P))  # Altitudes, Z[0] = 0 at highest altitude
    DZ = np.zeros(len(P) - 1)  # Altitude increments

    # Calculate altitude increments using the hydrostatic equation
    for i in range(1, len(P)):
        # Average temperature and pressure between layers
        T_avg = (T[i-1] + T[i]) / 2
        P_avg = (P[i-1] + P[i]) / 2
        dP = P[i] - P[i-1]  # Positive, since P[i] > P[i-1] (lower altitude, higher pressure)

        # Compute altitude increment: dz = -(R T / (g M P)) dP
        dz = -(R * T_avg / (g * M * P_avg)) * dP
        DZ[i-1] = dz
        Z[i] = Z[i-1] + dz  # Altitudes decrease (dz is negative)

    return np.abs(Z[::-1]), np.abs(DZ[::-1])


# In[3]:


def atm_matter_quantity(P_profile, T_profile,dZ, M, surface, g=3.71):
    """
    Calculates the total amount of matter (in moles) of a chemical species in the atmosphere.

    Parameters:
    - P_profile : array, partial pressure (Pa) of the species (from top to surface)
    - T_profile : array, temperature (K), in the same order as P_profile
    - dZ : array, thickness of atmosphere layers
    - M : molar mass of the species (kg/mol)
    - surface : surface area of the planet (in m²)
    - g : gravity (m/s²), default is 3.71 m/s² (Mars)

    Returns:
    - n_total : total amount of matter (mol) of the species
    """
    dZ = np.concatenate([ np.array([0]),dZ])
    # Calculate the mean temperatures and pressures between layers
#     T_mean = (T_profile[:-1] + T_profile[1:]) / 2
#     P_mean = (P_profile[:-1] + P_profile[1:]) / 2
    T_mean = T_profile
    P_mean = P_profile
    # Calculate the volume of each atmospheric layer (cylindrical volume)
    V_shell = dZ * surface

    # Calculate the amount of matter in each layer
    n_shell = (P_mean * V_shell) / (R * T_mean)

    # Sum the total amount of matter across all layers
    n_total = np.sum(n_shell)

    return n_total


# In[4]:


def compute_pressure_profile_integrated(altitudes, temperatures, ntot, surface, M=0.0377, g=3.71):
    """
    Calculate the atmospheric pressure profile using an integrated approach, optimized
    for accurate surface pressure.

    Parameters:
    - altitudes: List/array of altitudes (m), in descending order (highest to surface).
    - temperatures: List/array of temperatures (K), same order as altitudes.
    - ntot: Total number of moles in the atmosphere (mol).
    - surface: Surface area of the planet (m^2).
    - M: Mean molar mass of the atmosphere (kg/mol, default: 0.0377 for Mars).
    - g: Gravitational acceleration (m/s^2, default: 3.71 for Mars).

    Returns:
    - pressures: Array of pressures (Pa), in descending altitude order (highest to surface).
    """
    # Universal gas constant (J/(mol·K))
    R = 8.314

    # Convert inputs to NumPy arrays
    altitudes = np.array(altitudes, dtype=float)
    temperatures = np.array(temperatures, dtype=float)

    # Input validation
    if len(altitudes) != len(temperatures):
        raise ValueError("Altitudes and temperatures must have the same length.")
    if len(altitudes) < 2:
        raise ValueError("At least two points are required to compute the profile.")
    if not np.all(temperatures > 0):
        raise ValueError("Temperatures must be positive (in Kelvin).")
    if not np.all(np.diff(altitudes) < 0):
        raise ValueError("Altitudes must be monotonically decreasing (highest to surface).")
    if ntot <= 0:
        raise ValueError("Total number of moles must be positive.")
    if surface <= 0:
        raise ValueError("Surface area must be positive.")
    if M <= 0:
        raise ValueError("Molar mass must be positive.")
    if g <= 0:
        raise ValueError("Gravitational acceleration must be positive.")

    # Check for small altitude increments
    dz = -np.diff(altitudes)
    if np.any(dz < 1e-3):
        print("Warning: Small altitude increments (< 1 mm) may cause numerical instability.")

    # Reverse arrays for integration (ascending order)
    altitudes_asc = altitudes[::-1]
    temperatures_asc = temperatures[::-1]

    # Shift altitudes so surface is at z=0
    altitudes_asc = altitudes_asc - altitudes_asc[-1]

    # Compute number density profile
    pressures = np.zeros_like(altitudes_asc)
    n0 = 1.0  # Reference number density
    for i in range(len(altitudes_asc)):
        if i == 0:
            pressures[i] = n0 * R * temperatures_asc[i]
        else:
            z_range = altitudes_asc[:i+1]
            T_range = temperatures_asc[:i+1]
            integrand = M * g / (R * T_range)
            exponent = simpson(integrand, z_range)  # More accurate integration
            n_z = n0 * np.exp(-exponent)
            pressures[i] = n_z * R * temperatures_asc[i]

    # Normalize pressures using ntot
    n_z = pressures / (R * temperatures_asc)
    integral = simpson(n_z, altitudes_asc)  # Use simps for accuracy

    # Extrapolate integral to upper atmosphere (assume exponential decay)
    z_max = altitudes_asc[-1]
    T_avg = np.mean(temperatures_asc[-2:])  # Average temperature at top
    scale_height = R * T_avg / (M * g)  # ~10 km for Mars
    integral_upper = (n_z[-1] * scale_height)  # Approximate contribution above z_max
    total_integral = integral + integral_upper

    # Surface pressure
    scale_factor = ntot / (surface * total_integral)
    pressures = pressures * scale_factor

    # Return pressures in descending altitude order
    return pressures[::-1]*2/1.74350

