#!/usr/bin/env python
# coding: utf-8

from Constants import *
# from Main_evolution_function import *
import numpy as np
# from EXO_K_wo_ocean import *
from H2_atmospheric_flows import *
from CO2_atmospheric_flows import *
from CH4_atmospheric_flows import *

# ===========================================================
# Defining constants --> H2 , N2 and CH4
# ===========================================================

# Based on Kharecha et al, 2005

# Solubilities
alphaH     = 7.8e-4  # mol.L-1.bar-1
alphaG     = 1.4e-3  # mol.L-1.bar-1
alphaN     = 7.0e-4  # mol.L-1.bar-1
# Thermal diffusivities
thdH       = 5.0e-5  # cm2.s-1
thdG       = 1.8e-5  # cm2.s-1
#Piston velocities # Assuming no dependance on temperature
pvH        = 1.3e-2  # cm.s-1
pvG        = 4.5e-3  # cm.s-1
pvN        = 4.8e-3  # cm.s-1

# ===========================================================
# Defining constants --> CO2
# ===========================================================

def alphaC(T):
    """
    Computes the CO2 solubility depending on temperature

    Params:
     - T : float, temperature (K)

    Returns:
     - float, CO2 solubility (mol.L-1.bar-1)"""
    return np.exp(9345.17 / T - 167.8108 + 23.3585 * np.log(T) + (0.023517 - 2.3656e-4 * T + 4.7036e-7 * T**2) * 35.0)

pvC = 4.8e-3   # cm.s-1


def interface_flow(vp,alpha,p,c):
    """
    Computes the flux of a chemical specy at the interface between the ocean and the atmosphere
    Based on Kharecha et al, 2005

    Params:
     - vp : float, piston velocity (cm/s)
     - alpha : float, solubility (mol.L-1.bar-1)
     - p : float, partial pressure of the specy at the surface (bar)
     - c : float, concentration in the ocean (mol/L)

    Returns:
     - float, flow at the interface atmosphere -> ocean (mol/cm²/s)"""
    return vp*(alpha*p-c)*1e-3


def equilibre_des_flux_co2(P_profile, T_profile, fCO2, dZ, Z, M):
    """
    Approximates equilibrium flux contribution for CO2 
    combining dissolution and atmospheric flux components.
    """
    return alphaC(T_profile[-1]) * P_profile[-1] * 1e-5 * fCO2 + 1e3 * CO2_flow(P_profile, T_profile, fCO2, dZ, Z, M) / pvC

def equilibre_des_flux_ch4(P_profile, T_profile, fch4, Z):
    """
    Approximates equilibrium flux contribution for CH4 
    combining dissolution and atmospheric flux components.
    """
    return alphaG * P_profile[-1] * 1e-5 * fch4 + 1e3 * CH4_flow(P_profile, T_profile, fch4, Z) / pvG

def equilibre_des_flux_h2(fh2, T_profile, P_profile, M):
    """
    Approximates equilibrium flux contribution for H2 
    combining dissolution and atmospheric flux components.
    """
    return alphaH * P_profile[-1] * 1e-5 * fh2 + 1e3 * H2_flow(fh2, T_profile, P_profile, M) / pvH


from scipy.optimize import fsolve

def find_c_CH4(P_profile, T_profile, fCH4, Z, vp, alpha, p, 
               CH4_flow, interface_flow, c_init_guess=1.0):
    """
    Finds the aqueous concentration (c) of CH4 at which 
    the flux from the atmosphere equals the flux to the ocean.

    Parameters:
    - P_profile, T_profile, fCH4, Z: parameters for CH4_flow
    - vp, alpha, p: parameters for interface_flow
    - CH4_flow: function for atmospheric methane flux
    - interface_flow: function for dissolution flux
    - c_init_guess: initial guess for root finding

    Returns:
    - c (float): equilibrium aqueous concentration of CH4
    """
    def equation(c):
        return CH4_flow(P_profile, T_profile, fCH4, Z) - interface_flow(vp, alpha, p, c)

    c_solution, = fsolve(equation, c_init_guess, xtol=1e-9)
    return c_solution

def find_c_H2(P_profile, T_profile, fh2, Z, vp, alpha, p, M,
              H2_flow, interface_flow, c_init_guess=1.0):
    """
    Finds the aqueous concentration (c) of H2 where atmospheric 
    and aqueous fluxes are balanced.
    """
    def equation(c):
        return H2_flow(fh2, T_profile, P_profile, M) - interface_flow(vp, alpha, p, c)

    c_solution, = fsolve(equation, c_init_guess, xtol=1e-9)
    return c_solution

def find_c_CO2(P_profile, T_profile, fco2, Z, dZ, vp, alpha, p, M,
               CO2_flow, interface_flow, c_init_guess=1.0):
    """
    Finds the aqueous concentration (c) of CO2 where atmospheric 
    and aqueous fluxes are balanced.
    """
    def equation(c):
        return CO2_flow(P_profile, T_profile, fco2, dZ, Z, M) - interface_flow(vp, alpha, p, c)

    c_solution, = fsolve(equation, c_init_guess, xtol=1e-9)
    return c_solution

def equilibrium_concentration_value_CH4(P_profile, T_profile, fch4, Z, pvG, alphaG, M):
    """
    Estimates the equilibrium aqueous concentration of CH4 
    by averaging two methods: root-finding and flux-based estimation.
    """
    c1 = find_c_CH4(P_profile, T_profile, fch4, Z, pvG, alphaG, P_profile[-1]*fch4*1e-5, 
                    CH4_flow, interface_flow, c_init_guess=fch4/100)
    c2 = equilibre_des_flux_ch4(P_profile, T_profile, fch4, Z)
    return (c1 + c2) / 2

def equilibrium_concentration_value_H2(P_profile, T_profile, fh2, Z, pvH, alphaH, M):
    """
    Estimates the equilibrium aqueous concentration of H2 
    by averaging two methods: root-finding and flux-based estimation.
    """
    c1 = find_c_H2(P_profile, T_profile, fh2, Z, pvH, alphaH, P_profile[-1]*fh2*1e-5, M,
                   H2_flow, interface_flow, c_init_guess=fh2/100)
    c2 = equilibre_des_flux_h2(fh2, T_profile, P_profile, M)
    return (c1 + c2) / 2

def equilibrium_concentration_value_CO2(P_profile, T_profile, fco2, dZ, Z, pvC, M):
    """
    Estimates the equilibrium aqueous concentration of CO2 
    by averaging two methods: root-finding and flux-based estimation.
    """
    c1 = find_c_CO2(P_profile, T_profile, fco2, Z, dZ, pvC, alphaC(T_profile[-1]), P_profile[-1]*fco2*1e-5, M,
                    CO2_flow, interface_flow, c_init_guess=fco2/100)
    c2 = equilibre_des_flux_co2(P_profile, T_profile, fco2, dZ, Z, M)
    return (c1 + c2) / 2





