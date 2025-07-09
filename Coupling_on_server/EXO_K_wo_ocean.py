#!/usr/bin/env python
# coding: utf-8

# =========================================================================
# Importing libraries
# =========================================================================

import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from Constants import *

import exo_k as xk

xk.Settings().set_mks(True)

import sys
import os

# =========================================================================
# Defining main parameters
# =========================================================================

Nlay = 50  # Number of layers
Tstrat = 100.  # K => Temperature of the stratosphere
grav = 3.71  # m/s^2 => Surface gravity
Rp = None  # m => Planetary radius
albedo_surf = 0.161  # m => Surface albedo 
Tstar = 5778.  # K => Stellar blackbody temperature
flux_top_dw = 109

# =========================================================================
# Defining some useful fonctions
# =========================================================================

def equilibrium_state_values(model):
    time_evo = model.time_hist / xk.DAY
    teq = time_evo[-1]
    T = model.tlay_hist[:, -1]
    mask = np.where(time_evo > teq/2)
    T = T[mask]
    last_T_profile = model.tlay_hist[-1,:]
    last_P_profile = model.atm.play
    return teq,np.mean(T),last_T_profile,last_P_profile

def compute_R_over_cp(fN2, fCO2, fCH4, fH2):
    """
    Calcule R/cp pour un melange gazeux donne par les fractions molaires.
    Arguments:
        fN2, fCO2, fCH4, fH2 : fractions molaires de N2, CO2, CH4 et H2
    Retourne:
        R/cp (sans dimension)
    """

    # Verifier que la somme des fractions est proche de 1
    total_fraction = fN2 + fCO2 + fCH4 + fH2
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("La somme des fractions molaires doit etre egale a 1.")

    # Valeurs de gamma pour chaque gaz
    gamma_N2 = 1.4
    gamma_CO2 = 1.3
    gamma_CH4 = 1.32
    gamma_H2 = 1.41

    # Calcul de R/cp pour chaque gaz
    Rcp_N2 = 1 - 1 / gamma_N2
    Rcp_CO2 = 1 - 1 / gamma_CO2
    Rcp_CH4 = 1 - 1 / gamma_CH4
    Rcp_H2 = 1 - 1 / gamma_H2

    # Moyenne ponderee
    Rcp_total = (fN2 * Rcp_N2 +
                 fCO2 * Rcp_CO2 +
                 fCH4 * Rcp_CH4 +
                 fH2 * Rcp_H2)

    return Rcp_total

def defining_atm(h2,ch4,n2,ptop,psurf,t_surf,acceleration=None):
    if 0>h2 or h2>1 or 0>ch4 or ch4>1 or 0>n2 or n2>1 or (h2+ch4+n2)>1:
        raise ValueError("Not physical atmosphere composition")
    Tsurf = t_surf # K => Surface temperature
    composition = {'CO2': 'background', 'CH4': ch4, 'H2':h2, 'inactive_gas':n2}
    # Volume molar concentrations of the various species
    # One gas can be set to `background`.
    # It's volume mixing ratio will be automatically computed to fill the atmosphere.
    print('ptop',ptop,'psurf',psurf)
    evol_mars = xk.Atm_evolution(Nlay=Nlay, 
                                 psurf=psurf, 
                                 ptop=ptop,
                                 Tsurf=Tsurf, 
                                 Tstrat=Tstrat,
                                 grav=grav, 
                                 rcp=compute_R_over_cp(n2, 1-ch4-h2-n2, ch4, h2), 
                                 Rp=Rp,
                                 albedo_surf=albedo_surf,
                                 bg_vmr=composition,  # notice the change in argument name
                                 k_database=k_db_LR, 
                                 cia_database=cia_db_LR,
                                 rayleigh=True,
                                 flux_top_dw=flux_top_dw, 
                                 Tstar=Tstar,
                                 convection=True,
                                 diffusion=True,
                                 dTmax_use_kernel = 1
                                 )
    if acceleration != None : evol_mars.set_options(acceleration_mode=acceleration)
    

    return evol_mars



# =========================================================================
# Importing corrk and cia tables
# =========================================================================


datapath_ktables = 'Ktables/'
datapath_ciatables = 'ciatables/'

xk.Settings().set_search_path(datapath_ktables, path_type='ktable')
xk.Settings().set_search_path(datapath_ciatables, path_type='cia')

_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
k_db = xk.Kdatabase(['CO2','CH4'], '')

new_wn_points_right=np.array([
8140, 8160, 8180, 8200, 8220, 8240, 8260, 8280, 8300, 8320, 8340, 8360, 8380, 8400, 8420,
8440, 8460, 8480, 8500, 8520, 8540, 8560, 8580, 8600, 8620, 8640, 8660, 8680, 8700, 8720,
8740, 8760, 8780, 8800, 8820, 8840, 8860, 8880, 8900, 8920, 8940, 8960, 8980, 9000, 9020,
9040, 9060, 9080, 9100, 9120, 9140, 9160, 9180, 9200, 9220, 9240, 9260, 9280, 9300, 9320,
9340, 9360, 9380, 9400, 9420, 9440, 9460, 9480, 9500, 9520, 9540, 9560, 9580, 9600, 9620,
9640, 9660, 9680, 9700, 9720, 9740, 9760, 9780, 9800, 9820, 9840, 9860, 9880, 9900, 9920,
9940, 9960, 9980, 10000, 10020, 10040, 10060, 10080, 10100, 10120, 10140, 10160, 10180, 10200,
10220, 10240, 10260, 10280, 10300, 10320, 10340, 10360, 10380, 10400, 10420, 10440, 10460, 10480,
10500, 10520, 10540, 10560, 10580, 10600, 10620, 10640, 10660, 10680, 10700, 10720, 10740, 10760,
10780, 10800, 10820, 10840, 10860, 10880, 10900, 10920, 10940, 10960, 10980, 11000, 11020, 11040,
11060, 11080, 11100, 11120, 11140, 11160, 11180, 11200, 11220, 11240, 11260, 11280, 11300, 11320,
11340, 11360, 11380, 11400, 11420, 11440, 11460, 11480, 11500, 11520, 11540, 11560, 11580, 11600,
11620, 11640, 11660, 11680, 11700, 11720, 11740, 11760, 11780, 11800, 11820, 11840, 11860, 11880,
11900, 11920, 11940, 11960, 11980, 12000, 12020, 12040, 12060, 12080, 12100, 12120, 12140, 12160,
12180, 12200, 12220, 12240, 12260, 12280, 12300, 12320, 12340, 12360, 12380, 12400, 12420, 12440,
12460, 12480, 12500, 12520, 12540, 12560, 12580, 12600, 12620, 12640, 12660, 12680, 12700, 12720,
12740, 12760, 12780, 12800, 12820, 12840, 12860, 12880, 12900, 12920, 12940, 12960, 12980, 13000,
13020, 13040, 13060, 13080, 13100, 13120, 13140, 13160, 13180, 13200, 13220, 13240, 13260, 13280,
13300, 13320
])


k_db['CO2'].extend_spectral_range(wngrid_right=new_wn_points_right,remove_zeros=True)

Res = 10. 
wn_grid_LR = xk.wavenumber_grid_R(100., 35000., Res)


k_db = xk.Kdatabase(['CO2', 'CH4'], '')
k_db_LR = k_db.bin_down_cp(wnedges=wn_grid_LR, remove_zeros=True)

cia_db_LR = xk.CIAdatabase(molecules=['CO2', 'CH4', 'H2'], mks=True)
cia_db_LR.sample(k_db_LR.wns)
sys.stdout = _stdout



