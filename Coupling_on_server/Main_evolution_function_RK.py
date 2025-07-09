#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import sys
from Bio_model import *
from Constants import *
from Functions import *
from Photochem import *
from alive_progress import alive_bar
import time
from Interface_atmosphere_ocean import *
from Functions2 import *
from alive_progress import config_handler
from functools import partial
from scipy.integrate import solve_ivp

config_handler.set_global(spinner='classic', bar='classic', stats=False, force_tty=True)

def system_evolution_RK(P, T, Vwater, ratio, tf, surface, ntot,X0, CH4_frac_threshold=1, focean=0.3,
                     Atm_compo=(0.15, 0.85, 0, 0), concentration_ini=(0., 0., 0., 0.),
                     NC_0=10, rtol=1e-9, atol=1e-10, methode='LSODA', N=int(1e4),
                     firststep=None, minstep=1e-8):
    
    fluxes = {
        'dH_int': [], 'dC_int': [], 'dG_int': [], 'dN_int': [],
        'dH_atm': [], 'dG_atm': []
    }
    flux_times = []
    pressures = []
    Traits = ReturnTraits(T, Par, 1)
    concentration_ini = np.array(concentration_ini)
    nwat_H0, nwat_C0, nwat_G0, nwat_N0 = concentration_ini * (1e3 * Vwater)

    natm_H0 = ntot * Atm_compo[0]
    natm_C0 = ntot * Atm_compo[1]
    natm_G0 = ntot * Atm_compo[2]
    natm_N0 = ntot * Atm_compo[3]

    P0 = P
    ntot_initial = natm_H0 + natm_C0 + natm_G0 + natm_N0

    y0 = np.array([NC_0, X0, nwat_H0, nwat_C0, nwat_G0, nwat_N0,
                   natm_H0, natm_C0, natm_G0, natm_N0])

    scale_y = np.maximum(np.abs(y0), 1e-20)
    y0_scaled = y0 / scale_y

    nb_points = N
    t_eval = np.linspace(0, tf, nb_points)

    progress = {'index': 0}

    def stop_on_CH4_frac(t, y):
        y_phys = y * scale_y
        natm_H, natm_C, natm_G, natm_N = y_phys[6], y_phys[7], y_phys[8], y_phys[9]
        ntot = natm_H + natm_C + natm_G + natm_N
        frac_CH4 = natm_G / max(ntot, 1e-30)
        return frac_CH4 - CH4_frac_threshold

    stop_on_CH4_frac.terminal = True
    stop_on_CH4_frac.direction = 1

    def wrapped_rhs(t, y):
        y_phys = y * scale_y

        if wrapped_rhs.bar is not None:
            current_index = np.searchsorted(t_eval, t, side='right')
            if current_index > progress['index']:
                wrapped_rhs.bar()
                progress['index'] = current_index

        flux_times.append(t)

        NC, X, nwat_H, nwat_C, nwat_G, nwat_N, natm_H, natm_C, natm_G, natm_N = y_phys
        Atm_compo = (natm_H, natm_C, natm_G, natm_N)
        ntot = sum(Atm_compo)

        P = P0 * ntot / ntot_initial

        fH, fC, fG, fN = [n / max(ntot, 1e-30) for n in Atm_compo]

        Vcells = ratio * Vwater
        H = max(nwat_H / (1e3 * Vwater), 1e-10)
        C = max(nwat_C / (1e3 * Vwater), 1e-10)
        G = max(nwat_G / (1e3 * Vwater), 1e-10)
        N = max(nwat_N / (1e3 * Vwater), 1e-10)
        vec = np.array([H, C, G, N])

        dgcat, dNC, dX, qana, qcat, _, _, Catab, Anab = step_Bio(vec, T, "meth", NC, X, Traits, 1)

        dH_bio = NC * (Catab[0]*qcat + Anab[0]*qana) * Vcells/Vwater
        dC_bio = NC * (Catab[1]*qcat + Anab[1]*qana) * Vcells/Vwater
        dG_bio = NC * (Catab[3]*qcat + Anab[3]*qana) * Vcells/Vwater
        dN_bio = NC * (Catab[9]*qcat + Anab[9]*qana) * Vcells/Vwater

        factor = surface * 1e4 * 86400 * focean
        factorB = 86400 * surface * 1e4 / Av

        dH_int = interface_flow(pvH, alphaH, fH * P * 1e-5, H) * factor
        dC_int = interface_flow(pvC, alphaC(T), fC * P * 1e-5, C) * factor
        dG_int = interface_flow(pvG, alphaG, fG * P * 1e-5, G) * factor
        dN_int = interface_flow(pvN, alphaN, fN * P * 1e-5, N) * factor

        dH_atm = FH2_func(fH * P * 1e-5, fG * P * 1e-5) * factorB
        dG_atm = FCH4_func(fH * P * 1e-5, fG * P * 1e-5) * factorB

        fluxes['dH_int'].append(dH_int)
        fluxes['dC_int'].append(dC_int)
        fluxes['dG_int'].append(dG_int)
        fluxes['dN_int'].append(dN_int)
        fluxes['dH_atm'].append(dH_atm)
        fluxes['dG_atm'].append(dG_atm)

        pressures.append(P)

        dydt_phys = np.array([
            dNC, dX,
            dH_bio * 1e3 * Vwater + dH_int,
            dC_bio * 1e3 * Vwater + dC_int,
            dG_bio * 1e3 * Vwater + dG_int,
            dN_bio * 1e3 * Vwater + dN_int,
            -dH_int + dH_atm,
            -dC_int,
            -dG_int + dG_atm,
            -dN_int
        ])

        return dydt_phys / scale_y  

    wrapped_rhs.bar = None

    with alive_bar(nb_points, title='Integration') as bar:
        wrapped_rhs.bar = bar
        
        if methode=="LSODA":
            solution = solve_ivp(
                fun=wrapped_rhs,
                t_span=(0, tf),
                y0=y0_scaled,
                t_eval=t_eval,
                method=methode,
                rtol=rtol,
                atol=atol,
                events=stop_on_CH4_frac,
                first_step=firststep,
                min_step=minstep
            )
        else:
            solution = solve_ivp(
                fun=wrapped_rhs,
                t_span=(0, tf),
                y0=y0_scaled,
                t_eval=t_eval,
                method=methode,
                rtol=rtol,
                atol=atol,
                events=stop_on_CH4_frac,
                first_step=firststep,
            )

    y_scaled = solution.y * scale_y[:, np.newaxis]
    return y_scaled[0:2], y_scaled[2:6], y_scaled[6:], solution.t, fluxes, flux_times, pressures
