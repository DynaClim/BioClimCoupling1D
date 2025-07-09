#!/usr/bin/env python
# coding: utf-8

# Import necessary modules
import sys
import os
import numpy as np
import pickle
import time
from IPython.utils import io  # Used to suppress output
import xarray as xr  # For handling multi-dimensional data arrays

# Importing custom modules and functions from local files
from Main_evolution_function_RK import *
from EXO_K_wo_ocean import *
from Constants import *
from Functions2 import *
from Functions import *
from Bio_model import *

# Garbage collection and numerical tools
import gc
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator  # For 3D interpolation

# Define grid of iteration number values
x_vals = np.linspace(5, 50, num=8, dtype=int)
y_vals = np.linspace(30, 200, num=22, dtype=int)
doublets = [(x, y) for x in x_vals for y in y_vals] # Cartesian product of the two

# Load initial condition data for the grid
df = pd.read_csv("Grid_values_ini.csv")

# Get the task ID from command line argument, default to 0 if not provided
task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0 

# Check if task ID is within the grid bounds
if task_id < len(doublets): 
    niter1, niter2 = doublets[task_id]
else: 
    raise ValueError("Too many tasks")

# Extract values and prepare interpolation for chemical concentrations
points = df[['P', 'T', 'fG']].values  # Grid points
NCeq = df['NCeq'].values
cH = df['cH'].values
cC = df['cC'].values
cG = df['cG'].values
cN = df['cN'].values

# Interpolators for different species and total cells
interp_NCeq = LinearNDInterpolator(points, NCeq)
interp_cH = LinearNDInterpolator(points, cH)
interp_cC = LinearNDInterpolator(points, cC)
interp_cG = LinearNDInterpolator(points, cG)
interp_cN = LinearNDInterpolator(points, cN)

# Interpolation function
def interp_conc(P, T, fG):
    pt = np.array([[P, T, fG]])
    return {
        'NC': interp_NCeq(pt)[0],
        'cH': interp_cH(pt)[0],
        'cC': interp_cC(pt)[0],
        'cG': interp_cG(pt)[0],
        'cN': interp_cN(pt)[0]
    }

# Start the simulation timer
t_beginning = time.time()

# Print initial physical constants
print("Volume of the ocean :", format(550*Mars_surface, ".2e"), "m^3")
print("Ratio of the ocean over Mars surface :", round(fraction_ocean, 4))

# Initial surface temperature and pressure
T_surf = 300
Ptop = 1.0        # Pressure at top of atmosphere in Pa
Psurf = 2e5       # Surface pressure in Pa
fH = 0.15         # Hydrogen fraction
fG = 0            # Methane fraction
fN = 0.04         # Nitrogen fraction
fC = 1 - fN - fG - fH  # Carbon fraction

# First equilibrium calculation
print('=================================== First equilibrium =================================================')
print('Calculating atmospheric equilibrium...')

# Compute initial atmospheric equilibrium state
with io.capture_output() as _:
    evol_mars = defining_atm(fH, fG, fN, Ptop, Psurf, T_surf, acceleration=None)
    evol_mars.equilibrate(Fnet_tolerance=0.1, verbose=False)

# Extract equilibrium values
teq, Teq, T_profile, P_profile = equilibrium_state_values(evol_mars)
del evol_mars  # Free memory
gc.collect()

# Output equilibrium results
print('Atmospheric equilibrium : P_surf =', round(P_profile[-1]/1e5, 5), 'bar', ' and T_surf = ', round(Teq, 2), 'K')

# Compute atmospheric altitudes and total gas amount
Z, dZ = compute_altitudes(P_profile, T_profile, 0.03706)
ntotgasfinal = atm_matter_quantity(P_profile, T_profile, dZ, 0.03706, Mars_surface)

# Initialize output containers
Teq_list = [Teq]
t_list = [0]
bio_all = []
atmo_all = []
times_all = []
press_all = []
medium_all = []
timesp_all = []
X_all = []

# Initialize concentrations
cH, cC, cG, cN = 0, 0, 0, 0
NC = 1e4

# Initial concentration dictionary
dico_ini = {
    'NC': 1e4,
    'cH': 0,
    'cC': 0,
    'cG': 0,
    'cN': 0
}

# Get biological traits at equilibrium temperature
Traits = ReturnTraits(Teq, Par, 1)
X0 = 2 * Traits[3]  # Initial biomass

# Build methane concentration threshold list
CH4_threshold_list = np.concatenate([
    10**np.linspace(-7, -3, niter1), 
    np.linspace(1.001e-3, 0.045, niter2)
])

# Run the coupled evolution only if temperature is suitable for life
if Teq > 271:
    for i in range(niter1 + niter2):
        print('=================================== Iteration', i+1, '/', niter1+niter2, '===============================================')

        fgt = float(CH4_threshold_list[i])  # Set CH4 threshold

        # Run main evolution model
        Bio, Medium, Atmo, t, flux, flux_times, Pression = system_evolution_RK(
            Psurf, Teq, 550*Mars_surface, 1/11 * fraction_ocean, 1e6, Mars_surface, 
            ntotgasfinal, X0, CH4_frac_threshold=fgt,
            focean=fraction_ocean, 
            Atm_compo=(fH, fC, fG, fN),
            concentration_ini=(dico_ini['cH'], dico_ini['cC'], dico_ini['cG'], dico_ini['cN']),
            NC_0=dico_ini['NC'], 
            rtol=1e-7, atol=1e-20,
            methode='LSODA', N=int(1e5), firststep=None, minstep=np.nan
        )

        # Store results from this iteration
        atmo_all.append(Atmo)
        times_all.append(t)
        bio_all.append(Bio[0])
        X_all.append(Bio[1])
        press_all.append(Pression)
        medium_all.append(Medium)
        timesp_all.append(flux_times)

        # Update total gas and composition
        tfinal = t[-1]
        ntotgasfinal = np.sum(Atmo[:, -1])
        fG = Atmo[2][-1] / ntotgasfinal
        fC = Atmo[1][-1] / ntotgasfinal
        fH = Atmo[0][-1] / ntotgasfinal
        fN = Atmo[3][-1] / ntotgasfinal
        X0 = Bio[1][-1]
        cG = Medium[2][-1] / (Mars_surface * 550 * 1e3)
        cC = Medium[1][-1] / (Mars_surface * 550 * 1e3)
        cH = Medium[0][-1] / (Mars_surface * 550 * 1e3)
        cN = Medium[3][-1] / (Mars_surface * 550 * 1e3)
        Psurf = Pression[-1]

        t_list.append(t_list[-1] + tfinal)

        print('fG', round(fG, 6))
        print('Calculating atmospheric equilibrium...')

        # Recalculate atmospheric equilibrium
        CH4_threshold = CH4_threshold_list[i]
        with io.capture_output() as _:
            evol_mars = defining_atm(fH, fG, fN, Ptop, Psurf, T_surf, acceleration=None)
            evol_mars.equilibrate(Fnet_tolerance=0.1, verbose=False)

        # Get new equilibrium state
        teq, Teq, T_profile, P_profile = equilibrium_state_values(evol_mars)
        T_surf = Teq

        print('Atmospheric equilibrium : P_surf =', round(P_profile[-1]/1e5, 5), 'bar', ' and T_surf = ', round(Teq, 2), 'K')
        print("Actual time in years", round(t_list[-1]/365.25, 2))

        del evol_mars
        gc.collect()

        NC = Bio[0][-1]  # Update cell count

        # Update initial concentrations from interpolation
        dico_ini = interp_conc(Psurf, Teq, fG)

        # If any interpolated values are not finite, fallback to last known values
        if not all(math.isfinite(v) for v in dico_ini.values()): 
            print("dico_ini change")
            dico_ini = {
                'NC': Bio[0][-1],
                'cH': cH,
                'cC': cC,
                'cG': cG,
                'cN': cN
            }

        Teq_list.append(Teq)

        # Stop loop if temperature drops below habitable threshold
        if Teq < 271:
            break

    # End of loop — save results
    t_end = time.time()

    # Build data dictionary to pickle
    data = {
        "Teq_list": Teq_list,
        "press_all": press_all,
        "t_list": t_list,
        "bio_all": bio_all,
        "medium_all": medium_all,
        "atmo_all": atmo_all,
        "times_all": times_all,
        "timesp_all": timesp_all
    }

    # Save to file
    filename = f"Lib_Couplage_exok_bio_file_niter1_{niter1}_niter2_{niter2}.pkl"
    path = ""
    with open(path+filename, 'wb') as f:
        pickle.dump(data, f)

    # Print summary
    print('=========================================================================')
    print('               Total time', round(t_end - t_beginning, 2), 's')
    print('=========================================================================')
    print('=========================================================================')
    print('                           ***     END !     ***                         ')
    print('=========================================================================')

# If initial equilibrium temperature is too low, life cannot emerge
else: 
    print("No life possible")


