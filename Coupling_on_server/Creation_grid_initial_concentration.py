#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

# Import standard libraries
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import contextlib
import pandas as pd
import signal

# Import custom modules
from Main_evolution_function_RK import *
from EXO_K_wo_ocean import *
from Functions import *
from Bio_model import *

# Define the range of temperatures, pressures, and methane fractions
T_list = np.linspace(270,327,30)               # Surface temperature range [K]
P_array = np.linspace(1.2e5,2.1e5,10)           # Surface pressure range [Pa]
fG_list= np.linspace(0,0.0441,20)               # Methane volume mixing ratios

# Define initial atmospheric fractions (excluding CH₄)
fH0 = 0.15                                      # Initial hydrogen fraction
fN0 = 0.04                                      # Initial nitrogen fraction
fC0 = 0.81                                      # Initial carbon fraction

# Get the task index from command line argument
task_id = int(sys.argv[1])

# Safety check: task ID must be within array bounds
if task_id > len(P_array):
    raise ValueError("task_id > len(P_array)")

# Initialize result container
results = []
i = 0  # Iteration counter

# Define custom exception for timeout handling
class TimeoutException(Exception):
    pass

# Signal handler for timeout
def handler(signum, frame):
    raise TimeoutException()

# Assign signal handler to alarm signal
signal.signal(signal.SIGALRM, handler)

# Loop over selected pressure (based on task_id)
for P in [P_array[task_id]]:
    for T in T_list:
        # Compute biological traits at given temperature
        Traits = ReturnTraits(T, Par, 1)
        X0 = 2 * Traits[3]  # Initial biomass

        for fG in fG_list:
            print(i+1, '/ 1280')  # Show progress
            i += 1

            # Calculate adjusted atmospheric fractions given methane content
            x = fG / (1 + 4 * fG)
            fH = (fH0 - 4 * x) / (1 - 4 * x)
            fN = fN0 / (1 - 4 * x)
            fC = (fC0 - x) / (1 - 4 * x)

            # Skip simulations for uninhabitable scenarios (e.g., runaway temperature with high CH₄)
            if T > 310 and fG > 0.03:
                continue

            try:
                signal.alarm(60)  # ⏱️ Set timeout limit to 60 seconds

                # Run the system evolution model with defined parameters
                Bio, Medium, Atmo, t, flux, flux_times, P_final = system_evolution_RK(
                    P, T, 550 * Mars_surface, 1 / 11 * fraction_ocean, 2000,
                    Mars_surface,
                    1.75e20, X0,
                    focean=fraction_ocean,
                    Atm_compo=(fH, fC, fG, fN),
                    concentration_ini=(0, 0, 0, 0), NC_0=1e4,
                    rtol=1e-12, atol=1e-20, methode='LSODA',
                    N=int(1e5), firststep=None, minstep=np.nan
                )

                signal.alarm(0)  # ⏹️ Disable alarm after success

            except TimeoutException:
                print(f"⏱ Timeout at iteration {i}, T={T}, fG={fG}")
                continue
            except Exception as e:
                print(f"⚠ Error: {e}")
                continue

            # Extract final values after equilibrium
            NCeq = np.mean(Bio[0][np.where(t > (t[-1] - 100))])  # Average number of cells over last 100 units of time

            # Store results in list
            results.append({
                'P': P,
                'T': T,
                'fG': fG,
                'NCeq': NCeq,
                'cH': np.mean(Medium[0][np.where(t > (t[-1] - 100))]) / (Mars_surface * 550 * 1e3),
                'cC': np.mean(Medium[1][np.where(t > (t[-1] - 100))]) / (Mars_surface * 550 * 1e3),
                'cG': np.mean(Medium[2][np.where(t > (t[-1] - 100))]) / (Mars_surface * 550 * 1e3),
                'cN': np.mean(Medium[3][np.where(t > (t[-1] - 100))]) / (Mars_surface * 550 * 1e3)
            })

# Convert result list to DataFrame
df = pd.DataFrame(results)

# Save DataFrame to CSV file (named by pressure)
df.to_csv(f"Grid_initialisation_integration_P_{round(P_array[task_id]*1e-5,3)*100000}.csv", index=False)
