#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import contextlib
import pandas as pd
import signal

# Imports des modules
from Main_evolution_function_RK import *
from EXO_K_wo_ocean import *
from Functions import *
from Bio_model import *


T_list = np.linspace(270,327,30)
P_array = np.linspace(1.2e5,2.1e5,10)
fG_list= np.linspace(0,0.0441,20)

fH0 = 0.15
fN0 = 0.04
fC0 = 0.81

task_id = int(sys.argv[1])

if task_id > len(P_array):
    raise ValueError("task_id > len(P_array)")

results = []
i = 0

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

for P in [P_array[task_id]]:
    for T in T_list:
        Traits = ReturnTraits(T, Par, 1)
        X0 = 2 * Traits[3]
        for fG in fG_list:
            print(i+1, '/ 1280')
            i += 1

            x = fG / (1 + 4 * fG)
            fH = (fH0 - 4 * x) / (1 - 4 * x)
            fN = fN0 / (1 - 4 * x)
            fC = (fC0 - x) / (1 - 4 * x)

            if T > 310 and fG > 0.03:
                continue

            try:
                signal.alarm(60)  # ⏱️ Limite à 60 secondes
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
                signal.alarm(0)  # ⏹️ Annule l’alarme

            except TimeoutException:
                print(f"⏱ Timeout à l’itération {i}, T={T}, fG={fG}")
                continue
            except Exception as e:
                print(f"⚠ Erreur : {e}")
                continue

            NCeq = np.mean(Bio[0][np.where(t > (t[-1] - 100))])
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

# Conversion en DataFrame
df = pd.DataFrame(results)

# Enregistrement dans un fichier CSV
df.to_csv(f"Grid_initialisation_integration_P_{round(P_array[task_id]*1e-5,3)*100000}.csv", index=False)


