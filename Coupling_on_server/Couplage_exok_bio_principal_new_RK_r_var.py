#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import pickle
import time
from IPython.utils import io
import xarray as xr

# Imports des modules
from Main_evolution_function_RK import *
from EXO_K_wo_ocean import *
from Constants import *
from Functions2 import *
from Functions import *
from Bio_model import *

import gc

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator


# Charger le fichier CSV
df = pd.read_csv("Grid_values_ini.csv")  # Adaptez le chemin si besoin

task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # par défaut 0 si aucun argument

R_list = 10**np.linspace(-6,0,50)


# Extraire les colonnes d'entrée et de sortie
points = df[['P', 'T', 'fG']].values
NCeq = df['NCeq'].values
cH = df['cH'].values
cC = df['cC'].values
cG = df['cG'].values
cN = df['cN'].values

# Création des interpolateurs
interp_NCeq = LinearNDInterpolator(points, NCeq)
interp_cH = LinearNDInterpolator(points, cH)
interp_cC = LinearNDInterpolator(points, cC)
interp_cG = LinearNDInterpolator(points, cG)
interp_cN = LinearNDInterpolator(points, cN)

# Fonction d'interpolation
def interp_conc(P, T, fG):
    pt = np.array([[P, T, fG]])
    return {
        'NC': interp_NCeq(pt)[0],
        'cH': interp_cH(pt)[0],
        'cC': interp_cC(pt)[0],
        'cG': interp_cG(pt)[0],
        'cN': interp_cN(pt)[0]
    }

t_beginning = time.time()

tf = 1e5 #jours
print("Volume of the ocean :",format(550*Mars_surface,".2e"),"m^3")
print("Ratio of the ocean over Mars surface :",round(fraction_ocean,4))
T_surf = 300
Ptop = 1.0 #Pa
Psurf = 2e5 #Pa
fH = 0.15
fG = 0
fN = 0.04
fC = 1 - fN - fG - fH

print('=================================== First equilibrium =================================================')
print('Calculating atmospheric equilibrium...')

with io.capture_output() as _:
    evol_mars = defining_atm(fH, fG, fN, Ptop, Psurf, T_surf,acceleration=None)
    evol_mars.equilibrate(Fnet_tolerance=0.1, verbose=False)
teq,Teq,T_profile,P_profile = equilibrium_state_values(evol_mars)
del evol_mars
gc.collect()
print('Atmospheric equilibrium : P_surf =',round(P_profile[-1]/1e5,5),'bar',' and T_surf = ',round(Teq,2),'K')
Z,dZ = compute_altitudes(P_profile, T_profile, 0.03706)
ntotgasfinal = atm_matter_quantity(P_profile, T_profile,dZ,0.03706, Mars_surface)

Teq_list = [Teq]
t_list = [0]
bio_all = []
atmo_all = []
times_all = []
press_all = []
medium_all = []
timesp_all = []
X_all = []
cH,cC,cG,cN = 0,0,0,0
NC = 1e4

dico_ini={'NC':1e4,
         'cH':0,
         'cC':0,
         'cG':0,
         'cN':0}

Traits = ReturnTraits(Teq, Par, 1)
X0 = 2 * Traits[3]
niter1 = 50
niter2 = 150
CH4_threshold_list =  np.concatenate([10**np.linspace(-7,-3,niter1),np.linspace(1.001e-3,0.045,niter2)])

if task_id < len(R_list):
    r = R_list[task_id]
else : raise ValueError("task_id >= len(R_list)")

if Teq > 271:
    for i in range(niter1+niter2):
        print('=================================== Iteration',i+1,'/',niter1+niter2,'===============================================')

        fgt = float(CH4_threshold_list[i])

        Bio, Medium,Atmo, t,flux,flux_times,Pression=system_evolution_RK(Psurf, Teq, 550*Mars_surface,r * fraction_ocean, 1e6, Mars_surface, 
                                                             ntotgasfinal,X0,CH4_frac_threshold=fgt,
                                                focean=fraction_ocean, Atm_compo=(fH, fC, fG,fN),
                 concentration_ini=(dico_ini['cH'],dico_ini['cC'],dico_ini['cG'],dico_ini['cN']),
                                NC_0=dico_ini['NC']/(11*r), rtol=1e-7, atol=1e-20,methode='LSODA',N=int(1e5),firststep=None,minstep=np.nan)



        atmo_all.append(Atmo)
        times_all.append(t)
        bio_all.append(Bio[0])
        X_all.append(Bio[1])
        press_all.append(Pression)
        medium_all.append(Medium)
        timesp_all.append(flux_times)
        tfinal = t[-1]
        ntotgasfinal = np.sum(Atmo[:, -1])
        fG = Atmo[2][-1] / ntotgasfinal
        fC = Atmo[1][-1] / ntotgasfinal
        fH = Atmo[0][-1] / ntotgasfinal
        fN = Atmo[3][-1] / ntotgasfinal
        X0 = Bio[1][-1]
        cG = Medium[2][-1] / (Mars_surface*550*1e3)
        cC = Medium[1][-1] / (Mars_surface*550*1e3)
        cH = Medium[0][-1] / (Mars_surface*550*1e3)
        cN = Medium[3][-1] / (Mars_surface*550*1e3)

        Psurf = Pression[-1]

        t_list.append(t_list[-1]+tfinal)
        print('fG',round(fG,6))
        print('Calculating atmospheric equilibrium...')
        CH4_threshold = CH4_threshold_list[i]

        with io.capture_output() as _:
            evol_mars = defining_atm(fH, fG, fN, Ptop, Psurf, T_surf,acceleration=None)
            evol_mars.equilibrate(Fnet_tolerance=0.1, verbose=False)


        teq,Teq,T_profile,P_profile = equilibrium_state_values(evol_mars)
        T_surf=Teq
        print('Atmospheric equilibrium : P_surf =',round(P_profile[-1]/1e5,5),'bar',' and T_surf = ',round(Teq,2),'K')
        print("Actual time in years",round(t_list[-1]/(365.25),2))
        del evol_mars
        gc.collect()
        NC = Bio[0][-1]

        dico_ini = interp_conc(Psurf,Teq,fG)
        if not all(math.isfinite(v) for v in dico_ini.values()): print("dico_ini change") ; dico_ini = {'NC':Bio[0][-1],
			'cH':cH,
			'cC':cC,
			'cG':cG,
			'cN':cN}
        Teq_list.append(Teq)


        if Teq < 271:
            break




    t_end = time.time()

    data = {
        "Teq_list": Teq_list,
        "press_all": press_all,
        "t_list": t_list,
        "bio_all": bio_all,
        "medium_all": medium_all,
        "atmo_all": atmo_all,
        "times_all": times_all,
        "timesp_all":timesp_all

    }

    

    filename = f"Lib_Couplage_exok_bio_file_r_{r}.pkl"
    path = "Results_couplage_1D\\"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print('=========================================================================')
    print('               temps total',round(t_end-t_beginning,2),'s')
    print('=========================================================================')
    print('=========================================================================')
    print('                           ***     END !     ***                         ')
    print('=========================================================================')
else: print("No life possible")

