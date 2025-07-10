# How to Launch the 1D Bio‑Climatic Coupling on a Server

To launch the coupling on a server, simply upload this folder along with the `Ktables` and `ciatables` folders. Then navigate through the files in this directory and adjust the paths to match the server’s architecture. The codes will then be ready to run.

---

## 📄 Table of Contents
- [1. Biology‑related files](#1-biology-related-files)  
- [2. Useful coupling functions (physics)](#2-useful-coupling-functions-physics)  
  - [2.1. EXO_K_wo_ocean.py](#21-exo_k_wo_oceanpy)  
  - [2.2. Functions2.py](#22-functions2py)  
  - [2.3. Main_evolution_function_RK.py](#23-main_evolution_function_rkpy)  
  - [2.4. Photochem.py](#24-photochempy)  
  - [2.5. Interface_atmosphere_ocean.py](#25-interface_atmosphere_oceanpy)  
- [3. Data grids](#3-data-grids)  
- [4. Coupling scripts](#4-coupling-scripts)  
  - [4.1. BioClim_coupling_niter.py](#41-bioclim_coupling_niterpy)  
  - [4.2. BioClim_coupling_Vocean.py](#42-bioclim_coupling_voceanpy)  
  - [4.3. BioClim_coupling_r.py](#43-bioclim_coupling_rpy)  
- [5. Generating the initial concentration grid](#5-generating-the-initial-concentration-grid)

---

## 1. Biology‑related files

These files were largely written by Boris Sauterey and simulate biological evolution from a thermodynamic perspective. **DO NOT MODIFY THEM**.

The files are:
- `Acetogens1.py`
- `Bio.py`
- `Bio_model.py`
- `Bio_model_parameter.py`
- `Constants.py`
- `Functions.py`
- `Methanogens.py`

---

## 2. Useful coupling functions (physics)

### 2.1. `EXO_K_wo_ocean.py`

Defines atmosphere parameters using *Exok*. You can modify gravity, star temperature, etc. between lines 26–32 (see [Exok documentation](https://perso.astrophy.u-bordeaux.fr/~jleconte/exo_k-doc/tutorial-atm.html)). The script imports corrk and cia tables, extends them spectrally, and downscales resolution. **Only update lines 122 and 123** to point to your `Ktables` and `ciatables` directories.

### 2.2. `Functions2.py`

Contains physical functions to compute altitude from pressure and temperature, atmospheric mole quantities, and pressure profiles. See inline comments for details. **NO CHANGE REQUIRED**.

### 2.3. `Main_evolution_function_RK.py`

Implements `system_evolution_RK` to evolve atmosphere and ocean composition under biological influence, using an adaptive time-step solver (LSODA method). **NO CHANGE REQUIRED**.

### 2.4. `Photochem.py`

Interpolates the `Grid_no_header.csv` grid to return CH₄ and H₂ fluxes in the atmosphere, based on Zahnle & Kasting. **NO CHANGE REQUIRED**.

### 2.5. `Interface_atmosphere_ocean.py`

Returns interface fluxes using a stagnant boundary layer model (Karecha et al. 2005). **NO CHANGE REQUIRED**.

---

## 3. Data grids

- `Grid_no_header.csv`: used for CH₄ and H₂ atmospheric fluxes.  
- `Grid_values_ini.csv`: interpolated to give initial concentrations of cells and methanogens in the ocean.  
**NO CHANGE REQUIRED**.

---

## 4. Coupling scripts

These scripts perform multiple coupling simulations by varying three parameters: ocean volume ($V_{ocean}$), ratio ($r$), and number of climate‑model calls ($niter$). All outputs are ~$200$ MB files.

### 4.1. `BioClim_coupling_niter.py`

Varies the number of climate-model calls. For each `niter`, produces `Lib_Couplage_exok_bio_file_niter1_{niter1}_niter2_{niter2}.pkl` where:
- `niter1`: log‑spaced CH₄ thresholds between $10^{-7}$ and $10^{-3}$.
- `niter2`: linear‑spaced thresholds between $1.001\times 10^{-3}$ and $0.045$.

Use `Coupling_niter.sbatch` to run 88 parallel jobs. See script comments for details.

### 4.2. `BioClim_coupling_Vocean.py`

Varies ocean volume for a constant surface area, focusing on cells within the top 50 m globally. Outputs `Lib_Couplage_exok_bio_file_GEL_{GEL}.pkl` (GEL = global equivalent layer in meters). Launched via `Coupling_Vocean.sbatch`. Structurally similar to the $niter$ variant—refer to that for guidance.

### 4.3. `BioClim_coupling_r.py`

Varies the ratio of cell volume to total ocean volume : $r = \frac{V_{cells}}{V_{ocean}}$

Outputs `Lib_Couplage_exok_bio_file_r_{r}.pkl`. Launched via `Coupling_r.sbatch`. Same structure as above—see the $niter$ version.

---

## 5. Generating the initial concentration grid

`Creation_grid_initial_concentration.py` generates equilibrium concentration grids for a given pressure *P*, 30 temperatures (T) evenly spaced from 270 K to 327 K, and 20 CH₄ concentrations from 0 to 0.0441. Running `creation_grid.sbatch` across 10 pressures produces CSV grids named `Grid_initialisation_integration_P_{P}.csv`. You can then merge them into `Grid_values_ini.csv`, which contains concentrations of H₂, CO₂, CH₄, N₂ and methanogens for each (P, T, fCH₄) tuple.

⚠️ You **don't need** to regenerate this grid to run the coupling—it is already provided. Only regenerate it if you wish to extend the range or resolution of P, T, or fCH₄.

---

## ✅ Summary: How to launch

1. Upload all folders (`Coupling_on_server`, `Ktables`, `ciatables`) to the server.  
2. Adjust paths in all scripts to match the server environment.  
3. Use the appropriate `*.py` and `*.sbatch` scripts for your parameter study ($niter$, $V_{ocean}$, or $r$).  
4. Submit jobs via `sbatch`.  
5. Analyze the resulting `.pkl` files using the Jupyter notebooks in the [`Local_coupling_computation`](../Local_coupling_computation) section.
