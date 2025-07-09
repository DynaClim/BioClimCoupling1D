import numpy as np

###Environmental constants
Av           = 6e23                  # Avogadro number
R            = 8.31                  # Perfect gaz constant J.(K.mol)-1
S            = 1.4e18                # Ocean surface (cm2)
g            = 3.711                 # Gravitational constant (m.s-2)
M_CO2        = 44e-3                 # CO2 molar mass (kg.mol-1)
M_N2         = 28e-3                 # N2 molar mass (kg.mol-1)
M_CH4        = 16e-3                 # CH4 molar mass (kg.mol-1)
M_H2         = 2e-3                  # H2 molar mass (kg.mol-1)
kb           = 1.38e-23              # Boltzmann constant (J/K)
Mars_mass    = 6.39e23               # Mass of Mars (kg)
G            = 6.67e-11              # Gravitational constant (m3.kg-1.s-2)
Mars_radius  = 3.39e6                # Mean raduis of Mars (m)

### Physical properties of elements
# Molar mass (g.mol-1)
MH         = 1
MC         = 12
MG         = 16
MCO        = 28
MCO2       = 44
MN2        = 28
MCH3COOH   = 60

# Solubilities (mol.L-1.bar-1)
alphaH     = 7.8e-4
alphaG     = 1.4e-3
alphaCO    = 1e-3
alphaN2    = 7e-4

# Mars related constants calculated using a diagfi.nc file

import xarray as xr 


# HERE WE OPEN THE NETCDF FILE
diagfi_path = '/home/users/m/meyerfra/earlymars_FX/Biomodel_1D/GCM/'
data = xr.open_dataset(diagfi_path+'diagfi_earlymars_64x48x26_PC_newreact2_36Y_38Y.nc',

                       decode_times=False)


starfi = xr.open_dataset(diagfi_path+'startfi.nc',

                       decode_times=False)

Time=data['Time']
lat=data['latitude']
lon=data['longitude']

aire=data['aire']

Mars_surface = float(np.sum(aire)) # Surface of Mars (m²)

rnat=data['rnat']

rnat_ponderee = rnat * aire
somme_spatiale = rnat_ponderee.sum(dim=["latitude", "longitude"])
moyenne_spatiale = somme_spatiale / Mars_surface
rnat_global_mean = moyenne_spatiale.mean(dim="Time").item()
fraction_ocean = float(1-rnat_global_mean)

asr = data['ASR']
asr_ponderee = asr * aire
somme_spatiale = asr_ponderee.sum(dim=["latitude", "longitude"])
moyenne_spatiale = somme_spatiale / Mars_surface
asr_global_mean = 2*float(moyenne_spatiale.mean(dim="Time").item())

startfi = xr.open_dataset(diagfi_path+'startfi.nc',

                       decode_times=False)

from scipy.interpolate import RegularGridInterpolator

rnat_mean = data['rnat'].mean(dim='Time')
rnat_2d = rnat_mean.values
lat_rnat = data['latitude'].values
lon_rnat = data['longitude'].values

lat_sorted = np.argsort(lat_rnat)
lon_sorted = np.argsort(lon_rnat)
rnat_sorted = rnat_2d[np.ix_(lat_sorted, lon_sorted)]
lat_rnat_sorted = lat_rnat[lat_sorted]
lon_rnat_sorted = lon_rnat[lon_sorted]

Z = startfi['ZMEA'].values              
area = startfi['area'].values           
lat_phys = startfi['latitude'].values   
lon_phys = startfi['longitude'].values  
lat_phys_deg = np.rad2deg(lat_phys)
lon_phys_deg = np.rad2deg(lon_phys)
interp_func = RegularGridInterpolator(
    (lat_rnat_sorted, lon_rnat_sorted),
    rnat_sorted,
    bounds_error=False,
    fill_value=np.nan
)
coords = np.column_stack((lat_phys_deg, lon_phys_deg))
rnat_interp = interp_func(coords)
ocean_mask = (rnat_interp < 0.1)
Z_ocean = Z[ocean_mask]
z_sea = -2500
depth = np.zeros_like(Z)
depth[ocean_mask] = np.clip(z_sea - Z[ocean_mask], a_min=0, a_max=None)

# 4. Volume total d'eau
volume = np.sum(depth * area)
Mars_surface = np.sum(area)
equiv_depth = volume / Mars_surface
V_water = volume
