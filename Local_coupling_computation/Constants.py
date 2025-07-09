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
diagfi_path = ''
data = xr.open_dataset(diagfi_path + 'diagfi_reference.nc', decode_times=False)
starfi = xr.open_dataset(diagfi_path + 'startfi.nc', decode_times=False)

Time = data['Time']
lat = data['latitude']
lon = data['longitude']
aire = data['aire']

Mars_surface = float(np.sum(aire))  # Surface of Mars (m²)

# Weighted rnat and global mean
rnat_weighted = rnat * aire
spatial_sum = rnat_weighted.sum(dim=["latitude", "longitude"])
spatial_mean = spatial_sum / Mars_surface
rnat_global_mean = spatial_mean.mean(dim="Time").item()
fraction_ocean = float(1 - rnat_global_mean)

# ASR (Absorbed Solar Radiation) global mean
asr = data['ASR']
asr_weighted = asr * aire
spatial_sum = asr_weighted.sum(dim=["latitude", "longitude"])
spatial_mean = spatial_sum / Mars_surface
asr_global_mean = 2 * float(spatial_mean.mean(dim="Time").item())
