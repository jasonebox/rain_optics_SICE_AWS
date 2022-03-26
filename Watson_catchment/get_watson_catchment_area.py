#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import rasterio
from rasterio.mask import mask
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# %% prepare paths

# AW = False
# JEB = True

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals'

os.chdir(base_path)
    
# %% list files and load metatiffs

files = sorted(glob.glob('./gapless_data/*.tif'))

elev = rasterio.open('./metatiffs/elev_1km_1487x2687.tif').read(1)
lat = rasterio.open('./metatiffs/lat_1km_1487x2687.tif').read(1)
lon = rasterio.open('./metatiffs/lon_1km_1487x2687.tif').read(1)
msk = rasterio.open('./metatiffs/mask_1km_1487x2687.tif').read(1)

# %% total area of Watson catchment

watson_poly_gdf = gpd.read_file('../DVA/Watson/catchment_Watson.shp')
watson_poly = watson_poly_gdf.to_crs('epsg:3413').iloc[0].geometry

print(f'Watson catchment area: {watson_poly.area} m^2')
print(f'Watson catchment area: {watson_poly.area / 1e6} km^2')

# %% mask elevation raster with Watson polygon

elev_reader = rasterio.open('./metatiffs/elev_1km_1487x2687.tif')
masked_elev = rasterio.mask.mask(elev_reader, 
                                     [watson_poly], nodata=np.nan)[0][0, :, :]

# %% area of watson catchment per elevation bin

elev_bins = np.arange(50, 2100, 100)
elev_bins_center = np.arange(100, 2100, 100)

area_elev = []

for j in range(0, len(elev_bins) - 1):
    
    elev_mask = (masked_elev >= elev_bins[j])\
        & (masked_elev <= elev_bins[j + 1])
        
    area_elev.append(np.nansum(elev_mask))
    
area_elev_df = pd.DataFrame({'elevation_bin_meters': elev_bins_center,
                             'area_square_meters': area_elev})

area_elev_df.to_csv('../DVA/Watson/watson_area_elevation.csv', index=False)

# %% plot results

print('summed area',np.sum(area_elev_df.area_square_meters))

plt.figure()
plt.plot(area_elev_df.elevation_bin_meters, area_elev_df.area_square_meters,
         'o-')
plt.xlabel('Elevation, m')
plt.ylabel('Area km**2')
plt.grid(alpha=0.5)