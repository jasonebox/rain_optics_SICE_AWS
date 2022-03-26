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

AW = False
JEB = True

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals'

os.chdir(base_path)
    
# %% list files and load metatiffs

files = sorted(glob.glob('./L3_product/*.tif'))

elev = rasterio.open('./metatiffs/elev_1km_1487x2687.tif').read(1)
lat = rasterio.open('./metatiffs/lat_1km_1487x2687.tif').read(1)
lon = rasterio.open('./metatiffs/lon_1km_1487x2687.tif').read(1)
msk = rasterio.open('./metatiffs/mask_1km_1487x2687.tif').read(1)

# %% load Watson catchment polygon

watson_poly_gdf = gpd.read_file('../Watson_catchment/Watson/catchment_Watson.shp')
watson_poly = watson_poly_gdf.to_crs('epsg:3413').iloc[0].geometry

# %% mask elevation raster

elev_reader = rasterio.open('./metatiffs/elev_1km_1487x2687.tif')
masked_elev = rasterio.mask.mask(elev_reader, 
                                     [watson_poly], nodata=np.nan)[0][0, :, :]

# %% set elevation profile specs

elev_bins = np.arange(50, 2100, 100)
elev_bins_center = np.arange(100, 2100, 100)

# %% mask SICE data

column_names = ['year', 'doy']
[column_names.append(str(elv)) for elv in elev_bins_center]

results = np.zeros((len(files), len(column_names)))
results[:, :] = np.nan

for i, file in enumerate(files):
    
    albedo_reader = rasterio.open(file)
    masked_albedo = rasterio.mask.mask(albedo_reader, 
                                       [watson_poly], nodata=np.nan)[0][0, :, :]
    albedo_elev_profile = []
    
    for j in range(0, len(elev_bins) - 1):
        
        elev_mask = (masked_elev >= elev_bins[j])\
            & (masked_elev <= elev_bins[j + 1])
        results[i, j + 2] = np.nanmean(masked_albedo[elev_mask])
    
    doy = int(pd.to_datetime(file.split(os.sep)[-1].split('.')[0]).strftime("%j"))
    year = int(pd.to_datetime(file.split(os.sep)[-1].split('.')[0]).strftime("%Y"))
    date = file.split(os.sep)[-1].split('.')[0]
    
    results[i, 0] = int(year)
    results[i, 1] = int(doy)
    
    print(file)

results_df = pd.DataFrame(results, columns=column_names)

results_df.to_csv('../DVA/Watson/SICE_albedo_elevation_watson_082021.csv',
                  index=False)