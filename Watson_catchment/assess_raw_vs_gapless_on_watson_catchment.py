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

gapless_files = sorted(glob.glob('./gapless_data/*.tif'))
gapless_files = sorted(glob.glob('/media/adrien/Elements/show_case_v14_proc/'
                                 + 'Greenland/BBA_combination/2021*'))
raw_files = sorted(glob.glob('./data/*/BBA_combination.tif')) 
raw_files = sorted(glob.glob('/media/adrien/Elements/show_case_v14/Greenland/'
                             + '2021*/BBA_combination.tif'))

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

results = np.zeros((len(gapless_files), len(column_names)))
results[:, :] = np.nan

gapless_watson_albedo = []
gapless_dates = []

for i, file in enumerate(gapless_files):
    
    albedo_reader = rasterio.open(file)
    masked_albedo = rasterio.mask.mask(albedo_reader, 
                                       [watson_poly], nodata=np.nan)[0][0, :, :]
    
    gapless_watson_albedo.append(np.nanmean(masked_albedo))
    gapless_dates.append(pd.to_datetime(file.split(os.sep)[-1], format='%Y-%m-%d.tif'))
    print(file)
    
raw_watson_albedo = []
raw_dates = []

for i, file in enumerate(raw_files):
    
    albedo_reader = rasterio.open(file)
    masked_albedo = rasterio.mask.mask(albedo_reader, 
                                       [watson_poly], nodata=np.nan)[0][0, :, :]
    
    if i == 0:
        raw_albedo = masked_albedo
        raw_albedo[np.isfinite(raw_albedo)] = 0.86
    else:
        raw_albedo[np.isfinite(masked_albedo)] = masked_albedo[np.isfinite(masked_albedo)]
        
    raw_watson_albedo.append(np.nanmean(raw_albedo))
    raw_dates.append(pd.to_datetime(file.split(os.sep)[-2], format='%Y-%m-%d'))
    print(file)
    
hybrid_watson_albedo = []



#%% 

plt.figure()
plt.scatter(gapless_dates, gapless_watson_albedo)
plt.scatter(raw_dates, raw_watson_albedo)