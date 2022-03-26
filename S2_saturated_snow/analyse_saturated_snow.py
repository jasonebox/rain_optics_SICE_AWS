#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import glob
import os
import rasterio
import numpy as np

# %% set paths

if os.getlogin() == 'adrien':
    data_folder = '/home/adrien/Downloads/S2-saturated-snow'
    
# %% list files

files = sorted(glob.glob(f'{data_folder}/*.tiff'))

# %% load data 

dates = []
blue_pixels = []
mean_blue = []

for file in files:
        
    blue = rasterio.open(file).read(3)
    blue_255 = (blue / np.nanmax(blue)) * 255
    
    if np.sum(blue) > 0:
            
        cloud_mask = rasterio.open(file).read(4)
        
        blue_255[cloud_mask == 1] = np.nan
        
        dates.append(file.split(os.sep)[-1].split('.')[0])
        blue_pixels.append(np.nansum(blue_255 > 180))
        mean_blue.append(np.nanmean(blue_255))
        
        print(file)
        
    
