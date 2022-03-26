#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import glob
import pandas as pd
from pyproj import Transformer
import rasterio
import os
import numpy as np
from random import randrange

# %% paths

base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_full_spectra_at_sites/'\
    + 'results'

# %% SICE folders

folders = glob.glob('/media/adrien/Elements/rain_optics_SICE_AWS/2021-08*')

dates = [f.split(os.sep)[-1] for f in folders]

# %% data extraction

folder = folders[0]
files = sorted(glob.glob(f'{folder}/*.tif'))
files.pop(5)
files.pop(3)
files.pop(-1)
files.pop(-1)
files.pop(-1)

var_names = [f.split(os.sep)[-1].split('.')[0] for f in files]
var_names = ['x', 'y'] + var_names

spectra = []

for folder in folders:
    
    date = folder.split(os.sep)[-1]
    
    TOA_files = sorted(glob.glob(f'{folder}/*.tif'))
    TOA_files.pop(5)
    TOA_files.pop(3)
    TOA_files.pop(-1)
    TOA_files.pop(-1)
    TOA_files.pop(-1)
    
    SCDA_file = sorted(glob.glob(f'{folder}/*.tif'))[3]
    SCDA = rasterio.open(SCDA_file).read(1)
    
    diagno_file = sorted(glob.glob(f'{folder}/*.tif'))[5]
    diagno = rasterio.open(SCDA_file).read(1)
    
    
    # choice a random polluted snow pixel
    polluted_snow = np.where(diagno == 1)
    nb_values = len(np.where(diagno == 1)[0])
    
    random_pixel = randrange(nb_values)
    
    # get position 
    src = rasterio.open(SCDA_file)
    x, y = rasterio.transform.xy(src.transform, polluted_snow[0][random_pixel],
                                     polluted_snow[1][random_pixel])
    daily_spectrum = []       
    daily_spectrum.append(x)
    daily_spectrum.append(y)
    
    for i, file in enumerate(TOA_files):
        
        var_name = file.split(os.sep)[-1].split('.')[0]
        data = rasterio.open(file).read(1)
        data[~np.isfinite(SCDA)] = np.nan
        daily_spectrum.append(data[polluted_snow[0][random_pixel],
                                   polluted_snow[1][random_pixel]])
        
        # run check
        print(diagno[polluted_snow[0][random_pixel],
                     polluted_snow[1][random_pixel]])
        
    spectra.append(daily_spectrum)
    
spectra_df = pd.DataFrame(np.vstack(spectra), index=dates,
                       columns=var_names)
    
# %% 

spectra_df.to_csv(f'{base_path}/polluted_snow_full_spectra_at_random_pixels.csv')
    
        