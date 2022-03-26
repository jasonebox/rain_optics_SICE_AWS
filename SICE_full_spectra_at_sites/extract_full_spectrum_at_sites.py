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

# %% paths

base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_full_spectra_at_sites/'\
    + 'results'

# %% SICE folders

folders = glob.glob('/media/adrien/Elements/rain_optics_SICE_AWS/2021-08*')

# %% AWS positions and reproject to 3413

aws_info = pd.read_csv('/home/adrien/EO-IO/rain_optics_SICE_AWS/AWS_info/' 
                       + 'PROMICE_GC-Net_info.csv')

inProj = 'epsg:4326'
outProj = 'epsg:3413'

trf = Transformer.from_crs(inProj, outProj, always_xy=True)
aws_meast, aws_mnorth = trf.transform(aws_info.lon, aws_info.lat)

# %% get SICE rows and cols at station locations

ex = sorted(glob.glob(f'{folders[0]}/*.tif'))[0]

with rasterio.open(ex) as src:

    aws_rows, aws_cols = rasterio.transform.rowcol(src.transform, aws_meast, 
                                               aws_mnorth)

# %% data extraction

for folder in folders:
    
    date = folder.split(os.sep)[-1]
    
    TOA_files = sorted(glob.glob(f'{folder}/*.tif'))
    TOA_files.pop(3)
    
    SCDA_file = sorted(glob.glob(f'{folder}/*.tif'))[3]
    SCDA = rasterio.open(SCDA_file).read(1)
    
    var_names = [f.split(os.sep)[-1].split('.')[0] for f in 
                 sorted(glob.glob(f'{folder}/*.tif'))]
    
    daily_spectrum = pd.DataFrame(index=aws_info.name.values,
                                  columns=var_names)
   

    for file in TOA_files:
        
        var_name = file.split(os.sep)[-1].split('.')[0]
        data = rasterio.open(file).read(1)
        data[~np.isfinite(SCDA)] = np.nan
        daily_spectrum[var_name] = data[aws_rows, aws_cols]
        
    daily_spectrum.to_csv(f'{base_path}/{date}.csv')
    print(folder)
