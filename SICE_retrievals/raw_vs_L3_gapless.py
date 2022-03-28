# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import numpy as np
import glob
import rasterio
import os
import matplotlib.pyplot as plt
import pandas as pd

# %% setting required paths

if os.getlogin() == 'adrien':
    data_path_proc = '/media/adrien/Elements/show_case_v14_proc'
    data_path_raw = '/media/adrien/Elements/show_case_v14'
    out_path = '/home/adrien/EO-IO/SICE_showcase_study/SC_2021/'

# %% get L3 gapless

mask = rasterio.open('/home/adrien/EO-IO/SICE_AW_JEB/showcase/Greenland_1km.tif').read(1)

gris_mask = rasterio.open('/media/adrien/Elements/GrIS_mask_SICE_092021.tif').read(1)
hi = 0.86

SICE_proc_files = sorted(glob.glob(f'{data_path_proc}/Greenland/BBA_combination/2021*.tif'))

dates_p = []
bias_p = []
albedos_p = []
albedos_noic_p = []
bias_noic_p = []
    
for file in SICE_proc_files:
        
     dates_p.append(file.split(os.sep)[-1].split('.')[0])
     BBAc = rasterio.open(file).read(1)
     BBAc[mask!=220] = np.nan
     albedos_p.append(np.nanmean(BBAc))
     bias_p.append(np.sum(BBAc <= 0.565))
     
     albedos_noic_p.append(np.nanmean(BBAc[gris_mask == 1])) 
     bias_noic_p.append(np.sum((gris_mask == 1) & (BBAc<= 0.565))) 
     
     print(file)
         

# %% get raw gapless

SICE_raw_files = sorted(glob.glob(f'{data_path_raw}/Greenland/2021*/BBA_combination.tif'))

ex = rasterio.open(SICE_raw_files[0]).read(1)

dates_r = []
bias_r = []
albedos_r = []
albedos_noic_r = []
bias_noic_r = []
bias_nois_r_cml = []
albedos_r_cml = []
    
cloud_cover = []

# initialize cumul
hi = 0.86
cuml = np.zeros((np.shape(ex)[0], np.shape(ex)[1]))
cuml[:, :] = np.nan
# cuml[gris_mask == 1] = hi 

for file in SICE_raw_files:
        
    dates_r.append(file.split(os.sep)[-2])
    BBAc = rasterio.open(file).read(1)
    BBAc[mask!=220] = np.nan
     
    valid = [(BBAc > 0) & (BBAc < 1)] 
    cuml[valid] = BBAc[valid]
    bias_nois_r_cml.append(np.sum((gris_mask == 1) & (cuml <= 0.565)))
    
    albedos_r.append(np.nanmean(BBAc))
    albedos_r_cml.append(np.nanmean(cuml))
    bias_r.append(np.sum(BBAc <= 0.565))
     
    albedos_noic_r.append(np.nanmean(BBAc[gris_mask == 1])) 
    bias_noic_r.append(np.sum((gris_mask == 1) & (BBAc<= 0.565))) 
        
    cloud_cover.append(np.sum((gris_mask == 1) & (~np.isfinite(BBAc)))\
                       / np.sum(gris_mask == 1))
    print(file)
     
# %% compare the two products

plt.figure()
ax1 = plt.subplot(111)
ax1.plot(pd.to_datetime(dates_p), bias_noic_p, 'o-')
ax1.plot(pd.to_datetime(dates_r), bias_nois_r_cml, 'o-')

ax2 = ax1.twinx()
ax2.plot(pd.to_datetime(dates_r), cloud_cover, color='gray')