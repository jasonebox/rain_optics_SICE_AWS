#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import rasterio
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.animation as animation
import matplotlib as mpl 

data_path = '/media/adrien/Elements/show_case_v14'
files = glob.glob(f'{data_path}/Greenland/*/BBA_combination.tif')

# %% load geodata

base_path = '/media/adrien/Elements/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CryoClim/ancil/'


elev = rasterio.open(f'{base_path}/elev_1km_1487x2687.tif').read(1)

lat = rasterio.open(f'{base_path}/lat_1km_1487x2687.tif').read(1)
lon = rasterio.open(f'{base_path}/lon_1km_1487x2687.tif').read(1)
mask = rasterio.open(f'{base_path}/mask_1km_1487x2687.tif').read(1)

# %% extracting dates

dates = [pd.to_datetime(f.split(os.sep)[-2]) for f in files]

# %% mask by event dates

date1 = pd.to_datetime('2021-08-12')
date2 = pd.to_datetime('2021-08-20')

# sequence of dates between date1 and date2
files_event = [f for i, f in enumerate(files) 
               if (dates[i] >= date1) & (dates[i] <= date2)]

# only date1 and date2
files_event = [f for i, f in enumerate(files) 
               if (dates[i] == date1) or (dates[i] == date2)]


# %% mask by event area (JEB masking)

hilat = 67
lolat = hilat - 1

# keep elev masking for latter to keep a rectangular matrix 
area =  (lat > lolat) & (lat < hilat) & (lon < -43) & (elev > 1900)\
            &(elev < 3000)

row_min = np.nanmin(np.where(area)[0])
row_max = np.nanmax(np.where(area)[0])

col_min = np.nanmin(np.where(area)[1])
col_max = np.nanmax(np.where(area)[1])

# %% extract data for specific event

bba_stack = {}

for i, file in enumerate(files_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    date = file.split(os.sep)[-2]
    
    bba_area = bba[row_min: row_max, col_min: col_max] # [area]
    
    # plt.hist(bba_area.flatten())

    if len(bba_area) > 0:
        bba_stack[date] = bba_area
        
# %% animation

fig = plt.figure()
ax = fig.add_subplot(111)

implots = []

for date, bba in bba_stack.items():
    
    pltt = ax.imshow(bba, cmap='Blues_r', vmin=0.3, vmax=0.9, aspect='auto')
    
    implots.append([pltt])
     
ani = animation.ArtistAnimation(fig, implots, interval=500, blit=True,
                                repeat_delay=1)


# %% variation visualisation

class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    

d_bba = bba_stack[date2.strftime('%Y-%m-%d')]\
           - bba_stack[date1.strftime('%Y-%m-%d')]

fig = plt.figure(figsize=(15, 20))
ax1 = plt.subplot(311)
ax1plot = ax1.imshow(bba_stack[date1.strftime('%Y-%m-%d')], cmap='Blues_r', 
                     vmin=0.3, vmax=0.9, aspect='auto')
ax1.set_xticks([])
ax1.set_yticks([])
cbar = plt.colorbar(ax1plot,ax=ax1)
cbar.ax.tick_params(labelsize=19) 
plt.title("Albedo " + date1.strftime('%Y-%m-%d'), fontsize=22)
ax2 = plt.subplot(312)
ax2plot = ax2.imshow(bba_stack[date2.strftime('%Y-%m-%d')], cmap='Blues_r', 
                     vmin=0.3, vmax=0.9, aspect='auto')
cbar = plt.colorbar(ax2plot,ax=ax2)
cbar.ax.tick_params(labelsize=19) 
plt.title("Albedo " + date2.strftime('%Y-%m-%d'), fontsize=22)
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = plt.subplot(313)
ax3plot = ax3.imshow(d_bba, cmap='coolwarm_r', aspect='auto', 
                     norm=MidpointNormalize(-0.4, 0.1, 0.))
cbar = plt.colorbar(ax3plot,ax=ax3)
cbar.ax.tick_params(labelsize=19) 
ax3.set_xticks([])
ax3.set_yticks([])
plt.title('Albedo variations (' + date2.strftime('%Y-%m-%d') + ' - ' + 
          date1.strftime('%Y-%m-%d') + ')', fontsize=22)

plt.subplots_adjust(hspace=0.1)

# plt.savefig('/home/adrien/EO-IO/SICE-ROS/08-2021/082021.png', 
#             bbox_inches='tight')

# %% per-pixel variations

plt.figure()

plt.scatter(bba_stack[date1.strftime('%Y-%m-%d')], d_bba, color='gray',
            alpha=0.3)
plt.tick_params(labelsize=17)
plt.xlabel('Albedo before rainfall event', fontsize=20)
plt.ylabel('Albedo variations', fontsize=20)

