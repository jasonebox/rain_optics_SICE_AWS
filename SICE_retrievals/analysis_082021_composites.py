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
files = sorted(glob.glob(f'{data_path}/Greenland/*/BBA_combination.tif'))

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

date_event = pd.to_datetime('2021-08-14')
date1 = pd.to_datetime('2021-08-09')
date2 = pd.to_datetime('2021-08-19')

# sequence of dates between date1 and event
files_pre_event = [f for i, f in enumerate(files) 
                    if (dates[i] >= date1) & (dates[i] < date_event)]

files_post_event = [f for i, f in enumerate(files) 
                    if (dates[i] <= date2) & (dates[i] > date_event)]

# %% get shapes from ex file

ex = rasterio.open(files_pre_event[0]).read(1)

# %% extract data pre event

bba_pre_event = np.zeros((ex.shape[0], ex.shape[1], len(files_pre_event)))
bba_pre_event[:, :, :] = np.nan

for i, file in enumerate(files_pre_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    bba_pre_event[:, :, i] = bba
        
compo_bba_pre_event = np.nanmean(bba_pre_event, axis=-1)

# %% extract data post event

bba_post_event = np.zeros((ex.shape[0], ex.shape[1], len(files_pre_event)))
bba_post_event[:, :, :] = np.nan

for i, file in enumerate(files_post_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    bba_post_event[:, :, i] = bba
        
compo_bba_post_event = np.nanmean(bba_post_event, axis=-1)
        
# %% animation

# fig = plt.figure()
# ax = fig.add_subplot(111)

# implots = []

# for date, bba in bba_stack.items():
    
#     pltt = ax.imshow(bba, cmap='Blues_r', vmin=0.3, vmax=0.9, aspect='auto')
    
#     implots.append([pltt])
     
# ani = animation.ArtistAnimation(fig, implots, interval=500, blit=True,
#                                 repeat_delay=1)


# %% variation visualisation

class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    

d_bba = compo_bba_post_event - compo_bba_pre_event

fig = plt.figure(figsize=(15, 8))
ax1 = plt.subplot(131)
ax1plot = ax1.imshow(compo_bba_pre_event, cmap='Blues_r', 
                     vmin=0.3, vmax=0.9, aspect='auto')
ax1.set_xticks([])
ax1.set_yticks([])
cbar = plt.colorbar(ax1plot,ax=ax1)
cbar.ax.tick_params(labelsize=17) 
plt.title("Mean albedo \n" + files_pre_event[0].split(os.sep)[-2] + '/' + 
          files_pre_event[-1].split(os.sep)[-2], fontsize=22)

ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
ax2plot = ax2.imshow(compo_bba_post_event, cmap='Blues_r', 
                     vmin=0.3, vmax=0.9, aspect='auto')
cbar = plt.colorbar(ax2plot,ax=ax2)
cbar.ax.tick_params(labelsize=17) 
plt.title("Mean albedo \n" + files_post_event[0].split(os.sep)[-2] + '/' + 
          files_post_event[-1].split(os.sep)[-2], fontsize=22)
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
ax3plot = ax3.imshow(d_bba, cmap='coolwarm_r', aspect='auto', 
                     norm=MidpointNormalize(-0.4, 0.1, 0.))
cbar = plt.colorbar(ax3plot,ax=ax3)
cbar.ax.tick_params(labelsize=17) 
ax3.set_xticks([])
ax3.set_yticks([])
plt.title('Albedo variations', fontsize=22)

plt.subplots_adjust(hspace=0.1)

# %%

plt.savefig('/home/adrien/EO-IO/SICE-ROS/08-2021/figures/082021_composite_CW.png', 
            bbox_inches='tight')

