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

# prepare paths

AW = False
JEB = True

if AW:
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals'
elif JEB:
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals'

os.chdir(base_path)
    
files = sorted(glob.glob('./data/*/BBA_combination.tif'))
print(files)
print(len(files))

# load geodata

elev = rasterio.open('./metatiffs/elev_1km_1487x2687.tif').read(1)
lat = rasterio.open('./metatiffs/lat_1km_1487x2687.tif').read(1)
lon = rasterio.open('./metatiffs/lon_1km_1487x2687.tif').read(1)
mask = rasterio.open('./metatiffs/mask_1km_1487x2687.tif').read(1)
#load profile to further save outputs 
profile=rasterio.open('./metatiffs/mask_1km_1487x2687.tif').profile


# %% extracting dates

dates = [pd.to_datetime(f.split(os.sep)[-2]) for f in files]

# %% mask by event dates

date_event = pd.to_datetime('2021-08-14')
date1 = pd.to_datetime('2021-08-08')
date2 = pd.to_datetime('2021-08-20')

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


# extract data post event

bba_post_event = np.zeros((ex.shape[0], ex.shape[1], len(files_pre_event)))
bba_post_event[:, :, :] = np.nan

for i, file in enumerate(files_post_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    bba_post_event[:, :, i] = bba
        
compo_bba_post_event = np.nanmean(bba_post_event, axis=-1)


#%% map difference, write out composites

font_size=10
plt.rcParams["font.size"] = font_size
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)
plt.gcf().set_facecolor("k")

period_is_pre=0

for period_is_pre in range(2):
    if period_is_pre:
        a=compo_bba_pre_event
    else:
        a=compo_bba_post_event    
    
    print(np.nanmin(a),np.nanmax(a))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    img=plt.imshow(a,cmap='Greys_r',vmin=0.12,vmax=0.99)#,cmap='Blues_r',vmin=0.2,vmax=0.9)
    plt.axis('off')

    cbar = plt.colorbar(fraction=0.015, pad=-0.04)
    # cbar=plt.colorbar(img, shrink=0.5)
    cbar.ax.tick_params(labelsize=font_size)
    
    pre_name=files_pre_event[0].split(os.sep)[-2]+'_'+ \
              files_pre_event[-1].split(os.sep)[-2]
    post_name=files_post_event[0].split(os.sep)[-2]+'_'+ \
              files_post_event[-1].split(os.sep)[-2]
    
    if period_is_pre:
        fig_name_part=pre_name
    else:
        fig_name_part=post_name
        
    ly='p'
    
    if ly=='p':
        plt.savefig('./output_figs/'+fig_name_part+'.png', bbox_inches = 'tight',figsize = (16,12), dpi = 600, facecolor='k')
        
        fig_nam='./output_figs/'+fig_name_part
        cr=40
        crop_top=50
        crop_l=cr ; crop_r=20
        crop_bot=cr
        
        os.system('/usr/local/bin/convert '+fig_nam+'.png'+
                  ' -gravity north -chop 0x'+str(crop_top)+
                  ' -gravity east -chop '+str(crop_r)+
                  'x0 -gravity south -chop 0x'+str(crop_bot)+
                  ' -gravity west -chop '+str(crop_l)+'x0 '+
                  fig_nam+'.png')
    wo=0
    
    if wo:
        with rasterio.open(f'./output_tifs/'+pre_name+'_BBA_combination.tif', "w", **profile) as dst:
            dst.write(compo_bba_pre_event, 1)
        with rasterio.open(f'./output_tifs/'+post_name+'_BBA_combination.tif', "w", **profile) as dst:
            dst.write(compo_bba_post_event, 1)
        
#%%
do_gif=1
if do_gif == 1:
    print("making .gif")
    os.system('/usr/local/bin/convert -delay 50 -loop 0 ./output_figs/*.png ./output_figs/BBA_before_after.gif')


#%% plt difference transects

a=compo_bba_post_event-compo_bba_pre_event

rowx=1750
for rowx in range(1600,1800,40):
    plt.plot(elev[rowx,500:800],a[rowx,500:800],'.')
#%%
do_rest=0

if do_rest:
    # %% 
    
    dalbedo_cw = compo_bba_pre_event_cw - compo_bba_post_event_cw 
    
    data_mask = np.isfinite(dalbedo_cw)
    
    # # %% raw albedo elevation profile
    
    # plt.figure()
    # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_pre_event_cw[data_mask].flatten())
    # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_post_event_cw[data_mask].flatten())
    
    # %% 
    
    elev_cw_mskd = elev_cw[data_mask]
    albed_prevent = compo_bba_pre_event_cw[data_mask]
    albed_poevent = compo_bba_post_event_cw[data_mask]
    
    
    elev_bins = np.linspace(0, 3000, 100)
    
    albedo_prof_prevent = []
    albedo_prof_poevent = []
    
    albedo_prof_prevent_std = []
    albedo_prof_poevent_std = []
    
    for i in range(0, len(elev_bins) - 1):
        
        elev_mask = (elev_cw_mskd >= elev_bins[i])\
            & (elev_cw_mskd <= elev_bins[i + 1])
            
        albedo_prof_prevent.append(np.nanmean(albed_prevent[elev_mask]))
        albedo_prof_poevent.append(np.nanmean(albed_poevent[elev_mask]))
        
        albedo_prof_prevent_std.append(np.nanstd(albed_prevent[elev_mask]))
        albedo_prof_poevent_std.append(np.nanstd(albed_poevent[elev_mask]))
        
    plt.figure(figsize=(13, 9))
    color0='b' ; color1='r'
    plt.plot(elev_bins[:-1], albedo_prof_prevent, color=color0,
             label=files_pre_event[0].split(os.sep)[-2] + ' to ' + \
              files_pre_event[-1].split(os.sep)[-2])
    plt.scatter(elev_bins[:-1], albedo_prof_prevent, color='gray', alpha=0.5)
    plt.fill_between(elev_bins[:-1], np.array(albedo_prof_prevent) - np.array(albedo_prof_prevent_std),
                     np.array(albedo_prof_prevent) + np.array(albedo_prof_prevent_std),
                     alpha=0.2, color='gray')
    plt.fill_between(elev_bins[:-1], np.array(albedo_prof_poevent) - np.array(albedo_prof_poevent_std),
                     np.array(albedo_prof_poevent) + np.array(albedo_prof_poevent_std),
                     alpha=0.2, color='gray')
    plt.plot(elev_bins[:-1], albedo_prof_poevent, color=color1,
             label=files_post_event[0].split(os.sep)[-2] + ' to ' + \
              files_post_event[-1].split(os.sep)[-2])
    plt.scatter(elev_bins[:-1], albedo_prof_poevent, color='gray', alpha=0.5)
    plt.xlim([350, 3100])
    plt.grid(alpha=0.5)
    plt.tick_params(labelsize=18)
    plt.xlabel('Elevation, m', fontsize=20)
    plt.ylabel('Albedo, unitless', fontsize=20)
    
    plt.axvline(elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0], color=color0,
                LineStyle='--', label='snowline elevation before')
    
    plt.axvline(elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0], color=color1,
                LineStyle='--', label='snowline elevation after')
    plt.legend(fontsize=15, loc='lower right')
    # plt.title('Albedo/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
    #           fontsize=23)
    
    fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/AWS_info/AWS_results.txt'
    AWS=pd.read_csv(fn, delimiter='\t')
    n_AWS=len(AWS)
    print(AWS.columns)
    plt.plot(AWS["elevation, m"],AWS['alb0'],'s',color=color0)
    for i in range(n_AWS):
        plt.text(AWS["elevation, m"][i],AWS['alb0'][i],AWS.site[i],color=color0,rotation=0)
        plt.text(AWS["elevation, m"][i],AWS['alb1'][i],AWS.site[i],color=color1)
    plt.plot(AWS["elevation, m"],AWS['alb1'],'s',color=color1)
    
    ly='x'
    
    if ly=='p':
        plt.savefig('./figures/082021_albedo_elev_profile_CW.png', 
                bbox_inches='tight')
    
    #%%
    
    # print(df)
    # # %%
    
    # dalbedo_prof_perc = ((np.array(albedo_prof_poevent) - np.array(albedo_prof_prevent))\
    #     / np.array(albedo_prof_prevent)) * 100
    
    # plt.figure(figsize=(13, 9))
    # plt.plot(elev_bins[:-1], dalbedo_prof_perc, color='gray')
    # plt.scatter(elev_bins[:-1], dalbedo_prof_perc, color='gray', alpha=0.5)
    # plt.xlim([300, 2150])
    # plt.grid(alpha=0.5)
    # plt.tick_params(labelsize=18)
    # plt.xlabel('Elevation (meters)', fontsize=20)
    # plt.ylabel('Albedo decrease (%)', fontsize=20)
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0], color=color0,
    #             LineStyle='--', label='Pre-event snowline elevation')
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0], color=color1,
    #             LineStyle='--', label='Post-event snowline elevation')
    # plt.legend(fontsize=15)
    # plt.title('Albedo decrease/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
    #           fontsize=23)
        
    # # plt.savefig('.figures/082021_albedo_elev_profile_CW_perc_decrease_perc.png', 
    # #             bbox_inches='tight')
    
    # # %% 
    
    # dalbedo_prof = np.array(albedo_prof_poevent) - np.array(albedo_prof_prevent)
    
    
    # plt.figure(figsize=(13, 9))
    # plt.plot(elev_bins[:-1], dalbedo_prof, color='gray')
    # plt.scatter(elev_bins[:-1], dalbedo_prof, color='gray', alpha=0.5)
    # plt.xlim([300, 2150])
    # plt.grid(alpha=0.5)
    # plt.tick_params(labelsize=18)
    # plt.xlabel('Elevation (meters)', fontsize=20)
    # plt.ylabel('Albedo decrease (unitless)', fontsize=20)
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0], color=color0,
    #             LineStyle='--', label='Pre-event snowline elevation')
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0], color=color1,
    #             LineStyle='--', label='Post-event snowline elevation')
    # plt.legend(fontsize=15)
    # plt.title('Albedo decrease/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
    #           fontsize=23)
        
    # # plt.savefig('./figures/082021_albedo_elev_profile_CW_perc_decrease.png', 
    # #             bbox_inches='tight')
