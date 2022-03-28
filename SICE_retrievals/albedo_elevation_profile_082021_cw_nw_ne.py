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
from shapely import geometry
import geopandas as gpd

#%% graphics settings

font_size=20
plt.rcParams["font.size"] = font_size
params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k"}
plt.rcParams.update(params)
plt.gcf().set_facecolor("w")

# %% prepare paths

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals'

os.chdir(base_path)
    
files = sorted(glob.glob('./data/*/BBA_combination.tif'))
print(files)
print(len(files))

# %% load geodata

elev = rasterio.open('./metatiffs/elev_1km_1487x2687.tif').read(1)
lat = rasterio.open('./metatiffs/lat_1km_1487x2687.tif').read(1)
lon = rasterio.open('./metatiffs/lon_1km_1487x2687.tif').read(1)
mask = rasterio.open('./metatiffs/mask_1km_1487x2687.tif').read(1)
#load profile to further save outputs 
profile=rasterio.open('./metatiffs/mask_1km_1487x2687.tif').profile

# %% mask by event area

# hilat = 68
# lolat = hilat - 1

# # keep elev masking for latter to keep a rectangular matrix 
# area =  (lat > lolat) & (lat < hilat) & (lon < -44) & (elev > 1900)\
#             &(elev < 3000)

# row_min = np.nanmin(np.where(area)[0])
# row_max = np.nanmax(np.where(area)[0])

# col_min = np.nanmin(np.where(area)[1])
# col_max = np.nanmax(np.where(area)[1])

# center west area

region='CW'
# region='NO'
# region='NE'

if region=='NE':
    row_min = 320
    row_max = 600
    col_min = 800
    col_max = 1150
    min_elev=500
    max_elev=2140
    min_alb=0.44
    max_alb=0.90

if region=='NO':
    row_min = 350
    row_max = 650
    col_min = 200
    col_max = 500
    min_elev=500
    max_elev=2400
    min_alb=0.47
    max_alb=0.915

if region=='CW':
    row_min = 1600
    row_max = 2160
    col_min = 400
    col_max = 690
    min_elev=350
    max_elev=2500
    min_alb=0.2
    max_alb=0.91


elev_cw = elev[row_min: row_max, col_min: col_max]
lat_cw = lat[row_min: row_max, col_min: col_max]
lon_cw = lon[row_min: row_max, col_min: col_max]
mask_cw = mask[row_min: row_max, col_min: col_max]
print('min lat',np.nanmin(lat_cw[lat_cw>0]))
print('max lat',np.nanmax(lat_cw))

print('min lon',np.nanmin(lon_cw[lat_cw>0]))
print('max lon',np.nanmax(lon_cw[lat_cw>0]))

print('min elev',np.nanmin(elev_cw[elev_cw>400]))
print('max elev',np.nanmax(elev_cw))

plt.imshow(mask_cw)
plt.colorbar()
print(np.max(mask_cw))
print(np.sum(mask_cw==220))

# %% create Shapely polygon from rows/cols taken from an ex raster

def write_region_polygon(region):

    if region=='NE':
        row_min = 320
        row_max = 600
        col_min = 800
        col_max = 1150

    if region=='NO':
        row_min = 350
        row_max = 650
        col_min = 200
        col_max = 500
    
    if region=='CW':
        # row_min = 1565
        # row_max = 1900
        row_min = 1600
        row_max = 2160
        col_min = 400
        col_max = 690

     
    ## %% create Shapely polygon from positions taken from row cols of an ex raster 
    
    ex = rasterio.open(files[0])
    
    lrcorner_x, lrcorner_y = rasterio.transform.xy(ex.transform, row_max, col_max)
    trcorner_x, trcorner_y = rasterio.transform.xy(ex.transform, row_min, col_max)
    tlcorner_x, tlcorner_y = rasterio.transform.xy(ex.transform, row_min, col_min)
    llcorner_x, llcorner_y = rasterio.transform.xy(ex.transform, row_max, col_min)
    
    region_poly = geometry.Polygon([
                [lrcorner_x, lrcorner_y],
                [trcorner_x, trcorner_y],
                [tlcorner_x, tlcorner_y],
                [llcorner_x, llcorner_y]
                ])

    ## %% write Shapely polygon to shapefile

    gdr = gpd.GeoDataFrame({'geometry': [region_poly]}, crs='EPSG:3413')

    gdr.to_file(f'./{region}_polygon.shp')
    
    print(f'wrote ./{region}_polygon.shp')
    
    return None

# run
regions = ['NE', 'NO', 'CW']
[write_region_polygon(r) for r in regions]

# %% extracting dates

dates = [pd.to_datetime(f.split(os.sep)[-2]) for f in files]

# %% mask by event dates

date_event = pd.to_datetime('2021-08-14')
date1 = pd.to_datetime('2021-08-09')
date2 = pd.to_datetime('2021-08-19')
date3 = pd.to_datetime('2021-08-27')
date4 = pd.to_datetime('2021-08-31')

# date3 = pd.to_datetime('2021-08-24')
# date4 = pd.to_datetime('2021-08-28')

# sequence of dates between date1 and event
files_pre_event = [f for i, f in enumerate(files) 
                    if (dates[i] >= date1) & (dates[i] < date_event)]

files_post_event = [f for i, f in enumerate(files) 
                    if (dates[i] <= date2) & (dates[i] > date_event)]

files_post_post_event = [f for i, f in enumerate(files) 
                    if (dates[i] <= date4) & (dates[i] >= date3)]
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

compo_bba_pre_event_cw = compo_bba_pre_event[row_min: row_max, col_min: col_max]

# %% extract data post event

bba_post_event = np.zeros((ex.shape[0], ex.shape[1], len(files_pre_event)))
bba_post_event[:, :, :] = np.nan

for i, file in enumerate(files_post_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    bba_post_event[:, :, i] = bba
        
compo_bba_post_event = np.nanmean(bba_post_event, axis=-1)

compo_bba_post_event_cw = compo_bba_post_event[row_min: row_max, col_min: col_max]

# %% extract data post post event

bba_post_post_event = np.zeros((ex.shape[0], ex.shape[1], len(files_post_post_event)))
bba_post_post_event[:, :, :] = np.nan

for i, file in enumerate(files_post_post_event):
    
    bba = rasterio.open(file).read(1)
    bba[mask != 220] = np.nan
    bba[bba>=1] = np.nan
    
    bba_post_post_event[:, :, i] = bba
        
compo_bba_post_post_event = np.nanmean(bba_post_post_event, axis=-1)

compo_bba_post_post_event_cw = compo_bba_post_post_event[row_min: row_max, col_min: col_max]
#%% map difference, write out composites



a=compo_bba_post_event-compo_bba_pre_event
# a[row_min: row_max, col_min: col_max][((mask[row_min: row_max, col_min: col_max]>100)&(elev[row_min: row_max, col_min: col_max]<-44))]=0.4
a[row_min: row_max, col_min: col_max]=0.3
# plt.imshow(a,cmap='bwr')
# plt.colorbar()

wo=0

if wo:
    pre_name=files_pre_event[0].split(os.sep)[-2]+'_'+ \
              files_pre_event[-1].split(os.sep)[-2]
    with rasterio.open(f'./output_tifs/'+pre_name+'_BBA_combination.tif', "w", **profile) as dst:
        dst.write(compo_bba_pre_event, 1)
    post_name=files_post_event[0].split(os.sep)[-2]+'_'+ \
              files_post_event[-1].split(os.sep)[-2]
    with rasterio.open(f'./output_tifs/'+post_name+'_BBA_combination.tif', "w", **profile) as dst:
        dst.write(compo_bba_post_event, 1)
#%% plt difference transects

# a=compo_bba_post_event-compo_bba_pre_event

# rowx=1750
# for rowx in range(1600,1800,40):
#     plt.plot(elev[rowx,500:800],a[rowx,500:800],'.')
#%%
do_rest=1

if do_rest:
    # %% scatter of change v elevation
    
    dalbedo_region = compo_bba_pre_event_cw - compo_bba_post_event_cw     
    data_mask = np.isfinite(dalbedo_region)
    
    # # %% raw albedo elevation profile
    
    # plt.figure()
    # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_pre_event_cw[data_mask].flatten())
    # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_post_event_cw[data_mask].flatten())
    
    # %% graphic
    
    elev_cw_mskd = elev_cw[data_mask]
    albed_prevent = compo_bba_pre_event_cw[data_mask]
    albed_poevent = compo_bba_post_event_cw[data_mask]
    albed_po_poevent = compo_bba_post_post_event_cw[data_mask]
    
    elev_bins = np.linspace(0, 3000, 100)
    
    albedo_prof_prevent = []
    albedo_prof_poevent = []
    albedo_prof_po_poevent = []
    
    albedo_prof_prevent_std = []
    albedo_prof_poevent_std = []
    albedo_prof_po_poevent_std = []
    
    for i in range(0, len(elev_bins) - 1):
        
        elev_mask = (elev_cw_mskd >= elev_bins[i])\
            & (elev_cw_mskd <= elev_bins[i + 1])
            
        albedo_prof_prevent.append(np.nanmean(albed_prevent[elev_mask]))
        albedo_prof_poevent.append(np.nanmean(albed_poevent[elev_mask]))
        albedo_prof_po_poevent.append(np.nanmean(albed_po_poevent[elev_mask]))
        
        albedo_prof_prevent_std.append(np.nanstd(albed_prevent[elev_mask]))
        albedo_prof_poevent_std.append(np.nanstd(albed_poevent[elev_mask]))
        albedo_prof_po_poevent_std.append(np.nanstd(albed_po_poevent[elev_mask]))
        
    plt.figure(figsize=(13, 9))
    color0='b' ; color1='r' ; color2='m'

    # ---------------- albedo profile pre
    plt.plot(elev_bins[:-1], albedo_prof_prevent, color=color0,
             label=files_pre_event[0].split(os.sep)[-2] + ' to ' + \
              files_pre_event[-1].split(os.sep)[-2])
    plt.scatter(elev_bins[:-1], albedo_prof_prevent, color='gray', alpha=0.5)

    plt.fill_between(elev_bins[:-1], np.array(albedo_prof_prevent) - np.array(albedo_prof_prevent_std),
                     np.array(albedo_prof_prevent) + np.array(albedo_prof_prevent_std),
                     alpha=0.2, color='gray')

    # ---------------- albedo profile post
    plt.plot(elev_bins[:-1], albedo_prof_poevent, color=color1,
             label=files_post_event[0].split(os.sep)[-2] + ' to ' + \
              files_post_event[-1].split(os.sep)[-2])

    plt.scatter(elev_bins[:-1], albedo_prof_poevent, color='gray', alpha=0.5)

    plt.fill_between(elev_bins[:-1], np.array(albedo_prof_poevent) - np.array(albedo_prof_poevent_std),
                     np.array(albedo_prof_poevent) + np.array(albedo_prof_poevent_std),
                     alpha=0.2, color='gray',label='±1 standard deviation',zorder=20)

    plt_post_post=1
    if ((region=='NO')or(region=='NE')):plt_post_post=0
    if plt_post_post:
        # ---------------- albedo profile post post
        plt.plot(elev_bins[:-1], albedo_prof_po_poevent, color=color2,
                 label=files_post_post_event[0].split(os.sep)[-2] + ' to ' + \
                  files_post_post_event[-1].split(os.sep)[-2])
    
        plt.scatter(elev_bins[:-1], albedo_prof_po_poevent, color='gray', alpha=0.5)
    
        plt.fill_between(elev_bins[:-1], np.array(albedo_prof_po_poevent) - np.array(albedo_prof_po_poevent_std),
                         np.array(albedo_prof_po_poevent) + np.array(albedo_prof_po_poevent_std),
                         alpha=0.2, color='gray',zorder=20)

    plt.xlim(min_elev, max_elev)
    plt.ylim(min_alb, max_alb)
    plt.grid(alpha=0.5)
    plt.tick_params(labelsize=18)
    plt.xlabel('elevation, m', fontsize=font_size)
    plt.ylabel('albedo, unitless', fontsize=font_size)
    
    snowline_before=elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0]
    # snowline_before_m1std = elev_bins[:-1][np.array(albedo_prof_prevent) 
    #                                      - np.array(albedo_prof_prevent_std) 
    #                                      >= 0.565][0]
    # snowline_before_p1std = elev_bins[:-1][np.array(albedo_prof_prevent) 
    #                                        + np.array(albedo_prof_prevent_std) 
    #                                        >= 0.565][0]
    
    snowline_after=elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0]
    # snowline_after_m1std = elev_bins[:-1][np.array(albedo_prof_poevent) 
    #                                      - np.array(albedo_prof_poevent_std) 
    #                                      >= 0.565][0]
    # snowline_after_p1std = elev_bins[:-1][np.array(albedo_prof_poevent) 
    #                                        + np.array(albedo_prof_poevent_std) 
    #                                        >= 0.565][0]
    
    print("snowline_before",snowline_before)
    print("snowline_after",snowline_after)
    print("snowline change",snowline_after-snowline_before)
        
    # xos=25
    # x0=elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0]
    # plt.axvspan(x0-xos, x0+xos,  color=color0, alpha=0.3,label='pre-heatwave snowline')
    
    # plt.axvspan(snowline_before_m1std, snowline_before_p1std,  
    #             color=color0, alpha=0.3,label='pre-heatwave snowline')
    
    if region == 'CW':
        plt.axvspan(582.0, 778.7,  
                    color=color0, alpha=0.3,label='pre-heatwave snowline')
        snowline_before=(582.0+778.7)/2
        plt.axvspan(1426.2, 1585.1,  
                color=color1, alpha=0.3,label='post-heatwave snowline')
        
    if region == 'NE':
        plt.axvspan(582.0, 778.7,  
                    color=color0, alpha=0.3,label='pre-heatwave snowline')
        snowline_before=(582.0+778.7)/2
        
        plt.axvspan(727-78,727+78,  
                color=color1, alpha=0.3,label='post-heatwave snowline')
        
    if region == 'NO':
        plt.axvspan(582.0, 778.7,  
                    color=color0, alpha=0.3,label='pre-heatwave snowline')
        plt.axvspan(879-60, 879+60,  
                color=color1, alpha=0.3,label='post-heatwave snowline')
        snowline_before=(582.0+778.7)/2
        
        
    plt.axvline(snowline_before, linestyle='--', color='gray',zorder=20)
    
    # x0=snowline_after
    # plt.axvspan(x0-xos, x0+xos,  color=color1, alpha=0.3,label='post-heatwave snowline')
    
    # plt.axvspan(snowline_after_m1std, snowline_after_p1std,  
    #             color=color1, alpha=0.3,label='post-heatwave snowline')
    
    plt.axvline(snowline_after, linestyle='--', color='gray')
    
    plt.legend(fontsize=15, loc='lower right')
    # plt.title('Albedo/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
    #           fontsize=23)
    
    fn='../AWS_info/AWS_results.txt'
    AWS=pd.read_csv(fn, delimiter='\t')
    n_AWS=len(AWS)
    print(AWS.columns)
    
    el_os=15
    alb_os=0.006
    for i in range(n_AWS):
        if AWS.region[i]==region:
            if AWS.site[i]!='CP1':
                plt.text(AWS["elevation, m"][i]+el_os,AWS['alb0'][i]+alb_os,AWS.site[i],color=color0,rotation=0)
                plt.text(AWS["elevation, m"][i]+el_os,AWS['alb1'][i]+alb_os,AWS.site[i],color=color1)
                plt.plot(AWS["elevation, m"][i],AWS['alb1'][i],'s',color=color1)
                plt.plot(AWS["elevation, m"][i],AWS['alb0'][i],'s',color=color0)
    
    ly='p'
    
    if ly=='p':
        plt.savefig('./figures/082021_albedo_elev_profile_'+region+'.png', 
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
    # plt.xlabel('Elevation (meters)', fontsize=font_size)
    # plt.ylabel('Albedo decrease (%)', fontsize=font_size)
    
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
    # plt.xlabel('Elevation (meters)', fontsize=font_size)
    # plt.ylabel('Albedo decrease (unitless)', fontsize=font_size)
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0], color=color0,
    #             LineStyle='--', label='Pre-event snowline elevation')
    
    # plt.axvline(elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0], color=color1,
    #             LineStyle='--', label='Post-event snowline elevation')
    # plt.legend(fontsize=15)
    # plt.title('Albedo decrease/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
    #           fontsize=23)
        
    # # plt.savefig('./figures/082021_albedo_elev_profile_CW_perc_decrease.png', 
    # #             bbox_inches='tight')
