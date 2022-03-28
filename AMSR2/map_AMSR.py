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

if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/rain_optics_SICE_AWS/"

os.chdir(base_path)

# not needed for GRL fig
# files = sorted(glob.glob("/Users/jason/0_dat/AMSR2/*.tif"))
# print(files)
# print(len(files))
#%%
# load geodata

elev = rasterio.open("./metatiffs/elev_1km_1487x2687.tif").read(1)
lat = rasterio.open("./metatiffs/lat_1km_1487x2687.tif").read(1)
lon = rasterio.open("./metatiffs/lon_1km_1487x2687.tif").read(1)
mask = rasterio.open("./metatiffs/mask_1km_1487x2687.tif").read(1)
# mask[mask>2]=2 # include ice shelves as standard ice mass value of 2
# load profile to further save outputs
profile = rasterio.open("./metatiffs/mask_1km_1487x2687.tif").profile
# plt.hist(mask)
mask[mask == 220] = 2
mask[mask > 2] = 1
# mask[((mask > 2)&(mask <=200))] = 1
# mask[mask==0]=0

land = np.zeros((2687, 1487))
land = land.astype(np.float64)

ice = np.zeros((2687, 1487))
ice = land.astype(np.float64)

ice[mask == 0] = np.nan
ice[ice==0]=1
tot_ice_area=np.sum(ice==1)

# plt.imshow(ice)
# plt.colorbar()

#%%
land[mask == 1] = 1
land[mask == 0] = np.nan
land[mask > 1] = np.nan

# plt.imshow(land)
# plt.colorbar()

# with rasterio.open(
#     "/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals/output_tifs/land.tif",
#     "w",
#     **profile
# ) as dst:
#     dst.write(land, 1)

# %% map the good stuff

# fn = "/Users/jason/0_dat/AMSR2/melt-ASC-2021-08-13.tif"
fn = "./AMSR2/melt-ASC-2021-08-13.tif"
before = rasterio.open(fn).read(1)
before = before.astype(np.float64)

# plt.imshow(before)
# plt.colorbar()
# #%%
pixel_area=20*20 #km
pixel_area=1 # uncertain about scale

melt_area_time0=np.sum(before>0)*pixel_area


# fn = "/Users/jason/0_dat/AMSR2/melt-DSC-2021-08-14.tif"
fn = "./AMSR2/melt-DSC-2021-08-14.tif"
after = rasterio.open(fn).read(1)
melt_area_time1=np.sum(after>0)*pixel_area

# # fill in areas of melting ice
# v=elev<700
# after[v]=1
# before[v]=1

# v=((elev<1500) & (lat<73) & (lon<-44))
# after[v]=1
# before[v]=1

before[after == 1] += 1

before[mask == 0] = np.nan
before[land==1] = np.nan
tot_area=np.sum(before>=0)

melt_area_time2=np.sum(before>0)*pixel_area
print('melt-ASC-2021-08-13',melt_area_time0,melt_area_time0/tot_area)#/1e6)
print('melt-DSC-2021-08-14',melt_area_time1,melt_area_time1/tot_area)#/1e6)
print('melt-DSC-2021-08-14',melt_area_time2,melt_area_time2/tot_area)#/1e6)


# before[((after==1)or(after==1))]+=1

# before[mask == 1] = 1

plt.imshow(before)
# plt.imshow(after)
plt.colorbar()
# print(file,np.sum(a),np.sum(a)/areax)
# area[j,i]=np.sum(a)

# #%%
# with rasterio.open(
#     "/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals/output_tifs/13ASC-14DSC_AMSR2.tif",
#     "w",
#     **profile
# ) as dst:
#     dst.write(before, 1)

# #%%
# do_gif = 1
# if do_gif == 1:
#     print("making .gif")
#     os.system(
#         "/usr/local/bin/convert -delay 50 -loop 0 ./output_figs/*.png ./output_figs/BBA_before_after.gif"
#     )


# #%% plt difference transects

# a = compo_bba_post_event - compo_bba_pre_event

# rowx = 1750
# for rowx in range(1600, 1800, 40):
#     plt.plot(elev[rowx, 500:800], a[rowx, 500:800], ".")
# #%%
# do_rest = 0

# if do_rest:
#     # %%

#     dalbedo_cw = compo_bba_pre_event_cw - compo_bba_post_event_cw

#     data_mask = np.isfinite(dalbedo_cw)

#     # # %% raw albedo elevation profile

#     # plt.figure()
#     # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_pre_event_cw[data_mask].flatten())
#     # plt.scatter(elev_cw[data_mask].flatten(), compo_bba_post_event_cw[data_mask].flatten())

#     # %%

#     elev_cw_mskd = elev_cw[data_mask]
#     albed_prevent = compo_bba_pre_event_cw[data_mask]
#     albed_poevent = compo_bba_post_event_cw[data_mask]

#     elev_bins = np.linspace(0, 3000, 100)

#     albedo_prof_prevent = []
#     albedo_prof_poevent = []

#     albedo_prof_prevent_std = []
#     albedo_prof_poevent_std = []

#     for i in range(0, len(elev_bins) - 1):

#         elev_mask = (elev_cw_mskd >= elev_bins[i]) & (elev_cw_mskd <= elev_bins[i + 1])

#         albedo_prof_prevent.append(np.nanmean(albed_prevent[elev_mask]))
#         albedo_prof_poevent.append(np.nanmean(albed_poevent[elev_mask]))

#         albedo_prof_prevent_std.append(np.nanstd(albed_prevent[elev_mask]))
#         albedo_prof_poevent_std.append(np.nanstd(albed_poevent[elev_mask]))

#     plt.figure(figsize=(13, 9))
#     color0 = "b"
#     color1 = "r"
#     plt.plot(
#         elev_bins[:-1],
#         albedo_prof_prevent,
#         color=color0,
#         label=files_pre_event[0].split(os.sep)[-2]
#         + " to "
#         + files_pre_event[-1].split(os.sep)[-2],
#     )
#     plt.scatter(elev_bins[:-1], albedo_prof_prevent, color="gray", alpha=0.5)
#     plt.fill_between(
#         elev_bins[:-1],
#         np.array(albedo_prof_prevent) - np.array(albedo_prof_prevent_std),
#         np.array(albedo_prof_prevent) + np.array(albedo_prof_prevent_std),
#         alpha=0.2,
#         color="gray",
#     )
#     plt.fill_between(
#         elev_bins[:-1],
#         np.array(albedo_prof_poevent) - np.array(albedo_prof_poevent_std),
#         np.array(albedo_prof_poevent) + np.array(albedo_prof_poevent_std),
#         alpha=0.2,
#         color="gray",
#     )
#     plt.plot(
#         elev_bins[:-1],
#         albedo_prof_poevent,
#         color=color1,
#         label=files_post_event[0].split(os.sep)[-2]
#         + " to "
#         + files_post_event[-1].split(os.sep)[-2],
#     )
#     plt.scatter(elev_bins[:-1], albedo_prof_poevent, color="gray", alpha=0.5)
#     plt.xlim([350, 3100])
#     plt.grid(alpha=0.5)
#     plt.tick_params(labelsize=18)
#     plt.xlabel("Elevation, m", fontsize=20)
#     plt.ylabel("Albedo, unitless", fontsize=20)

#     plt.axvline(
#         elev_bins[:-1][np.array(albedo_prof_prevent) >= 0.565][0],
#         color=color0,
#         LineStyle="--",
#         label="snowline elevation before",
#     )

#     plt.axvline(
#         elev_bins[:-1][np.array(albedo_prof_poevent) >= 0.565][0],
#         color=color1,
#         LineStyle="--",
#         label="snowline elevation after",
#     )
#     plt.legend(fontsize=15, loc="lower right")
#     # plt.title('Albedo/Elevation profiles in CW Greenland \n before and after extreme 2021-08 rainfall event',
#     #           fontsize=23)

#     fn = "/Users/jason/Dropbox/rain_optics_SICE_AWS/AWS_info/AWS_results.txt"
#     AWS = pd.read_csv(fn, delimiter="\t")
#     n_AWS = len(AWS)
#     print(AWS.columns)
#     plt.plot(AWS["elevation, m"], AWS["alb0"], "s", color=color0)
#     for i in range(n_AWS):
#         plt.text(
#             AWS["elevation, m"][i],
#             AWS["alb0"][i],
#             AWS.site[i],
#             color=color0,
#             rotation=0,
#         )
#         plt.text(AWS["elevation, m"][i], AWS["alb1"][i], AWS.site[i], color=color1)
#     plt.plot(AWS["elevation, m"], AWS["alb1"], "s", color=color1)

#     ly = "x"

#     if ly == "p":
#         plt.savefig("./figures/082021_albedo_elev_profile_CW.png", bbox_inches="tight")

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
