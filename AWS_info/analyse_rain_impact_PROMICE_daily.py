#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:01:44 2021

@author: jeb@geus.dk

"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob

# from netCDF4 import Dataset
# from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime
import xarray as xr


# -------------------------------- chdir
if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/rain_optics_SICE_AWS"
elif os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/rain_optics_SICE_AWS"

os.chdir(base_path)

path = "./AWS_info/AWS_updates/"

sites = ["KAN_L"]
# sites=['KAN_M']
sites = ["KAN_U"]
# sites=['UPE_U']
# sites=['UPE_L']
# sites=['NUK_L']


# sites=['THU_U2']
# sites=['THU_L']
# sites=['KPC_U']

for st, site in enumerate(sites):
    # for st,site in enumerate(sites[0:1]):
    # for st,site in enumerate(sites[3:4]):

    fn = "/Users/jason/Dropbox/rain_optics_SICE_AWS/AWS_info/PROMICE_GC-Net_info.csv"
    meta = pd.read_csv(fn)
    print(meta.columns)
    for col in meta.columns[2:5]:
        meta[col] = pd.to_numeric(meta[col])
    print(meta)
    v = np.where(meta.name == site)
    v = v[0][0]

    fn = path + site + "_day_v03_upd.txt"
    df = pd.read_csv(fn, delim_whitespace=True)

    # df.index = pd.to_datetime(PROMICE_str_time, format='%Y %m %j %H')
    df[df == -999.0] = np.nan
    date_string = (
        df["Year"] * 10000 + df["MonthOfYear"] * 100 + df["DayOfMonth"] * 1
    ).apply(str)
    # print(date_string)
    df["time"] = pd.to_datetime(date_string, format="%Y%m%d")

    df.index = pd.to_datetime(date_string, format="%Y%m%d")

    # df = df.loc[df['time']>'2021-08-01',:]
    # df = df.reset_index(drop=True)

    # print(df)

    # x=df["AirTemperature(C)"]
    # print(x)
    print(df.columns)
    # print(df)

    # graphics settings
    fs = 25  # fontsize
    th = 1  # default line thickness
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.alpha"] = 1
    plt.rcParams["grid.color"] = "grey"
    plt.rcParams["font.size"] = fs
    plt.rcParams["figure.figsize"] = 25, 10
    plt.rcParams["mathtext.default"] = "regular"

    SAT = df["AirTemperature(C)"]
    ALB = df["ShortwaveRadiationUp(W/m2)"] / df["ShortwaveRadiationDown(W/m2)"]
    # ALB=df['Albedo_theta<70d']
    if sites == ["KPC_U"]:
        ALB -= 0.34

    # v=df['CloudCover']>0.8
    # ALB[v]=np.nan

    z_copy = df["DepthPressureTransducer_Cor(m)"]
    zboom = df["HeightSensorBoom(m)"]
    doy = df["DayOfYear"]
    year_PROMICE = df["Year"]

    t0 = datetime(2021, 8, 1)
    t1 = datetime(2021, 8, 23)
    t0 = datetime(2021, 8, 6)
    t1 = datetime(2021, 8, 22)

    if site == "KAN_L":
        # t0=datetime(2021, 8, 8) ; t1=datetime(2021, 8, 22)
        t0_pre_rain = datetime(2021, 8, 7)
        t1_pre_rain = datetime(2021, 8, 9)
        t0_post_rain = datetime(2021, 8, 14)
        t1_post_rain = datetime(2021, 8, 17)

    if site == "KAN_M":
        # t0=datetime(2021, 8, 8) ; t1=datetime(2021, 8, 22)
        t0_pre_rain = datetime(2021, 8, 8)
        t1_pre_rain = datetime(2021, 8, 9)
        t0_post_rain = datetime(2021, 8, 14)
        t1_post_rain = datetime(2021, 8, 17)

    if site == "KAN_U":
        # t0=datetime(2021, 8, 8) ; t1=datetime(2021, 8, 22)
        t0_pre_rain = datetime(2021, 8, 9)
        t1_pre_rain = datetime(2021, 8, 13)
        t0_post_rain = datetime(2021, 8, 15)
        t1_post_rain = datetime(2021, 8, 19)

    if site == "UPE_U":
        t0_pre_rain = datetime(2021, 8, 8)
        t1_pre_rain = datetime(2021, 8, 13)
        t0_post_rain = datetime(2021, 8, 14)
        t1_post_rain = datetime(2021, 8, 18)

    if site == "KPC_U":
        t0_pre_rain = datetime(2021, 8, 8)
        t1_pre_rain = datetime(2021, 8, 13)
        t0_post_rain = datetime(2021, 8, 14)
        t1_post_rain = datetime(2021, 8, 18)

    if site == "THU_U2":
        t0_pre_rain = datetime(2021, 8, 13)
        t1_pre_rain = datetime(2021, 8, 14)
        t0_post_rain = datetime(2021, 8, 15)
        t1_post_rain = datetime(2021, 8, 17)

    print("Alb pre rain")
    print(np.nanmean(ALB[t0_pre_rain:t1_pre_rain]))
    print("Alb post rain")
    print(np.nanmean(ALB[t0_post_rain:t1_post_rain]))
    print("Alb difference")
    print(
        np.nanmean(ALB[t0_post_rain:t1_post_rain])
        - np.nanmean(ALB[t0_pre_rain:t1_pre_rain])
    )

    plt.close()
    n_rows = 3
    fig, ax = plt.subplots(n_rows, 1, figsize=(10, 18))
    # fig, ax = plt.subplots(figsize=(10,18))
    #
    cc = 0  # ----------------------------------------------------------------------------------- upper wind and t
    pos = (
        "{:.0f}".format(meta.elev[v])
        + " m, N"
        + "{:.3f}".format(meta.lat[v])
        + "°N, "
        + "{:.3f}".format(-meta.lon[v])
        + "°W"
    )
    print(pos)
    ax[0].set_title(meta.name[v] + ", " + pos)
    # ax[0].set_title(site+', rain event 14 Aug, '+str(int(elev))+' m')

    ax[cc].plot(
        df["AirTemperature(C)"][t0:t1],
        drawstyle="steps",
        linewidth=th * 2,
        color="r",
        label="air T, °C",
    )
    # ax[cc].set_ylabel('°C', color='k')
    ax[cc].axhline(y=0, linestyle="--", linewidth=th * 1.5, color="grey")
    ax[cc].get_xaxis().set_visible(False)
    ax[cc].legend(prop={"size": fs})
    cc += 1

    # ax[cc].plot(df["Albedo_theta<70d"][t0:t1],linewidth=th*2, color='k',label='ALB')
    # ax[cc].plot(df['LongwaveRadiationDown(W/m2)'][t0:t1],linewidth=th*2, color='k',label='LWD')
    # ax[cc].plot(df['ShortwaveRadiationDown(W/m2)'][t0:t1],linewidth=th*2, color='k',label='SWD')
    # ax[cc].plot(df['ShortwaveRadiationUp(W/m2)'][t0:t1],linewidth=th*2, color='k',label='SWU')
    ax[cc].plot(
        ALB[t0:t1], drawstyle="steps", linewidth=th * 2, color="b", label="albedo"
    )
    ax[cc].set_ylabel("", color="k")
    ax[cc].get_xaxis().set_visible(False)
    ax[cc].legend(prop={"size": fs})
    cc += 1

    x1 = df["HeightSensorBoom(m)"][t0] - df["HeightSensorBoom(m)"][t0:t1]
    x2 = df["HeightStakes(m)"][t0] - df["HeightStakes(m)"][t0:t1]
    if site == "KAN_L":
        x2[datetime(2021, 8, 18) :] -= 2.76
        x2[datetime(2021, 8, 17) :] -= 0.9
    if site == "KAN_U":
        x1[datetime(2021, 8, 19) :] += 0.26
        x1[datetime(2021, 8, 20) :] += 0.65
        x2[datetime(2021, 8, 19) :] += 0.26
        x2[datetime(2021, 8, 20) :] += 0.75

    # ax[cc].plot(df["HeightSensorBoom(m)"][t0:t1],linewidth=th*2, color='k',label='Boom')
    ax[cc].plot(
        x2, drawstyle="steps", linewidth=th * 2, color="k", label="surface height 1, m"
    )
    # if site!='KAN_L':
    ax[cc].plot(
        x1, drawstyle="steps", linewidth=th * 2, color="r", label="surface height 2, m"
    )
    ax[cc].set_ylabel("", color="k")
    ax[cc].axhline(y=0, linestyle="--", linewidth=th * 1.5, color="grey")
    # ax[cc].set_ylabel('m', color='k')

    # ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))
    ax[cc].get_xaxis().set_visible(True)
    mult = 0.9
    ax[cc].legend(prop={"size": fs * mult})
    cc += 1

    cc -= 1
    # ax[cc].set_ylim(0,10)
    # ax[cc].set_yticks(np.arange(0, 12, 2))

    ax[cc].set_xlim(t0, t1)
    import matplotlib.dates as mdates

    # myFmt = mdates.DateFormatter('%b %d')
    # ax[cc].xaxis.set_major_formatter(myFmt)
    ax[cc].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90, ha="right")

    ly = "x"

    # plt.plot)
    if ly == "x":
        plt.show()

    if ly == "p":
        figpath = "/Users/jason/Dropbox/AWS/PROMICE/Figs/"
        figname = "/tmp/tmp.png"
        plt.savefig(figname, bbox_inches="tight", dpi=150)
        back_im = Image.open("/tmp/tmp.png")
        PROMICE_logo_fn = "/Users/jason/Dropbox/AWS/PROMICE/PROMICE logo.png"
        # msg='gm convert '+PROMICE_logo+' '+'-geometry x100 /tmp/PL.png ' ; print(msg) ; os.system(msg)
        pixelsx_size = 400
        size = pixelsx_size, pixelsx_size
        PROMICE_logo = Image.open(PROMICE_logo_fn, "r")
        PROMICE_logo.thumbnail(size, Image.ANTIALIAS)

        # back_im.save(figname, 'PNG')

        # PROMICE_logo = Image.open('/tmp/PL.png')
        # PROMICE_logo = Image.new("RGBA", PROMICE_logo.size, "WHITE") # Create a white rgba background

        # back_im = im1.copy()
        x0x, y0x = 210, 1070
        x0x, y0x = 1600, 1160

        back_im.paste(PROMICE_logo, (x0x, y0x), mask=PROMICE_logo)
        figname = figpath + site + "_" + varnam2[j] + "_update.png"
        back_im.save(figname)

        # figname=figpath+site+'_'+varnam2[j]+'_update.eps'
        # plt.savefig(figname)

        # os.system('open '+figname)
