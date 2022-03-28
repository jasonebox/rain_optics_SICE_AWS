#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:33:42 2021

@author: jason

SEMMIS notes from DVA:
    
H is surface height in snow and ice, for comparison with AWS measurements by PTA and SR50.

Precipitation is parameterized in the model:

precipitation = snowfall

precipitating = where(LRin[round(elev_bins/2),*] ge sigma*T[round(elev_bins/2),*]^4)

if total(precipitating) gt -1 then precipitation[*,precipitating] = 1

snowing = where(precipitation eq 1 and T-T_0 le T_solidprecip)

raining = where(precipitation eq 1 and T-T_0 gt T_solidprecip)

if total(snowing) gt -1 then snowfall[snowing] = prec_rate*rho_water/rho_snow[snowing]/3600.*dt_obs/dev ; in m of snow

if total(raining) gt -1 then rainfall[raining] = prec_rate/3600.*dt_obs/dev           ; in m of water

if total(raining) gt -1 then T_rain[raining] = T[raining]

print,'NB: precip is determined from LR at mid elevation only!

 
WS, T, RH are input, interpolated between AWSs.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from numpy.polynomial.polynomial import polyfit
import matplotlib.dates as mdates
from datetime import datetime

# set path
if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS'

os.chdir(base_path)

source_name='SICE'
# source_name='MODIS'

#%% read KAN_x hourly


site='KAN_M' ; site_elev='1269'
site='KAN_U' ; site_elev='1842'

fn='./AWS_info/AWS_data/'+site+'_hour_v03.txt'
KAN_x_hourly=pd.read_csv(fn, delim_whitespace=True)
KAN_x_hourly[KAN_x_hourly == -999.0] = np.nan

KAN_x_hourly.rename({'Year': 'year'}, axis=1, inplace=True)
KAN_x_hourly.rename({'MonthOfYear': 'month'}, axis=1, inplace=True)
KAN_x_hourly.rename({'DayOfMonth': 'day'}, axis=1, inplace=True)
KAN_x_hourly.rename({'HourOfDay(UTC)': 'hour'}, axis=1, inplace=True)

KAN_x_hourly["time"]=pd.to_datetime(KAN_x_hourly[['year', 'month', 'day', 'hour']])
KAN_x_hourly.index = pd.to_datetime(KAN_x_hourly.time)

KAN_x_hourly = KAN_x_hourly.loc[KAN_x_hourly['time']>='2021-08-01',:] 
KAN_x_hourly = KAN_x_hourly.loc[KAN_x_hourly['time']<'2021-09-01',:] 
KAN_x_hourly
KAN_x_hourly.columns

# ---------------------------------------------- compute albedo
KAN_x_hourly["ALB"] = np.nan
plusminus = 11
# df["SW Downward"] *= 10
# df["SW Upward"] *= 10
# df["SW Downward"] /= swd_coef
# df["SW Upward"] /= swu_coef
# df["SRnet"] = df["SW Downward"] - df["SW Upward"]
# df["SW Downward"][df["SRnet"] < 0] = np.nan
# df["SW Upward"][df["SRnet"] < 0] = np.nan
# plt.plot(df['SW Downward']-df['SW Upward'])
# plt.plot(df['SW Downward']-df['SW Upward'])

N=len(KAN_x_hourly)
for i in range(0 + plusminus, N - plusminus):
    KAN_x_hourly["ALB"][i] = np.nansum(KAN_x_hourly["ShortwaveRadiationUp(W/m2)"][i - plusminus : i + plusminus]
    ) / np.nansum(KAN_x_hourly["ShortwaveRadiationDown(W/m2)"][i - plusminus : i + plusminus])


#%% 

# t0=datetime(2021, 8, 7) ; t1=datetime(2021, 8, 31,12)
# th=1
# font_size=20
# fig, ax = plt.subplots(figsize=(18,10))

# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 1
# plt.rcParams['grid.color'] = "#cccccc"

# # ax.plot(ALB[str(lev)+'m'][t0:t1]/100,'grey',linewidth=th*4,label='SICE albedo')

# # ax.plot(KAN_x["ALB"][t0:t1],'b',drawstyle='steps',linewidth=th*4,label='KAN_x albedo')
# ax.plot(KAN_x_hourly["ALB"][t0:t1],'b',drawstyle='steps',linewidth=th*4,label='KAN_x albedo')
# # ax.plot(KAN_x_hourly['ShortwaveRadiationDown(W/m2)'][t0:t1],'b',drawstyle='steps',linewidth=th*4,label='KAN_x albedo')
# # ax.plot(KAN_x_hourly['ShortwaveRadiationUp(W/m2)'][t0:t1],'b',drawstyle='steps',linewidth=th*4,label='KAN_x albedo')

# # ax.axvspan(pd.to_datetime('2021-08-13-11'), pd.to_datetime('2021-08-14-18'), 
# #             color='r', alpha=0.5,label='heatwave with rain')

# # ax.axvspan(pd.to_datetime('2021-08-14-18'), pd.to_datetime('2021-08-20-10'), 
# #             color='g', alpha=0.5,label='heat & clouds, no rain')

# # ax.axvspan(pd.to_datetime('2021-08-20-10'), pd.to_datetime('2021-08-27-18'), 
# #             color='m', alpha=0.5,label='few clouds, albedo and latent\nfeedbacks')
# plt.xticks(rotation=90,ha='center')
# # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes

# mult=0.8
# plt.legend(prop={'size': font_size*mult})  
#%% read KAN_x

path='./AWS_info/AWS_data/'
fn=path+site+'_day_v03.txt'
KAN_x=pd.read_csv(fn, delim_whitespace=True)
KAN_x[KAN_x == -999.0] = np.nan
date_string=(KAN_x["Year"]*10000+KAN_x["MonthOfYear"]*100+KAN_x["DayOfMonth"]*1).apply(str)
# print(date_string)
KAN_x['time']=pd.to_datetime(date_string,format='%Y%m%d')
KAN_x.index = pd.to_datetime(date_string, format='%Y%m%d')
KAN_x["ALB"] = KAN_x['ShortwaveRadiationUp(W/m2)']/KAN_x['ShortwaveRadiationDown(W/m2)']

KAN_x = KAN_x.loc[KAN_x['time']>='2021-08-01',:] 
KAN_x = KAN_x.loc[KAN_x['time']<'2021-09-01',:] 

KAN_x["ALB"][KAN_x["ALB"]>0.86]=0.86
print(KAN_x.columns)
#%%
files=['GF(Wm-2)','M(Wm-2)','meltwater(mm)','H(m)','SHF(Wm-2)','rainHF(Wm-2)','LHF(Wm-2)','SRnet(Wm-2)','runoff(mm)','LRnet(Wm-2)','melt_int(Wm-2)','albedo(%)']

for i,file in enumerate(files):
    fn='./SEMMIS/input_data/'+source_name+'/'+file+'_IP2_2021_JB_bins.txt'

    if i==0:
        G=pd.read_csv(fn, delim_whitespace=True)
        cols=G.columns
        for col in cols[0:2]:G[col] = pd.to_numeric(G[col])
        
        G['time'] = pd.to_datetime(G['Year'].astype(str)+G['Day'].astype(str)+G['Hour'].astype(str), format='%Y%j%H')
        G.index = pd.to_datetime(G.time)

    if i==1:
        M=pd.read_csv(fn, delim_whitespace=True)
        cols=M.columns
        for col in cols[0:2]:M[col] = pd.to_numeric(M[col])
        
        M['time'] = pd.to_datetime(M['Year'].astype(str)+M['Day'].astype(str)+M['Hour'].astype(str), format='%Y%j%H')
        M.index = pd.to_datetime(M.time)

    # if i==2:
    #     meltwater=pd.read_csv(fn, delim_whitespace=True)
    #     cols=meltwater.columns
    #     for col in cols[0:2]:meltwater[col] = pd.to_numeric(meltwater[col])
        
    #     meltwater['time'] = pd.to_datetime(meltwater['Year'].astype(str)+meltwater['Day'].astype(str)+meltwater['Hour'].astype(str), format='%Y%j%H')
    #     meltwater.index = pd.to_datetime(meltwater.time)

    # if i==3:
    #     melt=pd.read_csv(fn, delim_whitespace=True)
    #     cols=melt.columns
    #     for col in cols[0:2]:melt[col] = pd.to_numeric(melt[col])
        
    #     melt['time'] = pd.to_datetime(melt['Year'].astype(str)+melt['Day'].astype(str)+melt['Hour'].astype(str), format='%Y%j%H')
    #     melt.index = pd.to_datetime(melt.time)

    if i==4:
        SHF=pd.read_csv(fn, delim_whitespace=True)
        cols=SHF.columns
        for col in cols[0:2]:SHF[col] = pd.to_numeric(SHF[col])
        
        SHF['time'] = pd.to_datetime(SHF['Year'].astype(str)+SHF['Day'].astype(str)+SHF['Hour'].astype(str), format='%Y%j%H')
        SHF.index = pd.to_datetime(SHF.time)

    if i==5:
        rainHF=pd.read_csv(fn, delim_whitespace=True)
        cols=rainHF.columns
        for col in cols[0:2]:rainHF[col] = pd.to_numeric(rainHF[col])
        
        rainHF['time'] = pd.to_datetime(rainHF['Year'].astype(str)+rainHF['Day'].astype(str)+rainHF['Hour'].astype(str), format='%Y%j%H')
        rainHF.index = pd.to_datetime(rainHF.time)
        
    if i==6:
        LHF=pd.read_csv(fn, delim_whitespace=True)
        cols=LHF.columns
        for col in cols[0:2]:LHF[col] = pd.to_numeric(LHF[col])
        
        LHF['time'] = pd.to_datetime(LHF['Year'].astype(str)+LHF['Day'].astype(str)+LHF['Hour'].astype(str), format='%Y%j%H')
        LHF.index = pd.to_datetime(LHF.time)

    if i==7:
        SRNet=pd.read_csv(fn, delim_whitespace=True)
        cols=SRNet.columns
        for col in cols[0:2]:SRNet[col] = pd.to_numeric(SRNet[col])
        
        SRNet['time'] = pd.to_datetime(SRNet['Year'].astype(str)+SRNet['Day'].astype(str)+SRNet['Hour'].astype(str), format='%Y%j%H')
        SRNet.index = pd.to_datetime(SRNet.time)

    if i==8:
        Runoff=pd.read_csv(fn, delim_whitespace=True)
        cols=Runoff.columns
        for col in cols[0:2]:Runoff[col] = pd.to_numeric(Runoff[col])
        
        Runoff['time'] = pd.to_datetime(Runoff['Year'].astype(str)+Runoff['Day'].astype(str)+Runoff['Hour'].astype(str), format='%Y%j%H')
        Runoff.index = pd.to_datetime(Runoff.time)

    if i==9:
        LRNet=pd.read_csv(fn, delim_whitespace=True)
        cols=LRNet.columns
        for col in cols[0:2]:LRNet[col] = pd.to_numeric(LRNet[col])
        
        LRNet['time'] = pd.to_datetime(LRNet['Year'].astype(str)+LRNet['Day'].astype(str)+LRNet['Hour'].astype(str), format='%Y%j%H')
        LRNet.index = pd.to_datetime(LRNet.time)    

    if i==11:
        ALB=pd.read_csv(fn, delim_whitespace=True)
        for col in cols[0:2]:ALB[col] = pd.to_numeric(ALB[col])
        
        ALB['time'] = pd.to_datetime(ALB['Year'].astype(str)+ALB['Day'].astype(str)+ALB['Hour'].astype(str), format='%Y%j%H')
        ALB.index = pd.to_datetime(ALB.time) 
    

    # if i==10:
    #     melt_int=pd.read_csv(fn, delim_whitespace=True)
    #     cols=melt_int.columns
    #     for col in cols[0:2]:melt_int[col] = pd.to_numeric(melt_int[col])
        
    #     melt_int['time'] = pd.to_datetime(melt_int['Year'].astype(str)+melt_int['Day'].astype(str)+melt_int['Hour'].astype(str), format='%Y%j%H')
    #     melt_int.index = pd.to_datetime(melt_int.time)    
#%%

# LRNet=pd.read_csv(fn, delim_whitespace=True)
# cols=LRNet.columns
# for col in cols[0:2]:LRNet[col] = pd.to_numeric(LRNet[col])

# LRNet['time'] = pd.to_datetime(LRNet['Year'].astype(str)+LRNet['Day'].astype(str)+LRNet['Hour'].astype(str), format='%Y%j%H')
# # LRNet.index = pd.to_datetime(LRNet.time)  

# #%%
# LRNet = LRNet[LRNet['time'] > '2021-08-01']
# LRNet = LRNet[~(LRNet['time'] >= '2021-09-01')]

# LRNet['doy']=LRNet['time'].dt.dayofyear

# print(LRNet.doy)
# #%%

# cols=['time','LRNet', 'SRNet']

# N=len(LRNet)

# df = pd.DataFrame(columns=cols)

# lev=2000

# # plt.plot(LRNet['doy'])
# LRNetx=[]
# timex=[]
# for j in range(221,243):
#     v=LRNet['doy']==j
#     v2=np.where(LRNet['doy']==j+1)
#     temp=np.nanmean(LRNet[str(lev)+'m'][v])
#     LRNetx.append(temp)
#     print(j,np.sum(v),temp)

#     #%%
#     # doyx.append(LRNet['time'][v2[0][0]].strftime('%j'))
#     # doyx.append(LRNet['time'][v2[0][0]].strftime('%b-%d'))
#     timex.append(LRNet['time'][v2[0][0]].strftime('%Y-%m-%d'))

# # df.index.name = 'index'
# df["time"]=pd.Series(timex)
# df.index = pd.to_datetime(df.time)

# df["LRNet"]=pd.Series(LRNetx)

# fig, ax = plt.subplots(figsize=(18,18))

# ax.plot(df.time,df['LRNet'])
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='right' )

# #%%

# fig, ax = plt.subplots(figsize=(10,10))

# plt.plot(SHF[str(lev)+'m'][t0:t1]-LHF[str(lev)+'m'][t0:t1],label='SHF minus LHF')
# ax.set_xlim(t0,t1)
# # ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))   #to get a tick every 15 minutes
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %HUTC'))

# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
# ax.set_ylabel('W m$^{-2}$', color='k')
# plt.legend()
# ax.axhline(y=0,linestyle='-',linewidth=th*1.5, color='k')
    
#%%
t0=datetime(2021, 8, 10) ; t1=datetime(2021, 8, 31)
t0=datetime(2021, 8, 7) ; t1=datetime(2021, 8, 31,12)
# t0=datetime(2021, 8, 13) ; t1=datetime(2021, 8, 15)

levs=np.arange(100,2100,100)

levs=[1300] # KAN_M
# levs=[1800] # KAN_U

ALB['time']=pd.to_datetime(ALB['time'])
ALB['doy']=ALB['time'].dt.dayofyear
doys_SEMMIS=ALB['doy'].values
KAN_x['time']=pd.to_datetime(KAN_x['time'])
KAN_x['doy']=KAN_x['time'].dt.dayofyear
doys_KAN_x=KAN_x['doy'].values

for lev in levs:
    print(lev)
    ALB['ALB'+str(lev)+'mx']=np.nan
    ALB['ALB'+str(lev)+'m_KAN_x']=np.nan
    ALB['ALB'+str(lev)+'m_KAN_x_bright']=np.nan
    ALB['SWD'+str(lev)+'m']=SRNet[str(lev)+'m']/((100-ALB[str(lev)+'m'])/100)
    # ALB['ALB'+str(lev)+'mx']=ALB[str(lev)+'m']/100
    for i in range(213,244):
        # ALB['ALB'+str(lev)+'mx'][doys_SEMMIS==i-1]
        temp=KAN_x["ALB"][doys_KAN_x==i].values
        # temp=temp.as
        print(i,temp)
        if np.isfinite(temp):
        # type(temp)
        # temp[0]
        # print(i,)
        # if site=='KAN_U':
            ALB['ALB'+str(lev)+'m_KAN_x_bright'][doys_SEMMIS==i]=temp[0]
            ALB['ALB'+str(lev)+'m_KAN_x'][doys_SEMMIS==i]=temp[0]
    #     print(i,sum(v),sum(v2),KAN_x["ALB"][v2])

    alb_thresh=0.8
    ALB['ALB'+str(lev)+'m_KAN_x_bright'][ALB['ALB'+str(lev)+'m_KAN_x_bright']<=alb_thresh]=alb_thresh
    # storing SRNet in the ALB dataframe
    ALB['SRNet'+str(lev)+'m_KAN_x']=ALB['SWD'+str(lev)+'m']*(1-ALB['ALB'+str(lev)+'m_KAN_x'])
    ALB['SRNet'+str(lev)+'m_KAN_x_bright']=ALB['SWD'+str(lev)+'m']*(1-ALB['ALB'+str(lev)+'m_KAN_x_bright'])

    # plt.plot(SRNet[str(lev)+'m'])
    # plt.plot(ALB['SRNet'+str(lev)+'m_KAN_x'])
    # plt.plot(ALB['SRNet'+str(lev)+'m_KAN_x_bright'])
    
    # plt.plot(ALB['ALB'+str(lev)+'mx'])
    # plt.plot(ALB['ALB'+str(lev)+'m_KAN_x'])
    # plt.plot((ALB['ALB'+str(lev)+'m_KAN_x']+ALB['ALB'+str(lev)+'mx'])/2)
        # plt.plot(SRNet[str(lev)+'m'])
##%%
    # M2 is the original SEMMIS with SICE albedo
    M['M2'+str(lev)+'m']=SHF[str(lev)+'m']+LHF[str(lev)+'m']+SRNet[str(lev)+'m']+LRNet[str(lev)+'m']+G[str(lev)+'m']+rainHF[str(lev)+'m']
    # M is the SEMMIS with KAN_x albedo
    M['M'+str(lev)+'m']=       SHF[str(lev)+'m']+LHF[str(lev)+'m']+ALB['SRNet'+str(lev)+'m_KAN_x']+       LRNet[str(lev)+'m']+G[str(lev)+'m']+rainHF[str(lev)+'m']

    M['M'+str(lev)+'m_bright']=SHF[str(lev)+'m']+LHF[str(lev)+'m']+ALB['SRNet'+str(lev)+'m_KAN_x_bright']+LRNet[str(lev)+'m']+G[str(lev)+'m']+rainHF[str(lev)+'m']

    temp=0 ; temp2=0  ; temp3=0
    ALB['meltwater_cum']=np.nan    
    ALB['meltwater_cum_bright']=np.nan    
    for i in range(len(Runoff)):
        if  ALB['time'][i]>t0:
            if M['M'+str(lev)+'m'][i]>0:
                temp+=M['M'+str(lev)+'m'][i]/3.34e5/1000*3600*1000
                temp2+=M['M2'+str(lev)+'m'][i]/3.34e5/1000*3600*1000
            if M['M'+str(lev)+'m_bright'][i]>0:
                temp3+=M['M'+str(lev)+'m_bright'][i]/3.34e5/1000*3600*1000
            ALB['meltwater_cum'][i]=temp
            ALB['meltwater_cum_bright'][i]=temp3
            # print(ALB.time[i],temp,temp2)        
   
    th=1 
    font_size=24
    # plt.rcParams['font.sans-serif'] = ['Georgia']
    plt.rcParams["font.size"] = font_size
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 1
    plt.rcParams['grid.color'] = "#cccccc"
    plt.rcParams["legend.facecolor"] ='w'
    plt.rcParams["mathtext.default"]='regular'
    plt.rcParams['grid.linewidth'] = th
    plt.rcParams['axes.linewidth'] = th #set the value globally
    plt.rcParams['legend.framealpha'] = 1

# #%% air T

#     fig, ax = plt.subplots(figsize=(18,10))

#     plt.rcParams['axes.grid'] = True
#     plt.rcParams['grid.alpha'] = 1
#     plt.rcParams['grid.color'] = "#cccccc"

#     # ax.plot(ALB[str(lev)+'m'][t0:t1]/100,'grey',linewidth=th*4,label='SICE albedo')

#     ax.plot(KAN_x['AirTemperature(C)'][t0:t1],'b',drawstyle='steps',linewidth=th*4,label='KAN_x air T')


#     ax.axvspan(pd.to_datetime('2021-08-7'), pd.to_datetime('2021-08-13-11'), 
#                 color='b', alpha=0.5,label='cool')
    
#     ax.axvspan(pd.to_datetime('2021-08-13-11'), pd.to_datetime('2021-08-14-19'), 
#                 color='r', alpha=0.5,label='heatwave with rain')

#     ax.axvspan(pd.to_datetime('2021-08-14-19'), pd.to_datetime('2021-08-19'), 
#                 color='g', alpha=0.5,label='heat & clouds, no rain')

#     ax.axvspan(pd.to_datetime('2021-08-19'), pd.to_datetime('2021-08-27'), 
#                 color='m', alpha=0.5,label='albedo feedback')

#     ax.axvspan(pd.to_datetime('2021-08-27'), pd.to_datetime('2021-08-31-12'), 
#                 color='k', alpha=0.5,label='ablation end')
    
#     plt.xticks(rotation=90,ha='center')
#     ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes

#     ax.axhline(y=0,linestyle='-',linewidth=th*1.5, color='k')

#     ax.set_xlim(t0,t1)

#     ax.set_ylabel('° C')
#     mult=1
#     plt.legend(prop={'size': font_size*mult})    
## %%
  

    plt.close()
    n_rows=3
    fig, ax = plt.subplots(n_rows,1,figsize=(18,20))
    plt.subplots_adjust(hspace = .001)
    # ax[0].set_title('Watson catchment energy budget data from SEMMIS, '+str(lev)+'m')

    cc=0
    #------------------------------------------ plot SEB
    ax[cc].plot(M['M'+str(lev)+'m'][t0:t1],'k',linewidth=th*2,label='melt energy')
    # ax[cc].plot(ALB['SWD'][t0:t1],'maroon',linewidth=th*2,label='SWD')
    ax[cc].plot(SRNet[str(lev)+'m'][t0:t1],'darkorange',linewidth=th*2,label='SRnet')
    ax[cc].plot(LRNet[str(lev)+'m'][t0:t1],'c',linewidth=th*2,label='LRnet')
    # ax[cc].plot(ALB[str(lev)+'m'][t0:t1],'grey',linewidth=th*4,label='SHF')
    ax[cc].plot(SHF[str(lev)+'m'][t0:t1],'r',linewidth=th*2,label='SHF')
    ax[cc].plot(LHF[str(lev)+'m'][t0:t1],'b',linewidth=th*2,label='LHF')
    ax[cc].plot(LHF[str(lev)+'m'][t0:t1]+LHF[str(lev)+'m'][t0:t1],'m',linewidth=th*2,label='SHF+LHF')
    # ax[cc].plot(rainHF[str(lev)+'m'][t0:t1],'grey',linewidth=th*4,label='rainHF')
    ax[cc].plot(G[str(lev)+'m'][t0:t1],'g',linewidth=th*2,label='G')

    print('G',np.max(G[str(lev)+'m'][t0:t1]))
    print('G',np.min(G[str(lev)+'m'][t0:t1]))
    ax[cc].get_xaxis().set_visible(False)
    ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')
    ax[cc].set_ylabel('W m$^{-2}$', color='k')
    mult=0.75
    ax[cc].legend(prop={'size': font_size*mult},loc=2)
    # if lev==1800: ax[cc].set_ylim(-125,270)
    ax[cc].set_xlim(t0,t1)

    mult=1.1
    xx0=0.2 ; yy0=0.96
    props = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor='w')
    msg=source_name+'\n'+str(lev)+' m elevation'
    msg=str(lev-50)+' to '+str(lev+50)+' m elevation'
    msg=site+', '+site_elev+' m elevation'
    ax[cc].text(xx0,yy0, msg,transform=ax[cc].transAxes, fontsize=font_size*mult,
            verticalalignment='top', bbox=props,rotation=0,color='grey', rotation_mode="anchor")  
    
    t0x=pd.to_datetime('2021-08-13-15')
    t1x=pd.to_datetime('2021-08-14-18')
    # plt.axvspan(t0x, t1x,color='gray', alpha=0.5,label='14 to 24 Aug')


    plt_air_T=1
    if plt_air_T:
        cc+=1

        ax[cc].plot(KAN_x_hourly['AirTemperature(C)'][t0:t1],'red',linewidth=th*2,label='hourly air temperature')
        xt=KAN_x_hourly.time[t0:t1]
        
        ax[cc].fill_between(xt, KAN_x_hourly['AirTemperature(C)'][t0:t1], where=(KAN_x_hourly['AirTemperature(C)'][t0:t1] > 0), color='r', alpha=.3,label='hourly air temperature > 0 °C')

        # ax[cc].step(KAN_x['AirTemperature(C)'][t0:t1],'maroon',
        #             alpha=0.8,
        #             where='mid',linewidth=th*2,label='daily air temperature')
        # ax[cc].axvspan(pd.to_datetime('2021-08-7'), pd.to_datetime('2021-08-13-11'), 
        #             color='b', alpha=0.3)#,label='cool')
        
        # ax[cc].axvspan(pd.to_datetime('2021-08-13-11'), pd.to_datetime('2021-08-14-19'), 
        #             color='r', alpha=0.3)#,label='heatwave with rain')
    
        # # ax[cc].axvspan(pd.to_datetime('2021-08-14-19'), pd.to_datetime('2021-08-19'), 
        # #             color='g', alpha=0.3)#,label='heat & clouds, no rain')
    
        # ax[cc].axvspan(pd.to_datetime('2021-08-19'), pd.to_datetime('2021-08-27'), 
        #             color='m', alpha=0.3)#,label='albedo feedback')
    
        # ax[cc].axvspan(pd.to_datetime('2021-08-27'), pd.to_datetime('2021-08-31-12'), 
        #             color='k', alpha=0.3)#,label='ablation end')
        
        ax[cc].set_ylabel('°C')
        
        ax[cc].xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes
    
        # ax[cc].axhline(y=0,linestyle='-',linewidth=th*1.5, color='k')
        ax[cc].spines['left'].set_color('maroon')
        ax[cc].tick_params(axis='y', colors='maroon')
        ax[cc].yaxis.label.set_color('maroon')

        
        ax[cc].set_xlim(t0,t1)
        ax[cc].get_xaxis().set_visible(False)

        mult=0.75
        ax[cc].legend(prop={'size': font_size*mult})#,loc=2)

    plt_alb=1
    if plt_alb:
        ax2 = ax[cc].twinx()

        # ax2.step(ALB['ALB'+str(lev)+'m_KAN_x'][t0:t1],'b',
        #             alpha=0.8,
        #             where='mid',linewidth=th*2,label='albedo')
        ax2.step(KAN_x["ALB"][t0:t1],'b',
                    alpha=0.8,
                    where='mid',linewidth=th*2,label='albedo')

        # ax2.set_ylabel('albedo')
        
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes
    
        ax2.axhline(y=0,linestyle='-',linewidth=th*1.5, color='k')
    
        ax2.set_xlim(t0,t1)
        ax2.get_xaxis().set_visible(False)
        if site=='KAN_M':ax2.set_ylim(0.18,0.9)
        if site=='KAN_U':ax2.set_ylim(0.6,0.9)
        ax2.spines['right'].set_color('b')
        ax2.tick_params(axis='y', colors='b')

        mult=0.75
        ax2.legend(prop={'size': font_size*mult})#,loc=2)

    plt_cumulative_melt=1
    if plt_cumulative_melt:
        print(str(lev)+' m elevation','melt rates')
        #------------------------------------------ plot cumulative melt and runoff
        cc+=1
        # ax[cc].plot(Runoff['Runoff_cum'][t0:t1],'-om',linewidth=th*2,label='runoff')
        ax[cc].plot(ALB['meltwater_cum'][t0:t1],'k',linewidth=th*3,label='melt',zorder=20)
        plot_melt_bright=1
        if plot_melt_bright:
        #     meltwater_flat=pd.read_csv('./SEMMIS/output/meltwater_cumSICE_const_alb_post-17Aug.csv')
        #     meltwater_flat.index = pd.to_datetime(meltwater_flat.time) 
        #     # meltwater_flat['meltwater_cum']
            ax[cc].plot(ALB['meltwater_cum_bright'][t0:t1],'grey',linestyle='--',
                    linewidth=th*3,label='melt: albedo≥'+"{:.2f}".format(alb_thresh),zorder=19)
            print('total melt with real albedo',ALB['meltwater_cum'][-1])
            print('total melt with constant albedo',ALB['meltwater_cum_bright'][-1])
            ax[cc].text(pd.to_datetime('2021-08-28'),ALB['meltwater_cum_bright'][-1]+3,
                        "{:.0f}".format(ALB['meltwater_cum_bright'][-1])+' mm',color='k',fontsize=22,alpha=1,ha='left')
            ax[cc].text(pd.to_datetime('2021-08-28'),ALB['meltwater_cum'][-1]+3,
                        "{:.0f}".format(ALB['meltwater_cum'][-1])+' mm',color='k',fontsize=22,alpha=1,ha='left')

        ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')
        ax[cc].set_ylabel('mm', color='k')
        mult=0.75
        ax[cc].legend(prop={'size': font_size*mult},loc=2)

        print('meltwater_cum',ALB['meltwater_cum'][-1])

        #________________________________________________________________________________
        def return_rate(df,t0,t1,colorx,y0,axnum,titlex):

            t0x=pd.to_datetime(t0)
            t1x=pd.to_datetime(t1)
            
            a_timedelta = t0x - datetime(1970, 1, 1) ; sec0 = a_timedelta.total_seconds()
            a_timedelta = t1x - datetime(1970, 1, 1) ; sec1 = a_timedelta.total_seconds()
            middle=(sec0+sec1)/2
            middle_time=datetime.fromtimestamp(middle)
            
            duration = t1x - t0x
            duration_in_s = duration.total_seconds()
            days  = duration_in_s/86400
            hours  = duration_in_s/3600
            # print()
            
            vt0x=np.where(df['time']==t0x)
            vt1x=np.where(df['time']==t1x)
            ax[axnum].axvspan(t0x, t1x,color=colorx, alpha=0.3,label='high melt')
                        
            df['doy']=df['time'].dt.dayofyear
            df['hour']=df['time'].dt.hour
            df['jt']=df['doy']+df['hour']/24
            df['meltwater_cum'][~np.isfinite(ALB['meltwater_cum'])]=0.
            v=np.where(((df['jt']>=df['jt'][vt0x[0]].values[0])&(df['jt']<=df['jt'][vt1x[0]].values[0])))

            x=ALB['jt'][v[0]].values
    
            y=df['meltwater_cum'][v[0]].values
            
            b, m = polyfit(x, y, 1)
            # xx=[x[0],x[-1]]
            # print('m,b,xx',m,b,xx)
            # yy=[xx[0]*m+b,xx[-1]*m+b]
            # ax[1].plot([t0x,t1x],yy,'-',linewidth=4,c='w',alpha=0.9)
            print("{:.1f}".format(m),'mm/d',t0,'to',t1,"{:.1f}".format(days)+'d,',"{:.0f}".format(hours)+'h')
            mm=y[-1]-y[0]
            msg="{:.0f}".format(mm)+' mm, '+\
                "{:.0f}".format(m)+' mm d$^{-1}$\n'+\
                t0x.strftime('%d %b %Hz')+'\nto\n'+t1x.strftime('%d %b %Hz')+\
                '\n'+"{:.0f}".format(hours)+'h, '+"{:.1f}".format(days)+'d'
            # msg="{:.0f}".format(m)+' mm d$^{-1}$\n'+"{:.0f}".format(hours)+'h, '+"{:.1f}".format(days)+'d'
            print(msg)
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()

            # note y0 was df['meltwater_cum'][-1]+2 and before that was y0=(y_min+y_max)/2
            ax[axnum].text(middle_time,y0,msg,color='k',fontsize=19,alpha=1,ha='center')
            ax[axnum].text(middle_time,y0+76,titlex,color=colorx,fontsize=20,alpha=1,ha='center')

            return(vt0x,vt1x,m)
        #________________________________________________________________________________

        annotate_plot=1
        if annotate_plot:
            axnum=2
            vt0x,vt1x,m0=return_rate(ALB,'2021-08-7','2021-08-13-11','b',15,axnum,'')
            vt0x,vt1x,m1=return_rate(ALB,'2021-08-13-11','2021-08-14-19','r',45,axnum,'AR heatwave\nwith rain')
            vt0x,vt1x,m1=return_rate(ALB,'2021-08-14-19','2021-08-20-00','g',92,axnum,'warm, cloudy,\nno rain')
            vt0x,vt1x,m2=return_rate(ALB,'2021-08-20-00','2021-08-27-00','m',19,axnum,'albedo feedback\ndominated melt')
            # vt0x,vt1x,m2=return_rate(ALB,'2021-08-27-00','2021-08-31-12','k',25,axnum,'ablation end')
            ax[cc].axvspan(pd.to_datetime('2021-08-27'), pd.to_datetime('2021-08-31-12'),
                           color='k', alpha=0.3)#,label='ablation end')
            ax[cc].text(pd.to_datetime('2021-08-29-10'),3,'end of ablation\nseason',color='k',fontsize=20,alpha=1,ha='center')

            print('after/before melt rate = ',"{:.1f}".format(m2/m0))
        
        # LRNet['doy']=LRNet['time'].dt.dayofyear
        # LRNet['hour']=LRNet['time'].dt.hour
        # LRNet['jt']=LRNet['doy']+LRNet['hour']/24

        # xx=[np.min(x),np.max(x)]
        # xx=[meltwater.time[meltwater.jt==meltwater.jt[v[0][0]]],meltwater.time[meltwater.jt==meltwater.jt[v[0][-1]]]]
        # yyy=[(b + m * xx[0]),(b + m * xx[1])]
        # ax[cc].plot(meltwater['jt'][v[0]],yyy, '--',c='grey',label= 'before')
        # meltwater['jt'][vt0x[0]].values[0],meltwater['jt'][vt1x[0]].values[0]
    
    ax[cc].set_xlim(t0,t1)
    ax[cc].xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes
    ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
    if lev==1800:ax[cc].set_ylim(0,155)

    xcolor='darkblue'
    ax[cc].xaxis.label.set_color(xcolor)
    ax[cc].tick_params(axis='x', colors=xcolor)
    mult=0.8
    ax[cc].text(-0.096,0.03, "day of\nAug.'21",transform=ax[cc].transAxes, fontsize=font_size*mult,
        verticalalignment='top',rotation=0,color=xcolor, rotation_mode="anchor")  

    ly='p'
    
    if ly=='p':
        plt.savefig('./SEMMIS/Figs/'+str(lev)+'m '+source_name+'.png', DPI=72,
                bbox_inches='tight')
        if levs[0]==1300:
            fignam='Fig S10 SEMMIS.pdf'
        if levs[0]==1800:
            fignam='Fig 4 SEMMIS.pdf'
        plt.savefig('./Figs_GRL/'+fignam,bbox_inches='tight')    
#%%
do_gif=0
if do_gif == 1:
    print("making .gif")
    os.system('/usr/local/bin/convert -delay 80 -loop 0 ./figs/2000*.png ./figs/anim2.gif')

